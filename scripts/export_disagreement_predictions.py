import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add src/ to python path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hcc_sempath.io.tile_package import TilePackageReader
from hcc_sempath.modeling.models import HCCSemPathModel, normalized_prototype_logits
from hcc_sempath.modeling.prototypes import PrototypeRegistry
from hcc_sempath.training.config import embedding_dim, teacher_dims, teacher_names
from hcc_sempath.training.datasets import _open_feature_source
from hcc_sempath.training.engine import _prepare_images
from hcc_sempath.training.prototype_images import load_prototype_image_bank, build_student_prototype_registry

# Define L1 names consistent with cli/annotate_prototypes.py
L1_PROTOTYPES = ["HCC-tumor", "Background-liver", "Inflammatory-stromal", "Degenerative-material"]

def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix == ".json":
            return json.load(handle)
        import yaml
        return yaml.safe_load(handle) or {}

def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _load_model(cfg: dict, checkpoint: Path, device: torch.device) -> HCCSemPathModel:
    names = teacher_names(cfg)
    dims = teacher_dims(cfg, names)
    model = HCCSemPathModel(
        backbone_name=cfg["model"]["backbone_name"],
        embedding_dim=embedding_dim(cfg),
        teacher_dims=dims,
        pretrained=False,
        projector_type=cfg["model"].get("projector_type", "linear"),
        projector_hidden_dim=int(cfg["model"].get("projector_hidden_dim", 2048)),
        teacher_head_type=cfg["model"].get("teacher_head_type", "linear"),
        grad_checkpointing=False,
    ).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    return model

def _read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def main() -> None:
    parser = argparse.ArgumentParser(description="Export predictions of all models on disagreement tiles.")
    parser.add_argument("--val-csv", default="artifacts/caches/local_cache/teacher_disagreement/val/teacher_disagreement_top500.csv")
    parser.add_argument("--exval-csv", default="artifacts/caches/local_cache/teacher_disagreement/exval/teacher_disagreement_top500.csv")
    parser.add_argument("--review-json", default="")
    parser.add_argument("--prototype-dir", default="artifacts/prototypes")
    parser.add_argument("--output-csv", default="artifacts/caches/local_cache/teacher_disagreement/teacher_disagreement_model_predictions.csv")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = _resolve_device()
    print(f"Using device: {device}")

    # 1. Load Disagreement Rows
    rows = []
    if args.review_json and Path(args.review_json).exists():
        print(f"Loading disagreement tiles directly from UI JSON: {args.review_json}")
        with Path(args.review_json).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        annotations = payload.get("annotations", {})
        sorted_keys = sorted(annotations.keys(), key=lambda k: int(k.split("-")[-1]) if "-" in k else k)
        for r_id in sorted_keys:
            item = annotations[r_id]
            row_data = dict(item)
            row_data["review_id"] = r_id
            row_data["package_path"] = item.get("package_path") or item.get("iac_path", "")
            row_data["row_idx"] = int(item.get("row_idx") or item.get("row", 0))
            rows.append(row_data)
        print(f"Loaded {len(rows)} tiles from JSON.")
    else:
        # Fallback to loading and shuffling CSVs (legacy behavior)
        val_path = Path(args.val_csv)
        exval_path = Path(args.exval_csv)
        seen = set()
        for path in [val_path, exval_path]:
            if path.exists():
                for row in _read_csv(path):
                    tile_id = row["tile_id"]
                    if tile_id in seen:
                        continue
                    seen.add(tile_id)
                    rows.append(row)
        print(f"Loaded {len(rows)} disagreement candidates from CSVs.")
        random.Random(args.seed).shuffle(rows)
        for idx, row in enumerate(rows, start=1):
            row["review_id"] = f"TD-{idx:04d}"

    # 3. Read Images
    print("Loading image tiles from JXL package caches...")
    from collections import defaultdict
    pkg_to_rows = defaultdict(list)
    
    for idx, row in enumerate(rows):
        pkg_path = row.get("package_path") or row.get("iac_path") or row.get("package")
        row_idx_str = row.get("row_idx") or row.get("row") or row.get("sample_row")
        
        if not pkg_path or not row_idx_str:
            print(f"Warning: missing path/row metadata for review_id={row['review_id']}. Skipping.")
            continue
            
        row_idx = int(row_idx_str)
        pkg_path = Path(pkg_path)
        if not pkg_path.is_absolute():
            pkg_path = REPO_ROOT / pkg_path
            
        pkg_to_rows[pkg_path].append((idx, row, row_idx))
        
    loaded_results = []
    for pkg_path, items in pkg_to_rows.items():
        try:
            reader = TilePackageReader(pkg_path)
        except Exception as e:
            print(f"Error opening package {pkg_path}: {e}")
            continue
            
        for idx, row, row_idx in items:
            try:
                pil_img = reader.read_image_at(row_idx).convert("RGB")
                arr = torch.from_numpy(np.array(pil_img, dtype=np.uint8, copy=True)).permute(2, 0, 1)
                loaded_results.append((idx, row, arr))
            except Exception as e:
                print(f"Error reading row {row_idx} from {pkg_path}: {e}")
        reader.close()
        
    # Sort back to preserve the original queue order
    loaded_results.sort(key=lambda x: x[0])
    
    images_list = [item[2] for item in loaded_results]
    valid_rows = [item[1] for item in loaded_results]
    
    print(f"Successfully loaded {len(images_list)} images for inference.")
    if not images_list:
        print("No images loaded. Exiting.")
        return


    # 4. Prepare batch inputs
    # Stack CHW tensors -> BCHW
    stacked_images = torch.stack(images_list) # BCHW shape
    
    # Define models
    model_paths = {
        "pred_full": {
            "checkpoint": "artifacts/models/hcc-sempath-full/checkpoints/best_scientific_score.pt",
            "config": "artifacts/models/hcc-sempath-full/resolved_config.json"
        },
        "pred_a0": {
            "checkpoint": "artifacts/experiments/ablation/a0_full_pamtd/checkpoints/best_scientific_score.pt",
            "config": "artifacts/experiments/ablation/a0_full_pamtd/resolved_config.json"
        },
        "pred_a1": {
            "checkpoint": "artifacts/experiments/ablation/a1_no_prototype/checkpoints/best_scientific_score.pt",
            "config": "artifacts/experiments/ablation/a1_no_prototype/resolved_config.json"
        },
        "pred_a2": {
            "checkpoint": "artifacts/experiments/ablation/a2_no_adjudication/checkpoints/best_scientific_score.pt",
            "config": "artifacts/experiments/ablation/a2_no_adjudication/resolved_config.json"
        },
        "pred_a3": {
            "checkpoint": "artifacts/experiments/ablation/a3_single_teacher/checkpoints/best_scientific_score.pt",
            "config": "artifacts/experiments/ablation/a3_single_teacher/resolved_config.json"
        },
        "pred_a4": {
            "checkpoint": "artifacts/experiments/ablation/a4_single_teacher_prototype/checkpoints/best_scientific_score.pt",
            "config": "artifacts/experiments/ablation/a4_single_teacher_prototype/resolved_config.json"
        },
        "pred_a5": {
            "checkpoint": "artifacts/experiments/ablation/a5_static_prototypes/checkpoints/best_scientific_score.pt",
            "config": "artifacts/experiments/ablation/a5_static_prototypes/resolved_config.json"
        },
        "pred_a6": {
            "checkpoint": "artifacts/experiments/ablation/a6_full_filter/checkpoints/best_scientific_score.pt",
            "config": "artifacts/experiments/ablation/a6_full_filter/resolved_config.json"
        }
    }

    # Load Prototype Image Bank for Dynamic Refresh
    proto_dir = Path(args.prototype_dir)
    image_bank_path = proto_dir / "zhcc_hcc_prototype_images.pt"
    if not image_bank_path.exists():
        print(f"Error: Prototype image bank not found at {image_bank_path}")
        return
        
    print("Loading prototype image bank...")
    image_bank = load_prototype_image_bank(image_bank_path)

    requested = set(args.models)
    if requested:
        unknown = requested.difference(model_paths)
        if unknown:
            raise ValueError(f"unknown model keys: {sorted(unknown)}")

    existing_by_id = {}
    output_path = Path(args.output_csv)
    if output_path.exists():
        existing_by_id = {row["review_id"]: row for row in _read_csv(output_path)}

    # Dictionary to store predictions for each model key.
    predictions = {}
    for key in model_paths:
        cached = [existing_by_id.get(row["review_id"], {}).get(key, "") for row in valid_rows]
        if not args.force and key not in requested and all(value not in {"", "N/A"} for value in cached):
            predictions[key] = cached
        elif not args.force and requested and key not in requested:
            predictions[key] = cached
        else:
            predictions[key] = []

    for key, paths in model_paths.items():
        if predictions[key]:
            print(f"Using cached L1 predictions for model: {key}")
            continue
        chk_path = Path(paths["checkpoint"])
        cfg_path = Path(paths["config"])
        if not chk_path.exists() or not cfg_path.exists():
            print(f"Warning: model {key} files missing ({chk_path} or {cfg_path}). Filling with N/A.")
            predictions[key] = ["N/A"] * len(valid_rows)
            continue
            
        print(f"Running inference for model: {key}...")
        cfg = _load_config(cfg_path)
        # Fix paths inside config
        cfg["data"]["zhcc_prototype_image_path"] = str(image_bank_path)
        
        # Load Model
        model = _load_model(cfg, chk_path, device)
        
        # Build registry
        print(f"Building dynamic student prototypes registry for {key}...")
        zhcc_registry = build_student_prototype_registry(
            model=model,
            image_bank=image_bank,
            cfg=cfg,
            device=device,
            batch_size=256
        ).to(device)
        
        # Batch inference
        batch_size = 64
        model_preds = []
        with torch.no_grad():
            for start_idx in range(0, len(valid_rows), batch_size):
                end_idx = min(start_idx + batch_size, len(valid_rows))
                batch_imgs = stacked_images[start_idx:end_idx].to(device)
                
                # Normalize batch images
                preprocessed_imgs = []
                for i in range(batch_imgs.shape[0]):
                    single_batch = {"images": batch_imgs[i].unsqueeze(0), "images_uint8": True}
                    preprocessed_imgs.append(_prepare_images(single_batch, cfg, device))
                preprocessed_imgs = torch.cat(preprocessed_imgs)
                
                outputs = model(preprocessed_imgs)
                embeddings = outputs["embedding_norm"] # (B, embedding_dim)
                
                # Cosine similarity to student prototypes
                primary_prototypes = zhcc_registry.primary_prototypes
                logits = normalized_prototype_logits(embeddings, primary_prototypes) # (B, num_primary)
                
                # Get L1 predictions
                pred_indices = logits.argmax(dim=1).cpu().numpy()
                primary_names = [zhcc_registry.names[idx] for idx in zhcc_registry.primary_indices]
                
                for idx in pred_indices:
                    model_preds.append(primary_names[int(idx)])
                    
        predictions[key] = model_preds
        del model, zhcc_registry
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 5. Export results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_rows = []
    for idx, row in enumerate(valid_rows):
        output_rows.append({
            "review_id": row["review_id"],
            "tile_id": row["tile_id"],
            "slide_id": row["slide_id"],
            "patient_id": row.get("patient_id") or row.get("patient") or "",
            "package_path": row.get("package_path") or row.get("iac_path") or row.get("package"),
            "row_idx": row.get("row_idx") or row.get("row") or row.get("sample_row"),
            "pred_full": predictions["pred_full"][idx],
            "pred_a0": predictions["pred_a0"][idx],
            "pred_a1": predictions["pred_a1"][idx],
            "pred_a2": predictions["pred_a2"][idx],
            "pred_a3": predictions["pred_a3"][idx],
            "pred_a4": predictions["pred_a4"][idx],
            "pred_a5": predictions["pred_a5"][idx],
            "pred_a6": predictions["pred_a6"][idx],
        })
        
    fieldnames = [
        "review_id", "tile_id", "slide_id", "patient_id", "package_path", "row_idx",
        "pred_full", "pred_a0", "pred_a1", "pred_a2", "pred_a3", "pred_a4",
        "pred_a5", "pred_a6",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
        
    print(f"Successfully exported predictions for all models to {output_path}")

if __name__ == "__main__":
    main()
