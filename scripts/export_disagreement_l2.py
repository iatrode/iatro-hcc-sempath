import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add src/ to python path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hcc_sempath.io.tile_package import TilePackageReader
from hcc_sempath.modeling.models import HCCSemPathModel, calibrated_attribute_scores, normalized_prototype_logits
from hcc_sempath.training.config import embedding_dim, teacher_dims, teacher_names
from hcc_sempath.training.engine import _prepare_images
from hcc_sempath.training.prototype_images import load_prototype_image_bank, build_student_prototype_registry

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


def _robust_calibration(cosine_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center each attribute at its median and map its IQR approximately to 0.25-0.75."""
    biases = np.median(cosine_scores, axis=0)
    q25, q75 = np.quantile(cosine_scores, [0.25, 0.75], axis=0)
    temperatures = np.maximum((q75 - q25) / (2.0 * np.log(3.0)), 1e-4)
    return biases.astype(np.float32), temperatures.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Level-2 predictions of all models on disagreement tiles.")
    parser.add_argument("--predictions-csv", default="artifacts/caches/local_cache/teacher_disagreement/teacher_disagreement_model_predictions.csv")
    parser.add_argument("--prototype-dir", default="artifacts/prototypes")
    parser.add_argument("--output-npz", default="artifacts/caches/local_cache/teacher_disagreement/teacher_disagreement_l2_probabilities.npz")
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = _resolve_device()
    print(f"Using device: {device}")

    # 1. Load rows from predictions CSV
    pred_path = Path(args.predictions_csv)
    if not pred_path.exists():
        print(f"Error: Predictions CSV not found at {pred_path}")
        return
    rows = _read_csv(pred_path)
    print(f"Loaded {len(rows)} candidates from predictions CSV.")

    # 2. Read Images
    print("Loading image tiles from JXL package caches...")
    from collections import defaultdict
    pkg_to_rows = defaultdict(list)
    
    for idx, row in enumerate(rows):
        pkg_path = row.get("package_path")
        row_idx_str = row.get("row_idx")
        
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

    # Stack BCHW tensors
    stacked_images = torch.stack(images_list)
    
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

    # Resume from the model-wise cache and only infer missing requested models.
    output_npz = Path(args.output_npz)
    npz_data = {}
    l2_names = None
    if output_npz.exists():
        with np.load(output_npz, allow_pickle=True) as cached:
            npz_data = {key: cached[key].copy() for key in cached.files}
        cached_ids = [str(value) for value in npz_data.get("review_ids", [])]
        current_ids = [row["review_id"] for row in valid_rows]
        if cached_ids and cached_ids != current_ids:
            raise RuntimeError("cached L2 review IDs do not match the current prediction rows")
        if "l2_names" in npz_data:
            l2_names = [str(value) for value in npz_data["l2_names"].tolist()]

    requested = set(args.models)
    if requested:
        unknown = requested.difference(model_paths)
        if unknown:
            raise ValueError(f"unknown model keys: {sorted(unknown)}")
        model_paths = {key: value for key, value in model_paths.items() if key in requested}

    for key, paths in model_paths.items():
        if not args.force and key in npz_data and f"raw_{key}" in npz_data:
            print(f"Using cached L2 scores for model: {key}")
            continue
        chk_path = Path(paths["checkpoint"])
        cfg_path = Path(paths["config"])
        if not chk_path.exists() or not cfg_path.exists():
            print(f"Warning: model {key} files missing. Skipping.")
            continue
            
        print(f"Running L2 inference for model: {key}...")
        cfg = _load_config(cfg_path)
        cfg["data"]["zhcc_prototype_image_path"] = str(image_bank_path)
        
        # Load Model & Registry
        model = _load_model(cfg, chk_path, device)
        zhcc_registry = build_student_prototype_registry(
            model=model,
            image_bank=image_bank,
            cfg=cfg,
            device=device,
            batch_size=256
        ).to(device)
        
        if l2_names is None:
            l2_names = [zhcc_registry.names[idx] for idx in zhcc_registry.attribute_indices]
            print(f"Level-2 attributes detected ({len(l2_names)}): {l2_names}")
            
        # Batch inference
        batch_size = 64
        model_cosine_scores = []
        with torch.no_grad():
            for start_idx in range(0, len(valid_rows), batch_size):
                end_idx = min(start_idx + batch_size, len(valid_rows))
                batch_imgs = stacked_images[start_idx:end_idx].to(device)
                
                # Preprocess batch
                preprocessed_imgs = []
                for i in range(batch_imgs.shape[0]):
                    single_batch = {"images": batch_imgs[i].unsqueeze(0), "images_uint8": True}
                    preprocessed_imgs.append(_prepare_images(single_batch, cfg, device))
                preprocessed_imgs = torch.cat(preprocessed_imgs)
                
                outputs = model(preprocessed_imgs)
                embeddings = outputs["embedding_norm"]
                
                # Raw cosine similarity is the ranking signal. Do not sharpen it before calibration.
                attr_prototypes = zhcc_registry.attribute_prototypes
                cosine_scores = normalized_prototype_logits(embeddings, attr_prototypes)
                model_cosine_scores.append(cosine_scores.cpu().numpy())
                
        cosine_matrix = np.concatenate(model_cosine_scores, axis=0)
        biases, temperatures = _robust_calibration(cosine_matrix)
        prob_matrix = calibrated_attribute_scores(
            torch.from_numpy(cosine_matrix),
            torch.from_numpy(biases),
            torch.from_numpy(temperatures),
        ).numpy()
        npz_data[key] = prob_matrix
        npz_data[f"raw_{key}"] = cosine_matrix
        npz_data[f"bias_{key}"] = biases
        npz_data[f"temperature_{key}"] = temperatures
        
        # Calculate statistics
        print(f"=== {key} Level-2 Statistics ===")
        mean_probs = prob_matrix.mean(axis=0)
        activation_rates = (prob_matrix > 0.5).mean(axis=0)
        for idx, attr_name in enumerate(l2_names):
            print(
                f"  {attr_name:<35}: Mean relative score={mean_probs[idx]:.4f}, "
                f"Above median={activation_rates[idx]*100:.1f}%"
            )
        print()
        
        del model, zhcc_registry
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save to NPZ
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    
    # Include metadata in NPZ
    npz_data["l2_names"] = np.array(l2_names)
    npz_data["review_ids"] = np.array([r["review_id"] for r in valid_rows])
    npz_data["l2_score_kind"] = np.array("median_iqr_relative_retrieval_score_v1")
    
    np.savez_compressed(output_npz, **npz_data)
    print(f"Successfully saved all Level-2 probabilities to {output_npz}")

if __name__ == "__main__":
    main()
