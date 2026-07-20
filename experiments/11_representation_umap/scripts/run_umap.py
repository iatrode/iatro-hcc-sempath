#!/usr/bin/env python
import argparse
import csv
import sys
import shutil
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap  # type: ignore

# Add src/ to python path
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from iatro.iac.adapters.tiles import TilePackageReader
from hcc_sempath.modeling.models import HCCSemPathModel
from hcc_sempath.training.config import embedding_dim, teacher_dims, teacher_names
from hcc_sempath.training.engine import _prepare_images

# Color mapping for Level-1 classes
CLASS_COLORS = {
    "HCC-tumor": "#E64B35",          # Red/Coral
    "Background-liver": "#4DBBD5",     # Cyan/Blue
    "Inflammatory-stromal": "#00A087"  # Green/Teal
}

def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _load_student(checkpoint_path: Path, config_path: Path, device: torch.device) -> HCCSemPathModel:
    with config_path.open("r", encoding="utf-8") as handle:
        import json
        cfg = json.load(handle)
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
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    return model

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 3-panel UMAP figure mapping the selected third row config (All Joint, Regular Independent, Conflict Independent).")
    parser.add_argument(
        "--predictions-csv",
        default="artifacts/caches/local_cache/teacher_disagreement/teacher_disagreement_model_predictions.csv",
        help="Path to predictions/annotations CSV containing expert L1 labels"
    )
    parser.add_argument(
        "--review-csv",
        default="annotations/reviews/teacher_disagreement/exval_1000/review.csv",
        help="Path to expert reviews CSV containing consensus labels"
    )
    parser.add_argument(
        "--student-checkpoint",
        default="artifacts/models/hcc-sempath-full/checkpoints/best_scientific_score.pt",
        help="Path to student model checkpoint"
    )
    parser.add_argument(
        "--student-config",
        default="artifacts/models/hcc-sempath-full/resolved_config.json",
        help="Path to student model config"
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/11_representation_umap/results",
        help="Output directory for results"
    )
    parser.add_argument("--n-neighbors", type=int, default=50, help="UMAP n_neighbors parameter (default: 50)")
    parser.add_argument("--min-dist", type=float, default=0.05, help="UMAP min_dist parameter (default: 0.05)")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    device = _resolve_device()
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load predictions & reviews
    pred_path = REPO_ROOT / args.predictions_csv
    if not pred_path.exists():
        print(f"Error: Predictions CSV not found at {pred_path}")
        return

    with pred_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    review_path = REPO_ROOT / args.review_csv
    review_labels = {}
    review_source_groups = {}
    if review_path.exists():
        with review_path.open("r", newline="", encoding="utf-8") as handle:
            for r in csv.DictReader(handle):
                review_labels[r["review_id"]] = r["l1"]
                review_source_groups[r["review_id"]] = r["source_group"]
    else:
        print(f"Error: Review CSV not found at {review_path}.")
        return

    # 2. Group by package
    pkg_to_rows = defaultdict(list)
    for idx, row in enumerate(rows):
        pkg_path = row.get("package_path") or row.get("iac_path") or row.get("package")
        row_idx_str = row.get("row_idx") or row.get("row") or row.get("sample_row")
        if pkg_path and row_idx_str:
            pkg_path = REPO_ROOT / pkg_path
            pkg_to_rows[pkg_path].append((idx, row, int(row_idx_str)))

    # 3. Pre-load valid images
    print("Pre-loading images into memory...")
    valid_tiles = []
    
    with (REPO_ROOT / args.student_config).open("r", encoding="utf-8") as handle:
        import json
        cfg = json.load(handle)

    for pkg_path, items in pkg_to_rows.items():
        if not pkg_path.exists():
            continue
        try:
            reader = TilePackageReader(pkg_path)
            for idx, row, row_idx in items:
                l1_label = review_labels.get(row["review_id"], "Unknown")
                if l1_label in ("Degenerative-material", "Unknown"):
                    continue
                try:
                    pil_img = reader.read_image_at(row_idx).convert("RGB")
                    arr = torch.from_numpy(np.array(pil_img, dtype=np.uint8, copy=True)).permute(2, 0, 1)
                    valid_tiles.append({
                        "image_tensor": arr,
                        "label": l1_label,
                        "source_group": review_source_groups.get(row["review_id"], "Unknown")
                    })
                except Exception as e:
                    print(f"Error reading row {row_idx}: {e}")
            reader.close()
        except Exception as e:
            print(f"Error package {pkg_path}: {e}")

    print(f"Pre-loaded {len(valid_tiles)} valid tiles.")
    if not valid_tiles:
        return

    # Load Full Model and run inference
    student_model = _load_student(REPO_ROOT / args.student_checkpoint, REPO_ROOT / args.student_config, device)
    embeddings = []
    labels = []
    source_groups = []
    
    print("Running model inference...")
    for tile in valid_tiles:
        batch = {"images": tile["image_tensor"].unsqueeze(0), "images_uint8": True}
        norm_image = _prepare_images(batch, cfg, device)
        with torch.no_grad():
            z = student_model(norm_image)["embedding_norm"][0]
        embeddings.append(z.cpu().numpy())
        labels.append(tile["label"])
        source_groups.append(tile["source_group"])

    embeddings_arr = np.stack(embeddings)
    print(f"Extracted {embeddings_arr.shape[0]} embeddings.")

    unique_labels = sorted(list(set(labels)))
    rand_mask = [i for i, sg in enumerate(source_groups) if sg == "random500"]
    top_mask = [i for i, sg in enumerate(source_groups) if sg == "top500"]

    print(f"Running projections with n_neighbors={args.n_neighbors}, min_dist={args.min_dist}...")
    
    # 1. Column 1: All Tiles (Joint UMAP)
    print("Fitting Joint UMAP (All Tiles)...")
    reducer_all = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist, metric="cosine", random_state=args.seed)
    coords_all = reducer_all.fit_transform(embeddings_arr)

    # 2. Column 2: Random500 (Independent UMAP)
    print("Fitting Independent UMAP (Random500)...")
    rand_embeddings = embeddings_arr[rand_mask]
    rand_labels = [labels[i] for i in rand_mask]
    # Separately reduce min_dist to 0.01 for Random500 to shrink point distances
    reducer_rand = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=0.01, metric="cosine", random_state=args.seed)
    coords_rand = reducer_rand.fit_transform(rand_embeddings)

    # 3. Column 3: Top500 (Independent UMAP)
    print("Fitting Independent UMAP (Top500)...")
    top_embeddings = embeddings_arr[top_mask]
    top_labels = [labels[i] for i in top_mask]
    # Set n_neighbors=100, min_dist=0.1 for Top500 to expand point distances while preserving global anchor pull and preventing line-like distortion
    reducer_top = umap.UMAP(n_neighbors=100, min_dist=0.1, metric="cosine", random_state=args.seed)
    coords_top = reducer_top.fit_transform(top_embeddings)

    # Plot 3-panel figure side-by-side
    print("Plotting figures...")
    fig, (ax_all, ax_rand, ax_top) = plt.subplots(1, 3, figsize=(22, 7))

    # Panel A: All Tiles (Joint)
    for label in unique_labels:
        idx_mask = [i for i, l in enumerate(labels) if l == label]
        ax_all.scatter(
            coords_all[idx_mask, 0],
            coords_all[idx_mask, 1],
            c=CLASS_COLORS.get(label, "#A6A6A6"),
            alpha=0.82,
            edgecolors="none",
            s=20
        )
    ax_all.spines["top"].set_visible(False)
    ax_all.spines["right"].set_visible(False)
    ax_all.set_xlabel("UMAP-1", fontsize=11, weight="bold")
    ax_all.set_ylabel("UMAP-2", fontsize=11, weight="bold")
    ax_all.set_title(f"All Tiles (Joint UMAP, n={coords_all.shape[0]})", fontsize=12, weight="bold")

    # Panel B: Random500 (Independent)
    for label in unique_labels:
        idx_mask = [i for i, l in enumerate(rand_labels) if l == label]
        ax_rand.scatter(
            coords_rand[idx_mask, 0],
            coords_rand[idx_mask, 1],
            c=CLASS_COLORS.get(label, "#A6A6A6"),
            label=label,
            alpha=0.82,
            edgecolors="none",
            s=20
        )
    ax_rand.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10, loc="best")
    ax_rand.spines["top"].set_visible(False)
    ax_rand.spines["right"].set_visible(False)
    ax_rand.set_xlabel("UMAP-1", fontsize=11, weight="bold")
    ax_rand.set_ylabel("UMAP-2", fontsize=11, weight="bold")
    ax_rand.set_title(f"Random500 (Independent UMAP, n={len(rand_mask)})", fontsize=12, weight="bold")

    # Panel C: Top500 (Independent)
    for label in unique_labels:
        idx_mask = [i for i, l in enumerate(top_labels) if l == label]
        ax_top.scatter(
            coords_top[idx_mask, 0],
            coords_top[idx_mask, 1],
            c=CLASS_COLORS.get(label, "#A6A6A6"),
            alpha=0.82,
            edgecolors="none",
            s=20
        )
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.set_xlabel("UMAP-1", fontsize=11, weight="bold")
    ax_top.set_ylabel("UMAP-2", fontsize=11, weight="bold")
    ax_top.set_title(f"Top500 (Independent UMAP, n={len(top_mask)})", fontsize=12, weight="bold")

    fig.suptitle(f"UMAP projection of HCC-SemPath ($z_{{HCC}}$) embedding space\n(n_neighbors={args.n_neighbors}, min_dist={args.min_dist})", fontsize=14, weight="bold", y=0.98)
    plt.tight_layout()

    # Save
    out_png = output_dir / "zhcc_umap.png"
    out_pdf = output_dir / "zhcc_umap.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated final 3-panel figure at {out_png}")

    # Copy to manuscript/figures/ if folder exists
    ms_figures_dir = REPO_ROOT / "manuscript" / "figures"
    if ms_figures_dir.exists():
        shutil.copyfile(out_png, ms_figures_dir / "zhcc_umap.png")
        shutil.copyfile(out_pdf, ms_figures_dir / "zhcc_umap.pdf")
        print(f"Copied figures to {ms_figures_dir}")

if __name__ == "__main__":
    main()
