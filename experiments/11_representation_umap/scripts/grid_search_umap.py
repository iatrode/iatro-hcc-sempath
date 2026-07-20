#!/usr/bin/env python
import csv
import sys
import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import silhouette_score
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
    predictions_csv = "artifacts/caches/local_cache/teacher_disagreement/teacher_disagreement_model_predictions.csv"
    review_csv = "annotations/reviews/teacher_disagreement/exval_1000/review.csv"
    student_checkpoint = "artifacts/models/hcc-sempath-full/checkpoints/best_scientific_score.pt"
    student_config = "artifacts/models/hcc-sempath-full/resolved_config.json"
    output_dir = Path("experiments/11_representation_umap/results/grid_search")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device()
    print(f"Using device: {device}")

    # Load predictions & reviews
    pred_path = REPO_ROOT / predictions_csv
    with pred_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    review_path = REPO_ROOT / review_csv
    review_labels = {}
    with review_path.open("r", newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            review_labels[r["review_id"]] = r["l1"]

    # Group by package to batch read images
    pkg_to_rows = defaultdict(list)
    for idx, row in enumerate(rows):
        pkg_path = row.get("package_path") or row.get("iac_path") or row.get("package")
        row_idx_str = row.get("row_idx") or row.get("row") or row.get("sample_row")
        if pkg_path and row_idx_str:
            pkg_path = REPO_ROOT / pkg_path
            pkg_to_rows[pkg_path].append((idx, row, int(row_idx_str)))

    # Load model
    student_model = _load_student(REPO_ROOT / student_checkpoint, REPO_ROOT / student_config, device)
    with (REPO_ROOT / student_config).open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    # Extract embeddings
    embeddings = []
    labels = []
    print("Extracting embeddings from Full Model...")
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
                    batch = {"images": arr.unsqueeze(0), "images_uint8": True}
                    norm_image = _prepare_images(batch, cfg, device)
                    with torch.no_grad():
                        z = student_model(norm_image)["embedding_norm"][0]
                    embeddings.append(z.cpu().numpy())
                    labels.append(l1_label)
                except Exception as e:
                    pass
            reader.close()
        except Exception as e:
            pass

    embeddings_arr = np.stack(embeddings)
    print(f"Extracted {embeddings_arr.shape[0]} embeddings.")

    # Grid search over UMAP parameters
    n_neighbors_list = [15, 30, 50, 100]
    min_dist_list = [0.00, 0.01, 0.05, 0.1, 0.3]
    metrics = ["cosine", "euclidean"]

    results = []

    for metric in metrics:
        for n_neighbors in n_neighbors_list:
            for min_dist in min_dist_list:
                print(f"Running metric={metric}, n_neighbors={n_neighbors}, min_dist={min_dist}...")
                try:
                    reducer = umap.UMAP(
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        metric=metric,
                        random_state=13
                    )
                    coords = reducer.fit_transform(embeddings_arr)
                    
                    # Compute Silhouette Score on UMAP coordinates
                    # Convert labels to integer class IDs
                    class_mapping = {l: i for i, l in enumerate(sorted(list(set(labels))))}
                    label_ids = np.array([class_mapping[l] for l in labels])
                    score = silhouette_score(coords, label_ids)
                    
                    results.append({
                        "metric": metric,
                        "n_neighbors": n_neighbors,
                        "min_dist": min_dist,
                        "silhouette": score
                    })

                    # Plot and save
                    fig, ax = plt.subplots(figsize=(8, 7))
                    for label in sorted(list(set(labels))):
                        idx_mask = [i for i, l in enumerate(labels) if l == label]
                        ax.scatter(
                            coords[idx_mask, 0],
                            coords[idx_mask, 1],
                            c=CLASS_COLORS.get(label, "#A6A6A6"),
                            label=label,
                            alpha=0.82,
                            edgecolors="none",
                            s=24
                        )
                    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10, loc="best")
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.set_xlabel("UMAP-1", fontsize=11, weight="bold")
                    ax.set_ylabel("UMAP-2", fontsize=11, weight="bold")
                    ax.set_title(f"UMAP (Full Model, metric={metric}, nn={n_neighbors}, md={min_dist})\nSilhouette Score: {score:.4f}", fontsize=12, weight="bold")
                    plt.tight_layout()
                    
                    png_path = output_dir / f"umap_{metric}_nn{n_neighbors}_md{str(min_dist).replace('.', 'p')}.png"
                    fig.savefig(png_path, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                except Exception as e:
                    print(f"Error for metric={metric}, nn={n_neighbors}, md={min_dist}: {e}")

    # Sort and print top 10 results
    results = sorted(results, key=lambda x: x["silhouette"], reverse=True)
    print("\n--- Top UMAP Hyperparameters by Silhouette Score ---")
    for idx, r in enumerate(results[:10]):
        print(f"{idx+1}. metric={r['metric']}, n_neighbors={r['n_neighbors']}, min_dist={r['min_dist']} -> Silhouette={r['silhouette']:.4f}")

if __name__ == "__main__":
    main()
