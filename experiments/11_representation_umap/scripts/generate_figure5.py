import os
os.environ["NUMBA_DISABLE_JIT"] = "1"

#!/usr/bin/env python
import csv
import json
import sys
import shutil
import gc
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap  # type: ignore
from PIL import Image
from timm.data import create_transform, resolve_model_data_config

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hcc_sempath.io.tile_package import TilePackageReader
from hcc_sempath.modeling.models import HCCSemPathModel
from hcc_sempath.teacher.cache import TimmTeacherEncoder, _resolve_model_spec

CLASS_COLORS = {
    "HCC-tumor": "#E64B35",          # Red/Coral
    "Background-liver": "#4DBBD5",     # Cyan/Blue
    "Inflammatory-stromal": "#00A087"  # Green/Teal
}

# Global list to store attention matrix from hook
attention_matrices = []

def attn_hook(module, input, output):
    if isinstance(output, tuple):
        output = output[0]
    attention_matrices.append(output.cpu())

def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _load_student(checkpoint_path: Path, config_path: Path, device: torch.device) -> tuple[HCCSemPathModel, dict]:
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    from hcc_sempath.training.config import embedding_dim, teacher_dims, teacher_names
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
    return model, cfg

def main() -> None:
    device = _resolve_device()
    print(f"Using device: {device}")

    # Paths
    predictions_csv = REPO_ROOT / "artifacts/caches/local_cache/teacher_disagreement/teacher_disagreement_model_predictions.csv"
    review_csv = REPO_ROOT / "annotations/reviews/teacher_disagreement/exval_1000/review.csv"
    student_checkpoint = REPO_ROOT / "artifacts/models/hcc-sempath-full/checkpoints/best_scientific_score.pt"
    student_config = REPO_ROOT / "artifacts/models/hcc-sempath-full/resolved_config.json"
    cache_dir = Path("/tmp/hcc_sempath_exval_selector/cache")

    # 1. Load predictions & reviews
    with review_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    # Filter valid rows (exval split and primary classes only)
    valid_rows = [
        row for row in rows 
        if row["l1"] in CLASS_COLORS and row["split"] == "exval"
    ]
    print(f"Loaded {len(valid_rows)} valid tiles for UMAP.")

    # Group rows by slide package for optimized I/O
    pkg_to_rows = defaultdict(list)
    for row in valid_rows:
        pkg_path = REPO_ROOT / row["package_path"]
        pkg_to_rows[pkg_path].append(row)

    # 3 target cases for Panel B rows
    cases = [
        {"id": "TD-0038", "label": "HCC-tumor"},
        {"id": "TD-0132", "label": "Background-liver"},
        {"id": "TD-0028", "label": "Inflammatory-stromal"}
    ]

    # Pre-extract original images for all 3 cases
    case_orig_imgs = {}
    for c in cases:
        rid = c["id"]
        orig_img_path = cache_dir / f"{rid}/original.png"
        if orig_img_path.exists():
            orig_img = Image.open(orig_img_path).convert("RGB").resize((224, 224))
        else:
            row_info = next(r for r in rows if r["review_id"] == rid)
            img_reader = TilePackageReader(REPO_ROOT / row_info["package_path"])
            orig_img = img_reader.read_image_at(int(row_info["row_idx"])).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            img_reader.close()
        case_orig_imgs[rid] = orig_img

    # 2. Extract Student embeddings & Student attention maps for all cases
    print("Loading student model...")
    student_model, cfg = _load_student(student_checkpoint, student_config, device)

    embeddings = []
    labels = []
    source_groups = []
    
    # Hook student model
    block = student_model.encoder.backbone.blocks[-1]
    block.attn.fused_attn = False
    
    case_attn_maps = {c["id"]: {} for c in cases}

    print("Running inference to collect UMAP embeddings...")
    for pkg_path, items in pkg_to_rows.items():
        if not pkg_path.exists():
            continue
        img_reader = TilePackageReader(pkg_path)
        for row in items:
            tile_id = row["tile_id"]
            row_idx = int(row["row_idx"])
            l1_label = row["l1"]
            source_group = row["source_group"]
            
            try:
                pil_img = img_reader.read_image_at(row_idx).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
                arr = torch.from_numpy(np.array(pil_img, dtype=np.uint8)).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32) / 255.0
                mean = torch.tensor(cfg["data"]["mean"], device=device).view(1, 3, 1, 1)
                std = torch.tensor(cfg["data"]["std"], device=device).view(1, 3, 1, 1)
                x = (arr - mean) / std
                
                with torch.no_grad():
                    z = student_model(x)["embedding_norm"][0].cpu().numpy()
                
                embeddings.append(z)
                labels.append(l1_label)
                source_groups.append(source_group)
                
            except Exception as e:
                print(f"Error processing tile {tile_id}: {e}")
                
        img_reader.close()

    # Hook Student to get attention maps for the 3 cases
    for c in cases:
        rid = c["id"]
        orig_img = case_orig_imgs[rid]
        arr = torch.from_numpy(np.array(orig_img, dtype=np.uint8)).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32) / 255.0
        mean = torch.tensor(cfg["data"]["mean"], device=device).view(1, 3, 1, 1)
        std = torch.tensor(cfg["data"]["std"], device=device).view(1, 3, 1, 1)
        x = (arr - mean) / std
        
        attention_matrices.clear()
        hook_handle = block.attn.attn_drop.register_forward_hook(attn_hook)
        with torch.no_grad():
            _ = student_model(x)
        hook_handle.remove()
        
        attn = attention_matrices[0][0]
        cls_attn = attn[:, 0, 1:].mean(dim=0).view(14, 14).numpy()
        vmin, vmax = np.percentile(cls_attn, 2), np.percentile(cls_attn, 98)
        cls_attn = np.clip((cls_attn - vmin) / (vmax - vmin + 1e-8), 0, 1)
        attn_img = Image.fromarray((cls_attn * 255).astype(np.uint8)).resize((224, 224), Image.Resampling.BILINEAR)
        case_attn_maps[rid]["student"] = attn_img

    del student_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Extracted student embeddings count: {len(embeddings)}")

    # 3. Fit UMAPs
    embeddings_arr = np.stack(embeddings)
    rand_mask = [i for i, sg in enumerate(source_groups) if sg == "random500"]
    top_mask = [i for i, sg in enumerate(source_groups) if sg == "top500"]

    print("Fitting Joint UMAP (All Tiles)...")
    reducer_all = umap.UMAP(n_neighbors=50, min_dist=0.05, metric="cosine", random_state=None)
    coords_all = reducer_all.fit_transform(embeddings_arr)

    print("Fitting Independent UMAP (Random500)...")
    reducer_rand = umap.UMAP(n_neighbors=50, min_dist=0.01, metric="cosine", random_state=None)
    coords_rand = reducer_rand.fit_transform(embeddings_arr[rand_mask])

    print("Fitting Independent UMAP (Top500)...")
    reducer_top = umap.UMAP(n_neighbors=100, min_dist=0.1, metric="cosine", random_state=None)
    coords_top = reducer_top.fit_transform(embeddings_arr[top_mask])

    # 4. Extract Teacher attention maps for all cases
    teachers = ["gigapath", "h_optimus_1", "uni2_h", "virchow2"]
    teacher_heads = {
        "gigapath": 5,
        "h_optimus_1": 17,
        "uni2_h": 14,
        "virchow2": 9
    }

    for t in teachers:
        print(f"Loading teacher model: {t}...")
        spec = _resolve_model_spec("artifacts/models/teachers/" + t)
        encoder = TimmTeacherEncoder(spec["model_name"], model_kwargs=spec["model_kwargs"]).to(device)
        encoder.eval()

        data_config = resolve_model_data_config(encoder.model)
        data_config["input_size"] = (3, 224, 224)
        transform = create_transform(**data_config, is_training=False)

        block = encoder.model.blocks[-1]
        block.attn.fused_attn = False
        head_idx = teacher_heads[t]
        
        for c in cases:
            rid = c["id"]
            orig_img = case_orig_imgs[rid]
            img_tensor = transform(orig_img).unsqueeze(0).to(device)
            
            attention_matrices.clear()
            hook_handle = block.attn.attn_drop.register_forward_hook(attn_hook)
            with torch.no_grad():
                _ = encoder(img_tensor)
            hook_handle.remove()

            # Extract average attention across all heads
            attn = attention_matrices[0][0]
            cls_to_patches = attn[:, 0, -196:].mean(dim=0).view(14, 14).numpy()
            vmin, vmax = np.percentile(cls_to_patches, 2), np.percentile(cls_to_patches, 98)
            cls_to_patches = np.clip((cls_to_patches - vmin) / (vmax - vmin + 1e-8), 0, 1)
            
            t_attn_img = Image.fromarray((cls_to_patches * 255).astype(np.uint8)).resize((224, 224), Image.Resampling.BILINEAR)
            case_attn_maps[rid][t] = t_attn_img

        del encoder
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 5. Plot unified figure
    print("Plotting unified Figure 5 (Grid 4x6)...")
    fig = plt.figure(figsize=(19, 14.2))
    # Height ratios: Row 0 (UMAP) has more space. Rows 1,2,3 (Attention) are equal.
    gs = fig.add_gridspec(4, 6, height_ratios=[1.3, 1.0, 1.0, 1.0], hspace=0.28, wspace=0.28)

    # --- Panel A: UMAPs ---
    ax_all = fig.add_subplot(gs[0, 0:2])
    ax_rand = fig.add_subplot(gs[0, 2:4])
    ax_top = fig.add_subplot(gs[0, 4:6])

    unique_labels = sorted(list(set(labels)))

    # Plot All Tiles
    for label in unique_labels:
        mask = [j for j, l in enumerate(labels) if l == label]
        ax_all.scatter(
            coords_all[mask, 0], coords_all[mask, 1],
            c=CLASS_COLORS[label], alpha=0.82, edgecolors="none", s=25
        )
    ax_all.set_title(f"All Tiles (Joint UMAP, n={coords_all.shape[0]})", fontsize=12, weight="bold", pad=8)

    # Plot Random500
    r_labels = [labels[j] for j in rand_mask]
    for label in unique_labels:
        mask = [j for j, l in enumerate(r_labels) if l == label]
        ax_rand.scatter(
            coords_rand[mask, 0], coords_rand[mask, 1],
            c=CLASS_COLORS[label], alpha=0.82, edgecolors="none", s=25, label=label
        )
    ax_rand.set_title(f"Random500 (Independent UMAP, n={len(rand_mask)})", fontsize=12, weight="bold", pad=8)
    ax_rand.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10, loc="best")

    # Plot Top500
    t_labels = [labels[j] for j in top_mask]
    for label in unique_labels:
        mask = [j for j, l in enumerate(t_labels) if l == label]
        ax_top.scatter(
            coords_top[mask, 0], coords_top[mask, 1],
            c=CLASS_COLORS[label], alpha=0.82, edgecolors="none", s=25
        )
    ax_top.set_title(f"Top500 (Independent UMAP, n={len(top_mask)})", fontsize=12, weight="bold", pad=8)

    # Format UMAP axes and enforce box aspect
    for ax in [ax_all, ax_rand, ax_top]:
        ax.set_xlabel("UMAP-1", fontsize=10, weight="bold")
        ax.set_ylabel("UMAP-2", fontsize=10, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_box_aspect(1)

    # --- Panel B: Microscopic Attention Maps (3x6 Grid) ---
    col_keys = ["he", "gigapath", "h_optimus_1", "uni2_h", "virchow2", "student"]
    col_titles = [
        "H&E Tile",
        "GigaPath\n(Mean Attention)",
        "H-Optimus-1\n(Mean Attention)",
        "UNI2-H\n(Mean Attention)",
        "Virchow2\n(Mean Attention)",
        "HCC-SemPath\n(Mean Attention)"
    ]

    for row_idx, c in enumerate(cases):
        rid = c["id"]
        label = c["label"]
        orig_img = case_orig_imgs[rid]
        
        # Subplot list for this row
        row_axes = []
        for col_idx, key in enumerate(col_keys):
            ax = fig.add_subplot(gs[row_idx + 1, col_idx])
            ax.set_box_aspect(1)
            row_axes.append(ax)
            
            if key == "he":
                ax.imshow(orig_img)
            else:
                ax.imshow(orig_img)
                ax.imshow(case_attn_maps[rid][key], cmap="jet", alpha=0.55)
            
            ax.axis("off")
            
            # Set column titles on the first row of Panel B (Row 1 in grid)
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=10, weight="bold", pad=8)

        # Add row labels on the left of Col 0 (H&E)
        ax_he = row_axes[0]
        ax_he.text(
            -0.15, 0.5, f"{label}\n({rid})",
            transform=ax_he.transAxes,
            ha="right", va="center",
            fontsize=11, weight="bold"
        )

    # Annotate A and B labels
    fig.text(0.015, 0.97, "A", fontsize=24, weight="bold")
    fig.text(0.015, 0.73, "B", fontsize=24, weight="bold")

    # Main Title
    fig.suptitle("Latent Space Manifolds and Microscopic Visual Attention Features", fontsize=16, weight="bold", y=0.99)
    
    # Left padding in tight_layout to make room for row labels
    plt.tight_layout(rect=[0.08, 0.01, 0.99, 0.98])

    # Save
    out_dir = REPO_ROOT / "manuscript/figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figure3_umap_attention.pdf", bbox_inches="tight")
    plt.close(fig)

    print("Figure 3 successfully generated and saved to manuscript/figures/figure3_umap_attention.pdf")

    # 6. Cleanup redundant/old files to prevent junk stacking
    redundant_paths = [
        REPO_ROOT / "manuscript/figures/figure5.png",
        REPO_ROOT / "manuscript/figures/figure5.pdf",
        REPO_ROOT / "manuscript/figures/figure5_umap_comparison.png",
        REPO_ROOT / "manuscript/figures/figure5_umap_comparison.pdf",
        REPO_ROOT / "manuscript/figures/zhcc_umap.png",
        REPO_ROOT / "manuscript/figures/zhcc_umap.pdf",
        REPO_ROOT / "experiments/11_representation_umap/results/figure5_umap_comparison.png",
        REPO_ROOT / "experiments/11_representation_umap/results/figure5_umap_comparison.pdf",
        REPO_ROOT / "experiments/11_representation_umap/results/zhcc_umap.png",
        REPO_ROOT / "experiments/11_representation_umap/results/zhcc_umap.pdf",
    ]
    for rp in redundant_paths:
        if rp.exists():
            try:
                rp.unlink()
                print(f"Removed redundant file: {rp.name}")
            except Exception as ex:
                print(f"Failed to remove {rp}: {ex}")

if __name__ == "__main__":
    main()
