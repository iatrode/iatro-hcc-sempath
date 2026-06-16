#!/usr/bin/env python
import json
import sys
import shutil
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hcc_sempath.modeling.models import HCCSemPathModel
from hcc_sempath.io.tile_package import TilePackageReader

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

# Global list to store attention matrix from hook
attention_matrices = []

def attn_hook(module, input, output):
    if isinstance(output, tuple):
        output = output[0]
    attention_matrices.append(output.cpu())

def main() -> None:
    device = _resolve_device()
    print(f"Using device: {device}")

    # Paths
    student_checkpoint = REPO_ROOT / "artifacts/models/hcc-sempath-full/checkpoints/best_scientific_score.pt"
    student_config = REPO_ROOT / "artifacts/models/hcc-sempath-full/resolved_config.json"
    cases_csv = REPO_ROOT / "experiments/06_attention_qc/configs/reviewed_attention_cases.csv"

    # Load Model
    model, cfg = _load_student(student_checkpoint, student_config, device)
    
    # Register hook on the last block attention dropout layer
    block = model.encoder.backbone.blocks[-1]
    block.attn.fused_attn = False
    hook_handle = block.attn.attn_drop.register_forward_hook(attn_hook)

    # 6 target cases for rows (primary + backup for each of the 3 categories)
    cases = [
        {"id": "TD-0167", "label": "HCC-tumor", "type": "Primary"},
        {"id": "TD-0199", "label": "HCC-tumor", "type": "Secondary"},
        {"id": "TD-0162", "label": "Background-liver", "type": "Primary"},
        {"id": "TD-0418", "label": "Background-liver", "type": "Secondary"},
        {"id": "TD-0342", "label": "Inflammatory-stromal", "type": "Primary"},
        {"id": "TD-0398", "label": "Inflammatory-stromal", "type": "Secondary"}
    ]

    # Load reviewed attention cases metadata
    with cases_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = {row["review_id"]: row for row in csv.DictReader(handle)}
    
    print("Generating attribution grid with Mean Attention across all heads...")
    # Setup figure: 6 rows, 3 columns
    fig, axes = plt.subplots(6, 3, figsize=(9.5, 14.2), gridspec_kw={"hspace": 0.08, "wspace": 0.08})
    
    for row_idx, case in enumerate(cases):
        rid = case["id"]
        label = case["label"]
        case_type = case["type"]
        
        row_info = rows[rid]
        
        # 1. Load original H&E image using TilePackageReader
        pkg_path = REPO_ROOT / row_info["package_path"]
        reader = TilePackageReader(pkg_path)
        orig_img = reader.read_image_at(int(row_info["row_idx"])).convert("RGB")
        reader.close()
        
        # 2. Extract attention map
        img_224 = orig_img.resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.array(img_224, dtype=np.uint8)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32) / 255.0
        mean = torch.tensor(cfg["data"]["mean"], device=device).view(1, 3, 1, 1)
        std = torch.tensor(cfg["data"]["std"], device=device).view(1, 3, 1, 1)
        x = (tensor - mean) / std

        attention_matrices.clear()
        with torch.no_grad():
            _ = model(x)
        
        # Shape: (num_heads, 197, 197)
        attn = attention_matrices[0][0]
        # Average attention from CLS token to all patches across all heads
        cls_attn = attn[:, 0, 1:].mean(dim=0).view(14, 14).numpy()
        # Normalize using robust percentile scaling
        vmin, vmax = np.percentile(cls_attn, 2), np.percentile(cls_attn, 98)
        cls_attn = np.clip((cls_attn - vmin) / (vmax - vmin + 1e-8), 0, 1)
        # Interpolate back to 224x224
        cls_attn_img = Image.fromarray((cls_attn * 255).astype(np.uint8)).resize((224, 224), Image.Resampling.BILINEAR)
        
        # 3. Load L1 map from results directory
        l1_img_path = REPO_ROOT / "experiments/06_attention_qc/results" / f"{row_info['tile_id']}.occlusion.png"
        l1_img = Image.open(l1_img_path).convert("RGB")
        
        # Plot Col 0: H&E tile
        ax_he = axes[row_idx, 0]
        ax_he.imshow(img_224)
        ax_he.axis("off")
        ax_he.set_box_aspect(1)
        
        # Plot Col 1: ViT Attention (Mean Attention)
        ax_attn = axes[row_idx, 1]
        ax_attn.imshow(img_224)
        ax_attn.imshow(cls_attn_img, cmap="jet", alpha=0.55)
        ax_attn.axis("off")
        ax_attn.set_box_aspect(1)
        
        # Plot Col 2: L1 Sensitivity
        ax_l1 = axes[row_idx, 2]
        ax_l1.imshow(l1_img.resize((224, 224), Image.Resampling.BILINEAR))
        ax_l1.axis("off")
        ax_l1.set_box_aspect(1)
        
        # Add row labels on the left of Col 0
        row_label = f"{label}\n({rid})"
        ax_he.text(
            -0.15, 0.5, row_label,
            transform=ax_he.transAxes,
            ha="right", va="center",
            fontsize=11, weight="bold"
        )
        
    # Col titles
    axes[0, 0].set_title("H&E Tile", fontsize=11, weight="bold", pad=8)
    axes[0, 1].set_title(f"ViT Attention\n(Mean Attention)", fontsize=11, weight="bold", pad=8)
    axes[0, 2].set_title("L1 Decision Sensitivity\n(Perturbed Attribution)", fontsize=11, weight="bold", pad=8)
    
    # Save output
    output_dir = REPO_ROOT / "manuscript/figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout(rect=[0.18, 0.01, 0.99, 0.98])
    
    fig.savefig(output_dir / "figure5_microscopic_attribution.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "figure5_microscopic_attribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Also save to experiments reports directory if exists
    exp_reports_dir = REPO_ROOT / "experiments/06_attention_qc/reports"
    if exp_reports_dir.exists():
        shutil.copyfile(output_dir / "figure5_microscopic_attribution.pdf", exp_reports_dir / "figure5_microscopic_attribution.pdf")
        print(f"Copied figure 5 PDF to {exp_reports_dir}")
    
    hook_handle.remove()
    print("Figure 5 successfully generated and saved to manuscript/figures/")

if __name__ == "__main__":
    main()
