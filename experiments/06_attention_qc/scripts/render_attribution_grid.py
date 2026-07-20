#!/usr/bin/env python
import csv
import json
import sys
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from iatro.iac.adapters.tiles import TilePackageReader
from hcc_sempath.modeling.models import HCCSemPathModel
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
    return model

# Global list to store attention matrix from hook
attention_matrices = []

def attn_hook(module, input, output):
    # output shape is expected to be (batch_size, num_heads, seq_len, seq_len)
    attention_matrices.append(output.cpu())

def main() -> None:
    device = _resolve_device()
    print(f"Using device: {device}")

    # Paths
    predictions_csv = REPO_ROOT / "artifacts/caches/local_cache/teacher_disagreement/teacher_disagreement_model_predictions.csv"
    review_csv = REPO_ROOT / "annotations/reviews/teacher_disagreement/exval_1000/review.csv"
    student_checkpoint = REPO_ROOT / "artifacts/models/hcc-sempath-full/checkpoints/best_scientific_score.pt"
    student_config = REPO_ROOT / "artifacts/models/hcc-sempath-full/resolved_config.json"
    cache_dir = Path("/tmp/hcc_sempath_exval_selector/cache")

    with student_config.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    # 1. Load predictions & reviews
    with predictions_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = {row["review_id"]: row for row in csv.DictReader(handle)}
    with review_csv.open("r", newline="", encoding="utf-8") as handle:
        reviews = {row["review_id"]: row for row in csv.DictReader(handle)}

    # Target 3 Primary Cases
    targets = {
        "HCC-tumor": "TD-0038",
        "Background-liver": "TD-0132",
        "Inflammatory-stromal": "TD-0028"
    }

    # Load Model
    model = _load_student(student_checkpoint, student_config, device)
    
    # Register hook on the last block attention dropout layer
    # For vit_small_patch16_224 in timm, we can hook attn_drop
    block = model.encoder.backbone.blocks[-1]
    block.attn.fused_attn = False
    hook_handle = block.attn.attn_drop.register_forward_hook(attn_hook)

    output_temp_dir = REPO_ROOT / "experiments/06_attention_qc/reports/temp_attention"
    output_temp_dir.mkdir(parents=True, exist_ok=True)

    for gName, rid in targets.items():
        print(f"Processing case {rid} for group {gName}...")
        case_info = reviews[rid]
        
        # Read image
        reader = TilePackageReader(REPO_ROOT / case_info["package_path"])
        pil_img = reader.read_image_at(int(case_info["row_idx"])).convert("RGB").resize((224, 224))
        reader.close()

        # Prepare tensor
        arr = np.array(pil_img, dtype=np.uint8)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32) / 255.0
        mean = torch.tensor(cfg["data"]["mean"], device=device).view(1, 3, 1, 1)
        std = torch.tensor(cfg["data"]["std"], device=device).view(1, 3, 1, 1)
        x = (tensor - mean) / std

        # Inference to trigger hook
        attention_matrices.clear()
        with torch.no_grad():
            _ = model(x)
        
        # We should have attention_matrices[0] with shape (1, num_heads, 197, 197)
        # where 197 is 1 CLS token + 196 patch tokens
        attn = attention_matrices[0][0] # shape: (num_heads, 197, 197)
        num_heads = attn.shape[0]
        print(f"Extracted attention map with {num_heads} heads.")

        # Plot all heads for selection
        fig, axes = plt.subplots(1, num_heads, figsize=(3 * num_heads, 3))
        for head_idx in range(num_heads):
            # Attention from CLS token to all patches (tokens 1 to 196)
            cls_attn = attn[head_idx, 0, 1:].view(14, 14).numpy()
            # Normalize for visual clarity
            cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
            
            # Interpolate to 224x224
            cls_attn_img = Image.fromarray((cls_attn * 255).astype(np.uint8)).resize((224, 224), Image.Resampling.BILINEAR)
            
            # Plot
            ax = axes[head_idx] if num_heads > 1 else axes
            ax.imshow(pil_img)
            ax.imshow(cls_attn_img, cmap="jet", alpha=0.55)
            ax.set_title(f"Head {head_idx}")
            ax.axis("off")
        
        fig.suptitle(f"Case {rid} ({gName}) - Layer 11 Attention Heads", fontsize=12, weight="bold")
        plt.tight_layout()
        fig.savefig(output_temp_dir / f"attention_heads_{rid}.png", dpi=150)
        plt.close(fig)
        print(f"Saved attention heads comparison for {rid} to {output_temp_dir}")

    hook_handle.remove()
    print("Hook removed successfully.")

if __name__ == "__main__":
    main()
