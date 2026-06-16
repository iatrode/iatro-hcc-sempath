#!/usr/bin/env python
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from timm.data import create_transform, resolve_model_data_config

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hcc_sempath.teacher.cache import TimmTeacherEncoder, _resolve_model_spec

# Global list to store attention matrix from hook
attention_matrices = []

def attn_hook(module, input, output):
    # output shape is expected to be (batch_size, num_heads, seq_len, seq_len)
    attention_matrices.append(output.cpu())

def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # TD-0038 image path
    orig_img_path = Path("/tmp/hcc_sempath_exval_selector/cache/TD-0038/original.png")
    orig_img = Image.open(orig_img_path).convert("RGB")

    teachers = ["gigapath", "h_optimus_1", "uni2_h", "virchow2"]
    
    output_temp_dir = REPO_ROOT / "experiments/06_attention_qc/reports/temp_attention"
    output_temp_dir.mkdir(parents=True, exist_ok=True)

    for t in teachers:
        print(f"Loading {t} model...")
        spec = _resolve_model_spec("artifacts/models/teachers/" + t)
        encoder = TimmTeacherEncoder(spec["model_name"], model_kwargs=spec["model_kwargs"]).to(device)
        encoder.eval()

        # Build transform
        data_config = resolve_model_data_config(encoder.model)
        data_config["input_size"] = (3, 224, 224)
        transform = create_transform(**data_config, is_training=False)

        # Hook setup
        block = encoder.model.blocks[-1]
        block.attn.fused_attn = False
        hook_handle = block.attn.attn_drop.register_forward_hook(attn_hook)

        # Inference
        img_tensor = transform(orig_img).unsqueeze(0).to(device)
        attention_matrices.clear()
        
        with torch.no_grad():
            _ = encoder(img_tensor)

        # Extract attention weights
        attn = attention_matrices[0][0] # shape: (num_heads, seq_len, seq_len)
        num_heads = attn.shape[0]
        
        # Determine sequence length
        seq_len = attn.shape[1]
        # Normally seq_len is 1 CLS + 196 patch tokens = 197.
        # But UNI2-h has reg_tokens = 8, so it is 1 CLS + 8 registers + 196 patches = 205.
        # Virchow2 has reg_tokens = 4, so it is 1 CLS + 4 registers + 196 patches = 201.
        # We need to extract the attention from CLS token (index 0) to all patch tokens (which are the last 196 tokens).
        # Let's dynamically find patch token indices: they are the last 196 tokens.
        cls_to_patches = attn[:, 0, -196:] # shape: (num_heads, 196)

        # Plot all heads
        print(f"Plotting {num_heads} heads for {t}...")
        fig_cols = 6
        fig_rows = (num_heads + fig_cols - 1) // fig_cols
        fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(3 * fig_cols, 3 * fig_rows))
        
        for head_idx in range(num_heads):
            cls_attn = cls_to_patches[head_idx].view(14, 14).numpy()
            # Normalize
            cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
            # Resize
            cls_attn_img = Image.fromarray((cls_attn * 255).astype(np.uint8)).resize((224, 224), Image.Resampling.BILINEAR)
            
            # Plot
            ax = axes.flat[head_idx]
            ax.imshow(orig_img.resize((224, 224)))
            ax.imshow(cls_attn_img, cmap="jet", alpha=0.55)
            ax.set_title(f"Head {head_idx}")
            ax.axis("off")
            
        # Hide extra axes
        for ax in axes.flat[num_heads:]:
            ax.axis("off")
            
        fig.suptitle(f"{t.upper()} - Layer 11 Attention Heads for TD-0038", fontsize=14, weight="bold")
        plt.tight_layout()
        fig.savefig(output_temp_dir / f"heads_scan_{t}.png", dpi=120)
        plt.close(fig)
        
        hook_handle.remove()
        print(f"Saved heads scan for {t} to {output_temp_dir}")

if __name__ == "__main__":
    main()
