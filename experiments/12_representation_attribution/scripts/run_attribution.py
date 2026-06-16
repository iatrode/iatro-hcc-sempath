#!/usr/bin/env python
import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add src/ to python path
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hcc_sempath.io.tile_package import TilePackageReader
from hcc_sempath.modeling.models import HCCSemPathModel
from hcc_sempath.teacher.cache import TimmTeacherEncoder
from hcc_sempath.training.config import embedding_dim, teacher_dims, teacher_names
from hcc_sempath.training.engine import _prepare_images

# Configuration maps for local teacher model paths
TEACHER_MODELS = {
    "gigapath": "artifacts/models/teachers/gigapath",
    "h_optimus_1": "artifacts/models/teachers/h_optimus_1",
    "uni2_h": "artifacts/models/teachers/uni2_h",
    "virchow2": "artifacts/models/teachers/virchow2",
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

@torch.no_grad()
def _compute_attribution(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,  # shape (3, H, W)
    device: torch.device,
    is_student: bool = False,
    batch_size: int = 32,
    patch_size: int = 16,
) -> np.ndarray:
    height, width = image_tensor.shape[-2:]
    grid_h = height // patch_size
    grid_w = width // patch_size

    # Prepare batches: HWC image input
    # Original embedding z
    img_input = image_tensor.unsqueeze(0).to(device)
    if is_student:
        z = model(img_input)["embedding_norm"][0]
    else:
        z = model(img_input)[0]
        z = F.normalize(z, dim=-1)

    variants = []
    for row in range(grid_h):
        for col in range(grid_w):
            variant = image_tensor.clone()
            y0 = row * patch_size
            x0 = col * patch_size
            # Mask the patch by setting to 0 (mean background color in normalized space)
            variant[:, y0 : y0 + patch_size, x0 : x0 + patch_size] = 0.0
            variants.append(variant)

    shifts = []
    for start in range(0, len(variants), batch_size):
        batch = torch.stack(variants[start : start + batch_size]).to(device)
        if is_student:
            z_p = model(batch)["embedding_norm"]
        else:
            z_p = model(batch)
            z_p = F.normalize(z_p, dim=-1)
        
        # Cosine distance = 1 - cosine_similarity
        dist = 1.0 - (z.unsqueeze(0) * z_p).sum(dim=-1)
        shifts.extend(dist.detach().cpu().tolist())

    heatmap = torch.tensor(shifts, dtype=torch.float32).reshape(1, 1, grid_h, grid_w)
    heatmap = F.interpolate(heatmap, size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    heatmap = heatmap.clamp_min(0)
    norm_max = heatmap.max().clamp_min(1e-6)
    heatmap = heatmap / norm_max
    return heatmap.numpy()

def _overlay(image: Image.Image, heatmap: np.ndarray) -> np.ndarray:
    base = np.asarray(image.resize((heatmap.shape[1], heatmap.shape[0])), dtype=np.float32) / 255.0
    color = plt.get_cmap("magma")(heatmap)[..., :3]
    return np.clip(0.58 * base + 0.42 * color, 0.0, 1.0)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run representation-level attribution contrast.")
    parser.add_argument(
        "--cases-csv",
        default="experiments/06_attention_qc/configs/reviewed_attention_cases.csv",
        help="Path to reviewed attention cases CSV"
    )
    parser.add_argument(
        "--tile-ids",
        nargs="*",
        default=["TD-0864", "TD-0477", "TD-0459", "TD-0363"],
        help="Specific tile review IDs to plot"
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
        default="experiments/12_representation_attribution/results",
        help="Output directory for results"
    )
    args = parser.parse_args()

    device = _resolve_device()
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load configuration and metadata
    cases_path = REPO_ROOT / args.cases_csv
    if not cases_path.exists():
        print(f"Error: Cases CSV not found at {cases_path}")
        return

    with cases_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    
    selected_rows = [row for row in rows if row["review_id"] in args.tile_ids]
    if not selected_rows:
        print(f"No selected tiles found matching ids: {args.tile_ids}")
        return

    # 2. Instantiate all models online
    print("Loading student model...")
    student_model = _load_student(REPO_ROOT / args.student_checkpoint, REPO_ROOT / args.student_config, device)

    teachers = {}
    for name, path in TEACHER_MODELS.items():
        full_path = REPO_ROOT / path
        if full_path.exists():
            print(f"Loading teacher model: {name}...")
            teachers[name] = TimmTeacherEncoder(str(full_path)).to(device)
            teachers[name].eval()
        else:
            print(f"Warning: Teacher path {full_path} not found. Skipping.")

    # 3. Read config to get norm mean/std
    with (REPO_ROOT / args.student_config).open("r", encoding="utf-8") as handle:
        import json
        cfg = json.load(handle)

    # 4. Generate panels
    fig, axes = plt.subplots(len(selected_rows), 6, figsize=(15, 2.5 * len(selected_rows)), squeeze=False)
    
    headers = ["H&E Tile", "GigaPath", "H-optimus-1", "UNI2-h", "Virchow2", "Student z_HCC"]
    for col_idx, header in enumerate(headers):
        axes[0, col_idx].set_title(header, fontsize=11, weight="bold")

    readers = {}
    try:
        for row_idx, row in enumerate(selected_rows):
            print(f"Processing tile {row['review_id']} ({row['tile_id']})...")
            pkg_path = Path(row["package_path"])
            if not pkg_path.is_absolute():
                pkg_path = REPO_ROOT / pkg_path
            
            if str(pkg_path) not in readers:
                readers[str(pkg_path)] = TilePackageReader(pkg_path)
            
            reader = readers[str(pkg_path)]
            pil_img = reader.read_image_at(int(row["row_idx"])).convert("RGB")
            
            # Preprocess image
            arr = torch.from_numpy(np.array(pil_img, dtype=np.uint8, copy=True)).permute(2, 0, 1)
            batch = {"images": arr.unsqueeze(0), "images_uint8": True}
            norm_image = _prepare_images(batch, cfg, device)[0]

            # Display H&E Tile
            axes[row_idx, 0].imshow(pil_img)
            axes[row_idx, 0].axis("off")
            axes[row_idx, 0].text(
                0.05, 0.05, row["review_id"],
                transform=axes[row_idx, 0].transAxes,
                color="white", weight="bold", fontsize=10,
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=2)
            )

            # Compute and display teacher attributions
            col_map = {"gigapath": 1, "h_optimus_1": 2, "uni2_h": 3, "virchow2": 4}
            for name, col in col_map.items():
                if name in teachers:
                    attr = _compute_attribution(teachers[name], norm_image, device, is_student=False)
                    overlay = _overlay(pil_img, attr)
                    axes[row_idx, col].imshow(overlay)
                else:
                    # Placeholder if teacher is missing
                    axes[row_idx, col].text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12)
                axes[row_idx, col].axis("off")

            # Compute and display student attribution
            attr_student = _compute_attribution(student_model, norm_image, device, is_student=True)
            overlay_student = _overlay(pil_img, attr_student)
            axes[row_idx, 5].imshow(overlay_student)
            axes[row_idx, 5].axis("off")

    finally:
        for r in readers.values():
            r.close()

    plt.tight_layout()
    
    # Save results
    output_pdf = output_dir / "representation_attribution_panel.pdf"
    output_png = output_dir / "representation_attribution_panel.png"
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Representation attribution panel generated successfully at {output_pdf}")

if __name__ == "__main__":
    main()
