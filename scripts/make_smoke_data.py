from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from hcc_sempath.manifests import write_tile_manifest
from hcc_sempath.tiling import tile_raster_image


def main() -> None:
    root = Path("smoke_data")
    slides_dir = root / "slides"
    tiles_dir = root / "tiles"
    cache_dir = root / "teacher_cache"
    slides_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(13)
    rows = []
    teacher_dim = 256
    specs = [("p_train", "s_train", "train"), ("p_val", "s_val", "val")]
    for patient_id, slide_id, split in specs:
        slide_path = slides_dir / f"{slide_id}.png"
        arr = (rng.random((448, 448, 3)) * 180).astype(np.uint8)
        Image.fromarray(arr).save(slide_path)
        rows.extend(
            tile_raster_image(
                image_path=slide_path,
                output_dir=tiles_dir,
                patient_id=patient_id,
                slide_id=slide_id,
                split=split,
                tile_size=224,
                min_tissue_fraction=0.0,
            )
        )
    for row in rows:
        np.save(cache_dir / f"{row['tile_id']}.npy", rng.normal(size=(teacher_dim,)).astype(np.float32))
    write_tile_manifest(root / "tile_manifest.csv", rows)
    torch.save({"anchors": torch.randn(7, teacher_dim)}, root / "anchors.pt")
    print("smoke_data_ok")


if __name__ == "__main__":
    main()
