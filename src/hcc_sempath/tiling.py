from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .manifests import write_tile_manifest


def tissue_fraction(rgb: np.ndarray, white_threshold: int = 220) -> float:
    gray = rgb.mean(axis=2)
    return float((gray < white_threshold).mean())


def iter_image_tiles(image: Image.Image, tile_size: int, min_tissue_fraction: float):
    image = image.convert("RGB")
    width, height = image.size
    for y in range(0, height - tile_size + 1, tile_size):
        for x in range(0, width - tile_size + 1, tile_size):
            tile = image.crop((x, y, x + tile_size, y + tile_size))
            arr = np.asarray(tile)
            if tissue_fraction(arr) >= min_tissue_fraction:
                yield x, y, tile


def select_read_level(level_downsamples: tuple[float, ...] | list[float], native_mpp: float, target_mpp: float) -> int:
    """Approximate OpenSlide's best level choice for target/native downsampling."""
    downsample_needed = target_mpp / native_mpp
    if downsample_needed <= 1:
        return 0
    return min(
        range(len(level_downsamples)),
        key=lambda idx: abs(float(level_downsamples[idx]) - downsample_needed),
    )


def tile_raster_image(
    image_path: str | Path,
    output_dir: str | Path,
    patient_id: str,
    slide_id: str,
    split: str,
    tile_size: int = 224,
    min_tissue_fraction: float = 0.1,
    overwrite_slide_dir: bool = False,
) -> list[dict]:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    slide_dir = output_dir / slide_id
    if overwrite_slide_dir and slide_dir.exists():
        shutil.rmtree(slide_dir)
    slide_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with Image.open(image_path) as image:
        for idx, (x, y, tile) in enumerate(iter_image_tiles(image, tile_size, min_tissue_fraction)):
            tile_id = f"{slide_id}_{idx:07d}"
            tile_path = slide_dir / f"{tile_id}.png"
            tile.save(tile_path)
            rows.append(
                {
                    "tile_id": tile_id,
                    "patient_id": patient_id,
                    "slide_id": slide_id,
                    "tile_path": str(tile_path),
                    "x": x,
                    "y": y,
                    "split": split,
                }
            )
    return rows


def tile_wsi(
    wsi_path: str | Path,
    output_dir: str | Path,
    patient_id: str,
    slide_id: str,
    split: str,
    tile_size: int = 224,
    min_tissue_fraction: float = 0.1,
    target_mpp: float = 0.5,
    native_mpp: float | None = None,
    native_mpp_y: float | None = None,
    max_tiles: int | None = None,
    overwrite_slide_dir: bool = False,
) -> list[dict]:
    try:
        import openslide
    except ImportError as exc:
        raise RuntimeError("openslide-python is required for WSI tiling") from exc
    output_dir = Path(output_dir)
    slide_dir = output_dir / slide_id
    if overwrite_slide_dir and slide_dir.exists():
        shutil.rmtree(slide_dir)
    slide_dir.mkdir(parents=True, exist_ok=True)
    slide = openslide.OpenSlide(str(wsi_path))
    if native_mpp is None:
        mpp_value = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        if mpp_value is None:
            raise ValueError("WSI is missing MPP metadata; pass --native-mpp explicitly")
        native_mpp = float(mpp_value)
    if native_mpp_y is None:
        mpp_y_value = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
        native_mpp_y = float(mpp_y_value) if mpp_y_value is not None else native_mpp
    downsample_needed = target_mpp / native_mpp
    level = slide.get_best_level_for_downsample(downsample_needed)
    level_downsample = float(slide.level_downsamples[level])
    scale_x = native_mpp / target_mpp
    scale_y = native_mpp_y / target_mpp
    level0_stride_x = max(1, round(tile_size / scale_x))
    level0_stride_y = max(1, round(tile_size / scale_y))
    level_read_w = max(1, round(level0_stride_x / level_downsample))
    level_read_h = max(1, round(level0_stride_y / level_downsample))
    width, height = slide.dimensions
    rows = []
    idx = 0
    for y in range(0, height - level0_stride_y + 1, level0_stride_y):
        for x in range(0, width - level0_stride_x + 1, level0_stride_x):
            tile = slide.read_region((x, y), level, (level_read_w, level_read_h)).convert("RGB")
            if tile.size != (tile_size, tile_size):
                tile = tile.resize((tile_size, tile_size), Image.Resampling.BICUBIC)
            if tissue_fraction(np.asarray(tile)) < min_tissue_fraction:
                continue
            tile_id = f"{slide_id}_{idx:07d}"
            tile_path = slide_dir / f"{tile_id}.png"
            tile.save(tile_path)
            rows.append(
                {
                    "tile_id": tile_id,
                    "patient_id": patient_id,
                    "slide_id": slide_id,
                    "tile_path": str(tile_path),
                    "x": x,
                    "y": y,
                    "split": split,
                }
            )
            idx += 1
            if max_tiles is not None and idx >= max_tiles:
                slide.close()
                return rows
    slide.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Tile a raster image or WSI into fixed-size patches.")
    parser.add_argument("--image", default="")
    parser.add_argument("--wsi", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--patient-id", required=True)
    parser.add_argument("--slide-id", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.1)
    parser.add_argument("--target-mpp", type=float, default=0.5)
    parser.add_argument("--native-mpp", type=float, default=None)
    parser.add_argument("--native-mpp-y", type=float, default=None)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--overwrite-slide-dir", action="store_true")
    args = parser.parse_args()
    if bool(args.image) == bool(args.wsi):
        raise ValueError("provide exactly one of --image or --wsi")
    rows = tile_raster_image(
        image_path=args.image,
        output_dir=args.output_dir,
        patient_id=args.patient_id,
        slide_id=args.slide_id,
        split=args.split,
        tile_size=args.tile_size,
        min_tissue_fraction=args.min_tissue_fraction,
        overwrite_slide_dir=args.overwrite_slide_dir,
    ) if args.image else tile_wsi(
        wsi_path=args.wsi,
        output_dir=args.output_dir,
        patient_id=args.patient_id,
        slide_id=args.slide_id,
        split=args.split,
        tile_size=args.tile_size,
        min_tissue_fraction=args.min_tissue_fraction,
        target_mpp=args.target_mpp,
        native_mpp=args.native_mpp,
        native_mpp_y=args.native_mpp_y,
        max_tiles=args.max_tiles,
        overwrite_slide_dir=args.overwrite_slide_dir,
    )
    write_tile_manifest(args.manifest_out, rows)
    print(f"wrote_tiles={len(rows)} manifest={args.manifest_out}")


if __name__ == "__main__":
    main()
