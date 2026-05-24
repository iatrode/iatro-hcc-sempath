from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm


MPP_X_PROPERTY = "openslide.mpp-x"
MPP_Y_PROPERTY = "openslide.mpp-y"
OBJECTIVE_POWER_PROPERTY = "openslide.objective-power"
APERIO_APP_MAG_PROPERTY = "aperio.AppMag"


def _float_property(properties, key: str) -> float | None:
    value = properties.get(key)
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def native_mpp_from_properties(properties) -> tuple[float | None, float | None]:
    mpp_x = _float_property(properties, MPP_X_PROPERTY)
    mpp_y = _float_property(properties, MPP_Y_PROPERTY)
    return mpp_x, mpp_y if mpp_y is not None else mpp_x


def objective_power_from_properties(properties) -> float | None:
    objective_power = _float_property(properties, OBJECTIVE_POWER_PROPERTY)
    if objective_power is not None:
        return objective_power
    return _float_property(properties, APERIO_APP_MAG_PROPERTY)


def infer_native_mpp_from_properties(properties) -> tuple[float | None, float | None, str | None]:
    mpp_x, mpp_y = native_mpp_from_properties(properties)
    if mpp_x is not None:
        return mpp_x, mpp_y, "metadata"

    objective_power = objective_power_from_properties(properties)
    if objective_power is None:
        return None, None, None
    if objective_power not in {10.0, 20.0, 40.0, 80.0}:
        return None, None, None

    inferred_mpp = 10.0 / objective_power
    return inferred_mpp, inferred_mpp, "objective_power"


def tissue_fraction(rgb: np.ndarray, white_threshold: int = 220, black_threshold: int = 8) -> float:
    gray = rgb.mean(axis=2)
    return float(((gray > black_threshold) & (gray < white_threshold)).mean())


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
    show_progress: bool = False,
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
        native_mpp, inferred_native_mpp_y, _ = infer_native_mpp_from_properties(slide.properties)
        if native_mpp is None:
            raise ValueError("WSI is missing MPP metadata or a supported objective power; pass --native-mpp explicitly")
    if native_mpp_y is None:
        native_mpp_y = inferred_native_mpp_y if "inferred_native_mpp_y" in locals() else native_mpp
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
    x_count = max(0, ((width - level0_stride_x) // level0_stride_x) + 1)
    y_count = max(0, ((height - level0_stride_y) // level0_stride_y) + 1)
    total_candidates = x_count * y_count
    progress = None
    if show_progress:
        print(
            "tiling_start "
            f"slide={slide_id} size={width}x{height} "
            f"native_mpp=({native_mpp:.4f},{native_mpp_y:.4f}) target_mpp={target_mpp:.4f} "
            f"level={level} level_downsample={level_downsample:.4f} "
            f"stride0=({level0_stride_x},{level0_stride_y}) candidates={total_candidates}",
            flush=True,
        )
        progress = tqdm(total=total_candidates, desc=f"Tiling {slide_id}", unit="tile")
    rows = []
    idx = 0
    try:
        for y in range(0, height - level0_stride_y + 1, level0_stride_y):
            for x in range(0, width - level0_stride_x + 1, level0_stride_x):
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(retained=idx, refresh=False)
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
                    return rows
    finally:
        if progress is not None:
            progress.close()
            tqdm.write(f"tiling_done slide={slide_id} retained={idx} candidates_seen={progress.n}")
        slide.close()
    return rows
