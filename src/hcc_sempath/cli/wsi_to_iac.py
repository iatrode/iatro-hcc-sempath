from __future__ import annotations

import os
from pathlib import Path
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import threading

import numpy as np
from PIL import Image
from tqdm import tqdm

from hcc_sempath.io.manifests import TileRecord
from hcc_sempath.io.qc import render_tile_package_qc
from hcc_sempath.io.tile_package import build_tile_package_from_records, encode_jxl_array
from hcc_sempath.io.tiling import tissue_fraction


_thread_state = threading.local()


def _default_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def _compression_stats(input_path: Path, output_path: Path) -> dict:
    input_bytes = input_path.stat().st_size
    package_bytes = output_path.stat().st_size
    compression_ratio = round(input_bytes / package_bytes, 4) if package_bytes > 0 else 0.0
    space_saving_pct = round((1.0 - package_bytes / input_bytes) * 100.0, 3) if input_bytes > 0 else 0.0
    return {
        "input_bytes": input_bytes,
        "package_bytes": package_bytes,
        "compression_ratio": compression_ratio,
        "space_saving_pct": space_saving_pct,
    }


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024.0
    return f"{size}B"


def _open_slide(wsi_path: Path):
    try:
        import openslide
    except ImportError as exc:
        raise RuntimeError("openslide-python is required for WSI packaging") from exc
    return openslide, openslide.OpenSlide(str(wsi_path))


def _get_thread_slide(wsi_path: Path):
    slide = getattr(_thread_state, "slide", None)
    if slide is None:
        _, slide = _open_slide(wsi_path)
        _thread_state.slide = slide
    return slide


def _choose_mask_level(slide, max_pixels: int = 12_000_000) -> int:
    candidates = []
    for idx, (width, height) in enumerate(slide.level_dimensions):
        pixels = int(width) * int(height)
        if pixels <= max_pixels:
            candidates.append((idx, pixels))
    if candidates:
        return min(candidates, key=lambda item: item[0])[0]
    return len(slide.level_dimensions) - 1


def _build_tissue_mask(slide, mask_level: int, white_threshold: int) -> np.ndarray:
    width, height = slide.level_dimensions[mask_level]
    image = slide.read_region((0, 0), mask_level, (width, height)).convert("RGB")
    arr = np.asarray(image)
    return arr.mean(axis=2) < white_threshold


def _candidate_grid(width: int, height: int, stride_x: int, stride_y: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(0, width - stride_x + 1, stride_x, dtype=np.int64)
    ys = np.arange(0, height - stride_y + 1, stride_y, dtype=np.int64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return grid_x.ravel(), grid_y.ravel()


def _prefilter_candidates(
    *,
    mask: np.ndarray,
    candidate_x: np.ndarray,
    candidate_y: np.ndarray,
    tile_w: int,
    tile_h: int,
    downsample: float,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    integral = np.pad(mask.astype(np.uint32), ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    x0 = np.maximum(0, (candidate_x / downsample).astype(np.int64))
    y0 = np.maximum(0, (candidate_y / downsample).astype(np.int64))
    x1 = np.minimum(mask.shape[1], ((candidate_x + tile_w) / downsample).astype(np.int64))
    y1 = np.minimum(mask.shape[0], ((candidate_y + tile_h) / downsample).astype(np.int64))
    x1 = np.maximum(x1, x0 + 1)
    y1 = np.maximum(y1, y0 + 1)
    x1 = np.minimum(x1, mask.shape[1])
    y1 = np.minimum(y1, mask.shape[0])
    area = np.maximum(1, (x1 - x0) * (y1 - y0))
    tissue = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    keep = tissue / area >= threshold
    return candidate_x[keep], candidate_y[keep]


def _flush_done(done, records: list[TileRecord], payloads: list[bytes]) -> None:
    for future in done:
        result = future.result()
        if result is None:
            continue
        _, record, payload = result
        records.append(record)
        payloads.append(payload)


def _read_encode_record(
    *,
    wsi_path: Path,
    tile_idx: int,
    x: int,
    y: int,
    slide_id: str,
    patient_id: str,
    split: str,
    level: int,
    level_read_w: int,
    level_read_h: int,
    tile_size: int,
    min_tissue_fraction: float,
    white_threshold: int,
    lossless: bool,
    distance: float,
    effort: int,
) -> tuple[int, TileRecord, bytes] | None:
    slide = _get_thread_slide(wsi_path)
    tile = slide.read_region((x, y), level, (level_read_w, level_read_h)).convert("RGB")
    if tile.size != (tile_size, tile_size):
        tile = tile.resize((tile_size, tile_size), Image.Resampling.BICUBIC)
    arr = np.asarray(tile).copy()
    if tissue_fraction(arr, white_threshold=white_threshold) < min_tissue_fraction:
        return None
    tile_id = f"{slide_id}_{tile_idx:07d}"
    record = TileRecord(
        tile_id=tile_id,
        patient_id=patient_id,
        slide_id=slide_id,
        tile_path=Path(f"tiles/{tile_id}.jxl"),
        x=x,
        y=y,
        split=split,
    )
    payload = encode_jxl_array(arr, lossless=lossless, distance=None if lossless else distance, effort=effort)
    return tile_idx, record, payload


def build_wsi_iac(
    *,
    wsi_path: str | Path,
    output_path: str | Path,
    patient_id: str | None = None,
    slide_id: str | None = None,
    split: str = "train",
    target_mpp: float = 0.5,
    native_mpp: float | None = None,
    native_mpp_y: float | None = None,
    tile_size: int = 224,
    min_tissue_fraction: float = 0.1,
    max_tiles: int | None = None,
    lossless: bool = False,
    distance: float = 1.0,
    effort: int = 7,
    workers: int = 1,
    white_threshold: int = 220,
    prefilter_tissue_fraction: float = 0.05,
    mask_max_pixels: int = 12_000_000,
    qc_out: str | Path | None = None,
    qc_max_tiles: int = 36,
    overwrite: bool = False,
    show_progress: bool = True,
) -> dict:
    wsi_path = Path(wsi_path)
    output_path = Path(output_path)
    slide_id = slide_id or wsi_path.stem
    patient_id = patient_id or slide_id

    openslide, slide = _open_slide(wsi_path)
    if native_mpp is None:
        mpp_value = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        if mpp_value is None:
            slide.close()
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
    x_count = max(0, ((width - level0_stride_x) // level0_stride_x) + 1)
    y_count = max(0, ((height - level0_stride_y) // level0_stride_y) + 1)
    total_candidates = x_count * y_count
    candidate_x, candidate_y = _candidate_grid(width, height, level0_stride_x, level0_stride_y)
    mask_level = _choose_mask_level(slide, max_pixels=mask_max_pixels)
    mask_downsample = float(slide.level_downsamples[mask_level])
    mask = _build_tissue_mask(slide, mask_level, white_threshold=white_threshold)
    selected_x, selected_y = _prefilter_candidates(
        mask=mask,
        candidate_x=candidate_x,
        candidate_y=candidate_y,
        tile_w=level0_stride_x,
        tile_h=level0_stride_y,
        downsample=mask_downsample,
        threshold=prefilter_tissue_fraction,
    )
    selected_candidates = len(selected_x)

    if show_progress:
        print(
            "wsi_direct_pack_start "
            f"slide={slide_id} size={width}x{height} "
            f"native_mpp=({native_mpp:.4f},{native_mpp_y:.4f}) target_mpp={target_mpp:.4f} "
            f"level={level} level_downsample={level_downsample:.4f} "
            f"stride0=({level0_stride_x},{level0_stride_y}) candidates={total_candidates} "
            f"mask_level={mask_level} mask_downsample={mask_downsample:.4f} "
            f"prefilter_tissue_fraction={prefilter_tissue_fraction:.4f} selected={selected_candidates}",
            flush=True,
        )
    progress = tqdm(total=selected_candidates, desc=f"Packing {slide_id}", unit="tile") if show_progress else None
    records: list[TileRecord] = []
    payloads: list[bytes] = []
    pending = set()
    submitted = 0
    workers = max(1, int(workers))

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for x, y in zip(selected_x, selected_y):
                if max_tiles is not None and len(records) >= max_tiles:
                    break
                pending.add(
                    executor.submit(
                        _read_encode_record,
                        wsi_path=wsi_path,
                        tile_idx=submitted,
                        x=int(x),
                        y=int(y),
                        slide_id=slide_id,
                        patient_id=patient_id,
                        split=split,
                        level=level,
                        level_read_w=level_read_w,
                        level_read_h=level_read_h,
                        tile_size=tile_size,
                        min_tissue_fraction=min_tissue_fraction,
                        white_threshold=white_threshold,
                        lossless=lossless,
                        distance=distance,
                        effort=effort,
                    )
                )
                submitted += 1
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(retained=len(records), refresh=False)
                if len(pending) >= workers * 4:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    _flush_done(done, records, payloads)
                    if max_tiles is not None and len(records) >= max_tiles:
                        break
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                _flush_done(done, records, payloads)
                if progress is not None:
                    progress.set_postfix(retained=len(records), refresh=False)
                if max_tiles is not None and len(records) >= max_tiles:
                    break
    finally:
        if progress is not None:
            progress.close()
            tqdm.write(f"wsi_direct_pack_tiles_done slide={slide_id} retained={len(records)} candidates_seen={progress.n}")
        slide.close()

    ordered = sorted(zip(records, payloads), key=lambda item: item[0].tile_id)
    if max_tiles is not None:
        ordered = ordered[:max_tiles]
    records = []
    payloads = [payload for _, payload in ordered]
    for tile_idx, (record, _) in enumerate(ordered):
        tile_id = f"{slide_id}_{tile_idx:07d}"
        records.append(
            TileRecord(
                tile_id=tile_id,
                patient_id=record.patient_id,
                slide_id=record.slide_id,
                tile_path=Path(f"tiles/{tile_id}.jxl"),
                x=record.x,
                y=record.y,
                split=record.split,
            )
        )
    if not records:
        raise ValueError(f"no tiles retained from {wsi_path}; lower --min-tissue-fraction or check the slide")

    build_tile_package_from_records(
        records=records,
        payloads=payloads,
        output_path=output_path,
        tile_width=tile_size,
        tile_height=tile_size,
        lossless=lossless,
        distance=None if lossless else distance,
        effort=effort,
        overwrite=overwrite,
        stride_x=level0_stride_x,
        stride_y=level0_stride_y,
        extra_header={
            "source": {
                "path": str(wsi_path),
                "bytes": wsi_path.stat().st_size,
                "width": width,
                "height": height,
                "native_mpp_x": native_mpp,
                "native_mpp_y": native_mpp_y,
            },
            "tiling": {
                "target_mpp": target_mpp,
                "openslide_level": level,
                "level_downsample": level_downsample,
                "level_read_width": level_read_w,
                "level_read_height": level_read_h,
                "min_tissue_fraction": min_tissue_fraction,
                "prefilter_tissue_fraction": prefilter_tissue_fraction,
                "mask_level": mask_level,
                "mask_downsample": mask_downsample,
                "mask_shape": list(mask.shape),
                "candidate_tiles": total_candidates,
                "prefiltered_tiles": selected_candidates,
                "retained_tiles": len(records),
                "max_tiles": max_tiles,
            },
        },
    )

    if qc_out:
        render_tile_package_qc(output_path, qc_out, max_tiles=qc_max_tiles)

    return {
        "tile_count": len(records),
        "output_path": output_path,
        "qc_path": None if qc_out is None else Path(qc_out),
        **_compression_stats(wsi_path, output_path),
    }
