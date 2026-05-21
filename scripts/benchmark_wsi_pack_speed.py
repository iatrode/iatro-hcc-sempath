from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import threading
import time
from pathlib import Path

import imagecodecs
import numpy as np
from PIL import Image
from tqdm import tqdm

from hcc_sempath.tiling import tissue_fraction


_thread_state = threading.local()


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024.0
    return f"{size}B"


def _grid(width: int, height: int, stride_x: int, stride_y: int) -> list[tuple[int, int]]:
    return [
        (x, y)
        for y in range(0, height - stride_y + 1, stride_y)
        for x in range(0, width - stride_x + 1, stride_x)
    ]


def _choose_mask_level(slide, max_pixels: int) -> int:
    candidates = []
    for idx, (width, height) in enumerate(slide.level_dimensions):
        pixels = int(width) * int(height)
        if pixels <= max_pixels:
            candidates.append((idx, pixels))
    if candidates:
        return min(candidates, key=lambda item: item[0])[0]
    return len(slide.level_dimensions) - 1


def _read_tile(slide, x: int, y: int, level: int, level_read_w: int, level_read_h: int, tile_size: int) -> np.ndarray:
    tile = slide.read_region((x, y), level, (level_read_w, level_read_h)).convert("RGB")
    if tile.size != (tile_size, tile_size):
        tile = tile.resize((tile_size, tile_size), Image.Resampling.BICUBIC)
    return np.asarray(tile).copy()


def _build_mask(slide, mask_level: int, white_threshold: int) -> tuple[np.ndarray, float]:
    width, height = slide.level_dimensions[mask_level]
    started = time.perf_counter()
    image = slide.read_region((0, 0), mask_level, (width, height)).convert("RGB")
    arr = np.asarray(image)
    mask = arr.mean(axis=2) < white_threshold
    return mask, time.perf_counter() - started


def _mask_fraction(mask: np.ndarray, x: int, y: int, tile_w: int, tile_h: int, downsample: float) -> float:
    x0 = max(0, int(x / downsample))
    y0 = max(0, int(y / downsample))
    x1 = min(mask.shape[1], max(x0 + 1, int((x + tile_w) / downsample)))
    y1 = min(mask.shape[0], max(y0 + 1, int((y + tile_h) / downsample)))
    if x0 >= x1 or y0 >= y1:
        return 0.0
    return float(mask[y0:y1, x0:x1].mean())


def _prefilter_candidates_integral(
    mask: np.ndarray,
    candidates: list[tuple[int, int]],
    tile_w: int,
    tile_h: int,
    downsample: float,
    threshold: float,
) -> list[tuple[int, int]]:
    coords = np.asarray(candidates, dtype=np.int64)
    xs = coords[:, 0]
    ys = coords[:, 1]
    integral = np.pad(mask.astype(np.uint32), ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    x0 = np.maximum(0, (xs / downsample).astype(np.int64))
    y0 = np.maximum(0, (ys / downsample).astype(np.int64))
    x1 = np.minimum(mask.shape[1], ((xs + tile_w) / downsample).astype(np.int64))
    y1 = np.minimum(mask.shape[0], ((ys + tile_h) / downsample).astype(np.int64))
    x1 = np.minimum(np.maximum(x1, x0 + 1), mask.shape[1])
    y1 = np.minimum(np.maximum(y1, y0 + 1), mask.shape[0])
    area = np.maximum(1, (x1 - x0) * (y1 - y0))
    tissue = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    keep = tissue / area >= threshold
    return [(int(x), int(y)) for x, y in coords[keep]]


def _row(method: str, elapsed: float, total_candidates: int, selected: int, highres_reads: int, retained: int) -> dict:
    return {
        "method": method,
        "elapsed_sec": round(elapsed, 6),
        "total_candidates": total_candidates,
        "selected_candidates": selected,
        "highres_reads": highres_reads,
        "retained_tiles": retained,
        "candidates_per_sec": round(total_candidates / elapsed, 3) if elapsed > 0 else 0.0,
        "highres_reads_per_sec": round(highres_reads / elapsed, 3) if elapsed > 0 else 0.0,
        "retained_tiles_per_sec": round(retained / elapsed, 3) if elapsed > 0 else 0.0,
    }


def _get_thread_slide(wsi_path: Path):
    slide = getattr(_thread_state, "slide", None)
    if slide is None:
        import openslide

        slide = openslide.OpenSlide(str(wsi_path))
        _thread_state.slide = slide
    return slide


def _confirm_tile(
    item: tuple[int, int],
    *,
    wsi_path: Path,
    level: int,
    level_read_w: int,
    level_read_h: int,
    tile_size: int,
    white_threshold: int,
    min_tissue_fraction: float,
    encode: bool,
    distance: float,
    effort: int,
) -> tuple[bool, int]:
    slide = _get_thread_slide(wsi_path)
    x, y = item
    arr = _read_tile(slide, x, y, level, level_read_w, level_read_h, tile_size)
    if tissue_fraction(arr, white_threshold=white_threshold) < min_tissue_fraction:
        return False, 0
    if not encode:
        return True, 0
    encoded = imagecodecs.jpegxl_encode(arr, lossless=False, distance=distance, effort=effort)
    return True, len(encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark WSI tile selection and encode strategies on one SVS/MRXS.")
    parser.add_argument("--wsi", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-mpp", type=float, default=0.5)
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.3)
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--effort", type=int, default=7)
    parser.add_argument("--white-threshold", type=int, default=220)
    parser.add_argument("--mask-max-pixels", type=int, default=12_000_000)
    parser.add_argument("--max-candidates", type=int, default=0, help="Debug cap; 0 means full slide.")
    parser.add_argument("--workers", type=int, default=1, help="Threaded benchmark workers for prefiltered high-res reads.")
    parser.add_argument("--skip-full-scan", action="store_true", help="Skip the slow current full-scan baseline.")
    parser.add_argument("--encode", action="store_true", help="Also benchmark JXL encoding for prefiltered retained tiles.")
    args = parser.parse_args()

    try:
        import openslide
    except ImportError as exc:
        raise RuntimeError("openslide-python is required") from exc

    wsi_path = Path(args.wsi)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slide = openslide.OpenSlide(str(wsi_path))
    try:
        native_mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
        native_mpp_y = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_Y, native_mpp))
        downsample_needed = args.target_mpp / native_mpp
        level = slide.get_best_level_for_downsample(downsample_needed)
        level_downsample = float(slide.level_downsamples[level])
        scale_x = native_mpp / args.target_mpp
        scale_y = native_mpp_y / args.target_mpp
        stride_x = max(1, round(args.tile_size / scale_x))
        stride_y = max(1, round(args.tile_size / scale_y))
        level_read_w = max(1, round(stride_x / level_downsample))
        level_read_h = max(1, round(stride_y / level_downsample))
        width, height = slide.dimensions
        candidates = _grid(width, height, stride_x, stride_y)
        if args.max_candidates > 0:
            candidates = candidates[: args.max_candidates]
        total_candidates = len(candidates)

        metadata = {
            "wsi_path": str(wsi_path),
            "wsi_bytes": wsi_path.stat().st_size,
            "wsi_size": _format_bytes(wsi_path.stat().st_size),
            "width": width,
            "height": height,
            "native_mpp_x": native_mpp,
            "native_mpp_y": native_mpp_y,
            "target_mpp": args.target_mpp,
            "tile_size": args.tile_size,
            "min_tissue_fraction": args.min_tissue_fraction,
            "openslide_level": level,
            "level_downsample": level_downsample,
            "stride_x": stride_x,
            "stride_y": stride_y,
            "level_read_width": level_read_w,
            "level_read_height": level_read_h,
            "total_candidates": total_candidates,
        }
        print("benchmark_start " + " ".join(f"{k}={v}" for k, v in metadata.items() if k != "wsi_path"))

        rows = []

        if not args.skip_full_scan:
            started = time.perf_counter()
            full_retained = 0
            for x, y in tqdm(candidates, desc="current_full_scan", unit="tile"):
                arr = _read_tile(slide, x, y, level, level_read_w, level_read_h, args.tile_size)
                if tissue_fraction(arr, white_threshold=args.white_threshold) >= args.min_tissue_fraction:
                    full_retained += 1
            rows.append(_row("current_full_scan_read_every_candidate", time.perf_counter() - started, total_candidates, total_candidates, total_candidates, full_retained))

        mask_level = _choose_mask_level(slide, args.mask_max_pixels)
        mask_downsample = float(slide.level_downsamples[mask_level])
        mask, mask_build_sec = _build_mask(slide, mask_level, args.white_threshold)
        started = time.perf_counter()
        selected = [
            (x, y)
            for x, y in tqdm(candidates, desc="mask_prefilter", unit="tile")
            if _mask_fraction(mask, x, y, stride_x, stride_y, mask_downsample) >= args.min_tissue_fraction
        ]
        prefilter_sec = time.perf_counter() - started
        rows.append(_row("mask_build", mask_build_sec, total_candidates, 0, 0, 0))
        rows[-1].update({"mask_level": mask_level, "mask_downsample": mask_downsample, "mask_shape": list(mask.shape)})
        rows.append(_row("mask_prefilter_all_candidates", prefilter_sec, total_candidates, len(selected), 0, len(selected)))
        rows[-1].update({"mask_level": mask_level, "mask_downsample": mask_downsample, "mask_shape": list(mask.shape)})

        started = time.perf_counter()
        selected_integral = _prefilter_candidates_integral(
            mask,
            candidates,
            stride_x,
            stride_y,
            mask_downsample,
            args.min_tissue_fraction,
        )
        integral_prefilter_sec = time.perf_counter() - started
        rows.append(_row("mask_prefilter_integral_all_candidates", integral_prefilter_sec, total_candidates, len(selected_integral), 0, len(selected_integral)))
        rows[-1].update({"mask_level": mask_level, "mask_downsample": mask_downsample, "mask_shape": list(mask.shape)})

        started = time.perf_counter()
        confirmed_retained = 0
        encoded_bytes = 0
        for x, y in tqdm(selected, desc="prefiltered_highres_confirm", unit="tile"):
            arr = _read_tile(slide, x, y, level, level_read_w, level_read_h, args.tile_size)
            if tissue_fraction(arr, white_threshold=args.white_threshold) < args.min_tissue_fraction:
                continue
            confirmed_retained += 1
            if args.encode:
                encoded_bytes += len(
                    imagecodecs.jpegxl_encode(arr, lossless=False, distance=args.distance, effort=args.effort)
                )
        rows.append(_row("mask_prefilter_then_highres_confirm" + ("_and_jxl_encode" if args.encode else ""), time.perf_counter() - started, total_candidates, len(selected), len(selected), confirmed_retained))
        rows[-1].update({"encoded_bytes": encoded_bytes, "encoded_size": _format_bytes(encoded_bytes)})

        if args.workers > 1:
            started = time.perf_counter()
            parallel_retained = 0
            parallel_encoded_bytes = 0
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                results = executor.map(
                    lambda item: _confirm_tile(
                        item,
                        wsi_path=wsi_path,
                        level=level,
                        level_read_w=level_read_w,
                        level_read_h=level_read_h,
                        tile_size=args.tile_size,
                        white_threshold=args.white_threshold,
                        min_tissue_fraction=args.min_tissue_fraction,
                        encode=args.encode,
                        distance=args.distance,
                        effort=args.effort,
                    ),
                    selected,
                )
                for retained, encoded_size in tqdm(results, total=len(selected), desc="prefiltered_highres_parallel", unit="tile"):
                    if retained:
                        parallel_retained += 1
                        parallel_encoded_bytes += encoded_size
            rows.append(_row(f"mask_prefilter_then_highres_confirm_{args.workers}workers" + ("_and_jxl_encode" if args.encode else ""), time.perf_counter() - started, total_candidates, len(selected), len(selected), parallel_retained))
            rows[-1].update({"encoded_bytes": parallel_encoded_bytes, "encoded_size": _format_bytes(parallel_encoded_bytes)})

        output_stem = wsi_path.stem.replace("/", "_")
        json_path = output_dir / f"{output_stem}.wsi_speed_benchmark.json"
        csv_path = output_dir / f"{output_stem}.wsi_speed_benchmark.csv"
        payload = {"metadata": metadata, "rows": rows}
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = sorted({key for row in rows for key in row.keys()})
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"benchmark_done json={json_path} csv={csv_path}")
        for row in rows:
            print(
                f"{row['method']} elapsed={row['elapsed_sec']:.3f}s "
                f"candidates_per_sec={row['candidates_per_sec']:.1f} "
                f"highres_reads_per_sec={row['highres_reads_per_sec']:.1f} "
                f"selected={row['selected_candidates']} retained={row['retained_tiles']}"
            )
    finally:
        slide.close()


if __name__ == "__main__":
    main()
