from __future__ import annotations

import argparse
import os
from pathlib import Path
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np
from PIL import Image
from tqdm import tqdm

from hcc_sempath.manifests import TileRecord
from hcc_sempath.qc import render_tile_package_qc
from hcc_sempath.tile_package import build_tile_package_from_records, encode_jxl_array
from hcc_sempath.tiling import tissue_fraction


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


def _flush_done(done, records: list[TileRecord], payloads: list[bytes]) -> None:
    for future in done:
        idx, record, payload = future.result()
        records.append(record)
        payloads.append(payload)


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

    if show_progress:
        print(
            "wsi_direct_pack_start "
            f"slide={slide_id} size={width}x{height} "
            f"native_mpp=({native_mpp:.4f},{native_mpp_y:.4f}) target_mpp={target_mpp:.4f} "
            f"level={level} level_downsample={level_downsample:.4f} "
            f"stride0=({level0_stride_x},{level0_stride_y}) candidates={total_candidates}",
            flush=True,
        )
    progress = tqdm(total=total_candidates, desc=f"Packing {slide_id}", unit="tile") if show_progress else None
    records: list[TileRecord] = []
    payloads: list[bytes] = []
    pending = set()
    idx = 0
    workers = max(1, int(workers))

    def encode_record(tile_idx: int, x: int, y: int, arr: np.ndarray):
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

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for y in range(0, height - level0_stride_y + 1, level0_stride_y):
                for x in range(0, width - level0_stride_x + 1, level0_stride_x):
                    tile = slide.read_region((x, y), level, (level_read_w, level_read_h)).convert("RGB")
                    if tile.size != (tile_size, tile_size):
                        tile = tile.resize((tile_size, tile_size), Image.Resampling.BICUBIC)
                    arr = np.asarray(tile).copy()
                    if progress is not None:
                        progress.update(1)
                        progress.set_postfix(retained=idx, refresh=False)
                    if tissue_fraction(arr) < min_tissue_fraction:
                        continue
                    pending.add(executor.submit(encode_record, idx, x, y, arr))
                    idx += 1
                    if len(pending) >= workers * 4:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        _flush_done(done, records, payloads)
                    if max_tiles is not None and idx >= max_tiles:
                        break
                if max_tiles is not None and idx >= max_tiles:
                    break
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                _flush_done(done, records, payloads)
    finally:
        if progress is not None:
            progress.close()
            tqdm.write(f"wsi_direct_pack_tiles_done slide={slide_id} retained={idx} candidates_seen={progress.n}")
        slide.close()

    ordered = sorted(zip(records, payloads), key=lambda item: item[0].tile_id)
    records = [record for record, _ in ordered]
    payloads = [payload for _, payload in ordered]
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
                "candidate_tiles": total_candidates,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an image-tile IatroCache package directly from an OpenSlide-readable WSI."
    )
    parser.add_argument("--wsi", required=True, help="Input WSI path, such as .svs or .mrxs.")
    parser.add_argument("--output", required=True, help="Output image-tile .iac path.")
    parser.add_argument("--patient-id", default=None)
    parser.add_argument("--slide-id", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--target-mpp", type=float, default=0.5)
    parser.add_argument("--native-mpp", type=float, default=None)
    parser.add_argument("--native-mpp-y", type=float, default=None)
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.1)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--lossless", action="store_true")
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--effort", type=int, default=7)
    parser.add_argument("--workers", type=int, default=_default_workers())
    parser.add_argument("--qc-out", default=None, help="Optional QC contact sheet path.")
    parser.add_argument("--qc-max-tiles", type=int, default=36)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    result = build_wsi_iac(
        wsi_path=args.wsi,
        output_path=args.output,
        patient_id=args.patient_id,
        slide_id=args.slide_id,
        split=args.split,
        target_mpp=args.target_mpp,
        native_mpp=args.native_mpp,
        native_mpp_y=args.native_mpp_y,
        tile_size=args.tile_size,
        min_tissue_fraction=args.min_tissue_fraction,
        max_tiles=args.max_tiles,
        lossless=args.lossless,
        distance=args.distance,
        effort=args.effort,
        workers=args.workers,
        qc_out=args.qc_out,
        qc_max_tiles=args.qc_max_tiles,
        overwrite=args.overwrite,
        show_progress=not args.no_progress,
    )
    print(
        "wsi_package_ok "
        f"tiles={result['tile_count']} output={result['output_path']} "
        f"target_mpp={args.target_mpp} tile_size={args.tile_size} "
        f"input={_format_bytes(result['input_bytes'])} package={_format_bytes(result['package_bytes'])} "
        f"input_bytes={result['input_bytes']} package_bytes={result['package_bytes']} "
        f"compression_ratio={result['compression_ratio']:.4f}x "
        f"space_saving_pct={result['space_saving_pct']:.3f}"
    )


if __name__ == "__main__":
    main()
