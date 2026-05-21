from __future__ import annotations

import argparse
import os
from pathlib import Path

from hcc_sempath.manifests import write_tile_manifest
from hcc_sempath.qc import render_tile_package_qc
from hcc_sempath.tile_package import build_tile_package
from hcc_sempath.tiling import tile_wsi


def _default_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def build_wsi_iac(
    *,
    wsi_path: str | Path,
    output_path: str | Path,
    work_dir: str | Path | None = None,
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
    work_dir = Path(work_dir) if work_dir else output_path.parent / f"{output_path.stem}_work"
    tiles_dir = work_dir / "tiles"
    manifest_path = work_dir / "tile_manifest.csv"

    rows = tile_wsi(
        wsi_path=wsi_path,
        output_dir=tiles_dir,
        patient_id=patient_id,
        slide_id=slide_id,
        split=split,
        tile_size=tile_size,
        min_tissue_fraction=min_tissue_fraction,
        target_mpp=target_mpp,
        native_mpp=native_mpp,
        native_mpp_y=native_mpp_y,
        max_tiles=max_tiles,
        overwrite_slide_dir=overwrite,
        show_progress=show_progress,
    )
    if not rows:
        raise ValueError(f"no tiles retained from {wsi_path}; lower --min-tissue-fraction or check the slide")
    write_tile_manifest(manifest_path, rows)

    build_tile_package(
        manifest_path=manifest_path,
        output_path=output_path,
        lossless=lossless,
        distance=None if lossless else distance,
        effort=effort,
        workers=workers,
        show_progress=show_progress,
        overwrite=overwrite,
    )

    if qc_out:
        render_tile_package_qc(output_path, qc_out, max_tiles=qc_max_tiles)

    return {
        "tile_count": len(rows),
        "manifest_path": manifest_path,
        "output_path": output_path,
        "work_dir": work_dir,
        "qc_path": None if qc_out is None else Path(qc_out),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an image-tile IatroCache package directly from an OpenSlide-readable WSI."
    )
    parser.add_argument("--wsi", required=True, help="Input WSI path, such as .svs or .mrxs.")
    parser.add_argument("--output", required=True, help="Output image-tile .iac path.")
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Intermediate directory for PNG tiles and tile_manifest.csv. Defaults beside the output package.",
    )
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
        work_dir=args.work_dir,
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
        f"tiles={result['tile_count']} manifest={result['manifest_path']} output={result['output_path']} "
        f"target_mpp={args.target_mpp} tile_size={args.tile_size}"
    )


if __name__ == "__main__":
    main()
