from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

from tqdm import tqdm

from hcc_sempath.cli.wsi_to_iac import _default_workers, _format_bytes, build_wsi_iac
from hcc_sempath.io.tile_package import read_package_metadata


WSI_SUFFIXES = {".svs", ".mrxs", ".ndpi", ".scn", ".tif", ".tiff"}


def _discover_wsi(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in WSI_SUFFIXES)


def _safe_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._") or "slide"


def _tcga_patient_id_from_name(path: Path) -> str:
    parts = path.stem.split(".")[0].split("-")
    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])
    return _safe_id(path.stem)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _compression_stats(input_path: Path, package_path: Path) -> dict:
    input_bytes = input_path.stat().st_size
    package_bytes = package_path.stat().st_size
    compression_ratio = round(input_bytes / package_bytes, 4) if package_bytes > 0 else 0.0
    space_saving_pct = round((1.0 - package_bytes / input_bytes) * 100.0, 3) if input_bytes > 0 else 0.0
    return {
        "input_bytes": input_bytes,
        "package_bytes": package_bytes,
        "compression_ratio": compression_ratio,
        "space_saving_pct": space_saving_pct,
    }


def _print_package_stats(status: str, slide_id: str, stats: dict) -> None:
    tqdm.write(
        f"tile_cache_{status} slide={slide_id} "
        f"input={_format_bytes(stats['input_bytes'])} package={_format_bytes(stats['package_bytes'])} "
        f"input_bytes={stats['input_bytes']} package_bytes={stats['package_bytes']} "
        f"compression_ratio={stats['compression_ratio']:.4f}x "
        f"space_saving_pct={stats['space_saving_pct']:.3f}"
    )


def _package_path_for_slide(output: Path, slide_id: str, total_slides: int) -> Path:
    if total_slides == 1 and output.suffix == ".iac":
        return output
    return output / f"{slide_id}.tiles.iac"


def _qc_path_for_slide(output: Path, slide_id: str, package_path: Path, enabled: bool, total_slides: int) -> Path | None:
    if not enabled:
        return None
    if total_slides == 1 and package_path.suffix == ".iac":
        return package_path.with_suffix(".qc.png")
    return output / f"{slide_id}.tiles.qc.png"


def _resolve_slides(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in WSI_SUFFIXES:
            supported = ", ".join(sorted(WSI_SUFFIXES))
            raise ValueError(f"input file is not a supported WSI extension: {input_path} ({supported})")
        return [input_path]
    if input_path.is_dir():
        slides = _discover_wsi(input_path)
        if slides:
            return slides
        raise FileNotFoundError(f"no WSI files found under {input_path}")
    raise FileNotFoundError(f"input path does not exist: {input_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build image-tile IatroCache packages directly from a WSI file or WSI directory."
    )
    parser.add_argument("--input", required=True, help="Input WSI file or directory containing WSI files.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output .tiles.iac path for one WSI, or output directory for a WSI directory.",
    )
    parser.add_argument("--patient-id", default=None, help="Patient id for a single WSI; defaults to slide id.")
    parser.add_argument("--slide-id", default=None, help="Slide id for a single WSI; defaults to file stem.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--target-mpp", type=float, default=0.5)
    parser.add_argument("--native-mpp", type=float, default=None, help="Override/fallback native MPP X.")
    parser.add_argument("--native-mpp-y", type=float, default=None, help="Override/fallback native MPP Y.")
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.1)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--lossless", action="store_true")
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--effort", type=int, default=7)
    parser.add_argument("--workers", type=int, default=_default_workers())
    parser.add_argument("--white-threshold", type=int, default=220)
    parser.add_argument("--prefilter-tissue-fraction", type=float, default=0.05)
    parser.add_argument("--mask-max-pixels", type=int, default=12_000_000)
    parser.add_argument("--qc", action="store_true", help="Write per-slide QC contact sheets beside IAC packages.")
    parser.add_argument("--qc-max-tiles", type=int, default=36)
    parser.add_argument("--limit", type=int, default=0, help="Debug limit for directory input; 0 means all slides.")
    parser.add_argument("--tcga-patient-id", action="store_true", help="Parse patient_id as TCGA-XX-YYYY from file names.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output = Path(args.output)
    slides = _resolve_slides(input_path)
    if args.limit > 0:
        slides = slides[: args.limit]
    if len(slides) > 1 and output.suffix == ".iac":
        raise ValueError("--output must be a directory when --input resolves to multiple WSI files")
    if len(slides) == 1 and output.suffix == ".iac":
        output.parent.mkdir(parents=True, exist_ok=True)
    else:
        output.mkdir(parents=True, exist_ok=True)

    print(
        "tile_cache_start "
        f"slides={len(slides)} input={input_path} output={output} "
        f"target_mpp={args.target_mpp} tile_size={args.tile_size} "
        f"min_tissue_fraction={args.min_tissue_fraction} distance={args.distance} workers={args.workers}",
        flush=True,
    )

    rows = []
    failures = []
    started = time.time()
    for wsi_path in tqdm(slides, desc="WSI", unit="slide", disable=args.no_progress):
        single = len(slides) == 1
        slide_id = _safe_id(args.slide_id) if single and args.slide_id else _safe_id(wsi_path.stem)
        patient_id = (
            args.patient_id
            if single and args.patient_id
            else _tcga_patient_id_from_name(wsi_path)
            if args.tcga_patient_id
            else slide_id
        )
        package_path = _package_path_for_slide(output, slide_id, len(slides))
        qc_path = _qc_path_for_slide(output, slide_id, package_path, args.qc, len(slides))

        if package_path.exists() and not args.overwrite:
            try:
                metadata = read_package_metadata(package_path)
                tile_count = int(metadata["num_records"])
            except Exception:
                tile_count = -1
            stats = _compression_stats(wsi_path, package_path)
            _print_package_stats("skipped", slide_id, stats)
            rows.append(
                {
                    "slide_id": slide_id,
                    "patient_id": patient_id,
                    "wsi_path": str(wsi_path),
                    "package_path": str(package_path),
                    "qc_path": "" if qc_path is None else str(qc_path),
                    "tile_count": tile_count,
                    "status": "skipped",
                    "target_mpp": args.target_mpp,
                    "tile_size": args.tile_size,
                    "min_tissue_fraction": args.min_tissue_fraction,
                    "prefilter_tissue_fraction": args.prefilter_tissue_fraction,
                    "distance": args.distance,
                    "input_bytes": stats["input_bytes"],
                    "package_bytes": stats["package_bytes"],
                    "compression_ratio": stats["compression_ratio"],
                    "space_saving_pct": stats["space_saving_pct"],
                    "error_type": "",
                    "error": "",
                }
            )
            continue

        try:
            result = build_wsi_iac(
                wsi_path=wsi_path,
                output_path=package_path,
                patient_id=patient_id,
                slide_id=slide_id,
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
                white_threshold=args.white_threshold,
                prefilter_tissue_fraction=args.prefilter_tissue_fraction,
                mask_max_pixels=args.mask_max_pixels,
                qc_out=qc_path,
                qc_max_tiles=args.qc_max_tiles,
                overwrite=args.overwrite,
                show_progress=not args.no_progress,
            )
            _print_package_stats("ok", slide_id, result)
            rows.append(
                {
                    "status": "ok",
                    "slide_id": slide_id,
                    "patient_id": patient_id,
                    "wsi_path": str(wsi_path),
                    "package_path": str(package_path),
                    "qc_path": "" if qc_path is None else str(qc_path),
                    "tile_count": result["tile_count"],
                    "target_mpp": args.target_mpp,
                    "tile_size": args.tile_size,
                    "min_tissue_fraction": args.min_tissue_fraction,
                    "prefilter_tissue_fraction": args.prefilter_tissue_fraction,
                    "distance": args.distance,
                    "input_bytes": result["input_bytes"],
                    "package_bytes": result["package_bytes"],
                    "compression_ratio": result["compression_ratio"],
                    "space_saving_pct": result["space_saving_pct"],
                    "error_type": "",
                    "error": "",
                }
            )
        except Exception as exc:
            failures.append({"slide_id": slide_id, "wsi_path": str(wsi_path), "error": str(exc)})
            rows.append(
                {
                    "slide_id": slide_id,
                    "patient_id": patient_id,
                    "wsi_path": str(wsi_path),
                    "package_path": str(package_path),
                    "qc_path": "" if qc_path is None else str(qc_path),
                    "tile_count": 0,
                    "status": "failed",
                    "target_mpp": args.target_mpp,
                    "tile_size": args.tile_size,
                    "min_tissue_fraction": args.min_tissue_fraction,
                    "prefilter_tissue_fraction": args.prefilter_tissue_fraction,
                    "distance": args.distance,
                    "input_bytes": wsi_path.stat().st_size if wsi_path.exists() else 0,
                    "package_bytes": 0,
                    "compression_ratio": 0.0,
                    "space_saving_pct": 0.0,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

        if len(slides) > 1:
            _write_json(
                output / "batch_progress.json",
                {
                    "input": str(input_path),
                    "output": str(output),
                    "processed": len(rows),
                    "total": len(slides),
                    "failures": len(failures),
                    "elapsed_sec": round(time.time() - started, 3),
                },
            )

    if len(slides) > 1:
        manifest_path = output / "packages.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "slide_id",
                "patient_id",
                "wsi_path",
                "package_path",
                "qc_path",
                "tile_count",
                "status",
                "target_mpp",
                "tile_size",
                "min_tissue_fraction",
                "prefilter_tissue_fraction",
                "distance",
                "input_bytes",
                "package_bytes",
                "compression_ratio",
                "space_saving_pct",
                "error_type",
                "error",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        completed = [row for row in rows if row["status"] in {"ok", "skipped"}]
        total_input_bytes = sum(int(row["input_bytes"]) for row in completed)
        total_package_bytes = sum(int(row["package_bytes"]) for row in completed)
        summary = {
            "input": str(input_path),
            "output": str(output),
            "total": len(slides),
            "ok": sum(1 for row in rows if row["status"] in {"ok", "skipped"}),
            "failed": sum(1 for row in rows if row["status"] == "failed"),
            "elapsed_sec": round(time.time() - started, 3),
            "input_bytes": total_input_bytes,
            "package_bytes": total_package_bytes,
            "compression_ratio": round(total_input_bytes / total_package_bytes, 4) if total_package_bytes > 0 else 0.0,
            "space_saving_pct": (
                round((1.0 - total_package_bytes / total_input_bytes) * 100.0, 3)
                if total_input_bytes > 0
                else 0.0
            ),
            "manifest": str(manifest_path),
        }
        _write_json(output / "batch_summary.json", summary)
        print(
            "tile_cache_done "
            f"total={summary['total']} ok={summary['ok']} failed={summary['failed']} "
            f"input={_format_bytes(summary['input_bytes'])} package={_format_bytes(summary['package_bytes'])} "
            f"compression_ratio={summary['compression_ratio']:.4f}x manifest={manifest_path}"
        )
    else:
        row = rows[0]
        if row["status"] == "failed":
            raise SystemExit(f"tile_cache_failed slide={row['slide_id']} error={row['error']}")
        print(
            "tile_cache_done "
            f"slide={row['slide_id']} tiles={row['tile_count']} output={row['package_path']} "
            f"compression_ratio={float(row['compression_ratio']):.4f}x"
        )


if __name__ == "__main__":
    main()
