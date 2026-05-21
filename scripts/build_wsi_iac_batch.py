from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from tqdm import tqdm

from hcc_sempath.cli.wsi_to_iac import build_wsi_iac
from hcc_sempath.tile_package import read_package_metadata


WSI_SUFFIXES = {".svs", ".mrxs"}


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
        f"wsi_package_{status} slide={slide_id} "
        f"input_bytes={stats['input_bytes']} package_bytes={stats['package_bytes']} "
        f"compression_ratio={stats['compression_ratio']:.4f}x "
        f"space_saving_pct={stats['space_saving_pct']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch package WSIs into per-slide IatroCache tile packages.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--target-mpp", type=float, default=0.5)
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.3)
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--effort", type=int, default=7)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--qc-max-tiles", type=int, default=36)
    parser.add_argument("--limit", type=int, default=0, help="Debug limit; 0 means all slides.")
    parser.add_argument("--tcga-patient-id", action="store_true", help="Parse patient_id as TCGA-XX-YYYY from file names.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-qc", action="store_true")
    parser.add_argument("--no-inner-progress", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    slides = _discover_wsi(input_root)
    if args.limit > 0:
        slides = slides[: args.limit]
    if not slides:
        raise FileNotFoundError(f"no WSI files found under {input_root}")
    print(
        "batch_start "
        f"slides={len(slides)} input_root={input_root} output_root={output_root} "
        f"target_mpp={args.target_mpp} tile_size={args.tile_size} "
        f"min_tissue_fraction={args.min_tissue_fraction} distance={args.distance} workers={args.workers}",
        flush=True,
    )

    manifest_rows = []
    failures = []
    started = time.time()
    for wsi_path in tqdm(slides, desc="WSI", unit="slide"):
        slide_id = _safe_id(wsi_path.stem)
        patient_id = _tcga_patient_id_from_name(wsi_path) if args.tcga_patient_id else slide_id
        slide_dir = output_root / slide_id
        package_path = slide_dir / "tiles.iac"
        qc_path = None if args.no_qc else slide_dir / "tile_package_qc.png"
        done_path = slide_dir / "done.json"
        fail_path = slide_dir / "failed.json"

        if package_path.exists() and done_path.exists() and not args.overwrite:
            try:
                metadata = read_package_metadata(package_path)
                tile_count = int(metadata["num_records"])
            except Exception:
                tile_count = -1
            stats = _compression_stats(wsi_path, package_path)
            _print_package_stats("skipped", slide_id, stats)
            manifest_rows.append(
                {
                    "slide_id": slide_id,
                    "patient_id": patient_id,
                    "wsi_path": str(wsi_path),
                    "package_path": str(package_path),
                    "qc_path": "" if qc_path is None else str(qc_path),
                    "tile_count": tile_count,
                    "status": "skipped",
                    **stats,
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
                native_mpp=None,
                native_mpp_y=None,
                tile_size=args.tile_size,
                min_tissue_fraction=args.min_tissue_fraction,
                max_tiles=args.max_tiles,
                lossless=False,
                distance=args.distance,
                effort=args.effort,
                workers=args.workers,
                qc_out=qc_path,
                qc_max_tiles=args.qc_max_tiles,
                overwrite=args.overwrite,
                show_progress=not args.no_inner_progress,
            )
            metadata = read_package_metadata(package_path)
            payload = {
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
                "distance": args.distance,
                "input_bytes": result["input_bytes"],
                "package_bytes": result["package_bytes"],
                "compression_ratio": result["compression_ratio"],
                "space_saving_pct": result["space_saving_pct"],
                "header": metadata,
            }
            _write_json(done_path, payload)
            if fail_path.exists():
                fail_path.unlink()
            _print_package_stats("ok", slide_id, result)
            manifest_rows.append(
                {
                    k: payload[k]
                    for k in (
                        "slide_id",
                        "patient_id",
                        "wsi_path",
                        "package_path",
                        "qc_path",
                        "tile_count",
                        "status",
                        "input_bytes",
                        "package_bytes",
                        "compression_ratio",
                        "space_saving_pct",
                    )
                }
            )
        except Exception as exc:
            failure = {
                "status": "failed",
                "slide_id": slide_id,
                "patient_id": patient_id,
                "wsi_path": str(wsi_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            _write_json(fail_path, failure)
            manifest_rows.append(
                {
                    "slide_id": slide_id,
                    "patient_id": patient_id,
                    "wsi_path": str(wsi_path),
                    "package_path": str(package_path),
                    "qc_path": "" if qc_path is None else str(qc_path),
                    "tile_count": 0,
                    "status": "failed",
                    "input_bytes": wsi_path.stat().st_size if wsi_path.exists() else 0,
                    "package_bytes": 0,
                    "compression_ratio": 0.0,
                    "space_saving_pct": 0.0,
                }
            )

        _write_json(
            output_root / "batch_progress.json",
            {
                "input_root": str(input_root),
                "output_root": str(output_root),
                "processed": len(manifest_rows),
                "total": len(slides),
                "failures": len(failures),
                "elapsed_sec": round(time.time() - started, 3),
            },
        )

    import csv

    manifest_path = output_root / "packages.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "slide_id",
            "patient_id",
            "wsi_path",
            "package_path",
            "qc_path",
            "tile_count",
            "status",
            "input_bytes",
            "package_bytes",
            "compression_ratio",
            "space_saving_pct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    completed_rows = [row for row in manifest_rows if row["status"] in {"ok", "skipped"}]
    total_input_bytes = sum(int(row["input_bytes"]) for row in completed_rows)
    total_package_bytes = sum(int(row["package_bytes"]) for row in completed_rows)
    overall_compression_ratio = round(total_input_bytes / total_package_bytes, 4) if total_package_bytes > 0 else 0.0
    overall_space_saving_pct = (
        round((1.0 - total_package_bytes / total_input_bytes) * 100.0, 3) if total_input_bytes > 0 else 0.0
    )
    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "total": len(slides),
        "ok": sum(1 for row in manifest_rows if row["status"] in {"ok", "skipped"}),
        "failed": sum(1 for row in manifest_rows if row["status"] == "failed"),
        "elapsed_sec": round(time.time() - started, 3),
        "input_bytes": total_input_bytes,
        "package_bytes": total_package_bytes,
        "compression_ratio": overall_compression_ratio,
        "space_saving_pct": overall_space_saving_pct,
        "manifest": str(manifest_path),
    }
    _write_json(output_root / "batch_summary.json", summary)
    print(
        "wsi_batch_done "
        f"total={summary['total']} ok={summary['ok']} failed={summary['failed']} "
        f"input_bytes={summary['input_bytes']} package_bytes={summary['package_bytes']} "
        f"compression_ratio={summary['compression_ratio']:.4f}x "
        f"space_saving_pct={summary['space_saving_pct']:.3f} "
        f"manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
