from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from iatro.iac import read_header, read_tables
from iatro.iac.adapters.tiles import TilePackageReader

from hcc_sempath.inference.inputs import (
    MaterializationOptions,
    materialize_input,
    plan_inputs,
)
from hcc_sempath.inference.model import load_release_model, prepare_images
from hcc_sempath.inference.predictions import (
    encode_prediction_payload,
    file_sha256,
    prediction_header,
    prediction_index_table,
    source_index_sha256,
    write_prediction_package,
)
from hcc_sempath.modeling.models import (
    SPATIAL_PATCH_PADDING,
    STUDENT_IMAGE_SIZE,
    STUDENT_PATCH_SIZE,
)
from hcc_sempath.release_hub import resolve_cached_release


def _source_split(index_table) -> str:
    if "split" not in index_table.column_names:
        raise ValueError("source tile index has no split column")
    values = {
        str(value)
        for value in index_table.column("split").to_pylist()
        if value not in (None, "")
    }
    if not values:
        return "unspecified"
    return next(iter(values)) if len(values) == 1 else "mixed"


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run SemPath on a pathology tile IAC, one 224px raster image, "
            "or a WSI. Raster and WSI inputs are first materialized as canonical "
            ".tile.path.iac packages; predictions are written as .pred.path.iac."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        help=(
            "Local release directory containing config.json and model.safetensors. "
            "Omit to use the release installed by `hcc-sempath download`."
        ),
    )
    parser.add_argument(
        "--hub",
        choices=("auto", "hf", "modelscope"),
        default="auto",
        help="Select a local downloaded release cache when --model is omitted.",
    )
    parser.add_argument("--cache-dir", type=Path, help="Optional release-cache root.")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help=(
            "Pathology .tile.path.iac, legacy .tiles.iac, 224px PNG/JPEG/WebP/BMP, "
            "WSI (.svs/.mrxs/.ndpi/.scn/.tif/.tiff), or directory; repeatable."
        ),
    )
    parser.add_argument("--output", required=True, type=Path, help="Output directory.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8, help="Inference tile decode/encode workers.")
    parser.add_argument("--tile-workers", type=int, default=8, help="WSI tile read/encode workers.")
    parser.add_argument("--target-mpp", type=float, default=0.5)
    parser.add_argument("--native-mpp", type=float, default=None, help="WSI/raster native MPP X override.")
    parser.add_argument("--native-mpp-y", type=float, default=None, help="WSI/raster native MPP Y override.")
    parser.add_argument("--split", default="inference", help="Split label recorded for generated tile packages.")
    parser.add_argument("--min-tissue-fraction", type=float, default=0.10)
    parser.add_argument("--prefilter-tissue-fraction", type=float, default=0.05)
    parser.add_argument("--white-threshold", type=int, default=220)
    parser.add_argument("--black-threshold", type=int, default=8)
    parser.add_argument("--mask-max-pixels", type=int, default=12_000_000)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--tile-lossless", action="store_true")
    parser.add_argument("--tile-distance", type=float, default=1.0)
    parser.add_argument("--tile-effort", type=int, default=7)
    parser.add_argument(
        "--spatial-dtype",
        choices=("uint8", "uint16", "float16"),
        default="float16",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.workers <= 0 or args.tile_workers <= 0:
        raise ValueError("batch size and worker counts must be positive")
    if args.target_mpp <= 0:
        raise ValueError("--target-mpp must be positive")
    if args.native_mpp is not None and args.native_mpp <= 0:
        raise ValueError("--native-mpp must be positive")
    if args.native_mpp_y is not None and args.native_mpp_y <= 0:
        raise ValueError("--native-mpp-y must be positive")
    for name in ("min_tissue_fraction", "prefilter_tissue_fraction"):
        value = float(getattr(args, name))
        if not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if not 0 <= args.black_threshold < args.white_threshold <= 255:
        raise ValueError("tissue thresholds must satisfy 0 <= black < white <= 255")
    if args.mask_max_pixels <= 0 or args.tile_effort <= 0:
        raise ValueError("mask pixel limit and tile effort must be positive")
    if args.max_tiles is not None and args.max_tiles <= 0:
        raise ValueError("--max-tiles must be positive")


def run(args: argparse.Namespace) -> dict:
    _validate_args(args)
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plans = plan_inputs(args.input, output_dir)
    existing = [plan.prediction_package for plan in plans if plan.prediction_package.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"prediction output already exists; pass --overwrite to replace it: {existing[0]}"
        )

    model_dir = resolve_cached_release(args.model, hub=args.hub, cache_dir=args.cache_dir)
    materialization = MaterializationOptions(
        split=args.split,
        target_mpp=args.target_mpp,
        native_mpp=args.native_mpp,
        native_mpp_y=args.native_mpp_y,
        min_tissue_fraction=args.min_tissue_fraction,
        prefilter_tissue_fraction=args.prefilter_tissue_fraction,
        white_threshold=args.white_threshold,
        black_threshold=args.black_threshold,
        mask_max_pixels=args.mask_max_pixels,
        max_tiles=args.max_tiles,
        workers=args.tile_workers,
        lossless=args.tile_lossless,
        distance=args.tile_distance,
        effort=args.tile_effort,
        overwrite=args.overwrite,
        show_progress=not args.no_progress,
    )
    packages = []
    for index, plan in enumerate(plans, start=1):
        print(
            f"input_prepare index={index}/{len(plans)} kind={plan.kind} source={plan.source_path}",
            flush=True,
        )
        packages.append(materialize_input(plan, materialization))

    device = torch.device(args.device)
    release = load_release_model(model_dir, device=device)
    checkpoint_digest = file_sha256(release.weights_path)
    total_tiles = sum(int(read_header(package)["num_records"]) for package in packages)
    records: list[dict] = []
    started = time.monotonic()

    with tqdm(
        total=total_tiles,
        desc="SemPath inference",
        unit="tile",
        disable=args.no_progress,
        dynamic_ncols=True,
    ) as progress:
        for plan, package_path in zip(plans, packages, strict=True):
            source_header, slide_table, source_index = read_tables(package_path)
            tile_size = (
                int(source_header["tile_height"]),
                int(source_header["tile_width"]),
            )
            if tile_size != (STUDENT_IMAGE_SIZE, STUDENT_IMAGE_SIZE):
                raise ValueError(
                    f"SemPath release requires {STUDENT_IMAGE_SIZE}px tiles, "
                    f"got {tile_size} from {package_path}"
                )
            rows = list(range(len(source_index)))
            source_digest = source_index_sha256(source_header, slide_table, source_index)
            reader = TilePackageReader(package_path)
            observed_grid: tuple[int, int] | None = None

            def payloads():
                nonlocal observed_grid
                encode = partial(encode_prediction_payload, spatial_dtype=args.spatial_dtype)
                with torch.inference_mode(), ThreadPoolExecutor(max_workers=args.workers) as executor:
                    for start in range(0, len(rows), args.batch_size):
                        batch_rows = rows[start : start + args.batch_size]
                        arrays = reader.read_arrays_at(batch_rows, workers=args.workers)
                        images = torch.from_numpy(np.stack(arrays, axis=0))
                        prepared = prepare_images(images, release, device)
                        with torch.autocast(
                            device_type=device.type,
                            dtype=torch.float16,
                            enabled=device.type == "cuda",
                        ):
                            result = release.model(prepared)
                        classification = result["classification_probabilities"].float().cpu().numpy()
                        instance = result["spatial_instance_probabilities"].float().cpu().numpy()
                        abundance = result["spatial_abundance_probabilities"].float().cpu().numpy()
                        grid = (int(instance.shape[-2]), int(instance.shape[-1]))
                        if observed_grid is None:
                            observed_grid = grid
                        elif observed_grid != grid:
                            raise ValueError("spatial grid dimensions changed within one source package")
                        for payload in executor.map(encode, classification, instance, abundance):
                            yield payload
                        progress.update(len(batch_rows))

            spatial_stride = int(release.config["model"]["spatial_output_stride"])
            expected_grid = tuple(
                (size + 2 * SPATIAL_PATCH_PADDING - STUDENT_PATCH_SIZE) // spatial_stride + 1
                for size in tile_size
            )
            header = prediction_header(
                source_path=package_path,
                source_header=source_header,
                source_index_digest=source_digest,
                checkpoint_path=release.weights_path,
                checkpoint_file_digest=checkpoint_digest,
                checkpoint_model_digest=release.model_digest,
                classification_names=list(release.classification_names),
                component_names=list(release.spatial_component_names),
                grid_shape=expected_grid,
                spatial_stride=spatial_stride,
                patch_size=STUDENT_PATCH_SIZE,
                patch_padding=SPATIAL_PATCH_PADDING,
                spatial_dtype=args.spatial_dtype,
                dataset_split=_source_split(source_index),
            )
            try:
                write_prediction_package(
                    plan.prediction_package,
                    header=header,
                    slide_table=slide_table,
                    index_table=prediction_index_table(source_index, rows),
                    payloads=payloads(),
                )
            finally:
                reader.close()
            if observed_grid != expected_grid:
                raise ValueError(
                    f"model grid dimensions mismatch: expected={expected_grid} got={observed_grid}"
                )
            records.append(
                {
                    "input_kind": plan.kind,
                    "input": str(plan.source_path),
                    "tile_package": str(package_path),
                    "prediction_package": str(plan.prediction_package),
                    "source_iac_index_sha256": source_digest,
                    "records": len(rows),
                    "bytes": plan.prediction_package.stat().st_size,
                }
            )

    manifest = {
        "schema_version": 2,
        "model": str(model_dir),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_model_sha256": release.model_digest,
        "spatial_dtype": args.spatial_dtype,
        "records": sum(item["records"] for item in records),
        "bytes": sum(item["bytes"] for item in records),
        "elapsed_seconds": time.monotonic() - started,
        "packages": records,
    }
    _write_json_atomic(output_dir / "inference_manifest.json", manifest)
    print(
        "inference_ok "
        f"packages={len(records)} records={manifest['records']} "
        f"bytes={manifest['bytes']} elapsed_seconds={manifest['elapsed_seconds']:.3f}",
        flush=True,
    )
    return manifest


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
