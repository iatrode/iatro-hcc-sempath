from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch

from iatro.iac import read_header, read_tables
from iatro.iac.adapters.tiles import TilePackageReader

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


def _input_packages(inputs: list[str]) -> list[Path]:
    packages: list[Path] = []
    for value in inputs:
        path = Path(value).resolve()
        if path.is_file():
            packages.append(path)
        elif path.is_dir():
            packages.extend(sorted(path.rglob("*.tiles.iac")))
        else:
            raise FileNotFoundError(path)
    packages = list(dict.fromkeys(packages))
    if not packages:
        raise ValueError("input did not resolve to any .tiles.iac packages")
    invalid = [
        path
        for path in packages
        if read_header(path).get("payload_type") != "image_tiles"
    ]
    if invalid:
        raise ValueError(f"input contains non-image tile IAC packages: {invalid[:3]}")
    return packages


def _output_paths(packages: list[Path], output_dir: Path) -> dict[Path, Path]:
    result = {
        package: output_dir / f"{package.name.removesuffix('.tiles.iac')}.predictions.iac"
        for package in packages
    }
    by_output: dict[Path, list[Path]] = {}
    for source, output in result.items():
        by_output.setdefault(output, []).append(source)
    collisions = {path: sources for path, sources in by_output.items() if len(sources) > 1}
    if collisions:
        sample = next(iter(collisions.values()))
        raise ValueError(f"duplicate tile package names would overwrite predictions: {sample}")
    return result


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a released SemPath model on image-tile IAC packages and write "
            "coordinate-reconstructable prediction IAC packages."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Release directory containing config.json and hcc_sempath_release.pt.",
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Image-tile IAC file or directory; repeatable.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
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
    args = parser.parse_args()
    if args.batch_size <= 0 or args.workers <= 0:
        raise ValueError("--batch-size and --workers must be positive")

    device = torch.device(args.device)
    release = load_release_model(args.model, device=device)
    packages = _input_packages(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = _output_paths(packages, args.output)
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"prediction output already exists; pass --overwrite to replace it: {existing[0]}"
        )

    checkpoint_digest = file_sha256(release.weights_path)
    records: list[dict] = []
    started = time.monotonic()
    for package_path in packages:
        source_header, slide_table, source_index = read_tables(package_path)
        tile_size = (int(source_header["tile_height"]), int(source_header["tile_width"]))
        if tile_size != (STUDENT_IMAGE_SIZE, STUDENT_IMAGE_SIZE):
            raise ValueError(
                f"SemPath release requires {STUDENT_IMAGE_SIZE}px tiles, "
                f"got {tile_size} from {package_path}"
            )
        rows = list(range(len(source_index)))
        source_digest = source_index_sha256(source_header, slide_table, source_index)
        output_path = outputs[package_path]
        reader = TilePackageReader(package_path)
        observed_grid: tuple[int, int] | None = None

        def payloads():
            nonlocal observed_grid
            encode = partial(
                encode_prediction_payload,
                spatial_dtype=args.spatial_dtype,
            )
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
                        raise ValueError("spatial grid shape changed within one source package")
                    yield from executor.map(encode, classification, instance, abundance)

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
                output_path,
                header=header,
                slide_table=slide_table,
                index_table=prediction_index_table(source_index, rows),
                payloads=payloads(),
            )
        finally:
            reader.close()
        if observed_grid != expected_grid:
            raise ValueError(
                f"model grid shape mismatch: expected={expected_grid} got={observed_grid}"
            )
        records.append(
            {
                "source_package": str(package_path),
                "prediction_package": str(output_path),
                "source_iac_index_sha256": source_digest,
                "records": len(rows),
                "bytes": output_path.stat().st_size,
            }
        )

    manifest = {
        "schema_version": 1,
        "model": str(args.model.resolve()),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_model_sha256": release.model_digest,
        "spatial_dtype": args.spatial_dtype,
        "records": sum(item["records"] for item in records),
        "bytes": sum(item["bytes"] for item in records),
        "elapsed_seconds": time.monotonic() - started,
        "packages": records,
    }
    _write_json_atomic(args.output / "manifest.json", manifest)
    print(
        "inference_ok "
        f"packages={len(records)} records={manifest['records']} "
        f"bytes={manifest['bytes']} elapsed_seconds={manifest['elapsed_seconds']:.3f}"
    )


if __name__ == "__main__":
    main()
