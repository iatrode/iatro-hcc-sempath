from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from iatro.iac import read_tables
from iatro.iac.adapters.tiles import TilePackageReader

from hcc_sempath.inference.predictions import (
    encode_prediction_payload,
    file_sha256,
    prediction_header,
    prediction_index_table,
    source_index_sha256,
    write_prediction_package,
)
from hcc_sempath.modeling.models import (
    HCCSemPathModel,
    SPATIAL_PATCH_PADDING,
    STUDENT_BACKBONE_NAME,
    STUDENT_PATCH_SIZE,
    model_state_sha256,
)
from hcc_sempath.training.config import embedding_dim, teacher_dims, teacher_names
from hcc_sempath.training.engine import _prepare_images
from hcc_sempath.training.manifest import load_training_manifest, manifest_tile_packages


def _model_from_checkpoint(payload: dict, device: torch.device) -> tuple[HCCSemPathModel, dict, list[str], list[str]]:
    cfg = payload.get("config")
    if not isinstance(cfg, dict):
        raise ValueError("checkpoint has no resolved training config")
    classification_names = [str(value) for value in cfg["model"]["classification_class_names"]]
    component_names = [str(value) for value in cfg["data"]["spatial_component_names"]]
    names = teacher_names(cfg)
    model = HCCSemPathModel(
        backbone_name=STUDENT_BACKBONE_NAME,
        embedding_dim=embedding_dim(cfg),
        teacher_dims=teacher_dims(cfg, names),
        pretrained=False,
        projector_type=cfg["model"].get("projector_type", "linear"),
        projector_hidden_dim=int(cfg["model"].get("projector_hidden_dim", 2048)),
        teacher_head_type=cfg["model"].get("teacher_head_type", "linear"),
        classification_num_classes=len(classification_names),
        spatial_num_components=len(component_names),
        spatial_dim=int(cfg["model"].get("spatial_dim", 256)),
        spatial_output_stride=int(cfg["model"].get("spatial_output_stride", 7)),
    ).to(device)
    if model.spatial_head is not None:
        model.spatial_head.use_local_branch = bool(cfg["model"].get("spatial_use_local_branch", True))
        model.spatial_head.use_semantic_branch = bool(cfg["model"].get("spatial_use_semantic_branch", True))
        model.spatial_head.use_context = bool(cfg["model"].get("spatial_use_context", True))
    model.load_state_dict({key.removeprefix("_orig_mod."): value for key, value in payload["model"].items()})
    model.eval()
    return model, cfg, classification_names, component_names


def _packages(args: argparse.Namespace) -> list[Path]:
    direct = [Path(value) for value in args.package]
    if args.manifest:
        manifest = load_training_manifest(args.manifest)
        direct.extend(manifest_tile_packages(manifest, args.split))
    packages = list(dict.fromkeys(path.resolve() for path in direct))
    if not packages:
        raise ValueError("pass at least one --package or --manifest with --split")
    return packages


def main() -> None:
    parser = argparse.ArgumentParser(description="Export reconstructable FULL-checkpoint tile predictions.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--package", action="append", default=[])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", choices=("train", "val", "exval"), default="train")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--max-tiles", type=int)
    parser.add_argument("--spatial-dtype", choices=("uint8", "uint16", "float16"), default="uint8")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.decode_workers <= 0:
        raise ValueError("batch size and decode workers must be positive")
    if args.max_tiles is not None and args.max_tiles <= 0:
        raise ValueError("--max-tiles must be positive")

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not bool(payload.get("run_complete")) or not bool(payload.get("selection_finalized")):
        raise ValueError("prediction export requires a finalized selected checkpoint")
    model, cfg, classification_names, component_names = _model_from_checkpoint(payload, device)
    checkpoint_digest = file_sha256(args.checkpoint)
    model_digest = model_state_sha256(payload["model"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict] = []
    remaining = args.max_tiles
    started = time.monotonic()

    for package_path in _packages(args):
        source_header, slide_table, source_index = read_tables(package_path)
        count = len(source_index) if remaining is None else min(len(source_index), remaining)
        if count <= 0:
            break
        rows = list(range(count))
        output_path = args.output_dir / (
            f"{package_path.parent.name}__"
            f"{package_path.name.removesuffix('.tiles.iac')}.predictions.iac"
        )
        reader = TilePackageReader(package_path)
        first_shape: tuple[int, int] | None = None

        def predicted_payloads():
            nonlocal first_shape
            with torch.inference_mode():
                for start in range(0, count, args.batch_size):
                    batch_rows = rows[start : start + args.batch_size]
                    arrays = reader.read_arrays_at(batch_rows, workers=args.decode_workers)
                    images = torch.from_numpy(np.stack(arrays, axis=0))
                    prepared = _prepare_images(
                        {"images": images, "images_uint8": True, "images_hwc": True},
                        cfg,
                        device,
                    )
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        result = model(prepared)
                    classification = result["classification_probabilities"].float().cpu().numpy()
                    instance = result["spatial_instance_probabilities"].float().cpu().numpy()
                    abundance = result["spatial_abundance_probabilities"].float().cpu().numpy()
                    shape = (int(instance.shape[-2]), int(instance.shape[-1]))
                    if first_shape is None:
                        first_shape = shape
                    elif first_shape != shape:
                        raise ValueError("spatial grid shape changed within one source package")
                    for offset in range(len(batch_rows)):
                        yield encode_prediction_payload(
                            classification[offset], instance[offset], abundance[offset],
                            spatial_dtype=args.spatial_dtype,
                        )

        expected_grid = tuple(
            (int(source_header[key]) + 2 * SPATIAL_PATCH_PADDING - STUDENT_PATCH_SIZE)
            // int(cfg["model"].get("spatial_output_stride", 7)) + 1
            for key in ("tile_height", "tile_width")
        )
        header = prediction_header(
            source_path=package_path,
            source_header=source_header,
            source_index_digest=source_index_sha256(source_header, slide_table, source_index),
            checkpoint_path=args.checkpoint,
            checkpoint_file_digest=checkpoint_digest,
            checkpoint_model_digest=model_digest,
            classification_names=classification_names,
            component_names=component_names,
            grid_shape=expected_grid,
            spatial_stride=int(cfg["model"].get("spatial_output_stride", 7)),
            patch_size=STUDENT_PATCH_SIZE,
            patch_padding=SPATIAL_PATCH_PADDING,
            spatial_dtype=args.spatial_dtype,
        )
        try:
            write_prediction_package(
                output_path,
                header=header,
                slide_table=slide_table,
                index_table=prediction_index_table(source_index, rows, split=args.split),
                payloads=predicted_payloads(),
            )
        finally:
            reader.close()
        if first_shape != expected_grid:
            raise ValueError(f"model grid shape mismatch: expected={expected_grid} got={first_shape}")
        outputs.append({
            "source_package": str(package_path),
            "prediction_package": str(output_path),
            "records": count,
            "bytes": output_path.stat().st_size,
            "sha256": file_sha256(output_path),
        })
        if remaining is not None:
            remaining -= count

    manifest = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_model_sha256": model_digest,
        "split": args.split,
        "spatial_dtype": args.spatial_dtype,
        "records": sum(int(item["records"]) for item in outputs),
        "bytes": sum(int(item["bytes"]) for item in outputs),
        "elapsed_seconds": time.monotonic() - started,
        "packages": outputs,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("prediction_export_ok " + " ".join(f"{key}={manifest[key]}" for key in ("records", "bytes", "elapsed_seconds")))


if __name__ == "__main__":
    main()
