from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from iatro.iac.adapters.tiles import (
    TilePackageReader,
    read_package_manifest,
    read_package_metadata,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hcc_sempath.modeling.models import (  # noqa: E402
    HCCSemPathModel,
    SPATIAL_PATCH_PADDING,
    STUDENT_BACKBONE_NAME,
    STUDENT_PATCH_SIZE,
    canonical_payload_sha256,
    model_state_sha256,
    validate_spatial_decoder_calibration,
)
from hcc_sempath.training.config import (  # noqa: E402
    _package_record_count,
    _select_package_fraction,
    embedding_dim,
    image_tile_package_paths,
    teacher_dims,
    teacher_names,
)
from hcc_sempath.training.engine import _prepare_images  # noqa: E402
from hcc_sempath.training.manifest import (  # noqa: E402
    load_training_manifest,
    manifest_tile_packages,
)
from hcc_sempath.training.roi import (  # noqa: E402
    build_spatial_roi_targets,
    load_spatial_tile_locations,
    load_spatial_validation_metadata,
    spatial_component_names,
)
from hcc_sempath.training.spatial_validation import (  # noqa: E402
    calibrate_spatial_decoder,
    evaluate_weak_spatial_supervision,
)
from hcc_sempath.training.train import (  # noqa: E402
    _assert_disjoint_package_cohorts,
    _package_cohort_ids,
    _paths_from_data,
    _target_rows_by_package,
    _verify_optimizer_visible_packages,
)


def _finalized_checkpoint(payload: dict) -> dict:
    cfg = payload.get("config")
    if not isinstance(cfg, dict):
        raise ValueError("checkpoint has no resolved training config")
    expected = int(cfg.get("train", {}).get("epochs", -1))
    checkpoint_epoch = int(payload.get("epoch", -1))
    expected_matches = (
        int(payload.get("expected_epochs", -1)) == expected
    )
    legacy_terminal = bool(
        payload.get("training_complete", False)
        and checkpoint_epoch == expected
        and expected_matches
    )
    finalized_selection = bool(
        payload.get("run_complete", False)
        and payload.get("selection_finalized", False)
        and checkpoint_epoch
        == int(payload.get("best_selection_epoch", -1))
        and expected_matches
    )
    joint_selection_run = bool(
        cfg.get("train", {}).get("selection_early_stop", False)
        or cfg.get("data", {}).get(
            "require_complete_expert_validation",
            False,
        )
    )
    accepted = (
        finalized_selection
        if joint_selection_run
        else (legacy_terminal or finalized_selection)
    )
    if not accepted:
        raise ValueError(
            "spatial calibration requires a finalized joint-selection "
            "checkpoint from a completed run"
        )
    return cfg


def _split_tile_packages(
    cfg: dict,
    split: str,
    *,
    apply_train_fraction: bool = True,
    manifest_override: Path | None = None,
) -> list[str]:
    manifest_path = manifest_override or cfg["data"].get("train_manifest_path")
    if manifest_path:
        manifest = load_training_manifest(manifest_path)
        paths = manifest_tile_packages(manifest, split)
        if split == "train" and apply_train_fraction:
            paths = _select_package_fraction(
                paths,
                {
                    path: _package_record_count(path)
                    for path in paths
                },
                fraction=float(
                    cfg["data"].get("train_tile_fraction", 1.0)
                ),
                seed=int(cfg.get("runtime", {}).get("seed", 13)),
            )
        return [str(path) for path in paths]
    key = f"{split}_image_tile_package_paths"
    if key in cfg["data"]:
        return list(dict.fromkeys(_paths_from_data(cfg, key)))
    if split in {"train", "val"}:
        return list(dict.fromkeys(image_tile_package_paths(cfg)))
    return []


def _load_validation_images(
    packages: list[str],
    target_locations: dict[str, tuple[str, int]],
) -> tuple[list[str], torch.Tensor, list[str], list[str]]:
    rows = _target_rows_by_package(packages, target_locations)
    tile_ids: list[str] = []
    images: list[torch.Tensor] = []
    slide_ids: list[str] = []
    selected_packages: list[str] = []
    for package_idx in sorted(rows):
        package_path = packages[package_idx]
        records = read_package_manifest(package_path)
        reader = TilePackageReader(package_path)
        try:
            for row_idx in rows[package_idx].tolist():
                tile_ids.append(str(records[row_idx].tile_id))
                slide_ids.append(str(records[row_idx].slide_id))
                image = np.asarray(
                    reader.read_image_at(row_idx).convert("RGB"),
                    dtype=np.uint8,
                ).copy()
                images.append(torch.from_numpy(image).permute(2, 0, 1))
        finally:
            reader.close()
        selected_packages.append(package_path)
    return tile_ids, torch.stack(images), selected_packages, slide_ids


def _optimizer_visible_packages(
    cfg: dict,
) -> list[str]:
    packages = cfg["data"].get(
        "optimizer_visible_tile_packages"
    )
    if not isinstance(packages, list) or not packages:
        raise ValueError(
            "checkpoint has no frozen optimizer-visible cohort contract"
        )
    resolved = [str(path) for path in packages]
    _verify_optimizer_visible_packages(cfg)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate the eleven-component decoder on an independent ROI split."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help=(
            "Relocated manifest used only to resolve immutable source images. "
            "Required when archived checkpoints contain obsolete host paths."
        ),
    )
    parser.add_argument(
        "--validation-split",
        action="append",
        choices=("val", "exval"),
        default=[],
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--image-cache",
        type=Path,
        help=(
            "Optional temporary decoded-image cache for repeated checkpoint "
            "evaluation. The annotation and package-pool digests must match."
        ),
    )
    parser.add_argument(
        "--supervisory-validation",
        action="store_true",
        help=(
            "Evaluate the checkpoint-selection supervision split even when "
            "its source cohorts overlap the optimizer-visible unlabeled "
            "population. The report is explicitly marked non-independent."
        ),
    )
    parser.add_argument("--output-calibration", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = _finalized_checkpoint(payload)
    device = torch.device(cfg["runtime"]["device"])
    names = spatial_component_names(args.annotation)
    frozen_names = [
        str(name)
        for name in cfg["data"].get("spatial_component_names", names)
    ]
    if frozen_names != names:
        raise ValueError(
            "validation annotation component order differs from the checkpoint"
        )
    validation_splits = {
        str(value)
        for value in (args.validation_split or ["val"])
    }
    package_splits = set(validation_splits)
    if args.supervisory_validation:
        # The checkpoint-selection bank may intentionally contain labels for
        # tiles sampled from the optimizer's train population. The annotation
        # split still controls which labels are evaluated; this wider package
        # pool only resolves the corresponding immutable source images.
        package_splits.add("train")
    packages = list(dict.fromkeys(
        path
        for split in sorted(package_splits)
        for path in _split_tile_packages(
            cfg,
            split,
            apply_train_fraction=not args.supervisory_validation,
            manifest_override=args.source_manifest,
        )
    ))
    if not packages:
        raise ValueError("checkpoint has no requested validation packages")
    metadata = read_package_metadata(packages[0])
    image_size = (
        int(metadata["tile_height"]),
        int(metadata["tile_width"]),
    )
    stride = int(cfg["model"].get("spatial_output_stride", 7))
    grid_size = tuple(
        (
            size + 2 * SPATIAL_PATCH_PADDING - STUDENT_PATCH_SIZE
        )
        // stride
        + 1
        for size in image_size
    )
    validation_targets = build_spatial_roi_targets(
        args.annotation,
        component_names=names,
        image_size=image_size,
        grid_size=grid_size,
        allowed_splits=validation_splits,
        point_tolerance_cells=int(
            cfg["loss"].get("spatial_point_tolerance_cells", 1)
        ),
    )
    if not validation_targets:
        raise ValueError("independent spatial validation split is empty")
    annotation_sha256 = hashlib.sha256(
        Path(args.annotation).read_bytes()
    ).hexdigest()
    package_pool_sha256 = hashlib.sha256(
        "\n".join(str(Path(path).resolve()) for path in packages).encode()
    ).hexdigest()
    cache_contract = {
        "version": 1,
        "annotation_sha256": annotation_sha256,
        "package_pool_sha256": package_pool_sha256,
        "validation_splits": sorted(validation_splits),
        "source_population_splits": sorted(package_splits),
    }
    if args.image_cache is not None and args.image_cache.exists():
        cached = torch.load(
            args.image_cache,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(cached, dict) or cached.get("contract") != cache_contract:
            raise ValueError(
                "decoded spatial image cache contract mismatch: "
                f"{args.image_cache}"
            )
        tile_ids = [str(value) for value in cached["tile_ids"]]
        images = cached["images"]
        validation_packages = [
            str(value) for value in cached["validation_packages"]
        ]
        slide_ids = [str(value) for value in cached["slide_ids"]]
        if not isinstance(images, torch.Tensor) or images.dtype != torch.uint8:
            raise ValueError("decoded spatial image cache must contain uint8 images")
    else:
        tile_ids, images, validation_packages, slide_ids = _load_validation_images(
            packages,
            load_spatial_tile_locations(
                args.annotation,
                allowed_splits=validation_splits,
            ),
        )
        if args.image_cache is not None:
            args.image_cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.image_cache.with_suffix(args.image_cache.suffix + ".tmp")
            torch.save(
                {
                    "contract": cache_contract,
                    "tile_ids": tile_ids,
                    "images": images,
                    "validation_packages": validation_packages,
                    "slide_ids": slide_ids,
                },
                temporary,
            )
            temporary.replace(args.image_cache)
    validation_metadata = load_spatial_validation_metadata(
        args.annotation,
        component_names=names,
        allowed_splits=validation_splits,
    )
    missing_metadata = sorted(
        set(tile_ids).difference(validation_metadata)
    )
    if missing_metadata:
        raise ValueError(
            "validation targets missing completeness metadata: "
            f"count={len(missing_metadata)}"
        )
    optimizer_packages = (
        [
            str(value)
            for value in cfg["data"].get(
                "optimizer_visible_tile_packages",
                [],
            )
        ]
        if args.supervisory_validation
        else _optimizer_visible_packages(cfg)
    )
    if not optimizer_packages:
        raise ValueError(
            "checkpoint has no frozen optimizer-visible cohort contract"
        )
    patient_slide_disjoint = not args.supervisory_validation
    if patient_slide_disjoint:
        _assert_disjoint_package_cohorts(
            optimizer_packages,
            list(dict.fromkeys(validation_packages)),
        )
    teacher_names_value = teacher_names(cfg)
    model = HCCSemPathModel(
        backbone_name=STUDENT_BACKBONE_NAME,
        embedding_dim=embedding_dim(cfg),
        teacher_dims=teacher_dims(cfg, teacher_names_value),
        pretrained=False,
        projector_type=cfg["model"].get("projector_type", "linear"),
        projector_hidden_dim=int(
            cfg["model"].get("projector_hidden_dim", 2048)
        ),
        teacher_head_type=cfg["model"].get("teacher_head_type", "linear"),
        classification_num_classes=len(cfg["model"]["classification_class_names"]),
        spatial_num_components=len(names),
        spatial_dim=int(cfg["model"].get("spatial_dim", 256)),
        spatial_output_stride=stride,
    ).to(device)
    if model.spatial_head is not None:
        model.spatial_head.use_local_branch = bool(
            cfg["model"].get("spatial_use_local_branch", True)
        )
        model.spatial_head.use_semantic_branch = bool(
            cfg["model"].get("spatial_use_semantic_branch", True)
        )
        model.spatial_head.use_context = bool(
            cfg["model"].get("spatial_use_context", True)
        )
    model.load_state_dict(
        {
            key.removeprefix("_orig_mod."): value
            for key, value in payload["model"].items()
        }
    )
    model.eval()

    instance_batches = []
    abundance_batches = []
    with torch.no_grad():
        for start in range(0, len(images), args.batch_size):
            batch = _prepare_images(
                {
                    "images": images[start : start + args.batch_size],
                    "images_uint8": True,
                },
                cfg,
                device,
            )
            outputs = model(batch)
            instance_batches.append(
                outputs["spatial_instance_probabilities"].cpu()
            )
            abundance_batches.append(
                outputs["spatial_abundance_probabilities"].cpu()
            )
    ordered_targets = [validation_targets[tile_id] for tile_id in tile_ids]
    ordered_metadata = [
        validation_metadata[tile_id]
        for tile_id in tile_ids
    ]
    research_contract = cfg.get("research_contract")
    if not isinstance(research_contract, dict):
        raise ValueError("checkpoint has no frozen research contract")
    validation_patients, validation_slides = _package_cohort_ids(
        list(dict.fromkeys(validation_packages))
    )
    protocol_contract = {
        "version": 1,
        "validation_role": (
            "independent_spatial_validation"
            if patient_slide_disjoint
            else "checkpoint_selection_supervision"
        ),
        "patient_slide_disjoint_from_training": patient_slide_disjoint,
        "component_names": names,
        "validation_splits": sorted(validation_splits),
        "spatial_output_stride": stride,
        "point_tolerance_cells": int(
            cfg["loss"].get("spatial_point_tolerance_cells", 1)
        ),
        "implicit_negative_weight": float(
            cfg["loss"].get(
                "spatial_implicit_negative_weight",
                0.05,
            )
        ),
        "brush_top_fraction": float(
            cfg["loss"].get("spatial_brush_top_fraction", 1.0)
        ),
        "completeness_contract": (
            "explicit_count_and_measurement_complete_v1"
        ),
    }
    calibration_provenance = {
        "checkpoint_model_sha256": model_state_sha256(
            payload["model"]
        ),
        "research_contract_sha256": canonical_payload_sha256(
            research_contract
        ),
        "validation_annotation_sha256": annotation_sha256,
        "validation_protocol_sha256": canonical_payload_sha256(
            protocol_contract
        ),
        "validation_cohort_sha256": canonical_payload_sha256(
            {
                "patient_ids": sorted(validation_patients),
                "slide_ids": sorted(validation_slides),
            }
        ),
        "optimizer_visible_contract_sha256": canonical_payload_sha256(
            {
                "packages": cfg["data"][
                    "optimizer_visible_tile_packages"
                ],
                "sizes": cfg["data"][
                    "optimizer_visible_tile_package_sizes"
                ],
                "sha256": cfg["data"][
                    "optimizer_visible_tile_package_sha256"
                ],
                "expert_split_exclusion_sha256": cfg["data"].get(
                    "expert_split_exclusion_sha256"
                ),
            }
        ),
        "supervision_assets_sha256": canonical_payload_sha256(
            cfg["data"].get("supervision_asset_sha256", {})
        ),
        "formal_asset_contract_sha256": str(
            cfg["data"].get("formal_asset_contract_sha256", "")
        ),
        "source_tree_sha256": str(
            cfg["data"].get("formal_source", {}).get(
                "source_tree_sha256",
                "",
            )
        ),
        "study_contract_sha256": str(
            cfg["data"].get(
                "formal_study_contract_sha256",
                "",
            )
        ),
        "selected_epoch": int(payload["epoch"]),
        "terminal_epoch": int(
            payload.get("run_terminal_epoch", payload["epoch"])
        ),
        "expected_epochs": int(payload["expected_epochs"]),
        "selection_finalized": bool(
            payload.get(
                "selection_finalized",
                int(payload["epoch"])
                == int(payload["expected_epochs"]),
            )
        ),
    }
    instance_probability = torch.cat(instance_batches)
    abundance_probability = torch.cat(abundance_batches)
    point_centers = torch.stack(
        [target.point_centers for target in ordered_targets]
    )
    brush_bag_ids = torch.stack(
        [target.brush_bag_ids for target in ordered_targets]
    )
    area_positive = torch.stack(
        [target.area_positive for target in ordered_targets]
    )
    explicit_negative = torch.stack(
        [target.explicit_negative for target in ordered_targets]
    )
    point_tolerance = int(
        cfg["loss"].get("spatial_point_tolerance_cells", 1)
    )
    brush_top_fraction = float(
        cfg["loss"].get("spatial_brush_top_fraction", 1.0)
    )
    if args.supervisory_validation:
        calibration, report = evaluate_weak_spatial_supervision(
            instance_probability=instance_probability,
            abundance_probability=abundance_probability,
            point_centers=point_centers,
            brush_bag_ids=brush_bag_ids,
            area_positive=area_positive,
            explicit_negative=explicit_negative,
            component_names=names,
            threshold=0.5,
            point_tolerance_cells=point_tolerance,
            nms_kernel=3,
            brush_top_fraction=brush_top_fraction,
        )
        calibration["provenance"] = calibration_provenance
    else:
        calibration, report = calibrate_spatial_decoder(
            instance_probability=instance_probability,
            abundance_probability=abundance_probability,
            point_centers=point_centers,
            brush_bag_ids=brush_bag_ids,
            area_positive=area_positive,
            explicit_negative=explicit_negative,
            implicit_negative=torch.stack(
                [target.implicit_negative for target in ordered_targets]
            ),
            count_complete=torch.stack(
                [item.count_complete for item in ordered_metadata]
            ),
            measurement_complete=torch.stack(
                [item.measurement_complete for item in ordered_metadata]
            ),
            geometry_modes=[
                item.geometry_modes for item in ordered_metadata
            ],
            slide_ids=slide_ids,
            calibration_provenance=calibration_provenance,
            component_names=names,
            output_stride=stride,
            point_tolerance_cells=point_tolerance,
            implicit_negative_weight=float(
                cfg["loss"].get("spatial_implicit_negative_weight", 0.05)
            ),
            brush_top_fraction=brush_top_fraction,
        )
        calibration = validate_spatial_decoder_calibration(
            calibration,
            names,
            expected_output_stride=stride,
            expected_model_state_sha256=calibration_provenance[
                "checkpoint_model_sha256"
            ],
            expected_research_contract_sha256=calibration_provenance[
                "research_contract_sha256"
            ],
            expected_optimizer_visible_contract_sha256=(
                calibration_provenance[
                    "optimizer_visible_contract_sha256"
                ]
            ),
            expected_supervision_assets_sha256=calibration_provenance[
                "supervision_assets_sha256"
            ],
            expected_formal_asset_contract_sha256=(
                calibration_provenance[
                    "formal_asset_contract_sha256"
                ]
            ),
            expected_source_tree_sha256=calibration_provenance[
                "source_tree_sha256"
            ],
            expected_study_contract_sha256=calibration_provenance[
                "study_contract_sha256"
            ],
        )
    report["protocol"].update(
        {
            "split": protocol_contract["validation_role"],
            "checkpoint_epoch": int(payload["epoch"]),
            "validation_splits": [
                str(value) for value in sorted(validation_splits)
            ],
            "source_population_splits": [
                str(value) for value in sorted(package_splits)
            ],
            "patient_slide_disjoint_from_training": patient_slide_disjoint,
            "validation_patient_count": len(validation_patients),
            "validation_slide_count": len(validation_slides),
            "optimizer_visible_package_count": len(
                optimizer_packages
            ),
            "validation_protocol_sha256": (
                calibration_provenance[
                    "validation_protocol_sha256"
                ]
            ),
        }
    )
    for output_path, value in (
        (Path(args.output_calibration), calibration),
        (Path(args.output_report), report),
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(value, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        (
            "spatial_supervision_metrics_ok "
            if args.supervisory_validation
            else "spatial_calibration_ok "
        )
        + f"tiles={len(tile_ids)} output={args.output_calibration}"
    )


if __name__ == "__main__":
    main()
