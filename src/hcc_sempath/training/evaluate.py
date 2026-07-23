from __future__ import annotations

import argparse
import copy

import torch
from torch.utils.data import DataLoader

from iatro.iac.adapters.tiles import read_package_metadata
from ..modeling.prototypes import PrototypeRegistry, load_prototype_registry
from ..modeling.models import (
    HCCSemPathModel,
    STUDENT_BACKBONE_NAME,
    STUDENT_IMAGE_SIZE,
)
from .config import (
    embedding_dim,
    image_tile_package_paths,
    load_config,
    manifest_data_paths,
    teacher_dims,
    teacher_feature_package_paths,
    teacher_names,
    validate_training_config,
)
from .datasets import (
    DistillationTileDataset,
    PackageSampledDistillationDataset,
    apply_split_overrides,
    collate_distillation,
    read_packaged_tile_records,
    validate_teacher_feature_package_pairs,
    validate_teacher_cache,
)
from .engine import collect_embeddings
from .manifest import load_training_manifest
from .metrics import evaluate_teacher_outputs
from .prototype_labels import DEFAULT_L1_CLASSES, load_prototype_labels
from .roi import spatial_component_names
from .train import (
    _PackageShuffleBatchLoader,
    _assert_disjoint_package_cohorts,
    _optimizer_visible_contract_sha256,
    _paths_from_data,
    _teacher_paths_from_data,
    _validation_package_keep_indices,
    _verify_frozen_supervision_assets,
)
from .utils import write_json


def _load_prototype_map(cfg: dict, dims: dict[str, int]) -> dict[str, PrototypeRegistry] | None:
    if float(cfg["loss"].get("semantic_weight", 0.0)) == 0:
        return None
    prototype_paths = cfg["data"].get("prototype_paths")
    if isinstance(prototype_paths, dict):
        return {name: load_prototype_registry(prototype_paths[name], expected_dim=dim) for name, dim in dims.items()}
    prototype_path = cfg["data"].get("prototype_path")
    if prototype_path is None:
        return None
    return {name: load_prototype_registry(prototype_path, expected_dim=dim) for name, dim in dims.items()}


def _prototype_source_splits(cfg: dict, split: str) -> set[str] | None:
    value = cfg["data"].get(f"prototype_supervision_{split}_splits", [split])
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return {str(item) for item in value}


def _checkpoint_config_contract(cfg: dict) -> dict:
    """Remove host-only and train-derived fields before strict comparison."""

    contract = copy.deepcopy(cfg)
    contract.pop("research_contract", None)
    contract.get("runtime", {}).pop("device", None)
    contract.get("runtime", {}).pop("output_dir", None)
    for key in (
        "num_workers",
        "prefetch_factor",
        "persistent_workers",
        "package_pin_memory",
        "expert_replay_tiles",
        "expert_replay_interval_batches",
        "expert_batch_size",
        "spatial_component_names",
        "optimizer_visible_tile_packages",
        "optimizer_visible_contract_sha256",
        "supervision_asset_sha256",
    ):
        contract.get("data", {}).pop(key, None)
    for key in (
        "log_interval",
        "progress",
        "progress_interval_sec",
        "tensorboard",
        "tensorboard_batch_interval",
        "tensorboard_log_dir",
    ):
        contract.get("train", {}).pop(key, None)
    return contract


def _use_checkpoint_config(requested: dict, payload: dict) -> dict:
    saved = payload.get("config")
    if not isinstance(saved, dict):
        raise ValueError("checkpoint has no resolved training config")
    if _checkpoint_config_contract(requested) != _checkpoint_config_contract(
        saved
    ):
        raise ValueError(
            "evaluation config differs from the checkpoint model/data contract"
        )
    cfg = copy.deepcopy(saved)
    cfg.setdefault("runtime", {}).update(
        {
            key: requested["runtime"][key]
            for key in ("device", "output_dir")
            if key in requested.get("runtime", {})
        }
    )
    for key in (
        "num_workers",
        "prefetch_factor",
        "persistent_workers",
        "package_pin_memory",
    ):
        if key in requested.get("data", {}):
            cfg.setdefault("data", {})[key] = requested["data"][key]
    return cfg


def _explicit_split_data_paths(
    cfg: dict,
    split: str,
) -> tuple[list[str], dict[str, list[str]]] | None:
    tile_key = f"{split}_image_tile_package_paths"
    teacher_key = f"{split}_teacher_feature_package_paths"
    has_tiles = tile_key in cfg.get("data", {})
    has_teachers = teacher_key in cfg.get("data", {})
    if has_tiles != has_teachers:
        raise ValueError(
            f"data.{tile_key} and data.{teacher_key} must be configured together"
        )
    if not has_tiles:
        return None
    return (
        _paths_from_data(cfg, tile_key),
        _teacher_paths_from_data(cfg, teacher_key),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained HCC-SemPath checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    args = parser.parse_args()
    requested_cfg = load_config(args.config)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = _use_checkpoint_config(requested_cfg, payload)
    _verify_frozen_supervision_assets(cfg)
    device = torch.device(cfg["runtime"]["device"])
    manifest_path = cfg["data"].get("train_manifest_path")
    if manifest_path:
        manifest = load_training_manifest(manifest_path)
        tile_packages, teacher_packages = manifest_data_paths(cfg, manifest, args.split)
        names = teacher_names(cfg)
    else:
        explicit_paths = _explicit_split_data_paths(cfg, args.split)
        if explicit_paths is None:
            tile_packages = image_tile_package_paths(cfg)
            teacher_packages = teacher_feature_package_paths(cfg)
        else:
            tile_packages, teacher_packages = explicit_paths
        names = list(teacher_packages)
    if args.split != "train":
        if manifest_path:
            train_packages, _ = manifest_data_paths(
                cfg,
                manifest,
                "train",
            )
            _assert_disjoint_package_cohorts(
                train_packages,
                tile_packages,
            )
        else:
            explicit_train = _explicit_split_data_paths(cfg, "train")
            if explicit_train is not None:
                _assert_disjoint_package_cohorts(
                    explicit_train[0],
                    tile_packages,
                )
        optimizer_packages = cfg["data"].get(
            "optimizer_visible_tile_packages"
        )
        optimizer_digest = str(
            cfg["data"].get(
                "optimizer_visible_contract_sha256",
                "",
            )
        )
        if (
            not isinstance(optimizer_packages, list)
            or not optimizer_packages
            or len(optimizer_digest) != 64
        ):
            raise ValueError(
                "checkpoint has no frozen optimizer-visible cohort contract"
            )
        optimizer_packages = [
            str(path) for path in optimizer_packages
        ]
        if (
            _optimizer_visible_contract_sha256(optimizer_packages)
            != optimizer_digest
        ):
            raise ValueError(
                "optimizer-visible cohort changed after checkpoint creation"
            )
        keep = _validation_package_keep_indices(
            tile_packages,
            optimizer_packages,
        )
        if len(keep) != len(tile_packages):
            tile_packages = [
                tile_packages[index] for index in keep
            ]
            teacher_packages = {
                name: [paths[index] for index in keep]
                for name, paths in teacher_packages.items()
            }
            if not tile_packages:
                raise ValueError(
                    "optimizer-visible cohort exclusion left an empty "
                    "evaluation split"
                )
    validate_training_config(cfg, names)
    dims = teacher_dims(cfg, names)
    tile_metadata = read_package_metadata(tile_packages[0])
    image_size = (int(tile_metadata["tile_height"]), int(tile_metadata["tile_width"]))
    if image_size != (STUDENT_IMAGE_SIZE, STUDENT_IMAGE_SIZE):
        raise ValueError(
            "checkpoint evaluation requires native "
            f"{STUDENT_IMAGE_SIZE}px tiles, got {image_size}"
        )
    for package_path in tile_packages[1:]:
        metadata = read_package_metadata(package_path)
        candidate_size = (int(metadata["tile_height"]), int(metadata["tile_width"]))
        if candidate_size != image_size:
            raise ValueError(f"tile package size mismatch: {package_path} has {candidate_size}, expected {image_size}")
    l1_class_names = [
        str(name)
        for name in cfg["model"].get("l1_class_names", DEFAULT_L1_CLASSES)
    ]
    frozen_spatial_names = cfg["data"].get("spatial_component_names")
    spatial_names = (
        [str(name) for name in frozen_spatial_names]
        if frozen_spatial_names
        else (
            spatial_component_names(cfg["data"]["spatial_manifest_path"])
            if cfg["data"].get("spatial_manifest_path")
            else []
        )
    )
    if not manifest_path:
        all_records = read_packaged_tile_records(tile_packages)
        all_records = apply_split_overrides(
            all_records,
            cfg["data"].get("split_manifest_path"),
            cfg["data"].get("split_key", "slide_id"),
        )
        records = [
            record
            for record in all_records
            if record.record.split == args.split
        ]
    prototype_labels = load_prototype_labels(
        cfg["data"].get("prototype_supervision_manifest_path"),
        l1_class_names,
        allowed_source_splits=_prototype_source_splits(cfg, args.split),
    )
    if manifest_path:
        validate_teacher_feature_package_pairs(
            tile_packages,
            teacher_packages,
            expected_dims=dims,
        )
        dataset = PackageSampledDistillationDataset(
            tile_packages,
            teacher_packages,
            image_size=image_size,
            mean=cfg["data"].get("mean"),
            std=cfg["data"].get("std"),
            tensor_collate=True,
            expected_dims=dims,
            prototype_labels=prototype_labels,
        )
        loader = _PackageShuffleBatchLoader(
            dataset,
            batch_size=int(cfg["train"]["batch_size"]),
            num_workers=int(cfg["data"]["num_workers"]),
            prefetch_batches=int(cfg["data"].get("prefetch_factor", 2)),
            collate_fn=dataset.collate,
            seed=int(cfg["runtime"]["seed"]) + 1,
            chunk_size=int(
                cfg["data"].get(
                    "package_chunk_size",
                    cfg["train"]["batch_size"],
                )
            ),
            buffer_batches=int(
                cfg["data"].get("package_buffer_batches", 4)
            ),
            reshuffle_each_epoch=False,
            pin_memory=bool(
                cfg["data"].get("package_pin_memory", False)
            ),
        )
    else:
        validate_teacher_cache(
            records,
            None,
            dims,
            teacher_cache_package_paths=teacher_packages,
        )
        dataset = DistillationTileDataset(
            records,
            None,
            image_size,
            mean=cfg["data"].get("mean"),
            std=cfg["data"].get("std"),
            teacher_cache_package_paths=teacher_packages,
            prototype_labels=prototype_labels,
        )
        loader = DataLoader(
            dataset,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=cfg["data"]["num_workers"],
            collate_fn=collate_distillation,
        )
    model = HCCSemPathModel(
        backbone_name=STUDENT_BACKBONE_NAME,
        embedding_dim=embedding_dim(cfg),
        teacher_dims=dims,
        pretrained=False,
        projector_type=cfg["model"].get("projector_type", "linear"),
        projector_hidden_dim=int(cfg["model"].get("projector_hidden_dim", 2048)),
        teacher_head_type=cfg["model"].get("teacher_head_type", "linear"),
        l1_num_classes=len(l1_class_names),
        spatial_num_components=(
            len(spatial_names)
        ),
        spatial_dim=int(cfg["model"].get("spatial_dim", 256)),
        spatial_output_stride=int(cfg["model"].get("spatial_output_stride", 7)),
    ).to(device)
    model.load_state_dict(
        {
            key.removeprefix("_orig_mod."): value
            for key, value in payload["model"].items()
        }
    )
    embeddings, student_by_teacher, teacher_by_name, supervised = collect_embeddings(
        model,
        loader,
        device,
        cfg=cfg,
        max_batches=cfg["train"].get("max_eval_batches", cfg["train"].get("max_val_batches")),
    )
    prototypes = _load_prototype_map(cfg, dims)
    eval_pairwise_max_samples = int(cfg["train"].get("eval_pairwise_max_samples", 4096))
    metrics = evaluate_teacher_outputs(
        student_by_teacher,
        teacher_by_name,
        prototypes,
        int(cfg["train"]["topk"]),
        max_pairwise_samples=eval_pairwise_max_samples,
    )
    mask = supervised["prototype_mask"].bool()
    logits = supervised["l1_logits"]
    metrics["l1_evaluated_tiles"] = float(mask.sum())
    metrics["l1_accuracy"] = (
        float(
            (
                logits[mask].argmax(dim=1)
                == supervised["prototype_level1"][mask]
            )
            .float()
            .mean()
        )
        if bool(mask.any()) and logits.numel() > 0
        else 0.0
    )
    write_json(f"{cfg['runtime']['output_dir']}/eval_{args.split}.json", metrics)
    print("eval_ok " + " ".join(f"{k}={v:.6f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
