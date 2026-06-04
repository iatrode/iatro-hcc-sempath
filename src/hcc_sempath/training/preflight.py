from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from ..io.tile_package import read_package_metadata
from ..modeling.models import HCCSemPathModel
from ..modeling.prototypes import PrototypeRegistry, load_prototype_registry
from .adjudication import prototype_adjudicated_teacher_weights
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
    apply_split_overrides,
    collate_distillation,
    read_packaged_tile_records,
    validate_teacher_cache,
)
from .losses import multi_teacher_distillation_loss
from .manifest import load_training_manifest
from .prototype_images import PrototypeImageBank, load_prototype_image_bank
from .prototype_labels import load_prototype_labels
from .utils import seed_everything


def _paths_from_data(cfg: dict, key: str) -> list[str]:
    value = cfg["data"].get(key)
    if value is None:
        raise ValueError(f"data.{key} is required")
    if isinstance(value, dict):
        return [str(path) for path in value.values()]
    return [str(path) for path in value]


def _teacher_paths_from_data(cfg: dict, key: str) -> dict[str, list[str]]:
    value = cfg["data"].get(key)
    if not isinstance(value, dict):
        raise ValueError(f"data.{key} must be a teacher->paths mapping")
    result = {}
    for name, paths in value.items():
        if isinstance(paths, (list, tuple)):
            result[str(name)] = [str(path) for path in paths]
        else:
            result[str(name)] = [str(paths)]
    return result


def _load_prototype_map(cfg: dict, dims: dict[str, int], device: torch.device) -> dict[str, PrototypeRegistry] | None:
    semantic_weight = float(cfg["loss"].get("semantic_weight", 0.0))
    prototype_filter_weight = float(cfg["loss"].get("prototype_filter_weight", 0.0))
    if semantic_weight == 0 and prototype_filter_weight == 0:
        return None
    prototype_paths = cfg["data"].get("prototype_paths")
    if isinstance(prototype_paths, dict):
        return {name: load_prototype_registry(prototype_paths[name], expected_dim=dim).to(device) for name, dim in dims.items()}
    prototype_path = cfg["data"].get("prototype_path")
    if prototype_path is None:
        raise ValueError(
            "data.prototype_path or data.prototype_paths is required when semantic_weight or prototype_filter_weight > 0"
        )
    return {name: load_prototype_registry(prototype_path, expected_dim=dim).to(device) for name, dim in dims.items()}


def _load_zhcc_prototypes(cfg: dict, device: torch.device) -> PrototypeRegistry | None:
    prototype_path = cfg["data"].get("zhcc_prototype_path")
    if prototype_path is None:
        if float(cfg["loss"].get("zhcc_response_weight", 0.0)) > 0:
            raise ValueError("data.zhcc_prototype_path is required only when loss.zhcc_response_weight > 0")
        return None
    return load_prototype_registry(prototype_path, expected_dim=embedding_dim(cfg)).to(device)


def _load_zhcc_image_bank(cfg: dict) -> PrototypeImageBank | None:
    image_path = cfg["data"].get("zhcc_prototype_image_path")
    if image_path is None:
        if float(cfg["loss"].get("zhcc_proto_weight", 0.0)) > 0:
            raise ValueError("data.zhcc_prototype_image_path is required when loss.zhcc_proto_weight > 0")
        return None
    return load_prototype_image_bank(image_path)


def _label_contract_registry(
    *,
    cfg: dict,
    prototypes: dict[str, PrototypeRegistry] | None,
    zhcc_prototypes: PrototypeRegistry | None,
    zhcc_image_bank: PrototypeImageBank | None,
) -> PrototypeRegistry | None:
    if zhcc_prototypes is not None:
        return zhcc_prototypes.to("cpu")
    if zhcc_image_bank is not None:
        return zhcc_image_bank.label_contract(embedding_dim(cfg))
    if prototypes:
        return next(iter(prototypes.values())).to("cpu")
    return None


def _prototype_source_splits(cfg: dict, key: str, default: list[str]) -> set[str] | None:
    value = cfg["data"].get(key, default)
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return {str(item) for item in value}


def _check_tile_package_sizes(package_paths: list[str]) -> tuple[int, int]:
    metadata = read_package_metadata(package_paths[0])
    image_size = (int(metadata["tile_height"]), int(metadata["tile_width"]))
    for package_path in package_paths[1:]:
        candidate = read_package_metadata(package_path)
        candidate_size = (int(candidate["tile_height"]), int(candidate["tile_width"]))
        if candidate_size != image_size:
            raise ValueError(f"tile package size mismatch: {package_path} has {candidate_size}, expected {image_size}")
    return image_size


def _sample_records(records: list, max_records: int | None) -> list:
    if max_records is None or max_records <= 0:
        return records
    return records[:max_records]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate HCC-SemPath training inputs before a long run.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-records", type=int, default=2048, help="Records per split to scan; use --full for all.")
    parser.add_argument("--full", action="store_true", help="Validate all records instead of the first --max-records per split.")
    parser.add_argument("--skip-model", action="store_true", help="Skip model construction and one-batch loss smoke test.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["runtime"]["seed"]))
    device = torch.device(cfg["runtime"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device is cuda but CUDA is not available")

    manifest_path = cfg["data"].get("train_manifest_path")
    explicit_split_packages = "train_image_tile_package_paths" in cfg["data"]
    if explicit_split_packages:
        train_tile_packages = _paths_from_data(cfg, "train_image_tile_package_paths")
        val_tile_packages = _paths_from_data(cfg, "val_image_tile_package_paths")
        train_teacher_packages = _teacher_paths_from_data(cfg, "train_teacher_feature_package_paths")
        val_teacher_packages = _teacher_paths_from_data(cfg, "val_teacher_feature_package_paths")
        names = list(train_teacher_packages)
    elif manifest_path:
        manifest = load_training_manifest(manifest_path)
        train_tile_packages, train_teacher_packages = manifest_data_paths(cfg, manifest, "train")
        val_tile_packages, val_teacher_packages = manifest_data_paths(cfg, manifest, "val")
        names = teacher_names(cfg)
    else:
        tile_packages = image_tile_package_paths(cfg)
        teacher_packages = teacher_feature_package_paths(cfg)
        train_tile_packages = tile_packages
        val_tile_packages = tile_packages
        train_teacher_packages = teacher_packages
        val_teacher_packages = teacher_packages
        names = list(teacher_packages)

    validate_training_config(cfg, names)
    dims = teacher_dims(cfg, names)
    image_size = _check_tile_package_sizes(sorted(set(train_tile_packages + val_tile_packages)))

    if manifest_path or explicit_split_packages:
        train_records = read_packaged_tile_records(train_tile_packages)
        val_records = read_packaged_tile_records(val_tile_packages)
    else:
        records = read_packaged_tile_records(train_tile_packages)
        records = apply_split_overrides(
            records,
            cfg["data"].get("split_manifest_path"),
            cfg["data"].get("split_key", "slide_id"),
        )
        train_records = [item for item in records if item.record.split == "train"]
        val_records = [item for item in records if item.record.split == "val"]
    if not train_records or not val_records:
        raise ValueError("manifest must contain non-empty train and val splits")

    max_records = None if args.full else args.max_records
    sampled_train = _sample_records(train_records, max_records)
    sampled_val = _sample_records(val_records, max_records)
    validate_teacher_cache(sampled_train, None, dims, teacher_cache_package_paths=train_teacher_packages)
    validate_teacher_cache(sampled_val, None, dims, teacher_cache_package_paths=val_teacher_packages)
    prototypes = _load_prototype_map(cfg, dims, device)
    zhcc_prototypes = _load_zhcc_prototypes(cfg, device)
    zhcc_image_bank = _load_zhcc_image_bank(cfg)
    label_contract = _label_contract_registry(
        cfg=cfg,
        prototypes=prototypes,
        zhcc_prototypes=zhcc_prototypes,
        zhcc_image_bank=zhcc_image_bank,
    )
    prototype_manifest_path = cfg["data"].get("prototype_supervision_manifest_path")
    prototype_label_required = (
        float(cfg["loss"].get("prototype_filter_weight", 0.0)) > 0
        and float(cfg["loss"].get("prototype_label_weight", 0.4)) > 0
    )
    if (float(cfg["loss"].get("zhcc_proto_weight", 0.0)) > 0 or prototype_label_required) and prototype_manifest_path is None:
        raise ValueError(
            "data.prototype_supervision_manifest_path is required when zhcc prototype supervision "
            "or prototype-label adjudication is enabled"
        )
    train_prototype_labels = load_prototype_labels(
        prototype_manifest_path,
        label_contract,
        allowed_source_splits=_prototype_source_splits(cfg, "prototype_supervision_train_splits", ["train"]),
    )

    if not args.skip_model:
        dataset = DistillationTileDataset(
            sampled_train,
            teacher_cache_dir=None,
            image_size=image_size,
            mean=cfg["data"].get("mean"),
            std=cfg["data"].get("std"),
            teacher_cache_package_paths=train_teacher_packages,
            prototype_labels=train_prototype_labels,
        )
        loader = DataLoader(
            dataset,
            batch_size=min(int(cfg["train"]["batch_size"]), len(dataset)),
            num_workers=0,
            collate_fn=collate_distillation,
        )
        batch = next(iter(loader))
        model = HCCSemPathModel(
            backbone_name=cfg["model"]["backbone_name"],
            embedding_dim=embedding_dim(cfg),
            teacher_dims=dims,
            pretrained=cfg["model"]["pretrained"],
        ).to(device)
        images = batch["images"].to(device)
        teachers = {name: tensor.to(device) for name, tensor in batch["teacher_features"].items()}
        outputs = model(images)
        if float(cfg["loss"].get("prototype_filter_weight", 0.0)) > 0:
            if prototypes is None:
                raise ValueError("prototype adjudication requires teacher prototype packages")
            teacher_sample_weights, _ = prototype_adjudicated_teacher_weights(
                teacher_by_name=teachers,
                prototypes_by_teacher=prototypes,
                zhcc_embedding_norm=outputs["embedding_norm"].detach(),
                zhcc_prototypes=zhcc_prototypes,
                prototype_mask=batch["prototype_mask"].to(device),
                prototype_level1=batch["prototype_level1"].to(device),
                prototype_level2=batch["prototype_level2"].to(device),
                alpha_min=float(cfg["loss"].get("prototype_filter_alpha_min", 0.25)),
                consensus_weight=float(cfg["loss"].get("consensus_weight", 0.4)),
                prototype_label_weight=float(cfg["loss"].get("prototype_label_weight", 0.4)),
                l1_agreement_weight=float(cfg["loss"].get("prototype_l1_agreement_weight", 0.5)),
                l2_agreement_weight=float(cfg["loss"].get("prototype_l2_agreement_weight", 0.5)),
                zhcc_response_weight=float(cfg["loss"].get("zhcc_response_weight", 0.2)),
                filter_strength=float(cfg["loss"].get("prototype_filter_weight", 0.0)),
            )
        else:
            teacher_sample_weights = None
        multi_teacher_distillation_loss(
            student_by_teacher=outputs["teacher_outputs"],
            teacher_by_name=teachers,
            prototypes_by_teacher=prototypes,
            relation_weight=float(cfg["loss"]["relation_weight"]),
            semantic_weight=float(cfg["loss"].get("semantic_weight", 0.0)),
            semantic_temperature=float(cfg["loss"]["semantic_temperature"]),
            teacher_weights=cfg["loss"].get("teacher_weights"),
            teacher_sample_weights=teacher_sample_weights,
            scale_relation_by_alpha=bool(cfg["loss"].get("scale_relation_by_alpha", False)),
        )

    print(
        "preflight_ok "
        f"teachers={','.join(names)} "
        f"train_records={len(train_records)} "
        f"val_records={len(val_records)} "
        f"checked_train={len(sampled_train)} "
        f"checked_val={len(sampled_val)}"
    )


if __name__ == "__main__":
    main()
