from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from ..io.tile_package import read_package_metadata
from ..modeling.prototypes import PrototypeRegistry, load_prototype_registry
from ..modeling.models import HCCSemPathModel, STUDENT_BACKBONE_NAME
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
from .engine import collect_embeddings
from .manifest import load_training_manifest
from .metrics import evaluate_teacher_outputs
from .prototype_labels import load_prototype_labels
from .utils import write_json
from .zhcc_metrics import evaluate_zhcc_prototypes


def _load_prototype_map(cfg: dict, dims: dict[str, int]) -> dict[str, PrototypeRegistry] | None:
    if float(cfg["loss"].get("semantic_weight", 0.0)) == 0 and float(cfg["loss"].get("prototype_filter_weight", 0.0)) == 0:
        return None
    prototype_paths = cfg["data"].get("prototype_paths")
    if isinstance(prototype_paths, dict):
        return {name: load_prototype_registry(prototype_paths[name], expected_dim=dim) for name, dim in dims.items()}
    prototype_path = cfg["data"].get("prototype_path")
    if prototype_path is None:
        return None
    return {name: load_prototype_registry(prototype_path, expected_dim=dim) for name, dim in dims.items()}


def _load_zhcc_prototypes(cfg: dict) -> PrototypeRegistry | None:
    prototype_path = cfg["data"].get("zhcc_prototype_path")
    if prototype_path is None:
        return None
    return load_prototype_registry(prototype_path, expected_dim=embedding_dim(cfg))


def _prototype_source_splits(cfg: dict, split: str) -> set[str] | None:
    value = cfg["data"].get(f"prototype_supervision_{split}_splits", [split])
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return {str(item) for item in value}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained HCC-SemPath checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(cfg["runtime"]["device"])
    manifest_path = cfg["data"].get("train_manifest_path")
    if manifest_path:
        manifest = load_training_manifest(manifest_path)
        tile_packages, teacher_packages = manifest_data_paths(cfg, manifest, args.split)
        names = teacher_names(cfg)
    else:
        tile_packages = image_tile_package_paths(cfg)
        teacher_packages = teacher_feature_package_paths(cfg)
        names = list(teacher_packages)
    validate_training_config(cfg, names)
    dims = teacher_dims(cfg, names)
    tile_metadata = read_package_metadata(tile_packages[0])
    image_size = (int(tile_metadata["tile_height"]), int(tile_metadata["tile_width"]))
    for package_path in tile_packages[1:]:
        metadata = read_package_metadata(package_path)
        candidate_size = (int(metadata["tile_height"]), int(metadata["tile_width"]))
        if candidate_size != image_size:
            raise ValueError(f"tile package size mismatch: {package_path} has {candidate_size}, expected {image_size}")
    if manifest_path:
        records = read_packaged_tile_records(tile_packages)
    else:
        all_records = read_packaged_tile_records(tile_packages)
        all_records = apply_split_overrides(
            all_records,
            cfg["data"].get("split_manifest_path"),
            cfg["data"].get("split_key", "slide_id"),
        )
        records = [record for record in all_records if record.record.split == args.split]
    validate_teacher_cache(
        records,
        None,
        dims,
        teacher_cache_package_paths=teacher_packages,
    )
    zhcc_prototypes = _load_zhcc_prototypes(cfg)
    prototype_labels = load_prototype_labels(
        cfg["data"].get("prototype_supervision_manifest_path"),
        zhcc_prototypes,
        allowed_source_splits=_prototype_source_splits(cfg, args.split),
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
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"], collate_fn=collate_distillation)
    model = HCCSemPathModel(
        backbone_name=STUDENT_BACKBONE_NAME,
        embedding_dim=embedding_dim(cfg),
        teacher_dims=dims,
        pretrained=False,
        projector_type=cfg["model"].get("projector_type", "linear"),
        projector_hidden_dim=int(cfg["model"].get("projector_hidden_dim", 2048)),
        teacher_head_type=cfg["model"].get("teacher_head_type", "linear"),
    ).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
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
    metrics.update(
        evaluate_zhcc_prototypes(
            embeddings,
            supervised["prototype_mask"],
            supervised["prototype_level1"],
            supervised["prototype_level2"],
            zhcc_prototypes,
            topk=int(cfg["train"]["topk"]),
            max_pairwise_samples=eval_pairwise_max_samples,
        )
    )
    write_json(f"{cfg['runtime']['output_dir']}/eval_{args.split}.json", metrics)
    print("eval_ok " + " ".join(f"{k}={v:.6f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
