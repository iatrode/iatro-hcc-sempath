from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from ..io.tile_package import read_package_metadata
from ..modeling.anchors import load_anchors
from ..modeling.models import HCCSemPathModel
from .config import (
    embedding_dim,
    image_tile_package_paths,
    load_config,
    manifest_data_paths,
    teacher_dims,
    teacher_feature_package_paths,
    teacher_names,
)
from .datasets import (
    DistillationTileDataset,
    apply_split_overrides,
    collate_distillation,
    read_packaged_tile_records,
    validate_teacher_cache,
)
from .engine import fit
from .manifest import load_training_manifest
from .utils import seed_everything


def _load_anchor_map(cfg: dict, dims: dict[str, int], device: torch.device) -> dict[str, torch.Tensor] | None:
    semantic_weight = float(cfg["loss"].get("semantic_weight", 0.0))
    if semantic_weight == 0:
        return None
    anchor_paths = cfg["data"].get("anchors_paths")
    if isinstance(anchor_paths, dict):
        return {name: load_anchors(anchor_paths[name], expected_dim=dim).to(device) for name, dim in dims.items()}
    anchor_path = cfg["data"].get("anchors_path")
    if anchor_path is None:
        raise ValueError("data.anchors_path or data.anchors_paths is required when semantic_weight > 0")
    return {name: load_anchors(anchor_path, expected_dim=dim).to(device) for name, dim in dims.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HCC-SemPath distillation model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["runtime"]["seed"]))
    device = torch.device(cfg["runtime"]["device"])
    manifest_path = cfg["data"].get("train_manifest_path")
    if manifest_path:
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
    dims = teacher_dims(cfg, names)
    all_tile_packages = sorted(set(train_tile_packages + val_tile_packages))
    tile_metadata = read_package_metadata(all_tile_packages[0])
    image_size = (int(tile_metadata["tile_height"]), int(tile_metadata["tile_width"]))
    for package_path in all_tile_packages[1:]:
        metadata = read_package_metadata(package_path)
        candidate_size = (int(metadata["tile_height"]), int(metadata["tile_width"]))
        if candidate_size != image_size:
            raise ValueError(f"tile package size mismatch: {package_path} has {candidate_size}, expected {image_size}")
    if manifest_path:
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
    validate_teacher_cache(
        train_records,
        None,
        dims,
        teacher_cache_package_paths=train_teacher_packages,
    )
    validate_teacher_cache(
        val_records,
        None,
        dims,
        teacher_cache_package_paths=val_teacher_packages,
    )
    common_dataset_kwargs = {
        "teacher_cache_dir": None,
        "image_size": image_size,
        "mean": cfg["data"].get("mean"),
        "std": cfg["data"].get("std"),
    }
    train_ds = DistillationTileDataset(
        train_records,
        **common_dataset_kwargs,
        teacher_cache_package_paths=train_teacher_packages,
    )
    val_ds = DistillationTileDataset(
        val_records,
        **common_dataset_kwargs,
        teacher_cache_package_paths=val_teacher_packages,
    )
    num_workers = int(cfg["data"]["num_workers"])
    loader_kwargs = {
        "batch_size": cfg["train"]["batch_size"],
        "num_workers": num_workers,
        "collate_fn": collate_distillation,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(cfg["data"].get("prefetch_factor", 2))
        loader_kwargs["persistent_workers"] = bool(cfg["data"].get("persistent_workers", True))
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    anchors = _load_anchor_map(cfg, dims, device)
    model = HCCSemPathModel(
        backbone_name=cfg["model"]["backbone_name"],
        embedding_dim=embedding_dim(cfg),
        teacher_dims=dims,
        pretrained=cfg["model"]["pretrained"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    if args.resume:
        payload = torch.load(args.resume, map_location=device)
        model.load_state_dict(payload["model"])
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
    metrics = fit(model, train_loader, val_loader, anchors, optimizer, device, cfg)
    print("train_ok " + " ".join(f"{k}={v}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
