from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import torch
from torch.utils.data import DataLoader
import numpy as np

from ..io.tile_package import read_package_metadata
from ..modeling.prototypes import PrototypeRegistry, load_prototype_registry
from ..modeling.models import HCCSemPathModel
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
    validate_teacher_cache,
)
from .engine import build_lr_scheduler, fit
from .manifest import load_training_manifest
from .prototype_labels import load_prototype_labels
from .utils import seed_everything


class _PackageShuffleBatchLoader:
    def __init__(
        self,
        dataset,
        *,
        batch_size: int,
        num_workers: int,
        prefetch_batches: int,
        collate_fn,
        seed: int = 13,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_workers = max(0, int(num_workers))
        self.prefetch_batches = max(0, int(prefetch_batches))
        self.collate_fn = collate_fn
        self.seed = int(seed)
        self.chunk_size = max(1, min(64, self.batch_size // 4))
        self.buffer_target = max(self.batch_size, self.batch_size * (self.prefetch_batches + 1))

    def _draw_batch(self, buffer: list[dict], rng: np.random.Generator) -> list[dict]:
        take = min(self.batch_size, len(buffer))
        chosen = rng.choice(len(buffer), size=take, replace=False)
        batch = [buffer[index] for index in chosen]
        for index in sorted((int(index) for index in chosen), reverse=True):
            buffer.pop(index)
        return batch

    def _ready_batches(self, buffer: list[dict], rng: np.random.Generator, *, final: bool = False):
        threshold = self.batch_size if final else self.buffer_target
        while len(buffer) >= threshold or (final and buffer):
            yield self.collate_fn(self._draw_batch(buffer, rng))

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        chunks = iter(self.dataset.iter_package_row_chunks(self.chunk_size, self.seed))
        buffer: list[dict] = []
        if self.num_workers <= 0:
            for package_idx, rows in chunks:
                buffer.extend(self.dataset.read_package_rows(package_idx, rows))
                yield from self._ready_batches(buffer, rng)
            yield from self._ready_batches(buffer, rng, final=True)
            return

        max_pending = max(1, self.num_workers * (self.prefetch_batches + 1))
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            pending: set[Future] = set()

            def fill_window() -> None:
                while len(pending) < max_pending:
                    try:
                        package_idx, rows = next(chunks)
                    except StopIteration:
                        return
                    pending.add(executor.submit(self.dataset.read_package_rows, package_idx, rows))

            fill_window()
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    buffer.extend(future.result())
                fill_window()
                yield from self._ready_batches(buffer, rng)
            yield from self._ready_batches(buffer, rng, final=True)

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


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


def _limit_records(records: list, limit: int, seed: int) -> list:
    if limit <= 0 or len(records) <= limit:
        return records
    groups: dict[str, list] = {}
    for item in records:
        package_path = getattr(item, "tile_package_path", None)
        key = str(package_path) if package_path is not None else item.record.slide_id
        groups.setdefault(key, []).append(item)
    if limit < len(groups):
        raise ValueError(
            f"max_records must be 0 or at least the selected package/group count so every group participates: "
            f"max_records={limit} groups={len(groups)}"
        )
    rng = np.random.default_rng(seed)
    group_keys = list(groups)
    group_sizes = np.asarray([len(groups[key]) for key in group_keys], dtype=np.int64)
    expected = group_sizes.astype(np.float64) * (limit / int(group_sizes.sum()))
    quotas = np.floor(expected).astype(np.int64)
    quotas = np.maximum(quotas, 1)
    quotas = np.minimum(quotas, group_sizes)
    overflow = int(quotas.sum() - limit)
    while overflow > 0:
        candidates = np.flatnonzero(quotas > 1)
        chosen = rng.choice(candidates, size=min(overflow, len(candidates)), replace=False)
        quotas[chosen] -= 1
        overflow = int(quotas.sum() - limit)
    remainder = int(limit - quotas.sum())
    if remainder > 0:
        capacity = group_sizes - quotas
        candidates = np.flatnonzero(capacity > 0)
        weights = expected[candidates] - np.floor(expected[candidates])
        if float(weights.sum()) <= 0:
            chosen = rng.choice(candidates, size=remainder, replace=False)
        else:
            chosen = rng.choice(candidates, size=remainder, replace=False, p=weights / weights.sum())
        quotas[chosen] += 1

    selected = []
    for key, quota in zip(group_keys, quotas):
        items = groups[key]
        if int(quota) >= len(items):
            selected.extend(items)
            continue
        stride = len(items) / int(quota)
        offset = float(rng.random()) * stride
        rows = np.floor(offset + np.arange(int(quota), dtype=np.float64) * stride).astype(np.int64)
        selected.extend(items[int(row)] for row in np.minimum(rows, len(items) - 1))
    rng.shuffle(selected)
    return selected


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
        if float(cfg["loss"].get("zhcc_proto_weight", 0.0)) > 0:
            raise ValueError("data.zhcc_prototype_path is required when loss.zhcc_proto_weight > 0")
        return None
    return load_prototype_registry(prototype_path, expected_dim=embedding_dim(cfg)).to(device)


def _prototype_source_splits(cfg: dict, key: str, default: list[str]) -> set[str] | None:
    value = cfg["data"].get(key, default)
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return {str(item) for item in value}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HCC-SemPath distillation model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["runtime"]["seed"]))
    device = torch.device(cfg["runtime"]["device"])
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
    zhcc_prototypes = _load_zhcc_prototypes(cfg, device)
    prototype_manifest_path = cfg["data"].get("prototype_supervision_manifest_path")
    if float(cfg["loss"].get("zhcc_proto_weight", 0.0)) > 0 and prototype_manifest_path is None:
        raise ValueError("data.prototype_supervision_manifest_path is required when loss.zhcc_proto_weight > 0")
    train_prototype_labels = load_prototype_labels(
        prototype_manifest_path,
        zhcc_prototypes.to("cpu") if zhcc_prototypes is not None else None,
        allowed_source_splits=_prototype_source_splits(cfg, "prototype_supervision_train_splits", ["train"]),
    )
    val_prototype_labels = load_prototype_labels(
        prototype_manifest_path,
        zhcc_prototypes.to("cpu") if zhcc_prototypes is not None else None,
        allowed_source_splits=_prototype_source_splits(cfg, "prototype_supervision_val_splits", ["val"]),
    )
    all_tile_packages = sorted(set(train_tile_packages + val_tile_packages))
    tile_metadata = read_package_metadata(all_tile_packages[0])
    image_size = (int(tile_metadata["tile_height"]), int(tile_metadata["tile_width"]))
    for package_path in all_tile_packages[1:]:
        metadata = read_package_metadata(package_path)
        candidate_size = (int(metadata["tile_height"]), int(metadata["tile_width"]))
        if candidate_size != image_size:
            raise ValueError(f"tile package size mismatch: {package_path} has {candidate_size}, expected {image_size}")
    dynamic_package_sampling = bool(cfg["data"].get("dynamic_package_sampling", False))
    if dynamic_package_sampling:
        common_dataset_kwargs = {
            "image_size": image_size,
            "mean": cfg["data"].get("mean"),
            "std": cfg["data"].get("std"),
        }
        train_ds = PackageSampledDistillationDataset(
            train_tile_packages,
            train_teacher_packages,
            **common_dataset_kwargs,
            max_records=int(cfg["data"].get("max_train_records", 0)),
            seed=int(cfg["runtime"]["seed"]),
            expected_dims=dims,
            train=True,
            augmentation=cfg.get("augmentation"),
            prototype_labels=train_prototype_labels,
        )
        val_ds = PackageSampledDistillationDataset(
            val_tile_packages,
            val_teacher_packages,
            **common_dataset_kwargs,
            max_records=int(cfg["data"].get("max_val_records", 0)),
            seed=int(cfg["runtime"]["seed"]) + 1,
            expected_dims=dims,
            train=False,
            augmentation=cfg.get("augmentation"),
            prototype_labels=val_prototype_labels,
        )
    elif manifest_path or explicit_split_packages:
        train_records = read_packaged_tile_records(train_tile_packages)
        val_records = read_packaged_tile_records(val_tile_packages)
        if not train_records or not val_records:
            raise ValueError("manifest must contain non-empty train and val splits")
        train_records = _limit_records(train_records, int(cfg["data"].get("max_train_records", 0)), int(cfg["runtime"]["seed"]))
        val_records = _limit_records(val_records, int(cfg["data"].get("max_val_records", 0)), int(cfg["runtime"]["seed"]) + 1)
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
            train=True,
            augmentation=cfg.get("augmentation"),
            prototype_labels=train_prototype_labels,
        )
        val_ds = DistillationTileDataset(
            val_records,
            **common_dataset_kwargs,
            teacher_cache_package_paths=val_teacher_packages,
            train=False,
            augmentation=cfg.get("augmentation"),
            prototype_labels=val_prototype_labels,
        )
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
        train_records = _limit_records(train_records, int(cfg["data"].get("max_train_records", 0)), int(cfg["runtime"]["seed"]))
        val_records = _limit_records(val_records, int(cfg["data"].get("max_val_records", 0)), int(cfg["runtime"]["seed"]) + 1)
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
            train=True,
            augmentation=cfg.get("augmentation"),
            prototype_labels=train_prototype_labels,
        )
        val_ds = DistillationTileDataset(
            val_records,
            **common_dataset_kwargs,
            teacher_cache_package_paths=val_teacher_packages,
            train=False,
            augmentation=cfg.get("augmentation"),
            prototype_labels=val_prototype_labels,
        )
    num_workers = int(cfg["data"]["num_workers"])
    loader_kwargs = {
        "batch_size": cfg["train"]["batch_size"],
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(cfg["data"].get("prefetch_factor", 2))
        loader_kwargs["persistent_workers"] = bool(cfg["data"].get("persistent_workers", True))
    if dynamic_package_sampling:
        prefetch_batches = int(cfg["data"].get("prefetch_factor", 2))
        train_loader = _PackageShuffleBatchLoader(
            train_ds,
            batch_size=int(cfg["train"]["batch_size"]),
            num_workers=num_workers,
            prefetch_batches=prefetch_batches,
            collate_fn=train_ds.collate,
            seed=int(cfg["runtime"]["seed"]),
        )
        val_loader = _PackageShuffleBatchLoader(
            val_ds,
            batch_size=int(cfg["train"]["batch_size"]),
            num_workers=num_workers,
            prefetch_batches=prefetch_batches,
            collate_fn=val_ds.collate,
            seed=int(cfg["runtime"]["seed"]) + 1,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            shuffle=True,
            collate_fn=collate_distillation,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_ds,
            shuffle=False,
            collate_fn=collate_distillation,
            **loader_kwargs,
        )
    prototypes = _load_prototype_map(cfg, dims, device)
    model = HCCSemPathModel(
        backbone_name=cfg["model"]["backbone_name"],
        embedding_dim=embedding_dim(cfg),
        teacher_dims=dims,
        pretrained=cfg["model"]["pretrained"],
        projector_type=cfg["model"].get("projector_type", "linear"),
        projector_hidden_dim=int(cfg["model"].get("projector_hidden_dim", 2048)),
        teacher_head_type=cfg["model"].get("teacher_head_type", "linear"),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = build_lr_scheduler(optimizer, cfg, len(train_loader))
    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_state["model"])
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
        if scheduler is not None and resume_state.get("scheduler") is not None:
            scheduler.load_state_dict(resume_state["scheduler"])
    metrics = fit(
        model,
        train_loader,
        val_loader,
        prototypes,
        optimizer,
        device,
        cfg,
        scheduler=scheduler,
        zhcc_prototypes=zhcc_prototypes,
        resume_state=resume_state,
    )
    print("train_ok " + " ".join(f"{k}={v}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
