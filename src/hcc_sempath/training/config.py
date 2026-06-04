from __future__ import annotations

from pathlib import Path
import random

import yaml

from ..io.feature_cache import FeatureCacheReader
from ..io.iatrocache import read_header
from .manifest import manifest_teacher_feature_packages_for_tiles, manifest_tile_packages


EXCLUDED_TEACHER_NAMES = {
    "h1",
    "h-1",
    "h_1",
    "h1-family",
    "h1_family",
    "h1family",
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key == "inherits":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    parent = cfg.get("inherits")
    if parent is None:
        return cfg
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _deep_merge(load_config(parent_path), cfg)


def _normalize_teacher_name(name: str) -> str:
    return str(name).strip().lower()


def validate_teacher_names(names: list[str]) -> None:
    if not names:
        raise ValueError("at least one teacher is required")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate teacher names: {duplicates}")
    excluded = [name for name in names if _normalize_teacher_name(name) in EXCLUDED_TEACHER_NAMES]
    if excluded:
        raise ValueError(f"excluded unsupported teacher name configured: {excluded}")


def _unexpected_keys(payload: dict | None, expected: set[str], label: str) -> None:
    if not isinstance(payload, dict):
        return
    extras = sorted(str(name) for name in payload if str(name) not in expected)
    if extras:
        raise ValueError(f"{label} contains unknown teacher entries: {extras}; expected={sorted(expected)}")


def teacher_names(cfg: dict) -> list[str]:
    teachers = cfg["data"].get("teachers")
    if teachers is not None:
        names = [str(teacher) for teacher in teachers]
        validate_teacher_names(names)
        return names
    dims = cfg["model"].get("teacher_dims")
    if isinstance(dims, dict):
        names = [str(name) for name in dims]
        validate_teacher_names(names)
        return names
    names = list(teacher_feature_package_paths(cfg))
    validate_teacher_names(names)
    return names


def image_tile_package_paths(cfg: dict) -> list[str]:
    data = cfg["data"]
    package_paths = data.get("image_tile_package_paths")
    if package_paths is not None:
        if isinstance(package_paths, dict):
            return [str(path) for path in package_paths.values()]
        return [str(path) for path in package_paths]
    package_path = data.get("image_tile_package_path")
    if package_path is None:
        raise ValueError("data.image_tile_package_path or data.image_tile_package_paths is required")
    return [str(package_path)]


def teacher_feature_package_paths(cfg: dict) -> dict[str, list[str]]:
    data = cfg["data"]
    package_paths = data.get("teacher_feature_package_paths")
    if isinstance(package_paths, dict):
        resolved = {}
        for name, value in package_paths.items():
            if isinstance(value, (list, tuple)):
                resolved[str(name)] = [str(path) for path in value]
            else:
                resolved[str(name)] = [str(value)]
        return resolved
    if package_paths is not None:
        paths = {}
        for path in package_paths:
            reader = FeatureCacheReader(path)
            try:
                name = str(reader.header.get("teacher") or Path(path).stem)
            finally:
                reader.close()
            if name in paths:
                raise ValueError(f"duplicate teacher feature package name: {name}")
            paths[name] = [str(path)]
        return paths
    package_path = data.get("teacher_feature_package_path")
    if package_path is None:
        raise ValueError("data.teacher_feature_package_path or data.teacher_feature_package_paths is required")
    reader = FeatureCacheReader(package_path)
    try:
        name = str(reader.header.get("teacher") or "teacher")
    finally:
        reader.close()
    return {name: [str(package_path)]}


def teacher_dims(cfg: dict, teacher_names: list[str]) -> dict[str, int]:
    model = cfg["model"]
    dims = model.get("teacher_dims")
    if isinstance(dims, dict):
        expected = set(teacher_names)
        _unexpected_keys(dims, expected, "model.teacher_dims")
        missing = sorted(name for name in teacher_names if name not in dims)
        if missing:
            raise ValueError(f"model.teacher_dims missing teacher entries: {missing}")
        return {name: int(dims[name]) for name in teacher_names}
    dim = int(model["teacher_dim"])
    return {name: dim for name in teacher_names}


def embedding_dim(cfg: dict) -> int:
    return int(cfg["model"].get("embedding_dim", cfg["model"].get("teacher_dim", 256)))


def _package_record_count(path: Path) -> int:
    return int(read_header(path)["num_records"])


def _select_package_fraction(
    packages: list[Path],
    counts: dict[Path, int],
    *,
    fraction: float,
    seed: int,
) -> list[Path]:
    if fraction >= 1.0:
        return sorted(packages)
    if fraction <= 0.0:
        raise ValueError(f"tile fraction must be > 0: {fraction}")
    target_tiles = max(1, round(sum(counts[path] for path in packages) * fraction))
    rng = random.Random(seed)
    shuffled = packages[:]
    rng.shuffle(shuffled)
    selected = []
    selected_tiles = 0
    for path in shuffled:
        selected.append(path)
        selected_tiles += counts[path]
        if selected_tiles >= target_tiles:
            break
    return sorted(selected)


def validate_training_config(cfg: dict, names: list[str]) -> None:
    validate_teacher_names(names)
    expected = set(names)
    _unexpected_keys(cfg.get("model", {}).get("teacher_dims"), expected, "model.teacher_dims")
    _unexpected_keys(cfg.get("loss", {}).get("teacher_weights"), expected, "loss.teacher_weights")

    semantic_weight = float(cfg.get("loss", {}).get("semantic_weight", 0.0))
    prototype_filter_weight = float(cfg.get("loss", {}).get("prototype_filter_weight", 0.0))
    loss_cfg = cfg.get("loss", {})
    for key in ("zhcc_primary_temperature", "zhcc_attribute_temperature", "primary_temperature", "attribute_temperature"):
        if key in loss_cfg and float(loss_cfg[key]) <= 0:
            raise ValueError(f"loss.{key} must be positive")
    l1_weight = float(loss_cfg.get("prototype_l1_agreement_weight", 0.5))
    l2_weight = float(loss_cfg.get("prototype_l2_agreement_weight", 0.5))
    if l1_weight < 0 or l2_weight < 0 or (l1_weight + l2_weight) <= 0:
        raise ValueError("prototype L1/L2 agreement weights must be non-negative and not both zero")
    prototype_paths = cfg.get("data", {}).get("prototype_paths")
    if isinstance(prototype_paths, dict):
        _unexpected_keys(prototype_paths, expected, "data.prototype_paths")
        if semantic_weight > 0 or prototype_filter_weight > 0:
            missing = sorted(name for name in names if name not in prototype_paths)
            if missing:
                raise ValueError(f"data.prototype_paths missing teacher entries: {missing}")


def manifest_data_paths(cfg: dict, manifest: dict, split: str) -> tuple[list[str], dict[str, list[str]]]:
    data = cfg["data"]
    feature_root = data.get("feature_root")
    if feature_root is None and not isinstance(manifest.get("feature_roots"), dict):
        raise ValueError("data.feature_root or manifest.feature_roots is required when data.train_manifest_path is used")
    teachers = teacher_names(cfg)
    tile_paths = manifest_tile_packages(manifest, split)
    fraction_key = "train_tile_fraction" if split == "train" else f"{split}_tile_fraction"
    fraction = float(data.get(fraction_key, 1.0))
    tile_paths = _select_package_fraction(
        tile_paths,
        {path: _package_record_count(path) for path in tile_paths},
        fraction=fraction,
        seed=int(cfg.get("runtime", {}).get("seed", 13)) + (0 if split == "train" else 1),
    )
    tile_packages = [str(path) for path in tile_paths]
    feature_packages = {
        name: [str(path) for path in paths]
        for name, paths in manifest_teacher_feature_packages_for_tiles(
            manifest=manifest,
            tile_paths=tile_paths,
            teachers=teachers,
            feature_root=feature_root,
            feature_suffix_template=data.get("feature_suffix_template", ".{teacher}.features.iac"),
        ).items()
    }
    return tile_packages, feature_packages
