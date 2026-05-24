from __future__ import annotations

from pathlib import Path

import yaml

from ..io.feature_cache import FeatureCacheReader
from .manifest import manifest_teacher_feature_packages, manifest_tile_packages


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def teacher_names(cfg: dict) -> list[str]:
    teachers = cfg["data"].get("teachers")
    if teachers is not None:
        return [str(teacher) for teacher in teachers]
    dims = cfg["model"].get("teacher_dims")
    if isinstance(dims, dict):
        return [str(name) for name in dims]
    return list(teacher_feature_package_paths(cfg))


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
        return {name: int(dims[name]) for name in teacher_names}
    dim = int(model["teacher_dim"])
    return {name: dim for name in teacher_names}


def embedding_dim(cfg: dict) -> int:
    return int(cfg["model"].get("embedding_dim", cfg["model"].get("teacher_dim", 256)))


def manifest_data_paths(cfg: dict, manifest: dict, split: str) -> tuple[list[str], dict[str, list[str]]]:
    data = cfg["data"]
    feature_root = data.get("feature_root")
    if feature_root is None:
        raise ValueError("data.feature_root is required when data.train_manifest_path is used")
    teachers = teacher_names(cfg)
    tile_packages = [str(path) for path in manifest_tile_packages(manifest, split)]
    feature_packages = {
        name: [str(path) for path in paths]
        for name, paths in manifest_teacher_feature_packages(
            manifest=manifest,
            split=split,
            teachers=teachers,
            feature_root=feature_root,
            feature_suffix_template=data.get("feature_suffix_template", ".{teacher}.features.iac"),
        ).items()
    }
    return tile_packages, feature_packages
