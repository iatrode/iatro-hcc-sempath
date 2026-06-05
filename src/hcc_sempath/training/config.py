from __future__ import annotations

from pathlib import Path
import random

import yaml

from ..io.feature_cache import FeatureCacheReader
from ..io.iatrocache import read_header
from .feature_pack_merge import MERGED_FEATURE_PAYLOAD_TYPE, MERGED_FEATURE_SUFFIX
from .manifest import manifest_teacher_feature_packages_for_tiles, manifest_tile_packages, package_stem


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
            header = read_header(path)
            if header.get("payload_type") == MERGED_FEATURE_PAYLOAD_TYPE:
                for name in header.get("teachers", []):
                    name = str(name)
                    paths.setdefault(name, []).append(str(path))
            else:
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


def _existing_merged_feature_package_for_tile(
    cfg: dict,
    manifest: dict,
    tile_path: Path,
    teachers: list[str],
    feature_root: str | Path | None,
) -> Path | None:
    if not bool(cfg.get("data", {}).get("prefer_merged_teacher_features", True)):
        return None
    stem = package_stem(tile_path, str(manifest.get("tile_suffix", ".tiles.iac")))
    candidates: list[Path] = []
    feature_roots = manifest.get("feature_roots")
    if isinstance(feature_roots, dict):
        first_root = feature_roots.get(teachers[0])
        if first_root is not None:
            candidates.append(Path(first_root) / tile_path.parent.name / f"{stem}{MERGED_FEATURE_SUFFIX}")
    elif feature_root is not None:
        root = Path(feature_root)
        candidates.extend(
            [
                root / teachers[0] / tile_path.parent.name / f"{stem}{MERGED_FEATURE_SUFFIX}",
                root / teachers[0] / f"{stem}{MERGED_FEATURE_SUFFIX}",
                root / tile_path.parent.name / f"{stem}{MERGED_FEATURE_SUFFIX}",
            ]
        )
    for path in candidates:
        if not path.exists():
            continue
        header = read_header(path)
        if header.get("payload_type") != MERGED_FEATURE_PAYLOAD_TYPE:
            raise ValueError(f"existing merged feature path has wrong payload_type: path={path}")
        missing = sorted(set(teachers).difference(str(name) for name in header.get("teachers", [])))
        if missing:
            raise ValueError(f"existing merged feature package missing teachers: path={path} teachers={missing}")
        tile_count = int(read_header(tile_path)["num_records"])
        if int(header.get("num_records", -1)) != tile_count:
            raise ValueError(
                f"existing merged feature/tile record count mismatch: path={path} "
                f"features={header.get('num_records')} tiles={tile_count}"
            )
        dims = cfg.get("model", {}).get("teacher_dims")
        if isinstance(dims, dict):
            merged_dims = {str(k): int(v) for k, v in header.get("teacher_dims", {}).items()}
            for teacher in teachers:
                if int(dims[teacher]) != int(merged_dims.get(teacher, -1)):
                    raise ValueError(
                        f"existing merged feature dim mismatch: teacher={teacher} "
                        f"expected={dims[teacher]} got={merged_dims.get(teacher)} path={path}"
                    )
        return path
    return None


def validate_training_config(cfg: dict, names: list[str]) -> None:
    validate_teacher_names(names)
    expected = set(names)
    _unexpected_keys(cfg.get("model", {}).get("teacher_dims"), expected, "model.teacher_dims")
    _unexpected_keys(cfg.get("loss", {}).get("teacher_weights"), expected, "loss.teacher_weights")

    semantic_weight = float(cfg.get("loss", {}).get("semantic_weight", 0.0))
    prototype_filter_weight = float(cfg.get("loss", {}).get("prototype_filter_weight", 0.0))
    zhcc_proto_weight = float(cfg.get("loss", {}).get("zhcc_proto_weight", 0.0))
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
        if semantic_weight > 0 or prototype_filter_weight > 0 or zhcc_proto_weight > 0:
            missing = sorted(name for name in names if name not in prototype_paths)
            if missing:
                raise ValueError(f"data.prototype_paths missing teacher entries: {missing}")
    elif semantic_weight > 0 or prototype_filter_weight > 0 or zhcc_proto_weight > 0:
        if cfg.get("data", {}).get("prototype_path") is None:
            raise ValueError(
                "data.prototype_path or data.prototype_paths is required when semantic_weight, "
                "prototype_filter_weight, or zhcc_proto_weight > 0"
            )


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
    feature_packages = {name: [] for name in teachers}
    for tile_path in tile_paths:
        merged_path = _existing_merged_feature_package_for_tile(cfg, manifest, tile_path, teachers, feature_root)
        if merged_path is not None:
            for name in teachers:
                feature_packages[name].append(str(merged_path))
            continue
        per_tile = manifest_teacher_feature_packages_for_tiles(
            manifest=manifest,
            tile_paths=[tile_path],
            teachers=teachers,
            feature_root=feature_root,
            feature_suffix_template=data.get("feature_suffix_template", ".{teacher}.features.iac"),
        )
        for name, paths in per_tile.items():
            feature_packages[name].append(str(paths[0]))
    return tile_packages, feature_packages
