from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import math
import random

import yaml

from iatro.iac.adapters.features import FeatureCacheReader
from iatro.iac import read_header
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


# Teacher-keyed maps describe the active teacher set. When a child config
# provides one, it should define that set wholesale rather than union-merge with
# the parent's teachers — otherwise a single-teacher override cannot drop the
# parent's other teachers and config validation rejects the leftovers.
_REPLACE_ON_OVERRIDE_KEYS = frozenset({"teacher_dims", "teacher_weights", "prototype_paths"})


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key == "inherits":
            continue
        if key in _REPLACE_ON_OVERRIDE_KEYS:
            result[key] = value
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
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


@lru_cache(maxsize=None)
def _package_record_count(path: Path) -> int:
    return int(read_header(path)["num_records"])


def _package_record_counts(paths: list[Path]) -> dict[Path, int]:
    unique = list(dict.fromkeys(paths))
    if not unique:
        return {}
    with ThreadPoolExecutor(max_workers=min(32, len(unique))) as executor:
        counts = executor.map(_package_record_count, unique)
        return dict(zip(unique, counts, strict=True))


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
            first_root_path = Path(first_root)
            candidates.extend(
                [
                    first_root_path.parent / "merged" / tile_path.parent.name / f"{stem}{MERGED_FEATURE_SUFFIX}",
                    first_root_path / tile_path.parent.name / f"{stem}{MERGED_FEATURE_SUFFIX}",
                ]
            )
        if feature_root is not None:
            candidates.append(Path(feature_root) / "merged" / tile_path.parent.name / f"{stem}{MERGED_FEATURE_SUFFIX}")
    elif feature_root is not None:
        root = Path(feature_root)
        candidates.extend(
            [
                root / "merged" / tile_path.parent.name / f"{stem}{MERGED_FEATURE_SUFFIX}",
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
    unsupported_model_keys = sorted(
        key for key in ("backbone_name", "pretrained", "roi_patch_size") if key in cfg.get("model", {})
    )
    if unsupported_model_keys:
        raise ValueError(
            "student backbone is a fixed pretrained DINOv2-S/14 research invariant; "
            f"remove model keys: {unsupported_model_keys}"
        )
    _unexpected_keys(cfg.get("model", {}).get("teacher_dims"), expected, "model.teacher_dims")
    _unexpected_keys(cfg.get("loss", {}).get("teacher_weights"), expected, "loss.teacher_weights")
    teacher_weights = cfg.get("loss", {}).get("teacher_weights")
    if isinstance(teacher_weights, dict):
        invalid = {
            str(name): value
            for name, value in teacher_weights.items()
            if not math.isfinite(float(value)) or float(value) < 0
        }
        if invalid:
            raise ValueError(
                "loss.teacher_weights must be finite and non-negative: "
                f"{invalid}"
            )
        if not any(float(value) > 0 for value in teacher_weights.values()):
            raise ValueError(
                "loss.teacher_weights requires at least one positive teacher"
            )

    unsupported_train_keys = sorted(
        key
        for key in (
            "pipeline_profile_interval",
            "batch_timing_interval",
            "system_profile_interval",
            "batch_profile_csv",
            "batch_profile_csv_interval",
            "batch_profile_fields",
            "system_profile_paths",
            "detailed_timing",
            "detailed_timing_sync",
            "torch_profile_batches",
            "torch_profile_row_limit",
            "torch_profile_record_shapes",
            "torch_profile_memory",
            "torch_profile_stack",
        )
        if key in cfg.get("train", {})
    )
    if unsupported_train_keys:
        raise ValueError(
            "unsupported profiling keys; use tqdm and TensorBoard metrics: "
            f"{unsupported_train_keys}"
        )
    if "warmup_epochs" in cfg.get("train", {}):
        raise ValueError(
            "train.warmup_epochs is unsupported for population-scale runs; "
            "use train.lr_warmup_steps"
        )

    semantic_weight = float(cfg.get("loss", {}).get("semantic_weight", 0.0))
    loss_cfg = cfg.get("loss", {})
    prototype_responses_enabled = (
        semantic_weight > 0
        or float(loss_cfg.get("prototype_filter_weight", 0.0)) > 0
        or float(loss_cfg.get("zhcc_response_weight", 0.0)) > 0
    )
    supported_loss_keys = {
        "teacher_weights",
        "feature_loss_type",
        "relation_weight",
        "semantic_weight",
        "semantic_temperature",
        "classification_temperature",
        "pamtd_classification_temperature",
        "prototype_filter_weight",
        "prototype_filter_alpha_min",
        "prototype_filter_start_step",
        "prototype_filter_ramp_steps",
        "prototype_consensus_weight",
        "prototype_label_weight",
        "prototype_student_weight",
        "classification_agreement_weight",
        "spatial_agreement_weight",
        "zhcc_response_weight",
        "zhcc_response_start_step",
        "zhcc_response_ramp_steps",
        "spatial_global_temperature",
        "classification_weight",
        "spatial_weight",
        "spatial_point_tolerance_cells",
        "spatial_abundance_point_weight",
        "spatial_brush_weight",
        "spatial_brush_top_fraction",
        "spatial_explicit_negative_weight",
        "spatial_implicit_negative_weight",
        "expert_supervision_start_step",
        "expert_supervision_ramp_steps",
        "spatial_detach_shared_encoder",
    }
    unknown_loss_keys = sorted(set(loss_cfg) - supported_loss_keys)
    if unknown_loss_keys:
        raise ValueError(
            f"unknown loss configuration keys: {unknown_loss_keys}"
        )
    unsupported_data_keys = sorted(
        key
        for key in (
            "zhcc_prototype_path",
            "zhcc_prototype_image_path",
            "roi_manifest_path",
            "roi_train_splits",
        )
        if key in cfg.get("data", {})
    )
    if unsupported_data_keys:
        raise ValueError(
            "unsupported data keys; use "
            f"data.spatial_manifest_path/spatial_train_splits: {unsupported_data_keys}"
        )
    unsupported_spatial_model_keys = sorted(
        key
        for key in (
            "roi_patch_dim",
            "roi_top_q",
            "roi_patch_temperature",
            "spatial_point_sigma",
        )
        if key in cfg.get("model", {})
    )
    if unsupported_spatial_model_keys:
        raise ValueError(
            "unsupported model keys; use model.spatial_* keys: "
            f"{unsupported_spatial_model_keys}"
        )
    for key in (
        "spatial_use_local_branch",
        "spatial_use_semantic_branch",
        "spatial_use_context",
    ):
        if key in cfg.get("model", {}) and not isinstance(cfg["model"][key], bool):
            raise ValueError(f"model.{key} must be boolean")
    if not (
        bool(cfg.get("model", {}).get("spatial_use_local_branch", True))
        or bool(cfg.get("model", {}).get("spatial_use_semantic_branch", True))
    ):
        raise ValueError(
            "model.spatial_use_local_branch and "
            "model.spatial_use_semantic_branch cannot both be false"
        )
    for key in (
        "relation_weight",
        "semantic_weight",
        "classification_weight",
        "spatial_weight",
        "spatial_abundance_point_weight",
        "spatial_brush_weight",
        "spatial_explicit_negative_weight",
        "spatial_implicit_negative_weight",
        "prototype_consensus_weight",
        "prototype_label_weight",
        "prototype_student_weight",
        "zhcc_response_weight",
    ):
        if key in loss_cfg and (
            not math.isfinite(float(loss_cfg[key]))
            or float(loss_cfg[key]) < 0
        ):
            raise ValueError(f"loss.{key} must be finite and non-negative")
    for key in (
        "expert_supervision_start_step",
        "expert_supervision_ramp_steps",
        "prototype_filter_start_step",
        "prototype_filter_ramp_steps",
        "zhcc_response_start_step",
        "zhcc_response_ramp_steps",
    ):
        if key in loss_cfg and int(loss_cfg[key]) < 0:
            raise ValueError(f"loss.{key} must be non-negative")
    if (
        "spatial_detach_shared_encoder" in loss_cfg
        and not isinstance(loss_cfg["spatial_detach_shared_encoder"], bool)
    ):
        raise ValueError(
            "loss.spatial_detach_shared_encoder must be boolean"
        )
    for key in (
        "semantic_temperature",
        "classification_temperature",
        "pamtd_classification_temperature",
        "spatial_global_temperature",
    ):
        if key in loss_cfg and (
            not math.isfinite(float(loss_cfg[key]))
            or float(loss_cfg[key]) <= 0
        ):
            raise ValueError(f"loss.{key} must be finite and positive")
    if str(loss_cfg.get("feature_loss_type", "cosine")) not in {
        "cosine",
        "cosine_plus_norm_mse",
        "cosine_plus_raw_mse",
    }:
        raise ValueError("loss.feature_loss_type is unsupported")
    for key in ("prototype_filter_weight", "prototype_filter_alpha_min"):
        value = float(loss_cfg.get(key, 0.0))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"loss.{key} must be in [0, 1]")
    reliability_mass = sum(
        float(loss_cfg.get(key, 1.0))
        for key in (
            "prototype_consensus_weight",
            "prototype_label_weight",
            "prototype_student_weight",
        )
    )
    adjudication_enabled = (
        float(loss_cfg.get("prototype_filter_weight", 0.0)) > 0
        or float(loss_cfg.get("zhcc_response_weight", 0.0)) > 0
    )
    if adjudication_enabled and reliability_mass <= 0:
        raise ValueError(
            "PAMT-D reliability requires at least one positive consensus, "
            "expert-label, or student-agreement coefficient"
        )
    if (
        adjudication_enabled
        and float(loss_cfg.get("prototype_filter_alpha_min", 0.25)) == 0
        and float(loss_cfg.get("prototype_consensus_weight", 1.0)) == 0
        and float(loss_cfg.get("prototype_student_weight", 1.0)) == 0
    ):
        raise ValueError(
            "PAMT-D population tiles can have zero reliability mass when "
            "alpha_min, consensus, and student-agreement coefficients are all zero"
        )
    point_tolerance = int(loss_cfg.get("spatial_point_tolerance_cells", 1))
    if point_tolerance < 0:
        raise ValueError("loss.spatial_point_tolerance_cells must be non-negative")
    brush_fraction = float(loss_cfg.get("spatial_brush_top_fraction", 0.25))
    if not math.isfinite(brush_fraction) or not (
        0.0 < brush_fraction <= 1.0
    ):
        raise ValueError("loss.spatial_brush_top_fraction must be in (0, 1]")
    if (
        "expert_replay_interval_batches" in cfg.get("data", {})
        and int(cfg["data"]["expert_replay_interval_batches"]) < 0
    ):
        raise ValueError(
            "data.expert_replay_interval_batches must be non-negative"
        )
    if (
        "expert_batch_size" in cfg.get("data", {})
        and int(cfg["data"]["expert_batch_size"]) <= 0
    ):
        raise ValueError("data.expert_batch_size must be positive")
    for key in (
        "dynamic_prototype_refresh_steps",
        "dynamic_spatial_prototype_refresh_steps",
    ):
        if key in cfg.get("train", {}) and int(cfg["train"][key]) < 0:
            raise ValueError(f"train.{key} must be non-negative")
    if int(cfg.get("train", {}).get("step_metrics_flush_steps", 50)) <= 0:
        raise ValueError("train.step_metrics_flush_steps must be positive")
    if int(
        cfg.get("train", {}).get("development_probe_interval_steps", 0)
    ) < 0:
        raise ValueError(
            "train.development_probe_interval_steps must be non-negative"
        )
    if int(cfg.get("train", {}).get("development_probe_batches", 64)) <= 0:
        raise ValueError("train.development_probe_batches must be positive")
    if int(cfg.get("train", {}).get("lr_warmup_steps", 0)) < 0:
        raise ValueError("train.lr_warmup_steps must be non-negative")
    if (
        "dynamic_prototype_batch_size" in cfg.get("train", {})
        and int(cfg["train"]["dynamic_prototype_batch_size"]) <= 0
    ):
        raise ValueError(
            "train.dynamic_prototype_batch_size must be positive"
        )
    if (
        cfg.get("data", {}).get("spatial_manifest_path")
        and bool(cfg.get("train", {}).get("early_stop_teacher_alignment", False))
    ):
        raise ValueError(
            "spatial training requires the prescribed terminal epoch; "
            "train.early_stop_teacher_alignment must be false"
        )
    prototype_paths = cfg.get("data", {}).get("prototype_paths")
    if isinstance(prototype_paths, dict):
        _unexpected_keys(prototype_paths, expected, "data.prototype_paths")
        if prototype_responses_enabled:
            missing = sorted(name for name in names if name not in prototype_paths)
            if missing:
                raise ValueError(f"data.prototype_paths missing teacher entries: {missing}")
    elif prototype_responses_enabled:
        if cfg.get("data", {}).get("prototype_path") is None:
            raise ValueError(
                "data.prototype_path or data.prototype_paths is required when "
                "semantic or PAMT-D prototype-response losses are enabled"
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
    if fraction < 1.0:
        tile_paths = _select_package_fraction(
            tile_paths,
            _package_record_counts(tile_paths),
            fraction=fraction,
            seed=int(cfg.get("runtime", {}).get("seed", 13)) + (0 if split == "train" else 1),
        )
    else:
        tile_paths = sorted(tile_paths)
    tile_packages = [str(path) for path in tile_paths]
    feature_packages = {name: [] for name in teachers}

    def resolve_feature_paths(tile_path: Path) -> dict[str, str]:
        merged_path = _existing_merged_feature_package_for_tile(cfg, manifest, tile_path, teachers, feature_root)
        if merged_path is not None:
            return {name: str(merged_path) for name in teachers}
        per_tile = manifest_teacher_feature_packages_for_tiles(
            manifest=manifest,
            tile_paths=[tile_path],
            teachers=teachers,
            feature_root=feature_root,
            feature_suffix_template=data.get("feature_suffix_template", ".{teacher}.features.iac"),
        )
        return {
            name: str(paths[0])
            for name, paths in per_tile.items()
        }

    with ThreadPoolExecutor(
        max_workers=min(32, max(1, len(tile_paths)))
    ) as executor:
        resolved_features = executor.map(resolve_feature_paths, tile_paths)
        for per_tile in resolved_features:
            for name in teachers:
                feature_packages[name].append(per_tile[name])
    return tile_packages, feature_packages
