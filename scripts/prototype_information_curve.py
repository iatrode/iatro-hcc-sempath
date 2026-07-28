#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hcc_sempath.annotation_information import (  # noqa: E402
    CurveObservation,
    aggregate_fixed_probe_curves,
    fixed_probe_curve,
    meaningful_reference_checkpoints,
    prepare_fixed_probe_split,
    tail_plateau,
)


@dataclass(frozen=True)
class PrototypeSample:
    tile_id: str
    level1_label: str
    level2_labels: tuple[str, ...]
    slide_id: str
    patient_id: str
    center: str


@dataclass
class Centers:
    level1: dict[str, np.ndarray]
    level2: dict[str, np.ndarray]
    level1_counts: dict[str, int]
    level2_counts: dict[str, int]


TEACHER_ALIASES: dict[str, tuple[str, ...]] = {
    "gigapath": ("gigapath", "prov-gigapath-local"),
    "h_optimus_1": ("h_optimus_1", "h1", "h-optimus-1", "h_optimus"),
    "uni2_h": ("uni2_h", "uni2"),
    "virchow2": ("virchow2",),
}

CANONICAL_TEACHER_BY_ALIAS = {
    alias: canonical
    for canonical, aliases in TEACHER_ALIASES.items()
    for alias in aliases
}

DEFAULT_PROTOTYPE_SAMPLE_COUNTS = "100,200,400,800,1200,1600,2000,3000"
DEFAULT_PLOT_FORMATS = "png,pdf"
DEFAULT_SEED = 13
DEFAULT_PROTOTYPE_SAMPLE_GROUP_KEY = "tile_id"
DEFAULT_INFOSPACE_TOPK = 5
DEFAULT_BOOTSTRAP_ITERATIONS = 500
DEFAULT_PLATEAU_NOVELTY_THRESHOLD = 0.02
DEFAULT_PLATEAU_DRIFT_THRESHOLD = 0.01
DEFAULT_PLATEAU_REDUNDANCY_THRESHOLD = 0.98
DEFAULT_PCA_LABEL_LEVELS = "l1,l2"
DEFAULT_MAX_PCA_CATEGORIES = 24
DEFAULT_WORKERS = 0
DEFAULT_BROWSER_UMAP_NEIGHBORS = 150
DEFAULT_BROWSER_UMAP_MIN_DIST = 0.7
DEFAULT_BROWSER_UMAP_RANDOM_STATE = 0
DEFAULT_ELBOW_MARGINAL_RATIO = 0.35
DEFAULT_FIXED_PROBE_RESAMPLES = 16
DEFAULT_PROBE_SLIDE_FRACTION = 0.20
DEFAULT_MIN_PROBE_SLIDES = 2
DEFAULT_FIXED_PROBE_SUPPORT = 0.80
DEFAULT_CONFIRMATION_INCREMENTS = 2
DEFAULT_CLASS_REFERENCE_COUNTS = (
    5,
    10,
    15,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    120,
    140,
    160,
    180,
    200,
    250,
    300,
    400,
    500,
    600,
    800,
)


def _log(message: str, *, enabled: bool = True) -> None:
    if enabled:
        print(f"[infospace] {message}", flush=True)


def _canonical_teacher_name(name: str) -> str:
    value = str(name).strip()
    return CANONICAL_TEACHER_BY_ALIAS.get(value, value)


def _teacher_aliases(name: str) -> tuple[str, ...]:
    canonical = _canonical_teacher_name(name)
    return TEACHER_ALIASES.get(canonical, (canonical,))


def _parse_int_list(value: str) -> list[int]:
    counts = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not counts:
        raise ValueError("expected at least one prototype_sample count")
    if any(count <= 0 for count in counts):
        raise ValueError(f"prototype_sample counts must be positive: {counts}")
    return sorted(dict.fromkeys(counts))


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _split_labels(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in str(value).replace("|", ";").split(";") if item.strip())


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _normalize_prototype_sample(row: dict[str, Any]) -> PrototypeSample:
    tile_id = str(row.get("tile_id", "")).strip()
    if not tile_id:
        raise ValueError("prototype_sample row missing tile_id")

    level1 = str(row.get("level1_label") or row.get("l1") or "").strip()
    if not level1:
        raise ValueError(f"prototype_sample row missing level1 label: tile_id={tile_id}")

    slide_id = str(row.get("slide_id") or row.get("slide") or tile_id).strip()
    return PrototypeSample(
        tile_id=tile_id,
        level1_label=level1,
        level2_labels=_split_labels(str(row.get("level2_labels") or row.get("l2") or "")),
        slide_id=slide_id,
        patient_id=str(row.get("patient_id") or row.get("patient") or slide_id).strip(),
        center=str(row.get("center") or row.get("dataset") or "").strip(),
    )


def _prototype_samples_from_annotation_json(path: Path) -> list[PrototypeSample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    annotations = payload.get("annotations")
    if not isinstance(annotations, dict):
        raise ValueError(f"annotation JSON missing annotations object: {path}")

    prototype_samples: list[PrototypeSample] = []
    for item in annotations.values():
        tile_id = str(item.get("tile_id", "")).strip()
        level1 = str(item.get("l1") or item.get("level1_label") or "").strip()
        if not tile_id or not level1:
            continue
        slide_id = str(item.get("slide") or item.get("slide_id") or tile_id).strip()
        dataset = str(item.get("dataset") or item.get("center") or "").strip()
        l2_raw = item.get("l2", item.get("level2_labels", []))
        if isinstance(l2_raw, str):
            l2 = _split_labels(l2_raw)
        else:
            l2 = tuple(str(label).strip() for label in l2_raw if str(label).strip())
        prototype_samples.append(
            PrototypeSample(
                tile_id=tile_id,
                level1_label=level1,
                level2_labels=l2,
                slide_id=slide_id,
                patient_id=str(item.get("patient_id") or slide_id).strip(),
                center=dataset,
            )
        )

    if not prototype_samples:
        raise ValueError(f"annotation JSON has no usable prototype_samples: {path}")
    return prototype_samples


def _load_prototype_samples_from_manifest(path: Path) -> list[PrototypeSample]:
    prototype_samples = [_normalize_prototype_sample(row) for row in _read_csv(path)]
    if not prototype_samples:
        raise ValueError(f"prototype_sample manifest has no usable prototype_samples: {path}")
    return prototype_samples


def _write_prototype_sample_manifest(path: Path, prototype_samples: list[PrototypeSample]) -> None:
    _write_csv(
        path,
        [
            {
                "tile_id": prototype_sample.tile_id,
                "level1_label": prototype_sample.level1_label,
                "level2_labels": ";".join(prototype_sample.level2_labels),
                "slide_id": prototype_sample.slide_id,
                "patient_id": prototype_sample.patient_id,
                "center": prototype_sample.center,
                "source": "prototype_sample_pool",
            }
            for prototype_sample in prototype_samples
        ],
    )


def _load_contract(path: Path | None, prototype_samples: list[PrototypeSample]) -> tuple[list[str], list[str]]:
    if path is None:
        level1: list[str] = []
        level2: list[str] = []
        for prototype_sample in prototype_samples:
            if prototype_sample.level1_label not in level1:
                level1.append(prototype_sample.level1_label)
            for label in prototype_sample.level2_labels:
                if label not in level2:
                    level2.append(label)
        return level1, level2

    payload = json.loads(path.read_text(encoding="utf-8"))
    level1 = [str(name) for name in payload.get("l1_prototypes", payload.get("level1", []))]
    level2 = [str(name) for name in payload.get("l2_prototypes", payload.get("level2", []))]
    if not level1:
        raise ValueError(f"prototype contract missing l1_prototypes/level1: {path}")
    return level1, level2


def _nested_subsets(
    prototype_samples: list[PrototypeSample],
    counts: list[int],
    seed: int,
    group_key: str,
) -> tuple[dict[int, list[PrototypeSample]], list[int]]:
    available_counts = [count for count in counts if count <= len(prototype_samples)]
    skipped_counts = [count for count in counts if count > len(prototype_samples)]
    if not available_counts:
        raise ValueError(f"all requested counts exceed available prototype_samples={len(prototype_samples)}")

    rng = random.Random(seed)
    groups: dict[str, list[PrototypeSample]] = {}
    for prototype_sample in prototype_samples:
        if group_key in {"tile_id", "slide_id", "patient_id", "center"}:
            key = str(getattr(prototype_sample, group_key) or prototype_sample.tile_id)
        else:
            key = prototype_sample.tile_id
        groups.setdefault(key, []).append(prototype_sample)

    group_keys = sorted(groups)
    rng.shuffle(group_keys)

    ordered: list[PrototypeSample] = []
    for key in group_keys:
        items = list(groups[key])
        rng.shuffle(items)
        ordered.extend(items)

    return {count: ordered[:count] for count in available_counts}, skipped_counts


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    denom = np.linalg.norm(matrix, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-8)
    return matrix / denom


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-8)
    return float(np.dot(a, b) / denom)


def _mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else math.nan


def _std(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) <= 1:
        return 0.0 if finite else math.nan
    mean = _mean(finite)
    return float((sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)) ** 0.5)


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, iterations: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    if iterations <= 0 or values.size == 1:
        value = float(values.mean())
        return value, value
    draws = rng.integers(0, values.size, size=(iterations, values.size))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _format_float(value: float) -> float | str:
    return "" if not math.isfinite(float(value)) else float(value)


def _as_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return math.nan
    return float(value)


def _quantiles_with_prefix(values: np.ndarray, prefix: str) -> dict[str, float | str]:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    keys = [f"{prefix}_q05", f"{prefix}_q25", f"{prefix}_q50", f"{prefix}_q75", f"{prefix}_q95"]
    if values.size == 0:
        return {key: "" for key in keys}
    return {
        f"{prefix}_q05": float(np.quantile(values, 0.05)),
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_q50": float(np.quantile(values, 0.50)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
        f"{prefix}_q95": float(np.quantile(values, 0.95)),
    }


def _topk_similarity_values(
    query_features: list[np.ndarray],
    reference_features: list[np.ndarray],
    *,
    k: int,
) -> np.ndarray:
    if not query_features or not reference_features:
        return np.asarray([], dtype=np.float32)
    query = _normalize_rows(np.stack(query_features))
    reference = _normalize_rows(np.stack(reference_features))
    sims = query @ reference.T
    k = max(1, min(int(k), sims.shape[1]))
    topk = np.partition(sims, -k, axis=1)[:, -k:]
    return topk.mean(axis=1).astype(np.float32)


def _novelty_values(
    new_features: list[np.ndarray],
    previous_features: list[np.ndarray],
    *,
    k: int,
) -> np.ndarray:
    if not new_features or not previous_features:
        return np.asarray([], dtype=np.float32)
    similarity = _topk_similarity_values(new_features, previous_features, k=k)
    return (1.0 - similarity).astype(np.float32)


def _redundancy_values(
    new_features: list[np.ndarray],
    previous_features: list[np.ndarray],
    *,
    k: int,
) -> np.ndarray:
    return _topk_similarity_values(new_features, previous_features, k=k)


class FeatureStore:
    def __init__(self, teacher_paths: dict[str, list[Path]]) -> None:
        from iatro.iac.adapters.features import FeatureCacheReader

        self._reader_cls = FeatureCacheReader
        self._paths = teacher_paths
        self._readers: dict[tuple[str, Path], Any] = {}
        self._paths_by_key: dict[str, dict[str, list[Path]]] = {
            teacher: self._index_paths(paths) for teacher, paths in teacher_paths.items()
        }

    @staticmethod
    def _package_keys(path: Path) -> list[str]:
        name = path.name
        suffixes = [
            ".prov-gigapath-local.features.iac",
            ".h_optimus_1.features.iac",
            ".h1.features.iac",
            ".uni2_h.features.iac",
            ".uni2.features.iac",
            ".virchow2.features.iac",
            ".features.iac",
        ]
        keys = []
        for suffix in suffixes:
            if name.endswith(suffix):
                keys.append(name[: -len(suffix)])
        keys.append(path.stem)
        return list(dict.fromkeys(key for key in keys if key))

    @classmethod
    def _index_paths(cls, paths: list[Path]) -> dict[str, list[Path]]:
        result: dict[str, list[Path]] = {}
        for path in paths:
            for key in cls._package_keys(path):
                result.setdefault(key, []).append(path)
        return result

    @staticmethod
    def _tile_keys(tile_id: str) -> list[str]:
        keys = [tile_id]
        if "_" in tile_id:
            keys.append(tile_id.rsplit("_", 1)[0])
        return list(dict.fromkeys(keys))

    def _reader(self, teacher: str, path: Path) -> Any:
        key = (teacher, path)
        if key not in self._readers:
            self._readers[key] = self._reader_cls(path)
        return self._readers[key]

    def read(self, teacher: str, tile_id: str) -> np.ndarray:
        last_error: FileNotFoundError | None = None
        candidate_paths: list[Path] = []
        for key in self._tile_keys(tile_id):
            candidate_paths.extend(self._paths_by_key[teacher].get(key, []))
        candidate_paths = list(dict.fromkeys([*candidate_paths, *self._paths[teacher]]))
        for path in candidate_paths:
            reader = self._reader(teacher, path)
            try:
                return reader.read_feature(tile_id)
            except FileNotFoundError as exc:
                last_error = exc
        raise FileNotFoundError(f"missing teacher feature: teacher={teacher} tile_id={tile_id}") from last_error

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()


def _teacher_paths_from_arg(value: str) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    if not value:
        return result
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"teacher feature package entry must be teacher=path: {item}")
        teacher, paths = item.split("=", 1)
        parsed = [Path(path) for path in paths.split("|") if path]
        if not parsed:
            raise ValueError(f"empty feature package paths for teacher={teacher}")
        result[_canonical_teacher_name(teacher)] = parsed
    return result


def _discover_teacher_paths(root: Path | None, teachers: list[str]) -> dict[str, list[Path]]:
    if root is None:
        return {}
    if root.is_file():
        if len(teachers) != 1:
            raise ValueError("--teacher-feature-root may be a file only when one teacher is requested")
        return {teachers[0]: [root]}

    result: dict[str, list[Path]] = {}
    for teacher in teachers:
        aliases = _teacher_aliases(teacher)
        direct_roots = [root / alias for alias in aliases if (root / alias).exists()]
        search_roots = direct_roots if direct_roots else [root]

        matches: list[Path] = []
        for search_root in search_roots:
            matches.extend(sorted(search_root.rglob("*.features.iac")))
            matches.extend(path for path in sorted(search_root.rglob("*features*.iac")) if path not in matches)

        teacher_matches = [
            path
            for path in matches
            if search_roots != [root] or any(alias in path.name or alias in path.parts for alias in aliases)
        ]
        if not teacher_matches:
            raise FileNotFoundError(
                f"no feature packages found for teacher={teacher} aliases={list(aliases)} under {root}"
            )
        result[teacher] = teacher_matches
    return result


def _discover_teachers_from_root(root: Path | None) -> list[str]:
    if root is None or not root.exists() or root.is_file():
        return []

    discovered: list[str] = []
    for canonical, aliases in TEACHER_ALIASES.items():
        if any((root / alias).exists() for alias in aliases):
            discovered.append(canonical)
            continue
        for alias in aliases:
            if any(root.rglob(f"*{alias}*.features.iac")):
                discovered.append(canonical)
                break
    return discovered


def _resolve_teacher_paths(args: argparse.Namespace) -> dict[str, list[Path]]:
    explicit = _teacher_paths_from_arg(args.teacher_feature_packages)
    root = Path(args.teacher_feature_root) if args.teacher_feature_root else None
    requested = _parse_str_list(getattr(args, "teachers", ""))

    if requested:
        teachers = list(dict.fromkeys(_canonical_teacher_name(teacher) for teacher in requested))
    elif explicit:
        teachers = list(explicit)
    else:
        teachers = _discover_teachers_from_root(root)

    if not teachers:
        raise ValueError("pass --teachers/--teacher-feature-packages, or provide recognizable packages under --teacher-feature-root")

    paths = _discover_teacher_paths(root, teachers)
    paths.update(explicit)

    missing = [teacher for teacher in teachers if teacher not in paths]
    if missing:
        raise ValueError(f"missing feature packages for teachers: {missing}")

    return {teacher: paths[teacher] for teacher in teachers}


def _feature_map(store: FeatureStore, teacher: str, tile_ids: list[str], *, verbose: bool) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    unique_ids = list(dict.fromkeys(tile_ids))
    for idx, tile_id in enumerate(unique_ids, start=1):
        result[tile_id] = store.read(teacher, tile_id)
        if verbose and (idx == 1 or idx == len(unique_ids) or idx % 250 == 0):
            _log(f"teacher={teacher}: read features {idx}/{len(unique_ids)}")
    return result


def _centers(
    prototype_samples: list[PrototypeSample],
    features: dict[str, np.ndarray],
    level1_names: list[str],
    level2_names: list[str],
) -> Centers:
    level1: dict[str, list[np.ndarray]] = {name: [] for name in level1_names}
    level2: dict[str, list[np.ndarray]] = {name: [] for name in level2_names}

    for prototype_sample in prototype_samples:
        feature = features[prototype_sample.tile_id]
        if prototype_sample.level1_label in level1:
            level1[prototype_sample.level1_label].append(feature)
        for label in prototype_sample.level2_labels:
            if label in level2:
                level2[label].append(feature)

    l1_centers = {name: np.stack(values).mean(axis=0).astype(np.float32) for name, values in level1.items() if values}
    l2_centers = {name: np.stack(values).mean(axis=0).astype(np.float32) for name, values in level2.items() if values}
    return Centers(
        level1=l1_centers,
        level2=l2_centers,
        level1_counts={name: len(values) for name, values in level1.items()},
        level2_counts={name: len(values) for name, values in level2.items()},
    )


def _center_for(centers: Centers | None, level: int, name: str) -> np.ndarray | None:
    if centers is None:
        return None
    return centers.level1.get(name) if level == 1 else centers.level2.get(name)


def _prototype_drift_values(current: Centers, previous: Centers | None) -> np.ndarray:
    if previous is None:
        return np.asarray([], dtype=np.float32)

    values: list[float] = []
    for name, center in current.level1.items():
        if name in previous.level1:
            values.append(1.0 - _cosine(center, previous.level1[name]))
    for name, center in current.level2.items():
        if name in previous.level2:
            values.append(1.0 - _cosine(center, previous.level2[name]))
    return np.asarray(values, dtype=np.float32)


def _prototype_sample_has_prototype(prototype_sample: PrototypeSample, level: int, name: str) -> bool:
    if level == 1:
        return prototype_sample.level1_label == name
    return name in prototype_sample.level2_labels


def _metric_summary(values: np.ndarray, prefix: str, rng: np.random.Generator, iterations: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float32)
    ci_low, ci_high = _bootstrap_ci(values, rng, iterations)
    result: dict[str, Any] = {
        prefix: _format_float(float(values.mean()) if values.size else math.nan),
        f"{prefix}_ci_low": _format_float(ci_low),
        f"{prefix}_ci_high": _format_float(ci_high),
    }
    result.update(_quantiles_with_prefix(values, prefix))
    return result


def _prototype_rows(
    *,
    teacher: str,
    prototype_sample_count: int,
    prototype_samples: list[PrototypeSample],
    previous_prototype_samples: list[PrototypeSample] | None,
    features: dict[str, np.ndarray],
    centers: Centers,
    previous_centers: Centers | None,
    level1_names: list[str],
    level2_names: list[str],
    topk: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_prototype_samples = previous_prototype_samples or []

    for level, names in [(1, level1_names), (2, level2_names)]:
        for name in names:
            selected = [prototype_sample for prototype_sample in prototype_samples if _prototype_sample_has_prototype(prototype_sample, level, name)]
            previous_selected = [prototype_sample for prototype_sample in previous_prototype_samples if _prototype_sample_has_prototype(prototype_sample, level, name)]
            new_selected = selected[len(previous_selected) :]

            new_features = [features[prototype_sample.tile_id] for prototype_sample in new_selected]
            previous_global_features = [features[prototype_sample.tile_id] for prototype_sample in previous_prototype_samples]
            previous_local_features = [features[prototype_sample.tile_id] for prototype_sample in previous_selected]

            global_novelty = _novelty_values(new_features, previous_global_features, k=topk)
            local_novelty = _novelty_values(new_features, previous_local_features, k=topk)
            global_redundancy = _redundancy_values(new_features, previous_global_features, k=topk)
            local_redundancy = _redundancy_values(new_features, previous_local_features, k=topk)

            current_center = _center_for(centers, level, name)
            previous_center = _center_for(previous_centers, level, name)
            drift = (
                1.0 - _cosine(current_center, previous_center)
                if current_center is not None and previous_center is not None
                else math.nan
            )

            row: dict[str, Any] = {
                "prototype_sample_count": prototype_sample_count,
                "teacher": teacher,
                "level": level,
                "prototype": name,
                "prototype_tile_count": len(selected),
                "new_prototype_tile_count": len(new_selected),
                "prototype_tile_fraction": _format_float(len(selected) / len(prototype_samples) if prototype_samples else math.nan),
                "center_available": str(current_center is not None).lower(),
                "prototype_drift": _format_float(drift),
                "global_infospace_novelty": _format_float(float(global_novelty.mean()) if global_novelty.size else math.nan),
                "local_infospace_novelty": _format_float(float(local_novelty.mean()) if local_novelty.size else math.nan),
                "global_infospace_redundancy": _format_float(float(global_redundancy.mean()) if global_redundancy.size else math.nan),
                "local_infospace_redundancy": _format_float(float(local_redundancy.mean()) if local_redundancy.size else math.nan),
            }
            row.update(_quantiles_with_prefix(global_novelty, "global_infospace_novelty"))
            row.update(_quantiles_with_prefix(local_novelty, "local_infospace_novelty"))
            rows.append(row)

    return rows


def _teacher_curve(
    *,
    teacher: str,
    subsets: dict[int, list[PrototypeSample]],
    features: dict[str, np.ndarray],
    level1_names: list[str],
    level2_names: list[str],
    bootstrap_iterations: int,
    seed: int,
    topk: int,
    verbose: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    prototype_rows: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    rng = np.random.default_rng(seed)

    for count in sorted(subsets):
        _log(f"teacher={teacher}: compute infospace metrics at N={count}", enabled=verbose)
        prototype_samples = subsets[count]
        centers = _centers(prototype_samples, features, level1_names, level2_names)

        if previous is None:
            new_prototype_samples = prototype_samples
            previous_prototype_samples: list[PrototypeSample] = []
            novelty = np.asarray([], dtype=np.float32)
            redundancy = np.asarray([], dtype=np.float32)
        else:
            previous_prototype_samples = previous["prototype_samples"]
            new_prototype_samples = prototype_samples[len(previous_prototype_samples) :]
            new_features = [features[prototype_sample.tile_id] for prototype_sample in new_prototype_samples]
            previous_features = [features[prototype_sample.tile_id] for prototype_sample in previous_prototype_samples]
            novelty = _novelty_values(new_features, previous_features, k=topk)
            redundancy = _redundancy_values(new_features, previous_features, k=topk)

        drift_values = _prototype_drift_values(centers, previous["centers"] if previous else None)
        missing_l1 = [name for name in level1_names if name not in centers.level1]
        missing_l2 = [name for name in level2_names if name not in centers.level2]

        row: dict[str, Any] = {
            "prototype_sample_count": count,
            "teacher": teacher,
            "new_tile_count": len(new_prototype_samples),
            "previous_tile_count": len(previous_prototype_samples),
            "infospace_topk": topk,
            "prototype_drift": _format_float(float(drift_values.mean()) if drift_values.size else math.nan),
            "available_l1_centers": len(centers.level1),
            "available_l2_centers": len(centers.level2),
            "missing_l1_centers": ";".join(missing_l1),
            "missing_l2_centers": ";".join(missing_l2),
        }
        row.update(_metric_summary(novelty, "infospace_novelty", rng, bootstrap_iterations))
        row.update(_metric_summary(redundancy, "infospace_redundancy", rng, bootstrap_iterations))
        rows.append(row)

        prototype_rows.extend(
            _prototype_rows(
                teacher=teacher,
                prototype_sample_count=count,
                prototype_samples=prototype_samples,
                previous_prototype_samples=previous_prototype_samples,
                features=features,
                centers=centers,
                previous_centers=previous["centers"] if previous else None,
                level1_names=level1_names,
                level2_names=level2_names,
                topk=topk,
            )
        )

        previous = {
            "prototype_samples": prototype_samples,
            "centers": centers,
        }

    return rows, prototype_rows


def _aggregate_rows(
    teacher_rows: list[dict[str, Any]],
    *,
    plateau_novelty_threshold: float,
    plateau_drift_threshold: float,
    plateau_redundancy_threshold: float,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    counts = sorted({int(row["prototype_sample_count"]) for row in teacher_rows})
    consecutive_plateau = 0

    quantile_suffixes = ["q05", "q25", "q50", "q75", "q95"]

    for count in counts:
        rows = [row for row in teacher_rows if int(row["prototype_sample_count"]) == count]
        novelty = _mean([_as_float(row, "infospace_novelty") for row in rows])
        novelty_ci_low = _mean([_as_float(row, "infospace_novelty_ci_low") for row in rows])
        novelty_ci_high = _mean([_as_float(row, "infospace_novelty_ci_high") for row in rows])
        redundancy = _mean([_as_float(row, "infospace_redundancy") for row in rows])
        redundancy_ci_low = _mean([_as_float(row, "infospace_redundancy_ci_low") for row in rows])
        redundancy_ci_high = _mean([_as_float(row, "infospace_redundancy_ci_high") for row in rows])
        drift = _mean([_as_float(row, "prototype_drift") for row in rows])

        low_novelty = math.isfinite(novelty) and novelty <= plateau_novelty_threshold
        high_redundancy = math.isfinite(redundancy) and redundancy >= plateau_redundancy_threshold
        low_drift = math.isfinite(drift) and drift <= plateau_drift_threshold
        interval_plateau = low_novelty and high_redundancy and low_drift
        consecutive_plateau = consecutive_plateau + 1 if interval_plateau else 0

        aggregate: dict[str, Any] = {
            "prototype_sample_count": count,
            "teacher_count": len(rows),
            "new_tile_count_mean": _format_float(_mean([_as_float(row, "new_tile_count") for row in rows])),
            "previous_tile_count_mean": _format_float(_mean([_as_float(row, "previous_tile_count") for row in rows])),
            "infospace_novelty_mean": _format_float(novelty),
            "infospace_novelty_std": _format_float(_std([_as_float(row, "infospace_novelty") for row in rows])),
            "infospace_novelty_ci_low_mean": _format_float(novelty_ci_low),
            "infospace_novelty_ci_high_mean": _format_float(novelty_ci_high),
            "infospace_redundancy_mean": _format_float(redundancy),
            "infospace_redundancy_std": _format_float(_std([_as_float(row, "infospace_redundancy") for row in rows])),
            "infospace_redundancy_ci_low_mean": _format_float(redundancy_ci_low),
            "infospace_redundancy_ci_high_mean": _format_float(redundancy_ci_high),
            "prototype_drift_mean": _format_float(drift),
            "low_novelty": str(low_novelty).lower(),
            "high_redundancy": str(high_redundancy).lower(),
            "low_drift": str(low_drift).lower(),
            "plateau_interval": str(interval_plateau).lower(),
            "plateau_consensus": str(consecutive_plateau >= 2).lower(),
        }
        for prefix in ["infospace_novelty", "infospace_redundancy"]:
            for suffix in quantile_suffixes:
                key = f"{prefix}_{suffix}"
                aggregate[f"{key}_mean"] = _format_float(_mean([_as_float(row, key) for row in rows]))
        result.append(aggregate)

    _annotate_marginal_utility(result)
    return result


def _annotate_marginal_utility(aggregate_rows: list[dict[str, Any]]) -> None:
    positive_drop_per_100: list[float] = []
    previous: dict[str, Any] | None = None

    for row in aggregate_rows:
        count = int(row["prototype_sample_count"])
        novelty = _as_float(row, "infospace_novelty_mean")
        drift = _as_float(row, "prototype_drift_mean")

        if previous is None:
            row["marginal_prototype_sample_count_delta"] = ""
            row["marginal_novelty_drop"] = ""
            row["marginal_novelty_drop_per_100_tiles"] = ""
            row["marginal_drift_drop"] = ""
            previous = row
            continue

        previous_count = int(previous["prototype_sample_count"])
        previous_novelty = _as_float(previous, "infospace_novelty_mean")
        previous_drift = _as_float(previous, "prototype_drift_mean")
        delta = count - previous_count
        novelty_drop = previous_novelty - novelty if math.isfinite(previous_novelty) and math.isfinite(novelty) else math.nan
        drop_per_100 = novelty_drop / delta * 100.0 if delta > 0 and math.isfinite(novelty_drop) else math.nan
        drift_drop = previous_drift - drift if math.isfinite(previous_drift) and math.isfinite(drift) else math.nan

        row["marginal_prototype_sample_count_delta"] = delta
        row["marginal_novelty_drop"] = _format_float(novelty_drop)
        row["marginal_novelty_drop_per_100_tiles"] = _format_float(drop_per_100)
        row["marginal_drift_drop"] = _format_float(drift_drop)
        if math.isfinite(drop_per_100) and drop_per_100 > 0:
            positive_drop_per_100.append(drop_per_100)
        previous = row

    best_drop = max(positive_drop_per_100) if positive_drop_per_100 else math.nan
    for row in aggregate_rows:
        drop_per_100 = _as_float(row, "marginal_novelty_drop_per_100_tiles")
        ratio = drop_per_100 / best_drop if math.isfinite(drop_per_100) and math.isfinite(best_drop) and best_drop > 0 else math.nan
        row["marginal_utility_ratio"] = _format_float(ratio)
        row["elbow_candidate"] = str(
            math.isfinite(ratio)
            and ratio <= DEFAULT_ELBOW_MARGINAL_RATIO
            and str(row.get("low_drift", "")).lower() == "true"
        ).lower()


def _recommendation(aggregate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    elbow_rows = [row for row in aggregate_rows if str(row.get("elbow_candidate", "")).lower() == "true"]
    if elbow_rows:
        onset = elbow_rows[0]
        onset_count = int(onset["prototype_sample_count"])
        counts = [int(row["prototype_sample_count"]) for row in aggregate_rows]
        onset_index = counts.index(onset_count)
        confirmation_count = counts[min(onset_index + 1, len(counts) - 1)]
        conservative_index = min(onset_index + 2, len(counts) - 1)
        conservative_count = counts[conservative_index]
        return {
            "recommended_prototype_sample_count": confirmation_count,
            "elbow_onset_count": onset_count,
            "conservative_prototype_sample_count": conservative_count,
            "reason": (
                "marginal utility elbow: novelty drop per 100 prototype_samples falls below "
                f"{DEFAULT_ELBOW_MARGINAL_RATIO:.0%} of the best observed marginal gain while prototype drift is low"
            ),
            "marginal_utility_ratio_threshold": DEFAULT_ELBOW_MARGINAL_RATIO,
            "novelty_drop_per_100_at_elbow": _format_float(_as_float(onset, "marginal_novelty_drop_per_100_tiles")),
            "prototype_drift_at_elbow": _format_float(_as_float(onset, "prototype_drift_mean")),
        }

    plateau_counts = [int(row["prototype_sample_count"]) for row in aggregate_rows if row["plateau_consensus"] == "true"]
    available_counts = [int(row["prototype_sample_count"]) for row in aggregate_rows]
    if plateau_counts:
        return {
            "recommended_prototype_sample_count": plateau_counts[0],
            "reason": "strict novelty plateau; this is stronger than the marginal utility elbow criterion",
        }
    return {
        "recommended_prototype_sample_count": available_counts[-1] if available_counts else None,
        "reason": "no marginal utility elbow detected within available prototype_samples",
    }


def _fixed_probe_seed(unit: str, seed: int, resample: int) -> int:
    digest = hashlib.sha256(unit.encode("utf-8")).digest()
    return (
        seed
        + int.from_bytes(digest[:4], "little")
        + resample * 100_003
    )


def _reference_checkpoints(
    capacity: int,
    requested: list[int],
) -> list[int]:
    return meaningful_reference_checkpoints(capacity, requested)


def _json_curve_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            key: _format_float(value)
            if isinstance(value, float)
            else value
            for key, value in row.items()
        }
        for row in rows
    ]


def _fixed_probe_unit(
    *,
    unit: str,
    prototype_samples: list[PrototypeSample],
    features_by_teacher: dict[str, dict[str, np.ndarray]],
    requested_counts: list[int],
    seed: int,
    resamples: int,
    topk: int,
    probe_slide_fraction: float,
    min_probe_slides: int,
    marginal_ratio_threshold: float,
    drift_threshold: float,
    support_threshold: float,
    confirmation_increments: int,
    min_slides: int,
    min_increments: int,
) -> dict[str, Any]:
    selected_samples = [
        sample
        for sample in prototype_samples
        if unit == "__all_l1__" or sample.level1_label == unit
    ]
    slides = {sample.slide_id for sample in selected_samples}
    if not selected_samples or len(slides) < min_probe_slides + 2:
        return {
            "status": "not_assessable",
            "enough_now": False,
            "reason": (
                "fixed-probe split requires at least "
                f"{min_probe_slides + 2} slides; observed {len(slides)}"
            ),
            "positive_tile_count": len(selected_samples),
            "positive_slide_count": len(slides),
            "reference_capacity": 0,
            "checkpoints": [],
            "tail_plateau_support": 0.0,
            "recommended_reference_tile_count": None,
            "curve": [],
        }
    curves: list[list[dict[str, object]]] = []
    splits_by_teacher: dict[str, list[Any]] = {}
    for teacher, feature_map in features_by_teacher.items():
        observations = [
            CurveObservation(
                tile_id=sample.tile_id,
                slide_id=sample.slide_id,
                stratum=(
                    sample.level1_label
                    if unit == "__all_l1__"
                    else unit
                ),
                feature=feature_map[sample.tile_id],
            )
            for sample in selected_samples
        ]
        splits_by_teacher[teacher] = [
            prepare_fixed_probe_split(
                observations,
                seed=_fixed_probe_seed(unit, seed, resample),
                probe_slide_fraction=probe_slide_fraction,
                min_probe_slides=min_probe_slides,
            )
            for resample in range(resamples)
        ]

    all_splits = [
        split
        for teacher_splits in splits_by_teacher.values()
        for split in teacher_splits
    ]
    capacity = min(
        len(split.reference_tile_order) for split in all_splits
    )
    checkpoints = _reference_checkpoints(capacity, requested_counts)
    for teacher_splits in splits_by_teacher.values():
        curves.extend(
            [
                list(
                    fixed_probe_curve(
                        split,
                        checkpoints,
                        topk=topk,
                    )
                )
                for split in teacher_splits
            ]
        )
    aggregate = list(aggregate_fixed_probe_curves(curves))
    decisions = [
        tail_plateau(
            curve,
            marginal_ratio_threshold=marginal_ratio_threshold,
            drift_threshold=drift_threshold,
            confirmation_increments=confirmation_increments,
        )
        for curve in curves
    ]
    observed = [
        onset
        for stable, onset in decisions
        if stable and onset is not None
    ]
    support = len(observed) / len(curves)
    enough_curve = len(checkpoints) - 1 >= min_increments
    enough_slides = len(slides) >= min_slides
    ready = (
        enough_curve
        and enough_slides
        and support >= support_threshold
    )
    if not enough_curve or not enough_slides:
        reason = (
            f"requires {min_increments} reference increments and "
            f"{min_slides} slides; observed {len(checkpoints) - 1} "
            f"increments and {len(slides)} slides"
        )
        status = "not_assessable"
    elif ready:
        reason = (
            "fixed-probe remaining novelty is monotone and its final "
            "gains and centre drift pass repeated confirmation"
        )
        status = "provisionally_stable"
    else:
        reason = (
            "the fixed-probe tail has not passed consecutive "
            "low-gain confirmation"
        )
        status = "still_growing"
    return {
        "status": status,
        "enough_now": ready,
        "reason": reason,
        "positive_tile_count": len(selected_samples),
        "positive_slide_count": len(slides),
        "reference_capacity": capacity,
        "checkpoints": checkpoints,
        "tail_plateau_support": support,
        "recommended_reference_tile_count": (
            int(round(float(np.median(observed))))
            if observed
            else None
        ),
        "curve": _json_curve_rows(aggregate),
    }


def _fixed_probe_l1_information(
    *,
    prototype_samples: list[PrototypeSample],
    features_by_teacher: dict[str, dict[str, np.ndarray]],
    level1_names: list[str],
    requested_counts: list[int],
    seed: int,
    resamples: int,
    topk: int,
    probe_slide_fraction: float,
    min_probe_slides: int,
    marginal_ratio_threshold: float,
    drift_threshold: float,
    support_threshold: float,
    confirmation_increments: int,
    min_slides: int,
    min_increments: int,
) -> dict[str, Any]:
    global_result = _fixed_probe_unit(
        unit="__all_l1__",
        prototype_samples=prototype_samples,
        features_by_teacher=features_by_teacher,
        requested_counts=requested_counts,
        seed=seed,
        resamples=resamples,
        topk=topk,
        probe_slide_fraction=probe_slide_fraction,
        min_probe_slides=min_probe_slides,
        marginal_ratio_threshold=marginal_ratio_threshold,
        drift_threshold=drift_threshold,
        support_threshold=support_threshold,
        confirmation_increments=confirmation_increments,
        min_slides=min_slides,
        min_increments=min_increments,
    )
    classes = {
        name: _fixed_probe_unit(
            unit=name,
            prototype_samples=prototype_samples,
            features_by_teacher=features_by_teacher,
            requested_counts=list(DEFAULT_CLASS_REFERENCE_COUNTS),
            seed=seed,
            resamples=resamples,
            topk=topk,
            probe_slide_fraction=probe_slide_fraction,
            min_probe_slides=min_probe_slides,
            marginal_ratio_threshold=marginal_ratio_threshold,
            drift_threshold=drift_threshold,
            support_threshold=support_threshold,
            confirmation_increments=confirmation_increments,
            min_slides=min_slides,
            min_increments=min_increments,
        )
        for name in level1_names
    }
    ready = bool(classes) and global_result["enough_now"] and all(
        result["enough_now"] for result in classes.values()
    )
    return {
        "status": (
            "provisionally_stable" if ready else "still_growing"
        ),
        "enough_now": ready,
        "global": global_result,
        "classes": classes,
        "method": {
            "primary_curve": (
                "fixed slide-separated probe remaining novelty under a "
                "nested reference set; monotone non-increasing by construction"
            ),
            "teacher_aggregation": (
                "equal teacher contribution after per-teacher fixed-probe curves"
            ),
            "class_aggregation": (
                "global classification probe coverage weights classes equally"
            ),
            "resamples": resamples,
            "probe_slide_fraction": probe_slide_fraction,
            "minimum_probe_slides": min_probe_slides,
            "topk": topk,
            "tail_plateau_support_threshold": support_threshold,
            "confirmation_increments": confirmation_increments,
        },
    }


def _pca_2d(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if matrix.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32)
    matrix = _normalize_rows(matrix)
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(matrix, full_matrices=False)
    components = vt[: min(2, vt.shape[0])]
    coords = matrix @ components.T
    if coords.shape[1] == 1:
        coords = np.concatenate([coords, np.zeros((coords.shape[0], 1), dtype=coords.dtype)], axis=1)
    return coords.astype(np.float32)


def _standardize_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.size == 0:
        return coords
    coords = coords - coords.mean(axis=0, keepdims=True)
    scale = coords.std(axis=0, keepdims=True)
    scale = np.maximum(scale, 1e-8)
    return coords / scale


def _align_coords_to_reference(coords: np.ndarray, reference: np.ndarray) -> np.ndarray:
    coords = _standardize_coords(coords)
    reference = _standardize_coords(reference)
    if coords.shape[0] < 2 or reference.shape[0] < 2:
        return coords
    matrix = coords.T @ reference
    try:
        u, _, vt = np.linalg.svd(matrix, full_matrices=False)
        rotation = u @ vt
        return _standardize_coords(coords @ rotation)
    except np.linalg.LinAlgError:
        return coords


def _browser_matched_fused_features(
    pca_features_by_teacher: dict[str, dict[str, np.ndarray]],
    teacher_names: list[str],
    common_tile_ids: list[str],
) -> np.ndarray:
    matrices = []
    for teacher in teacher_names:
        matrix = np.stack([pca_features_by_teacher[teacher][tile_id] for tile_id in common_tile_ids])
        matrices.append(_normalize_rows(matrix))

    dims = {matrix.shape[1] for matrix in matrices}
    if len(dims) == 1:
        return _normalize_rows(np.stack(matrices, axis=0).mean(axis=0))
    return _normalize_rows(np.concatenate(matrices, axis=1))


def _browser_matched_umap_2d(features: np.ndarray) -> np.ndarray:
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(os.getenv("TMPDIR", "/tmp")) / "numba_cache"))
    try:
        import umap  # type: ignore
    except Exception as exc:
        raise RuntimeError("umap-learn is required for browser-matched classification UMAP QC") from exc

    x = _normalize_rows(features)
    n = x.shape[0]
    if n < 3:
        return _standardize_coords(_pca_2d(x))

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=max(2, min(DEFAULT_BROWSER_UMAP_NEIGHBORS, n - 1)),
        min_dist=DEFAULT_BROWSER_UMAP_MIN_DIST,
        metric="cosine",
        random_state=DEFAULT_BROWSER_UMAP_RANDOM_STATE,
        n_jobs=1,
        low_memory=False,
    )
    return _standardize_coords(reducer.fit_transform(x))


def _l1_distribution_ellipses(coords: np.ndarray, labels: list[str]) -> dict[str, dict[str, float]]:
    # 95% covariance ellipse in 2D: chi-square(df=2, p=0.95) = 5.991464547.
    chi2_95_2d = 5.991464547107979
    result: dict[str, dict[str, float]] = {}
    coords = np.asarray(coords, dtype=np.float32)

    for label in sorted(set(labels)):
        idx = [i for i, value in enumerate(labels) if value == label]
        if len(idx) < 3:
            continue
        points = coords[idx]
        center = points.mean(axis=0)
        cov = np.cov(points.T)
        cov = np.asarray(cov, dtype=np.float64)
        if cov.shape != (2, 2) or not np.isfinite(cov).all():
            continue
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-10)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        result[label] = {
            "cx": float(center[0]),
            "cy": float(center[1]),
            "rx": float(math.sqrt(float(eigvals[0]) * chi2_95_2d)),
            "ry": float(math.sqrt(float(eigvals[1]) * chi2_95_2d)),
            "angle": float(math.degrees(math.atan2(float(eigvecs[1, 0]), float(eigvecs[0, 0])))),
            "n": int(len(idx)),
        }
    return result


def _prototype_sample_labels(prototype_sample: PrototypeSample, level: str) -> tuple[str, ...]:
    if level == "l1":
        return (prototype_sample.level1_label,)
    labels = tuple(label for label in prototype_sample.level2_labels if label)
    return labels if labels else ("none",)


def _compress_multilabels(labels_by_prototype_sample: list[tuple[str, ...]], max_categories: int) -> list[tuple[str, ...]]:
    if max_categories <= 0:
        return labels_by_prototype_sample

    counts: dict[str, int] = {}
    for labels in labels_by_prototype_sample:
        for label in labels:
            counts[label] = counts.get(label, 0) + 1

    keep = {
        label
        for label, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:max_categories]
    }

    compressed: list[tuple[str, ...]] = []
    for labels in labels_by_prototype_sample:
        mapped: list[str] = []
        for label in labels:
            mapped.append(label if label in keep else "OTHER")
        deduped = tuple(dict.fromkeys(mapped))
        compressed.append(deduped if deduped else ("none",))
    return compressed


def _scatter_by_multilabel(
    ax: Any,
    coords: np.ndarray,
    labels_by_prototype_sample: list[tuple[str, ...]],
    *,
    point_size: float = 12.0,
    alpha: float = 0.58,
) -> None:
    unique_labels = sorted({label for labels in labels_by_prototype_sample for label in labels})
    for label in unique_labels:
        idx = [i for i, labels in enumerate(labels_by_prototype_sample) if label in labels]
        if not idx:
            continue
        ax.scatter(coords[idx, 0], coords[idx, 1], s=point_size, alpha=alpha, label=label)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, linewidth=0.4, alpha=0.25)


def _plot_pca_qc(
    *,
    plt: Any,
    figure_dir: Path,
    prototype_samples: list[PrototypeSample],
    pca_features_by_teacher: dict[str, dict[str, np.ndarray]],
    label_levels: list[str],
    max_categories: int,
    formats: list[str],
    written: list[str],
    verbose: bool,
) -> None:
    if not prototype_samples or not pca_features_by_teacher:
        return

    teacher_names = sorted(pca_features_by_teacher)
    common_tile_ids = [
        prototype_sample.tile_id
        for prototype_sample in prototype_samples
        if all(prototype_sample.tile_id in pca_features_by_teacher[teacher] for teacher in teacher_names)
    ]
    if len(common_tile_ids) < 3:
        _log("PCA QC skipped: fewer than 3 common tiles across teachers", enabled=verbose)
        return

    prototype_samples_by_id = {prototype_sample.tile_id: prototype_sample for prototype_sample in prototype_samples}
    common_prototype_samples = [prototype_samples_by_id[tile_id] for tile_id in common_tile_ids]

    teacher_coords: dict[str, np.ndarray] = {}
    for teacher in teacher_names:
        matrix = np.stack([pca_features_by_teacher[teacher][tile_id] for tile_id in common_tile_ids])
        teacher_coords[teacher] = _standardize_coords(_pca_2d(matrix))

    reference_teacher = teacher_names[0]
    reference_coords = teacher_coords[reference_teacher]
    aligned = {
        teacher: (
            teacher_coords[teacher]
            if teacher == reference_teacher
            else _align_coords_to_reference(teacher_coords[teacher], reference_coords)
        )
        for teacher in teacher_names
    }
    average_coords = _standardize_coords(np.mean(np.stack([aligned[teacher] for teacher in teacher_names]), axis=0))

    ncols = min(2, len(teacher_names))
    nrows = int(math.ceil(len(teacher_names) / ncols))

    for level in label_levels:
        if level not in {"l1", "l2"}:
            continue
        raw_labels = [_prototype_sample_labels(prototype_sample, level) for prototype_sample in common_prototype_samples]
        labels_by_prototype_sample = _compress_multilabels(raw_labels, max_categories)
        level_name = "classification" if level == "l1" else "spatial"
        label_note = "single-label" if level == "l1" else "multi-label expanded; one tile may appear under multiple labels"

        _log(f"plot PCA QC: teacher-averaged {level_name}, tiles={len(common_prototype_samples)}", enabled=verbose)
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        _scatter_by_multilabel(ax, average_coords, labels_by_prototype_sample)
        ax.set_title(f"{level_name} QC PCA at max N={len(prototype_samples)}: teacher-averaged ({label_note})")
        ax.set_xlabel("Averaged PCA-1")
        ax.set_ylabel("Averaged PCA-2")
        ax.legend(fontsize=7, markerscale=1.5, ncols=2)
        for fmt in formats:
            path = figure_dir / f"infospace_pca_qc_{level}_teacher_average.{fmt}"
            fig.savefig(path, dpi=220)
            written.append(str(path))
        plt.close(fig)

        _log(f"plot PCA QC: per-teacher subplots {level_name}", enabled=verbose)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), constrained_layout=True)
        axes_arr = np.asarray(axes).reshape(-1)
        for ax, teacher in zip(axes_arr, teacher_names):
            _scatter_by_multilabel(ax, teacher_coords[teacher], labels_by_prototype_sample, point_size=10.0, alpha=0.52)
            ax.set_title(f"{teacher}")
            ax.set_xlabel("PCA-1")
            ax.set_ylabel("PCA-2")
        for ax in axes_arr[len(teacher_names) :]:
            ax.axis("off")
        handles, legend_labels = axes_arr[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, legend_labels, loc="outside lower center", ncols=min(4, max(1, len(legend_labels))), fontsize=7)
        fig.suptitle(f"{level_name} QC PCA at max N={len(prototype_samples)}: individual teachers ({label_note})", y=1.02)
        for fmt in formats:
            path = figure_dir / f"infospace_pca_qc_{level}_by_teacher.{fmt}"
            fig.savefig(path, dpi=220)
            written.append(str(path))
        plt.close(fig)


def _plot_browser_matched_l1_umap_qc(
    *,
    plt: Any,
    figure_dir: Path,
    prototype_samples: list[PrototypeSample],
    pca_features_by_teacher: dict[str, dict[str, np.ndarray]],
    formats: list[str],
    written: list[str],
    verbose: bool,
) -> None:
    if not prototype_samples or not pca_features_by_teacher:
        return

    teacher_names = sorted(pca_features_by_teacher)
    common_tile_ids = [
        prototype_sample.tile_id
        for prototype_sample in prototype_samples
        if all(prototype_sample.tile_id in pca_features_by_teacher[teacher] for teacher in teacher_names)
    ]
    if len(common_tile_ids) < 3:
        _log("browser-matched classification UMAP QC skipped: fewer than 3 common tiles across teachers", enabled=verbose)
        return

    prototype_samples_by_id = {prototype_sample.tile_id: prototype_sample for prototype_sample in prototype_samples}
    common_prototype_samples = [prototype_samples_by_id[tile_id] for tile_id in common_tile_ids]
    labels = [prototype_sample.level1_label for prototype_sample in common_prototype_samples]
    label_names = sorted(set(labels))

    _log(
        "plot browser-matched classification UMAP QC: "
        f"tiles={len(common_prototype_samples)} n_neighbors={DEFAULT_BROWSER_UMAP_NEIGHBORS} "
        f"min_dist={DEFAULT_BROWSER_UMAP_MIN_DIST} random_state={DEFAULT_BROWSER_UMAP_RANDOM_STATE}",
        enabled=verbose,
    )
    try:
        fused = _browser_matched_fused_features(pca_features_by_teacher, teacher_names, common_tile_ids)
        coords = _browser_matched_umap_2d(fused)
    except RuntimeError as exc:
        _log(f"browser-matched classification UMAP QC skipped: {exc}", enabled=verbose)
        return

    palette = dict(zip(label_names, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    ellipses = _l1_distribution_ellipses(coords, labels)

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    for label in label_names:
        idx = [i for i, value in enumerate(labels) if value == label]
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=12.0,
            alpha=0.58,
            label=label,
            color=palette.get(label),
            linewidths=0,
        )

    try:
        from matplotlib.patches import Ellipse

        for label, ellipse in ellipses.items():
            patch = Ellipse(
                (ellipse["cx"], ellipse["cy"]),
                width=2.0 * ellipse["rx"],
                height=2.0 * ellipse["ry"],
                angle=ellipse["angle"],
                facecolor="none",
                edgecolor=palette.get(label, "#111827"),
                linewidth=1.2,
                alpha=0.75,
            )
            ax.add_patch(patch)
    except Exception:
        pass

    ax.set_title(
        "Classification QC UMAP at max N="
        f"{len(prototype_samples)}: browser-matched fused teacher features\n"
        f"UMAP cosine, n_neighbors={DEFAULT_BROWSER_UMAP_NEIGHBORS}, "
        f"min_dist={DEFAULT_BROWSER_UMAP_MIN_DIST}, random_state={DEFAULT_BROWSER_UMAP_RANDOM_STATE}"
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, linewidth=0.4, alpha=0.25)
    ax.legend(fontsize=8, markerscale=1.5, ncols=2)

    for fmt in formats:
        path = figure_dir / f"infospace_umap_qc_l1_browser_matched.{fmt}"
        fig.savefig(path, dpi=220)
        written.append(str(path))
    plt.close(fig)


def _plot_infospace_distribution(
    *,
    plt: Any,
    figure_dir: Path,
    aggregate_rows: list[dict[str, Any]],
    formats: list[str],
    written: list[str],
) -> None:
    counts = [int(row["prototype_sample_count"]) for row in aggregate_rows]
    q05 = [_as_float(row, "infospace_novelty_q05_mean") for row in aggregate_rows]
    q25 = [_as_float(row, "infospace_novelty_q25_mean") for row in aggregate_rows]
    q50 = [_as_float(row, "infospace_novelty_q50_mean") for row in aggregate_rows]
    q75 = [_as_float(row, "infospace_novelty_q75_mean") for row in aggregate_rows]
    q95 = [_as_float(row, "infospace_novelty_q95_mean") for row in aggregate_rows]

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.fill_between(counts, q05, q95, alpha=0.12, linewidth=0, label="5–95%")
    ax.fill_between(counts, q25, q75, alpha=0.22, linewidth=0, label="25–75%")
    ax.plot(counts, q50, marker="o", linewidth=1.8, label="median")
    ax.set_title("Distribution of newly added tile novelty in teacher infospace")
    ax.set_xlabel("Prototype tile count N")
    ax.set_ylabel("Novelty: 1 - mean top-k cosine to previous tiles")
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend(fontsize=8)
    for fmt in formats:
        path = figure_dir / f"infospace_novelty_distribution.{fmt}"
        fig.savefig(path, dpi=220)
        written.append(str(path))
    plt.close(fig)


def _plot_prototype_curves(
    *,
    plt: Any,
    figure_dir: Path,
    prototype_rows: list[dict[str, Any]],
    level: int,
    formats: list[str],
    written: list[str],
) -> None:
    rows = [row for row in prototype_rows if int(row["level"]) == level]
    if not rows:
        return
    counts = sorted({int(row["prototype_sample_count"]) for row in rows})
    prototypes = sorted({str(row["prototype"]) for row in rows})
    if not counts or not prototypes:
        return

    def mean_for(count: int, prototype: str, key: str) -> float:
        return _mean([
            _as_float(row, key)
            for row in rows
            if int(row["prototype_sample_count"]) == count and str(row["prototype"]) == prototype
        ])

    fig, axes = plt.subplots(3, 1, figsize=(11, max(8, 0.34 * len(prototypes) + 6)), constrained_layout=True)
    panels = [
        ("prototype_tile_count", f"L{level} selected tile count"),
        ("global_infospace_novelty", f"L{level} global infospace novelty of newly added tiles"),
        ("local_infospace_novelty", f"L{level} same-prototype infospace novelty of newly added tiles"),
    ]
    for ax, (key, title) in zip(axes, panels):
        for prototype in prototypes:
            ax.plot(counts, [mean_for(count, prototype, key) for count in counts], marker="o", linewidth=1.2, label=prototype)
        ax.set_title(title)
        ax.set_xlabel("Prototype tile count N")
        ax.grid(True, linewidth=0.5, alpha=0.35)
    axes[0].legend(fontsize=7, ncols=2)
    for fmt in formats:
        path = figure_dir / f"infospace_level{level}_prototype_curves.{fmt}"
        fig.savefig(path, dpi=220)
        written.append(str(path))
    plt.close(fig)


def _plot_prototype_audit(
    *,
    plt: Any,
    figure_dir: Path,
    prototype_rows: list[dict[str, Any]],
    level: int,
    formats: list[str],
    written: list[str],
) -> None:
    rows = [row for row in prototype_rows if int(row["level"]) == level]
    if not rows:
        return
    final_count = max(int(row["prototype_sample_count"]) for row in rows)
    final_rows = [row for row in rows if int(row["prototype_sample_count"]) == final_count]
    prototypes = sorted({str(row["prototype"]) for row in final_rows})
    if not prototypes:
        return

    columns = [
        "prototype_tile_count",
        "new_prototype_tile_count",
        "prototype_tile_fraction",
        "global_infospace_novelty",
        "local_infospace_novelty",
        "global_infospace_redundancy",
        "local_infospace_redundancy",
        "prototype_drift",
    ]
    labels = ["n", "new n", "frac", "global nov", "local nov", "global red", "local red", "drift"]

    matrix: list[list[float]] = []
    text_matrix: list[list[str]] = []
    for prototype in prototypes:
        items = [row for row in final_rows if str(row["prototype"]) == prototype]
        values = [_mean([_as_float(row, key) for row in items]) for key in columns]
        matrix.append(values)
        text_row: list[str] = []
        for key, value in zip(columns, values):
            if not math.isfinite(value):
                text_row.append("NA")
            elif key.endswith("_count"):
                text_row.append(str(int(round(value))))
            elif key.endswith("_fraction"):
                text_row.append(f"{value:.2%}")
            else:
                text_row.append(f"{value:.3f}")
        text_matrix.append(text_row)

    arr = np.asarray(matrix, dtype=np.float32)
    normalized = np.zeros_like(arr, dtype=np.float32)
    for col in range(arr.shape[1]):
        values = arr[:, col]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            normalized[:, col] = np.nan
        else:
            low = float(finite.min())
            high = float(finite.max())
            normalized[:, col] = 0.5 if high <= low else (values - low) / (high - low)

    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.38 * len(prototypes) + 1.8)), constrained_layout=True)
    image = ax.imshow(normalized, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title(f"L{level} prototype audit at final N={final_count}")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(prototypes)), prototypes)
    ax.tick_params(axis="both", labelsize=8)
    for y, row in enumerate(text_matrix):
        for x, value in enumerate(row):
            ax.text(x, y, value, ha="center", va="center", fontsize=7, fontweight="bold")
    cbar = fig.colorbar(image, ax=ax, shrink=0.75)
    cbar.set_label("Column-normalized value")
    for fmt in formats:
        path = figure_dir / f"infospace_level{level}_prototype_audit.{fmt}"
        fig.savefig(path, dpi=220)
        written.append(str(path))
    plt.close(fig)


def _plot_curves(
    *,
    output_root: Path,
    teacher_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    fixed_probe_information: dict[str, Any],
    prototype_rows: list[dict[str, Any]],
    max_subset: list[PrototypeSample],
    pca_features_by_teacher: dict[str, dict[str, np.ndarray]],
    pca_label_levels: list[str],
    max_pca_categories: int,
    formats: list[str],
    no_pca: bool,
    verbose: bool,
) -> list[str]:
    os.environ.setdefault("MPLCONFIGDIR", str(output_root / ".matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to render figures; pass --no-plots to skip figures") from exc

    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    counts = [int(row["prototype_sample_count"]) for row in aggregate_rows]
    elbow_counts = [int(row["prototype_sample_count"]) for row in aggregate_rows if str(row.get("elbow_candidate", "")).lower() == "true"]
    plateau_counts = [int(row["prototype_sample_count"]) for row in aggregate_rows if row["plateau_consensus"] == "true"]
    decision_count = elbow_counts[0] if elbow_counts else (plateau_counts[0] if plateau_counts else None)

    def save(fig: Any, stem: str) -> None:
        for fmt in formats:
            path = figure_dir / f"{stem}.{fmt}"
            fig.savefig(path, dpi=220)
            written.append(str(path))

    _log("plot fixed-probe coverage curves", enabled=verbose)
    fixed_units = [
        ("all classification", fixed_probe_information["global"]),
        *sorted(fixed_probe_information["classes"].items()),
    ]
    ncols = 3
    nrows = int(math.ceil(len(fixed_units) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 3.8 * nrows),
        constrained_layout=True,
    )
    axes_arr = np.asarray(axes).reshape(-1)
    for axis, (name, unit) in zip(axes_arr, fixed_units):
        curve = list(unit.get("curve", []))
        if curve:
            x = [int(row["sample_count"]) for row in curve]
            y = [
                _as_float(row, "remaining_novelty_mean")
                for row in curve
            ]
            low = [
                _as_float(row, "remaining_novelty_mean_ci_low")
                for row in curve
            ]
            high = [
                _as_float(row, "remaining_novelty_mean_ci_high")
                for row in curve
            ]
            axis.plot(x, y, marker="o", linewidth=1.8)
            axis.fill_between(x, low, high, alpha=0.18, linewidth=0)
        axis.set_title(
            f"{name}\n{unit['status']} · N={unit['positive_tile_count']}"
        )
        axis.set_xlabel("nested reference tile count")
        axis.set_ylabel("fixed-probe remaining novelty")
        axis.grid(True, linewidth=0.5, alpha=0.35)
    for axis in axes_arr[len(fixed_units) :]:
        axis.axis("off")
    fig.suptitle(
        "Classification annotation coverage: fixed probe under nested reference growth"
    )
    save(fig, "infospace_information_summary")
    plt.close(fig)

    _log("plot secondary discovery diagnostic", enabled=verbose)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    panels = [
        ("infospace_novelty_mean", "New-batch infospace novelty", axes[0, 0]),
        ("infospace_redundancy_mean", "New-batch infospace redundancy", axes[0, 1]),
        ("prototype_drift_mean", "Prototype center drift", axes[1, 0]),
        ("infospace_novelty_q95_mean", "High-tail novelty (q95)", axes[1, 1]),
    ]
    for key, title, ax in panels:
        values = [_as_float(row, key) for row in aggregate_rows]
        ax.plot(counts, values, marker="o", linewidth=1.8)
        if key == "infospace_novelty_mean":
            ax.fill_between(
                counts,
                [_as_float(row, "infospace_novelty_ci_low_mean") for row in aggregate_rows],
                [_as_float(row, "infospace_novelty_ci_high_mean") for row in aggregate_rows],
                alpha=0.18,
                linewidth=0,
            )
        if key == "infospace_redundancy_mean":
            ax.fill_between(
                counts,
                [_as_float(row, "infospace_redundancy_ci_low_mean") for row in aggregate_rows],
                [_as_float(row, "infospace_redundancy_ci_high_mean") for row in aggregate_rows],
                alpha=0.18,
                linewidth=0,
            )
        if decision_count is not None:
            ax.axvline(decision_count, linestyle="--", linewidth=0.9, alpha=0.65)
        ax.set_title(title)
        ax.set_xlabel("Prototype tile count N")
        ax.grid(True, linewidth=0.5, alpha=0.35)
    fig.suptitle(
        "Secondary new-batch discovery diagnostic — not a stopping signal"
    )
    save(fig, "infospace_discovery_diagnostic_summary")
    plt.close(fig)

    _plot_infospace_distribution(plt=plt, figure_dir=figure_dir, aggregate_rows=aggregate_rows, formats=formats, written=written)

    teacher_names = sorted({row["teacher"] for row in teacher_rows})
    if len(teacher_names) > 1:
        _log("plot teacher audit curves", enabled=verbose)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        panels = [
            ("infospace_novelty", "Novelty by teacher", axes[0, 0]),
            ("infospace_redundancy", "Redundancy by teacher", axes[0, 1]),
            ("prototype_drift", "Prototype drift by teacher", axes[1, 0]),
            ("infospace_novelty_q95", "q95 novelty by teacher", axes[1, 1]),
        ]
        for key, title, ax in panels:
            for teacher in teacher_names:
                rows = [row for row in teacher_rows if row["teacher"] == teacher]
                ax.plot(
                    [int(row["prototype_sample_count"]) for row in rows],
                    [_as_float(row, key) for row in rows],
                    marker="o",
                    linewidth=1.4,
                    label=teacher,
                )
            if decision_count is not None:
                ax.axvline(decision_count, linestyle="--", linewidth=0.9, alpha=0.65)
            ax.set_title(title)
            ax.set_xlabel("Prototype tile count N")
            ax.grid(True, linewidth=0.5, alpha=0.35)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(
            "Secondary new-batch discovery diagnostic by teacher"
        )
        save(fig, "infospace_discovery_diagnostic_teacher_audit")
        plt.close(fig)

    plotted_levels = sorted(
        {int(row["level"]) for row in prototype_rows}
    )
    for level in plotted_levels:
        _log(f"plot L{level} prototype curves/audit", enabled=verbose)
        _plot_prototype_curves(
            plt=plt,
            figure_dir=figure_dir,
            prototype_rows=prototype_rows,
            level=level,
            formats=formats,
            written=written,
        )
        _plot_prototype_audit(
            plt=plt,
            figure_dir=figure_dir,
            prototype_rows=prototype_rows,
            level=level,
            formats=formats,
            written=written,
        )

    if not no_pca:
        _plot_pca_qc(
            plt=plt,
            figure_dir=figure_dir,
            prototype_samples=max_subset,
            pca_features_by_teacher=pca_features_by_teacher,
            label_levels=pca_label_levels,
            max_categories=max_pca_categories,
            formats=formats,
            written=written,
            verbose=verbose,
        )
        _plot_browser_matched_l1_umap_qc(
            plt=plt,
            figure_dir=figure_dir,
            prototype_samples=max_subset,
            pca_features_by_teacher=pca_features_by_teacher,
            formats=formats,
            written=written,
            verbose=verbose,
        )

    return written


def _require_plot_backend(args: argparse.Namespace) -> None:
    if bool(getattr(args, "no_plots", False)):
        return
    if importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError("matplotlib is required before reading teacher features; pass --no-plots to skip figures")


def _process_teacher_worker(
    *,
    teacher: str,
    teacher_paths: list[Path],
    needed_tile_ids: list[str],
    subsets: dict[int, list[PrototypeSample]],
    level1_names: list[str],
    level2_names: list[str],
    bootstrap_iterations: int,
    seed: int,
    topk: int,
    verbose: bool,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    _log(f"teacher={teacher}: start worker", enabled=verbose)
    store = FeatureStore({teacher: teacher_paths})
    try:
        _log(f"teacher={teacher}: start feature loading", enabled=verbose)
        features = _feature_map(store, teacher, needed_tile_ids, verbose=verbose)
        _log(f"teacher={teacher}: feature loading done", enabled=verbose)
        teacher_rows, prototype_rows = _teacher_curve(
            teacher=teacher,
            subsets=subsets,
            features=features,
            level1_names=level1_names,
            level2_names=level2_names,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
            topk=topk,
            verbose=verbose,
        )
        pca_features = {tile_id: features[tile_id] for tile_id in needed_tile_ids}
        _log(f"teacher={teacher}: worker done", enabled=verbose)
        return teacher, teacher_rows, prototype_rows, pca_features
    finally:
        store.close()


def run(args: argparse.Namespace) -> dict[str, Any]:
    verbose = not bool(getattr(args, "quiet", False))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _require_plot_backend(args)

    seed = int(args.seed)
    prototype_sample_counts = _parse_int_list(args.prototype_sample_counts)
    plot_formats = _parse_str_list(getattr(args, "plot_formats", DEFAULT_PLOT_FORMATS)) or _parse_str_list(DEFAULT_PLOT_FORMATS)
    pca_label_levels = [value.lower() for value in _parse_str_list(getattr(args, "pca_label_levels", DEFAULT_PCA_LABEL_LEVELS))]
    infospace_topk = int(getattr(args, "infospace_topk", DEFAULT_INFOSPACE_TOPK))
    prototype_sample_group_key = getattr(args, "prototype_sample_group_key", DEFAULT_PROTOTYPE_SAMPLE_GROUP_KEY)
    workers = int(getattr(args, "workers", DEFAULT_WORKERS))
    bootstrap_iterations = int(getattr(args, "bootstrap_iterations", DEFAULT_BOOTSTRAP_ITERATIONS))
    plateau_novelty_threshold = float(getattr(args, "plateau_novelty_threshold", DEFAULT_PLATEAU_NOVELTY_THRESHOLD))
    plateau_drift_threshold = float(getattr(args, "plateau_drift_threshold", DEFAULT_PLATEAU_DRIFT_THRESHOLD))
    plateau_redundancy_threshold = float(getattr(args, "plateau_redundancy_threshold", DEFAULT_PLATEAU_REDUNDANCY_THRESHOLD))
    fixed_probe_resamples = int(
        getattr(
            args,
            "fixed_probe_resamples",
            DEFAULT_FIXED_PROBE_RESAMPLES,
        )
    )
    probe_slide_fraction = float(
        getattr(
            args,
            "probe_slide_fraction",
            DEFAULT_PROBE_SLIDE_FRACTION,
        )
    )
    min_probe_slides = int(
        getattr(args, "min_probe_slides", DEFAULT_MIN_PROBE_SLIDES)
    )
    fixed_probe_support = float(
        getattr(
            args,
            "fixed_probe_support",
            DEFAULT_FIXED_PROBE_SUPPORT,
        )
    )
    confirmation_increments = int(
        getattr(
            args,
            "confirmation_increments",
            DEFAULT_CONFIRMATION_INCREMENTS,
        )
    )

    _log("load prototype_samples", enabled=verbose)
    generated_manifest_info: dict[str, Any] = {}
    if args.annotation_json:
        prototype_samples = _prototype_samples_from_annotation_json(Path(args.annotation_json))
        input_dir = output_root / "inputs"
        prototype_sample_manifest = input_dir / "prototype_sample_pool.csv"
        _write_prototype_sample_manifest(prototype_sample_manifest, prototype_samples)
        generated_manifest_info = {
            "annotation_json": str(args.annotation_json),
            "generated_prototype_sample_manifest": str(prototype_sample_manifest),
            "prototype_sample_count": len(prototype_samples),
            "seed": seed,
        }
    else:
        if not args.prototype_sample_manifest:
            raise ValueError("--prototype-sample-manifest is required unless --annotation-json is used")
        prototype_sample_manifest = Path(args.prototype_sample_manifest)
        prototype_samples = _load_prototype_samples_from_manifest(prototype_sample_manifest)

    _log(f"prototype_sample pool loaded: n={len(prototype_samples)}", enabled=verbose)
    subsets, skipped_counts = _nested_subsets(prototype_samples, prototype_sample_counts, seed, prototype_sample_group_key)
    if skipped_counts:
        _log(f"skip requested counts > prototype_sample_pool_count: {skipped_counts}", enabled=verbose)
    _log(f"available prototype_sample counts: {sorted(subsets)}", enabled=verbose)

    level1_names, level2_names = _load_contract(Path(args.prototype_contract) if args.prototype_contract else None, prototype_samples)
    prototype_levels = {
        value.lower()
        for value in _parse_str_list(
            getattr(args, "prototype_levels", "l1,l2")
        )
    }
    if not prototype_levels or not prototype_levels <= {"l1", "l2"}:
        raise ValueError(
            "--prototype-levels must contain l1, l2, or both"
        )
    if "l1" not in prototype_levels:
        level1_names = []
    if "l2" not in prototype_levels:
        level2_names = []
    _log(
        "prototype labels: "
        f"classification={len(level1_names)} spatial={len(level2_names)}",
        enabled=verbose,
    )

    teacher_paths = _resolve_teacher_paths(args)
    _log(f"teachers resolved: {list(teacher_paths)}", enabled=verbose)

    max_count = max(subsets)
    max_subset = subsets[max_count]
    needed_tile_ids = [prototype_sample.tile_id for prototype_sample in max_subset]

    report = {
        "sweep_type": "infospace_information_curve",
        "objective": (
            "estimate annotation coverage from a fixed slide-separated probe "
            "under nested reference growth before main-model training"
        ),
        "prototype_sample_manifest": str(prototype_sample_manifest),
        "prototype_contract": str(args.prototype_contract or ""),
        "prototype_sample_counts_requested": prototype_sample_counts,
        "prototype_sample_counts_available": sorted(subsets),
        "prototype_sample_counts_skipped": skipped_counts,
        "nested_subsets": True,
        "validation_split_used": False,
        "coverage_metric_used": True,
        "does_not_train": True,
        "seed": seed,
        "prototype_sample_group_key": prototype_sample_group_key,
        "prototype_sample_pool_count": len(prototype_samples),
        "max_subset_count": len(max_subset),
        "infospace_topk": infospace_topk,
        "workers_requested": workers,
        "teachers": list(teacher_paths),
        "teacher_feature_packages": {teacher: [str(path) for path in paths] for teacher, paths in teacher_paths.items()},
        "level1_prototypes": level1_names,
        "level2_prototypes": level2_names,
        "prototype_levels": sorted(prototype_levels),
        "generated_manifests": generated_manifest_info,
    }

    _log("write nested subset manifests", enabled=verbose)
    subset_rows = []
    for count, subset in sorted(subsets.items()):
        subset_path = output_root / f"N{count}" / "prototype_samples.csv"
        _write_csv(
            subset_path,
            [
                {
                    "tile_id": prototype_sample.tile_id,
                    "level1_label": prototype_sample.level1_label,
                    "level2_labels": ";".join(prototype_sample.level2_labels),
                    "slide_id": prototype_sample.slide_id,
                    "patient_id": prototype_sample.patient_id,
                    "center": prototype_sample.center,
                }
                for prototype_sample in subset
            ],
        )
        subset_rows.append({"prototype_sample_count": count, "prototype_sample_count_actual": len(subset), "path": str(subset_path)})
    _write_csv(output_root / "nested_subsets.csv", subset_rows)

    teacher_rows: list[dict[str, Any]] = []
    prototype_rows: list[dict[str, Any]] = []
    pca_features_by_teacher: dict[str, dict[str, np.ndarray]] = {}

    requested_workers = workers
    if requested_workers <= 0:
        worker_count = min(len(teacher_paths), os.cpu_count() or 1)
    else:
        worker_count = min(max(1, requested_workers), len(teacher_paths))
    worker_count = max(1, worker_count)
    _log(f"teacher workers: {worker_count}", enabled=verbose)

    worker_kwargs = [
        {
            "teacher": teacher,
            "teacher_paths": paths,
            "needed_tile_ids": needed_tile_ids,
            "subsets": subsets,
            "level1_names": level1_names,
            "level2_names": level2_names,
            "bootstrap_iterations": bootstrap_iterations,
            "seed": seed,
            "topk": infospace_topk,
            "verbose": verbose,
        }
        for teacher, paths in teacher_paths.items()
    ]

    if worker_count == 1:
        for kwargs in worker_kwargs:
            teacher, next_teacher_rows, next_prototype_rows, pca_features = _process_teacher_worker(**kwargs)
            teacher_rows.extend(next_teacher_rows)
            prototype_rows.extend(next_prototype_rows)
            pca_features_by_teacher[teacher] = pca_features
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_process_teacher_worker, **kwargs) for kwargs in worker_kwargs]
            for future in as_completed(futures):
                teacher, next_teacher_rows, next_prototype_rows, pca_features = future.result()
                teacher_rows.extend(next_teacher_rows)
                prototype_rows.extend(next_prototype_rows)
                pca_features_by_teacher[teacher] = pca_features

    _log("aggregate teacher metrics", enabled=verbose)
    aggregate_rows = _aggregate_rows(
        teacher_rows,
        plateau_novelty_threshold=plateau_novelty_threshold,
        plateau_drift_threshold=plateau_drift_threshold,
        plateau_redundancy_threshold=plateau_redundancy_threshold,
    )
    discovery_recommendation = _recommendation(aggregate_rows)
    fixed_probe_information = _fixed_probe_l1_information(
        prototype_samples=max_subset,
        features_by_teacher=pca_features_by_teacher,
        level1_names=level1_names,
        requested_counts=prototype_sample_counts,
        seed=seed,
        resamples=fixed_probe_resamples,
        topk=infospace_topk,
        probe_slide_fraction=probe_slide_fraction,
        min_probe_slides=min_probe_slides,
        marginal_ratio_threshold=DEFAULT_ELBOW_MARGINAL_RATIO,
        drift_threshold=plateau_drift_threshold,
        support_threshold=fixed_probe_support,
        confirmation_increments=confirmation_increments,
        min_slides=5,
        min_increments=3,
    )

    _log("write CSV/JSON outputs", enabled=verbose)
    _write_csv(output_root / "infospace_information_by_teacher.csv", teacher_rows)
    _write_csv(output_root / "infospace_information_by_prototype.csv", prototype_rows)
    _write_csv(output_root / "infospace_information_summary.csv", aggregate_rows)
    fixed_curve_rows = [
        {
            "unit": unit,
            **row,
        }
        for unit, value in [
            ("__all_l1__", fixed_probe_information["global"]),
            *sorted(fixed_probe_information["classes"].items()),
        ]
        for row in value.get("curve", [])
    ]
    _write_csv(
        output_root / "infospace_fixed_probe_information.csv",
        fixed_curve_rows,
    )

    figure_paths: list[str] = []
    if not bool(getattr(args, "no_plots", False)):
        _log("render figures", enabled=verbose)
        figure_paths = _plot_curves(
            output_root=output_root,
            teacher_rows=teacher_rows,
            aggregate_rows=aggregate_rows,
            fixed_probe_information=fixed_probe_information,
            prototype_rows=prototype_rows,
            max_subset=max_subset,
            pca_features_by_teacher=pca_features_by_teacher,
            pca_label_levels=pca_label_levels,
            max_pca_categories=int(getattr(args, "max_pca_categories", DEFAULT_MAX_PCA_CATEGORIES)),
            formats=plot_formats,
            no_pca=bool(getattr(args, "no_pca", False)),
            verbose=verbose,
        )

    report.update(
        {
            "fixed_probe_information": fixed_probe_information,
            "discovery_diagnostic": {
                "claim_scope": (
                    "new-batch novelty only; non-monotone and never used "
                    "to stop annotation"
                ),
                "recommendation": discovery_recommendation,
            },
            "figures": figure_paths,
        }
    )
    _write_json(output_root / "infospace_information_report.json", report)

    _log(
        "done "
        f"output_root={output_root} teachers={len(teacher_paths)} counts={len(subsets)} "
        f"fixed_probe_status={fixed_probe_information['status']} "
        f"figures={len(figure_paths)}",
        enabled=verbose,
    )
    return {
        "teacher": teacher_rows,
        "prototype": prototype_rows,
        "summary": aggregate_rows,
        "fixed_probe_information": fixed_probe_information,
        "discovery_diagnostic": {
            "recommendation": discovery_recommendation,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute teacher-feature infospace novelty decay curves for prototype tile selection. "
            "No validation split, no coverage metric, no training."
        )
    )
    parser.add_argument("--teacher-feature-root", default="")
    parser.add_argument("--teacher-feature-packages", default="")
    parser.add_argument("--teachers", default="")
    parser.add_argument("--annotation-json", default="")
    parser.add_argument("--prototype-sample-manifest", default="")
    parser.add_argument("--prototype-contract", default="")
    parser.add_argument(
        "--prototype-levels",
        default="l1,l2",
        help="Comma-separated prototype levels included in the curve: l1, l2",
    )
    parser.add_argument("--output-root", default="outputs/infospace_information_curve")
    parser.add_argument("--prototype-sample-counts", default=DEFAULT_PROTOTYPE_SAMPLE_COUNTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--prototype-sample-group-key", default=DEFAULT_PROTOTYPE_SAMPLE_GROUP_KEY)
    parser.add_argument("--infospace-topk", type=int, default=DEFAULT_INFOSPACE_TOPK)
    parser.add_argument("--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS)
    parser.add_argument("--plateau-novelty-threshold", type=float, default=DEFAULT_PLATEAU_NOVELTY_THRESHOLD)
    parser.add_argument("--plateau-drift-threshold", type=float, default=DEFAULT_PLATEAU_DRIFT_THRESHOLD)
    parser.add_argument("--plateau-redundancy-threshold", type=float, default=DEFAULT_PLATEAU_REDUNDANCY_THRESHOLD)
    parser.add_argument(
        "--fixed-probe-resamples",
        type=int,
        default=DEFAULT_FIXED_PROBE_RESAMPLES,
    )
    parser.add_argument(
        "--probe-slide-fraction",
        type=float,
        default=DEFAULT_PROBE_SLIDE_FRACTION,
    )
    parser.add_argument(
        "--min-probe-slides",
        type=int,
        default=DEFAULT_MIN_PROBE_SLIDES,
    )
    parser.add_argument(
        "--fixed-probe-support",
        type=float,
        default=DEFAULT_FIXED_PROBE_SUPPORT,
    )
    parser.add_argument(
        "--confirmation-increments",
        type=int,
        default=DEFAULT_CONFIRMATION_INCREMENTS,
    )
    parser.add_argument("--plot-formats", default=DEFAULT_PLOT_FORMATS)
    parser.add_argument("--pca-label-levels", default=DEFAULT_PCA_LABEL_LEVELS)
    parser.add_argument("--max-pca-categories", type=int, default=DEFAULT_MAX_PCA_CATEGORIES)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="teacher-level parallel workers; <=0 means auto")
    parser.add_argument("--no-pca", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
