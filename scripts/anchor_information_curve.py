#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
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


@dataclass(frozen=True)
class Anchor:
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

DEFAULT_ANCHOR_COUNTS = "100,200,400,800,1200,1600,2000,3000"
DEFAULT_PLOT_FORMATS = "png,pdf"
DEFAULT_SEED = 13
DEFAULT_ANCHOR_GROUP_KEY = "tile_id"
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
        raise ValueError("expected at least one anchor count")
    if any(count <= 0 for count in counts):
        raise ValueError(f"anchor counts must be positive: {counts}")
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


def _normalize_anchor(row: dict[str, Any]) -> Anchor:
    tile_id = str(row.get("tile_id", "")).strip()
    if not tile_id:
        raise ValueError("anchor row missing tile_id")

    level1 = str(row.get("level1_label") or row.get("l1") or "").strip()
    if not level1:
        raise ValueError(f"anchor row missing level1 label: tile_id={tile_id}")

    slide_id = str(row.get("slide_id") or row.get("slide") or tile_id).strip()
    return Anchor(
        tile_id=tile_id,
        level1_label=level1,
        level2_labels=_split_labels(str(row.get("level2_labels") or row.get("l2") or "")),
        slide_id=slide_id,
        patient_id=str(row.get("patient_id") or row.get("patient") or slide_id).strip(),
        center=str(row.get("center") or row.get("dataset") or "").strip(),
    )


def _anchors_from_annotation_json(path: Path) -> list[Anchor]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    annotations = payload.get("annotations")
    if not isinstance(annotations, dict):
        raise ValueError(f"annotation JSON missing annotations object: {path}")

    anchors: list[Anchor] = []
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
        anchors.append(
            Anchor(
                tile_id=tile_id,
                level1_label=level1,
                level2_labels=l2,
                slide_id=slide_id,
                patient_id=str(item.get("patient_id") or slide_id).strip(),
                center=dataset,
            )
        )

    if not anchors:
        raise ValueError(f"annotation JSON has no usable anchors: {path}")
    return anchors


def _load_anchors_from_manifest(path: Path) -> list[Anchor]:
    anchors = [_normalize_anchor(row) for row in _read_csv(path)]
    if not anchors:
        raise ValueError(f"anchor manifest has no usable anchors: {path}")
    return anchors


def _write_anchor_manifest(path: Path, anchors: list[Anchor]) -> None:
    _write_csv(
        path,
        [
            {
                "tile_id": anchor.tile_id,
                "level1_label": anchor.level1_label,
                "level2_labels": ";".join(anchor.level2_labels),
                "slide_id": anchor.slide_id,
                "patient_id": anchor.patient_id,
                "center": anchor.center,
                "source": "anchor_pool",
            }
            for anchor in anchors
        ],
    )


def _load_contract(path: Path | None, anchors: list[Anchor]) -> tuple[list[str], list[str]]:
    if path is None:
        level1: list[str] = []
        level2: list[str] = []
        for anchor in anchors:
            if anchor.level1_label not in level1:
                level1.append(anchor.level1_label)
            for label in anchor.level2_labels:
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
    anchors: list[Anchor],
    counts: list[int],
    seed: int,
    group_key: str,
) -> tuple[dict[int, list[Anchor]], list[int]]:
    available_counts = [count for count in counts if count <= len(anchors)]
    skipped_counts = [count for count in counts if count > len(anchors)]
    if not available_counts:
        raise ValueError(f"all requested counts exceed available anchors={len(anchors)}")

    rng = random.Random(seed)
    groups: dict[str, list[Anchor]] = {}
    for anchor in anchors:
        if group_key in {"tile_id", "slide_id", "patient_id", "center"}:
            key = str(getattr(anchor, group_key) or anchor.tile_id)
        else:
            key = anchor.tile_id
        groups.setdefault(key, []).append(anchor)

    group_keys = sorted(groups)
    rng.shuffle(group_keys)

    ordered: list[Anchor] = []
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
        from hcc_sempath.io.feature_cache import FeatureCacheReader

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
    anchors: list[Anchor],
    features: dict[str, np.ndarray],
    level1_names: list[str],
    level2_names: list[str],
) -> Centers:
    level1: dict[str, list[np.ndarray]] = {name: [] for name in level1_names}
    level2: dict[str, list[np.ndarray]] = {name: [] for name in level2_names}

    for anchor in anchors:
        feature = features[anchor.tile_id]
        if anchor.level1_label in level1:
            level1[anchor.level1_label].append(feature)
        for label in anchor.level2_labels:
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


def _anchor_has_prototype(anchor: Anchor, level: int, name: str) -> bool:
    if level == 1:
        return anchor.level1_label == name
    return name in anchor.level2_labels


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
    anchor_count: int,
    anchors: list[Anchor],
    previous_anchors: list[Anchor] | None,
    features: dict[str, np.ndarray],
    centers: Centers,
    previous_centers: Centers | None,
    level1_names: list[str],
    level2_names: list[str],
    topk: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_anchors = previous_anchors or []

    for level, names in [(1, level1_names), (2, level2_names)]:
        for name in names:
            selected = [anchor for anchor in anchors if _anchor_has_prototype(anchor, level, name)]
            previous_selected = [anchor for anchor in previous_anchors if _anchor_has_prototype(anchor, level, name)]
            new_selected = selected[len(previous_selected) :]

            new_features = [features[anchor.tile_id] for anchor in new_selected]
            previous_global_features = [features[anchor.tile_id] for anchor in previous_anchors]
            previous_local_features = [features[anchor.tile_id] for anchor in previous_selected]

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
                "anchor_count": anchor_count,
                "teacher": teacher,
                "level": level,
                "prototype": name,
                "prototype_tile_count": len(selected),
                "new_prototype_tile_count": len(new_selected),
                "prototype_tile_fraction": _format_float(len(selected) / len(anchors) if anchors else math.nan),
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
    subsets: dict[int, list[Anchor]],
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
        anchors = subsets[count]
        centers = _centers(anchors, features, level1_names, level2_names)

        if previous is None:
            new_anchors = anchors
            previous_anchors: list[Anchor] = []
            novelty = np.asarray([], dtype=np.float32)
            redundancy = np.asarray([], dtype=np.float32)
        else:
            previous_anchors = previous["anchors"]
            new_anchors = anchors[len(previous_anchors) :]
            new_features = [features[anchor.tile_id] for anchor in new_anchors]
            previous_features = [features[anchor.tile_id] for anchor in previous_anchors]
            novelty = _novelty_values(new_features, previous_features, k=topk)
            redundancy = _redundancy_values(new_features, previous_features, k=topk)

        drift_values = _prototype_drift_values(centers, previous["centers"] if previous else None)
        missing_l1 = [name for name in level1_names if name not in centers.level1]
        missing_l2 = [name for name in level2_names if name not in centers.level2]

        row: dict[str, Any] = {
            "anchor_count": count,
            "teacher": teacher,
            "new_tile_count": len(new_anchors),
            "previous_tile_count": len(previous_anchors),
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
                anchor_count=count,
                anchors=anchors,
                previous_anchors=previous_anchors,
                features=features,
                centers=centers,
                previous_centers=previous["centers"] if previous else None,
                level1_names=level1_names,
                level2_names=level2_names,
                topk=topk,
            )
        )

        previous = {
            "anchors": anchors,
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
    counts = sorted({int(row["anchor_count"]) for row in teacher_rows})
    consecutive_plateau = 0

    quantile_suffixes = ["q05", "q25", "q50", "q75", "q95"]

    for count in counts:
        rows = [row for row in teacher_rows if int(row["anchor_count"]) == count]
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
            "anchor_count": count,
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
        count = int(row["anchor_count"])
        novelty = _as_float(row, "infospace_novelty_mean")
        drift = _as_float(row, "prototype_drift_mean")

        if previous is None:
            row["marginal_anchor_count_delta"] = ""
            row["marginal_novelty_drop"] = ""
            row["marginal_novelty_drop_per_100_tiles"] = ""
            row["marginal_drift_drop"] = ""
            previous = row
            continue

        previous_count = int(previous["anchor_count"])
        previous_novelty = _as_float(previous, "infospace_novelty_mean")
        previous_drift = _as_float(previous, "prototype_drift_mean")
        delta = count - previous_count
        novelty_drop = previous_novelty - novelty if math.isfinite(previous_novelty) and math.isfinite(novelty) else math.nan
        drop_per_100 = novelty_drop / delta * 100.0 if delta > 0 and math.isfinite(novelty_drop) else math.nan
        drift_drop = previous_drift - drift if math.isfinite(previous_drift) and math.isfinite(drift) else math.nan

        row["marginal_anchor_count_delta"] = delta
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
        onset_count = int(onset["anchor_count"])
        counts = [int(row["anchor_count"]) for row in aggregate_rows]
        onset_index = counts.index(onset_count)
        confirmation_count = counts[min(onset_index + 1, len(counts) - 1)]
        conservative_index = min(onset_index + 2, len(counts) - 1)
        conservative_count = counts[conservative_index]
        return {
            "recommended_anchor_count": confirmation_count,
            "elbow_onset_count": onset_count,
            "conservative_anchor_count": conservative_count,
            "reason": (
                "marginal utility elbow: novelty drop per 100 anchors falls below "
                f"{DEFAULT_ELBOW_MARGINAL_RATIO:.0%} of the best observed marginal gain while prototype drift is low"
            ),
            "marginal_utility_ratio_threshold": DEFAULT_ELBOW_MARGINAL_RATIO,
            "novelty_drop_per_100_at_elbow": _format_float(_as_float(onset, "marginal_novelty_drop_per_100_tiles")),
            "prototype_drift_at_elbow": _format_float(_as_float(onset, "prototype_drift_mean")),
        }

    plateau_counts = [int(row["anchor_count"]) for row in aggregate_rows if row["plateau_consensus"] == "true"]
    available_counts = [int(row["anchor_count"]) for row in aggregate_rows]
    if plateau_counts:
        return {
            "recommended_anchor_count": plateau_counts[0],
            "reason": "strict novelty plateau; this is stronger than the marginal utility elbow criterion",
        }
    return {
        "recommended_anchor_count": available_counts[-1] if available_counts else None,
        "reason": "no marginal utility elbow detected within available anchors",
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
        raise RuntimeError("umap-learn is required for browser-matched L1 UMAP QC") from exc

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


def _anchor_labels(anchor: Anchor, level: str) -> tuple[str, ...]:
    if level == "l1":
        return (anchor.level1_label,)
    labels = tuple(label for label in anchor.level2_labels if label)
    return labels if labels else ("none",)


def _compress_multilabels(labels_by_anchor: list[tuple[str, ...]], max_categories: int) -> list[tuple[str, ...]]:
    if max_categories <= 0:
        return labels_by_anchor

    counts: dict[str, int] = {}
    for labels in labels_by_anchor:
        for label in labels:
            counts[label] = counts.get(label, 0) + 1

    keep = {
        label
        for label, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:max_categories]
    }

    compressed: list[tuple[str, ...]] = []
    for labels in labels_by_anchor:
        mapped: list[str] = []
        for label in labels:
            mapped.append(label if label in keep else "OTHER")
        deduped = tuple(dict.fromkeys(mapped))
        compressed.append(deduped if deduped else ("none",))
    return compressed


def _scatter_by_multilabel(
    ax: Any,
    coords: np.ndarray,
    labels_by_anchor: list[tuple[str, ...]],
    *,
    point_size: float = 12.0,
    alpha: float = 0.58,
) -> None:
    unique_labels = sorted({label for labels in labels_by_anchor for label in labels})
    for label in unique_labels:
        idx = [i for i, labels in enumerate(labels_by_anchor) if label in labels]
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
    anchors: list[Anchor],
    pca_features_by_teacher: dict[str, dict[str, np.ndarray]],
    label_levels: list[str],
    max_categories: int,
    formats: list[str],
    written: list[str],
    verbose: bool,
) -> None:
    if not anchors or not pca_features_by_teacher:
        return

    teacher_names = sorted(pca_features_by_teacher)
    common_tile_ids = [
        anchor.tile_id
        for anchor in anchors
        if all(anchor.tile_id in pca_features_by_teacher[teacher] for teacher in teacher_names)
    ]
    if len(common_tile_ids) < 3:
        _log("PCA QC skipped: fewer than 3 common tiles across teachers", enabled=verbose)
        return

    anchors_by_id = {anchor.tile_id: anchor for anchor in anchors}
    common_anchors = [anchors_by_id[tile_id] for tile_id in common_tile_ids]

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
        raw_labels = [_anchor_labels(anchor, level) for anchor in common_anchors]
        labels_by_anchor = _compress_multilabels(raw_labels, max_categories)
        level_name = "L1" if level == "l1" else "L2"
        label_note = "single-label" if level == "l1" else "multi-label expanded; one tile may appear under multiple labels"

        _log(f"plot PCA QC: teacher-averaged {level_name}, tiles={len(common_anchors)}", enabled=verbose)
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        _scatter_by_multilabel(ax, average_coords, labels_by_anchor)
        ax.set_title(f"{level_name} QC PCA at max N={len(anchors)}: teacher-averaged ({label_note})")
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
            _scatter_by_multilabel(ax, teacher_coords[teacher], labels_by_anchor, point_size=10.0, alpha=0.52)
            ax.set_title(f"{teacher}")
            ax.set_xlabel("PCA-1")
            ax.set_ylabel("PCA-2")
        for ax in axes_arr[len(teacher_names) :]:
            ax.axis("off")
        handles, legend_labels = axes_arr[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, legend_labels, loc="outside lower center", ncols=min(4, max(1, len(legend_labels))), fontsize=7)
        fig.suptitle(f"{level_name} QC PCA at max N={len(anchors)}: individual teachers ({label_note})", y=1.02)
        for fmt in formats:
            path = figure_dir / f"infospace_pca_qc_{level}_by_teacher.{fmt}"
            fig.savefig(path, dpi=220)
            written.append(str(path))
        plt.close(fig)


def _plot_browser_matched_l1_umap_qc(
    *,
    plt: Any,
    figure_dir: Path,
    anchors: list[Anchor],
    pca_features_by_teacher: dict[str, dict[str, np.ndarray]],
    formats: list[str],
    written: list[str],
    verbose: bool,
) -> None:
    if not anchors or not pca_features_by_teacher:
        return

    teacher_names = sorted(pca_features_by_teacher)
    common_tile_ids = [
        anchor.tile_id
        for anchor in anchors
        if all(anchor.tile_id in pca_features_by_teacher[teacher] for teacher in teacher_names)
    ]
    if len(common_tile_ids) < 3:
        _log("browser-matched L1 UMAP QC skipped: fewer than 3 common tiles across teachers", enabled=verbose)
        return

    anchors_by_id = {anchor.tile_id: anchor for anchor in anchors}
    common_anchors = [anchors_by_id[tile_id] for tile_id in common_tile_ids]
    labels = [anchor.level1_label for anchor in common_anchors]
    label_names = sorted(set(labels))

    _log(
        "plot browser-matched L1 UMAP QC: "
        f"tiles={len(common_anchors)} n_neighbors={DEFAULT_BROWSER_UMAP_NEIGHBORS} "
        f"min_dist={DEFAULT_BROWSER_UMAP_MIN_DIST} random_state={DEFAULT_BROWSER_UMAP_RANDOM_STATE}",
        enabled=verbose,
    )
    try:
        fused = _browser_matched_fused_features(pca_features_by_teacher, teacher_names, common_tile_ids)
        coords = _browser_matched_umap_2d(fused)
    except RuntimeError as exc:
        _log(f"browser-matched L1 UMAP QC skipped: {exc}", enabled=verbose)
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
        "L1 QC UMAP at max N="
        f"{len(anchors)}: browser-matched fused teacher features\n"
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
    counts = [int(row["anchor_count"]) for row in aggregate_rows]
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
    counts = sorted({int(row["anchor_count"]) for row in rows})
    prototypes = sorted({str(row["prototype"]) for row in rows})
    if not counts or not prototypes:
        return

    def mean_for(count: int, prototype: str, key: str) -> float:
        return _mean([
            _as_float(row, key)
            for row in rows
            if int(row["anchor_count"]) == count and str(row["prototype"]) == prototype
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
    final_count = max(int(row["anchor_count"]) for row in rows)
    final_rows = [row for row in rows if int(row["anchor_count"]) == final_count]
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
    prototype_rows: list[dict[str, Any]],
    max_subset: list[Anchor],
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

    counts = [int(row["anchor_count"]) for row in aggregate_rows]
    elbow_counts = [int(row["anchor_count"]) for row in aggregate_rows if str(row.get("elbow_candidate", "")).lower() == "true"]
    plateau_counts = [int(row["anchor_count"]) for row in aggregate_rows if row["plateau_consensus"] == "true"]
    decision_count = elbow_counts[0] if elbow_counts else (plateau_counts[0] if plateau_counts else None)

    def save(fig: Any, stem: str) -> None:
        for fmt in formats:
            path = figure_dir / f"{stem}.{fmt}"
            fig.savefig(path, dpi=220)
            written.append(str(path))

    _log("plot summary curves", enabled=verbose)
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
    save(fig, "infospace_information_summary")
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
                    [int(row["anchor_count"]) for row in rows],
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
        save(fig, "infospace_information_teacher_audit")
        plt.close(fig)

    for level in (1, 2):
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
            anchors=max_subset,
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
            anchors=max_subset,
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
    subsets: dict[int, list[Anchor]],
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
    anchor_counts = _parse_int_list(args.anchor_counts)
    plot_formats = _parse_str_list(args.plot_formats) or _parse_str_list(DEFAULT_PLOT_FORMATS)
    pca_label_levels = [value.lower() for value in _parse_str_list(args.pca_label_levels)]

    _log("load anchors", enabled=verbose)
    generated_manifest_info: dict[str, Any] = {}
    if args.annotation_json:
        anchors = _anchors_from_annotation_json(Path(args.annotation_json))
        input_dir = output_root / "inputs"
        anchor_manifest = input_dir / "anchor_pool.csv"
        _write_anchor_manifest(anchor_manifest, anchors)
        generated_manifest_info = {
            "annotation_json": str(args.annotation_json),
            "generated_anchor_manifest": str(anchor_manifest),
            "anchor_count": len(anchors),
            "seed": seed,
        }
    else:
        if not args.anchor_manifest:
            raise ValueError("--anchor-manifest is required unless --annotation-json is used")
        anchor_manifest = Path(args.anchor_manifest)
        anchors = _load_anchors_from_manifest(anchor_manifest)

    _log(f"anchor pool loaded: n={len(anchors)}", enabled=verbose)
    subsets, skipped_counts = _nested_subsets(anchors, anchor_counts, seed, args.anchor_group_key)
    if skipped_counts:
        _log(f"skip requested counts > anchor_pool_count: {skipped_counts}", enabled=verbose)
    _log(f"available anchor counts: {sorted(subsets)}", enabled=verbose)

    level1_names, level2_names = _load_contract(Path(args.prototype_contract) if args.prototype_contract else None, anchors)
    _log(f"prototype labels: L1={len(level1_names)} L2={len(level2_names)}", enabled=verbose)

    teacher_paths = _resolve_teacher_paths(args)
    _log(f"teachers resolved: {list(teacher_paths)}", enabled=verbose)

    max_count = max(subsets)
    max_subset = subsets[max_count]
    needed_tile_ids = [anchor.tile_id for anchor in max_subset]

    report = {
        "sweep_type": "infospace_information_curve",
        "objective": "estimate decay of newly added tile novelty in teacher-feature infospace before main-model training",
        "anchor_manifest": str(anchor_manifest),
        "prototype_contract": str(args.prototype_contract or ""),
        "anchor_counts_requested": anchor_counts,
        "anchor_counts_available": sorted(subsets),
        "anchor_counts_skipped": skipped_counts,
        "nested_subsets": True,
        "validation_split_used": False,
        "coverage_metric_used": False,
        "does_not_train": True,
        "seed": seed,
        "anchor_group_key": args.anchor_group_key,
        "anchor_pool_count": len(anchors),
        "max_subset_count": len(max_subset),
        "infospace_topk": int(args.infospace_topk),
        "workers_requested": int(args.workers),
        "teachers": list(teacher_paths),
        "teacher_feature_packages": {teacher: [str(path) for path in paths] for teacher, paths in teacher_paths.items()},
        "level1_prototypes": level1_names,
        "level2_prototypes": level2_names,
        "generated_manifests": generated_manifest_info,
    }

    _log("write nested subset manifests", enabled=verbose)
    subset_rows = []
    for count, subset in sorted(subsets.items()):
        subset_path = output_root / f"N{count}" / "anchors.csv"
        _write_csv(
            subset_path,
            [
                {
                    "tile_id": anchor.tile_id,
                    "level1_label": anchor.level1_label,
                    "level2_labels": ";".join(anchor.level2_labels),
                    "slide_id": anchor.slide_id,
                    "patient_id": anchor.patient_id,
                    "center": anchor.center,
                }
                for anchor in subset
            ],
        )
        subset_rows.append({"anchor_count": count, "anchor_count_actual": len(subset), "path": str(subset_path)})
    _write_csv(output_root / "nested_subsets.csv", subset_rows)

    teacher_rows: list[dict[str, Any]] = []
    prototype_rows: list[dict[str, Any]] = []
    pca_features_by_teacher: dict[str, dict[str, np.ndarray]] = {}

    requested_workers = int(args.workers)
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
            "bootstrap_iterations": int(args.bootstrap_iterations),
            "seed": seed,
            "topk": int(args.infospace_topk),
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
        plateau_novelty_threshold=float(args.plateau_novelty_threshold),
        plateau_drift_threshold=float(args.plateau_drift_threshold),
        plateau_redundancy_threshold=float(args.plateau_redundancy_threshold),
    )
    recommendation = _recommendation(aggregate_rows)

    _log("write CSV/JSON outputs", enabled=verbose)
    _write_csv(output_root / "infospace_information_by_teacher.csv", teacher_rows)
    _write_csv(output_root / "infospace_information_by_prototype.csv", prototype_rows)
    _write_csv(output_root / "infospace_information_summary.csv", aggregate_rows)

    figure_paths: list[str] = []
    if not bool(args.no_plots):
        _log("render figures", enabled=verbose)
        figure_paths = _plot_curves(
            output_root=output_root,
            teacher_rows=teacher_rows,
            aggregate_rows=aggregate_rows,
            prototype_rows=prototype_rows,
            max_subset=max_subset,
            pca_features_by_teacher=pca_features_by_teacher,
            pca_label_levels=pca_label_levels,
            max_pca_categories=int(args.max_pca_categories),
            formats=plot_formats,
            no_pca=bool(args.no_pca),
            verbose=verbose,
        )

    report.update({"recommendation": recommendation, "figures": figure_paths})
    _write_json(output_root / "infospace_information_report.json", report)

    _log(
        "done "
        f"output_root={output_root} teachers={len(teacher_paths)} counts={len(subsets)} "
        f"recommended_anchor_count={recommendation['recommended_anchor_count']} figures={len(figure_paths)}",
        enabled=verbose,
    )
    return {"teacher": teacher_rows, "prototype": prototype_rows, "summary": aggregate_rows, "recommendation": recommendation}


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
    parser.add_argument("--anchor-manifest", default="")
    parser.add_argument("--prototype-contract", default="")
    parser.add_argument("--output-root", default="outputs/infospace_information_curve")
    parser.add_argument("--anchor-counts", default=DEFAULT_ANCHOR_COUNTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--anchor-group-key", default=DEFAULT_ANCHOR_GROUP_KEY)
    parser.add_argument("--infospace-topk", type=int, default=DEFAULT_INFOSPACE_TOPK)
    parser.add_argument("--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS)
    parser.add_argument("--plateau-novelty-threshold", type=float, default=DEFAULT_PLATEAU_NOVELTY_THRESHOLD)
    parser.add_argument("--plateau-drift-threshold", type=float, default=DEFAULT_PLATEAU_DRIFT_THRESHOLD)
    parser.add_argument("--plateau-redundancy-threshold", type=float, default=DEFAULT_PLATEAU_REDUNDANCY_THRESHOLD)
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
