from __future__ import annotations

import argparse
import csv
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

DEFAULT_PLOT_FORMATS = ("png", "pdf")
DEFAULT_ANCHOR_COUNTS = "100,200,400,800,1200,1600,2000,3000"
DEFAULT_SEED = 13
DEFAULT_LOCKED_VAL_FRACTION = 0.2
DEFAULT_LOCKED_VAL_COUNT = 0
DEFAULT_ANCHOR_GROUP_KEY = "tile_id"
DEFAULT_BOOTSTRAP_ITERATIONS = 500
DEFAULT_PLATEAU_DELTA_EPSILON = 0.005
DEFAULT_PLATEAU_DRIFT_THRESHOLD = 0.01
DEFAULT_PLATEAU_REDUNDANCY_THRESHOLD = 0.95
OBSOLETE_FIGURE_STEMS = (
)
OBSOLETE_JSON_OUTPUTS = ("anchor_information_plan.json", "anchor_information_recommendation.json")


def _canonical_teacher_name(name: str) -> str:
    value = str(name).strip()
    return CANONICAL_TEACHER_BY_ALIAS.get(value, value)


def _teacher_aliases(name: str) -> tuple[str, ...]:
    canonical = _canonical_teacher_name(name)
    return TEACHER_ALIASES.get(canonical, (canonical,))


def _parse_int_list(value: str) -> list[int]:
    counts = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not counts:
        raise ValueError("expected at least one anchor count")
    if any(count <= 0 for count in counts):
        raise ValueError(f"anchor counts must be positive: {counts}")
    return sorted(dict.fromkeys(counts))


def _parse_str_list(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one value")
    return values


def _parse_optional_str_list(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _split_labels(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in str(value).replace("|", ";").split(";") if item.strip())


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_anchor_manifest(path: Path, anchors: list[Anchor], source_split: str) -> None:
    rows = [
        {
            "tile_id": anchor.tile_id,
            "level1_label": anchor.level1_label,
            "level2_labels": ";".join(anchor.level2_labels),
            "slide_id": anchor.slide_id,
            "patient_id": anchor.patient_id,
            "center": anchor.center,
            "source_split": source_split,
        }
        for anchor in anchors
    ]
    _write_csv(path, rows)


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
        center=str(row.get("center") or "").strip(),
    )


def _anchors_from_annotation_json(path: Path) -> list[Anchor]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    annotations = payload.get("annotations")
    if not isinstance(annotations, dict):
        raise ValueError(f"annotation JSON missing annotations object: {path}")
    anchors = []
    for item in annotations.values():
        tile_id = str(item.get("tile_id", "")).strip()
        level1 = str(item.get("l1") or item.get("level1_label") or "").strip()
        if not tile_id or not level1:
            continue
        slide_id = str(item.get("slide") or item.get("slide_id") or tile_id).strip()
        dataset = str(item.get("dataset") or "").strip()
        anchors.append(
            Anchor(
                tile_id=tile_id,
                level1_label=level1,
                level2_labels=tuple(str(label).strip() for label in item.get("l2", []) if str(label).strip()),
                slide_id=slide_id,
                patient_id=str(item.get("patient_id") or slide_id).strip(),
                center=dataset,
            )
        )
    if not anchors:
        raise ValueError(f"annotation JSON has no usable anchors: {path}")
    return anchors


def _stable_score(anchor: Anchor, seed: int) -> float:
    payload = f"{seed}|{anchor.level1_label}|{anchor.slide_id}|{anchor.tile_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _split_anchors(
    anchors: list[Anchor],
    *,
    locked_val_fraction: float,
    locked_val_count: int,
    seed: int,
) -> tuple[list[Anchor], list[Anchor]]:
    if locked_val_count < 0:
        raise ValueError("--locked-val-count must be non-negative")
    if not 0.0 <= locked_val_fraction < 1.0:
        raise ValueError("--locked-val-fraction must be in [0, 1)")
    groups: dict[str, list[Anchor]] = {}
    for anchor in anchors:
        groups.setdefault(anchor.level1_label, []).append(anchor)
    locked_val: list[Anchor] = []
    if locked_val_count > 0:
        ranked = sorted(anchors, key=lambda anchor: _stable_score(anchor, seed))
        locked_val = ranked[: min(locked_val_count, max(0, len(anchors) - 1))]
    else:
        for label, rows in sorted(groups.items()):
            rows = sorted(rows, key=lambda anchor: _stable_score(anchor, seed))
            count = int(round(len(rows) * locked_val_fraction))
            if len(rows) >= 5 and locked_val_fraction > 0:
                count = max(1, count)
            count = min(count, max(0, len(rows) - 1))
            locked_val.extend(rows[:count])
    locked_ids = {anchor.tile_id for anchor in locked_val}
    train = [anchor for anchor in anchors if anchor.tile_id not in locked_ids]
    return train, locked_val


def _load_anchors(path: Path, *, exclude_tile_ids: set[str] | None = None) -> list[Anchor]:
    exclude_tile_ids = exclude_tile_ids or set()
    anchors = []
    for row in _read_csv(path):
        tile_id = str(row.get("tile_id", "")).strip()
        source_split = str(row.get("source_split") or row.get("split") or "").strip().lower()
        if tile_id in exclude_tile_ids or source_split == "val":
            continue
        anchors.append(_normalize_anchor(row))
    return anchors


def _load_locked_val(path: Path) -> list[Anchor]:
    return [_normalize_anchor(row) for row in _read_csv(path)]


def _load_candidate_tiles(path: Path | None, locked_val: list[Anchor]) -> list[str]:
    if path is None:
        return [anchor.tile_id for anchor in locked_val]
    rows = _read_csv(path)
    tile_ids = [str(row.get("tile_id", "")).strip() for row in rows]
    tile_ids = [tile_id for tile_id in tile_ids if tile_id]
    if not tile_ids:
        raise ValueError(f"candidate manifest has no tile_id values: {path}")
    return tile_ids


def _load_contract(path: Path | None, anchors: list[Anchor], locked_val: list[Anchor]) -> tuple[list[str], list[str]]:
    if path is None:
        level1: list[str] = []
        level2: list[str] = []
        for anchor in [*anchors, *locked_val]:
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


def _nested_subsets(anchors: list[Anchor], counts: list[int], seed: int, group_key: str) -> dict[int, list[Anchor]]:
    available_counts = [count for count in counts if count <= len(anchors)]
    if not available_counts:
        raise ValueError(f"all requested counts exceed available anchors={len(anchors)}")
    rng = random.Random(seed)
    groups: dict[str, list[Anchor]] = {}
    for anchor in anchors:
        key = getattr(anchor, group_key) if group_key in {"tile_id", "slide_id", "patient_id", "center"} else anchor.tile_id
        groups.setdefault(str(key or anchor.tile_id), []).append(anchor)
    group_keys = sorted(groups)
    rng.shuffle(group_keys)
    ordered: list[Anchor] = []
    for key in group_keys:
        items = list(groups[key])
        rng.shuffle(items)
        ordered.extend(items)
    return {count: ordered[:count] for count in available_counts}


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    denom = np.linalg.norm(matrix, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-8)
    return matrix / denom


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-8)
    return float(np.dot(a, b) / denom)


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(sum(finite) / len(finite)) if finite else math.nan


def _std(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
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
    result = {}
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
    if root is None or not root.exists():
        return []
    discovered: list[str] = []
    if root.is_file():
        return discovered
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
    requested = _parse_optional_str_list(getattr(args, "teachers", ""))
    if requested:
        teachers = list(dict.fromkeys(_canonical_teacher_name(teacher) for teacher in requested))
    elif explicit:
        teachers = list(explicit)
    else:
        teachers = _discover_teachers_from_root(root)
    if not teachers:
        raise ValueError("--teacher-feature-root must contain recognizable teacher directories or packages")
    paths = _discover_teacher_paths(root, teachers)
    paths.update(explicit)
    missing = [teacher for teacher in teachers if teacher not in paths]
    if missing:
        raise ValueError(f"missing feature packages for teachers: {missing}")
    return {teacher: paths[teacher] for teacher in teachers}


def _feature_map(store: FeatureStore, teacher: str, tile_ids: list[str]) -> dict[str, np.ndarray]:
    result = {}
    for tile_id in dict.fromkeys(tile_ids):
        result[tile_id] = store.read(teacher, tile_id)
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


def _coverage_values(query_features: list[np.ndarray], anchor_features: list[np.ndarray]) -> np.ndarray:
    if not query_features or not anchor_features:
        return np.asarray([], dtype=np.float32)
    query = _normalize_rows(np.stack(query_features))
    anchors = _normalize_rows(np.stack(anchor_features))
    return (query @ anchors.T).max(axis=1).astype(np.float32)


def _center_for(centers: Centers, level: int, name: str) -> np.ndarray | None:
    return centers.level1.get(name) if level == 1 else centers.level2.get(name)


def _center_count(centers: Centers, level: int, name: str) -> int:
    return centers.level1_counts.get(name, 0) if level == 1 else centers.level2_counts.get(name, 0)


def _anchor_has_prototype(anchor: Anchor, level: int, name: str) -> bool:
    if level == 1:
        return anchor.level1_label == name
    return name in anchor.level2_labels


def _prototype_drift_values(current: Centers, previous: Centers | None) -> np.ndarray:
    if previous is None:
        return np.asarray([], dtype=np.float32)
    values = []
    for name, center in current.level1.items():
        if name in previous.level1:
            values.append(1.0 - _cosine(center, previous.level1[name]))
    for name, center in current.level2.items():
        if name in previous.level2:
            values.append(1.0 - _cosine(center, previous.level2[name]))
    return np.asarray(values, dtype=np.float32)


def _l1_agreement_values(locked_val: list[Anchor], features: dict[str, np.ndarray], centers: Centers) -> np.ndarray:
    if not centers.level1:
        return np.asarray([], dtype=np.float32)
    names = list(centers.level1)
    center_matrix = _normalize_rows(np.stack([centers.level1[name] for name in names]))
    values = []
    for anchor in locked_val:
        if anchor.level1_label not in centers.level1:
            continue
        feature = _normalize_rows(np.expand_dims(features[anchor.tile_id], axis=0))
        predicted = names[int((feature @ center_matrix.T).argmax(axis=1)[0])]
        values.append(1.0 if predicted == anchor.level1_label else 0.0)
    return np.asarray(values, dtype=np.float32)


def _row_auc(scores: dict[str, float], positives: set[str]) -> float | None:
    negatives = [label for label in scores if label not in positives]
    positives = {label for label in positives if label in scores}
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0
    for positive in positives:
        for negative in negatives:
            total += 1
            if scores[positive] > scores[negative]:
                wins += 1.0
            elif scores[positive] == scores[negative]:
                wins += 0.5
    return wins / total if total else None


def _l2_agreement_values(locked_val: list[Anchor], features: dict[str, np.ndarray], centers: Centers) -> np.ndarray:
    if not centers.level2:
        return np.asarray([], dtype=np.float32)
    names = list(centers.level2)
    center_matrix = _normalize_rows(np.stack([centers.level2[name] for name in names]))
    values = []
    for anchor in locked_val:
        feature = _normalize_rows(np.expand_dims(features[anchor.tile_id], axis=0))
        sims = (feature @ center_matrix.T).reshape(-1)
        auc = _row_auc(dict(zip(names, (float(value) for value in sims))), set(anchor.level2_labels))
        if auc is not None:
            values.append(auc)
    return np.asarray(values, dtype=np.float32)


def _binary_auc(scores: list[float], labels: list[int]) -> float | None:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0
    for positive in positives:
        for negative in negatives:
            total += 1
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / total if total else None


def _l1_prediction(anchor: Anchor, features: dict[str, np.ndarray], centers: Centers) -> str | None:
    if not centers.level1:
        return None
    names = list(centers.level1)
    center_matrix = _normalize_rows(np.stack([centers.level1[name] for name in names]))
    feature = _normalize_rows(np.expand_dims(features[anchor.tile_id], axis=0))
    return names[int((feature @ center_matrix.T).argmax(axis=1)[0])]


def _prototype_rows(
    *,
    teacher: str,
    anchor_count: int,
    anchors: list[Anchor],
    previous_anchors: list[Anchor] | None,
    locked_val: list[Anchor],
    features: dict[str, np.ndarray],
    centers: Centers,
    previous_centers: Centers | None,
    level1_names: list[str],
    level2_names: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for level, names in [(1, level1_names), (2, level2_names)]:
        for name in names:
            center = _center_for(centers, level, name)
            previous_center = _center_for(previous_centers, level, name) if previous_centers is not None else None
            matching_val = [anchor for anchor in locked_val if _anchor_has_prototype(anchor, level, name)]
            matching_anchor_features = [
                features[anchor.tile_id] for anchor in anchors if _anchor_has_prototype(anchor, level, name)
            ]
            matching_val_features = [features[anchor.tile_id] for anchor in matching_val]
            coverage_values = _coverage_values(matching_val_features, matching_anchor_features)
            drift = 1.0 - _cosine(center, previous_center) if center is not None and previous_center is not None else math.nan
            if previous_anchors is None:
                redundancy_values = np.asarray([], dtype=np.float32)
            else:
                new_anchor_features = [
                    features[anchor.tile_id]
                    for anchor in anchors[len(previous_anchors) :]
                    if _anchor_has_prototype(anchor, level, name)
                ]
                previous_anchor_features = [
                    features[anchor.tile_id] for anchor in previous_anchors if _anchor_has_prototype(anchor, level, name)
                ]
                redundancy_values = _coverage_values(new_anchor_features, previous_anchor_features)
            if center is None:
                agreement = math.nan
            elif level == 1:
                l1_values = [
                    1.0 if _l1_prediction(anchor, features, centers) == name else 0.0
                    for anchor in matching_val
                ]
                agreement = _mean(l1_values)
            else:
                scores = []
                labels = []
                for anchor in locked_val:
                    scores.append(_cosine(features[anchor.tile_id], center))
                    labels.append(1 if name in anchor.level2_labels else 0)
                auc = _binary_auc(scores, labels)
                agreement = math.nan if auc is None else float(auc)
            rows.append(
                {
                    "anchor_count": anchor_count,
                    "teacher": teacher,
                    "level": level,
                    "prototype": name,
                    "train_anchor_count": _center_count(centers, level, name),
                    "locked_val_count": len(matching_val),
                    "center_available": str(center is not None).lower(),
                    "val_coverage": _format_float(float(coverage_values.mean()) if coverage_values.size else math.nan),
                    "val_agreement": _format_float(agreement),
                    "prototype_drift": _format_float(drift),
                    "redundancy": _format_float(float(redundancy_values.mean()) if redundancy_values.size else math.nan),
                }
            )
    return rows


def _format_float(value: float) -> float | str:
    return "" if not math.isfinite(float(value)) else float(value)


def _teacher_curve(
    *,
    teacher: str,
    subsets: dict[int, list[Anchor]],
    locked_val: list[Anchor],
    candidate_tiles: list[str],
    features: dict[str, np.ndarray],
    level1_names: list[str],
    level2_names: list[str],
    bootstrap_iterations: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    prototype_rows = []
    previous: dict[str, Any] | None = None
    rng = np.random.default_rng(seed)
    for count in sorted(subsets):
        anchors = subsets[count]
        centers = _centers(anchors, features, level1_names, level2_names)
        anchor_vectors = [features[anchor.tile_id] for anchor in anchors]
        candidate_vectors = [features[tile_id] for tile_id in candidate_tiles if tile_id in features]
        coverage_values = _coverage_values(candidate_vectors, anchor_vectors)
        l1_values = _l1_agreement_values(locked_val, features, centers)
        l2_values = _l2_agreement_values(locked_val, features, centers)
        val_values = np.concatenate([l1_values, l2_values]) if l1_values.size or l2_values.size else np.asarray([])
        drift_values = _prototype_drift_values(centers, previous["centers"] if previous else None)
        if previous is None:
            redundancy_values = np.asarray([], dtype=np.float32)
            delta_coverage_values = np.asarray([], dtype=np.float32)
            delta_val_values = np.asarray([], dtype=np.float32)
        else:
            previous_anchors = previous["anchors"]
            new_anchors = anchors[len(previous_anchors) :]
            redundancy_values = _coverage_values(
                [features[anchor.tile_id] for anchor in new_anchors],
                [features[anchor.tile_id] for anchor in previous_anchors],
            )
            delta_coverage_values = coverage_values - previous["coverage_values"]
            if val_values.size and previous["val_values"].size == val_values.size:
                delta_val_values = val_values - previous["val_values"]
            else:
                delta_val_values = np.asarray([], dtype=np.float32)
        coverage_ci = _bootstrap_ci(delta_coverage_values, rng, bootstrap_iterations)
        val_ci = _bootstrap_ci(delta_val_values, rng, bootstrap_iterations)
        missing_l1 = [name for name in level1_names if name not in centers.level1]
        missing_l2 = [name for name in level2_names if name not in centers.level2]
        row = {
            "anchor_count": count,
            "teacher": teacher,
            "coverage": _format_float(float(coverage_values.mean()) if coverage_values.size else math.nan),
            "delta_coverage": _format_float(float(delta_coverage_values.mean()) if delta_coverage_values.size else math.nan),
            "delta_coverage_ci_low": _format_float(coverage_ci[0]),
            "delta_coverage_ci_high": _format_float(coverage_ci[1]),
            "val_agreement_l1": _format_float(float(l1_values.mean()) if l1_values.size else math.nan),
            "val_agreement_l2": _format_float(float(l2_values.mean()) if l2_values.size else math.nan),
            "val_agreement": _format_float(float(val_values.mean()) if val_values.size else math.nan),
            "delta_val_agreement": _format_float(float(delta_val_values.mean()) if delta_val_values.size else math.nan),
            "delta_val_agreement_ci_low": _format_float(val_ci[0]),
            "delta_val_agreement_ci_high": _format_float(val_ci[1]),
            "prototype_drift": _format_float(float(drift_values.mean()) if drift_values.size else math.nan),
            "redundancy": _format_float(float(redundancy_values.mean()) if redundancy_values.size else math.nan),
            "available_l1_centers": len(centers.level1),
            "available_l2_centers": len(centers.level2),
            "missing_l1_centers": ";".join(missing_l1),
            "missing_l2_centers": ";".join(missing_l2),
            "candidate_count": int(coverage_values.size),
            "locked_val_l1_count": int(l1_values.size),
            "locked_val_l2_count": int(l2_values.size),
        }
        rows.append(row)
        prototype_rows.extend(
            _prototype_rows(
                teacher=teacher,
                anchor_count=count,
                anchors=anchors,
                previous_anchors=previous["anchors"] if previous else None,
                locked_val=locked_val,
                features=features,
                centers=centers,
                previous_centers=previous["centers"] if previous else None,
                level1_names=level1_names,
                level2_names=level2_names,
            )
        )
        previous = {
            "anchors": anchors,
            "centers": centers,
            "coverage_values": coverage_values,
            "val_values": val_values,
        }
    return rows, prototype_rows


def _as_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return math.nan
    return float(value)


def _aggregate_rows(
    teacher_rows: list[dict[str, Any]],
    *,
    plateau_delta_epsilon: float,
    plateau_drift_threshold: float,
    plateau_redundancy_threshold: float,
) -> list[dict[str, Any]]:
    result = []
    counts = sorted({int(row["anchor_count"]) for row in teacher_rows})
    consecutive_plateau = 0
    for count in counts:
        rows = [row for row in teacher_rows if int(row["anchor_count"]) == count]
        delta_coverage_ci_low = _mean([_as_float(row, "delta_coverage_ci_low") for row in rows])
        delta_coverage_ci_high = _mean([_as_float(row, "delta_coverage_ci_high") for row in rows])
        delta_val_ci_low = _mean([_as_float(row, "delta_val_agreement_ci_low") for row in rows])
        delta_val_ci_high = _mean([_as_float(row, "delta_val_agreement_ci_high") for row in rows])
        drift = _mean([_as_float(row, "prototype_drift") for row in rows])
        redundancy = _mean([_as_float(row, "redundancy") for row in rows])
        delta_coverage = _mean([_as_float(row, "delta_coverage") for row in rows])
        delta_val = _mean([_as_float(row, "delta_val_agreement") for row in rows])
        coverage_noise = (
            math.isfinite(delta_coverage_ci_low)
            and math.isfinite(delta_coverage_ci_high)
            and delta_coverage_ci_low <= 0.0 <= delta_coverage_ci_high
        ) or (math.isfinite(delta_coverage) and abs(delta_coverage) <= plateau_delta_epsilon)
        val_noise = (
            math.isfinite(delta_val_ci_low)
            and math.isfinite(delta_val_ci_high)
            and delta_val_ci_low <= 0.0 <= delta_val_ci_high
        ) or (math.isfinite(delta_val) and abs(delta_val) <= plateau_delta_epsilon)
        low_drift = math.isfinite(drift) and drift <= plateau_drift_threshold
        high_redundancy = math.isfinite(redundancy) and redundancy >= plateau_redundancy_threshold
        interval_plateau = coverage_noise and val_noise and low_drift and high_redundancy
        consecutive_plateau = consecutive_plateau + 1 if interval_plateau else 0
        result.append(
            {
                "anchor_count": count,
                "teacher_count": len(rows),
                "coverage_mean": _format_float(_mean([_as_float(row, "coverage") for row in rows])),
                "coverage_std": _format_float(_std([_as_float(row, "coverage") for row in rows])),
                "delta_coverage_mean": _format_float(delta_coverage),
                "delta_coverage_ci_low_mean": _format_float(delta_coverage_ci_low),
                "delta_coverage_ci_high_mean": _format_float(delta_coverage_ci_high),
                "val_agreement_l1_mean": _format_float(_mean([_as_float(row, "val_agreement_l1") for row in rows])),
                "val_agreement_l2_mean": _format_float(_mean([_as_float(row, "val_agreement_l2") for row in rows])),
                "val_agreement_mean": _format_float(_mean([_as_float(row, "val_agreement") for row in rows])),
                "delta_val_agreement_mean": _format_float(delta_val),
                "delta_val_agreement_ci_low_mean": _format_float(delta_val_ci_low),
                "delta_val_agreement_ci_high_mean": _format_float(delta_val_ci_high),
                "prototype_drift_mean": _format_float(drift),
                "redundancy_mean": _format_float(redundancy),
                "coverage_noise": str(coverage_noise).lower(),
                "val_agreement_noise": str(val_noise).lower(),
                "low_drift": str(low_drift).lower(),
                "high_redundancy": str(high_redundancy).lower(),
                "plateau_interval": str(interval_plateau).lower(),
                "plateau_consensus": str(consecutive_plateau >= 2).lower(),
            }
        )
    return result


def _recommendation(aggregate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    plateau_counts = [int(row["anchor_count"]) for row in aggregate_rows if row["plateau_consensus"] == "true"]
    available_counts = [int(row["anchor_count"]) for row in aggregate_rows]
    if plateau_counts:
        return {"recommended_anchor_count": plateau_counts[0], "reason": "first consecutive plateau upper bound"}
    return {
        "recommended_anchor_count": available_counts[-1] if available_counts else None,
        "reason": "no plateau detected within available anchors",
    }


def _finite_or_nan(values: list[float]) -> list[float]:
    return [value if math.isfinite(value) else math.nan for value in values]


def _plot_ci_band(
    ax: Any,
    x: list[int],
    low: list[float],
    high: list[float],
    *,
    color: str | None = None,
    alpha: float = 0.18,
) -> None:
    low_array = np.asarray(_finite_or_nan(low), dtype=np.float32)
    high_array = np.asarray(_finite_or_nan(high), dtype=np.float32)
    if np.isfinite(low_array).any() and np.isfinite(high_array).any():
        ax.fill_between(x, low_array, high_array, color=color, alpha=alpha, linewidth=0)


def _mark_plateau(ax: Any, aggregate_rows: list[dict[str, Any]]) -> None:
    plateau_counts = [int(row["anchor_count"]) for row in aggregate_rows if row["plateau_consensus"] == "true"]
    if not plateau_counts:
        return
    count = plateau_counts[0]
    ax.axvline(count, color="#111827", linestyle="--", linewidth=0.9, alpha=0.65)
    ax.text(
        count,
        0.98,
        f"plateau N={count}",
        transform=ax.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=7,
        color="#111827",
        rotation=90,
    )


def _save_figure(fig: Any, figure_dir: Path, stem: str, formats: list[str], written: list[str]) -> None:
    for fmt in formats:
        path = figure_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=220)
        written.append(str(path))


def _remove_obsolete_figures(figure_dir: Path, formats: list[str]) -> None:
    for stem in OBSOLETE_FIGURE_STEMS:
        for fmt in formats:
            path = figure_dir / f"{stem}.{fmt}"
            if path.exists():
                path.unlink()


def _format_cell(value: float, *, count: bool = False) -> str:
    if not math.isfinite(value):
        return "NA"
    if count:
        return str(int(round(value)))
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3f}"


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(matrix, dtype=np.float32)
    for col in range(matrix.shape[1]):
        values = matrix[:, col]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            normalized[:, col] = np.nan
            continue
        low = float(finite.min())
        high = float(finite.max())
        if high <= low:
            normalized[:, col] = 0.5
        else:
            normalized[:, col] = (values - low) / (high - low)
    return normalized


def _prototype_audit_matrix(
    prototype_rows: list[dict[str, Any]],
    *,
    level: int,
) -> tuple[list[str], list[str], np.ndarray, list[list[str]]] | None:
    rows = [row for row in prototype_rows if int(row["level"]) == level]
    if not rows:
        return None
    final_count = max(int(row["anchor_count"]) for row in rows)
    final_rows = [row for row in rows if int(row["anchor_count"]) == final_count]
    prototypes = sorted({row["prototype"] for row in final_rows})
    if not prototypes:
        return None

    columns = ["train n", "val n", "center T", "val coverage", "val agreement", "drift", "redundancy"]
    matrix_rows: list[list[float]] = []
    text_rows: list[list[str]] = []
    for prototype in prototypes:
        items = [row for row in final_rows if row["prototype"] == prototype]
        teacher_count = len(items)
        center_count = sum(1 for row in items if str(row.get("center_available", "")).lower() == "true")
        values = [
            _mean([_as_float(row, "train_anchor_count") for row in items]),
            _mean([_as_float(row, "locked_val_count") for row in items]),
            float(center_count) / float(teacher_count) if teacher_count else math.nan,
            _mean([_as_float(row, "val_coverage") for row in items]),
            _mean([_as_float(row, "val_agreement") for row in items]),
            _mean([_as_float(row, "prototype_drift") for row in items]),
            _mean([_as_float(row, "redundancy") for row in items]),
        ]
        matrix_rows.append(values)
        text_rows.append(
            [
                _format_cell(values[0], count=True),
                _format_cell(values[1], count=True),
                f"{center_count}/{teacher_count}",
                _format_cell(values[3]),
                _format_cell(values[4]),
                _format_cell(values[5]),
                _format_cell(values[6]),
            ]
        )
    return prototypes, columns, np.asarray(matrix_rows, dtype=np.float32), text_rows


def _plot_prototype_audit_heatmap(
    *,
    plt: Any,
    figure_dir: Path,
    prototype_rows: list[dict[str, Any]],
    formats: list[str],
    written: list[str],
) -> None:
    for level, suffix in [(1, "level1"), (2, "level2")]:
        payload = _prototype_audit_matrix(prototype_rows, level=level)
        if payload is None:
            continue
        prototypes, columns, matrix, text_rows = payload
        height = max(4.5, 0.42 * len(prototypes) + 1.8)
        fig, ax = plt.subplots(figsize=(12, height), constrained_layout=True)
        image = ax.imshow(_normalize_columns(matrix), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"Prototype audit at final anchor count: {suffix}")
        ax.set_xticks(range(len(columns)), columns, rotation=30, ha="right")
        ax.set_yticks(range(len(prototypes)), prototypes)
        ax.tick_params(axis="both", labelsize=8)
        color_matrix = _normalize_columns(matrix)
        for y, row in enumerate(text_rows):
            for x, text in enumerate(row):
                color_value = color_matrix[y, x]
                text_color = "white" if not math.isfinite(float(color_value)) or float(color_value) < 0.58 else "#111827"
                ax.text(x, y, text, ha="center", va="center", color=text_color, fontsize=7, fontweight="bold")
        ax.set_xlabel("Auditable criterion")
        ax.set_ylabel("Prototype")
        cbar = fig.colorbar(image, ax=ax, shrink=0.75)
        cbar.set_label("Column-normalized intensity")
        _save_figure(fig, figure_dir, f"anchor_information_{suffix}_prototype_audit", formats, written)
        plt.close(fig)


def _plot_curves(
    *,
    output_root: Path,
    teacher_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    prototype_rows: list[dict[str, Any]],
    formats: list[str],
) -> list[str]:
    os.environ.setdefault("MPLCONFIGDIR", str(output_root / ".matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to render anchor information audit figures. "
            "Install it in the active hcc-sempath environment or run with --no-plots."
        ) from exc
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    _remove_obsolete_figures(figure_dir, formats)
    written: list[str] = []

    counts = [int(row["anchor_count"]) for row in aggregate_rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    panels = [
        (
            "delta_coverage_mean",
            "delta_coverage_ci_low_mean",
            "delta_coverage_ci_high_mean",
            "Delta coverage with 95% CI",
            axes[0, 0],
        ),
        (
            "delta_val_agreement_mean",
            "delta_val_agreement_ci_low_mean",
            "delta_val_agreement_ci_high_mean",
            "Delta validation agreement with 95% CI",
            axes[0, 1],
        ),
        ("prototype_drift_mean", "Prototype drift", axes[1, 0]),
        ("redundancy_mean", "Redundancy", axes[1, 1]),
    ]
    for panel in panels:
        if len(panel) == 5:
            key, low_key, high_key, title, ax = panel
        else:
            key, title, ax = panel
            low_key = high_key = ""
        values = [_as_float(row, key) for row in aggregate_rows]
        (line,) = ax.plot(counts, values, marker="o", linewidth=1.8)
        if low_key and high_key:
            _plot_ci_band(
                ax,
                counts,
                [_as_float(row, low_key) for row in aggregate_rows],
                [_as_float(row, high_key) for row in aggregate_rows],
                color=line.get_color(),
            )
            ax.axhline(0.0, color="#6b7280", linestyle=":", linewidth=0.9)
        _mark_plateau(ax, aggregate_rows)
        ax.set_title(title)
        ax.set_xlabel("Anchor count")
        ax.grid(True, linewidth=0.5, alpha=0.35)
    _save_figure(fig, figure_dir, "anchor_information_summary", formats, written)
    plt.close(fig)

    for level, suffix in [(1, "level1"), (2, "level2")]:
        rows = [row for row in prototype_rows if int(row["level"]) == level]
        prototypes = sorted({row["prototype"] for row in rows})
        if not rows or not prototypes:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
        for prototype in prototypes:
            series = [row for row in rows if row["prototype"] == prototype]
            x = [int(row["anchor_count"]) for row in series]
            agreement = [_as_float(row, "val_agreement") for row in series]
            drift = [_as_float(row, "prototype_drift") for row in series]
            axes[0].plot(x, agreement, marker="o", linewidth=1.2, label=prototype)
            axes[1].plot(x, drift, marker="o", linewidth=1.2, label=prototype)
        axes[0].set_title(f"{suffix} validation agreement by prototype")
        axes[1].set_title(f"{suffix} prototype drift by prototype")
        for ax in axes:
            ax.set_xlabel("Anchor count")
            ax.grid(True, linewidth=0.5, alpha=0.35)
        axes[0].legend(fontsize=7, ncols=2)
        axes[1].legend(fontsize=7, ncols=2)
        _save_figure(fig, figure_dir, f"anchor_information_{suffix}_prototypes", formats, written)
        plt.close(fig)

    teacher_names = sorted({row["teacher"] for row in teacher_rows})
    if len(teacher_names) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        panels = [
            ("coverage", "Coverage by teacher", axes[0, 0]),
            ("val_agreement", "Validation agreement by teacher", axes[0, 1]),
            ("prototype_drift", "Prototype drift by teacher", axes[1, 0]),
            ("redundancy", "Redundancy by teacher", axes[1, 1]),
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
            ax.set_title(title)
            ax.set_xlabel("Anchor count")
            ax.grid(True, linewidth=0.5, alpha=0.35)
        axes[0, 0].legend(fontsize=8)
        _save_figure(fig, figure_dir, "anchor_information_teacher_audit", formats, written)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        for teacher in teacher_names:
            rows = [row for row in teacher_rows if row["teacher"] == teacher]
            x = [int(row["anchor_count"]) for row in rows]
            coverage_values = [_as_float(row, "delta_coverage") for row in rows]
            val_values = [_as_float(row, "delta_val_agreement") for row in rows]
            (line,) = axes[0].plot(x, coverage_values, marker="o", linewidth=1.3, label=teacher)
            _plot_ci_band(
                axes[0],
                x,
                [_as_float(row, "delta_coverage_ci_low") for row in rows],
                [_as_float(row, "delta_coverage_ci_high") for row in rows],
                color=line.get_color(),
                alpha=0.12,
            )
            (line,) = axes[1].plot(x, val_values, marker="o", linewidth=1.3, label=teacher)
            _plot_ci_band(
                axes[1],
                x,
                [_as_float(row, "delta_val_agreement_ci_low") for row in rows],
                [_as_float(row, "delta_val_agreement_ci_high") for row in rows],
                color=line.get_color(),
                alpha=0.12,
            )
        for ax, title in [
            (axes[0], "Teacher delta coverage with 95% CI"),
            (axes[1], "Teacher delta validation agreement with 95% CI"),
        ]:
            ax.axhline(0.0, color="#6b7280", linestyle=":", linewidth=0.9)
            ax.set_title(title)
            ax.set_xlabel("Anchor count")
            ax.grid(True, linewidth=0.5, alpha=0.35)
        axes[0].legend(fontsize=8)
        _save_figure(fig, figure_dir, "anchor_information_teacher_delta_ci", formats, written)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for teacher in teacher_names:
            rows = [row for row in teacher_rows if row["teacher"] == teacher]
            ax.plot(
                [int(row["anchor_count"]) for row in rows],
                [_as_float(row, "coverage") for row in rows],
                marker="o",
                linewidth=1.4,
                label=teacher,
            )
        ax.set_title("Coverage by teacher")
        ax.set_xlabel("Anchor count")
        ax.grid(True, linewidth=0.5, alpha=0.35)
        ax.legend(fontsize=8)
        _save_figure(fig, figure_dir, "anchor_information_teacher_coverage", formats, written)
        plt.close(fig)
    _plot_prototype_audit_heatmap(
        plt=plt,
        figure_dir=figure_dir,
        prototype_rows=prototype_rows,
        formats=formats,
        written=written,
    )
    return written


def _require_plot_backend(args: argparse.Namespace) -> None:
    if bool(getattr(args, "no_plots", False)):
        return
    if importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError(
            "matplotlib is required before reading teacher features because this run is expected to render figures. "
            "Install matplotlib in the active hcc-sempath environment or pass --no-plots."
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _require_plot_backend(args)
    seed = int(getattr(args, "seed", DEFAULT_SEED))
    locked_val_fraction = float(getattr(args, "locked_val_fraction", DEFAULT_LOCKED_VAL_FRACTION))
    locked_val_count = int(getattr(args, "locked_val_count", DEFAULT_LOCKED_VAL_COUNT))
    anchor_group_key = str(getattr(args, "anchor_group_key", DEFAULT_ANCHOR_GROUP_KEY))
    anchor_counts = _parse_int_list(getattr(args, "anchor_counts", DEFAULT_ANCHOR_COUNTS))
    bootstrap_iterations = int(getattr(args, "bootstrap_iterations", DEFAULT_BOOTSTRAP_ITERATIONS))
    generated_manifest_info: dict[str, Any] = {}
    if args.annotation_json:
        all_anchors = _anchors_from_annotation_json(Path(args.annotation_json))
        anchors, locked_val = _split_anchors(
            all_anchors,
            locked_val_fraction=locked_val_fraction,
            locked_val_count=locked_val_count,
            seed=seed,
        )
        input_dir = output_root / "inputs"
        anchor_manifest = input_dir / "anchor_pool.csv"
        locked_val_manifest = input_dir / "anchor_locked_val.csv"
        _write_anchor_manifest(anchor_manifest, anchors, "train")
        _write_anchor_manifest(locked_val_manifest, locked_val, "val")
        generated_manifest_info = {
            "annotation_json": str(args.annotation_json),
            "generated_anchor_manifest": str(anchor_manifest),
            "generated_locked_val_manifest": str(locked_val_manifest),
            "annotation_count": len(all_anchors),
            "train_anchor_count": len(anchors),
            "locked_val_count": len(locked_val),
            "locked_val_fraction": locked_val_fraction,
            "locked_val_count_requested": locked_val_count,
            "seed": seed,
        }
    else:
        if not args.anchor_manifest or not args.locked_val_manifest:
            raise ValueError("--anchor-manifest and --locked-val-manifest are required unless --annotation-json is used")
        locked_val = _load_locked_val(Path(args.locked_val_manifest))
        locked_val_tile_ids = {anchor.tile_id for anchor in locked_val}
        anchors = _load_anchors(Path(args.anchor_manifest), exclude_tile_ids=locked_val_tile_ids)
        anchor_manifest = Path(args.anchor_manifest)
        locked_val_manifest = Path(args.locked_val_manifest)
    subsets = _nested_subsets(anchors, anchor_counts, seed, anchor_group_key)
    level1_names, level2_names = _load_contract(Path(args.prototype_contract) if args.prototype_contract else None, anchors, locked_val)
    candidate_tiles = _load_candidate_tiles(Path(args.candidate_manifest) if args.candidate_manifest else None, locked_val)
    teacher_paths = _resolve_teacher_paths(args)
    all_anchor_tile_ids = [anchor.tile_id for subset in subsets.values() for anchor in subset]
    locked_val_tile_ids_ordered = [anchor.tile_id for anchor in locked_val]
    needed_tile_ids = list(dict.fromkeys([*all_anchor_tile_ids, *locked_val_tile_ids_ordered, *candidate_tiles]))

    report = {
        "sweep_type": "anchor_information_curve",
        "anchor_manifest": str(anchor_manifest),
        "locked_val_manifest": str(locked_val_manifest),
        "generated_manifests": generated_manifest_info,
        "candidate_manifest": str(args.candidate_manifest or ""),
        "prototype_contract": str(args.prototype_contract or ""),
        "anchor_counts_requested": anchor_counts,
        "anchor_counts_available": sorted(subsets),
        "nested_subsets": True,
        "locked_validation_reused_for_all_counts": True,
        "seed": seed,
        "anchor_group_key": anchor_group_key,
        "teachers": list(teacher_paths),
        "teacher_feature_packages": {teacher: [str(path) for path in paths] for teacher, paths in teacher_paths.items()},
        "level1_prototypes": level1_names,
        "level2_prototypes": level2_names,
        "does_not_train": True,
    }
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

    store = FeatureStore(teacher_paths)
    try:
        teacher_rows: list[dict[str, Any]] = []
        prototype_rows: list[dict[str, Any]] = []
        for teacher in teacher_paths:
            features = _feature_map(store, teacher, needed_tile_ids)
            next_teacher_rows, next_prototype_rows = _teacher_curve(
                teacher=teacher,
                subsets=subsets,
                locked_val=locked_val,
                candidate_tiles=candidate_tiles,
                features=features,
                level1_names=level1_names,
                level2_names=level2_names,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
            )
            teacher_rows.extend(next_teacher_rows)
            prototype_rows.extend(next_prototype_rows)
    finally:
        store.close()

    aggregate_rows = _aggregate_rows(
        teacher_rows,
        plateau_delta_epsilon=float(getattr(args, "plateau_delta_epsilon", 0.005)),
        plateau_drift_threshold=float(getattr(args, "plateau_drift_threshold", 0.01)),
        plateau_redundancy_threshold=float(getattr(args, "plateau_redundancy_threshold", 0.95)),
    )
    recommendation = _recommendation(aggregate_rows)
    _write_csv(output_root / "anchor_information_by_teacher.csv", teacher_rows)
    _write_csv(output_root / "anchor_information_by_prototype.csv", prototype_rows)
    _write_csv(output_root / "anchor_information_summary.csv", aggregate_rows)
    figure_paths = [] if bool(getattr(args, "no_plots", False)) else _plot_curves(
        output_root=output_root,
        teacher_rows=teacher_rows,
        aggregate_rows=aggregate_rows,
        prototype_rows=prototype_rows,
        formats=list(DEFAULT_PLOT_FORMATS),
    )
    report.update({"recommendation": recommendation, "figures": figure_paths})
    for name in OBSOLETE_JSON_OUTPUTS:
        path = output_root / name
        if path.exists():
            path.unlink()
    _write_json(output_root / "anchor_information_report.json", report)
    print(
        "anchor_information_curve_ok "
        f"output_root={output_root} teachers={len(teacher_paths)} counts={len(subsets)} "
        f"recommended_anchor_count={recommendation['recommended_anchor_count']} figures={len(figure_paths)}"
    )
    for path in figure_paths:
        print(f"figure={path}")
    return {"teacher": teacher_rows, "prototype": prototype_rows, "summary": aggregate_rows, "recommendation": recommendation}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute anchor information saturation curves from cached teacher features.")
    parser.add_argument("--teacher-feature-root", default="")
    parser.add_argument("--teacher-feature-packages", default="")
    parser.add_argument("--annotation-json", default="")
    parser.add_argument("--anchor-manifest", default="")
    parser.add_argument("--locked-val-manifest", default="")
    parser.add_argument("--candidate-manifest", default="")
    parser.add_argument("--prototype-contract", default="")
    parser.add_argument("--output-root", default="outputs/anchor_information_curve")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
