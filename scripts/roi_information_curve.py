#!/usr/bin/env python3
"""Audit whether current L2 ROI tiles cover all four teacher feature spaces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hcc_sempath.annotation_information import (  # noqa: E402
    CurveObservation,
    aggregate_fixed_probe_curves,
    fixed_probe_curve,
    meaningful_reference_checkpoints,
    prepare_fixed_probe_split,
    slide_round_robin_order,
    tail_low_gain,
)


DEFAULT_COUNTS = (
    "5,10,15,20,30,40,50,60,70,80,90,100,"
    "120,140,160,180,200,225,250,275,300,350,400,450,500"
)
DEFAULT_SEED = 13
DEFAULT_RESAMPLES = 32
DEFAULT_TOPK = 5
DEFAULT_ELBOW_RATIO = 0.35
DEFAULT_MIN_SLIDES = 5
DEFAULT_MIN_INCREMENTS = 3
DEFAULT_ELBOW_SUPPORT = 0.80
DEFAULT_MAX_ZERO_GEOMETRY_FRACTION = 0.01
DEFAULT_PROBE_SLIDE_FRACTION = 0.20
DEFAULT_MIN_PROBE_SLIDES = 2
DEFAULT_CONFIRMATION_INCREMENTS = 3
ELBOW_SENSITIVITIES = (0.25, 0.35, 0.50)
EXPECTED_TEACHERS = frozenset(
    {"gigapath", "h_optimus_1", "uni2_h", "virchow2"}
)
IMAGE_SIZE = (224, 224)
GRID_SIZE = (16, 16)
STATUS_VALUES = {
    "provisionally_stable",
    "still_growing",
    "coverage_limited",
    "not_assessable",
}


def _geometry_modality(kind: str) -> str:
    normalized = str(kind).lower()
    if normalized == "point":
        return "point"
    if normalized == "circle":
        return "circle"
    return "brush"


@dataclass(frozen=True)
class RoiFeatureSample:
    tile_id: str
    slide_id: str
    attribute: str
    teacher: str
    feature: np.ndarray


class TeacherFeatureStore:
    """Read selected tiles without indexing every row in the full caches."""

    def __init__(self, teacher_paths: dict[str, list[Path]]) -> None:
        from iatro.iac.adapters.features import FeatureCacheReader

        self._reader_cls = FeatureCacheReader
        self._paths = teacher_paths
        self._readers: dict[tuple[str, Path], Any] = {}
        self._paths_by_key = {
            teacher: self._index_paths(paths)
            for teacher, paths in teacher_paths.items()
        }

    @staticmethod
    def _package_keys(path: Path) -> list[str]:
        name = path.name
        suffixes = (
            ".prov-gigapath-local.features.iac",
            ".h_optimus_1.features.iac",
            ".h1.features.iac",
            ".uni2_h.features.iac",
            ".uni2.features.iac",
            ".virchow2.features.iac",
            ".features.iac",
        )
        keys = [
            name[: -len(suffix)]
            for suffix in suffixes
            if name.endswith(suffix)
        ]
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
        candidates: list[Path] = []
        for key in self._tile_keys(tile_id):
            candidates.extend(self._paths_by_key[teacher].get(key, []))
        candidates = list(
            dict.fromkeys([*candidates, *self._paths[teacher]])
        )
        for path in candidates:
            try:
                return self._reader(teacher, path).read_feature(tile_id)
            except FileNotFoundError as exc:
                last_error = exc
        raise FileNotFoundError(
            f"missing teacher feature: teacher={teacher} tile_id={tile_id}"
        ) from last_error

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()


def _teacher_paths_from_arg(value: str) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for item in str(value or "").split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(
                "teacher package entry must be teacher=path: "
                f"{item}"
            )
        teacher, raw_paths = item.split("=", 1)
        paths = [Path(path) for path in raw_paths.split("|") if path]
        if not paths:
            raise ValueError(f"empty feature packages for teacher={teacher}")
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"teacher feature packages are missing: {missing[:3]}"
            )
        result[teacher] = paths
    if not result:
        raise ValueError("--teacher-feature-packages is required")
    return result


def _log(message: str) -> None:
    print(f"[roi-information] {message}", flush=True)


def _parse_int_list(value: str) -> list[int]:
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError("--sample-counts requires positive comma-separated integers")
    return values


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)


def _finite(value: float) -> float | str:
    return float(value) if math.isfinite(float(value)) else ""


def _quantile(values: Iterable[float], q: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, q)) if array.size else math.nan


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else math.nan


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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _active_attributes(payload: dict[str, Any]) -> list[str]:
    definitions = payload.get("label_definitions", {}).get("l2", [])
    active = [
        str(item["id"])
        for item in definitions
        if isinstance(item, dict) and bool(item.get("active", True)) and item.get("id")
    ]
    attributes = active or [str(value) for value in payload.get("l2_prototypes", [])]
    if not attributes:
        raise ValueError("annotation state has no active L2 ROI attributes")
    return attributes


def _load_annotation_state(path: Path) -> tuple[dict[str, Any], list[str], dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    annotations = payload.get("annotations")
    if not isinstance(annotations, dict):
        raise ValueError(f"annotation state requires an annotations object: {path}")
    attributes = _active_attributes(payload)
    items: dict[str, dict[str, Any]] = {}
    for key, raw in annotations.items():
        if not isinstance(raw, dict) or not raw.get("tile_id"):
            continue
        if str(raw.get("split", "train")) != "train":
            continue
        item = dict(raw)
        tile_id = str(item["tile_id"])
        if tile_id in items:
            raise ValueError(f"duplicate tile_id in annotation state: {tile_id}")
        item["_annotation_key"] = str(key)
        items[tile_id] = item
    if not items:
        raise ValueError(f"annotation state has no train ROI tiles: {path}")
    return payload, attributes, items


def _coverage_rows(
    attributes: list[str],
    items: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    from hcc_sempath.training.roi import geometry_token_mask

    geometry_counts: dict[str, Counter[str]] = {name: Counter() for name in attributes}
    zero_token_geometries = Counter()
    explicit_negative_masks: dict[tuple[str, str], Any] = {}
    explicit_positive_masks: dict[tuple[str, str], Any] = {}
    positive_modalities: dict[tuple[str, str], set[str]] = defaultdict(set)

    for tile_id, item in items.items():
        for record in item.get("roi", []):
            attribute = str(record.get("attribute", ""))
            if attribute not in geometry_counts:
                continue
            geometry = record.get("geometry")
            state = str(record.get("state", "positive")).lower()
            if geometry is None:
                if state == "negative" and bool(record.get("review_complete", False)):
                    geometry_counts[attribute]["review_complete"] += 1
                    explicit_negative_masks[(tile_id, attribute)] = torch.ones(
                        GRID_SIZE,
                        dtype=torch.bool,
                    )
                continue
            kind = str(geometry.get("type", "unknown")).lower()
            geometry_counts[attribute][f"{state}_{kind}"] += 1
            mask = geometry_token_mask(geometry, image_size=IMAGE_SIZE, grid_size=GRID_SIZE)
            if not bool(mask.any()):
                zero_token_geometries[attribute] += 1
            store = explicit_positive_masks if state == "positive" else explicit_negative_masks
            key = (tile_id, attribute)
            store[key] = mask if key not in store else (store[key] | mask)
            if state == "positive":
                positive_modalities[key].add(_geometry_modality(kind))

    conflicts = Counter()
    for key in explicit_positive_masks.keys() & explicit_negative_masks.keys():
        overlap = explicit_positive_masks[key] & explicit_negative_masks[key]
        conflicts[key[1]] += int(overlap.sum().item())

    rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    for index, attribute in enumerate(attributes):
        positive_tiles: list[str] = []
        negative_tiles: list[str] = []
        positive_tokens: list[int] = []
        negative_token_count = 0
        occupancy = np.zeros(GRID_SIZE, dtype=np.int64)
        slide_counts = Counter()
        negative_slides: set[str] = set()
        point_tiles = 0
        circle_tiles = 0
        brush_tiles = 0
        for (tile_id, candidate), positive in explicit_positive_masks.items():
            if candidate != attribute:
                continue
            slide = str(items[tile_id].get("slide") or items[tile_id].get("slide_id") or tile_id)
            positive_count = int(positive.sum().item())
            if positive_count:
                positive_tiles.append(tile_id)
                positive_tokens.append(positive_count)
                slide_counts[slide] += 1
                occupancy += positive.numpy().astype(np.int64)
                modalities = positive_modalities[(tile_id, attribute)]
                point_tiles += int("point" in modalities)
                circle_tiles += int("circle" in modalities)
                brush_tiles += int("brush" in modalities)
        for (tile_id, candidate), negative in explicit_negative_masks.items():
            if candidate != attribute:
                continue
            slide = str(items[tile_id].get("slide") or items[tile_id].get("slide_id") or tile_id)
            negative_count = int(negative.sum().item())
            if negative_count:
                negative_tiles.append(tile_id)
                negative_slides.add(slide)
                negative_token_count += negative_count

        total_positive = sum(positive_tokens)
        shares = np.asarray(list(slide_counts.values()), dtype=np.float64)
        shares = shares / shares.sum() if shares.size else shares
        effective_slides = float(1.0 / np.square(shares).sum()) if shares.size else 0.0
        max_slide_share = float(shares.max()) if shares.size else 0.0
        occupied = occupancy[occupancy > 0].astype(np.float64)
        if occupied.size > 1:
            probabilities = occupied / occupied.sum()
            occupancy_entropy = float(-(probabilities * np.log(probabilities)).sum() / math.log(occupancy.size))
        else:
            occupancy_entropy = 0.0
        counts = geometry_counts[attribute]
        positive_geometry_count = sum(
            count for key, count in counts.items() if key.startswith("positive_")
        )
        zero_geometry_fraction = (
            float(zero_token_geometries[attribute]) / positive_geometry_count
            if positive_geometry_count
            else 0.0
        )
        row = {
            "attribute": attribute,
            "positive_tile_count": len(positive_tiles),
            "positive_slide_count": len(slide_counts),
            "effective_positive_slide_count": effective_slides,
            "max_positive_slide_share": max_slide_share,
            "point_positive_tile_count": point_tiles,
            "circle_positive_tile_count": circle_tiles,
            "brush_positive_tile_count": brush_tiles,
            "positive_token_count": total_positive,
            "positive_tokens_per_tile_median": _quantile(positive_tokens, 0.50),
            "positive_tokens_per_tile_q25": _quantile(positive_tokens, 0.25),
            "positive_tokens_per_tile_q75": _quantile(positive_tokens, 0.75),
            "negative_reviewed_tile_count": len(negative_tiles),
            "negative_reviewed_slide_count": len(negative_slides),
            "negative_reviewed_token_count": negative_token_count,
            "occupied_patch_fraction": float((occupancy > 0).mean()),
            "occupancy_entropy": occupancy_entropy,
            "zero_token_geometry_count": int(zero_token_geometries[attribute]),
            "zero_token_geometry_fraction": zero_geometry_fraction,
            "explicit_conflict_token_count": int(conflicts[attribute]),
            "positive_geometry_count": positive_geometry_count,
            "point_geometry_count": counts["positive_point"],
            "brush_geometry_count": counts["positive_brush"] + counts["positive_polyline"],
            "circle_geometry_count": counts["positive_circle"],
            "polygon_geometry_count": counts["positive_polygon"] + counts["positive_freehand"],
            "review_complete_count": counts["review_complete"],
        }
        rows.append(row)
        detail[attribute] = {
            "positive_tile_ids": positive_tiles,
            "positive_token_counts": dict(zip(positive_tiles, positive_tokens)),
        }
    return rows, detail


def _extract_teacher_features(
    items: dict[str, dict[str, Any]],
    positive_tile_ids: dict[str, list[str]],
    teacher_paths: dict[str, list[Path]],
) -> tuple[list[RoiFeatureSample], dict[str, Any]]:
    needed_tile_ids = sorted(
        {
            tile_id
            for tile_ids in positive_tile_ids.values()
            for tile_id in tile_ids
        }
    )
    attributes_by_tile: dict[str, list[str]] = defaultdict(list)
    for attribute, tile_ids in positive_tile_ids.items():
        for tile_id in tile_ids:
            attributes_by_tile[tile_id].append(attribute)
    samples: list[RoiFeatureSample] = []
    dimensions: dict[str, int] = {}
    store = TeacherFeatureStore(teacher_paths)
    try:
        for index, tile_id in enumerate(needed_tile_ids, start=1):
            item = items[tile_id]
            slide_id = str(
                item.get("slide") or item.get("slide_id") or tile_id
            )
            for teacher in teacher_paths:
                feature = np.asarray(
                    store.read(teacher, tile_id),
                    dtype=np.float32,
                )
                if feature.ndim != 1:
                    raise ValueError(
                        "teacher feature must be one-dimensional: "
                        f"teacher={teacher} tile={tile_id} "
                        f"shape={feature.shape}"
                    )
                previous_dim = dimensions.setdefault(
                    teacher,
                    int(feature.shape[0]),
                )
                if previous_dim != int(feature.shape[0]):
                    raise ValueError(
                        "teacher feature dimension changed: "
                        f"teacher={teacher} expected={previous_dim} "
                        f"observed={feature.shape[0]}"
                    )
                feature = _normalize(feature)
                for attribute in attributes_by_tile[tile_id]:
                    samples.append(
                        RoiFeatureSample(
                            tile_id=tile_id,
                            slide_id=slide_id,
                            attribute=attribute,
                            teacher=teacher,
                            feature=feature,
                        )
                    )
            if index == 1 or index == len(needed_tile_ids) or index % 50 == 0:
                _log(
                    "read teacher caches "
                    f"{index}/{len(needed_tile_ids)} positive ROI tiles"
                )
    finally:
        store.close()
    metadata = {
        "source": "four frozen teacher feature caches used by training",
        "teachers": list(teacher_paths),
        "teacher_dimensions": dimensions,
        "teacher_feature_packages": {
            teacher: [str(path) for path in paths]
            for teacher, paths in teacher_paths.items()
        },
        "representation": (
            "one L2-normalized cached global feature per positive ROI tile "
            "and teacher; no raw-DINO or trained-student substitute"
        ),
        "positive_tile_count": len(needed_tile_ids),
    }
    return samples, metadata


def _stable_seed(attribute: str, seed: int, resample: int) -> int:
    digest = hashlib.sha256(attribute.encode("utf-8")).digest()
    return seed + int.from_bytes(digest[:4], "little") + resample * 100_003


def _slide_round_robin(samples: list[RoiFeatureSample], seed: int) -> list[str]:
    return slide_round_robin_order(
        [
            CurveObservation(
                tile_id=sample.tile_id,
                slide_id=sample.slide_id,
                stratum=sample.attribute,
                feature=sample.feature,
            )
            for sample in samples
        ],
        seed=seed,
    )


def _checkpoints(total: int, requested: list[int]) -> list[int]:
    return meaningful_reference_checkpoints(total, requested)


def _one_curve(
    samples: list[RoiFeatureSample],
    checkpoints: list[int],
    *,
    seed: int,
    topk: int,
    probe_slide_fraction: float = DEFAULT_PROBE_SLIDE_FRACTION,
    min_probe_slides: int = DEFAULT_MIN_PROBE_SLIDES,
) -> list[dict[str, Any]]:
    observations = [
        CurveObservation(
            tile_id=sample.tile_id,
            slide_id=sample.slide_id,
            stratum=sample.attribute,
            feature=sample.feature,
        )
        for sample in samples
    ]
    split = prepare_fixed_probe_split(
        observations,
        seed=seed,
        probe_slide_fraction=probe_slide_fraction,
        min_probe_slides=min_probe_slides,
    )
    rows = fixed_probe_curve(split, checkpoints, topk=topk)
    return [dict(row) for row in rows]


def _status(
    *,
    checkpoints: list[int],
    coverage: dict[str, Any],
    support: float,
    min_slides: int,
    min_increments: int,
    support_threshold: float,
    max_zero_geometry_fraction: float,
) -> tuple[str, str]:
    slides = int(coverage["positive_slide_count"])
    if len(checkpoints) - 1 < min_increments or slides < min_slides:
        return (
            "not_assessable",
            f"requires at least {min_increments} information increments and {min_slides} slides; "
            f"observed {len(checkpoints) - 1} increments and {slides} slides",
        )
    tail_stable = support >= support_threshold
    coverage_issue = (
        float(coverage["effective_positive_slide_count"]) < min_slides
        or float(coverage["max_positive_slide_share"]) > 0.50
        or float(coverage["zero_token_geometry_fraction"]) > max_zero_geometry_fraction
        or int(coverage["explicit_conflict_token_count"]) > 0
    )
    if tail_stable and coverage_issue:
        return (
            "coverage_limited",
            "all teacher-space curves have low-gain tails, but slide or ROI "
            "geometry QC fails",
        )
    if tail_stable:
        return (
            "provisionally_stable",
            "all four teacher spaces have confirmed low-gain fixed-probe tails",
        )
    return (
        "still_growing",
        "at least one teacher space lacks consecutive low-gain confirmation",
    )


def evaluate_information(
    samples: list[RoiFeatureSample],
    coverage_rows: list[dict[str, Any]],
    attributes: list[str],
    *,
    requested_counts: list[int],
    seed: int,
    resamples: int,
    topk: int,
    elbow_ratio: float,
    min_slides: int,
    min_increments: int,
    support_threshold: float,
    max_zero_geometry_fraction: float = DEFAULT_MAX_ZERO_GEOMETRY_FRACTION,
    probe_slide_fraction: float = DEFAULT_PROBE_SLIDE_FRACTION,
    min_probe_slides: int = DEFAULT_MIN_PROBE_SLIDES,
    confirmation_increments: int = DEFAULT_CONFIRMATION_INCREMENTS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_attribute_teacher: dict[
        tuple[str, str],
        list[RoiFeatureSample],
    ] = defaultdict(list)
    for sample in samples:
        by_attribute_teacher[(sample.attribute, sample.teacher)].append(
            sample
        )
    teachers = sorted({sample.teacher for sample in samples})
    if not teachers:
        raise ValueError("teacher-space L2 audit requires teacher features")
    coverage_by_attribute = {row["attribute"]: row for row in coverage_rows}
    summary_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    report_attributes: dict[str, Any] = {}

    for attribute in attributes:
        samples_by_teacher = {
            teacher: by_attribute_teacher.get((attribute, teacher), [])
            for teacher in teachers
        }
        if any(not values for values in samples_by_teacher.values()):
            status, reason = "not_assessable", "no positive ROI tiles"
            summary = {**coverage_by_attribute[attribute], "status": status, "enough_now": False, "reason": reason}
            summary_rows.append(summary)
            report_attributes[attribute] = {"status": status, "reason": reason, "checkpoints": []}
            continue
        try:
            splits_by_teacher = {
                teacher: [
                    prepare_fixed_probe_split(
                        [
                            CurveObservation(
                                tile_id=sample.tile_id,
                                slide_id=sample.slide_id,
                                stratum=attribute,
                                feature=sample.feature,
                            )
                            for sample in teacher_samples
                        ],
                        seed=_stable_seed(attribute, seed, resample),
                        probe_slide_fraction=probe_slide_fraction,
                        min_probe_slides=min_probe_slides,
                    )
                    for resample in range(resamples)
                ]
                for teacher, teacher_samples in samples_by_teacher.items()
            }
        except ValueError as exc:
            status, reason = "not_assessable", str(exc)
            summary = {
                **coverage_by_attribute[attribute],
                "status": status,
                "enough_now": False,
                "reason": reason,
            }
            summary_rows.append(summary)
            report_attributes[attribute] = {
                "status": status,
                "reason": reason,
                "checkpoints": [],
            }
            continue
        reference_capacity = min(
            len(split.reference_tile_order)
            for teacher_splits in splits_by_teacher.values()
            for split in teacher_splits
        )
        checkpoints = _checkpoints(reference_capacity, requested_counts)
        curves_by_teacher = {
            teacher: [
                [
                    dict(row)
                    for row in fixed_probe_curve(
                        split,
                        checkpoints,
                        topk=topk,
                    )
                ]
                for split in teacher_splits
            ]
            for teacher, teacher_splits in splits_by_teacher.items()
        }
        curves = [
            curve
            for teacher_curves in curves_by_teacher.values()
            for curve in teacher_curves
        ]
        aggregate = [
            dict(row)
            for row in aggregate_fixed_probe_curves(curves)
        ]
        aggregate_by_teacher = {
            teacher: [
                dict(row)
                for row in aggregate_fixed_probe_curves(teacher_curves)
            ]
            for teacher, teacher_curves in curves_by_teacher.items()
        }
        supports_by_ratio: dict[str, float] = {}
        supports_by_teacher_ratio: dict[
            str,
            dict[str, float],
        ] = {}
        recommendations: dict[str, int | None] = {}
        recommendations_by_teacher_ratio: dict[
            str,
            dict[str, int | None],
        ] = {}
        for ratio in ELBOW_SENSITIVITIES:
            ratio_key = f"{ratio:.2f}"
            teacher_support: dict[str, float] = {}
            teacher_recommendation: dict[str, int | None] = {}
            for teacher, teacher_curves in curves_by_teacher.items():
                decisions = [
                    tail_low_gain(
                        curve,
                        marginal_ratio_threshold=ratio,
                        confirmation_increments=confirmation_increments,
                    )
                    for curve in teacher_curves
                ]
                observed = [
                    onset
                    for stable, onset in decisions
                    if stable and onset is not None
                ]
                teacher_support[teacher] = len(observed) / resamples
                teacher_recommendation[teacher] = (
                    int(round(float(np.median(observed))))
                    if observed
                    else None
                )
            supports_by_teacher_ratio[ratio_key] = teacher_support
            supports_by_ratio[ratio_key] = min(teacher_support.values())
            recommendations_by_teacher_ratio[ratio_key] = (
                teacher_recommendation
            )
            observed_recommendations = [
                value
                for value in teacher_recommendation.values()
                if value is not None
            ]
            recommendations[ratio_key] = (
                max(observed_recommendations)
                if len(observed_recommendations) == len(teachers)
                else None
            )
        primary_key = f"{elbow_ratio:.2f}"
        primary_support = supports_by_ratio[primary_key]
        status, reason = _status(
            checkpoints=checkpoints,
            coverage=coverage_by_attribute[attribute],
            support=primary_support,
            min_slides=min_slides,
            min_increments=min_increments,
            support_threshold=support_threshold,
            max_zero_geometry_fraction=max_zero_geometry_fraction,
        )
        if status not in STATUS_VALUES:
            raise AssertionError(status)
        recommended = recommendations[primary_key]
        summary = {
            **coverage_by_attribute[attribute],
            "status": status,
            "enough_now": status == "provisionally_stable",
            "reason": reason,
            "teacher_low_gain_support_min": primary_support,
            "recommended_reference_tile_count_if_stable": recommended or "",
            "reference_capacity": reference_capacity,
            "last_remaining_novelty_mean": _finite(
                float(aggregate[-1]["remaining_novelty_mean"])
            ),
            "last_remaining_novelty_q95": _finite(
                float(aggregate[-1]["remaining_novelty_q95"])
            ),
            "last_information_gain_per_100_tiles": _finite(
                float(aggregate[-1]["information_gain_per_100_tiles"])
            ),
        }
        summary_rows.append(summary)
        attribute_curve_rows = [
            {
                "attribute": attribute,
                **{
                    key: _finite(value)
                    if isinstance(value, float)
                    else value
                    for key, value in row.items()
                    if not key.startswith("center_drift")
                },
            }
            for row in aggregate
        ]
        curve_rows.extend(attribute_curve_rows)
        report_attributes[attribute] = {
            "status": status,
            "enough_now": status == "provisionally_stable",
            "reason": reason,
            "checkpoints": checkpoints,
            "reference_capacity": reference_capacity,
            "probe_slide_fraction": probe_slide_fraction,
            "probe_slide_count_mean": aggregate[-1][
                "probe_slide_count"
            ],
            "teacher_low_gain_support_by_ratio": supports_by_ratio,
            "teacher_low_gain_support_by_teacher_ratio": (
                supports_by_teacher_ratio
            ),
            "recommended_reference_tile_count_by_ratio": recommendations,
            "recommended_reference_tile_count_by_teacher_ratio": (
                recommendations_by_teacher_ratio
            ),
            "teacher_curves": {
                teacher: [
                    {
                        key: _finite(value)
                        if isinstance(value, float)
                        else value
                        for key, value in row.items()
                        if not key.startswith("center_drift")
                    }
                    for row in teacher_curve
                ]
                for teacher, teacher_curve in aggregate_by_teacher.items()
            },
            "coverage": coverage_by_attribute[attribute],
            "curve": attribute_curve_rows,
        }
    return summary_rows, curve_rows, report_attributes


def _plot(
    summary_rows: list[dict[str, Any]],
    attribute_report: dict[str, Any],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    attributes = [row["attribute"] for row in summary_rows]
    figure, axes = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)
    for attribute, axis in zip(attributes, axes.flat):
        teacher_curves = attribute_report[attribute].get(
            "teacher_curves",
            {},
        )
        for teacher, rows in teacher_curves.items():
            axis.plot(
                [int(row["sample_count"]) for row in rows],
                [
                    float(row["remaining_novelty_mean"])
                    if row["remaining_novelty_mean"] != ""
                    else np.nan
                    for row in rows
                ],
                marker="o",
                markersize=3,
                linewidth=1.4,
                label=teacher,
            )
        summary = next(row for row in summary_rows if row["attribute"] == attribute)
        axis.set_title(f"{attribute}\n{summary['status']} · N={summary['positive_tile_count']}", fontsize=9)
        axis.set_xlabel("reference positive ROI tiles")
        axis.set_ylabel("remaining novelty", color="#1f77b4")
        axis.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="outside lower center",
            ncols=len(labels),
            fontsize=8,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.suptitle(
        "Current L2 ROI coverage across four teacher spaces "
        "(monotone fixed-probe novelty)",
        fontsize=14,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    annotation_path = Path(args.annotation_json).resolve()
    output_root = Path(args.output_root).resolve()
    payload, attributes, items = _load_annotation_state(annotation_path)
    _log(f"loaded {len(items)} current ROI tiles across {len(attributes)} attributes")
    coverage_rows, coverage_detail = _coverage_rows(attributes, items)
    teacher_paths = _teacher_paths_from_arg(
        args.teacher_feature_packages
    )
    if set(teacher_paths) != EXPECTED_TEACHERS:
        raise ValueError(
            "L2 task-space audit requires exactly the four training teachers: "
            f"expected={sorted(EXPECTED_TEACHERS)} "
            f"observed={sorted(teacher_paths)}"
        )
    _log(
        "reading task-aligned caches for teachers: "
        + ", ".join(teacher_paths)
    )
    samples, feature_metadata = _extract_teacher_features(
        items,
        {
            attribute: list(
                coverage_detail[attribute]["positive_tile_ids"]
            )
            for attribute in attributes
        },
        teacher_paths,
    )
    summary_rows, curve_rows, attribute_report = evaluate_information(
        samples,
        coverage_rows,
        attributes,
        requested_counts=_parse_int_list(args.sample_counts),
        seed=args.seed,
        resamples=args.resamples,
        topk=args.topk,
        elbow_ratio=args.elbow_ratio,
        min_slides=args.min_slides,
        min_increments=args.min_increments,
        support_threshold=args.elbow_support,
        max_zero_geometry_fraction=args.max_zero_geometry_fraction,
        probe_slide_fraction=args.probe_slide_fraction,
        min_probe_slides=args.min_probe_slides,
        confirmation_increments=args.confirmation_increments,
    )
    report = {
        "audit_type": "roi_annotation_information_curve",
        "claim_scope": "pre-training annotation information coverage; not downstream model performance",
        "does_not_train": True,
        "legacy_tile_level_l2_used": False,
        "annotation_json": str(annotation_path),
        "annotation_state_version": payload.get("version"),
        "annotation_tile_count": len(items),
        "attributes": attribute_report,
        "feature_source": feature_metadata,
        "method": {
            "primary_observation": (
                "one cached feature for each positive ROI tile in each of "
                "the four frozen teacher spaces used by distillation"
            ),
            "curve_unit": "reference unique positive tile per component",
            "primary_curve": (
                "remaining novelty of one fixed slide-separated probe while "
                "the nested reference set grows; monotone non-increasing by construction"
            ),
            "teacher_gate": (
                "coverage is computed separately for every teacher; the "
                "component passes only when every teacher reaches the "
                "required repeated low-gain support"
            ),
            "geometry_role": (
                "point, circle, brush, slide balance, rasterization, and "
                "explicit conflicts are independent annotation QC; raw RGB "
                "or untrained DINO features never substitute for teacher space"
            ),
            "sampling": (
                "repeated fixed slide-level probe split plus nested "
                "slide-aware reference order"
            ),
            "probe_slide_fraction": args.probe_slide_fraction,
            "minimum_probe_slides": args.min_probe_slides,
            "resamples": args.resamples,
            "seed": args.seed,
            "topk": args.topk,
            "elbow_marginal_ratio": args.elbow_ratio,
            "elbow_ratio_sensitivity": list(ELBOW_SENSITIVITIES),
            "required_teacher_low_gain_support": args.elbow_support,
            "confirmation_increments": args.confirmation_increments,
            "minimum_slides": args.min_slides,
            "minimum_information_increments": args.min_increments,
            "maximum_zero_token_geometry_fraction": args.max_zero_geometry_fraction,
            "negative_evidence": "coverage QC only; no single negative center is assumed",
        },
        "status_meanings": {
            "provisionally_stable": "every teacher space has a confirmed low-gain tail",
            "still_growing": "at least one teacher space lacks consecutive low-gain confirmation",
            "coverage_limited": "teacher curves pass but slide/geometry coverage fails QC",
            "not_assessable": "too few tiles, increments, or independent slides to test stability",
        },
    }
    _write_csv(output_root / "roi_information_summary.csv", summary_rows)
    _write_csv(output_root / "roi_information_curve.csv", curve_rows)
    teacher_curve_rows = [
        {
            "attribute": attribute,
            "teacher": teacher,
            **row,
        }
        for attribute, value in attribute_report.items()
        for teacher, rows in value.get("teacher_curves", {}).items()
        for row in rows
    ]
    _write_csv(
        output_root / "roi_information_curve_by_teacher.csv",
        teacher_curve_rows,
    )
    _write_json(output_root / "roi_information_report.json", report)
    if not args.no_plots:
        _plot(
            summary_rows,
            attribute_report,
            output_root / "roi_information_curves.png",
        )
    for row in summary_rows:
        _log(
            f"{row['attribute']}: {row['status']} "
            f"({row['positive_tile_count']} tiles / {row['positive_slide_count']} slides)"
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-json",
        default=str(REPO_ROOT / "annotations" / "hcc_l2_roi_v2.json"),
        help="Current ROI annotation state JSON",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "artifacts" / "diagnostics" / "roi_information_curve_current"),
    )
    parser.add_argument(
        "--teacher-feature-packages",
        default="",
        help="Comma-separated teacher=package|package mappings",
    )
    parser.add_argument("--sample-counts", default=DEFAULT_COUNTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--elbow-ratio", type=float, choices=ELBOW_SENSITIVITIES, default=DEFAULT_ELBOW_RATIO)
    parser.add_argument("--elbow-support", type=float, default=DEFAULT_ELBOW_SUPPORT)
    parser.add_argument("--min-slides", type=int, default=DEFAULT_MIN_SLIDES)
    parser.add_argument("--min-increments", type=int, default=DEFAULT_MIN_INCREMENTS)
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
        "--confirmation-increments",
        type=int,
        default=DEFAULT_CONFIRMATION_INCREMENTS,
    )
    parser.add_argument(
        "--max-zero-geometry-fraction",
        type=float,
        default=DEFAULT_MAX_ZERO_GEOMETRY_FRACTION,
        help="Geometry QC failure threshold; isolated rasterization misses remain reported but do not veto a class",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.resamples < 1:
        raise ValueError("--resamples must be positive")
    if not 0.0 < args.probe_slide_fraction < 0.5:
        raise ValueError("--probe-slide-fraction must be between 0 and 0.5")
    if args.min_probe_slides < 1 or args.confirmation_increments < 2:
        raise ValueError(
            "--min-probe-slides must be positive and "
            "--confirmation-increments must be at least 2"
        )
    run(args)


if __name__ == "__main__":
    main()
