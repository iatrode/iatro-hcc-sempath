"""Fixed-probe information coverage curves for annotation sufficiency audits."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CurveObservation:
    tile_id: str
    slide_id: str
    stratum: str
    feature: np.ndarray


@dataclass(frozen=True)
class FixedProbeSplit:
    observations_by_tile: dict[str, tuple[CurveObservation, ...]]
    reference_tile_order: tuple[str, ...]
    probe_observations: tuple[CurveObservation, ...]
    reference_slide_count: int
    probe_slide_count: int


def meaningful_reference_checkpoints(
    capacity: int,
    requested: Iterable[int],
) -> list[int]:
    """Keep nested checkpoints and discard an unstable tiny remainder."""
    if capacity < 1:
        return []
    counts = sorted(
        {
            int(value)
            for value in requested
            if 0 < int(value) <= capacity
        }
    )
    if capacity in counts:
        return counts
    if not counts:
        return [capacity]
    previous_gap = (
        counts[-1] - counts[-2]
        if len(counts) >= 2
        else counts[-1]
    )
    minimum_tail_increment = max(
        5,
        int(math.ceil(previous_gap / 2.0)),
    )
    if capacity - counts[-1] >= minimum_tail_increment:
        counts.append(capacity)
    return counts


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    return value / max(float(np.linalg.norm(value)), 1e-8)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float32)
    return value / np.maximum(
        np.linalg.norm(value, axis=1, keepdims=True),
        1e-8,
    )


def _mean(values: Iterable[float]) -> float:
    finite = [
        float(value)
        for value in values
        if math.isfinite(float(value))
    ]
    return float(np.mean(finite)) if finite else math.nan


def _quantile(values: Iterable[float], q: float) -> float:
    finite = np.asarray(
        [
            float(value)
            for value in values
            if math.isfinite(float(value))
        ],
        dtype=np.float64,
    )
    return float(np.quantile(finite, q)) if finite.size else math.nan


def slide_round_robin_order(
    observations: Iterable[CurveObservation],
    *,
    seed: int,
) -> list[str]:
    rng = random.Random(seed)
    tile_to_slide: dict[str, str] = {}
    for observation in observations:
        previous = tile_to_slide.setdefault(
            observation.tile_id,
            observation.slide_id,
        )
        if previous != observation.slide_id:
            raise ValueError(
                "tile appears under multiple slides: "
                f"{observation.tile_id}"
            )
    grouped: dict[str, list[str]] = defaultdict(list)
    for tile_id, slide_id in tile_to_slide.items():
        grouped[slide_id].append(tile_id)
    slides = sorted(grouped)
    rng.shuffle(slides)
    for slide in slides:
        rng.shuffle(grouped[slide])
    order: list[str] = []
    depth = 0
    while True:
        added = False
        round_slides = list(slides)
        rng.shuffle(round_slides)
        for slide in round_slides:
            if depth < len(grouped[slide]):
                order.append(grouped[slide][depth])
                added = True
        if not added:
            return order
        depth += 1


def prepare_fixed_probe_split(
    observations: Iterable[CurveObservation],
    *,
    seed: int,
    probe_slide_fraction: float,
    min_probe_slides: int,
) -> FixedProbeSplit:
    values = list(observations)
    if not values:
        raise ValueError("fixed-probe curve requires observations")
    if not 0.0 < probe_slide_fraction < 0.5:
        raise ValueError("probe_slide_fraction must be between 0 and 0.5")
    if min_probe_slides < 1:
        raise ValueError("min_probe_slides must be positive")

    observations_by_tile: dict[str, list[CurveObservation]] = defaultdict(
        list
    )
    slide_by_tile: dict[str, str] = {}
    for observation in values:
        previous = slide_by_tile.setdefault(
            observation.tile_id,
            observation.slide_id,
        )
        if previous != observation.slide_id:
            raise ValueError(
                "tile appears under multiple slides: "
                f"{observation.tile_id}"
            )
        observations_by_tile[observation.tile_id].append(observation)

    slides = sorted(set(slide_by_tile.values()))
    if len(slides) < min_probe_slides + 2:
        raise ValueError(
            "fixed-probe curve requires at least "
            f"{min_probe_slides + 2} slides; observed {len(slides)}"
        )
    rng = random.Random(seed)
    rng.shuffle(slides)
    probe_count = max(
        min_probe_slides,
        int(round(len(slides) * probe_slide_fraction)),
    )
    probe_count = min(probe_count, len(slides) - 2)
    probe_slides = set(slides[:probe_count])
    reference_slides = set(slides[probe_count:])

    probe = tuple(
        observation
        for observation in values
        if observation.slide_id in probe_slides
    )
    reference_observations = [
        observation
        for observation in values
        if observation.slide_id in reference_slides
    ]
    if not probe or not reference_observations:
        raise ValueError("fixed-probe split produced an empty side")
    reference_order = tuple(
        slide_round_robin_order(
            reference_observations,
            seed=seed + 17,
        )
    )
    return FixedProbeSplit(
        observations_by_tile={
            tile_id: tuple(items)
            for tile_id, items in observations_by_tile.items()
        },
        reference_tile_order=reference_order,
        probe_observations=probe,
        reference_slide_count=len(reference_slides),
        probe_slide_count=len(probe_slides),
    )


def _fixed_k_novelty_from_similarities(
    similarities: np.ndarray,
    *,
    reference_observation_count: int,
    topk: int,
) -> np.ndarray:
    if similarities.ndim != 2:
        raise ValueError("similarities must be a matrix")
    if topk < 1:
        raise ValueError("topk must be positive")
    available = min(reference_observation_count, similarities.shape[1])
    if available <= 0:
        return np.full(
            similarities.shape[0],
            2.0,
            dtype=np.float32,
        )
    current = similarities[:, :available]
    keep = min(topk, available)
    top = np.partition(current, -keep, axis=1)[:, -keep:]
    if keep < topk:
        padding = np.full(
            (top.shape[0], topk - keep),
            -1.0,
            dtype=top.dtype,
        )
        top = np.concatenate([top, padding], axis=1)
    return (1.0 - top.mean(axis=1)).astype(np.float32)


def _slide_stratum_balanced_center(
    observations: list[CurveObservation],
) -> np.ndarray:
    by_stratum_slide: dict[
        str,
        dict[str, list[np.ndarray]],
    ] = defaultdict(lambda: defaultdict(list))
    for observation in observations:
        by_stratum_slide[observation.stratum][
            observation.slide_id
        ].append(observation.feature)
    stratum_centers: list[np.ndarray] = []
    for by_slide in by_stratum_slide.values():
        slide_centers = [
            _normalize(np.stack(features).mean(axis=0))
            for features in by_slide.values()
        ]
        stratum_centers.append(
            _normalize(np.stack(slide_centers).mean(axis=0))
        )
    # This helper is only called on one common feature plane. Callers with
    # heterogeneous teacher dimensions compute centres per stratum upstream.
    dimensions = {center.shape for center in stratum_centers}
    if len(dimensions) != 1:
        raise ValueError(
            "centre drift requires equal feature dimensions across strata"
        )
    return _normalize(np.stack(stratum_centers).mean(axis=0))


def fixed_probe_curve(
    split: FixedProbeSplit,
    reference_counts: list[int],
    *,
    topk: int,
) -> list[dict[str, object]]:
    if not reference_counts:
        return []
    if reference_counts != sorted(set(reference_counts)):
        raise ValueError("reference_counts must be sorted and unique")
    if reference_counts[-1] > len(split.reference_tile_order):
        raise ValueError("reference count exceeds split reference pool")

    probe_by_stratum: dict[str, list[CurveObservation]] = defaultdict(list)
    for observation in split.probe_observations:
        probe_by_stratum[observation.stratum].append(observation)
    reference_by_stratum: dict[str, list[CurveObservation]] = defaultdict(
        list
    )
    reference_prefix_by_stratum: dict[str, list[int]] = {
        stratum: [] for stratum in probe_by_stratum
    }
    for tile_id in split.reference_tile_order:
        for observation in split.observations_by_tile[tile_id]:
            if observation.stratum in probe_by_stratum:
                reference_by_stratum[observation.stratum].append(
                    observation
                )
        for stratum in probe_by_stratum:
            reference_prefix_by_stratum[stratum].append(
                len(reference_by_stratum[stratum])
            )

    similarity_by_stratum: dict[str, np.ndarray] = {}
    for stratum, probe in probe_by_stratum.items():
        reference = reference_by_stratum[stratum]
        if not reference:
            similarity_by_stratum[stratum] = np.empty(
                (len(probe), 0),
                dtype=np.float32,
            )
            continue
        query = _normalize_rows(
            np.stack([observation.feature for observation in probe])
        )
        keys = _normalize_rows(
            np.stack(
                [observation.feature for observation in reference]
            )
        )
        similarity_by_stratum[stratum] = query @ keys.T

    rows: list[dict[str, object]] = []
    previous_center: np.ndarray | None = None
    previous_count = 0
    for count in reference_counts:
        current_tiles = split.reference_tile_order[:count]
        current = [
            observation
            for tile_id in current_tiles
            for observation in split.observations_by_tile[tile_id]
        ]
        per_stratum: dict[str, dict[str, float]] = {}
        stratum_means: list[float] = []
        stratum_q50: list[float] = []
        stratum_q95: list[float] = []
        for stratum in sorted(probe_by_stratum):
            available = reference_prefix_by_stratum[stratum][count - 1]
            novelty = _fixed_k_novelty_from_similarities(
                similarity_by_stratum[stratum],
                reference_observation_count=available,
                topk=topk,
            )
            mean = float(novelty.mean())
            q50 = float(np.quantile(novelty, 0.50))
            q95 = float(np.quantile(novelty, 0.95))
            per_stratum[stratum] = {
                "mean": mean,
                "q50": q50,
                "q95": q95,
                "probe_observation_count": len(
                    probe_by_stratum[stratum]
                ),
                "reference_observation_count": available,
            }
            stratum_means.append(mean)
            stratum_q50.append(q50)
            stratum_q95.append(q95)

        center = _slide_stratum_balanced_center(current)
        drift = (
            1.0 - float(np.dot(center, previous_center))
            if previous_center is not None
            else math.nan
        )
        row: dict[str, object] = {
            "sample_count": count,
            "new_tile_count": count - previous_count,
            "reference_observation_count": len(current),
            "reference_slide_count": len(
                {observation.slide_id for observation in current}
            ),
            "probe_observation_count": len(split.probe_observations),
            "probe_slide_count": split.probe_slide_count,
            "remaining_novelty_mean": _mean(stratum_means),
            "remaining_novelty_q50": _mean(stratum_q50),
            "remaining_novelty_q95": _mean(stratum_q95),
            "remaining_redundancy_mean": 1.0 - _mean(stratum_means),
            "center_drift": drift,
            "remaining_novelty_by_stratum": per_stratum,
        }
        if rows:
            previous_novelty = float(
                rows[-1]["remaining_novelty_mean"]
            )
            current_novelty = float(row["remaining_novelty_mean"])
            increase = current_novelty - previous_novelty
            if increase > 1e-6:
                raise AssertionError(
                    "fixed-probe remaining novelty increased: "
                    f"{previous_novelty} -> {current_novelty}"
                )
            gain = max(0.0, previous_novelty - current_novelty)
            row["information_gain"] = gain
            row["information_gain_per_100_tiles"] = (
                gain / (count - previous_count) * 100.0
            )
        else:
            row["information_gain"] = math.nan
            row["information_gain_per_100_tiles"] = math.nan
        rows.append(row)
        previous_count = count
        previous_center = center
    return rows


def aggregate_fixed_probe_curves(
    curves: list[list[dict[str, object]]],
) -> list[dict[str, object]]:
    if not curves:
        return []
    counts = [
        [int(row["sample_count"]) for row in curve]
        for curve in curves
    ]
    if any(values != counts[0] for values in counts[1:]):
        raise ValueError("fixed-probe curves do not share checkpoints")
    metrics = [
        "new_tile_count",
        "reference_observation_count",
        "reference_slide_count",
        "probe_observation_count",
        "probe_slide_count",
        "remaining_novelty_mean",
        "remaining_novelty_q50",
        "remaining_novelty_q95",
        "remaining_redundancy_mean",
        "center_drift",
        "information_gain",
        "information_gain_per_100_tiles",
    ]
    rows: list[dict[str, object]] = []
    for index, count in enumerate(counts[0]):
        source = [curve[index] for curve in curves]
        row: dict[str, object] = {"sample_count": count}
        for metric in metrics:
            values = [_as_finite(item.get(metric)) for item in source]
            row[metric] = _mean(values)
            row[f"{metric}_ci_low"] = _quantile(values, 0.025)
            row[f"{metric}_ci_high"] = _quantile(values, 0.975)
        rows.append(row)
    for previous, current in zip(rows, rows[1:]):
        if (
            float(current["remaining_novelty_mean"])
            > float(previous["remaining_novelty_mean"]) + 1e-6
        ):
            raise AssertionError(
                "aggregate fixed-probe remaining novelty is not monotone"
            )
    return rows


def _as_finite(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def tail_plateau(
    curve: list[dict[str, object]],
    *,
    marginal_ratio_threshold: float,
    drift_threshold: float,
    confirmation_increments: int,
) -> tuple[bool, int | None]:
    intervals = [
        row
        for row in curve
        if math.isfinite(
            _as_finite(row.get("information_gain_per_100_tiles"))
        )
    ]
    if len(intervals) < confirmation_increments:
        return False, None
    gains = [
        _as_finite(row["information_gain_per_100_tiles"])
        for row in intervals
    ]
    best_gain = max(gains, default=0.0)
    threshold = best_gain * marginal_ratio_threshold
    tail = intervals[-confirmation_increments:]
    stable = all(
        _as_finite(row["information_gain_per_100_tiles"])
        <= threshold + 1e-12
        and _as_finite(row["center_drift"]) <= drift_threshold
        for row in tail
    )
    onset = int(tail[0]["sample_count"]) if stable else None
    return stable, onset


def tail_low_gain(
    curve: list[dict[str, object]],
    *,
    marginal_ratio_threshold: float,
    confirmation_increments: int,
) -> tuple[bool, int | None]:
    """Confirm a low-gain tail without imposing a centroid model.

    This is the L2 spatial-annotation gate. Unlike L1 prototypes, an L2
    component is deliberately multi-modal, so movement of one global centre is
    not a scientifically valid stopping condition.
    """
    intervals = [
        row
        for row in curve
        if math.isfinite(
            _as_finite(row.get("information_gain_per_100_tiles"))
        )
    ]
    if len(intervals) < confirmation_increments:
        return False, None
    gains = [
        _as_finite(row["information_gain_per_100_tiles"])
        for row in intervals
    ]
    threshold = max(gains, default=0.0) * marginal_ratio_threshold
    tail = intervals[-confirmation_increments:]
    stable = all(
        _as_finite(row["information_gain_per_100_tiles"])
        <= threshold + 1e-12
        for row in tail
    )
    onset = int(tail[0]["sample_count"]) if stable else None
    return stable, onset
