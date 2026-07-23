from __future__ import annotations

import math

import numpy as np

from hcc_sempath.annotation_information import (
    CurveObservation,
    fixed_probe_curve,
    meaningful_reference_checkpoints,
    prepare_fixed_probe_split,
    tail_plateau,
)


def _observations() -> list[CurveObservation]:
    return [
        CurveObservation(
            tile_id=f"tile-{index}",
            slide_id=f"slide-{index}",
            stratum="cell",
            feature=np.asarray(
                [1.0, index / 10.0, (index / 10.0) ** 2],
                dtype=np.float32,
            ),
        )
        for index in range(8)
    ]


def test_fixed_probe_is_slide_disjoint_and_curve_is_monotone() -> None:
    split = prepare_fixed_probe_split(
        _observations(),
        seed=13,
        probe_slide_fraction=0.25,
        min_probe_slides=2,
    )
    probe_slides = {
        observation.slide_id for observation in split.probe_observations
    }
    reference_slides = {
        observation.slide_id
        for tile_id in split.reference_tile_order
        for observation in split.observations_by_tile[tile_id]
    }

    assert probe_slides.isdisjoint(reference_slides)

    curve = fixed_probe_curve(
        split,
        list(range(1, len(split.reference_tile_order) + 1)),
        topk=3,
    )
    remaining = [
        float(row["remaining_novelty_mean"]) for row in curve
    ]
    gains = [
        float(row["information_gain"]) for row in curve[1:]
    ]

    assert all(
        current <= previous + 1e-6
        for previous, current in zip(remaining, remaining[1:])
    )
    assert all(gain >= 0.0 for gain in gains)


def test_tiny_reference_remainder_is_not_a_tail_confirmation_step() -> None:
    assert meaningful_reference_checkpoints(
        181,
        [100, 120, 140, 160, 180],
    ) == [100, 120, 140, 160, 180]
    assert meaningful_reference_checkpoints(
        190,
        [100, 120, 140, 160, 180],
    ) == [100, 120, 140, 160, 180, 190]


def test_tail_confirmation_rejects_a_late_gain_rebound() -> None:
    curve = [
        {
            "sample_count": 10,
            "information_gain_per_100_tiles": math.nan,
            "center_drift": math.nan,
        },
        {
            "sample_count": 20,
            "information_gain_per_100_tiles": 10.0,
            "center_drift": 0.001,
        },
        {
            "sample_count": 30,
            "information_gain_per_100_tiles": 1.0,
            "center_drift": 0.001,
        },
        {
            "sample_count": 40,
            "information_gain_per_100_tiles": 8.0,
            "center_drift": 0.001,
        },
    ]

    stable, onset = tail_plateau(
        curve,
        marginal_ratio_threshold=0.35,
        drift_threshold=0.01,
        confirmation_increments=2,
    )

    assert stable is False
    assert onset is None


def test_tail_confirmation_accepts_only_consecutive_low_gain_tail() -> None:
    curve = [
        {
            "sample_count": 10,
            "information_gain_per_100_tiles": math.nan,
            "center_drift": math.nan,
        },
        {
            "sample_count": 20,
            "information_gain_per_100_tiles": 10.0,
            "center_drift": 0.001,
        },
        {
            "sample_count": 30,
            "information_gain_per_100_tiles": 2.0,
            "center_drift": 0.001,
        },
        {
            "sample_count": 40,
            "information_gain_per_100_tiles": 1.0,
            "center_drift": 0.001,
        },
    ]

    stable, onset = tail_plateau(
        curve,
        marginal_ratio_threshold=0.35,
        drift_threshold=0.01,
        confirmation_increments=2,
    )

    assert stable is True
    assert onset == 30
