from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _load_module():
    script = Path(__file__).resolve().parents[1] / "research" / "scripts" / "roi_information_curve.py"
    spec = importlib.util.spec_from_file_location("roi_information_curve", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sample(
    module,
    index: int,
    slide: str,
    attribute: str = "a",
    teacher: str = "t1",
):
    angle = index * 0.01
    feature = np.asarray([1.0, angle, angle * angle], dtype=np.float32)
    feature /= np.linalg.norm(feature)
    return module.RoiFeatureSample(
        tile_id=f"t{index}",
        slide_id=slide,
        attribute=attribute,
        teacher=teacher,
        feature=feature,
    )


def test_slide_round_robin_prioritizes_independent_slides() -> None:
    module = _load_module()
    samples = [
        _sample(module, 0, "s1"),
        _sample(module, 1, "s1"),
        _sample(module, 2, "s1"),
        _sample(module, 3, "s2"),
        _sample(module, 4, "s2"),
        _sample(module, 5, "s3"),
    ]
    order = module._slide_round_robin(samples, seed=13)
    slide_by_tile = {sample.tile_id: sample.slide_id for sample in samples}
    assert {slide_by_tile[tile_id] for tile_id in order[:3]} == {"s1", "s2", "s3"}
    assert set(order) == {sample.tile_id for sample in samples}


def test_annotation_loader_uses_active_current_roi_taxonomy(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "annotations.json"
    path.write_text(
        json.dumps(
            {
                "spatial_prototypes": ["fallback"],
                "label_definitions": {
                    "spatial": [
                        {"id": "current-a", "active": True},
                        {"id": "retired", "active": False},
                    ]
                },
                "annotations": {
                    "one": {"tile_id": "t1", "split": "train", "roi": []},
                    "skip": {"tile_id": "t2", "split": "val", "roi": []},
                },
            }
        ),
        encoding="utf-8",
    )
    _, attributes, items = module._load_annotation_state(path)
    assert attributes == ["current-a"]
    assert list(items) == ["t1"]


def test_small_attribute_is_reported_not_assessable() -> None:
    module = _load_module()
    samples = [_sample(module, index, f"s{index}") for index in range(4)]
    coverage = [
        {
            "attribute": "a",
            "positive_tile_count": 4,
            "positive_slide_count": 4,
            "effective_positive_slide_count": 4.0,
            "max_positive_slide_share": 0.25,
            "positive_token_count": 4,
            "positive_tokens_per_tile_median": 1.0,
            "positive_tokens_per_tile_q25": 1.0,
            "positive_tokens_per_tile_q75": 1.0,
            "negative_reviewed_tile_count": 0,
            "negative_reviewed_slide_count": 0,
            "negative_reviewed_token_count": 0,
            "occupied_patch_fraction": 0.01,
            "occupancy_entropy": 0.0,
            "zero_token_geometry_count": 0,
            "zero_token_geometry_fraction": 0.0,
            "explicit_conflict_token_count": 0,
            "positive_geometry_count": 4,
            "point_geometry_count": 4,
            "brush_geometry_count": 0,
            "circle_geometry_count": 0,
            "polygon_geometry_count": 0,
            "review_complete_count": 0,
        }
    ]
    summary, curves, report = module.evaluate_information(
        samples,
        coverage,
        ["a"],
        requested_counts=[2, 4],
        seed=13,
        resamples=4,
        topk=2,
        elbow_ratio=0.35,
        min_slides=5,
        min_increments=3,
        support_threshold=0.8,
        max_zero_geometry_fraction=0.01,
    )
    assert summary[0]["status"] == "not_assessable"
    assert summary[0]["enough_now"] is False
    assert report["a"]["checkpoints"] == [2]
    assert len(curves) == 1


def test_curve_is_nested_and_reports_finite_metrics_after_first_increment() -> None:
    module = _load_module()
    samples = [_sample(module, index, f"s{index % 6}") for index in range(24)]
    curve = module._one_curve(
        samples,
        [4, 8, 12, 16],
        seed=9,
        topk=3,
    )
    assert [row["sample_count"] for row in curve] == [4, 8, 12, 16]
    novelty = [float(row["remaining_novelty_mean"]) for row in curve]
    assert all(
        current <= previous + 1e-6
        for previous, current in zip(novelty, novelty[1:])
    )
    assert all(
        np.isfinite(row["information_gain_per_100_tiles"])
        for row in curve[1:]
    )
    assert all(np.isfinite(row["center_drift"]) for row in curve[1:])


def test_parser_defaults_to_pretraining_audit_configuration() -> None:
    module = _load_module()
    args = module.build_parser().parse_args(
        ["--annotation-json", "/private/study/spatial_state.json"]
    )
    assert args.annotation_json == "/private/study/spatial_state.json"
    assert args.resamples == 16
    assert args.elbow_ratio == 0.35
    assert args.teacher_feature_packages == ""
    assert args.probe_slide_fraction == 0.20
    assert args.confirmation_increments == 2


def test_coverage_keeps_point_circle_and_brush_modalities_separate() -> None:
    module = _load_module()
    items = {
        "t1": {
            "tile_id": "t1",
            "slide": "s1",
            "roi": [
                {
                    "attribute": "a",
                    "state": "positive",
                    "geometry": {"type": "point", "point": [112, 112]},
                },
                {
                    "attribute": "a",
                    "state": "positive",
                    "geometry": {
                        "type": "circle",
                        "center": [160, 160],
                        "radius": 16,
                    },
                },
                {
                    "attribute": "a",
                    "state": "positive",
                    "geometry": {
                        "type": "brush",
                        "points": [[20, 20], [80, 80]],
                        "width": 12,
                    },
                },
            ],
        },
        "t2": {
            "tile_id": "t2",
            "slide": "s2",
            "roi": [
                {
                    "attribute": "a",
                    "state": "negative",
                    "review_complete": True,
                    "geometry": None,
                }
            ],
        },
    }
    rows, detail = module._coverage_rows(["a"], items)

    assert rows[0]["positive_tile_count"] == 1
    assert rows[0]["point_positive_tile_count"] == 1
    assert rows[0]["circle_positive_tile_count"] == 1
    assert rows[0]["brush_positive_tile_count"] == 1
    assert rows[0]["negative_reviewed_tile_count"] == 1
    assert detail["a"]["positive_tile_ids"] == ["t1"]


def test_curve_counts_unique_tiles_not_teacher_observations() -> None:
    module = _load_module()
    samples = [
        _sample(module, 0, "s1"),
        module.RoiFeatureSample(
            tile_id="t0",
            slide_id="s1",
            attribute="a",
            teacher="t2",
            feature=np.asarray([0.9, 0.1, 0.0], dtype=np.float32),
        ),
        _sample(module, 1, "s2"),
        _sample(module, 2, "s3"),
    ]
    order = module._slide_round_robin(samples, seed=3)

    assert len(order) == 3
    assert len(set(order)) == 3


def test_spatial_uses_the_same_pooled_tail_support_as_classification(
    monkeypatch,
) -> None:
    module = _load_module()
    samples = [
        _sample(module, index, f"s{index % 6}", teacher=teacher)
        for teacher in ("t1", "t2")
        for index in range(24)
    ]
    coverage = [
        {
            "attribute": "a",
            "positive_tile_count": 24,
            "positive_slide_count": 6,
            "effective_positive_slide_count": 6.0,
            "max_positive_slide_share": 1.0 / 6.0,
            "positive_token_count": 24,
            "positive_tokens_per_tile_median": 1.0,
            "positive_tokens_per_tile_q25": 1.0,
            "positive_tokens_per_tile_q75": 1.0,
            "negative_reviewed_tile_count": 0,
            "negative_reviewed_slide_count": 0,
            "negative_reviewed_token_count": 0,
            "occupied_patch_fraction": 0.1,
            "occupancy_entropy": 1.0,
            "zero_token_geometry_count": 0,
            "zero_token_geometry_fraction": 0.0,
            "explicit_conflict_token_count": 0,
            "positive_geometry_count": 24,
            "point_geometry_count": 24,
            "brush_geometry_count": 0,
            "circle_geometry_count": 0,
            "polygon_geometry_count": 0,
            "review_complete_count": 0,
        }
    ]
    call_count = 0

    def fake_tail_plateau(*args, **kwargs):
        nonlocal call_count
        teacher_index = (call_count // 4) % 2
        call_count += 1
        return (teacher_index == 0, 12 if teacher_index == 0 else None)

    monkeypatch.setattr(module, "tail_plateau", fake_tail_plateau)
    summary, _, report = module.evaluate_information(
        samples,
        coverage,
        ["a"],
        requested_counts=[4, 8, 12, 16],
        seed=13,
        resamples=4,
        topk=2,
        elbow_ratio=0.35,
        min_slides=5,
        min_increments=3,
        support_threshold=0.8,
    )

    assert summary[0]["status"] == "still_growing"
    assert report["a"]["tail_plateau_support_by_teacher_ratio"][
        "0.35"
    ] == {"t1": 1.0, "t2": 0.0}
    assert report["a"]["tail_plateau_support_by_ratio"]["0.35"] == 0.5
    assert all("center_drift" in row for row in report["a"]["curve"])
