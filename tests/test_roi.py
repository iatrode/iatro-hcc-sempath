from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from hcc_sempath.training.roi import (
    build_spatial_roi_targets,
    geometry_token_mask,
    load_spatial_validation_metadata,
)
from hcc_sempath.training.spatial_losses import (
    _maximum_cardinality_score_matching,
    l1_classification_loss,
    spatial_morphometry_loss,
)


def test_spatial_geometry_supports_point_brush_circle_and_polygon() -> None:
    common = {"image_size": (224, 224), "grid_size": (32, 32)}
    geometries = [
        {"type": "point", "x": 112, "y": 112},
        {"type": "brush", "points": [[20, 20], [200, 200]], "width": 16},
        {"type": "circle", "center": [112, 112], "radius": 40},
        {"type": "polygon", "points": [[20, 20], [100, 20], [60, 100]]},
    ]
    for geometry in geometries:
        mask = geometry_token_mask(geometry, **common)
        assert mask.shape == (32, 32)
        assert mask.dtype == torch.bool
        assert mask.any()


def _write_records(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps(records), encoding="utf-8")


def test_masked_l1_loss_matches_selected_cross_entropy_and_handles_empty() -> None:
    logits = torch.tensor(
        [[2.0, -1.0], [-0.5, 1.5], [1.0, 0.0]],
        requires_grad=True,
    )
    mask = torch.tensor([True, False, True])
    targets = torch.tensor([0, -1, 1])
    loss, parts = l1_classification_loss(logits, mask, targets)
    expected = torch.nn.functional.cross_entropy(
        logits[mask],
        targets[mask],
    )
    torch.testing.assert_close(loss, expected)
    assert parts["l1_supervised_tiles"].item() == 2

    empty, empty_parts = l1_classification_loss(
        logits,
        torch.zeros(3, dtype=torch.bool),
        torch.full((3,), -1, dtype=torch.long),
    )
    assert empty.item() == 0.0
    assert empty_parts["l1_accuracy"].item() == 0.0


def test_validation_completeness_is_explicit_and_geometry_preserved(
    tmp_path: Path,
) -> None:
    path = tmp_path / "validation.json"
    names = [
        "hepatocellular-parenchyma-present",
        "necrosis-present",
    ]
    path.write_text(
        json.dumps(
            {
                "annotations": {
                    "tile": {
                        "tile_id": "tile",
                        "split": "val",
                        "roi_count_complete": [names[0]],
                        "roi_measurement_complete": {
                            names[0]: True,
                            names[1]: True,
                        },
                        "roi": [
                            {
                                "attribute": names[0],
                                "state": "positive",
                                "geometry": {
                                    "type": "point",
                                    "point": [0.5, 0.5],
                                    "coordinate_space": "normalized",
                                },
                            },
                            {
                                "attribute": names[1],
                                "state": "positive",
                                "geometry": {
                                    "type": "brush",
                                    "points": [[0.1, 0.1], [0.2, 0.2]],
                                    "width": 0.05,
                                    "coordinate_space": "normalized",
                                },
                            },
                        ],
                    },
                    "negative-tile": {
                        "tile_id": "negative-tile",
                        "split": "val",
                        "roi_measurement_complete": [names[1]],
                        "roi": [],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    metadata = load_spatial_validation_metadata(
        path,
        component_names=names,
        allowed_splits={"val"},
    )["tile"]

    assert metadata.count_complete.tolist() == [True, False]
    assert metadata.measurement_complete.tolist() == [True, True]
    assert metadata.geometry_modes == (("point",), ("brush",))
    targets = build_spatial_roi_targets(
        path,
        component_names=names,
        image_size=(224, 224),
        grid_size=(32, 32),
        allowed_splits={"val"},
    )
    assert "negative-tile" in targets


def test_brush_is_positive_bag_and_not_point_or_solid_area_target(
    tmp_path: Path,
) -> None:
    path = tmp_path / "spatial.json"
    _write_records(
        path,
        [
            {
                "tile_id": "dense",
                "attribute": "immune",
                "split": "train",
                "state": "positive",
                "geometry": {
                    "type": "brush",
                    "points": [[40, 40], [180, 180]],
                    "width": 20,
                },
            }
        ],
    )
    target = build_spatial_roi_targets(
        path,
        component_names=["immune"],
        image_size=(224, 224),
        grid_size=(32, 32),
        allowed_splits={"train"},
    )["dense"]

    assert not target.point_centers.any()
    assert target.brush_mask.any()
    assert target.brush_bag_ids.max().item() == 1
    assert not target.explicit_negative.any()
    assert not target.implicit_negative[target.brush_mask].any()
    assert target.implicit_negative.any()


def test_continuous_component_brush_is_area_only(tmp_path: Path) -> None:
    path = tmp_path / "spatial.json"
    _write_records(
        path,
        [
            {
                "tile_id": "necrosis",
                "attribute": "necrosis-present",
                "split": "train",
                "state": "positive",
                "geometry": {
                    "type": "brush",
                    "points": [[40, 40], [180, 180]],
                    "width": 20,
                },
            }
        ],
    )
    target = build_spatial_roi_targets(
        path,
        component_names=["necrosis-present"],
        image_size=(224, 224),
        grid_size=(32, 32),
    )["necrosis"]

    assert not target.point_centers.any()
    assert not target.brush_mask.any()
    assert target.area_positive.any()


def test_bile_pigment_all_geometries_supervise_burden_not_count(
    tmp_path: Path,
) -> None:
    path = tmp_path / "spatial.json"
    common = {
        "tile_id": "pigment",
        "attribute": "bile-pigment-present",
        "split": "train",
        "state": "positive",
    }
    _write_records(
        path,
        [
            {**common, "geometry": {"type": "point", "point": [30, 30]}},
            {
                **common,
                "geometry": {
                    "type": "circle",
                    "center": [100, 100],
                    "radius": 20,
                },
            },
            {
                **common,
                "geometry": {
                    "type": "brush",
                    "points": [[140, 140], [190, 190]],
                    "width": 18,
                },
            },
        ],
    )
    target = build_spatial_roi_targets(
        path,
        component_names=["bile-pigment-present"],
        image_size=(224, 224),
        grid_size=(32, 32),
    )["pigment"]

    assert not target.point_centers.any()
    assert not target.brush_mask.any()
    assert target.area_positive.any()


def test_large_structure_circle_and_brush_each_add_one_instance_with_area(
    tmp_path: Path,
) -> None:
    path = tmp_path / "spatial.json"
    common = {
        "tile_id": "duct",
        "attribute": "ductular-portal-present",
        "split": "train",
        "state": "positive",
    }
    _write_records(
        path,
        [
            {
                **common,
                "geometry": {
                    "type": "circle",
                    "center": [60, 60],
                    "radius": 18,
                },
            },
            {
                **common,
                "geometry": {
                    "type": "brush",
                    "points": [[140, 140], [190, 170]],
                    "width": 20,
                },
            },
        ],
    )
    target = build_spatial_roi_targets(
        path,
        component_names=["ductular-portal-present"],
        image_size=(224, 224),
        grid_size=(32, 32),
    )["duct"]

    assert target.point_centers.sum().item() == 2
    assert not target.brush_mask.any()
    assert target.area_positive.any()


def test_large_structure_point_has_center_without_invented_extent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "spatial.json"
    _write_records(
        path,
        [
            {
                "tile_id": "vacuole",
                "attribute": "steatosis-vacuolation-present",
                "split": "train",
                "state": "positive",
                "geometry": {"type": "point", "point": [112, 112]},
            }
        ],
    )
    target = build_spatial_roi_targets(
        path,
        component_names=["steatosis-vacuolation-present"],
        image_size=(224, 224),
        grid_size=(32, 32),
        point_tolerance_cells=1,
    )["vacuole"]

    assert target.point_centers.sum().item() == 1
    assert not target.area_positive.any()


def test_bile_point_is_one_burden_seed_not_a_tolerance_sized_extent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "spatial.json"
    _write_records(
        path,
        [
            {
                "tile_id": "pigment",
                "attribute": "bile-pigment-present",
                "split": "train",
                "state": "positive",
                "geometry": {"type": "point", "point": [112, 112]},
            }
        ],
    )
    target = build_spatial_roi_targets(
        path,
        component_names=["bile-pigment-present"],
        image_size=(224, 224),
        grid_size=(32, 32),
        point_tolerance_cells=1,
    )["pigment"]

    assert not target.point_centers.any()
    assert target.area_positive.sum().item() == 1


def test_connected_structure_brush_strokes_form_one_instance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "spatial.json"
    common = {
        "tile_id": "vessel",
        "attribute": "vascular-structure-present",
        "split": "train",
        "state": "positive",
    }
    _write_records(
        path,
        [
            {
                **common,
                "geometry": {
                    "type": "brush",
                    "points": [[40, 80], [120, 80]],
                    "width": 24,
                },
            },
            {
                **common,
                "geometry": {
                    "type": "brush",
                    "points": [[105, 80], [180, 115]],
                    "width": 24,
                },
            },
        ],
    )
    target = build_spatial_roi_targets(
        path,
        component_names=["vascular-structure-present"],
        image_size=(224, 224),
        grid_size=(32, 32),
    )["vessel"]

    assert target.point_centers.sum().item() == 1
    assert not target.brush_mask.any()
    assert target.area_positive.any()


def test_disconnected_structure_brush_strokes_remain_separate_instances(
    tmp_path: Path,
) -> None:
    path = tmp_path / "spatial.json"
    common = {
        "tile_id": "vessels",
        "attribute": "vascular-structure-present",
        "split": "train",
        "state": "positive",
    }
    _write_records(
        path,
        [
            {
                **common,
                "geometry": {
                    "type": "brush",
                    "points": [[25, 35], [65, 35]],
                    "width": 12,
                },
            },
            {
                **common,
                "geometry": {
                    "type": "brush",
                    "points": [[155, 175], [200, 175]],
                    "width": 12,
                },
            },
        ],
    )
    target = build_spatial_roi_targets(
        path,
        component_names=["vascular-structure-present"],
        image_size=(224, 224),
        grid_size=(32, 32),
    )["vessels"]

    assert target.point_centers.sum().item() == 2
    assert target.area_positive.any()


def test_point_uses_local_tolerance_and_weak_unmarked_background(
    tmp_path: Path,
) -> None:
    path = tmp_path / "spatial.json"
    _write_records(
        path,
        [
            {
                "tile_id": "sparse",
                "attribute": "immune",
                "split": "train",
                "state": "positive",
                "geometry": {"type": "point", "point": [112, 112]},
            }
        ],
    )
    target = build_spatial_roi_targets(
        path,
        component_names=["immune"],
        image_size=(224, 224),
        grid_size=(32, 32),
        allowed_splits={"train"},
        point_tolerance_cells=1,
    )["sparse"]

    assert target.point_centers.sum().item() == 1
    row, col = (target.point_centers[0] > 0).nonzero(as_tuple=False)[0]
    assert not target.implicit_negative[0, row, col]
    assert not target.implicit_negative[0, row - 1 : row + 2, col - 1 : col + 2].any()
    assert target.implicit_negative.any()
    assert not target.brush_mask.any()
    assert not target.explicit_negative.any()


def test_circle_is_one_large_instance_and_not_a_brush_bag(
    tmp_path: Path,
) -> None:
    path = tmp_path / "spatial.json"
    common = {
        "tile_id": "mixed",
        "attribute": "hemorrhage",
        "split": "train",
        "state": "positive",
    }
    _write_records(
        path,
        [
            {**common, "geometry": {"type": "point", "point": [35, 42]}},
            {
                **common,
                "geometry": {
                    "type": "circle",
                    "center": [150, 150],
                    "radius": 24,
                },
            },
        ],
    )
    target = build_spatial_roi_targets(
        path,
        component_names=["hemorrhage"],
        image_size=(224, 224),
        grid_size=(32, 32),
        allowed_splits={"train"},
    )["mixed"]

    assert target.point_centers.sum().item() == 2
    assert not target.brush_mask.any()
    assert not target.area_positive.any()
    assert not target.explicit_negative.any()


def test_explicit_negative_is_strong_and_absent_component_remains_ignore(
    tmp_path: Path,
) -> None:
    path = tmp_path / "spatial.json"
    _write_records(
        path,
        [
            {
                "tile_id": "negative",
                "attribute": "immune",
                "split": "train",
                "state": "negative",
                "review_complete": True,
                "geometry": None,
            }
        ],
    )
    target = build_spatial_roi_targets(
        path,
        component_names=["immune", "vascular"],
        image_size=(224, 224),
        grid_size=(32, 32),
        allowed_splits={"train"},
    )["negative"]

    assert target.explicit_negative[0].all()
    assert not target.implicit_negative[0].any()
    assert target.supervised.tolist() == [True, False]


def test_roi_reviewed_does_not_promote_unmentioned_components_to_negative(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "annotations": {
                    "one": {
                        "tile_id": "one",
                        "split": "train",
                        "roi_reviewed": True,
                        "roi_complete_all": False,
                        "roi": [
                            {
                                "attribute": "immune",
                                "state": "positive",
                                "geometry": {
                                    "type": "point",
                                    "point": [100, 100],
                                },
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    target = build_spatial_roi_targets(
        path,
        component_names=["immune", "vascular"],
        image_size=(224, 224),
        grid_size=(32, 32),
    )["one"]

    assert target.supervised.tolist() == [True, False]


def test_positive_and_explicit_negative_overlap_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "spatial.json"
    common = {
        "tile_id": "conflict",
        "attribute": "immune",
        "split": "train",
    }
    _write_records(
        path,
        [
            {
                **common,
                "state": "positive",
                "geometry": {"type": "point", "point": [112, 112]},
            },
            {
                **common,
                "state": "negative",
                "geometry": {
                    "type": "circle",
                    "center": [112, 112],
                    "radius": 20,
                },
            },
        ],
    )
    with pytest.raises(ValueError, match="overlap"):
        build_spatial_roi_targets(
            path,
            component_names=["immune"],
            image_size=(224, 224),
            grid_size=(32, 32),
        )


def test_spatial_loss_routes_point_brush_and_negative_gradients() -> None:
    instance_logits = torch.zeros((1, 1, 5, 5), requires_grad=True)
    abundance_logits = torch.zeros((1, 1, 5, 5), requires_grad=True)
    point_centers = torch.zeros_like(instance_logits)
    point_centers[0, 0, 1, 1] = 1
    brush_bag_ids = torch.zeros_like(instance_logits, dtype=torch.long)
    brush_bag_ids[0, 0, 3:, 3:] = 1
    area_positive = torch.zeros_like(instance_logits, dtype=torch.bool)
    explicit_negative = torch.zeros_like(instance_logits, dtype=torch.bool)
    explicit_negative[0, 0, 0, 4] = True
    implicit_negative = torch.zeros_like(instance_logits, dtype=torch.bool)
    implicit_negative[0, 0, 4, 0] = True

    loss, parts = spatial_morphometry_loss(
        instance_logits=instance_logits,
        abundance_logits=abundance_logits,
        point_centers=point_centers,
        brush_bag_ids=brush_bag_ids,
        area_positive=area_positive,
        explicit_negative=explicit_negative,
        implicit_negative=implicit_negative,
    )
    loss.backward()

    assert instance_logits.grad is not None
    assert instance_logits.grad.abs().sum() > 0
    assert abundance_logits.grad is not None
    assert abundance_logits.grad.abs().sum() > 0
    assert parts["l2_point_count"].item() == 1
    assert parts["l2_brush_bag_count"].item() == 1
    assert parts["l2_explicit_negative_pairs"].item() == 1
    assert parts["l2_implicit_negative_pairs"].item() == 1


def test_implicit_background_direct_gradient_is_twenty_times_weaker() -> None:
    def negative_gradient(*, explicit: bool) -> tuple[float, float]:
        instance = torch.zeros((1, 1, 3, 3), requires_grad=True)
        measurement = torch.zeros_like(instance, requires_grad=True)
        full = torch.ones_like(instance, dtype=torch.bool)
        empty = torch.zeros_like(full)
        loss, _ = spatial_morphometry_loss(
            instance_logits=instance,
            abundance_logits=measurement,
            point_centers=torch.zeros_like(instance),
            brush_bag_ids=torch.zeros_like(instance, dtype=torch.long),
            area_positive=empty,
            explicit_negative=full if explicit else empty,
            implicit_negative=empty if explicit else full,
            explicit_negative_weight=1.0,
            implicit_negative_weight=0.05,
        )
        loss.backward()
        gradient = instance.grad.abs().sum() + measurement.grad.abs().sum()
        return float(loss.detach()), float(gradient)

    explicit_loss, explicit_gradient = negative_gradient(explicit=True)
    implicit_loss, implicit_gradient = negative_gradient(explicit=False)

    assert explicit_loss == pytest.approx(20.0 * implicit_loss)
    assert explicit_gradient == pytest.approx(20.0 * implicit_gradient)


def test_area_only_loss_does_not_train_instance_channel() -> None:
    instance_logits = torch.zeros((1, 1, 5, 5), requires_grad=True)
    measurement_logits = torch.zeros((1, 1, 5, 5), requires_grad=True)
    zeros_float = torch.zeros_like(instance_logits)
    zeros_long = torch.zeros_like(instance_logits, dtype=torch.long)
    zeros_bool = torch.zeros_like(instance_logits, dtype=torch.bool)
    area_positive = zeros_bool.clone()
    area_positive[0, 0, 2, 2] = True

    loss, parts = spatial_morphometry_loss(
        instance_logits=instance_logits,
        abundance_logits=measurement_logits,
        point_centers=zeros_float,
        brush_bag_ids=zeros_long,
        area_positive=area_positive,
        explicit_negative=zeros_bool,
        implicit_negative=zeros_bool,
        component_names=["necrosis-present"],
    )
    loss.backward()

    assert instance_logits.grad is not None
    assert instance_logits.grad.abs().sum().item() == 0
    assert measurement_logits.grad is not None
    assert measurement_logits.grad.abs().sum().item() > 0
    assert parts["l2_area_supervised_pairs"].item() == 1


def test_structure_point_does_not_train_unknown_measurement_extent() -> None:
    instance_logits = torch.zeros((1, 1, 5, 5), requires_grad=True)
    measurement_logits = torch.zeros((1, 1, 5, 5), requires_grad=True)
    point = torch.zeros_like(instance_logits)
    point[0, 0, 2, 2] = 1
    implicit = torch.ones_like(instance_logits, dtype=torch.bool)
    implicit[0, 0, 1:4, 1:4] = False
    zeros_long = torch.zeros_like(instance_logits, dtype=torch.long)
    zeros_bool = torch.zeros_like(instance_logits, dtype=torch.bool)

    loss, _ = spatial_morphometry_loss(
        instance_logits=instance_logits,
        abundance_logits=measurement_logits,
        point_centers=point,
        brush_bag_ids=zeros_long,
        area_positive=zeros_bool,
        explicit_negative=zeros_bool,
        implicit_negative=implicit,
        component_names=["vascular-structure-present"],
        point_tolerance_cells=1,
    )
    loss.backward()

    assert instance_logits.grad is not None
    assert instance_logits.grad.abs().sum().item() > 0
    assert measurement_logits.grad is not None
    assert measurement_logits.grad.abs().sum().item() == 0


def test_structure_extent_penalizes_a_second_instance_peak() -> None:
    one_peak = torch.full((1, 1, 7, 7), -5.0)
    one_peak[0, 0, 3, 3] = 5.0
    two_peaks = one_peak.clone()
    two_peaks[0, 0, 1, 1] = 5.0
    point = torch.zeros_like(one_peak)
    point[0, 0, 3, 3] = 1
    area = torch.zeros_like(one_peak, dtype=torch.bool)
    area[0, 0, 1:6, 1:6] = True
    zeros_long = torch.zeros_like(one_peak, dtype=torch.long)
    zeros_bool = torch.zeros_like(one_peak, dtype=torch.bool)
    measurement = torch.zeros_like(one_peak)

    one_loss, _ = spatial_morphometry_loss(
        instance_logits=one_peak,
        abundance_logits=measurement,
        point_centers=point,
        brush_bag_ids=zeros_long,
        area_positive=area,
        explicit_negative=zeros_bool,
        implicit_negative=zeros_bool,
        component_names=["vascular-structure-present"],
        point_tolerance_cells=0,
    )
    two_loss, _ = spatial_morphometry_loss(
        instance_logits=two_peaks,
        abundance_logits=measurement,
        point_centers=point,
        brush_bag_ids=zeros_long,
        area_positive=area,
        explicit_negative=zeros_bool,
        implicit_negative=zeros_bool,
        component_names=["vascular-structure-present"],
        point_tolerance_cells=0,
    )

    assert two_loss > one_loss


def test_point_peak_accepts_response_within_tolerance() -> None:
    instance_logits = torch.full((1, 1, 5, 5), -5.0)
    instance_logits[0, 0, 2, 3] = 5.0
    abundance_logits = instance_logits.clone()
    point_centers = torch.zeros_like(instance_logits)
    point_centers[0, 0, 2, 2] = 1
    zeros_long = torch.zeros_like(instance_logits, dtype=torch.long)
    zeros_bool = torch.zeros_like(instance_logits, dtype=torch.bool)

    tolerant, _ = spatial_morphometry_loss(
        instance_logits=instance_logits,
        abundance_logits=abundance_logits,
        point_centers=point_centers,
        brush_bag_ids=zeros_long,
        area_positive=zeros_bool,
        explicit_negative=zeros_bool,
        implicit_negative=zeros_bool,
        point_tolerance_cells=1,
        abundance_point_weight=0.0,
    )
    exact, _ = spatial_morphometry_loss(
        instance_logits=instance_logits,
        abundance_logits=abundance_logits,
        point_centers=point_centers,
        brush_bag_ids=zeros_long,
        area_positive=zeros_bool,
        explicit_negative=zeros_bool,
        implicit_negative=zeros_bool,
        point_tolerance_cells=0,
        abundance_point_weight=0.0,
    )

    assert tolerant < exact


def test_adjacent_clicks_cannot_share_one_response_peak() -> None:
    one_peak = torch.full((1, 1, 5, 5), -5.0)
    one_peak[0, 0, 2, 2] = 5.0
    two_peaks = one_peak.clone()
    two_peaks[0, 0, 2, 3] = 5.0
    point_centers = torch.zeros_like(one_peak)
    point_centers[0, 0, 2, 1] = 1
    point_centers[0, 0, 2, 3] = 1
    zeros_long = torch.zeros_like(one_peak, dtype=torch.long)
    zeros_bool = torch.zeros_like(one_peak, dtype=torch.bool)

    shared, _ = spatial_morphometry_loss(
        instance_logits=one_peak,
        abundance_logits=one_peak,
        point_centers=point_centers,
        brush_bag_ids=zeros_long,
        area_positive=zeros_bool,
        explicit_negative=zeros_bool,
        implicit_negative=zeros_bool,
        point_tolerance_cells=1,
        abundance_point_weight=0.0,
    )
    distinct, _ = spatial_morphometry_loss(
        instance_logits=two_peaks,
        abundance_logits=two_peaks,
        point_centers=point_centers,
        brush_bag_ids=zeros_long,
        area_positive=zeros_bool,
        explicit_negative=zeros_bool,
        implicit_negative=zeros_bool,
        point_tolerance_cells=1,
        abundance_point_weight=0.0,
    )

    assert distinct < shared


def test_one_click_penalizes_a_second_peak_inside_its_tolerance() -> None:
    one_peak = torch.full((1, 1, 5, 5), -5.0)
    one_peak[0, 0, 2, 1] = 5.0
    two_peaks = one_peak.clone()
    two_peaks[0, 0, 2, 3] = 5.0
    point_centers = torch.zeros_like(one_peak)
    point_centers[0, 0, 2, 2] = 1
    zeros_long = torch.zeros_like(one_peak, dtype=torch.long)
    zeros_bool = torch.zeros_like(one_peak, dtype=torch.bool)

    one_loss, _ = spatial_morphometry_loss(
        instance_logits=one_peak,
        abundance_logits=one_peak,
        point_centers=point_centers,
        brush_bag_ids=zeros_long,
        area_positive=zeros_bool,
        explicit_negative=zeros_bool,
        implicit_negative=zeros_bool,
        point_tolerance_cells=1,
        abundance_point_weight=0.0,
    )
    two_loss, _ = spatial_morphometry_loss(
        instance_logits=two_peaks,
        abundance_logits=two_peaks,
        point_centers=point_centers,
        brush_bag_ids=zeros_long,
        area_positive=zeros_bool,
        explicit_negative=zeros_bool,
        implicit_negative=zeros_bool,
        point_tolerance_cells=1,
        abundance_point_weight=0.0,
    )

    assert two_loss > one_loss


def test_click_matching_maximizes_cardinality_before_peak_score() -> None:
    scarce = [0, 1, 3, 4]
    candidates = [list(scarce) for _ in range(4)]
    candidates.append([0, 1, 2, 3, 4, 5])
    scores = torch.tensor([[10.0, 9.0, -10.0], [8.0, 7.0, -9.0]])

    matched = _maximum_cardinality_score_matching(
        candidates,
        scores,
        width=3,
    )

    assert len(matched) == 5
    assert len(set(matched.values())) == 5
    assert matched[4] in {2, 5}
