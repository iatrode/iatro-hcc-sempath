from __future__ import annotations

import json

import pytest
import torch

from hcc_sempath.modeling.models import SpatialMorphometryHead
from hcc_sempath.training.roi import build_spatial_roi_targets
from hcc_sempath.training.spatial_losses import (
    _brush_bag_pair_terms,
    _point_center_pair_terms,
    _routed_negative_loss,
    spatial_morphometry_loss,
)


def test_point_supervision_is_fixed_at_clicked_center() -> None:
    centers = torch.zeros((1, 1, 5, 5))
    centers[0, 0, 2, 2] = 1
    centered = torch.full_like(centers, -5.0)
    centered[0, 0, 2, 2] = 5.0
    shifted = torch.full_like(centers, -5.0)
    shifted[0, 0, 2, 3] = 5.0

    centered_pair, _, _ = _point_center_pair_terms(
        centered,
        centers,
        tolerance_cells=1,
    )
    shifted_pair, _, _ = _point_center_pair_terms(
        shifted,
        centers,
        tolerance_cells=1,
    )

    assert centered_pair.item() < shifted_pair.item()


def test_point_center_and_local_noncenter_receive_opposite_gradients() -> None:
    logits = torch.zeros((1, 1, 5, 5), requires_grad=True)
    centers = torch.zeros_like(logits)
    centers[0, 0, 2, 2] = 1
    pair, _, _ = _point_center_pair_terms(
        logits,
        centers,
        tolerance_cells=1,
    )
    pair.sum().backward()

    assert logits.grad[0, 0, 2, 2] < 0
    assert logits.grad[0, 0, 2, 3] > 0
    assert logits.grad[0, 0, 0, 0] == 0


def test_full_brush_support_trains_every_selected_cell() -> None:
    logits = torch.zeros((1, 1, 4, 4), requires_grad=True)
    bag_ids = torch.zeros_like(logits, dtype=torch.long)
    bag_ids[0, 0, 1:3, 1:3] = 1
    pair, supervised, bags, pairs = _brush_bag_pair_terms(
        logits,
        bag_ids,
        top_fraction=1.0,
    )
    pair.sum().backward()

    assert supervised.item()
    assert (bags, pairs) == (1, 1)
    assert torch.all(logits.grad[0, 0, 1:3, 1:3] < 0)
    assert torch.count_nonzero(logits.grad) == 4


def test_complete_negative_hard_tail_does_not_dilute_sparse_peak() -> None:
    instance = torch.zeros((1, 1, 32, 32), requires_grad=True)
    measurement = torch.zeros_like(instance, requires_grad=True)
    with torch.no_grad():
        instance[0, 0, 7, 9] = 8
        measurement[0, 0, 7, 9] = 8
    mask = torch.ones_like(instance, dtype=torch.bool)
    countable = torch.ones((1, 1, 1, 1), dtype=torch.bool)

    loss, _, _ = _routed_negative_loss(
        instance,
        measurement,
        mask,
        countable,
        hard_tail_topk=4,
        hard_tail_weight=0.5,
        sum_heads=True,
    )
    loss.backward()

    peak = instance.grad[0, 0, 7, 9].abs()
    median = instance.grad.abs().median()
    assert peak > 100 * median


def test_cell_circle_routes_center_and_full_measurement_support(
    tmp_path,
) -> None:
    manifest = tmp_path / "spatial.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "tile_id": "tile",
                    "attribute": "hemorrhage",
                    "split": "train",
                    "state": "positive",
                    "geometry": {
                        "type": "circle",
                        "center": [112, 112],
                        "radius": 28,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    target = build_spatial_roi_targets(
        manifest,
        component_names=["hemorrhage"],
        image_size=(224, 224),
        grid_size=(32, 32),
        allowed_splits={"train"},
    )["tile"]

    assert target.point_centers.sum().item() == pytest.approx(1)
    assert target.instance_exclusion_support.any()
    assert target.brush_mask.sum().item() > 1


def test_range_center_is_not_reused_as_measurement_point_seed(
    tmp_path,
) -> None:
    manifest = tmp_path / "spatial.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "tile_id": "tile",
                    "attribute": "hemorrhage",
                    "split": "train",
                    "state": "positive",
                    "geometry": {
                        "type": "circle",
                        "center": [112, 112],
                        "radius": 28,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    target = build_spatial_roi_targets(
        manifest,
        component_names=["hemorrhage"],
        image_size=(224, 224),
        grid_size=(32, 32),
        allowed_splits={"train"},
    )["tile"]
    logits = torch.zeros((1, 1, 32, 32))
    _, parts = spatial_morphometry_loss(
        instance_logits=logits,
        abundance_logits=logits,
        point_centers=target.point_centers[None],
        instance_exclusion_support=target.instance_exclusion_support[None],
        brush_bag_ids=target.brush_bag_ids[None],
        area_positive=target.area_positive[None],
        explicit_negative=target.explicit_negative[None],
        implicit_negative=target.implicit_negative[None],
        component_names=["hemorrhage"],
        brush_top_fraction=1.0,
    )

    assert parts["spatial_abundance_point"].item() == 0
    assert parts["spatial_brush_bag"].item() > 0


def test_range_center_is_not_reused_by_measurement_prototype() -> None:
    head = SpatialMorphometryHead(
        student_dim=2,
        component_count=1,
        spatial_dim=2,
    )
    features = torch.zeros((1, 2, 2, 2))
    features[0, :, 0, 0] = torch.tensor([1.0, 0.0])
    features[0, :, 1, 1] = torch.tensor([0.0, 1.0])
    point = torch.zeros((1, 1, 2, 2))
    point[0, 0, 0, 0] = 1
    exclusion = torch.zeros_like(point, dtype=torch.bool)
    exclusion[0, 0, 0, 0] = True
    brush = torch.zeros_like(point, dtype=torch.long)
    brush[0, 0, 1, 1] = 1
    empty = torch.zeros_like(point, dtype=torch.bool)

    observations = head.prototype_observation_sums(
        features,
        point_centers=point,
        instance_exclusion_support=exclusion,
        brush_bag_ids=brush,
        area_positive=empty,
        explicit_negative=empty,
        implicit_negative=empty,
    )

    torch.testing.assert_close(
        observations["instance"][0][0],
        torch.tensor([1.0, 0.0]),
    )
    torch.testing.assert_close(
        observations["measurement"][0][0],
        torch.tensor([0.0, 1.0]),
    )
