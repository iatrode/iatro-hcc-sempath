from __future__ import annotations

import json
from pathlib import Path

import torch

from hcc_sempath.training.roi import build_roi_targets, geometry_token_mask
from hcc_sempath.training.zhcc_losses import roi_guided_level2_loss


def test_roi_geometry_supports_point_brush_circle_and_polygon() -> None:
    common = {"image_size": (224, 224), "grid_size": (14, 14)}
    geometries = [
        {"type": "point", "x": 112, "y": 112, "radius": 12},
        {"type": "brush", "points": [[20, 20], [200, 200]], "width": 16},
        {"type": "circle", "center": [112, 112], "radius": 40},
        {"type": "polygon", "points": [[20, 20], [100, 20], [60, 100]]},
    ]
    for geometry in geometries:
        mask = geometry_token_mask(geometry, **common)
        assert mask.shape == (14, 14)
        assert mask.any()


def test_partial_roi_ignores_unmarked_tokens_and_complete_review_creates_negatives(tmp_path: Path) -> None:
    path = tmp_path / "roi.json"
    path.write_text(
        json.dumps(
            [
                {
                    "tile_id": "partial",
                    "attribute": "necrosis",
                    "split": "train",
                    "state": "positive",
                    "geometry": {"type": "point", "x": 112, "y": 112, "radius": 12},
                },
                {
                    "tile_id": "complete",
                    "attribute": "necrosis",
                    "split": "train",
                    "review_complete": True,
                },
                {
                    "tile_id": "complete",
                    "attribute": "necrosis",
                    "split": "train",
                    "state": "positive",
                    "geometry": {"type": "circle", "center": [112, 112], "radius": 30},
                },
            ]
        ),
        encoding="utf-8",
    )
    targets = build_roi_targets(
        path,
        attribute_names=["necrosis"],
        image_size=(224, 224),
        grid_size=(14, 14),
        allowed_splits={"train"},
    )
    partial = targets["partial"]
    assert 0 < partial.valid.sum() < 14 * 14
    assert torch.equal(partial.target.bool(), partial.valid)
    complete = targets["complete"]
    assert complete.valid.all()
    assert complete.target.any()
    assert (~complete.target.bool()).any()


def test_roi_guided_loss_routes_local_signal_only_toward_global() -> None:
    patch_logits = torch.zeros((1, 2, 2, 2), requires_grad=True)
    local_logits = torch.tensor([[1.0, 0.0]], requires_grad=True)
    global_logits = torch.zeros((1, 2), requires_grad=True)
    target = torch.zeros_like(patch_logits)
    target[0, 0, 0, 0] = 1
    valid = torch.zeros_like(patch_logits, dtype=torch.bool)
    valid[0, 0, 0, 0] = True
    consistency = torch.tensor([[True, False]])

    roi_loss, consistency_loss, diagnostics = roi_guided_level2_loss(
        patch_logits=patch_logits,
        local_logits=local_logits,
        global_logits=global_logits,
        roi_target=target,
        roi_valid=valid,
        roi_consistency=consistency,
    )
    (roi_loss + consistency_loss).backward()
    assert patch_logits.grad is not None and patch_logits.grad.abs().sum() > 0
    assert global_logits.grad is not None and global_logits.grad.abs().sum() > 0
    assert local_logits.grad is None
    assert diagnostics["roi_valid_tokens"].item() == 1
    assert "roi_attr_0_activation_fraction" in diagnostics
    assert "roi_attr_0_all_zero_rate" in diagnostics
    assert "roi_attr_0_all_one_rate" in diagnostics
    assert "roi_attr_0_broadcast_rate" in diagnostics


def test_point_geometry_always_marks_its_patch_below_patch_radius() -> None:
    # A sub-patch radius must still mark the patch the click lands in; otherwise
    # off-center clicks on the DINOv2-S/14 grid silently rasterize to zero tokens.
    common = {"image_size": (224, 224), "grid_size": (16, 16)}
    for cx, cy in [(3, 3), (45, 7), (100, 100), (217, 9), (130, 200)]:
        geometry = {
            "type": "point",
            "coordinate_space": "pixel",
            "point": [cx, cy],
            "radius": 0.5,
        }
        mask = geometry_token_mask(geometry, **common)
        assert int(mask.sum()) >= 1
        assert mask[int(cy / 14), int(cx / 14)]


def test_degenerate_map_rates_count_only_roi_supervised_tiles() -> None:
    # One supervised tile with a healthy map; many unsupervised tiles pinned to an
    # all-zero/broadcast map must not pollute the degenerate-map diagnostics.
    patch_logits = torch.full((8, 1, 4, 4), -6.0)
    patch_logits[0] = torch.tensor([5.0, -5.0]).repeat(8).reshape(1, 4, 4)
    local_logits = torch.zeros((8, 1))
    global_logits = torch.zeros((8, 1))
    target = torch.zeros_like(patch_logits)
    valid = torch.zeros_like(patch_logits, dtype=torch.bool)
    valid[0] = True
    consistency = torch.zeros((8, 1), dtype=torch.bool)
    consistency[0] = True

    _, _, diagnostics = roi_guided_level2_loss(
        patch_logits=patch_logits,
        local_logits=local_logits,
        global_logits=global_logits,
        roi_target=target,
        roi_valid=valid,
        roi_consistency=consistency,
    )
    # The single supervised tile is non-degenerate, so all degenerate rates are 0
    # even though 7/8 unsupervised tiles look all-zero and broadcast.
    assert diagnostics["roi_attr_0_all_zero_rate"].item() == 0.0
    assert diagnostics["roi_attr_0_broadcast_rate"].item() == 0.0

