from __future__ import annotations

import torch
import torch.nn.functional as F

from hcc_sempath.modeling.models import HCCSemPathModel
from hcc_sempath.training.engine import (
    PrototypeRefreshState,
    _maybe_refresh_prototypes,
    _refresh_global_prototypes,
)


def _prototype_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    positives: torch.Tensor,
) -> dict:
    size, component_count = positives.shape
    point = positives.view(size, component_count, 1, 1).float()
    zeros_bool = torch.zeros_like(point, dtype=torch.bool)
    return {
        "tile_id": [f"tile-{int(label)}" for label in labels],
        "images": images,
        "teacher_features": {
            "teacher": torch.stack(
                [
                    torch.tensor(
                        [float(label + 1), float(4 - label)]
                    )
                    for label in labels
                ]
            )
        },
        "prototype_mask": torch.ones(size, dtype=torch.bool),
        "prototype_level1": labels,
        "l2_point_centers": point,
        "l2_brush_bag_ids": torch.zeros_like(point, dtype=torch.long),
        "l2_area_positive": zeros_bool,
        "l2_explicit_negative": zeros_bool,
        "l2_implicit_negative": zeros_bool,
        "l2_spatial_supervised": positives.bool(),
    }


def test_full_bank_prototype_refresh_is_chunk_invariant_and_tracks_student() -> None:
    torch.manual_seed(31)
    model = HCCSemPathModel(
        backbone_name="vit_tiny_patch16_224",
        embedding_dim=8,
        teacher_dims={"teacher": 2},
        pretrained=False,
        l1_num_classes=4,
        spatial_num_components=2,
        spatial_dim=8,
    )
    images = torch.randn(4, 3, 224, 224)
    labels = torch.arange(4)
    positives = torch.tensor(
        [
            [True, False],
            [False, True],
            [False, False],
            [False, False],
        ]
    )
    loader = [
        _prototype_batch(images[:2], labels[:2], positives[:2]),
        _prototype_batch(images[2:], labels[2:], positives[2:]),
    ]
    cfg = {
        "data": {
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
        }
    }

    metrics = _refresh_global_prototypes(
        model,
        loader,
        cfg,
        torch.device("cpu"),
    )
    with torch.no_grad():
        expected = F.normalize(model.encode(images), dim=-1)
    first = model.l1_prototypes.clone()

    assert metrics["tiles"] == 4
    assert metrics["l1_observations"] == 4
    assert metrics["l2_positive_observations"] == 2
    torch.testing.assert_close(model.l1_prototype_counts, torch.ones(4))
    torch.testing.assert_close(
        model.global_l2_prototype_counts,
        torch.ones(2),
    )
    torch.testing.assert_close(first, expected)

    with torch.no_grad():
        model.encoder.projector[1].weight.add_(
            torch.randn_like(model.encoder.projector[1].weight) * 0.1
        )
    _refresh_global_prototypes(
        model,
        loader,
        cfg,
        torch.device("cpu"),
    )

    assert not torch.allclose(model.l1_prototypes, first)
    torch.testing.assert_close(model.l1_prototype_counts, torch.ones(4))


def test_prototype_response_updates_zhcc_but_not_detached_centroids() -> None:
    torch.manual_seed(37)
    model = HCCSemPathModel(
        backbone_name="vit_tiny_patch16_224",
        embedding_dim=8,
        teacher_dims={},
        pretrained=False,
        l1_num_classes=4,
    )
    model.replace_l1_prototypes(
        torch.randn(4, 8),
        torch.ones(4),
    )
    before = model.l1_prototypes.clone()
    outputs = model(
        torch.randn(3, 3, 224, 224),
        run_spatial=False,
    )
    loss = F.cross_entropy(
        outputs["l1_logits"],
        torch.tensor([0, 1, 2]),
    )

    loss.backward()

    projector_grad = model.encoder.projector[1].weight.grad
    assert projector_grad is not None
    assert float(projector_grad.abs().sum()) > 0
    assert model.l1_prototypes.grad is None
    torch.testing.assert_close(model.l1_prototypes, before)


def test_spatial_prototypes_refresh_on_global_step_clock(
    monkeypatch,
) -> None:
    refresh_steps: list[int] = []
    state = PrototypeRefreshState(
        global_loader=None,
        spatial_loader=object(),
    )

    def fake_refresh(model, loader, cfg, device):
        del model, loader, cfg, device
        refresh_steps.append(current_step[0])
        return {
            "tiles": 1,
            "positive_observations": 1,
            "seconds": 0.0,
        }

    monkeypatch.setattr(
        "hcc_sempath.training.engine._refresh_spatial_prototypes",
        fake_refresh,
    )
    cfg = {
        "train": {
            "dynamic_spatial_prototype_refresh_steps": 500,
        }
    }
    current_step = [0]
    for step in (0, 499, 500, 999, 1000, 2500, 2999, 3000):
        current_step[0] = step
        _maybe_refresh_prototypes(
            model=object(),
            cfg=cfg,
            device=torch.device("cpu"),
            state=state,
            global_step=step,
        )

    assert refresh_steps == [0, 500, 1000, 2500, 3000]
    assert state.last_spatial_global_step == 3000
