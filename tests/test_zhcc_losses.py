from __future__ import annotations

import pytest
import torch

from hcc_sempath.modeling.prototypes import PrototypeRegistry
from hcc_sempath.training.zhcc_losses import zhcc_prototype_loss


def _registry(attribute_count: int = 1) -> PrototypeRegistry:
    names = ["primary_tumor", "primary_non_tumor"] + [f"attribute_{idx}" for idx in range(attribute_count)]
    return PrototypeRegistry(
        prototypes=torch.randn(len(names), 4),
        names=names,
        groups=["primary", "primary"] + ["attribute"] * attribute_count,
        levels=[1, 1] + [2] * attribute_count,
        exclusive=[True, True] + [False] * attribute_count,
    )


def test_zhcc_prototype_loss_accepts_valid_two_level_labels() -> None:
    loss, parts = zhcc_prototype_loss(
        embedding_norm=torch.randn(2, 4),
        prototype_mask=torch.tensor([True, True]),
        prototype_level1=torch.tensor([0, 1]),
        prototype_level2=torch.tensor([[1.0], [0.0]]),
        prototypes=_registry(attribute_count=1),
    )

    assert loss.ndim == 0
    assert parts["zhcc_proto"].ndim == 0


def test_zhcc_prototype_loss_rejects_level1_out_of_range() -> None:
    with pytest.raises(ValueError, match="prototype_level1 target out of range"):
        zhcc_prototype_loss(
            embedding_norm=torch.randn(1, 4),
            prototype_mask=torch.tensor([True]),
            prototype_level1=torch.tensor([2]),
            prototype_level2=torch.tensor([[1.0]]),
            prototypes=_registry(attribute_count=1),
        )


def test_zhcc_prototype_loss_rejects_level2_width_mismatch() -> None:
    with pytest.raises(ValueError, match="prototype_level2 width mismatch"):
        zhcc_prototype_loss(
            embedding_norm=torch.randn(1, 4),
            prototype_mask=torch.tensor([True]),
            prototype_level1=torch.tensor([0]),
            prototype_level2=torch.tensor([[1.0, 0.0]]),
            prototypes=_registry(attribute_count=1),
        )


def test_zhcc_prototype_loss_allows_zero_width_level2_when_no_attributes() -> None:
    loss, parts = zhcc_prototype_loss(
        embedding_norm=torch.randn(1, 4),
        prototype_mask=torch.tensor([True]),
        prototype_level1=torch.tensor([0]),
        prototype_level2=torch.zeros((1, 0)),
        prototypes=_registry(attribute_count=0),
    )

    assert loss.ndim == 0
    assert parts["zhcc_l2"].item() == 0.0
