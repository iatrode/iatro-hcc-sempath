from __future__ import annotations

import pytest
import torch

from hcc_sempath.modeling.prototypes import PrototypeRegistry
from hcc_sempath.training.zhcc_losses import (
    prototype_response,
    teacher_semantic_response_target,
    zhcc_prototype_loss,
    zhcc_response_distillation_loss,
)


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


def test_zhcc_response_distillation_uses_teacher_soft_targets_and_updates_student() -> None:
    registry = PrototypeRegistry(
        prototypes=torch.eye(4)[:3],
        names=["primary_tumor", "primary_non_tumor", "lymphocyte_rich"],
        groups=["primary", "primary", "attribute"],
        levels=[1, 1, 2],
        exclusive=[True, True, False],
    )
    teachers = {
        "a": torch.tensor([[2.0, 0.0, 1.0, 0.0], [0.0, 2.0, 0.0, 0.0]]),
        "b": torch.tensor([[2.0, 0.0, 1.0, 0.0], [0.0, 2.0, 0.0, 0.0]]),
    }
    target_primary, target_attributes = teacher_semantic_response_target(
        teacher_by_name=teachers,
        prototypes_by_teacher={"a": registry, "b": registry},
        target_registry=registry,
        primary_temperature=0.1,
        attribute_temperature=0.1,
    )
    student = torch.randn(2, 4, requires_grad=True)

    loss, parts = zhcc_response_distillation_loss(
        embedding_norm=student,
        prototypes=registry,
        target_primary=target_primary,
        target_attributes=target_attributes,
        primary_temperature=0.1,
        attribute_temperature=0.1,
    )
    loss.backward()

    assert loss.ndim == 0
    assert parts["zhcc_response"].ndim == 0
    assert target_primary.requires_grad is False
    assert student.grad is not None


def test_teacher_prototype_response_is_bounded_before_sigmoid_or_softmax() -> None:
    registry = PrototypeRegistry(
        prototypes=torch.eye(4)[:3],
        names=["primary_tumor", "primary_non_tumor", "lymphocyte_rich"],
        groups=["primary", "primary", "attribute"],
        levels=[1, 1, 2],
        exclusive=[True, True, False],
    )
    features = torch.tensor([[1000.0, 0.0, 1000.0, 0.0]], dtype=torch.float16)

    primary, attributes = prototype_response(
        features,
        registry,
        primary_names=["primary_tumor", "primary_non_tumor"],
        attribute_names=["lymphocyte_rich"],
        label="teacher",
        primary_temperature=0.1,
        attribute_temperature=0.1,
    )

    assert primary.dtype == torch.float32
    assert attributes.dtype == torch.float32
    assert primary.min().item() > 1e-4
    assert primary.max().item() < 1.0 - 1e-4
    assert attributes.min().item() > 1e-4
    assert attributes.max().item() < 1.0 - 1e-4


def test_zhcc_losses_remain_float32_with_half_precision_inputs() -> None:
    registry = PrototypeRegistry(
        prototypes=torch.eye(4)[:3],
        names=["primary_tumor", "primary_non_tumor", "lymphocyte_rich"],
        groups=["primary", "primary", "attribute"],
        levels=[1, 1, 2],
        exclusive=[True, True, False],
    )
    student = torch.tensor([[1000.0, 0.0, 1000.0, 0.0]], dtype=torch.float16, requires_grad=True)

    loss, _ = zhcc_prototype_loss(
        embedding_norm=student,
        prototype_mask=torch.tensor([True]),
        prototype_level1=torch.tensor([0]),
        prototype_level2=torch.tensor([[1.0]]),
        prototypes=registry,
        primary_temperature=0.1,
        attribute_temperature=0.1,
    )
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert student.grad is not None
