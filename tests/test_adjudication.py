from __future__ import annotations

import pytest
import torch

from hcc_sempath.modeling.prototypes import PrototypeRegistry
from hcc_sempath.training.adjudication import (
    attribute_adjudicated_level2_target,
    prototype_adjudicated_teacher_weights,
)


def _registry(prototypes: torch.Tensor, names: list[str] | None = None) -> PrototypeRegistry:
    return PrototypeRegistry(
        prototypes=prototypes.float(),
        names=names or ["primary_tumor", "primary_non_tumor", "lymphocyte_rich"],
        groups=["primary_state", "primary_state", "immune"],
        levels=[1, 1, 2],
        exclusive=[True, True, False],
    )


def test_prototype_adjudication_prefers_prototype_label_consistent_teacher() -> None:
    prototypes = torch.eye(3)
    teacher_by_name = {
        "aligned": torch.tensor([[5.0, 0.0, 5.0], [4.0, 0.0, 4.0]]),
        "conflicting": torch.tensor([[0.0, 5.0, 0.0], [0.0, 4.0, 0.0]]),
    }

    alpha, diagnostics = prototype_adjudicated_teacher_weights(
        teacher_by_name=teacher_by_name,
        prototypes_by_teacher={"aligned": _registry(prototypes), "conflicting": _registry(prototypes)},
        zhcc_embedding_norm=torch.tensor([[5.0, 0.0, 5.0], [4.0, 0.0, 4.0]]),
        zhcc_prototypes=_registry(prototypes),
        prototype_mask=torch.tensor([True, False]),
        prototype_level1=torch.tensor([0, -1]),
        prototype_level2=torch.tensor([[1.0], [0.0]]),
        alpha_min=0.25,
        consensus_weight=0.0,
        prototype_label_weight=0.5,
        zhcc_response_weight=0.5,
    )

    assert alpha["aligned"][0] > alpha["conflicting"][0]
    assert alpha["aligned"].min() >= 0.25
    assert alpha["aligned"].max() <= 1.0
    assert "alpha_mean/aligned" in diagnostics
    assert "rejected_fraction_alpha_lt_0.5/conflicting" in diagnostics


def test_prototype_adjudication_aligns_prototype_names_not_tensor_order() -> None:
    zhcc_registry = _registry(torch.eye(3))
    reversed_registry = _registry(
        torch.eye(3)[[1, 0, 2]],
        names=["primary_non_tumor", "primary_tumor", "lymphocyte_rich"],
    )

    alpha, _ = prototype_adjudicated_teacher_weights(
        teacher_by_name={"teacher": torch.tensor([[5.0, 0.0, 5.0]])},
        prototypes_by_teacher={"teacher": reversed_registry},
        zhcc_embedding_norm=torch.tensor([[5.0, 0.0, 5.0]]),
        zhcc_prototypes=zhcc_registry,
        prototype_mask=torch.tensor([True]),
        prototype_level1=torch.tensor([0]),
        prototype_level2=torch.tensor([[1.0]]),
        alpha_min=0.25,
        consensus_weight=0.0,
        prototype_label_weight=0.5,
        zhcc_response_weight=0.5,
    )

    assert alpha["teacher"].item() > 0.75


def test_filter_strength_interpolates_raw_alpha() -> None:
    registry = _registry(torch.eye(3))
    kwargs = {
        "teacher_by_name": {"teacher": torch.tensor([[0.0, 5.0, 0.0]])},
        "prototypes_by_teacher": {"teacher": registry},
        "zhcc_embedding_norm": torch.tensor([[5.0, 0.0, 5.0]]),
        "zhcc_prototypes": registry,
        "prototype_mask": torch.tensor([False]),
        "prototype_level1": torch.tensor([-1]),
        "prototype_level2": torch.tensor([[0.0]]),
        "alpha_min": 0.25,
        "consensus_weight": 0.0,
        "prototype_label_weight": 0.0,
        "zhcc_response_weight": 1.0,
    }

    alpha_off, diag_off = prototype_adjudicated_teacher_weights(**kwargs, filter_strength=0.0)
    alpha_half, _ = prototype_adjudicated_teacher_weights(**kwargs, filter_strength=0.5)
    alpha_full, diag_full = prototype_adjudicated_teacher_weights(**kwargs, filter_strength=1.0)

    assert alpha_off["teacher"].item() == 1.0
    assert alpha_full["teacher"].item() == pytest.approx(diag_full["alpha_raw_mean/teacher"].item())
    assert alpha_full["teacher"].item() < alpha_half["teacher"].item() < alpha_off["teacher"].item()
    assert diag_off["alpha_effective_mean/teacher"].item() == 1.0


def test_prototype_label_weight_only_changes_labeled_tiles() -> None:
    registry = _registry(torch.eye(3))
    common = {
        "teacher_by_name": {"teacher": torch.tensor([[5.0, 0.0, 5.0], [5.0, 0.0, 5.0]])},
        "prototypes_by_teacher": {"teacher": registry},
        "zhcc_embedding_norm": torch.tensor([[5.0, 0.0, 5.0], [5.0, 0.0, 5.0]]),
        "zhcc_prototypes": registry,
        "prototype_mask": torch.tensor([True, False]),
        "prototype_level1": torch.tensor([1, -1]),
        "prototype_level2": torch.tensor([[0.0], [0.0]]),
        "alpha_min": 0.25,
        "consensus_weight": 0.5,
        "zhcc_response_weight": 0.5,
        "filter_strength": 1.0,
    }

    no_prototype_label, _ = prototype_adjudicated_teacher_weights(**common, prototype_label_weight=0.0)
    with_prototype_label, _ = prototype_adjudicated_teacher_weights(**common, prototype_label_weight=0.8)

    assert with_prototype_label["teacher"][0] < no_prototype_label["teacher"][0]
    assert abs(with_prototype_label["teacher"][1].item() - no_prototype_label["teacher"][1].item()) < 1e-6


def _attribute_registry() -> PrototypeRegistry:
    return PrototypeRegistry(
        prototypes=torch.eye(4),
        names=["primary_tumor", "primary_non_tumor", "necrosis", "bile"],
        groups=["primary", "primary", "attribute", "attribute"],
        levels=[1, 1, 2, 2],
        exclusive=[True, True, False, False],
    )


def test_attribute_level2_adjudication_is_attribute_specific() -> None:
    registry = _attribute_registry()
    teachers = {
        "mixed": torch.tensor([[5.0, 0.0, 5.0, 0.0]]),
        "reference": torch.tensor([[5.0, 0.0, 5.0, 5.0]]),
    }

    target = attribute_adjudicated_level2_target(
        teacher_by_name=teachers,
        prototypes_by_teacher={"mixed": registry, "reference": registry},
        target_registry=registry,
        teacher_attribute_prior={"mixed": {"necrosis": 1.0, "bile": 0.05}, "reference": {"necrosis": 1.0, "bile": 1.0}},
        prototype_mask=torch.tensor([True]),
        prototype_level2=torch.tensor([[1.0, 1.0]]),
        attribute_temperature=0.1,
    )

    assert target.target.shape == (1, 2)
    assert target.reliability_by_teacher["mixed"][0, 0] > target.reliability_by_teacher["mixed"][0, 1]
    assert target.target[0, 0] > 0.90
    assert target.target[0, 1] > 0.80
    assert "l2_attr_reliability_mean/mixed" in target.diagnostics


def test_attribute_level2_uncertainty_gate_penalizes_agreed_ambiguous_more_than_conflict() -> None:
    registry = _attribute_registry()
    ambiguous = {
        "a": torch.tensor([[5.0, 0.0, 0.0, 0.0]]),
        "b": torch.tensor([[5.0, 0.0, 0.0, 0.0]]),
    }
    conflicting = {
        "a": torch.tensor([[5.0, 0.0, 5.0, 0.0]]),
        "b": torch.tensor([[5.0, 0.0, -5.0, 0.0]]),
    }

    ambiguous_target = attribute_adjudicated_level2_target(
        teacher_by_name=ambiguous,
        prototypes_by_teacher={"a": registry, "b": registry},
        target_registry=registry,
        prototype_mask=torch.tensor([False]),
        prototype_level2=torch.tensor([[0.0, 0.0]]),
        attribute_temperature=0.1,
        uncertainty_eta=0.5,
    )
    conflicting_target = attribute_adjudicated_level2_target(
        teacher_by_name=conflicting,
        prototypes_by_teacher={"a": registry, "b": registry},
        target_registry=registry,
        prototype_mask=torch.tensor([False]),
        prototype_level2=torch.tensor([[0.0, 0.0]]),
        attribute_temperature=0.1,
        uncertainty_eta=0.5,
    )

    assert ambiguous_target.gate[0, 0] < conflicting_target.gate[0, 0]


def test_attribute_level2_anchor_sets_gate_and_complete_negative_weight() -> None:
    registry = _attribute_registry()
    teachers = {"teacher": torch.tensor([[5.0, 0.0, 0.0, 0.0]])}

    target = attribute_adjudicated_level2_target(
        teacher_by_name=teachers,
        prototypes_by_teacher={"teacher": registry},
        target_registry=registry,
        prototype_mask=torch.tensor([True]),
        prototype_level2=torch.tensor([[0.0, 1.0]]),
        attribute_temperature=0.1,
    )

    assert target.gate.flatten().tolist() == pytest.approx([1.0, 1.0])
    assert target.negative_weight[0, 0].item() == pytest.approx(1.0)
