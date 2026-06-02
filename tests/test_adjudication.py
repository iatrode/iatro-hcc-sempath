from __future__ import annotations

import torch

from hcc_sempath.modeling.prototypes import PrototypeRegistry
from hcc_sempath.training.adjudication import prototype_adjudicated_teacher_weights


def _registry(prototypes: torch.Tensor, names: list[str] | None = None) -> PrototypeRegistry:
    return PrototypeRegistry(
        prototypes=prototypes.float(),
        names=names or ["primary_tumor", "primary_non_tumor", "lymphocyte_rich"],
        groups=["primary_state", "primary_state", "immune"],
        levels=[1, 1, 2],
        exclusive=[True, True, False],
    )


def test_prototype_adjudication_prefers_anchor_consistent_teacher() -> None:
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
        anchor_weight=0.5,
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
        anchor_weight=0.5,
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
        "anchor_weight": 0.0,
        "zhcc_response_weight": 1.0,
    }

    alpha_off, diag_off = prototype_adjudicated_teacher_weights(**kwargs, filter_strength=0.0)
    alpha_half, _ = prototype_adjudicated_teacher_weights(**kwargs, filter_strength=0.5)
    alpha_full, diag_full = prototype_adjudicated_teacher_weights(**kwargs, filter_strength=1.0)

    assert alpha_off["teacher"].item() == 1.0
    assert alpha_full["teacher"].item() == diag_full["alpha_raw_mean/teacher"].item()
    assert alpha_full["teacher"].item() < alpha_half["teacher"].item() < alpha_off["teacher"].item()
    assert diag_off["alpha_effective_mean/teacher"].item() == 1.0


def test_anchor_weight_only_changes_anchor_tiles() -> None:
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

    no_anchor, _ = prototype_adjudicated_teacher_weights(**common, anchor_weight=0.0)
    with_anchor, _ = prototype_adjudicated_teacher_weights(**common, anchor_weight=0.8)

    assert with_anchor["teacher"][0] < no_anchor["teacher"][0]
    assert abs(with_anchor["teacher"][1].item() - no_anchor["teacher"][1].item()) < 1e-6
