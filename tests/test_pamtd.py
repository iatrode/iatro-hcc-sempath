from __future__ import annotations

import pytest
import torch

from hcc_sempath.modeling.models import clamp_probability
from hcc_sempath.modeling.prototypes import PrototypeRegistry
from hcc_sempath.spatial_schema import DEFAULT_SPATIAL_COMPONENTS
from hcc_sempath.training.losses import multi_teacher_distillation_loss
from hcc_sempath.training.pamtd import (
    prototype_adjudicated_teacher_target,
    prototype_response_distillation_loss,
)
from hcc_sempath.training.prototype_labels import DEFAULT_CLASSIFICATION_CLASSES


CLASSIFICATION_COUNT = len(DEFAULT_CLASSIFICATION_CLASSES)
SPATIAL_COUNT = len(DEFAULT_SPATIAL_COMPONENTS)


def _registry() -> PrototypeRegistry:
    return PrototypeRegistry(
        prototypes=torch.eye(CLASSIFICATION_COUNT),
        names=list(DEFAULT_CLASSIFICATION_CLASSES),
    )


def test_pamtd_returns_per_tile_teacher_weights_and_classification_target() -> None:
    teacher_by_name = {
        "a": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
        ),
        "b": torch.tensor(
            [[0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.8, 0.2, 0.0, 0.0, 0.0]]
        ),
    }
    result = prototype_adjudicated_teacher_target(
        teacher_by_name=teacher_by_name,
        prototypes_by_teacher={"a": _registry(), "b": _registry()},
        student_classification_response=torch.full(
            (2, CLASSIFICATION_COUNT),
            1.0 / CLASSIFICATION_COUNT,
        ),
        class_names=DEFAULT_CLASSIFICATION_CLASSES,
        classification_mask=torch.tensor([True, False]),
        classification_target=torch.tensor([0, -1]),
        filter_strength=1.0,
        alpha_min=0.25,
        classification_temperature=0.1,
    )

    assert set(result.teacher_sample_weights) == {"a", "b"}
    for weight in result.teacher_sample_weights.values():
        assert weight.shape == (2,)
        assert torch.all((weight >= 0.25) & (weight <= 1.0))
    assert result.classification_target.shape == (2, CLASSIFICATION_COUNT)
    assert result.response_sample_weight.shape == (2,)
    assert result.classification_target[0].argmax().item() == 0
    torch.testing.assert_close(
        result.classification_target.sum(dim=1),
        torch.ones(2),
    )


def test_spatial_component_response_only_adjudicates_teacher_reliability() -> None:
    teacher = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
    )
    spatial_prototypes = torch.stack(
        [
            torch.roll(
                torch.eye(CLASSIFICATION_COUNT)[0],
                shifts=index % CLASSIFICATION_COUNT,
            )
            for index in range(SPATIAL_COUNT)
        ]
    )
    result = prototype_adjudicated_teacher_target(
        teacher_by_name={"a": teacher},
        prototypes_by_teacher={"a": _registry()},
        student_classification_response=torch.full(
            (2, CLASSIFICATION_COUNT),
            1.0 / CLASSIFICATION_COUNT,
        ),
        class_names=DEFAULT_CLASSIFICATION_CLASSES,
        teacher_spatial_prototypes={
            "a": (spatial_prototypes, torch.ones(SPATIAL_COUNT))
        },
        student_spatial_response=torch.full((2, SPATIAL_COUNT), 0.5),
        spatial_target=torch.zeros((2, SPATIAL_COUNT)),
        spatial_known=torch.ones((2, SPATIAL_COUNT), dtype=torch.bool),
        classification_temperature=0.1,
        spatial_temperature=0.1,
    )

    # The deployable target remains the fixed classification response. spatial contributes
    # only to the teacher reliability used by PAMT-D.
    assert result.classification_target.shape == (2, CLASSIFICATION_COUNT)
    assert result.teacher_sample_weights["a"].shape == (2,)

    population_result = prototype_adjudicated_teacher_target(
        teacher_by_name={"a": teacher},
        prototypes_by_teacher={"a": _registry()},
        student_classification_response=torch.full(
            (2, CLASSIFICATION_COUNT),
            1.0 / CLASSIFICATION_COUNT,
        ),
        class_names=DEFAULT_CLASSIFICATION_CLASSES,
        teacher_spatial_prototypes={
            "a": (spatial_prototypes, torch.ones(SPATIAL_COUNT))
        },
        student_spatial_response=torch.full((2, SPATIAL_COUNT), 0.5),
        classification_temperature=0.1,
        spatial_temperature=0.1,
    )
    assert population_result.classification_target.shape == (2, CLASSIFICATION_COUNT)


def test_uninitialized_spatial_prototypes_are_exactly_classification_only() -> None:
    teacher = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
    )
    common = {
        "teacher_by_name": {"a": teacher},
        "prototypes_by_teacher": {"a": _registry()},
        "student_classification_response": torch.full(
            (2, CLASSIFICATION_COUNT),
            1.0 / CLASSIFICATION_COUNT,
        ),
        "class_names": DEFAULT_CLASSIFICATION_CLASSES,
        "classification_temperature": 0.1,
        "spatial_temperature": 0.1,
    }
    baseline = prototype_adjudicated_teacher_target(**common)
    with_empty_spatial = prototype_adjudicated_teacher_target(
        **common,
        teacher_spatial_prototypes={
            "a": (
                torch.randn(SPATIAL_COUNT, CLASSIFICATION_COUNT),
                torch.zeros(SPATIAL_COUNT),
            )
        },
        student_spatial_response=torch.rand(2, SPATIAL_COUNT),
    )

    torch.testing.assert_close(
        with_empty_spatial.classification_target,
        baseline.classification_target,
    )
    torch.testing.assert_close(
        with_empty_spatial.response_sample_weight,
        baseline.response_sample_weight,
    )
    torch.testing.assert_close(
        with_empty_spatial.teacher_sample_weights["a"],
        baseline.teacher_sample_weights["a"],
    )


def test_response_distillation_does_not_apply_fixed_temperature_twice() -> None:
    logits = torch.tensor([[2.0, -1.0]])
    target = torch.tensor([[0.25, 0.75]])
    temperature = 0.5

    actual = prototype_response_distillation_loss(
        logits,
        target,
        temperature=temperature,
    )
    expected = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(logits, dim=-1),
        clamp_probability(target, normalize=True),
        reduction="batchmean",
    ) * temperature**2

    torch.testing.assert_close(actual, expected)


def test_teacher_alpha_changes_cross_teacher_feature_gradient_weight() -> None:
    common = {
        "student_by_teacher": {
            "bad": torch.tensor([[1.0, 0.0]], requires_grad=True),
            "good": torch.tensor([[1.0, 0.0]], requires_grad=True),
        },
        "teacher_by_name": {
            "bad": torch.tensor([[0.0, 1.0]]),
            "good": torch.tensor([[1.0, 0.0]]),
        },
        "prototypes_by_teacher": None,
        "relation_weight": 0.0,
        "semantic_weight": 0.0,
        "semantic_temperature": 1.0,
    }
    uniform, _ = multi_teacher_distillation_loss(
        **common,
        teacher_sample_weights={
            "bad": torch.ones(1),
            "good": torch.ones(1),
        },
    )
    downweighted, _ = multi_teacher_distillation_loss(
        **common,
        teacher_sample_weights={
            "bad": torch.full((1,), 0.25),
            "good": torch.ones(1),
        },
    )

    assert uniform.item() == pytest.approx(0.5)
    assert downweighted.item() == pytest.approx(0.2)
    assert downweighted < uniform


def test_base_teacher_weight_and_tile_reliability_share_one_normalizer() -> None:
    total, parts = multi_teacher_distillation_loss(
        student_by_teacher={
            "bad": torch.tensor([[1.0, 0.0]], requires_grad=True),
            "good": torch.tensor([[1.0, 0.0]], requires_grad=True),
        },
        teacher_by_name={
            "bad": torch.tensor([[0.0, 1.0]]),
            "good": torch.tensor([[1.0, 0.0]]),
        },
        prototypes_by_teacher=None,
        relation_weight=0.0,
        semantic_weight=0.0,
        semantic_temperature=1.0,
        teacher_weights={"bad": 2.0, "good": 1.0},
        teacher_sample_weights={
            "bad": torch.tensor([0.25]),
            "good": torch.tensor([1.0]),
        },
    )

    # bad mass = 2 * .25 = .5; good mass = 1 * 1 = 1.
    assert parts["feature"].item() == pytest.approx(1.0 / 3.0)
    torch.testing.assert_close(total, parts["feature"])


def test_teacher_alpha_jointly_normalizes_relation_pairs() -> None:
    common = {
        "student_by_teacher": {
            "bad": torch.tensor(
                [[1.0, 0.0], [1.0, 0.0]],
                requires_grad=True,
            ),
            "good": torch.tensor(
                [[1.0, 0.0], [0.0, 1.0]],
                requires_grad=True,
            ),
        },
        "teacher_by_name": {
            "bad": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "good": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        },
        "prototypes_by_teacher": None,
        "relation_weight": 1.0,
        "semantic_weight": 0.0,
        "semantic_temperature": 1.0,
    }
    _, uniform_parts = multi_teacher_distillation_loss(
        **common,
        teacher_sample_weights={
            "bad": torch.ones(2),
            "good": torch.ones(2),
        },
    )
    _, downweighted_parts = multi_teacher_distillation_loss(
        **common,
        teacher_sample_weights={
            "bad": torch.full((2,), 0.25),
            "good": torch.ones(2),
        },
    )

    assert uniform_parts["relation"].item() == pytest.approx(0.25)
    assert downweighted_parts["relation"].item() == pytest.approx(
        0.125 / 4.25
    )


def test_teacher_alpha_jointly_normalizes_semantic_kl() -> None:
    common = {
        "student_by_teacher": {
            "bad": torch.tensor(
                [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
                requires_grad=True,
            ),
            "good": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                requires_grad=True,
            ),
        },
        "teacher_by_name": {
            "bad": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            "good": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        },
        "prototypes_by_teacher": {
            "bad": _registry(),
            "good": _registry(),
        },
        "relation_weight": 0.0,
        "semantic_weight": 1.0,
        "semantic_temperature": 1.0,
    }
    _, uniform_parts = multi_teacher_distillation_loss(
        **common,
        teacher_sample_weights={
            "bad": torch.ones(1),
            "good": torch.ones(1),
        },
    )
    _, downweighted_parts = multi_teacher_distillation_loss(
        **common,
        teacher_sample_weights={
            "bad": torch.full((1,), 0.25),
            "good": torch.ones(1),
        },
    )

    assert downweighted_parts["semantic"] < uniform_parts["semantic"]


def test_response_distillation_uses_adjudicated_tile_mass() -> None:
    logits = torch.tensor([[3.0, -3.0], [-3.0, 3.0]])
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    loss = prototype_response_distillation_loss(
        logits,
        target,
        temperature=0.5,
        sample_weight=torch.tensor([1.0, 0.0]),
    )
    expected = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(logits[:1], dim=-1),
        clamp_probability(target[:1], normalize=True),
        reduction="batchmean",
    ) * 0.5**2
    torch.testing.assert_close(loss, expected)


def test_zero_weight_teacher_cannot_change_consensus_or_target() -> None:
    common = {
        "prototypes_by_teacher": {"a": _registry()},
        "student_classification_response": torch.full(
            (1, CLASSIFICATION_COUNT),
            1.0 / CLASSIFICATION_COUNT,
        ),
        "class_names": DEFAULT_CLASSIFICATION_CLASSES,
        "teacher_weights": {"a": 1.0},
        "classification_temperature": 0.1,
    }
    baseline = prototype_adjudicated_teacher_target(
        teacher_by_name={"a": torch.eye(CLASSIFICATION_COUNT)[:1]},
        **common,
    )
    with_zero = prototype_adjudicated_teacher_target(
        teacher_by_name={
            "a": torch.eye(CLASSIFICATION_COUNT)[:1],
            "ignored": torch.eye(CLASSIFICATION_COUNT)[-1:],
        },
        prototypes_by_teacher={
            "a": _registry(),
            "ignored": _registry(),
        },
        student_classification_response=common["student_classification_response"],
        class_names=DEFAULT_CLASSIFICATION_CLASSES,
        teacher_weights={"a": 1.0, "ignored": 0.0},
        classification_temperature=0.1,
    )

    torch.testing.assert_close(
        with_zero.classification_target,
        baseline.classification_target,
    )
    torch.testing.assert_close(
        with_zero.teacher_sample_weights["a"],
        baseline.teacher_sample_weights["a"],
    )
    assert torch.count_nonzero(
        with_zero.teacher_sample_weights["ignored"]
    ).item() == 0


def test_zero_reliability_mass_has_valid_fallback_but_no_response_weight() -> None:
    result = prototype_adjudicated_teacher_target(
        teacher_by_name={
            "a": torch.eye(CLASSIFICATION_COUNT)[:1],
            "b": torch.eye(CLASSIFICATION_COUNT)[1:2],
        },
        prototypes_by_teacher={"a": _registry(), "b": _registry()},
        student_classification_response=torch.full(
            (1, CLASSIFICATION_COUNT),
            1.0 / CLASSIFICATION_COUNT,
        ),
        class_names=DEFAULT_CLASSIFICATION_CLASSES,
        filter_strength=1.0,
        alpha_min=0.0,
        consensus_weight=0.0,
        prototype_label_weight=0.0,
        student_agreement_weight=0.0,
        classification_temperature=0.1,
    )

    torch.testing.assert_close(
        result.classification_target.sum(dim=1),
        torch.ones(1),
    )
    assert result.response_sample_weight.item() == 0.0
    assert all(
        weight.item() == 0.0
        for weight in result.teacher_sample_weights.values()
    )


def test_multi_teacher_rejects_zero_total_alpha_and_invalid_base_weights() -> None:
    kwargs = {
        "student_by_teacher": {
            "a": torch.tensor([[1.0, 0.0]], requires_grad=True)
        },
        "teacher_by_name": {"a": torch.tensor([[0.0, 1.0]])},
        "prototypes_by_teacher": None,
        "relation_weight": 0.0,
        "semantic_weight": 0.0,
        "semantic_temperature": 1.0,
    }
    with pytest.raises(ValueError, match="zero total mass"):
        multi_teacher_distillation_loss(
            **kwargs,
            teacher_sample_weights={"a": torch.zeros(1)},
        )
    for value in (-1.0, float("nan")):
        with pytest.raises(ValueError, match="finite and non-negative"):
            multi_teacher_distillation_loss(
                **kwargs,
                teacher_weights={"a": value},
            )
    with pytest.raises(ValueError, match="finite and non-negative"):
        multi_teacher_distillation_loss(
            **kwargs,
            teacher_sample_weights={"a": torch.tensor([float("nan")])},
        )


def test_teacher_targets_and_alpha_are_gradient_stops() -> None:
    student = torch.tensor([[1.0, 0.0]], requires_grad=True)
    teacher = torch.tensor([[0.0, 1.0]], requires_grad=True)
    alpha = torch.tensor([0.5], requires_grad=True)
    loss, _ = multi_teacher_distillation_loss(
        student_by_teacher={"a": student},
        teacher_by_name={"a": teacher},
        prototypes_by_teacher=None,
        relation_weight=0.0,
        semantic_weight=0.0,
        semantic_temperature=1.0,
        teacher_sample_weights={"a": alpha},
    )
    loss.backward()

    assert student.grad is not None
    assert teacher.grad is None
    assert alpha.grad is None


def test_common_alpha_scaling_is_normalization_invariant() -> None:
    kwargs = {
        "student_by_teacher": {
            "a": torch.tensor(
                [[1.0, 0.0], [1.0, 0.0]],
                requires_grad=True,
            )
        },
        "teacher_by_name": {
            "a": torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        },
        "prototypes_by_teacher": None,
        "relation_weight": 1.0,
        "semantic_weight": 0.0,
        "semantic_temperature": 1.0,
    }
    full, full_parts = multi_teacher_distillation_loss(
        **kwargs,
        teacher_sample_weights={"a": torch.ones(2)},
    )
    scaled, scaled_parts = multi_teacher_distillation_loss(
        **kwargs,
        teacher_sample_weights={"a": torch.full((2,), 0.2)},
    )
    torch.testing.assert_close(full, scaled)
    torch.testing.assert_close(
        full_parts["relation"],
        scaled_parts["relation"],
    )


def test_batch_size_one_relation_is_finite_zero() -> None:
    total, parts = multi_teacher_distillation_loss(
        student_by_teacher={
            "a": torch.tensor([[1.0, 0.0]], requires_grad=True)
        },
        teacher_by_name={"a": torch.tensor([[0.0, 1.0]])},
        prototypes_by_teacher=None,
        relation_weight=1.0,
        semantic_weight=0.0,
        semantic_temperature=1.0,
    )
    assert torch.isfinite(total)
    assert parts["relation"].item() == 0.0


def test_distillation_is_finite_under_cpu_autocast() -> None:
    student_a = torch.randn(3, CLASSIFICATION_COUNT, requires_grad=True)
    student_b = torch.randn(3, CLASSIFICATION_COUNT, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        total, parts = multi_teacher_distillation_loss(
            student_by_teacher={"a": student_a, "b": student_b},
            teacher_by_name={
                "a": torch.randn(3, CLASSIFICATION_COUNT),
                "b": torch.randn(3, CLASSIFICATION_COUNT),
            },
            prototypes_by_teacher={"a": _registry(), "b": _registry()},
            relation_weight=0.05,
            semantic_weight=0.02,
            semantic_temperature=0.5,
            teacher_weights={"a": 1.0, "b": 0.5},
            teacher_sample_weights={
                "a": torch.tensor([1.0, 0.5, 0.25]),
                "b": torch.tensor([0.25, 0.5, 1.0]),
            },
        )
    total.backward()

    assert torch.isfinite(total)
    assert all(torch.isfinite(value) for value in parts.values())
    assert torch.isfinite(student_a.grad).all()
    assert torch.isfinite(student_b.grad).all()


def test_response_distillation_rejects_invalid_sample_mass() -> None:
    logits = torch.tensor([[1.0, -1.0]])
    target = torch.tensor([[1.0, 0.0]])
    with pytest.raises(ValueError, match="finite and non-negative"):
        prototype_response_distillation_loss(
            logits,
            target,
            temperature=0.1,
            sample_weight=torch.tensor([float("nan")]),
        )
    with pytest.raises(ValueError, match="zero total mass"):
        prototype_response_distillation_loss(
            logits,
            target,
            temperature=0.1,
            sample_weight=torch.zeros(1),
        )
