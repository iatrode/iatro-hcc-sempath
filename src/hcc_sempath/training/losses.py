from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..modeling.models import bounded_logits, clamp_probability, normalized_prototype_logits
from ..modeling.prototypes import PrototypeRegistry


def _validate_nonnegative_finite(
    value: torch.Tensor,
    message: str,
) -> None:
    invalid = (~torch.isfinite(value) | (value < 0)).any()
    assert_async = getattr(torch, "_assert_async", None)
    if invalid.device.type == "cuda" and assert_async is not None:
        assert_async(~invalid, message)
        return
    if bool(invalid):
        raise ValueError(message)


def feature_distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    loss_type: str = "cosine",
) -> torch.Tensor:
    return feature_distillation_loss_per_sample(student, teacher, loss_type=loss_type).mean()


def feature_distillation_loss_per_sample(
    student: torch.Tensor,
    teacher: torch.Tensor,
    loss_type: str = "cosine",
) -> torch.Tensor:
    student = student.float()
    teacher = teacher.detach().float()
    cosine = 1.0 - F.cosine_similarity(student, teacher, dim=-1)
    if loss_type == "cosine":
        return cosine
    student_norm = F.normalize(student, dim=-1)
    teacher_norm = F.normalize(teacher, dim=-1)
    if loss_type == "cosine_plus_norm_mse":
        mse = F.mse_loss(student_norm, teacher_norm, reduction="none").mean(dim=-1)
        return cosine + mse
    if loss_type == "cosine_plus_raw_mse":
        mse = F.mse_loss(student, teacher, reduction="none").mean(dim=-1)
        return cosine + mse
    raise ValueError(f"unsupported feature_loss_type: {loss_type}")


def relation_distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    difference = relation_distillation_loss_matrix(student, teacher)
    if sample_weight is None:
        return difference.mean()
    weight = _validated_sample_weight(
        sample_weight,
        difference.shape[0],
        difference,
        label="relation",
    )
    pair_weight = weight[:, None] * weight[None, :]
    return (difference * pair_weight).sum() / pair_weight.sum().clamp_min(1e-6)


def relation_distillation_loss_matrix(
    student: torch.Tensor,
    teacher: torch.Tensor,
) -> torch.Tensor:
    student = student.float()
    teacher = teacher.detach().float()
    student_norm = F.normalize(student, dim=-1)
    teacher_norm = F.normalize(teacher, dim=-1)
    student_rel = student_norm @ student_norm.transpose(0, 1)
    teacher_rel = teacher_norm @ teacher_norm.transpose(0, 1)
    return (student_rel - teacher_rel).square()


def semantic_distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    prototypes: PrototypeRegistry,
    classification_temperature: float = 1.0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Preserve the fixed classification semantic axis."""

    per_sample = classification_prototype_kl_loss_per_sample(
        student,
        teacher,
        prototypes,
        temperature=classification_temperature,
    )
    return _weighted_sample_mean(
        per_sample,
        sample_weight,
        label="semantic",
    )


def classification_prototype_kl_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    prototypes: PrototypeRegistry,
    temperature: float = 1.0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    per_sample = classification_prototype_kl_loss_per_sample(
        student,
        teacher,
        prototypes,
        temperature=temperature,
    )
    return _weighted_sample_mean(
        per_sample,
        sample_weight,
        label="semantic",
    )


def classification_prototype_kl_loss_per_sample(
    student: torch.Tensor,
    teacher: torch.Tensor,
    prototypes: PrototypeRegistry,
    temperature: float = 1.0,
) -> torch.Tensor:
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError(
            f"semantic temperature must be positive, got {temperature}"
        )
    classification_prototypes = prototypes.prototypes
    student_logits = bounded_logits(
        normalized_prototype_logits(student, classification_prototypes)
        / temperature
    )
    teacher_logits = bounded_logits(
        normalized_prototype_logits(teacher, classification_prototypes)
        / temperature
    )
    teacher_target = clamp_probability(
        F.softmax(teacher_logits, dim=-1),
        normalize=True,
    ).detach()
    return F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        teacher_target,
        reduction="none",
    ).sum(dim=-1) * (temperature**2)


def _validated_sample_weight(
    sample_weight: torch.Tensor,
    expected_size: int,
    reference: torch.Tensor,
    *,
    label: str,
) -> torch.Tensor:
    weight = sample_weight.detach().to(
        device=reference.device,
        dtype=reference.dtype,
    ).view(-1)
    if weight.shape != (expected_size,):
        raise ValueError(
            f"{label} sample weight shape mismatch: "
            f"weight={tuple(weight.shape)} "
            f"expected={(expected_size,)}"
        )
    _validate_nonnegative_finite(
        weight,
        f"{label} sample weights must be finite and non-negative",
    )
    return weight


def _weighted_sample_mean(
    values: torch.Tensor,
    sample_weight: torch.Tensor | None,
    *,
    label: str,
) -> torch.Tensor:
    if sample_weight is None:
        return values.mean()
    weight = _validated_sample_weight(
        sample_weight,
        values.shape[0],
        values,
        label=label,
    )
    denominator = weight.sum()
    invalid = denominator <= 0
    assert_async = getattr(torch, "_assert_async", None)
    if invalid.device.type == "cuda" and assert_async is not None:
        assert_async(
            ~invalid,
            f"{label} sample weights have zero total mass",
        )
    elif bool(invalid):
        raise ValueError(f"{label} sample weights have zero total mass")
    return (values * weight).sum() / denominator


def total_distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    prototypes: PrototypeRegistry | None,
    relation_weight: float,
    semantic_weight: float,
    semantic_temperature: float,
    feature_loss_type: str = "cosine",
    classification_temperature: float | None = None,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    feature_per_sample = feature_distillation_loss_per_sample(
        student,
        teacher,
        loss_type=feature_loss_type,
    )
    feature = _weighted_sample_mean(
        feature_per_sample,
        sample_weight,
        label="feature",
    )
    relation = relation_distillation_loss(
        student,
        teacher,
        sample_weight=sample_weight,
    )
    if prototypes is None or semantic_weight == 0:
        semantic = feature.new_zeros(())
    else:
        semantic = semantic_distillation_loss(
            student=student,
            teacher=teacher,
            prototypes=prototypes,
            classification_temperature=semantic_temperature if classification_temperature is None else classification_temperature,
            sample_weight=sample_weight,
        )
    total = feature + relation_weight * relation + semantic_weight * semantic
    return total, {
        "feature": feature.detach(),
        "relation": relation.detach(),
        "semantic": semantic.detach(),
    }


def multi_teacher_distillation_loss(
    student_by_teacher: dict[str, torch.Tensor],
    teacher_by_name: dict[str, torch.Tensor],
    prototypes_by_teacher: dict[str, PrototypeRegistry] | None,
    relation_weight: float,
    semantic_weight: float,
    semantic_temperature: float,
    teacher_weights: dict[str, float] | None = None,
    feature_loss_type: str = "cosine",
    classification_temperature: float | None = None,
    teacher_sample_weights: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not student_by_teacher:
        raise ValueError("multi-teacher distillation requires at least one teacher")
    if set(student_by_teacher) != set(teacher_by_name):
        raise ValueError(
            f"student/teacher names differ: student={sorted(student_by_teacher)} teacher={sorted(teacher_by_name)}"
        )
    if (
        teacher_sample_weights is not None
        and set(teacher_sample_weights) != set(student_by_teacher)
    ):
        raise ValueError(
            "teacher sample weights must match student heads: "
            f"weights={sorted(teacher_sample_weights)} "
            f"student={sorted(student_by_teacher)}"
        )
    unknown_teacher_weights = set(teacher_weights or {}).difference(
        student_by_teacher
    )
    if unknown_teacher_weights:
        raise ValueError(
            "teacher weights contain unknown teachers: "
            f"{sorted(unknown_teacher_weights)}"
        )
    if semantic_weight != 0 and (
        prototypes_by_teacher is None
        or set(prototypes_by_teacher) != set(student_by_teacher)
    ):
        raise ValueError(
            "semantic distillation requires one prototype registry per "
            f"teacher: prototypes={sorted(prototypes_by_teacher or {})} "
            f"student={sorted(student_by_teacher)}"
        )
    zero = next(iter(student_by_teacher.values())).new_zeros(
        (),
        dtype=torch.float32,
    )
    feature_numerator = zero
    feature_denominator = zero
    relation_numerator = zero
    relation_denominator = zero
    semantic_numerator = zero
    semantic_denominator = zero
    teacher_alignment: dict[str, torch.Tensor] = {}
    positive_teacher_count = 0
    for name in sorted(student_by_teacher):
        teacher_weight = float((teacher_weights or {}).get(name, 1.0))
        if not math.isfinite(teacher_weight) or teacher_weight < 0:
            raise ValueError(
                f"teacher weight must be finite and non-negative: "
                f"teacher={name} weight={teacher_weight}"
            )
        if teacher_weight == 0:
            continue
        positive_teacher_count += 1
        student = student_by_teacher[name]
        teacher = teacher_by_name[name].detach()
        sample_weight = (
            teacher_sample_weights.get(name)
            if teacher_sample_weights is not None
            else torch.ones(
                student.shape[0],
                device=student.device,
                dtype=torch.float32,
            )
        )
        feature_values = feature_distillation_loss_per_sample(
            student,
            teacher,
            loss_type=feature_loss_type,
        )
        cosine_distance = (
            feature_values
            if feature_loss_type == "cosine"
            else feature_distillation_loss_per_sample(
                student,
                teacher,
                loss_type="cosine",
            )
        )
        teacher_alignment[f"{name}_feature_cosine"] = (
            1.0 - cosine_distance.mean()
        ).detach()
        alpha = _validated_sample_weight(
            sample_weight,
            student.shape[0],
            feature_values,
            label=f"teacher={name}",
        )
        weighted_alpha = alpha * teacher_weight

        feature_numerator = feature_numerator + (
            feature_values * weighted_alpha
        ).sum()
        feature_denominator = (
            feature_denominator + weighted_alpha.sum()
        )

        relation_values = relation_distillation_loss_matrix(
            student,
            teacher,
        )
        pair_weight = (
            alpha[:, None] * alpha[None, :] * teacher_weight
        )
        relation_numerator = relation_numerator + (
            relation_values * pair_weight
        ).sum()
        relation_denominator = (
            relation_denominator + pair_weight.sum()
        )

        prototypes = prototypes_by_teacher.get(name) if prototypes_by_teacher else None
        if prototypes is not None and semantic_weight != 0:
            semantic_values = classification_prototype_kl_loss_per_sample(
                student,
                teacher,
                prototypes,
                temperature=(
                    semantic_temperature
                    if classification_temperature is None
                    else classification_temperature
                ),
            )
            semantic_numerator = semantic_numerator + (
                semantic_values * weighted_alpha
            ).sum()
            semantic_denominator = (
                semantic_denominator + weighted_alpha.sum()
            )

    if positive_teacher_count == 0:
        raise ValueError("at least one teacher must have a positive loss weight")
    if float(feature_denominator.detach()) <= 0:
        raise ValueError(
            "teacher sample weights have zero total mass across the batch"
        )
    if float(relation_denominator.detach()) <= 0:
        raise ValueError(
            "teacher pair weights have zero total mass across the batch"
        )
    feature = feature_numerator / feature_denominator
    relation = relation_numerator / relation_denominator
    semantic = (
        semantic_numerator / semantic_denominator
        if semantic_weight != 0
        else zero
    )
    total = (
        feature
        + float(relation_weight) * relation
        + float(semantic_weight) * semantic
    )
    return total, {
        "feature": feature.detach(),
        "relation": relation.detach(),
        "semantic": semantic.detach(),
        **teacher_alignment,
    }
