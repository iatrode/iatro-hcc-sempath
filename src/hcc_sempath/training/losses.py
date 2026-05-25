from __future__ import annotations

import torch
import torch.nn.functional as F

from ..modeling.models import normalized_prototype_logits
from ..modeling.prototypes import PrototypeRegistry


def feature_distillation_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return feature_distillation_loss_per_sample(student, teacher).mean()


def feature_distillation_loss_per_sample(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    cosine = 1.0 - F.cosine_similarity(student, teacher, dim=-1)
    student_norm = F.normalize(student, dim=-1)
    teacher_norm = F.normalize(teacher, dim=-1)
    mse = F.mse_loss(student_norm, teacher_norm, reduction="none").mean(dim=-1)
    return cosine + mse


def relation_distillation_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    student_norm = F.normalize(student, dim=-1)
    teacher_norm = F.normalize(teacher, dim=-1)
    student_rel = student_norm @ student_norm.transpose(0, 1)
    teacher_rel = teacher_norm @ teacher_norm.transpose(0, 1)
    return F.mse_loss(student_rel, teacher_rel)


def semantic_distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    prototypes: PrototypeRegistry,
    temperature: float = 1.0,
) -> torch.Tensor:
    primary = primary_prototype_kl_loss(student, teacher, prototypes, temperature=temperature)
    attributes = attribute_prototype_bce_loss(student, teacher, prototypes, temperature=temperature)
    return primary + attributes


def primary_prototype_kl_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    prototypes: PrototypeRegistry,
    temperature: float = 1.0,
) -> torch.Tensor:
    primary = prototypes.primary_prototypes
    student_logits = normalized_prototype_logits(student, primary) / temperature
    teacher_logits = normalized_prototype_logits(teacher, primary) / temperature
    return (
        F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction="batchmean",
        )
        * (temperature**2)
    )


def attribute_prototype_bce_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    prototypes: PrototypeRegistry,
    temperature: float = 1.0,
) -> torch.Tensor:
    if not prototypes.attribute_indices:
        return student.new_zeros(())
    attributes = prototypes.attribute_prototypes
    student_logits = normalized_prototype_logits(student, attributes) / temperature
    with torch.no_grad():
        teacher_targets = torch.sigmoid(normalized_prototype_logits(teacher, attributes) / temperature)
    return F.binary_cross_entropy_with_logits(student_logits, teacher_targets)


@torch.no_grad()
def prototype_teacher_reliability(
    student: torch.Tensor,
    teacher: torch.Tensor,
    prototypes: PrototypeRegistry,
    alpha_min: float = 0.25,
) -> torch.Tensor:
    primary = prototypes.primary_prototypes
    student_primary = F.softmax(normalized_prototype_logits(student, primary), dim=-1)
    teacher_primary = F.softmax(normalized_prototype_logits(teacher, primary), dim=-1)
    primary_agreement = (student_primary * teacher_primary).sum(dim=-1)
    agreements = [primary_agreement]
    if prototypes.attribute_indices:
        attributes = prototypes.attribute_prototypes
        student_attr = torch.sigmoid(normalized_prototype_logits(student, attributes))
        teacher_attr = torch.sigmoid(normalized_prototype_logits(teacher, attributes))
        attr_agreement = 1.0 - (student_attr - teacher_attr).abs().mean(dim=-1)
        agreements.append(attr_agreement.clamp(0.0, 1.0))
    agreement = torch.stack(agreements, dim=0).mean(dim=0).clamp(0.0, 1.0)
    return (alpha_min + (1.0 - alpha_min) * agreement).clamp(alpha_min, 1.0)


def total_distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    prototypes: PrototypeRegistry | None,
    relation_weight: float,
    semantic_weight: float,
    semantic_temperature: float,
    prototype_filter_weight: float = 0.0,
    prototype_filter_alpha_min: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    reliability = None
    if prototypes is not None and prototype_filter_weight > 0:
        reliability = prototype_teacher_reliability(
            student=student,
            teacher=teacher,
            prototypes=prototypes,
            alpha_min=prototype_filter_alpha_min,
        )
        feature_per_sample = feature_distillation_loss_per_sample(student, teacher)
        filtered_feature = (reliability * feature_per_sample).mean()
        raw_feature = feature_per_sample.mean()
        feature = (1.0 - prototype_filter_weight) * raw_feature + prototype_filter_weight * filtered_feature
    else:
        feature = feature_distillation_loss(student, teacher)
    relation = relation_distillation_loss(student, teacher)
    if prototypes is None or semantic_weight == 0:
        semantic = feature.new_zeros(())
    else:
        semantic = semantic_distillation_loss(
            student=student,
            teacher=teacher,
            prototypes=prototypes,
            temperature=semantic_temperature,
        )
    relation_scale = reliability.mean() if reliability is not None else relation.new_ones(())
    total = feature + relation_weight * relation_scale * relation + semantic_weight * semantic
    return total, {
        "feature": feature.detach(),
        "relation": relation.detach(),
        "semantic": semantic.detach(),
        "reliability": relation_scale.detach(),
    }


def multi_teacher_distillation_loss(
    student_by_teacher: dict[str, torch.Tensor],
    teacher_by_name: dict[str, torch.Tensor],
    prototypes_by_teacher: dict[str, PrototypeRegistry] | None,
    relation_weight: float,
    semantic_weight: float,
    semantic_temperature: float,
    teacher_weights: dict[str, float] | None = None,
    prototype_filter_weight: float = 0.0,
    prototype_filter_alpha_min: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if set(student_by_teacher) != set(teacher_by_name):
        raise ValueError(
            f"student/teacher names differ: student={sorted(student_by_teacher)} teacher={sorted(teacher_by_name)}"
        )
    total = None
    totals = {
        "feature": next(iter(student_by_teacher.values())).new_zeros(()),
        "relation": next(iter(student_by_teacher.values())).new_zeros(()),
        "semantic": next(iter(student_by_teacher.values())).new_zeros(()),
        "reliability": next(iter(student_by_teacher.values())).new_zeros(()),
    }
    weight_sum = 0.0
    for name in sorted(student_by_teacher):
        weight = float((teacher_weights or {}).get(name, 1.0))
        if weight <= 0:
            continue
        prototypes = prototypes_by_teacher.get(name) if prototypes_by_teacher else None
        loss, parts = total_distillation_loss(
            student=student_by_teacher[name],
            teacher=teacher_by_name[name],
            prototypes=prototypes,
            relation_weight=relation_weight,
            semantic_weight=semantic_weight,
            semantic_temperature=semantic_temperature,
            prototype_filter_weight=prototype_filter_weight,
            prototype_filter_alpha_min=prototype_filter_alpha_min,
        )
        total = weight * loss if total is None else total + weight * loss
        for key in totals:
            totals[key] = totals[key] + weight * parts[key]
        weight_sum += weight
    if total is None or weight_sum == 0:
        raise ValueError("at least one teacher must have a positive loss weight")
    total = total / weight_sum
    return total, {key: value / weight_sum for key, value in totals.items()}
