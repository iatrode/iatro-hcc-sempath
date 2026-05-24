from __future__ import annotations

import torch
import torch.nn.functional as F

from ..modeling.models import normalized_anchor_logits


def feature_distillation_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    cosine = 1.0 - F.cosine_similarity(student, teacher, dim=-1).mean()
    student_norm = F.normalize(student, dim=-1)
    teacher_norm = F.normalize(teacher, dim=-1)
    mse = F.mse_loss(student_norm, teacher_norm)
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
    anchors: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    student_logits = normalized_anchor_logits(student, anchors) / temperature
    teacher_logits = normalized_anchor_logits(teacher, anchors) / temperature
    return F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)


def total_distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    anchors: torch.Tensor | None,
    relation_weight: float,
    semantic_weight: float,
    semantic_temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    feature = feature_distillation_loss(student, teacher)
    relation = relation_distillation_loss(student, teacher)
    if anchors is None or semantic_weight == 0:
        semantic = feature.new_zeros(())
    else:
        semantic = semantic_distillation_loss(
            student=student,
            teacher=teacher,
            anchors=anchors,
            temperature=semantic_temperature,
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
    anchors_by_teacher: dict[str, torch.Tensor] | None,
    relation_weight: float,
    semantic_weight: float,
    semantic_temperature: float,
    teacher_weights: dict[str, float] | None = None,
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
    }
    weight_sum = 0.0
    for name in sorted(student_by_teacher):
        weight = float((teacher_weights or {}).get(name, 1.0))
        if weight <= 0:
            continue
        anchors = anchors_by_teacher.get(name) if anchors_by_teacher else None
        loss, parts = total_distillation_loss(
            student=student_by_teacher[name],
            teacher=teacher_by_name[name],
            anchors=anchors,
            relation_weight=relation_weight,
            semantic_weight=semantic_weight,
            semantic_temperature=semantic_temperature,
        )
        total = weight * loss if total is None else total + weight * loss
        for key in totals:
            totals[key] = totals[key] + weight * parts[key]
        weight_sum += weight
    if total is None or weight_sum == 0:
        raise ValueError("at least one teacher must have a positive loss weight")
    total = total / weight_sum
    return total, {key: value / weight_sum for key, value in totals.items()}
