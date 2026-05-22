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
    anchors: torch.Tensor,
    relation_weight: float,
    semantic_weight: float,
    semantic_temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    feature = feature_distillation_loss(student, teacher)
    relation = relation_distillation_loss(student, teacher)
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
