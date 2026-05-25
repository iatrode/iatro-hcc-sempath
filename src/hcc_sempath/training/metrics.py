from __future__ import annotations

import torch
import torch.nn.functional as F

from .losses import relation_distillation_loss, semantic_distillation_loss
from ..modeling.models import normalized_prototype_logits
from ..modeling.prototypes import PrototypeRegistry


def feature_cosine(student: torch.Tensor, teacher: torch.Tensor) -> float:
    return float(F.cosine_similarity(student, teacher, dim=-1).mean().cpu())


def retrieval_overlap(student: torch.Tensor, teacher: torch.Tensor, topk: int) -> float:
    k = min(topk + 1, student.shape[0])
    student_sim = F.normalize(student, dim=-1) @ F.normalize(student, dim=-1).T
    teacher_sim = F.normalize(teacher, dim=-1) @ F.normalize(teacher, dim=-1).T
    student_idx = student_sim.topk(k=k, dim=1).indices[:, 1:]
    teacher_idx = teacher_sim.topk(k=k, dim=1).indices[:, 1:]
    overlap = []
    for s_row, t_row in zip(student_idx, teacher_idx):
        overlap.append(len(set(s_row.tolist()) & set(t_row.tolist())) / max(1, k - 1))
    return float(sum(overlap) / max(1, len(overlap)))


def prototype_response_correlation(student: torch.Tensor, teacher: torch.Tensor, prototypes: PrototypeRegistry) -> float:
    student_logits = normalized_prototype_logits(student, prototypes.prototypes).flatten()
    teacher_logits = normalized_prototype_logits(teacher, prototypes.prototypes).flatten()
    matrix = torch.corrcoef(torch.stack([student_logits, teacher_logits]))
    return float(matrix[0, 1].cpu())


def evaluate_embeddings(
    student: torch.Tensor,
    teacher: torch.Tensor,
    prototypes: PrototypeRegistry | None,
    topk: int,
) -> dict[str, float]:
    metrics = {
        "feature_cosine": feature_cosine(student, teacher),
        "relation_mse": float(relation_distillation_loss(student, teacher).cpu()),
        "retrieval_overlap": retrieval_overlap(student, teacher, topk=topk),
    }
    if prototypes is not None:
        metrics["prototype_semantic_loss"] = float(semantic_distillation_loss(student, teacher, prototypes).cpu())
        metrics["prototype_response_corr"] = prototype_response_correlation(student, teacher, prototypes)
    return metrics


def evaluate_teacher_outputs(
    student_by_teacher: dict[str, torch.Tensor],
    teacher_by_name: dict[str, torch.Tensor],
    prototypes_by_teacher: dict[str, PrototypeRegistry] | None,
    topk: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name in sorted(student_by_teacher):
        teacher_metrics = evaluate_embeddings(
            student_by_teacher[name],
            teacher_by_name[name],
            prototypes_by_teacher.get(name) if prototypes_by_teacher else None,
            topk,
        )
        metrics.update({f"{name}_{key}": value for key, value in teacher_metrics.items()})
    return metrics
