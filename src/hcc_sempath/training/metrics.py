from __future__ import annotations

import torch
import torch.nn.functional as F

from .losses import relation_distillation_loss, semantic_distillation_loss
from ..modeling.models import normalized_prototype_logits
from ..modeling.prototypes import PrototypeRegistry


def feature_cosine(student: torch.Tensor, teacher: torch.Tensor) -> float:
    return float(F.cosine_similarity(student, teacher, dim=-1).mean().cpu())


def _deterministic_subset(
    student: torch.Tensor,
    teacher: torch.Tensor,
    max_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_samples = int(max_samples)
    if max_samples <= 0 or student.shape[0] <= max_samples:
        return student, teacher
    idx = torch.linspace(0, student.shape[0] - 1, steps=max_samples, device=student.device).long()
    return student.index_select(0, idx), teacher.index_select(0, idx)


def retrieval_overlap(student: torch.Tensor, teacher: torch.Tensor, topk: int) -> float:
    k = min(topk + 1, student.shape[0])
    if k <= 1:
        return 0.0
    student_sim = F.normalize(student, dim=-1) @ F.normalize(student, dim=-1).T
    teacher_sim = F.normalize(teacher, dim=-1) @ F.normalize(teacher, dim=-1).T
    student_idx = student_sim.topk(k=k, dim=1).indices[:, 1:]
    teacher_idx = teacher_sim.topk(k=k, dim=1).indices[:, 1:]
    overlap = student_idx.unsqueeze(-1).eq(teacher_idx.unsqueeze(-2)).any(dim=-1)
    return float(overlap.float().mean().cpu())


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
    max_pairwise_samples: int = 4096,
) -> dict[str, float]:
    pairwise_student, pairwise_teacher = _deterministic_subset(student, teacher, max_pairwise_samples)
    metrics = {
        "feature_cosine": feature_cosine(student, teacher),
        "relation_mse": float(relation_distillation_loss(pairwise_student, pairwise_teacher).cpu()),
        "retrieval_overlap": retrieval_overlap(pairwise_student, pairwise_teacher, topk=topk),
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
    max_pairwise_samples: int = 4096,
    evaluation_device: torch.device | str | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    device = (
        torch.device(evaluation_device)
        if evaluation_device is not None
        else None
    )
    with torch.inference_mode():
        for name in sorted(student_by_teacher):
            student = student_by_teacher[name]
            teacher = teacher_by_name[name]
            prototypes = (
                prototypes_by_teacher.get(name)
                if prototypes_by_teacher
                else None
            )
            if device is not None:
                student = student.to(device, non_blocking=True)
                teacher = teacher.to(device, non_blocking=True)
                prototypes = (
                    prototypes.to(device)
                    if prototypes is not None
                    else None
                )
            teacher_metrics = evaluate_embeddings(
                student,
                teacher,
                prototypes,
                topk,
                max_pairwise_samples=max_pairwise_samples,
            )
            metrics.update(
                {f"{name}_{key}": value for key, value in teacher_metrics.items()}
            )
            del student, teacher, prototypes
    return metrics
