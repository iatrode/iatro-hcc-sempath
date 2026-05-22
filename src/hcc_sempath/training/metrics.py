from __future__ import annotations

import torch
import torch.nn.functional as F

from .losses import relation_distillation_loss, semantic_distillation_loss
from ..modeling.models import normalized_anchor_logits


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


def anchor_response_correlation(student: torch.Tensor, teacher: torch.Tensor, anchors: torch.Tensor) -> float:
    student_logits = normalized_anchor_logits(student, anchors).flatten()
    teacher_logits = normalized_anchor_logits(teacher, anchors).flatten()
    matrix = torch.corrcoef(torch.stack([student_logits, teacher_logits]))
    return float(matrix[0, 1].cpu())


def evaluate_embeddings(student: torch.Tensor, teacher: torch.Tensor, anchors: torch.Tensor, topk: int) -> dict[str, float]:
    return {
        "feature_cosine": feature_cosine(student, teacher),
        "relation_mse": float(relation_distillation_loss(student, teacher).cpu()),
        "semantic_kl": float(semantic_distillation_loss(student, teacher, anchors).cpu()),
        "retrieval_overlap": retrieval_overlap(student, teacher, topk=topk),
        "anchor_response_corr": anchor_response_correlation(student, teacher, anchors),
    }
