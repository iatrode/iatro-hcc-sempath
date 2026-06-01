from __future__ import annotations

import torch
import torch.nn.functional as F

from ..modeling.models import normalized_prototype_logits
from ..modeling.prototypes import PrototypeRegistry


def _macro_auc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    aucs: list[float] = []
    for idx in range(targets.shape[1]):
        y = targets[:, idx].bool()
        pos = logits[y, idx]
        neg = logits[~y, idx]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        aucs.append(float((pos[:, None] > neg[None, :]).float().mean().cpu()))
    return float(sum(aucs) / len(aucs)) if aucs else 0.0


def _macro_f1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits > 0
    scores: list[float] = []
    for idx in range(targets.shape[1]):
        pred = preds[:, idx]
        target = targets[:, idx].bool()
        tp = int((pred & target).sum())
        fp = int((pred & ~target).sum())
        fn = int((~pred & target).sum())
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(sum(scores) / len(scores)) if scores else 0.0


def _prototype_topk_precision(primary_logits: torch.Tensor, attr_logits: torch.Tensor, level1: torch.Tensor, level2: torch.Tensor) -> float:
    if primary_logits.shape[0] == 0:
        return 0.0
    all_logits = torch.cat([primary_logits, attr_logits], dim=1) if attr_logits.numel() else primary_logits
    scores = []
    attr_offset = primary_logits.shape[1]
    for row_idx in range(all_logits.shape[0]):
        positives = {int(level1[row_idx])}
        positives.update(int(attr_offset + idx) for idx in torch.nonzero(level2[row_idx] > 0, as_tuple=False).flatten())
        k = max(1, min(len(positives), all_logits.shape[1]))
        pred = set(int(idx) for idx in all_logits[row_idx].topk(k=k).indices.tolist())
        scores.append(len(pred & positives) / k)
    return float(sum(scores) / len(scores))


def _neighborhood_purity(embedding: torch.Tensor, level1: torch.Tensor, level2: torch.Tensor, topk: int) -> tuple[float, float]:
    if embedding.shape[0] <= 1:
        return 0.0, 0.0
    k = min(max(1, int(topk)), embedding.shape[0] - 1)
    sim = F.normalize(embedding, dim=-1) @ F.normalize(embedding, dim=-1).T
    indices = sim.topk(k=k + 1, dim=1).indices[:, 1:]
    l1_scores = []
    l2_scores = []
    for row_idx, neighbors in enumerate(indices):
        l1_scores.append(float((level1[neighbors] == level1[row_idx]).float().mean().cpu()))
        target = level2[row_idx].bool()
        if not bool(target.any()):
            l2_scores.append(0.0)
            continue
        overlap = (level2[neighbors].bool() & target).float().sum(dim=1)
        denom = (level2[neighbors].bool() | target).float().sum(dim=1).clamp_min(1.0)
        l2_scores.append(float((overlap / denom).mean().cpu()))
    return float(sum(l1_scores) / len(l1_scores)), float(sum(l2_scores) / len(l2_scores))


def evaluate_zhcc_prototypes(
    embedding_norm: torch.Tensor,
    prototype_mask: torch.Tensor,
    prototype_level1: torch.Tensor,
    prototype_level2: torch.Tensor,
    prototypes: PrototypeRegistry | None,
    *,
    topk: int = 10,
) -> dict[str, float]:
    count = int(prototype_mask.sum().item())
    metrics = {"zhcc_prototype_count": float(count)}
    if prototypes is None or count == 0:
        metrics.update(
            {
                "zhcc_level1_accuracy": 0.0,
                "zhcc_level2_macro_f1": 0.0,
                "zhcc_level2_macro_auc": 0.0,
                "zhcc_prototype_topk_precision": 0.0,
                "zhcc_neighborhood_purity_l1": 0.0,
                "zhcc_neighborhood_purity_l2": 0.0,
            }
        )
        return metrics
    embedding = embedding_norm[prototype_mask]
    level1 = prototype_level1[prototype_mask].long()
    level2 = prototype_level2[prototype_mask].float()
    primary_logits = normalized_prototype_logits(embedding, prototypes.primary_prototypes)
    attr_logits = (
        normalized_prototype_logits(embedding, prototypes.attribute_prototypes)
        if prototypes.attribute_indices
        else embedding.new_zeros((embedding.shape[0], 0))
    )
    l1_pred = primary_logits.argmax(dim=1)
    purity_l1, purity_l2 = _neighborhood_purity(embedding, level1, level2, topk=topk)
    metrics.update(
        {
            "zhcc_level1_accuracy": float((l1_pred == level1).float().mean().cpu()),
            "zhcc_level2_macro_f1": _macro_f1(attr_logits, level2) if attr_logits.numel() else 0.0,
            "zhcc_level2_macro_auc": _macro_auc(attr_logits, level2) if attr_logits.numel() else 0.0,
            "zhcc_prototype_topk_precision": _prototype_topk_precision(primary_logits, attr_logits, level1, level2),
            "zhcc_neighborhood_purity_l1": purity_l1,
            "zhcc_neighborhood_purity_l2": purity_l2,
        }
    )
    return metrics
