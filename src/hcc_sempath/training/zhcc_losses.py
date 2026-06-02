from __future__ import annotations

import torch
import torch.nn.functional as F

from ..modeling.models import normalized_prototype_logits
from ..modeling.prototypes import PrototypeRegistry


def zhcc_prototype_loss(
    embedding_norm: torch.Tensor,
    prototype_mask: torch.Tensor,
    prototype_level1: torch.Tensor,
    prototype_level2: torch.Tensor,
    prototypes: PrototypeRegistry,
    *,
    level2_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not bool(prototype_mask.any()):
        zero = embedding_norm.new_zeros(())
        return zero, {"zhcc_proto": zero.detach(), "zhcc_l1": zero.detach(), "zhcc_l2": zero.detach()}

    supervised_embedding = embedding_norm[prototype_mask]
    l1_targets = prototype_level1[prototype_mask].long()
    l2_targets = prototype_level2[prototype_mask].to(supervised_embedding.dtype)
    num_primary = len(prototypes.primary_indices)
    num_attributes = len(prototypes.attribute_indices)
    if l1_targets.numel() > 0:
        l1_min = int(l1_targets.min().item())
        l1_max = int(l1_targets.max().item())
        if l1_min < 0 or l1_max >= num_primary:
            raise ValueError(
                f"prototype_level1 target out of range: min={l1_min} max={l1_max} num_primary={num_primary}"
            )
    if l2_targets.shape[1] != num_attributes:
        raise ValueError(
            f"prototype_level2 width mismatch: labels={l2_targets.shape[1]} attributes={num_attributes}"
        )
    primary_logits = normalized_prototype_logits(supervised_embedding, prototypes.primary_prototypes)
    l1 = F.cross_entropy(primary_logits, l1_targets)
    if num_attributes > 0:
        attribute_logits = normalized_prototype_logits(supervised_embedding, prototypes.attribute_prototypes)
        l2 = F.binary_cross_entropy_with_logits(attribute_logits, l2_targets)
    else:
        l2 = embedding_norm.new_zeros(())
    total = l1 + float(level2_weight) * l2
    return total, {
        "zhcc_proto": total.detach(),
        "zhcc_l1": l1.detach(),
        "zhcc_l2": l2.detach(),
    }
