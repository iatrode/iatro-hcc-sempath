from __future__ import annotations

import torch
import torch.nn.functional as F

from ..modeling.models import normalized_prototype_logits
from ..modeling.prototypes import PrototypeRegistry


def _prototype_names(registry: PrototypeRegistry, indices: list[int]) -> list[str]:
    return [registry.names[idx] for idx in indices]


def _positions_for_names(
    *,
    registry: PrototypeRegistry,
    source_indices: list[int],
    target_names: list[str],
    label: str,
    device: torch.device,
) -> torch.Tensor:
    source_names = _prototype_names(registry, source_indices)
    positions = {name: idx for idx, name in enumerate(source_names)}
    missing = [name for name in target_names if name not in positions]
    if missing:
        raise ValueError(f"{label} prototype package is missing required prototype names: {missing}")
    return torch.tensor([positions[name] for name in target_names], dtype=torch.long, device=device)


def _teacher_response(
    features: torch.Tensor,
    registry: PrototypeRegistry,
    *,
    primary_names: list[str],
    attribute_names: list[str],
    label: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    primary_logits = normalized_prototype_logits(features, registry.primary_prototypes)
    primary_positions = _positions_for_names(
        registry=registry,
        source_indices=registry.primary_indices,
        target_names=primary_names,
        label=label,
        device=features.device,
    )
    primary = F.softmax(primary_logits, dim=-1).index_select(dim=1, index=primary_positions)
    if not attribute_names:
        return primary, features.new_zeros((features.shape[0], 0))
    attribute_logits = normalized_prototype_logits(features, registry.attribute_prototypes)
    attribute_positions = _positions_for_names(
        registry=registry,
        source_indices=registry.attribute_indices,
        target_names=attribute_names,
        label=label,
        device=features.device,
    )
    attributes = torch.sigmoid(attribute_logits).index_select(dim=1, index=attribute_positions)
    return primary, attributes


def _agreement(
    left_primary: torch.Tensor,
    left_attributes: torch.Tensor,
    right_primary: torch.Tensor,
    right_attributes: torch.Tensor,
) -> torch.Tensor:
    primary_agreement = (left_primary * right_primary).sum(dim=-1).clamp(0.0, 1.0)
    if left_attributes.shape[1] == 0:
        return primary_agreement
    attribute_agreement = 1.0 - (left_attributes - right_attributes).abs().mean(dim=-1)
    return (0.5 * primary_agreement + 0.5 * attribute_agreement.clamp(0.0, 1.0)).clamp(0.0, 1.0)


def _anchor_agreement(
    *,
    primary_response: torch.Tensor,
    attribute_response: torch.Tensor,
    fallback: torch.Tensor,
    prototype_mask: torch.Tensor,
    prototype_level1: torch.Tensor,
    prototype_level2: torch.Tensor,
) -> torch.Tensor:
    anchor = fallback.clone()
    if not bool(prototype_mask.any()):
        return anchor
    out_of_range = (prototype_level1[prototype_mask] < 0) | (
        prototype_level1[prototype_mask] >= primary_response.shape[1]
    )
    if bool(out_of_range.any()):
        raise ValueError("prototype_level1 contains labels outside the level-1 prototype range")
    masked_primary = primary_response[prototype_mask]
    masked_level1 = prototype_level1[prototype_mask].long()
    primary_agreement = masked_primary.gather(1, masked_level1[:, None]).squeeze(1)
    if attribute_response.shape[1] == 0:
        anchor[prototype_mask] = primary_agreement.clamp(0.0, 1.0)
        return anchor
    if prototype_level2.shape[1] != attribute_response.shape[1]:
        raise ValueError(
            f"prototype_level2 width does not match level-2 prototypes: "
            f"labels={prototype_level2.shape[1]} prototypes={attribute_response.shape[1]}"
        )
    level2_targets = prototype_level2[prototype_mask].to(attribute_response.dtype)
    attribute_agreement = 1.0 - (attribute_response[prototype_mask] - level2_targets).abs().mean(dim=-1)
    anchor[prototype_mask] = (0.5 * primary_agreement + 0.5 * attribute_agreement.clamp(0.0, 1.0)).clamp(0.0, 1.0)
    return anchor


def _diagnostics(
    name: str,
    alpha: torch.Tensor,
    consensus: torch.Tensor,
    anchor: torch.Tensor,
    zhcc: torch.Tensor,
) -> dict[str, torch.Tensor]:
    alpha_float = alpha.float()
    return {
        f"alpha_mean/{name}": alpha_float.mean().detach(),
        f"alpha_p25/{name}": torch.quantile(alpha_float, 0.25).detach(),
        f"alpha_p50/{name}": torch.quantile(alpha_float, 0.50).detach(),
        f"alpha_p75/{name}": torch.quantile(alpha_float, 0.75).detach(),
        f"consensus_mean/{name}": consensus.float().mean().detach(),
        f"anchor_agreement_mean/{name}": anchor.float().mean().detach(),
        f"zhcc_agreement_mean/{name}": zhcc.float().mean().detach(),
        f"rejected_fraction_alpha_lt_0.5/{name}": (alpha_float < 0.5).float().mean().detach(),
    }


@torch.no_grad()
def prototype_adjudicated_teacher_weights(
    *,
    teacher_by_name: dict[str, torch.Tensor],
    prototypes_by_teacher: dict[str, PrototypeRegistry],
    zhcc_embedding_norm: torch.Tensor,
    zhcc_prototypes: PrototypeRegistry,
    prototype_mask: torch.Tensor,
    prototype_level1: torch.Tensor,
    prototype_level2: torch.Tensor,
    alpha_min: float = 0.25,
    consensus_weight: float = 0.4,
    anchor_weight: float = 0.4,
    zhcc_response_weight: float = 0.2,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if set(teacher_by_name) != set(prototypes_by_teacher):
        raise ValueError(
            f"teacher/prototype names differ: teacher={sorted(teacher_by_name)} "
            f"prototypes={sorted(prototypes_by_teacher)}"
        )
    if not teacher_by_name:
        raise ValueError("at least one teacher is required for prototype adjudication")
    if not 0.0 <= float(alpha_min) <= 1.0:
        raise ValueError(f"alpha_min must be in [0, 1], got {alpha_min}")

    primary_names = _prototype_names(zhcc_prototypes, zhcc_prototypes.primary_indices)
    attribute_names = _prototype_names(zhcc_prototypes, zhcc_prototypes.attribute_indices)
    zhcc_primary, zhcc_attributes = _teacher_response(
        zhcc_embedding_norm,
        zhcc_prototypes,
        primary_names=primary_names,
        attribute_names=attribute_names,
        label="zhcc",
    )

    responses = {
        name: _teacher_response(
            features,
            prototypes_by_teacher[name],
            primary_names=primary_names,
            attribute_names=attribute_names,
            label=name,
        )
        for name, features in teacher_by_name.items()
    }

    names = sorted(responses)
    primary_stack = torch.stack([responses[name][0] for name in names], dim=0)
    attribute_stack = torch.stack([responses[name][1] for name in names], dim=0)
    alpha_by_teacher: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, torch.Tensor] = {}
    alpha_min_value = float(alpha_min)

    for idx, name in enumerate(names):
        primary, attributes = responses[name]
        if len(names) == 1:
            consensus = primary.new_ones(primary.shape[0])
        else:
            mean_other_primary = (primary_stack.sum(dim=0) - primary_stack[idx]) / float(len(names) - 1)
            mean_other_attributes = (attribute_stack.sum(dim=0) - attribute_stack[idx]) / float(len(names) - 1)
            consensus = _agreement(primary, attributes, mean_other_primary, mean_other_attributes)
        anchor = _anchor_agreement(
            primary_response=primary,
            attribute_response=attributes,
            fallback=consensus,
            prototype_mask=prototype_mask.to(primary.device),
            prototype_level1=prototype_level1.to(primary.device),
            prototype_level2=prototype_level2.to(primary.device),
        )
        zhcc = _agreement(primary, attributes, zhcc_primary, zhcc_attributes)
        reliability = (
            float(consensus_weight) * consensus
            + float(anchor_weight) * anchor
            + float(zhcc_response_weight) * zhcc
        ).clamp(0.0, 1.0)
        alpha = (alpha_min_value + (1.0 - alpha_min_value) * reliability).clamp(alpha_min_value, 1.0)
        alpha_by_teacher[name] = alpha
        diagnostics.update(_diagnostics(name, alpha, consensus, anchor, zhcc))

    return alpha_by_teacher, diagnostics
