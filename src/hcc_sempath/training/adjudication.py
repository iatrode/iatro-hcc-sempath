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
    *,
    l1_weight: float = 0.5,
    l2_weight: float = 0.5,
) -> torch.Tensor:
    primary_agreement = (left_primary * right_primary).sum(dim=-1).clamp(0.0, 1.0)
    if left_attributes.shape[1] == 0:
        return primary_agreement
    attribute_agreement = 1.0 - (left_attributes - right_attributes).abs().mean(dim=-1)
    denom = max(float(l1_weight) + float(l2_weight), 1e-6)
    return (
        (float(l1_weight) * primary_agreement + float(l2_weight) * attribute_agreement.clamp(0.0, 1.0)) / denom
    ).clamp(0.0, 1.0)


def _prototype_label_agreement(
    *,
    primary_response: torch.Tensor,
    attribute_response: torch.Tensor,
    prototype_mask: torch.Tensor,
    prototype_level1: torch.Tensor,
    prototype_level2: torch.Tensor,
    l1_weight: float = 0.5,
    l2_weight: float = 0.5,
) -> torch.Tensor:
    mask = prototype_mask.to(device=primary_response.device, dtype=torch.bool)
    prototype_agreement = primary_response.new_zeros(primary_response.shape[0])
    if not bool(prototype_mask.any()):
        return prototype_agreement
    out_of_range = (prototype_level1[mask] < 0) | (
        prototype_level1[mask] >= primary_response.shape[1]
    )
    if bool(out_of_range.any()):
        raise ValueError("prototype_level1 contains labels outside the level-1 prototype range")
    masked_primary = primary_response[mask]
    masked_level1 = prototype_level1[mask].long()
    primary_agreement = masked_primary.gather(1, masked_level1[:, None]).squeeze(1)
    if attribute_response.shape[1] == 0:
        prototype_agreement[mask] = primary_agreement.clamp(0.0, 1.0)
        return prototype_agreement
    if prototype_level2.shape[1] != attribute_response.shape[1]:
        raise ValueError(
            f"prototype_level2 width does not match level-2 prototypes: "
            f"labels={prototype_level2.shape[1]} prototypes={attribute_response.shape[1]}"
        )
    level2_targets = prototype_level2[mask].to(attribute_response.dtype)
    attribute_agreement = 1.0 - (attribute_response[mask] - level2_targets).abs().mean(dim=-1)
    denom = max(float(l1_weight) + float(l2_weight), 1e-6)
    prototype_agreement[mask] = (
        (float(l1_weight) * primary_agreement + float(l2_weight) * attribute_agreement.clamp(0.0, 1.0)) / denom
    ).clamp(0.0, 1.0)
    return prototype_agreement


def _combine_reliability(
    *,
    consensus: torch.Tensor,
    prototype_label: torch.Tensor,
    zhcc: torch.Tensor,
    prototype_mask: torch.Tensor,
    consensus_weight: float,
    prototype_label_weight: float,
    zhcc_response_weight: float,
) -> torch.Tensor:
    w_c = float(consensus_weight)
    w_p = float(prototype_label_weight)
    w_z = float(zhcc_response_weight)
    mask = prototype_mask.to(device=consensus.device, dtype=torch.bool)
    reliability = consensus.new_empty(consensus.shape)

    prototype_denom = max(w_c + w_p + w_z, 1e-6)
    reliability_with_prototype = (w_c * consensus + w_p * prototype_label + w_z * zhcc) / prototype_denom

    non_prototype_denom = w_c + w_z
    if non_prototype_denom <= 0:
        reliability_without_prototype = consensus
    else:
        reliability_without_prototype = (w_c * consensus + w_z * zhcc) / non_prototype_denom

    reliability[mask] = reliability_with_prototype[mask]
    reliability[~mask] = reliability_without_prototype[~mask]
    return reliability.clamp(0.0, 1.0)


def _diagnostics(
    name: str,
    alpha_raw: torch.Tensor,
    alpha_effective: torch.Tensor,
    consensus: torch.Tensor,
    prototype_label: torch.Tensor,
    zhcc: torch.Tensor,
) -> dict[str, torch.Tensor]:
    raw = alpha_raw.float()
    effective = alpha_effective.float()
    return {
        f"alpha_mean/{name}": effective.mean().detach(),
        f"alpha_p25/{name}": torch.quantile(effective, 0.25).detach(),
        f"alpha_p50/{name}": torch.quantile(effective, 0.50).detach(),
        f"alpha_p75/{name}": torch.quantile(effective, 0.75).detach(),
        f"alpha_raw_mean/{name}": raw.mean().detach(),
        f"alpha_raw_p25/{name}": torch.quantile(raw, 0.25).detach(),
        f"alpha_raw_p50/{name}": torch.quantile(raw, 0.50).detach(),
        f"alpha_raw_p75/{name}": torch.quantile(raw, 0.75).detach(),
        f"alpha_effective_mean/{name}": effective.mean().detach(),
        f"alpha_effective_p25/{name}": torch.quantile(effective, 0.25).detach(),
        f"alpha_effective_p50/{name}": torch.quantile(effective, 0.50).detach(),
        f"alpha_effective_p75/{name}": torch.quantile(effective, 0.75).detach(),
        f"consensus_mean/{name}": consensus.float().mean().detach(),
        f"prototype_label_agreement_mean/{name}": prototype_label.float().mean().detach(),
        f"zhcc_agreement_mean/{name}": zhcc.float().mean().detach(),
        f"rejected_fraction_alpha_lt_0.5/{name}": (effective < 0.5).float().mean().detach(),
    }


@torch.no_grad()
def prototype_adjudicated_teacher_weights(
    *,
    teacher_by_name: dict[str, torch.Tensor],
    prototypes_by_teacher: dict[str, PrototypeRegistry],
    zhcc_embedding_norm: torch.Tensor | None = None,
    zhcc_prototypes: PrototypeRegistry | None = None,
    prototype_mask: torch.Tensor,
    prototype_level1: torch.Tensor,
    prototype_level2: torch.Tensor,
    alpha_min: float = 0.25,
    consensus_weight: float = 0.4,
    prototype_label_weight: float = 0.4,
    l1_agreement_weight: float = 0.5,
    l2_agreement_weight: float = 0.5,
    zhcc_response_weight: float = 0.2,
    filter_strength: float = 1.0,
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
    filter_strength_value = float(filter_strength)
    if not 0.0 <= filter_strength_value <= 1.0:
        raise ValueError(f"filter_strength must be in [0, 1], got {filter_strength}")

    label_registry = zhcc_prototypes if zhcc_prototypes is not None else next(iter(prototypes_by_teacher.values()))
    primary_names = _prototype_names(label_registry, label_registry.primary_indices)
    attribute_names = _prototype_names(label_registry, label_registry.attribute_indices)
    if float(zhcc_response_weight) > 0:
        if zhcc_embedding_norm is None or zhcc_prototypes is None:
            raise ValueError("zhcc_response_weight > 0 requires zhcc_embedding_norm and zhcc_prototypes")
        zhcc_primary, zhcc_attributes = _teacher_response(
            zhcc_embedding_norm,
            zhcc_prototypes,
            primary_names=primary_names,
            attribute_names=attribute_names,
            label="zhcc",
        )
    else:
        first_teacher = next(iter(teacher_by_name.values()))
        zhcc_primary = first_teacher.new_zeros((first_teacher.shape[0], len(primary_names)))
        zhcc_attributes = first_teacher.new_zeros((first_teacher.shape[0], len(attribute_names)))

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
            consensus = _agreement(
                primary,
                attributes,
                mean_other_primary,
                mean_other_attributes,
                l1_weight=l1_agreement_weight,
                l2_weight=l2_agreement_weight,
            )
        prototype_label = _prototype_label_agreement(
            primary_response=primary,
            attribute_response=attributes,
            prototype_mask=prototype_mask.to(primary.device),
            prototype_level1=prototype_level1.to(primary.device),
            prototype_level2=prototype_level2.to(primary.device),
            l1_weight=l1_agreement_weight,
            l2_weight=l2_agreement_weight,
        )
        if float(zhcc_response_weight) > 0:
            zhcc = _agreement(
                primary,
                attributes,
                zhcc_primary,
                zhcc_attributes,
                l1_weight=l1_agreement_weight,
                l2_weight=l2_agreement_weight,
            )
        else:
            zhcc = consensus.new_zeros(consensus.shape)
        reliability = _combine_reliability(
            consensus=consensus,
            prototype_label=prototype_label,
            zhcc=zhcc,
            prototype_mask=prototype_mask.to(primary.device),
            consensus_weight=consensus_weight,
            prototype_label_weight=prototype_label_weight,
            zhcc_response_weight=zhcc_response_weight,
        )
        alpha_raw = (alpha_min_value + (1.0 - alpha_min_value) * reliability).clamp(alpha_min_value, 1.0)
        alpha_effective = 1.0 - filter_strength_value * (1.0 - alpha_raw)
        alpha_effective = alpha_effective.clamp(alpha_min_value, 1.0)
        alpha_by_teacher[name] = alpha_effective
        diagnostics.update(_diagnostics(name, alpha_raw, alpha_effective, consensus, prototype_label, zhcc))

    return alpha_by_teacher, diagnostics
