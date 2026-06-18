from __future__ import annotations

import torch
import torch.nn.functional as F

from ..modeling.models import PROB_EPS, bounded_logits, clamp_probability, normalized_prototype_logits
from ..modeling.prototypes import PrototypeRegistry


def _prototype_names(registry: PrototypeRegistry, indices: list[int]) -> list[str]:
    return [registry.names[idx] for idx in indices]


_POSITIONS_CACHE: dict[tuple, torch.Tensor] = {}


def _positions_for_names(
    *,
    registry: PrototypeRegistry,
    source_indices: list[int],
    target_names: list[str],
    label: str,
    device: torch.device,
) -> torch.Tensor:
    key = (id(registry), tuple(source_indices), tuple(target_names), device)
    if key in _POSITIONS_CACHE:
        return _POSITIONS_CACHE[key]
    source_names = _prototype_names(registry, source_indices)
    positions = {name: idx for idx, name in enumerate(source_names)}
    missing = [name for name in target_names if name not in positions]
    if missing:
        raise ValueError(f"{label} prototype package is missing required prototype names: {missing}")
    tensor = torch.tensor([positions[name] for name in target_names], dtype=torch.long, device=device)
    _POSITIONS_CACHE[key] = tensor
    return tensor


def prototype_response(
    features: torch.Tensor,
    registry: PrototypeRegistry,
    *,
    primary_names: list[str],
    attribute_names: list[str],
    label: str,
    primary_temperature: float = 0.1,
    attribute_temperature: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if primary_temperature <= 0:
        raise ValueError(f"primary_temperature must be positive, got {primary_temperature}")
    if attribute_temperature <= 0:
        raise ValueError(f"attribute_temperature must be positive, got {attribute_temperature}")
    primary_logits = bounded_logits(
        normalized_prototype_logits(features, registry.primary_prototypes) / float(primary_temperature)
    )
    primary_positions = _positions_for_names(
        registry=registry,
        source_indices=registry.primary_indices,
        target_names=primary_names,
        label=label,
        device=features.device,
    )
    primary = clamp_probability(F.softmax(primary_logits, dim=-1)).index_select(
        dim=1, index=primary_positions
    )
    primary = primary / primary.sum(dim=1, keepdim=True).clamp_min(PROB_EPS)
    if not attribute_names:
        return primary, features.new_zeros((features.shape[0], 0))
    attribute_logits = bounded_logits(
        normalized_prototype_logits(features, registry.attribute_prototypes) / float(attribute_temperature)
    )
    attribute_positions = _positions_for_names(
        registry=registry,
        source_indices=registry.attribute_indices,
        target_names=attribute_names,
        label=label,
        device=features.device,
    )
    attributes = clamp_probability(torch.sigmoid(attribute_logits)).index_select(dim=1, index=attribute_positions)
    return primary, attributes


@torch.no_grad()
def teacher_semantic_response_target(
    *,
    teacher_by_name: dict[str, torch.Tensor],
    prototypes_by_teacher: dict[str, PrototypeRegistry],
    target_registry: PrototypeRegistry,
    teacher_weights: dict[str, float] | None = None,
    teacher_sample_weights: dict[str, torch.Tensor] | None = None,
    primary_temperature: float = 0.1,
    attribute_temperature: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if set(teacher_by_name) != set(prototypes_by_teacher):
        raise ValueError(
            f"teacher/prototype names differ: teacher={sorted(teacher_by_name)} "
            f"prototypes={sorted(prototypes_by_teacher)}"
        )
    if teacher_sample_weights is not None and set(teacher_sample_weights) != set(teacher_by_name):
        raise ValueError(
            f"teacher_sample_weights must match teachers: "
            f"weights={sorted(teacher_sample_weights)} teachers={sorted(teacher_by_name)}"
        )
    primary_names = _prototype_names(target_registry, target_registry.primary_indices)
    attribute_names = _prototype_names(target_registry, target_registry.attribute_indices)
    primary_sum = None
    attribute_sum = None
    weight_sum = None
    for name in sorted(teacher_by_name):
        base_weight = float((teacher_weights or {}).get(name, 1.0))
        if base_weight <= 0:
            continue
        primary, attributes = prototype_response(
            teacher_by_name[name],
            prototypes_by_teacher[name],
            primary_names=primary_names,
            attribute_names=attribute_names,
            label=name,
            primary_temperature=primary_temperature,
            attribute_temperature=attribute_temperature,
        )
        sample_weight = teacher_sample_weights.get(name) if teacher_sample_weights is not None else None
        if sample_weight is None:
            weight = primary.new_full((primary.shape[0], 1), base_weight)
        else:
            weight = sample_weight.to(device=primary.device, dtype=primary.dtype).view(-1, 1) * base_weight
        primary_sum = primary * weight if primary_sum is None else primary_sum + primary * weight
        attribute_sum = attributes * weight if attribute_sum is None else attribute_sum + attributes * weight
        weight_sum = weight if weight_sum is None else weight_sum + weight
    if primary_sum is None or attribute_sum is None or weight_sum is None:
        raise ValueError("at least one teacher must have a positive loss weight")
    denom = weight_sum.clamp_min(1e-6)
    primary_target = clamp_probability(primary_sum / denom, normalize=True)
    primary_target = primary_target / primary_target.sum(dim=1, keepdim=True).clamp_min(PROB_EPS)
    attribute_target = clamp_probability(attribute_sum / denom)
    return primary_target, attribute_target


def zhcc_response_distillation_loss(
    *,
    embedding_norm: torch.Tensor,
    prototypes: PrototypeRegistry,
    target_primary: torch.Tensor,
    target_attributes: torch.Tensor,
    primary_temperature: float = 0.1,
    attribute_temperature: float = 0.1,
    level2_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if primary_temperature <= 0:
        raise ValueError(f"primary_temperature must be positive, got {primary_temperature}")
    if attribute_temperature <= 0:
        raise ValueError(f"attribute_temperature must be positive, got {attribute_temperature}")
    primary_logits = bounded_logits(
        normalized_prototype_logits(embedding_norm, prototypes.primary_prototypes) / float(primary_temperature)
    )
    if target_primary.shape != primary_logits.shape:
        raise ValueError(f"target_primary shape mismatch: target={tuple(target_primary.shape)} logits={tuple(primary_logits.shape)}")
    l1 = (
        F.kl_div(
            F.log_softmax(primary_logits, dim=-1),
            clamp_probability(target_primary.to(device=embedding_norm.device), normalize=True),
            reduction="batchmean",
        )
        * (float(primary_temperature) ** 2)
    )
    if prototypes.attribute_indices:
        attribute_logits = bounded_logits(
            normalized_prototype_logits(embedding_norm, prototypes.attribute_prototypes) / float(attribute_temperature)
        )
        if target_attributes.shape != attribute_logits.shape:
            raise ValueError(
                f"target_attributes shape mismatch: target={tuple(target_attributes.shape)} "
                f"logits={tuple(attribute_logits.shape)}"
            )
        l2 = F.binary_cross_entropy_with_logits(
            attribute_logits,
            clamp_probability(target_attributes.to(device=embedding_norm.device)),
        ) * (float(attribute_temperature) ** 2)
    else:
        l2 = embedding_norm.new_zeros(())
    total = l1 + float(level2_weight) * l2
    return total, {
        "zhcc_proto": total.detach(),
        "zhcc_response": total.detach(),
        "zhcc_l1": l1.detach(),
        "zhcc_l2": l2.detach(),
    }


def zhcc_prototype_loss(
    embedding_norm: torch.Tensor,
    prototype_mask: torch.Tensor,
    prototype_level1: torch.Tensor,
    prototype_level2: torch.Tensor,
    prototypes: PrototypeRegistry,
    *,
    level2_weight: float = 0.5,
    primary_temperature: float = 0.1,
    attribute_temperature: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if primary_temperature <= 0:
        raise ValueError(f"primary_temperature must be positive, got {primary_temperature}")
    if attribute_temperature <= 0:
        raise ValueError(f"attribute_temperature must be positive, got {attribute_temperature}")
    if not bool(prototype_mask.any()):
        zero = embedding_norm.new_zeros(())
        return zero, {"zhcc_proto": zero.detach(), "zhcc_l1": zero.detach(), "zhcc_l2": zero.detach()}

    supervised_embedding = embedding_norm[prototype_mask].float()
    l1_targets = prototype_level1[prototype_mask].long()
    l2_targets = prototype_level2[prototype_mask].float()
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
    primary_logits = bounded_logits(
        normalized_prototype_logits(supervised_embedding, prototypes.primary_prototypes) / float(primary_temperature)
    )
    l1 = F.cross_entropy(primary_logits, l1_targets)
    if num_attributes > 0:
        attribute_logits = bounded_logits(
            normalized_prototype_logits(supervised_embedding, prototypes.attribute_prototypes) / float(attribute_temperature)
        )
        l2 = F.binary_cross_entropy_with_logits(attribute_logits, l2_targets)
    else:
        l2 = embedding_norm.new_zeros(())
    total = l1 + float(level2_weight) * l2
    return total, {
        "zhcc_proto": total.detach(),
        "zhcc_l1": l1.detach(),
        "zhcc_l2": l2.detach(),
    }
