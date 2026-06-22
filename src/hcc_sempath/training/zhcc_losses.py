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
    # Do not key only by id(registry): Python may recycle object ids across
    # short-lived PrototypeRegistry instances in the same process, which can
    # silently reuse a stale name-position map when registries use different
    # prototype ordering.
    key = (
        tuple(registry.names),
        tuple(registry.primary_indices),
        tuple(registry.attribute_indices),
        tuple(source_indices),
        tuple(target_names),
        device,
    )
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


def global_attribute_logits(
    embedding_norm: torch.Tensor,
    prototypes: PrototypeRegistry,
    *,
    attribute_temperature: float = 0.1,
) -> torch.Tensor:
    if attribute_temperature <= 0:
        raise ValueError(f"attribute_temperature must be positive, got {attribute_temperature}")
    if not prototypes.attribute_indices:
        return embedding_norm.new_zeros((embedding_norm.shape[0], 0))
    return bounded_logits(
        normalized_prototype_logits(embedding_norm, prototypes.attribute_prototypes)
        / float(attribute_temperature)
    )


def roi_guided_level2_loss(
    *,
    patch_logits: torch.Tensor,
    local_logits: torch.Tensor,
    global_logits: torch.Tensor,
    roi_target: torch.Tensor,
    roi_valid: torch.Tensor,
    roi_consistency: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """ROI-token supervision plus one-way local-to-global transfer.

    Unmarked tokens are ignored. The detached local response may reshape the
    deployable global L2 readout, while global predictions never supervise the
    spatial map.
    """
    if patch_logits.shape != roi_target.shape or patch_logits.shape != roi_valid.shape:
        raise ValueError(
            f"ROI tensor shape mismatch: logits={tuple(patch_logits.shape)} "
            f"target={tuple(roi_target.shape)} valid={tuple(roi_valid.shape)}"
        )
    if local_logits.shape != global_logits.shape or local_logits.shape != roi_consistency.shape:
        raise ValueError(
            f"ROI consistency shape mismatch: local={tuple(local_logits.shape)} "
            f"global={tuple(global_logits.shape)} mask={tuple(roi_consistency.shape)}"
        )
    target = roi_target.to(device=patch_logits.device, dtype=patch_logits.dtype)
    valid = roi_valid.to(device=patch_logits.device, dtype=patch_logits.dtype)
    token_terms = F.binary_cross_entropy_with_logits(patch_logits, target, reduction="none")
    valid_count = valid.sum()
    roi_loss = (token_terms * valid).sum() / valid_count.clamp_min(1.0)
    roi_loss = torch.where(valid_count > 0, roi_loss, patch_logits.sum() * 0.0)

    consistency = roi_consistency.to(device=global_logits.device, dtype=global_logits.dtype)
    local_target = torch.sigmoid(local_logits.detach())
    consistency_terms = F.binary_cross_entropy_with_logits(global_logits, local_target, reduction="none")
    consistency_count = consistency.sum()
    consistency_loss = (consistency_terms * consistency).sum() / consistency_count.clamp_min(1.0)
    consistency_loss = torch.where(
        consistency_count > 0, consistency_loss, global_logits.sum() * 0.0
    )
    probabilities = torch.sigmoid(patch_logits.detach())
    activation = probabilities >= 0.5
    # Degenerate-map rates must reflect ROI-supervised tiles only. A batch is mostly
    # unsupervised tiles (valid==0) whose untrained maps look all-zero/broadcast; if
    # those were counted, the diagnostics would mask the health of the supervised maps
    # that Gate R1 actually certifies.
    supervised_tile = valid.bool().flatten(2).any(dim=2)  # [B, K] per (tile, attribute)
    diagnostics = {
        "roi_valid_tokens": valid_count.detach(),
        "roi_supervised_attributes": consistency_count.detach(),
        "roi_activation_fraction": activation.float().mean(),
    }
    for attribute_idx in range(patch_logits.shape[1]):
        attribute_activation = activation[:, attribute_idx]
        attribute_probability = probabilities[:, attribute_idx]
        supervised = supervised_tile[:, attribute_idx]
        supervised_count = supervised.sum().clamp_min(1)
        per_tile_all_zero = (~attribute_activation).all(dim=(-2, -1))
        per_tile_all_one = attribute_activation.all(dim=(-2, -1))
        per_tile_broadcast = attribute_probability.var(dim=(-2, -1), unbiased=False) < 1e-8
        diagnostics[f"roi_attr_{attribute_idx}_activation_fraction"] = attribute_activation.float().mean()
        diagnostics[f"roi_attr_{attribute_idx}_all_zero_rate"] = (
            (per_tile_all_zero & supervised).sum() / supervised_count
        )
        diagnostics[f"roi_attr_{attribute_idx}_all_one_rate"] = (
            (per_tile_all_one & supervised).sum() / supervised_count
        )
        diagnostics[f"roi_attr_{attribute_idx}_broadcast_rate"] = (
            (per_tile_broadcast & supervised).sum() / supervised_count
        )
    return roi_loss, consistency_loss, diagnostics


def zhcc_response_distillation_loss(
    *,
    embedding_norm: torch.Tensor,
    prototypes: PrototypeRegistry,
    target_primary: torch.Tensor,
    target_attributes: torch.Tensor,
    attribute_gate: torch.Tensor | None = None,
    attribute_negative_weight: torch.Tensor | None = None,
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
        attribute_logits = global_attribute_logits(
            embedding_norm, prototypes, attribute_temperature=attribute_temperature
        )
        if target_attributes.shape != attribute_logits.shape:
            raise ValueError(
                f"target_attributes shape mismatch: target={tuple(target_attributes.shape)} "
                f"logits={tuple(attribute_logits.shape)}"
            )
        target = clamp_probability(target_attributes.to(device=embedding_norm.device))
        if attribute_gate is None and attribute_negative_weight is None:
            l2 = F.binary_cross_entropy_with_logits(attribute_logits, target) * (float(attribute_temperature) ** 2)
        else:
            gate = (
                torch.ones_like(attribute_logits)
                if attribute_gate is None
                else attribute_gate.to(device=embedding_norm.device, dtype=attribute_logits.dtype)
            )
            negative_weight = (
                torch.ones_like(attribute_logits)
                if attribute_negative_weight is None
                else attribute_negative_weight.to(device=embedding_norm.device, dtype=attribute_logits.dtype)
            )
            if gate.shape != attribute_logits.shape:
                raise ValueError(
                    f"attribute_gate shape mismatch: gate={tuple(gate.shape)} "
                    f"logits={tuple(attribute_logits.shape)}"
                )
            if negative_weight.shape != attribute_logits.shape:
                raise ValueError(
                    f"attribute_negative_weight shape mismatch: "
                    f"negative_weight={tuple(negative_weight.shape)} logits={tuple(attribute_logits.shape)}"
                )
            positive_term = -target * F.logsigmoid(attribute_logits)
            negative_term = -negative_weight.clamp(0.0, 1.0) * (1.0 - target) * F.logsigmoid(-attribute_logits)
            normalizer = (
                gate * (target + negative_weight.clamp(0.0, 1.0) * (1.0 - target))
            ).sum().clamp_min(1e-6)
            l2 = (gate.clamp(0.0, 1.0) * (positive_term + negative_term)).sum() / normalizer
            l2 = l2 * (float(attribute_temperature) ** 2)
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
