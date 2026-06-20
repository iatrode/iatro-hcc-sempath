from __future__ import annotations

from dataclasses import dataclass

import torch

from ..modeling.prototypes import PrototypeRegistry
from .zhcc_losses import prototype_response


def _prototype_names(registry: PrototypeRegistry, indices: list[int]) -> list[str]:
    return [registry.names[idx] for idx in indices]


def _teacher_response(
    features: torch.Tensor,
    registry: PrototypeRegistry,
    *,
    primary_names: list[str],
    attribute_names: list[str],
    label: str,
    primary_temperature: float = 0.1,
    attribute_temperature: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    return prototype_response(
        features,
        registry,
        primary_names=primary_names,
        attribute_names=attribute_names,
        label=label,
        primary_temperature=primary_temperature,
        attribute_temperature=attribute_temperature,
    )


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


@dataclass(frozen=True)
class AttributeAdjudicationTarget:
    target: torch.Tensor
    gate: torch.Tensor
    negative_weight: torch.Tensor
    reliability_by_teacher: dict[str, torch.Tensor]
    diagnostics: dict[str, torch.Tensor]


def _weighted_teacher_variance(
    responses: torch.Tensor,
    weights: torch.Tensor,
    mean: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    numerator = (weights * (responses - mean.unsqueeze(0)).square()).sum(dim=0)
    denom = weights.sum(dim=0).clamp_min(eps)
    variance = numerator / denom
    bernoulli_ceiling = (mean * (1.0 - mean)).clamp_min(eps)
    return (variance / bernoulli_ceiling).clamp(0.0, 1.0)


def _normalized_binary_entropy(probability: torch.Tensor, *, eps: float) -> torch.Tensor:
    p = probability.clamp(eps, 1.0 - eps)
    return (-(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)) / torch.log(p.new_tensor(2.0))).clamp(0.0, 1.0)


def _prior_tensor_for_teacher(
    *,
    name: str,
    attribute_names: list[str],
    payload: dict[str, torch.Tensor | list[float] | tuple[float, ...] | dict[str, float]] | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if payload is None or name not in payload:
        return torch.ones(len(attribute_names), device=device, dtype=dtype)
    value = payload[name]
    if isinstance(value, dict):
        missing = [attribute for attribute in attribute_names if attribute not in value]
        if missing:
            raise ValueError(f"teacher_attribute_prior[{name!r}] missing attributes: {missing}")
        tensor = torch.tensor([float(value[attribute]) for attribute in attribute_names], device=device, dtype=dtype)
    else:
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.shape != (len(attribute_names),):
        raise ValueError(
            f"teacher_attribute_prior[{name!r}] must have shape=({len(attribute_names)},), got {tuple(tensor.shape)}"
        )
    return tensor.clamp(0.0, 1.0)


@torch.no_grad()
def attribute_adjudicated_level2_target(
    *,
    teacher_by_name: dict[str, torch.Tensor],
    prototypes_by_teacher: dict[str, PrototypeRegistry],
    target_registry: PrototypeRegistry,
    teacher_weights: dict[str, float] | None = None,
    teacher_attribute_prior: dict[str, torch.Tensor | list[float] | tuple[float, ...] | dict[str, float]] | None = None,
    prototype_mask: torch.Tensor | None = None,
    prototype_level2: torch.Tensor | None = None,
    consensus_weight: float = 1.0,
    prototype_label_weight: float = 1.0,
    uncertainty_eta: float = 0.5,
    negative_evidence_power: float = 2.0,
    epsilon: float = 1e-6,
    primary_temperature: float = 0.1,
    attribute_temperature: float = 0.1,
) -> AttributeAdjudicationTarget:
    """Build an independently adjudicated multi-label Level-2 teacher target.

    This path intentionally estimates teacher reliability per attribute. The
    resulting weights are used only for the Level-2 semantic response target;
    global tile-level alpha remains the feature/relation distillation contract.
    """
    if set(teacher_by_name) != set(prototypes_by_teacher):
        raise ValueError(
            f"teacher/prototype names differ: teacher={sorted(teacher_by_name)} "
            f"prototypes={sorted(prototypes_by_teacher)}"
        )
    if not teacher_by_name:
        raise ValueError("at least one teacher is required for Level-2 adjudication")
    eta = float(uncertainty_eta)
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"uncertainty_eta must be in [0, 1], got {uncertainty_eta}")
    neg_power = float(negative_evidence_power)
    if neg_power <= 0:
        raise ValueError(f"negative_evidence_power must be positive, got {negative_evidence_power}")
    eps = float(epsilon)
    if eps <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    primary_names = _prototype_names(target_registry, target_registry.primary_indices)
    attribute_names = _prototype_names(target_registry, target_registry.attribute_indices)
    first_teacher = next(iter(teacher_by_name.values()))
    batch_size = int(first_teacher.shape[0])
    device = first_teacher.device
    if not attribute_names:
        empty = first_teacher.new_zeros((batch_size, 0))
        return AttributeAdjudicationTarget(empty, empty, empty, {}, {})

    responses = {
        name: _teacher_response(
            features,
            prototypes_by_teacher[name],
            primary_names=primary_names,
            attribute_names=attribute_names,
            label=name,
            primary_temperature=primary_temperature,
            attribute_temperature=attribute_temperature,
        )[1]
        for name, features in teacher_by_name.items()
    }
    names = sorted(responses)
    attribute_stack = torch.stack([responses[name] for name in names], dim=0)
    if teacher_weights is None:
        base_weights = attribute_stack.new_ones((len(names), 1, 1))
    else:
        base_weights = attribute_stack.new_tensor(
            [float(teacher_weights.get(name, 1.0)) for name in names]
        ).view(-1, 1, 1)
    if bool((base_weights <= 0).all()):
        raise ValueError("at least one teacher must have a positive loss weight")

    if prototype_mask is None:
        proto_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
    else:
        proto_mask = prototype_mask.to(device=device, dtype=torch.bool)
    if proto_mask.shape != (batch_size,):
        raise ValueError(f"prototype_mask must have shape=({batch_size},), got {tuple(proto_mask.shape)}")
    if prototype_level2 is None:
        proto_l2 = first_teacher.new_zeros((batch_size, len(attribute_names)))
    else:
        proto_l2 = prototype_level2.to(device=device, dtype=attribute_stack.dtype)
    if proto_l2.shape != (batch_size, len(attribute_names)):
        raise ValueError(
            f"prototype_level2 width mismatch: labels={tuple(proto_l2.shape)} "
            f"expected=({batch_size}, {len(attribute_names)})"
        )

    reliability_by_teacher: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, torch.Tensor] = {}
    weighted_response_sum = attribute_stack.new_zeros((batch_size, len(attribute_names)))
    reliability_sum = attribute_stack.new_zeros((batch_size, len(attribute_names)))
    consensus_w = max(float(consensus_weight), 0.0)
    prototype_w = max(float(prototype_label_weight), 0.0)

    for idx, name in enumerate(names):
        attributes = responses[name]
        if len(names) == 1:
            consensus = torch.ones_like(attributes)
        else:
            mean_other = (attribute_stack.sum(dim=0) - attribute_stack[idx]) / float(len(names) - 1)
            consensus = (1.0 - (attributes - mean_other).abs()).clamp(0.0, 1.0)
        proto_agreement = (1.0 - (attributes - proto_l2).abs()).clamp(0.0, 1.0)
        numerator = consensus_w * consensus
        denominator = attributes.new_full(attributes.shape, consensus_w)
        if prototype_w > 0:
            numerator = numerator + prototype_w * proto_agreement * proto_mask[:, None].to(attributes.dtype)
            denominator = denominator + prototype_w * proto_mask[:, None].to(attributes.dtype)
        local_confidence = torch.where(
            denominator > 0,
            numerator / denominator.clamp_min(eps),
            torch.ones_like(attributes),
        ).clamp(0.0, 1.0)
        prior = _prior_tensor_for_teacher(
            name=name,
            attribute_names=attribute_names,
            payload=teacher_attribute_prior,
            device=device,
            dtype=attributes.dtype,
        ).view(1, -1)
        reliability = (eps + (1.0 - eps) * (prior * local_confidence).clamp(0.0, 1.0)).clamp(eps, 1.0)
        reliability_by_teacher[name] = reliability
        effective_weight = reliability * base_weights[idx].clamp_min(0.0)
        weighted_response_sum = weighted_response_sum + effective_weight * attributes
        reliability_sum = reliability_sum + effective_weight
        diagnostics[f"l2_attr_reliability_mean/{name}"] = reliability.float().mean().detach()
        diagnostics[f"l2_attr_reliability_min/{name}"] = reliability.float().min().detach()

    target = (weighted_response_sum / reliability_sum.clamp_min(eps)).clamp(0.0, 1.0)
    response_weights = base_weights.clamp_min(0.0) * torch.stack(
        [reliability_by_teacher[name] for name in names],
        dim=0,
    )
    variance = _weighted_teacher_variance(attribute_stack, response_weights, target, eps=eps)
    uncertainty = _normalized_binary_entropy(target, eps=eps) * (1.0 - variance)
    gate = (1.0 - eta * uncertainty).clamp(0.0, 1.0)
    gate = torch.where(proto_mask[:, None], torch.ones_like(gate), gate)

    negative_weight = ((1.0 - target).clamp(0.0, 1.0).pow(neg_power) * (1.0 - variance)).clamp(0.0, 1.0)
    anchored_negative = proto_mask[:, None] & (proto_l2 <= 0.0)
    negative_weight = torch.where(anchored_negative, torch.ones_like(negative_weight), negative_weight)
    diagnostics["l2_attr_target_mean"] = target.float().mean().detach()
    diagnostics["l2_attr_gate_mean"] = gate.float().mean().detach()
    diagnostics["l2_attr_negative_weight_mean"] = negative_weight.float().mean().detach()
    diagnostics["l2_attr_teacher_variance_mean"] = variance.float().mean().detach()
    return AttributeAdjudicationTarget(
        target=target.detach(),
        gate=gate.detach(),
        negative_weight=negative_weight.detach(),
        reliability_by_teacher={name: value.detach() for name, value in reliability_by_teacher.items()},
        diagnostics=diagnostics,
    )


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
    primary_temperature: float = 0.1,
    attribute_temperature: float = 0.1,
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
            primary_temperature=primary_temperature,
            attribute_temperature=attribute_temperature,
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
            primary_temperature=primary_temperature,
            attribute_temperature=attribute_temperature,
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
