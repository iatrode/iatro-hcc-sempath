from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from ..modeling.models import (
    bounded_logits,
    clamp_probability,
    normalized_prototype_logits,
)
from ..modeling.prototypes import PrototypeRegistry


def _raise_if_true(condition: torch.Tensor, message: str) -> None:
    reduced = condition.any()
    assert_async = getattr(torch, "_assert_async", None)
    if reduced.device.type == "cuda" and assert_async is not None:
        assert_async(~reduced, message)
        return
    if bool(reduced):
        raise ValueError(message)


@dataclass(frozen=True)
class PAMTDAdjudication:
    """Per-tile teacher reliability and the shared semantic response target."""

    teacher_sample_weights: dict[str, torch.Tensor]
    primary_target: torch.Tensor
    response_sample_weight: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


def _primary_response(
    features: torch.Tensor,
    registry: PrototypeRegistry,
    *,
    class_names: list[str] | tuple[str, ...],
    temperature: float,
) -> torch.Tensor:
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError(f"primary temperature must be positive, got {temperature}")
    available = [
        registry.names[index]
        for index in registry.primary_indices
    ]
    positions = {name: index for index, name in enumerate(available)}
    missing = [name for name in class_names if name not in positions]
    if missing:
        raise ValueError(
            f"teacher prototype registry is missing L1 classes: {missing}"
        )
    logits = bounded_logits(
        normalized_prototype_logits(
            features,
            registry.primary_prototypes,
        )
        / float(temperature)
    )
    response = F.softmax(logits, dim=-1)
    order = [positions[name] for name in class_names]
    ordered_response = (
        response
        if order == list(range(len(order)))
        else response.index_select(
            1,
            torch.tensor(
                order,
                device=features.device,
                dtype=torch.long,
            ),
        )
    )
    return clamp_probability(
        ordered_response,
        normalize=True,
    )


def _global_l2_response(
    features: torch.Tensor,
    prototypes: torch.Tensor,
    counts: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError(f"L2 temperature must be positive, got {temperature}")
    logits = bounded_logits(
        normalized_prototype_logits(features, prototypes)
        / float(temperature)
    )
    ready = (counts > 0).view(1, -1)
    return torch.where(
        ready,
        clamp_probability(torch.sigmoid(logits)),
        torch.full_like(logits, 0.5),
    )


def _l1_agreement(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left * right).sum(dim=-1).clamp(0.0, 1.0)


def _l2_agreement(
    left: torch.Tensor,
    right: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    valid = valid.to(device=left.device, dtype=left.dtype)
    valid_count = valid.sum(dim=-1)
    difference = (left - right).abs() * valid
    agreement = 1.0 - difference.sum(dim=-1) / valid_count.clamp_min(1.0)
    return torch.where(
        valid_count > 0,
        agreement.clamp(0.0, 1.0),
        torch.ones_like(agreement),
    )


@torch.no_grad()
def prototype_adjudicated_teacher_target(
    *,
    teacher_by_name: dict[str, torch.Tensor],
    prototypes_by_teacher: dict[str, PrototypeRegistry],
    student_primary_response: torch.Tensor,
    class_names: list[str] | tuple[str, ...],
    teacher_l2_prototypes: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    student_l2_response: torch.Tensor | None = None,
    l1_mask: torch.Tensor | None = None,
    l1_target: torch.Tensor | None = None,
    l2_target: torch.Tensor | None = None,
    l2_known: torch.Tensor | None = None,
    teacher_weights: dict[str, float] | None = None,
    filter_strength: float = 1.0,
    alpha_min: float = 0.25,
    consensus_weight: float = 1.0,
    prototype_label_weight: float = 1.0,
    student_agreement_weight: float = 1.0,
    l1_agreement_weight: float = 0.5,
    l2_agreement_weight: float = 0.5,
    primary_temperature: float = 0.1,
    l2_temperature: float = 0.1,
) -> PAMTDAdjudication:
    """Restore PAMT-D without turning spatial L2 back into a tile classifier.

    L2 global responses are used only as a common reliability coordinate.
    Human geometry remains the sole source of spatial targets and outputs.
    """

    if set(teacher_by_name) != set(prototypes_by_teacher):
        raise ValueError(
            "teacher/prototype names differ: "
            f"teacher={sorted(teacher_by_name)} "
            f"prototypes={sorted(prototypes_by_teacher)}"
        )
    if not teacher_by_name:
        raise ValueError("PAMT-D requires at least one teacher")
    if not class_names:
        raise ValueError("PAMT-D requires the ordered L1 class names")
    if not 0.0 <= float(filter_strength) <= 1.0:
        raise ValueError(
            f"filter_strength must be in [0, 1], got {filter_strength}"
        )
    if not 0.0 <= float(alpha_min) <= 1.0:
        raise ValueError(f"alpha_min must be in [0, 1], got {alpha_min}")

    names = sorted(teacher_by_name)
    unknown_teacher_weights = set(teacher_weights or {}).difference(names)
    if unknown_teacher_weights:
        raise ValueError(
            "teacher weights contain unknown teachers: "
            f"{sorted(unknown_teacher_weights)}"
        )
    base_weights: dict[str, float] = {}
    for name in names:
        value = float((teacher_weights or {}).get(name, 1.0))
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                "teacher weight must be finite and non-negative: "
                f"teacher={name} weight={value}"
            )
        base_weights[name] = value
    active_names = [name for name in names if base_weights[name] > 0]
    if not active_names:
        raise ValueError("at least one teacher must have a positive base weight")
    for label, value in (
        ("consensus_weight", consensus_weight),
        ("prototype_label_weight", prototype_label_weight),
        ("student_agreement_weight", student_agreement_weight),
        ("l1_agreement_weight", l1_agreement_weight),
        ("l2_agreement_weight", l2_agreement_weight),
    ):
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{label} must be finite and non-negative")

    primary = {
        name: _primary_response(
            teacher_by_name[name],
            prototypes_by_teacher[name],
            class_names=class_names,
            temperature=primary_temperature,
        )
        for name in active_names
    }
    primary_stack = torch.stack(
        [primary[name] for name in active_names],
        dim=0,
    )
    if student_primary_response.shape != primary_stack.shape[1:]:
        raise ValueError(
            "student L1 response shape mismatch: "
            f"student={tuple(student_primary_response.shape)} "
            f"expected={tuple(primary_stack.shape[1:])}"
        )
    student_primary_response = clamp_probability(
        student_primary_response.to(
            device=primary_stack.device,
            dtype=primary_stack.dtype,
        ),
        normalize=True,
    )

    l2_by_teacher: dict[str, torch.Tensor] = {}
    if teacher_l2_prototypes:
        for name in active_names:
            state = teacher_l2_prototypes.get(name)
            if state is None:
                continue
            l2_by_teacher[name] = _global_l2_response(
                teacher_by_name[name],
                state[0],
                state[1],
                temperature=l2_temperature,
            )
    l2_ready = (
        len(l2_by_teacher) == len(active_names)
        and student_l2_response is not None
        and student_l2_response.shape[1] > 0
    )
    if l2_ready:
        valid_components = torch.stack(
            [
                teacher_l2_prototypes[name][1] > 0
                for name in active_names
            ],
            dim=0,
        ).all(dim=0)
        l2_available = valid_components.any()
        use_l2 = True
    else:
        valid_components = None
        l2_available = None
        use_l2 = False
    l2_stack = (
        torch.stack(
            [l2_by_teacher[name] for name in active_names],
            dim=0,
        )
        if use_l2
        else None
    )
    l2_valid = (
        valid_components.view(1, -1).expand(primary_stack.shape[1], -1)
        if use_l2 and valid_components is not None
        else None
    )

    sample_weights: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, torch.Tensor] = {}
    weighted_primary = torch.zeros_like(primary_stack[0])
    primary_weight_sum = torch.zeros(
        (primary_stack.shape[1], 1),
        device=primary_stack.device,
        dtype=primary_stack.dtype,
    )
    base_weight_sum = sum(base_weights[name] for name in active_names)
    base_primary = sum(
        primary[name] * base_weights[name]
        for name in active_names
    )
    base_primary_target = base_primary / float(base_weight_sum)
    base_l2 = (
        sum(
            l2_by_teacher[name] * base_weights[name]
            for name in active_names
        )
        if use_l2
        else None
    )
    l1_mask_value = (
        torch.zeros(primary_stack.shape[1], dtype=torch.bool, device=primary_stack.device)
        if l1_mask is None
        else l1_mask.to(device=primary_stack.device, dtype=torch.bool)
    )
    l1_target_value = (
        torch.full(
            (primary_stack.shape[1],),
            -1,
            dtype=torch.long,
            device=primary_stack.device,
        )
        if l1_target is None
        else l1_target.to(device=primary_stack.device, dtype=torch.long)
    )
    invalid_l1 = l1_mask_value & (
        (l1_target_value < 0)
        | (l1_target_value >= primary_stack.shape[-1])
    )
    _raise_if_true(
        invalid_l1,
        "L1 expert target is outside the prototype response range",
    )
    l2_target_value = (
        None
        if l2_target is None
        else l2_target.to(device=primary_stack.device, dtype=primary_stack.dtype)
    )
    l2_known_value = (
        None
        if l2_known is None
        else l2_known.to(device=primary_stack.device, dtype=torch.bool)
    )

    for name in active_names:
        if len(active_names) == 1:
            consensus = primary[name].new_ones(primary[name].shape[0])
        else:
            peer_weight = base_weight_sum - base_weights[name]
            consensus_primary = (
                base_primary - primary[name] * base_weights[name]
            ) / float(peer_weight)
            consensus = _l1_agreement(primary[name], consensus_primary)
        student_agreement = _l1_agreement(
            primary[name],
            student_primary_response,
        )
        if use_l2 and l2_stack is not None and l2_valid is not None:
            if len(active_names) == 1:
                l2_consensus = consensus.new_ones(consensus.shape)
            else:
                assert base_l2 is not None
                peer_weight = base_weight_sum - base_weights[name]
                consensus_l2 = (
                    base_l2
                    - l2_by_teacher[name] * base_weights[name]
                ) / float(peer_weight)
                l2_consensus = _l2_agreement(
                    l2_by_teacher[name],
                    consensus_l2,
                    l2_valid,
                )
            l2_student = _l2_agreement(
                l2_by_teacher[name],
                student_l2_response,
                l2_valid,
            )
            assert l2_available is not None
            l2_consensus = torch.where(
                l2_available,
                l2_consensus,
                consensus,
            )
            l2_student = torch.where(
                l2_available,
                l2_student,
                student_agreement,
            )
            axis_weight = float(l1_agreement_weight) + float(l2_agreement_weight)
            if axis_weight > 0:
                consensus = (
                    float(l1_agreement_weight) * consensus
                    + float(l2_agreement_weight) * l2_consensus
                ) / axis_weight
                student_agreement = (
                    float(l1_agreement_weight) * student_agreement
                    + float(l2_agreement_weight) * l2_student
                ) / axis_weight

        expert_sum = torch.zeros_like(consensus)
        expert_denominator = torch.zeros_like(consensus)
        safe_target = l1_target_value.clamp(
            0,
            primary_stack.shape[-1] - 1,
        )
        l1_expert = primary[name].gather(
            1,
            safe_target.view(-1, 1),
        ).squeeze(1)
        expert_sum = expert_sum + torch.where(
            l1_mask_value,
            l1_expert,
            torch.zeros_like(l1_expert),
        )
        expert_denominator = (
            expert_denominator + l1_mask_value.float()
        )
        if (
            use_l2
            and l2_target_value is not None
            and l2_known_value is not None
            and l2_valid is not None
        ):
            valid = l2_known_value & l2_valid
            valid_count = valid.sum(dim=-1)
            l2_expert = 1.0 - (
                (l2_by_teacher[name] - l2_target_value).abs()
                * valid.float()
            ).sum(dim=-1) / valid_count.clamp_min(1)
            expert_sum = expert_sum + torch.where(
                valid_count > 0,
                l2_expert,
                torch.zeros_like(l2_expert),
            )
            expert_denominator = expert_denominator + (valid_count > 0).float()
        expert = expert_sum / expert_denominator.clamp_min(1.0)
        has_expert = expert_denominator > 0

        numerator = (
            float(consensus_weight) * consensus
            + float(student_agreement_weight) * student_agreement
            + float(prototype_label_weight) * expert * has_expert.float()
        )
        denominator = (
            float(consensus_weight)
            + float(student_agreement_weight)
            + float(prototype_label_weight) * has_expert.float()
        )
        reliability = (numerator / denominator.clamp_min(1e-6)).clamp(0.0, 1.0)
        alpha_raw = float(alpha_min) + (1.0 - float(alpha_min)) * reliability
        alpha = 1.0 - float(filter_strength) * (1.0 - alpha_raw)
        sample_weights[name] = alpha.detach()
        diagnostics[f"{name}_alpha_mean"] = alpha.mean().detach()
        diagnostics[f"{name}_reliability_mean"] = reliability.mean().detach()

        base_weight = base_weights[name]
        combined = alpha.view(-1, 1) * base_weight
        weighted_primary = weighted_primary + primary[name] * combined
        primary_weight_sum = primary_weight_sum + combined

    for name in names:
        if name not in sample_weights:
            zeros = primary_stack.new_zeros(primary_stack.shape[1])
            sample_weights[name] = zeros
            diagnostics[f"{name}_alpha_mean"] = zeros.mean()
            diagnostics[f"{name}_reliability_mean"] = zeros.mean()

    has_response_mass = primary_weight_sum > 0
    primary_target = torch.where(
        has_response_mass,
        weighted_primary / primary_weight_sum.clamp_min(1e-12),
        base_primary_target,
    )
    primary_target = clamp_probability(
        primary_target,
        normalize=True,
    )
    response_sample_weight = (
        primary_weight_sum.squeeze(1) / float(base_weight_sum)
    ).clamp(0.0, 1.0)
    diagnostics["teacher_alpha_mean"] = torch.stack(
        [sample_weights[name].mean() for name in active_names]
    ).mean()
    diagnostics["response_sample_weight_mean"] = (
        response_sample_weight.mean().detach()
    )
    return PAMTDAdjudication(
        teacher_sample_weights=sample_weights,
        primary_target=primary_target.detach(),
        response_sample_weight=response_sample_weight.detach(),
        diagnostics=diagnostics,
    )


def prototype_response_distillation_loss(
    fixed_temperature_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Distil the adjudicated response in one fixed-temperature coordinate.

    ``fixed_temperature_logits`` must already be
    ``bounded(cosine / temperature)``. The temperature is not applied again;
    it remains an argument only for the conventional ``temperature**2``
    gradient scale.
    """

    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError(f"response temperature must be positive, got {temperature}")
    target = clamp_probability(
        target.detach().to(
            device=fixed_temperature_logits.device,
            dtype=torch.float32,
        ),
        normalize=True,
    )
    if target.shape != fixed_temperature_logits.shape:
        raise ValueError(
            "response target shape mismatch: "
            f"target={tuple(target.shape)} "
            f"logits={tuple(fixed_temperature_logits.shape)}"
        )
    per_sample = F.kl_div(
        F.log_softmax(
            fixed_temperature_logits.float(),
            dim=-1,
        ),
        target,
        reduction="none",
    ).sum(dim=-1)
    if sample_weight is None:
        loss = per_sample.mean()
    else:
        weight = sample_weight.detach().to(
            device=per_sample.device,
            dtype=per_sample.dtype,
        ).view(-1)
        if weight.shape != per_sample.shape:
            raise ValueError(
                "response sample weight shape mismatch: "
                f"weight={tuple(weight.shape)} "
                f"expected={tuple(per_sample.shape)}"
            )
        _raise_if_true(
            ~torch.isfinite(weight) | (weight < 0),
            "response sample weights must be finite and non-negative",
        )
        denominator = weight.sum()
        invalid = denominator <= 0
        assert_async = getattr(torch, "_assert_async", None)
        if invalid.device.type == "cuda" and assert_async is not None:
            assert_async(
                ~invalid,
                "response sample weights have zero total mass",
            )
        elif bool(invalid):
            raise ValueError("response sample weights have zero total mass")
        loss = (per_sample * weight).sum() / denominator
    return loss * float(temperature) ** 2
