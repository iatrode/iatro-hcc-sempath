from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from hcc_sempath.spatial_schema import (
    CELL_INSTANCE_DENSITY,
    DEFAULT_SPATIAL_COMPONENTS,
    STRUCTURE_INSTANCE_AREA,
    spatial_component_specs,
)


def _mean_supervised_pair(
    values: torch.Tensor,
    supervised: torch.Tensor,
    zero_source: torch.Tensor,
) -> torch.Tensor:
    supervised = supervised.to(device=values.device, dtype=torch.bool)
    if not bool(supervised.any()):
        return zero_source.sum() * 0.0
    return values[supervised].mean()


def l1_classification_loss(
    l1_logits: torch.Tensor,
    prototype_mask: torch.Tensor,
    prototype_level1: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mask = prototype_mask.to(device=l1_logits.device, dtype=torch.bool)
    if not bool(mask.any()):
        zero = l1_logits.sum() * 0.0
        return zero, {
            "l1": zero.detach(),
            "l1_supervised_tiles": mask.sum().detach(),
            "l1_accuracy": zero.detach(),
        }
    targets = prototype_level1.to(
        device=l1_logits.device,
        dtype=torch.long,
    )[mask]
    logits = l1_logits[mask]
    if int(targets.min()) < 0 or int(targets.max()) >= l1_logits.shape[1]:
        raise ValueError(
            f"L1 target out of range: min={int(targets.min())} "
            f"max={int(targets.max())} classes={l1_logits.shape[1]}"
        )
    loss = F.cross_entropy(logits, targets)
    accuracy = (logits.argmax(dim=1) == targets).float().mean()
    return loss, {
        "l1": loss.detach(),
        "l1_supervised_tiles": mask.sum().detach(),
        "l1_accuracy": accuracy.detach(),
    }


def _minimum_cost_assignment(cost: list[list[float]]) -> list[int]:
    """Solve a rectangular row assignment with the Hungarian algorithm."""

    row_count = len(cost)
    if row_count == 0:
        return []
    column_count = len(cost[0])
    if column_count < row_count or any(
        len(row) != column_count for row in cost
    ):
        raise ValueError("assignment cost must be rectangular with columns >= rows")
    u = [0.0] * (row_count + 1)
    v = [0.0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    path = [0] * (column_count + 1)
    for row_idx in range(1, row_count + 1):
        matched_row[0] = row_idx
        minimum = [float("inf")] * (column_count + 1)
        used = [False] * (column_count + 1)
        column = 0
        while True:
            used[column] = True
            current_row = matched_row[column]
            delta = float("inf")
            next_column = 0
            for candidate_column in range(1, column_count + 1):
                if used[candidate_column]:
                    continue
                reduced = (
                    cost[current_row - 1][candidate_column - 1]
                    - u[current_row]
                    - v[candidate_column]
                )
                if reduced < minimum[candidate_column]:
                    minimum[candidate_column] = reduced
                    path[candidate_column] = column
                if minimum[candidate_column] < delta:
                    delta = minimum[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(column_count + 1):
                if used[candidate_column]:
                    u[matched_row[candidate_column]] += delta
                    v[candidate_column] -= delta
                else:
                    minimum[candidate_column] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = path[column]
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break
    assignment = [-1] * row_count
    for column in range(1, column_count + 1):
        row = matched_row[column]
        if row:
            assignment[row - 1] = column - 1
    return assignment


def _maximum_cardinality_score_matching(
    candidates: list[list[int]],
    scores: torch.Tensor,
    *,
    width: int,
) -> dict[int, int]:
    """Match clicks to cells: maximum cardinality first, score second."""

    if not candidates:
        return {}
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError("non-finite instance logits in point matching")
    cells = sorted({cell for choices in candidates for cell in choices})
    positions = {cell: index for index, cell in enumerate(cells)}
    row_count = len(candidates)
    cardinality_penalty = float(4 * row_count + 1)
    forbidden = 2.0 * cardinality_penalty
    cost = [
        [forbidden] * len(cells) + [cardinality_penalty] * row_count
        for _ in range(row_count)
    ]
    for click_idx, choices in enumerate(candidates):
        for cell in choices:
            score = float(scores[cell // width, cell % width])
            # Bounded secondary cost keeps one additional real match more
            # valuable than every possible score rearrangement combined.
            cost[click_idx][positions[cell]] = 1.0 - math.tanh(score)
    assignment = _minimum_cost_assignment(cost)
    matched: dict[int, int] = {}
    for click_idx, column in enumerate(assignment):
        if column < len(cells):
            cell = cells[column]
            if cell in candidates[click_idx]:
                matched[click_idx] = cell
    return matched


def _point_peak_loss(
    logits: torch.Tensor,
    point_centers: torch.Tensor,
    *,
    tolerance_cells: int,
    exclusive: bool = True,
    exclusion_support: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match each click to one peak and suppress extras in known object support."""

    if tolerance_cells < 0:
        raise ValueError(
            f"tolerance_cells must be non-negative, got {tolerance_cells}"
        )
    centers = point_centers.to(device=logits.device, dtype=torch.float32)
    if exclusion_support is not None:
        if exclusion_support.shape != logits.shape:
            raise ValueError(
                "point exclusion support shape mismatch: "
                f"support={tuple(exclusion_support.shape)} "
                f"logits={tuple(logits.shape)}"
            )
        exclusion_support = exclusion_support.to(
            device=logits.device,
            dtype=torch.bool,
        )
    point_count = centers.flatten(2).sum(dim=2)
    supervised = point_count > 0
    pair_loss = logits.new_zeros(point_count.shape)
    height, width = logits.shape[-2:]
    for batch_idx, component_idx in supervised.nonzero().tolist():
        click_cells: list[tuple[int, int]] = []
        for row, col in (
            centers[batch_idx, component_idx] > 0
        ).nonzero().tolist():
            multiplicity = max(
                1,
                int(round(float(centers[batch_idx, component_idx, row, col]))),
            )
            click_cells.extend([(int(row), int(col))] * multiplicity)

        candidates: list[list[int]] = []
        scores = logits[batch_idx, component_idx].detach()
        for row, col in click_cells:
            cells = [
                candidate_row * width + candidate_col
                for candidate_row in range(
                    max(0, row - tolerance_cells),
                    min(height, row + tolerance_cells + 1),
                )
                for candidate_col in range(
                    max(0, col - tolerance_cells),
                    min(width, col + tolerance_cells + 1),
                )
            ]
            cells.sort(
                key=lambda cell: float(
                    scores[cell // width, cell % width]
                ),
                reverse=True,
            )
            candidates.append(cells)

        matched_cell = _maximum_cardinality_score_matching(
            candidates,
            scores,
            width=width,
        )

        selected_logits: list[torch.Tensor] = []
        selected_cells: set[int] = set()
        for click_idx in range(len(click_cells)):
            # More clicks than resolvable grid cells can occur after coordinate
            # quantization. Reuse is then unavoidable and remains visible in
            # the count target rather than silently discarding the click.
            cell = matched_cell.get(click_idx, candidates[click_idx][0])
            selected_cells.add(cell)
            selected_logits.append(
                logits[
                    batch_idx,
                    component_idx,
                    cell // width,
                    cell % width,
                ]
            )
        positive_loss = torch.stack(
            [F.softplus(-value) for value in selected_logits]
        ).mean()
        if exclusive:
            exclusive_cells = {
                cell for choices in candidates for cell in choices
            }
            if exclusion_support is not None:
                exclusive_cells.update(
                    int(row) * width + int(col)
                    for row, col in exclusion_support[
                        batch_idx,
                        component_idx,
                    ].nonzero().tolist()
                )
            extra_cells = sorted(exclusive_cells - selected_cells)
            if extra_cells:
                extra_logits = torch.stack(
                    [
                        logits[
                            batch_idx,
                            component_idx,
                            cell // width,
                            cell % width,
                        ]
                        for cell in extra_cells
                    ]
                )
                positive_loss = positive_loss + F.softplus(
                    extra_logits
                ).mean()
        pair_loss[batch_idx, component_idx] = positive_loss
    return (
        _mean_supervised_pair(pair_loss, supervised, logits),
        supervised,
        point_count,
    )


def _routed_negative_loss(
    instance_logits: torch.Tensor,
    measurement_logits: torch.Tensor,
    mask: torch.Tensor,
    countable: torch.Tensor,
    *,
    instance_pair_valid: torch.Tensor | None = None,
    measurement_pair_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply negatives to valid heads without halving area-only components."""

    mask = mask.to(device=instance_logits.device, dtype=torch.bool)
    valid_count = mask.flatten(2).sum(dim=2)
    instance_pair = (
        (F.softplus(instance_logits) * mask).flatten(2).sum(dim=2)
        / valid_count.clamp_min(1)
    )
    measurement_pair = (
        (F.softplus(measurement_logits) * mask).flatten(2).sum(dim=2)
        / valid_count.clamp_min(1)
    )
    supervised = valid_count > 0
    countable_pair = countable.reshape(1, -1)
    instance_supervised = supervised & countable_pair
    measurement_supervised = supervised
    if instance_pair_valid is not None:
        instance_supervised &= instance_pair_valid.to(
            device=instance_logits.device,
            dtype=torch.bool,
        )
    if measurement_pair_valid is not None:
        measurement_supervised &= measurement_pair_valid.to(
            device=instance_logits.device,
            dtype=torch.bool,
        )
    objective_count = (
        instance_supervised.float() + measurement_supervised.float()
    )
    pair_loss = (
        instance_pair * instance_supervised.float()
        + measurement_pair * measurement_supervised.float()
    ) / objective_count.clamp_min(1.0)
    any_supervised = objective_count > 0
    return (
        _mean_supervised_pair(
            pair_loss,
            any_supervised,
            instance_logits,
        ),
        instance_supervised,
        measurement_supervised,
    )


def _positive_area_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Positive occupancy support without inventing unmarked boundaries."""

    mask = mask.to(device=logits.device, dtype=torch.bool)
    valid_count = mask.flatten(2).sum(dim=2)
    pair_loss = (
        (F.softplus(-logits) * mask).flatten(2).sum(dim=2)
        / valid_count.clamp_min(1)
    )
    supervised = valid_count > 0
    return _mean_supervised_pair(pair_loss, supervised, logits), supervised


def _brush_bag_loss(
    logits: torch.Tensor,
    bag_ids: torch.Tensor,
    *,
    top_fraction: float,
) -> tuple[torch.Tensor, int, int]:
    """Positive multiple-instance loss over dense-cell brush bags.

    Only the strongest configured fraction of cells contributes to a bag's
    positive score. This requires distributed evidence without declaring every
    covered cell positive or inventing instance centres.
    """

    if not 0.0 < top_fraction <= 1.0:
        raise ValueError(
            f"brush top_fraction must be in (0, 1], got {top_fraction}"
        )
    ids = bag_ids.to(device=logits.device, dtype=torch.long)
    pair_losses: list[torch.Tensor] = []
    bag_count = 0
    pair_count = 0
    for batch_idx in range(logits.shape[0]):
        for component_idx in range(logits.shape[1]):
            component_ids = torch.unique(ids[batch_idx, component_idx])
            component_ids = component_ids[component_ids > 0]
            if component_ids.numel() == 0:
                continue
            losses: list[torch.Tensor] = []
            for bag_id in component_ids.tolist():
                values = logits[batch_idx, component_idx][
                    ids[batch_idx, component_idx] == int(bag_id)
                ]
                if values.numel() == 0:  # pragma: no cover - guarded by unique
                    continue
                keep = max(1, int(math.ceil(values.numel() * top_fraction)))
                evidence = torch.topk(values, keep, sorted=False).values.mean()
                losses.append(F.softplus(-evidence))
                bag_count += 1
            if losses:
                pair_losses.append(torch.stack(losses).mean())
                pair_count += 1
    if not pair_losses:
        return logits.sum() * 0.0, 0, 0
    return torch.stack(pair_losses).mean(), bag_count, pair_count


def spatial_morphometry_loss(
    *,
    instance_logits: torch.Tensor,
    abundance_logits: torch.Tensor,
    point_centers: torch.Tensor,
    brush_bag_ids: torch.Tensor,
    area_positive: torch.Tensor,
    explicit_negative: torch.Tensor,
    implicit_negative: torch.Tensor,
    component_names: list[str] | tuple[str, ...] | None = None,
    point_tolerance_cells: int = 1,
    abundance_point_weight: float = 0.5,
    brush_weight: float = 1.0,
    brush_top_fraction: float = 0.25,
    explicit_negative_weight: float = 1.0,
    implicit_negative_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Class-routed spatial supervision normalized per tile/component pair."""

    expected = instance_logits.shape
    if abundance_logits.shape != expected:
        raise ValueError(
            "instance/abundance shape mismatch: "
            f"instance={tuple(expected)} abundance={tuple(abundance_logits.shape)}"
        )
    for name, value in (
        ("point_centers", point_centers),
        ("brush_bag_ids", brush_bag_ids),
        ("area_positive", area_positive),
        ("explicit_negative", explicit_negative),
        ("implicit_negative", implicit_negative),
    ):
        if value.shape != expected:
            raise ValueError(
                f"{name} shape mismatch: got={tuple(value.shape)} "
                f"expected={tuple(expected)}"
            )
    for name, value in (
        ("abundance_point_weight", abundance_point_weight),
        ("brush_weight", brush_weight),
        ("explicit_negative_weight", explicit_negative_weight),
        ("implicit_negative_weight", implicit_negative_weight),
    ):
        if float(value) < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")

    if component_names is None:
        if instance_logits.shape[1] == len(DEFAULT_SPATIAL_COMPONENTS):
            resolved_names = DEFAULT_SPATIAL_COMPONENTS
            specs = spatial_component_specs(resolved_names)
        else:
            resolved_names = tuple(
                f"synthetic-{index}"
                for index in range(instance_logits.shape[1])
            )
            specs = spatial_component_specs(
                resolved_names,
                unknown_mode=CELL_INSTANCE_DENSITY,
            )
    else:
        resolved_names = tuple(str(name) for name in component_names)
        if len(resolved_names) != instance_logits.shape[1]:
            raise ValueError(
                "component_names/logit channel mismatch: "
                f"names={len(resolved_names)} channels={instance_logits.shape[1]}"
            )
        specs = spatial_component_specs(
            resolved_names,
            unknown_mode=CELL_INSTANCE_DENSITY,
        )

    countable = torch.tensor(
        [spec.supports_instance_count for spec in specs],
        dtype=torch.bool,
        device=instance_logits.device,
    ).view(1, -1, 1, 1)
    density = torch.tensor(
        [spec.supports_density for spec in specs],
        dtype=torch.bool,
        device=instance_logits.device,
    ).view(1, -1, 1, 1)
    area = torch.tensor(
        [spec.supports_area for spec in specs],
        dtype=torch.bool,
        device=instance_logits.device,
    ).view(1, -1, 1, 1)
    structure = torch.tensor(
        [spec.mode == STRUCTURE_INSTANCE_AREA for spec in specs],
        dtype=torch.bool,
        device=instance_logits.device,
    ).view(1, -1, 1, 1)

    point_bool = point_centers > 0
    bag_bool = brush_bag_ids > 0
    area_bool = area_positive.to(dtype=torch.bool)
    if bool((point_bool & ~countable).any()):
        raise ValueError("non-countable component received instance centres")
    if bool((bag_bool & ~density).any()):
        raise ValueError("non-density component received a brush density bag")
    if bool((area_bool & ~area).any()):
        raise ValueError("non-area component received occupied-area support")

    positive_support = point_bool | bag_bool | area_bool
    explicit_bool = explicit_negative.to(dtype=torch.bool)
    implicit_bool = implicit_negative.to(dtype=torch.bool)
    if bool((positive_support & explicit_bool).any()):
        raise ValueError("positive and explicit-negative spatial targets overlap")
    if bool((positive_support & implicit_bool).any()):
        raise ValueError("positive and implicit-negative spatial targets overlap")
    if bool((explicit_bool & implicit_bool).any()):
        raise ValueError("explicit and implicit negative targets overlap")

    instance_point, point_pairs, point_counts = _point_peak_loss(
        instance_logits,
        point_centers,
        tolerance_cells=point_tolerance_cells,
        exclusion_support=area_bool & structure,
    )
    abundance_point, _, _ = _point_peak_loss(
        abundance_logits,
        point_centers * density.to(dtype=point_centers.dtype),
        tolerance_cells=point_tolerance_cells,
        exclusive=False,
    )
    brush_bag, brush_bags, brush_pairs = _brush_bag_loss(
        abundance_logits,
        brush_bag_ids,
        top_fraction=brush_top_fraction,
    )
    area_positive_loss, area_pairs = _positive_area_loss(
        abundance_logits,
        area_bool,
    )
    (
        explicit_loss,
        explicit_pairs_instance,
        explicit_pairs_abundance,
    ) = _routed_negative_loss(
        instance_logits,
        abundance_logits,
        explicit_bool,
        countable,
    )
    (
        implicit_loss,
        implicit_pairs_instance,
        implicit_pairs_abundance,
    ) = _routed_negative_loss(
        instance_logits,
        abundance_logits,
        implicit_bool,
        countable,
        instance_pair_valid=(
            positive_support.flatten(2).any(dim=2)
            | implicit_bool.flatten(2).all(dim=2)
        ),
        measurement_pair_valid=(
            (
                (point_bool & density)
                | bag_bool
                | area_bool
            )
            .flatten(2)
            .any(dim=2)
            | implicit_bool.flatten(2).all(dim=2)
        ),
    )

    total = (
        instance_point
        + float(abundance_point_weight) * abundance_point
        + float(brush_weight) * brush_bag
        + area_positive_loss
        + float(explicit_negative_weight) * explicit_loss
        + float(implicit_negative_weight) * implicit_loss
    )
    return total, {
        "l2_spatial": total.detach(),
        "l2_instance_point": instance_point.detach(),
        "l2_abundance_point": abundance_point.detach(),
        "l2_brush_bag": brush_bag.detach(),
        "l2_area_positive": area_positive_loss.detach(),
        "l2_explicit_negative": explicit_loss.detach(),
        "l2_implicit_negative": implicit_loss.detach(),
        "l2_point_supervised_pairs": point_pairs.sum().detach(),
        "l2_point_count": point_counts.sum().detach(),
        "l2_brush_supervised_pairs": torch.tensor(
            brush_pairs,
            device=instance_logits.device,
        ),
        "l2_brush_bag_count": torch.tensor(
            brush_bags,
            device=instance_logits.device,
        ),
        "l2_area_supervised_pairs": area_pairs.sum().detach(),
        "l2_explicit_negative_pairs": torch.maximum(
            explicit_pairs_instance.sum(),
            explicit_pairs_abundance.sum(),
        ).detach(),
        "l2_implicit_negative_pairs": torch.maximum(
            implicit_pairs_instance.sum(),
            implicit_pairs_abundance.sum(),
        ).detach(),
    }
