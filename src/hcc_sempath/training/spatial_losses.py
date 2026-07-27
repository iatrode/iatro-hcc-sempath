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
    """Average within each component, then equally across active components."""

    supervised = supervised.to(device=values.device, dtype=torch.bool)
    weight = supervised.to(dtype=values.dtype)
    if values.ndim != 2 or supervised.shape != values.shape:
        raise ValueError(
            "component-balanced pair reduction expects [tile, component] "
            f"values and mask, got values={tuple(values.shape)} "
            f"supervised={tuple(supervised.shape)}"
        )
    component_count = weight.sum(dim=0)
    component_mean = (
        (values * weight).sum(dim=0) / component_count.clamp_min(1)
    )
    active = component_count > 0
    active_weight = active.to(dtype=values.dtype)
    mean = (
        component_mean * active_weight
    ).sum() / active_weight.sum().clamp_min(1)
    return torch.where(
        active.any(),
        mean,
        zero_source.sum() * 0.0,
    )


def _raise_if_true(condition: torch.Tensor, message: str) -> None:
    """Validate a device tensor without synchronizing the CUDA hot path."""

    reduced = condition.any()
    assert_async = getattr(torch, "_assert_async", None)
    if reduced.device.type == "cuda" and assert_async is not None:
        assert_async(~reduced, message)
        return
    if bool(reduced):
        raise ValueError(message)


def l1_classification_loss(
    l1_logits: torch.Tensor,
    prototype_mask: torch.Tensor,
    prototype_level1: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mask = prototype_mask.to(device=l1_logits.device, dtype=torch.bool)
    targets = prototype_level1.to(
        device=l1_logits.device,
        dtype=torch.long,
    )
    _raise_if_true(
        mask
        & ((targets < 0) | (targets >= l1_logits.shape[1])),
        f"L1 target out of range for {l1_logits.shape[1]} classes",
    )
    weight = mask.to(dtype=l1_logits.dtype)
    supervised_count = weight.sum()
    safe_targets = torch.where(mask, targets, torch.zeros_like(targets))
    per_tile = F.cross_entropy(
        l1_logits,
        safe_targets,
        reduction="none",
    )
    loss = (per_tile * weight).sum() / supervised_count.clamp_min(1)
    accuracy = (
        (l1_logits.argmax(dim=1) == safe_targets).to(dtype=l1_logits.dtype)
        * weight
    ).sum() / supervised_count.clamp_min(1)
    return loss, {
        "l1": loss.detach(),
        "l1_supervised_tiles": supervised_count.detach(),
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
    point_centers_host: torch.Tensor | None = None,
    exclusion_support_host: torch.Tensor | None = None,
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
    height, width = logits.shape[-2:]
    centers_cpu = (
        centers.detach().to(device="cpu")
        if point_centers_host is None
        else point_centers_host.detach().to(
            device="cpu",
            dtype=torch.float32,
        )
    )
    if centers_cpu.shape != logits.shape:
        raise ValueError(
            "host point-centre shape mismatch: "
            f"host={tuple(centers_cpu.shape)} logits={tuple(logits.shape)}"
        )
    supervised_cpu = (
        centers_cpu.flatten(2).sum(dim=2) > 0
    )
    if not bool(supervised_cpu.any()):
        return (
            logits.sum() * 0.0,
            supervised,
            point_count,
        )
    supervised_pairs = supervised_cpu.nonzero()
    pair_batch = supervised_pairs[:, 0].to(device=logits.device)
    pair_component = supervised_pairs[:, 1].to(device=logits.device)
    scores_cpu = logits.detach()[
        pair_batch,
        pair_component,
    ].float().to(device="cpu")
    exclusion_cpu = None
    if exclusion_support is not None:
        host_exclusion = (
            exclusion_support.detach().to(device="cpu")
            if exclusion_support_host is None
            else exclusion_support_host.detach().to(
                device="cpu",
                dtype=torch.bool,
            )
        )
        if host_exclusion.shape != logits.shape:
            raise ValueError(
                "host point-exclusion shape mismatch: "
                f"host={tuple(host_exclusion.shape)} logits={tuple(logits.shape)}"
            )
        exclusion_cpu = host_exclusion[
            supervised_pairs[:, 0],
            supervised_pairs[:, 1],
        ]
    selected_flat_indices: list[int] = []
    selected_pair_indices: list[int] = []
    extra_flat_indices: list[int] = []
    extra_pair_indices: list[int] = []
    component_count = logits.shape[1]
    plane_size = height * width
    for pair_idx, (batch_idx, component_idx) in enumerate(
        supervised_pairs.tolist()
    ):
        click_cells: list[tuple[int, int]] = []
        for row, col in (
            centers_cpu[batch_idx, component_idx] > 0
        ).nonzero().tolist():
            multiplicity = max(
                1,
                int(
                    round(
                        float(
                            centers_cpu[
                                batch_idx,
                                component_idx,
                                row,
                                col,
                            ]
                        )
                    )
                ),
            )
            click_cells.extend([(int(row), int(col))] * multiplicity)

        candidates: list[list[int]] = []
        scores = scores_cpu[pair_idx]
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

        selected_cells_ordered: list[int] = []
        selected_cells: set[int] = set()
        for click_idx in range(len(click_cells)):
            # More clicks than resolvable grid cells can occur after coordinate
            # quantization. Reuse is then unavoidable and remains visible in
            # the count target rather than silently discarding the click.
            cell = matched_cell.get(click_idx, candidates[click_idx][0])
            selected_cells.add(cell)
            selected_cells_ordered.append(cell)
        plane_offset = (
            int(batch_idx) * component_count + int(component_idx)
        ) * plane_size
        selected_flat_indices.extend(
            plane_offset + cell for cell in selected_cells_ordered
        )
        selected_pair_indices.extend(
            [pair_idx] * len(selected_cells_ordered)
        )
        if exclusive:
            exclusive_cells = {
                cell for choices in candidates for cell in choices
            }
            if exclusion_cpu is not None:
                exclusive_cells.update(
                    int(row) * width + int(col)
                    for row, col in exclusion_cpu[pair_idx].nonzero().tolist()
                )
            extra_cells = sorted(exclusive_cells - selected_cells)
            extra_flat_indices.extend(
                plane_offset + cell for cell in extra_cells
            )
            extra_pair_indices.extend([pair_idx] * len(extra_cells))

    pair_count = int(supervised_pairs.shape[0])
    flat_logits = logits.flatten()

    def _grouped_mean(
        flat_indices: list[int],
        group_indices: list[int],
        *,
        positive: bool,
    ) -> torch.Tensor:
        if not flat_indices:
            return logits.new_zeros((pair_count,))
        index = torch.tensor(
            flat_indices,
            device=logits.device,
            dtype=torch.long,
        )
        group = torch.tensor(
            group_indices,
            device=logits.device,
            dtype=torch.long,
        )
        values = flat_logits.index_select(0, index)
        losses = F.softplus(-values if positive else values)
        sums = logits.new_zeros((pair_count,)).scatter_add(
            0,
            group,
            losses,
        )
        counts = torch.bincount(
            group,
            minlength=pair_count,
        ).to(dtype=logits.dtype)
        return sums / counts.clamp_min(1)

    pair_loss = _grouped_mean(
        selected_flat_indices,
        selected_pair_indices,
        positive=True,
    )
    if exclusive:
        pair_loss = pair_loss + _grouped_mean(
            extra_flat_indices,
            extra_pair_indices,
            positive=False,
        )
    pair_matrix = logits.new_zeros(
        (logits.shape[0], component_count)
    )
    pair_matrix[pair_batch, pair_component] = pair_loss
    return (
        _mean_supervised_pair(
            pair_matrix,
            supervised,
            logits,
        ),
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
    bag_ids_host: torch.Tensor | None = None,
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
    ids_cpu = (
        bag_ids.detach().to(device="cpu", dtype=torch.long)
        if bag_ids_host is None
        else bag_ids_host.detach().to(device="cpu", dtype=torch.long)
    )
    if ids_cpu.shape != logits.shape:
        raise ValueError(
            "host brush-bag shape mismatch: "
            f"host={tuple(ids_cpu.shape)} logits={tuple(logits.shape)}"
        )
    component_count = logits.shape[1]
    plane_size = logits.shape[2] * logits.shape[3]
    flat_indices: list[torch.Tensor] = []
    bag_sizes: list[int] = []
    keep_counts: list[int] = []
    bag_pairs: list[int] = []
    supervised_pairs: set[int] = set()
    for batch_idx in range(logits.shape[0]):
        for component_idx in range(component_count):
            pair_index = batch_idx * component_count + component_idx
            pair_ids = ids_cpu[batch_idx, component_idx].flatten()
            for bag_id in torch.unique(pair_ids[pair_ids > 0]).tolist():
                local = pair_ids.eq(int(bag_id)).nonzero().flatten()
                size = int(local.numel())
                flat_indices.append(local + pair_index * plane_size)
                bag_sizes.append(size)
                keep_counts.append(
                    max(1, int(math.ceil(size * top_fraction)))
                )
                bag_pairs.append(pair_index)
                supervised_pairs.add(pair_index)
    bag_count = len(flat_indices)
    pair_count = len(supervised_pairs)
    if bag_count == 0:
        return logits.sum() * 0.0, 0, 0

    device = logits.device
    indices = torch.cat(flat_indices).to(device=device)
    sizes = torch.tensor(bag_sizes, device=device, dtype=torch.long)
    keep = torch.tensor(keep_counts, device=device, dtype=torch.long)
    pairs = torch.tensor(bag_pairs, device=device, dtype=torch.long)
    values = logits.flatten().index_select(0, indices)
    bag_index = torch.repeat_interleave(
        torch.arange(bag_count, device=device),
        sizes,
    )
    # Stable value sort followed by stable bag sort is a lexicographic
    # (bag ascending, score descending) ordering without one topk launch per
    # brush bag.
    score_order = torch.argsort(
        values,
        descending=True,
        stable=True,
    )
    grouped_order = score_order[
        torch.argsort(bag_index[score_order], stable=True)
    ]
    sorted_values = values[grouped_order]
    offsets = torch.cumsum(sizes, dim=0) - sizes
    rank = torch.arange(values.numel(), device=device) - torch.repeat_interleave(
        offsets,
        sizes,
    )
    selected = rank < torch.repeat_interleave(keep, sizes)
    sorted_bag_index = bag_index[grouped_order]
    bag_sums = logits.new_zeros((bag_count,)).scatter_add(
        0,
        sorted_bag_index,
        torch.where(selected, sorted_values, torch.zeros_like(sorted_values)),
    )
    evidence = bag_sums / keep.to(dtype=logits.dtype)
    bag_losses = F.softplus(-evidence)
    flat_pair_loss = logits.new_zeros(
        (logits.shape[0] * component_count,)
    ).scatter_add(0, pairs, bag_losses)
    flat_pair_count = torch.bincount(
        pairs,
        minlength=flat_pair_loss.numel(),
    ).to(dtype=logits.dtype)
    pair_loss = (
        flat_pair_loss / flat_pair_count.clamp_min(1)
    ).view(logits.shape[0], component_count)
    supervised = flat_pair_count.view(
        logits.shape[0],
        component_count,
    ) > 0
    return (
        _mean_supervised_pair(pair_loss, supervised, logits),
        bag_count,
        pair_count,
    )


def spatial_morphometry_loss(
    *,
    instance_logits: torch.Tensor,
    abundance_logits: torch.Tensor,
    point_centers: torch.Tensor,
    brush_bag_ids: torch.Tensor,
    area_positive: torch.Tensor,
    explicit_negative: torch.Tensor,
    implicit_negative: torch.Tensor,
    instance_exclusion_support: torch.Tensor | None = None,
    point_centers_host: torch.Tensor | None = None,
    instance_exclusion_support_host: torch.Tensor | None = None,
    brush_bag_ids_host: torch.Tensor | None = None,
    area_positive_host: torch.Tensor | None = None,
    component_names: list[str] | tuple[str, ...] | None = None,
    point_tolerance_cells: int = 1,
    abundance_point_weight: float = 0.5,
    brush_weight: float = 1.0,
    brush_top_fraction: float = 0.25,
    explicit_negative_weight: float = 1.0,
    implicit_negative_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Class-routed supervision balanced across components per objective."""

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
    if (
        instance_exclusion_support is not None
        and instance_exclusion_support.shape != expected
    ):
        raise ValueError(
            "instance_exclusion_support shape mismatch: "
            f"got={tuple(instance_exclusion_support.shape)} "
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
    density_flags = [spec.supports_density for spec in specs]
    density = torch.tensor(
        density_flags,
        dtype=torch.bool,
        device=instance_logits.device,
    ).view(1, -1, 1, 1)
    area = torch.tensor(
        [spec.supports_area for spec in specs],
        dtype=torch.bool,
        device=instance_logits.device,
    ).view(1, -1, 1, 1)
    structure_flags = [
        spec.mode == STRUCTURE_INSTANCE_AREA for spec in specs
    ]
    structure = torch.tensor(
        structure_flags,
        dtype=torch.bool,
        device=instance_logits.device,
    ).view(1, -1, 1, 1)
    density_host = torch.tensor(
        density_flags,
        dtype=torch.bool,
    ).view(1, -1, 1, 1)
    structure_host = torch.tensor(
        structure_flags,
        dtype=torch.bool,
    ).view(1, -1, 1, 1)
    resolved_point_host = (
        None
        if point_centers_host is None
        else point_centers_host.detach().to(device="cpu")
    )
    resolved_area_host = (
        None
        if area_positive_host is None
        else area_positive_host.detach().to(
            device="cpu",
            dtype=torch.bool,
        )
    )
    exclusion_bool = (
        torch.zeros_like(area_positive, dtype=torch.bool)
        if instance_exclusion_support is None
        else instance_exclusion_support.to(dtype=torch.bool)
    )
    resolved_exclusion_host = (
        None
        if instance_exclusion_support_host is None
        else instance_exclusion_support_host.detach().to(
            device="cpu",
            dtype=torch.bool,
        )
    )

    point_bool = point_centers > 0
    bag_bool = brush_bag_ids > 0
    area_bool = area_positive.to(dtype=torch.bool)
    _raise_if_true(
        point_bool & ~countable,
        "non-countable component received instance centres",
    )
    _raise_if_true(
        bag_bool & ~density,
        "non-density component received a brush density bag",
    )
    _raise_if_true(
        area_bool & ~area,
        "non-area component received occupied-area support",
    )
    _raise_if_true(
        exclusion_bool & ~countable,
        "non-countable component received instance exclusion support",
    )
    _raise_if_true(
        exclusion_bool.flatten(2).any(dim=2)
        & ~point_bool.flatten(2).any(dim=2),
        "instance exclusion support requires an instance centre",
    )

    positive_support = point_bool | bag_bool | area_bool
    explicit_bool = explicit_negative.to(dtype=torch.bool)
    implicit_bool = implicit_negative.to(dtype=torch.bool)
    _raise_if_true(
        positive_support & explicit_bool,
        "positive and explicit-negative spatial targets overlap",
    )
    _raise_if_true(
        positive_support & implicit_bool,
        "positive and implicit-negative spatial targets overlap",
    )
    _raise_if_true(
        explicit_bool & implicit_bool,
        "explicit and implicit negative targets overlap",
    )
    _raise_if_true(
        exclusion_bool & explicit_bool,
        "instance exclusion support and explicit-negative targets overlap",
    )

    instance_point, point_pairs, point_counts = _point_peak_loss(
        instance_logits,
        point_centers,
        tolerance_cells=point_tolerance_cells,
        exclusion_support=(
            exclusion_bool | (area_bool & structure)
        ),
        point_centers_host=resolved_point_host,
        exclusion_support_host=(
            None
            if (
                resolved_area_host is None
                or resolved_exclusion_host is None
            )
            else (
                resolved_exclusion_host
                | (resolved_area_host & structure_host)
            )
        ),
    )
    abundance_point, _, _ = _point_peak_loss(
        abundance_logits,
        point_centers * density.to(dtype=point_centers.dtype),
        tolerance_cells=point_tolerance_cells,
        exclusive=False,
        point_centers_host=(
            None
            if resolved_point_host is None
            else resolved_point_host
            * density_host.to(dtype=resolved_point_host.dtype)
        ),
    )
    brush_bag, brush_bags, brush_pairs = _brush_bag_loss(
        abundance_logits,
        brush_bag_ids,
        top_fraction=brush_top_fraction,
        bag_ids_host=brush_bag_ids_host,
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
