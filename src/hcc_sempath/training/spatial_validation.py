from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from hcc_sempath.modeling.models import (
    _collapse_peak_plateaus,
    _sparse_connected_components_8,
)
from hcc_sempath.spatial_schema import (
    DEFAULT_SPATIAL_COMPONENTS,
    spatial_component_specs,
)


def _maximum_point_matches(
    truth: list[tuple[int, int]],
    predicted: list[tuple[int, int]],
    tolerance: int,
) -> int:
    edges = [
        [
            pred_idx
            for pred_idx, (pred_row, pred_col) in enumerate(predicted)
            if max(abs(row - pred_row), abs(col - pred_col)) <= tolerance
        ]
        for row, col in truth
    ]
    matched_truth = [-1] * len(predicted)

    def augment(truth_idx: int, seen: set[int]) -> bool:
        for pred_idx in edges[truth_idx]:
            if pred_idx in seen:
                continue
            seen.add(pred_idx)
            if (
                matched_truth[pred_idx] < 0
                or augment(matched_truth[pred_idx], seen)
            ):
                matched_truth[pred_idx] = truth_idx
                return True
        return False

    return sum(
        augment(truth_idx, set())
        for truth_idx in range(len(truth))
    )


def _peak_mask(
    probability: torch.Tensor,
    *,
    threshold: float,
    kernel: int,
) -> torch.Tensor:
    pooled = F.max_pool2d(
        probability[:, None],
        kernel_size=kernel,
        stride=1,
        padding=kernel // 2,
    )[:, 0]
    candidates = (probability >= threshold) & (probability >= pooled)
    return _collapse_peak_plateaus(
        candidates[:, None],
        probability[:, None],
    )[:, 0]


def _f1(tp: float, fp: float, fn: float) -> float:
    denominator = 2.0 * tp + fp + fn
    return 0.0 if denominator <= 0 else 2.0 * tp / denominator


def _connected_component_count(
    mask: torch.Tensor,
    *,
    minimum_cells: int,
) -> int:
    return sum(
        len(component) >= minimum_cells
        for component in _sparse_connected_components_8(mask)
    )


def _point_locations(
    centers: torch.Tensor,
) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for row, col in (centers > 0).nonzero().tolist():
        points.extend(
            [(int(row), int(col))]
            * max(1, int(round(float(centers[row, col]))))
        )
    return points


def _binary_roc_auc(scores: list[float], labels: list[int]) -> float | None:
    """Return tie-aware binary ROC AUC without an optional sklearn dependency."""

    positives = sum(labels)
    negatives = len(labels) - positives
    if positives <= 0 or negatives <= 0:
        return None
    order = sorted(range(len(scores)), key=scores.__getitem__)
    rank_sum = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        rank_sum += average_rank * sum(labels[order[index]] for index in range(start, end))
        start = end
    return (
        rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def evaluate_weak_spatial_supervision(
    *,
    instance_probability: torch.Tensor,
    abundance_probability: torch.Tensor,
    point_centers: torch.Tensor,
    brush_bag_ids: torch.Tensor,
    area_positive: torch.Tensor,
    explicit_negative: torch.Tensor,
    implicit_negative: torch.Tensor,
    component_names: Sequence[str] = DEFAULT_SPATIAL_COMPONENTS,
    threshold: float = 0.5,
    point_tolerance_cells: int = 1,
    nms_kernel: int = 3,
    brush_top_fraction: float = 1.0,
) -> tuple[dict, dict]:
    """Evaluate incomplete checkpoint-selection marks without inventing misses.

    Positive points, bags, and occupied-area marks are not exhaustive masks, so
    unmarked predictions are never counted as false positives. False-positive
    rates use annotator-confirmed explicit-negative support only. Tile-component
    ROC AUC is reported only when both marked-positive and explicit-negative
    pairs exist for a component.
    """

    expected = instance_probability.shape
    for label, tensor in (
        ("abundance_probability", abundance_probability),
        ("point_centers", point_centers),
        ("brush_bag_ids", brush_bag_ids),
        ("area_positive", area_positive),
        ("explicit_negative", explicit_negative),
        ("implicit_negative", implicit_negative),
    ):
        if tensor.shape != expected:
            raise ValueError(
                f"{label} shape mismatch: got={tuple(tensor.shape)} "
                f"expected={tuple(expected)}"
            )
    names = [str(value) for value in component_names]
    if expected[1] != len(names):
        raise ValueError("weak spatial evaluation component count/order mismatch")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("weak spatial evaluation threshold must be in [0, 1]")
    if point_tolerance_cells < 0:
        raise ValueError("point tolerance must be non-negative")
    if nms_kernel <= 0 or nms_kernel % 2 == 0:
        raise ValueError("NMS kernel must be a positive odd integer")
    if not 0.0 < brush_top_fraction <= 1.0:
        raise ValueError("brush_top_fraction must be in (0, 1]")

    point_bool = point_centers > 0
    bag_bool = brush_bag_ids > 0
    area_bool = area_positive.to(dtype=torch.bool)
    explicit_bool = explicit_negative.to(dtype=torch.bool)
    implicit_bool = implicit_negative.to(dtype=torch.bool)
    instance_peaks = _peak_mask(
        instance_probability.flatten(0, 1),
        threshold=threshold,
        kernel=nms_kernel,
    ).reshape(expected)
    components: dict[str, dict[str, float | int | None]] = {}
    for component_idx, name in enumerate(names):
        point_total = point_hits = 0
        point_confidence: list[float] = []
        for tile_idx in range(expected[0]):
            truth = _point_locations(point_centers[tile_idx, component_idx])
            if not truth:
                continue
            predicted = [
                (int(row), int(col))
                for row, col in instance_peaks[tile_idx, component_idx].nonzero().tolist()
            ]
            point_total += len(truth)
            point_hits += _maximum_point_matches(
                truth,
                predicted,
                point_tolerance_cells,
            )
            plane = instance_probability[tile_idx, component_idx]
            for row, col in truth:
                r0 = max(0, row - point_tolerance_cells)
                r1 = min(plane.shape[0], row + point_tolerance_cells + 1)
                c0 = max(0, col - point_tolerance_cells)
                c1 = min(plane.shape[1], col + point_tolerance_cells + 1)
                point_confidence.append(float(plane[r0:r1, c0:c1].max()))

        bag_total = bag_hits = 0
        for tile_idx in range(expected[0]):
            ids = torch.unique(brush_bag_ids[tile_idx, component_idx])
            for bag_id in ids[ids > 0].tolist():
                values = abundance_probability[tile_idx, component_idx][
                    brush_bag_ids[tile_idx, component_idx] == int(bag_id)
                ]
                keep = max(1, int(math.ceil(values.numel() * brush_top_fraction)))
                score = float(torch.topk(values, keep, sorted=False).values.mean())
                bag_total += 1
                bag_hits += int(score >= threshold)

        positive_area = area_bool[:, component_idx]
        area_total = int(positive_area.sum())
        area_hits = int(
            (
                (abundance_probability[:, component_idx] >= threshold)
                & positive_area
            ).sum()
        )
        negative = explicit_bool[:, component_idx]
        negative_total = int(negative.sum())
        abundance_negative_fp = int(
            (
                (abundance_probability[:, component_idx] >= threshold)
                & negative
            ).sum()
        )
        instance_negative_fp = int(
            (
                (instance_probability[:, component_idx] >= threshold)
                & negative
            ).sum()
        )

        nonassigned = implicit_bool[:, component_idx]
        nonassigned_total = int(nonassigned.sum())
        abundance_nonassigned = abundance_probability[:, component_idx][
            nonassigned
        ]
        instance_nonassigned = instance_probability[:, component_idx][
            nonassigned
        ]

        scores: list[float] = []
        labels: list[int] = []
        for tile_idx in range(expected[0]):
            has_point = bool(point_bool[tile_idx, component_idx].any())
            has_measurement = bool(
                bag_bool[tile_idx, component_idx].any()
                or area_bool[tile_idx, component_idx].any()
            )
            if has_point or has_measurement:
                score = 0.0
                if has_point:
                    score = max(
                        score,
                        float(instance_probability[tile_idx, component_idx].max()),
                    )
                if has_measurement:
                    score = max(
                        score,
                        float(abundance_probability[tile_idx, component_idx].max()),
                    )
                scores.append(score)
                labels.append(1)
            elif bool(negative[tile_idx].any()):
                scores.append(float(torch.maximum(
                    instance_probability[tile_idx, component_idx][negative[tile_idx]].max(),
                    abundance_probability[tile_idx, component_idx][negative[tile_idx]].max(),
                )))
                labels.append(0)

        predicted_labels = [int(score >= threshold) for score in scores]
        tile_tp = sum(
            prediction == 1 and label == 1
            for prediction, label in zip(predicted_labels, labels)
        )
        tile_fp = sum(
            prediction == 1 and label == 0
            for prediction, label in zip(predicted_labels, labels)
        )
        tile_fn = sum(
            prediction == 0 and label == 1
            for prediction, label in zip(predicted_labels, labels)
        )
        tile_precision = (
            tile_tp / (tile_tp + tile_fp)
            if tile_tp + tile_fp > 0
            else None
        )
        tile_recall = (
            tile_tp / (tile_tp + tile_fn)
            if tile_tp + tile_fn > 0
            else None
        )
        components[name] = {
            "point_count": point_total,
            "point_hit_rate": point_hits / point_total if point_total else None,
            "point_mean_local_confidence": (
                sum(point_confidence) / len(point_confidence)
                if point_confidence
                else None
            ),
            "brush_bag_count": bag_total,
            "brush_bag_recall": bag_hits / bag_total if bag_total else None,
            "positive_area_cells": area_total,
            "positive_area_recall": area_hits / area_total if area_total else None,
            "explicit_negative_cells": negative_total,
            "abundance_explicit_negative_fpr": (
                abundance_negative_fp / negative_total if negative_total else None
            ),
            "instance_explicit_negative_fpr": (
                instance_negative_fp / negative_total if negative_total else None
            ),
            "nonassigned_cells": nonassigned_total,
            "abundance_nonassigned_mean_response": (
                float(abundance_nonassigned.mean())
                if nonassigned_total
                else None
            ),
            "instance_nonassigned_mean_response": (
                float(instance_nonassigned.mean())
                if nonassigned_total
                else None
            ),
            "abundance_nonassigned_high_response_rate": (
                float((abundance_nonassigned >= threshold).float().mean())
                if nonassigned_total
                else None
            ),
            "instance_nonassigned_high_response_rate": (
                float((instance_nonassigned >= threshold).float().mean())
                if nonassigned_total
                else None
            ),
            "tile_component_positive_pairs": sum(labels),
            "tile_component_explicit_negative_pairs": len(labels) - sum(labels),
            "tile_component_precision": tile_precision,
            "tile_component_recall": tile_recall,
            "tile_component_f1": (
                _f1(tile_tp, tile_fp, tile_fn)
                if tile_tp + tile_fn > 0
                else None
            ),
            "tile_component_roc_auc": _binary_roc_auc(scores, labels),
        }

    macro_keys = (
        "point_hit_rate",
        "point_mean_local_confidence",
        "brush_bag_recall",
        "positive_area_recall",
        "abundance_explicit_negative_fpr",
        "instance_explicit_negative_fpr",
        "abundance_nonassigned_mean_response",
        "instance_nonassigned_mean_response",
        "abundance_nonassigned_high_response_rate",
        "instance_nonassigned_high_response_rate",
        "tile_component_precision",
        "tile_component_recall",
        "tile_component_f1",
        "tile_component_roc_auc",
    )
    macro = {}
    for key in macro_keys:
        values = [
            float(item[key])
            for item in components.values()
            if item[key] is not None
        ]
        macro[key] = sum(values) / len(values) if values else None
    readout = {
        "version": 1,
        "role": "checkpoint_selection_supervision_metric_readout",
        "spatial_component_names": names,
        "instance_threshold": float(threshold),
        "abundance_threshold": float(threshold),
        "nms_kernel": int(nms_kernel),
        "point_tolerance_cells": int(point_tolerance_cells),
    }
    report = {
        "protocol": {
            "split": "checkpoint_selection_supervision",
            "tile_count": int(expected[0]),
            "labels_exhaustive": False,
            "unmarked_predictions_counted_as_false_positive": False,
            "threshold": float(threshold),
            "point_tolerance_cells": int(point_tolerance_cells),
            "nms_kernel": int(nms_kernel),
        },
        "macro": macro,
        "components": components,
    }
    return readout, report


def _macro_summary(
    pair_stats: dict[int, dict[str, float]],
    *,
    slide_ids: Sequence[str],
    geometry_modes: Sequence[Sequence[Sequence[str]]],
    component_idx: int,
    include_count_error: bool,
) -> dict:
    by_slide: dict[str, dict[str, float]] = {}
    by_geometry: dict[str, dict[str, float]] = {}
    for batch_idx, stats in pair_stats.items():
        slide = str(slide_ids[batch_idx])
        slide_stats = by_slide.setdefault(
            slide,
            {"tp": 0.0, "fp": 0.0, "fn": 0.0, "pairs": 0.0, "count_error": 0.0},
        )
        for key in ("tp", "fp", "fn", "count_error"):
            slide_stats[key] += float(stats.get(key, 0.0))
        slide_stats["pairs"] += 1.0
        modes = geometry_modes[batch_idx][component_idx]
        if not modes:
            modes = ("negative",)
        elif len(modes) > 1:
            # Targets have already been combined at tile/component level, so
            # copying one aggregate score into every source geometry would be
            # false stratification. Mixed pairs remain their own stratum.
            modes = ("mixed",)
        for mode in modes:
            geometry_stats = by_geometry.setdefault(
                str(mode),
                {
                    "tp": 0.0,
                    "fp": 0.0,
                    "fn": 0.0,
                    "pairs": 0.0,
                    "count_error": 0.0,
                    "slides": set(),
                },
            )
            for key in ("tp", "fp", "fn", "count_error"):
                geometry_stats[key] += float(stats.get(key, 0.0))
            geometry_stats["pairs"] += 1.0
            geometry_stats["slides"].add(slide)

    positive_slide_f1 = [
        _f1(value["tp"], value["fp"], value["fn"])
        for value in by_slide.values()
        if value["tp"] + value["fn"] > 0
    ]
    negative_slide_fp = [
        value["fp"]
        for value in by_slide.values()
        if value["tp"] + value["fn"] <= 0
    ]
    result: dict[str, object] = {
        "independent_slide_count": len(by_slide),
        "slide_macro_f1": (
            sum(positive_slide_f1) / len(positive_slide_f1)
            if positive_slide_f1
            else None
        ),
        "positive_slide_count": len(positive_slide_f1),
        "negative_slide_count": len(negative_slide_fp),
        "negative_slide_mean_weighted_fp": (
            sum(negative_slide_fp) / len(negative_slide_fp)
            if negative_slide_fp
            else 0.0
        ),
        "geometry_strata": {},
    }
    if include_count_error:
        slide_count_mae = [
            value["count_error"] / max(1.0, value["pairs"])
            for value in by_slide.values()
        ]
        result["slide_macro_count_mae"] = (
            sum(slide_count_mae) / len(slide_count_mae)
            if slide_count_mae
            else 0.0
        )
    strata = result["geometry_strata"]
    assert isinstance(strata, dict)
    for mode, value in sorted(by_geometry.items()):
        payload = {
            "pair_count": int(value["pairs"]),
            "independent_slide_count": len(value["slides"]),
            "f1": (
                _f1(value["tp"], value["fp"], value["fn"])
                if value["tp"] + value["fn"] > 0
                else None
            ),
            "weighted_false_positive": value["fp"],
        }
        if include_count_error:
            payload["count_mae"] = value["count_error"] / max(
                1.0,
                value["pairs"],
            )
        strata[mode] = payload
    return result


def calibrate_spatial_decoder(
    *,
    instance_probability: torch.Tensor,
    abundance_probability: torch.Tensor,
    point_centers: torch.Tensor,
    brush_bag_ids: torch.Tensor,
    area_positive: torch.Tensor,
    explicit_negative: torch.Tensor,
    implicit_negative: torch.Tensor,
    count_complete: torch.Tensor,
    measurement_complete: torch.Tensor,
    geometry_modes: Sequence[Sequence[Sequence[str]]],
    slide_ids: Sequence[str],
    calibration_provenance: dict,
    component_names: Sequence[str] = DEFAULT_SPATIAL_COMPONENTS,
    output_stride: int,
    point_tolerance_cells: int,
    threshold_grid: Sequence[float] | None = None,
    nms_kernels: Sequence[int] = (3, 5, 7),
    implicit_negative_weight: float = 0.05,
    brush_top_fraction: float = 1.0,
    focus_minimum_grid: Sequence[int] = tuple(range(1, 10)),
) -> tuple[dict, dict]:
    """Freeze decoder values from explicitly complete independent labels.

    Training marks remain weak supervision. Calibration only consumes
    tile/component pairs whose independent annotation explicitly declares the
    count or measurement endpoint complete.
    """

    names = [str(name) for name in component_names]
    specs = spatial_component_specs(names)
    expected = instance_probability.shape
    if abundance_probability.shape != expected:
        raise ValueError("instance and abundance validation maps differ")
    for label, tensor in (
        ("point_centers", point_centers),
        ("brush_bag_ids", brush_bag_ids),
        ("area_positive", area_positive),
        ("explicit_negative", explicit_negative),
        ("implicit_negative", implicit_negative),
    ):
        if tensor.shape != expected:
            raise ValueError(
                f"{label} validation shape mismatch: "
                f"got={tuple(tensor.shape)} expected={tuple(expected)}"
            )
    if expected[1] != len(names):
        raise ValueError("validation component count/order mismatch")
    pair_shape = expected[:2]
    if count_complete.shape != pair_shape:
        raise ValueError("count_complete must have shape [tile, component]")
    if measurement_complete.shape != pair_shape:
        raise ValueError(
            "measurement_complete must have shape [tile, component]"
        )
    if len(slide_ids) != expected[0] or any(
        not str(value) for value in slide_ids
    ):
        raise ValueError("slide_ids must identify every validation tile")
    if len(geometry_modes) != expected[0] or any(
        len(row) != len(names) for row in geometry_modes
    ):
        raise ValueError(
            "geometry_modes must have shape [tile, component]"
        )
    if not isinstance(calibration_provenance, dict):
        raise ValueError("calibration_provenance is required")
    if output_stride <= 0 or point_tolerance_cells < 0:
        raise ValueError("output_stride must be positive and tolerance non-negative")
    if not math.isfinite(implicit_negative_weight) or not (
        0.0 <= implicit_negative_weight <= 1.0
    ):
        raise ValueError("implicit_negative_weight must be in [0, 1]")
    if not 0.0 < brush_top_fraction <= 1.0:
        raise ValueError("brush_top_fraction must be in (0, 1]")
    thresholds = [
        float(value)
        for value in (
            threshold_grid
            if threshold_grid is not None
            else torch.linspace(0.1, 0.9, 17).tolist()
        )
    ]
    if not thresholds or any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in thresholds
    ):
        raise ValueError("threshold grid must contain finite values in [0, 1]")
    kernels = [int(value) for value in nms_kernels]
    if not kernels or any(value <= 0 or value % 2 == 0 for value in kernels):
        raise ValueError("NMS kernels must be positive odd integers")

    instance_thresholds: list[float] = []
    abundance_thresholds: list[float] = []
    selected_kernels: list[int] = []
    component_report: dict[str, dict[str, float | int | str]] = {}

    point_bool = point_centers > 0
    bag_bool = brush_bag_ids > 0
    area_bool = area_positive.to(dtype=torch.bool)
    explicit_bool = explicit_negative.to(dtype=torch.bool)
    implicit_bool = implicit_negative.to(dtype=torch.bool)
    count_complete = count_complete.to(dtype=torch.bool)
    measurement_complete = measurement_complete.to(dtype=torch.bool)

    for component_idx, spec in enumerate(specs):
        report: dict[str, float | int | str] = {"mode": spec.mode}
        if spec.supports_instance_count:
            eligible = count_complete[:, component_idx]
            if bool(
                (
                    eligible
                    & bag_bool[:, component_idx].flatten(1).any(dim=1)
                ).any()
            ):
                raise ValueError(
                    "count-complete cell pairs cannot contain dense brush "
                    f"bags: {spec.name}"
                )
            if not bool(eligible.any()):
                raise ValueError(
                    "independent spatial validation has no count-complete "
                    f"pairs for {spec.name}"
                )
            truth_counts = point_centers[
                eligible,
                component_idx,
            ].flatten(1).sum(dim=1)
            if not bool((truth_counts > 0).any()) or not bool(
                (truth_counts == 0).any()
            ):
                raise ValueError(
                    "count calibration requires positive and complete-negative "
                    f"pairs for {spec.name}"
                )
            best_instance: tuple[float, float, float, int] | None = None
            best_instance_stats: tuple[int, int, int] | None = None
            best_instance_pairs: dict[int, dict[str, float]] | None = None
            probability = instance_probability[eligible, component_idx]
            truth_tensor = point_centers[eligible, component_idx]
            eligible_indices = eligible.nonzero().flatten().tolist()
            for kernel in kernels:
                for threshold in thresholds:
                    predicted_mask = _peak_mask(
                        probability,
                        threshold=threshold,
                        kernel=kernel,
                    )
                    tp = fp = fn = 0
                    count_error = 0.0
                    pair_stats: dict[int, dict[str, float]] = {}
                    for pair_idx in range(predicted_mask.shape[0]):
                        truth = _point_locations(truth_tensor[pair_idx])
                        predicted = [
                            (int(row), int(col))
                            for row, col in predicted_mask[
                                pair_idx
                            ].nonzero().tolist()
                        ]
                        matched = _maximum_point_matches(
                            truth,
                            predicted,
                            point_tolerance_cells,
                        )
                        tp += matched
                        fp += len(predicted) - matched
                        fn += len(truth) - matched
                        pair_error = abs(len(predicted) - len(truth))
                        count_error += pair_error
                        pair_stats[eligible_indices[pair_idx]] = {
                            "tp": float(matched),
                            "fp": float(len(predicted) - matched),
                            "fn": float(len(truth) - matched),
                            "count_error": float(pair_error),
                        }
                    score = _f1(tp, fp, fn)
                    count_mae = count_error / max(
                        1,
                        int(eligible.sum()),
                    )
                    candidate = (
                        score,
                        -count_mae,
                        threshold,
                        -kernel,
                    )
                    if best_instance is None or candidate > best_instance:
                        best_instance = candidate
                        best_instance_stats = (tp, fp, fn)
                        best_instance_pairs = pair_stats
            assert best_instance is not None
            assert best_instance_stats is not None
            assert best_instance_pairs is not None
            instance_thresholds.append(float(best_instance[2]))
            selected_kernels.append(int(-best_instance[3]))
            report.update(
                {
                    "instance_pairs": int(eligible.sum()),
                    "instance_f1": float(best_instance[0]),
                    "instance_tp": best_instance_stats[0],
                    "instance_fp": best_instance_stats[1],
                    "instance_fn": best_instance_stats[2],
                    "instance_count_mae": float(-best_instance[1]),
                    **_macro_summary(
                        best_instance_pairs,
                        slide_ids=slide_ids,
                        geometry_modes=geometry_modes,
                        component_idx=component_idx,
                        include_count_error=True,
                    ),
                }
            )
        else:
            instance_thresholds.append(1.0)
            selected_kernels.append(1)
            report["instance_pairs"] = 0

        density_point = (
            point_bool[:, component_idx]
            if spec.supports_density
            else torch.zeros_like(point_bool[:, component_idx])
        )
        positive_area = area_bool[:, component_idx]
        positive_bag = bag_bool[:, component_idx]
        valid_pair = measurement_complete[:, component_idx]
        if bool(
            (
                valid_pair
                & point_bool[:, component_idx].flatten(1).any(dim=1)
                & ~density_point.flatten(1).any(dim=1)
                & ~positive_area.flatten(1).any(dim=1)
            ).any()
        ):
            raise ValueError(
                "measurement-complete structure point has no annotated "
                f"extent: {spec.name}"
            )
        positive_pair = (
            density_point
            | positive_bag
            | positive_area
        ).flatten(1).any(dim=1)
        if not bool((valid_pair & positive_pair).any()):
            raise ValueError(
                "independent spatial validation has no complete "
                "measurement-positive "
                f"pairs for {spec.name}"
            )
        if not bool((valid_pair & ~positive_pair).any()):
            raise ValueError(
                "measurement calibration requires complete-negative pairs "
                f"for {spec.name}"
            )
        probability = abundance_probability[:, component_idx]
        best_abundance: tuple[float, float] | None = None
        best_stats: tuple[float, float, float] | None = None
        best_measurement_pairs: dict[int, dict[str, float]] | None = None
        for threshold in thresholds:
            predicted = probability >= threshold
            tp = fp = fn = 0.0
            pair_stats: dict[int, dict[str, float]] = {}
            for batch_idx in valid_pair.nonzero().flatten().tolist():
                pair_tp = pair_fp = pair_fn = 0.0
                point_truth = _point_locations(
                    point_centers[batch_idx, component_idx]
                    if spec.supports_density
                    else torch.zeros_like(
                        point_centers[batch_idx, component_idx]
                    )
                )
                predicted_cells = [
                    (int(row), int(col))
                    for row, col in predicted[batch_idx].nonzero().tolist()
                ]
                if point_truth:
                    matched = _maximum_point_matches(
                        point_truth,
                        predicted_cells,
                        point_tolerance_cells,
                    )
                    pair_tp += float(matched)
                    pair_fn += float(len(point_truth) - matched)
                component_bags = torch.unique(
                    brush_bag_ids[batch_idx, component_idx]
                )
                for bag_id in component_bags[
                    component_bags > 0
                ].tolist():
                    values = probability[batch_idx][
                        brush_bag_ids[batch_idx, component_idx]
                        == int(bag_id)
                    ]
                    keep = max(
                        1,
                        int(math.ceil(
                            values.numel() * brush_top_fraction
                        )),
                    )
                    detected = (
                        float(
                            torch.topk(
                                values,
                                keep,
                                sorted=False,
                            ).values.mean()
                        )
                        >= threshold
                    )
                    pair_tp += float(detected)
                    pair_fn += float(not detected)
                area = positive_area[batch_idx]
                pair_tp += float((predicted[batch_idx] & area).sum())
                pair_fn += float((~predicted[batch_idx] & area).sum())
                pair_explicit = explicit_bool[
                    batch_idx,
                    component_idx,
                ]
                pair_implicit = implicit_bool[
                    batch_idx,
                    component_idx,
                ]
                if (
                    not bool(pair_explicit.any())
                    and not bool(pair_implicit.any())
                    and not bool(positive_pair[batch_idx])
                ):
                    # A validation-only measurement-complete declaration with
                    # no positive geometry is an exhaustive negative endpoint.
                    pair_explicit = torch.ones_like(pair_explicit)
                negative = (
                    pair_explicit
                    | pair_implicit
                ) & ~(
                    density_point[batch_idx]
                    | positive_bag[batch_idx]
                    | area
                )
                pair_fp += float(
                    (
                        predicted[batch_idx]
                        & pair_explicit
                        & negative
                    ).sum()
                )
                pair_fp += implicit_negative_weight * float(
                    (
                        predicted[batch_idx]
                        & pair_implicit
                        & negative
                    ).sum()
                )
                tp += pair_tp
                fp += pair_fp
                fn += pair_fn
                pair_stats[batch_idx] = {
                    "tp": pair_tp,
                    "fp": pair_fp,
                    "fn": pair_fn,
                }
            score = _f1(tp, fp, fn)
            candidate = (score, threshold)
            if best_abundance is None or candidate > best_abundance:
                best_abundance = candidate
                best_stats = (tp, fp, fn)
                best_measurement_pairs = pair_stats
        assert best_abundance is not None and best_stats is not None
        assert best_measurement_pairs is not None
        abundance_thresholds.append(float(best_abundance[1]))
        report.update(
            {
                "measurement_pairs": int(valid_pair.sum()),
                "measurement_weighted_f1": float(best_abundance[0]),
                "measurement_tp": best_stats[0],
                "measurement_weighted_fp": best_stats[1],
                "measurement_fn": best_stats[2],
                "measurement_validation": _macro_summary(
                    best_measurement_pairs,
                    slide_ids=slide_ids,
                    geometry_modes=geometry_modes,
                    component_idx=component_idx,
                    include_count_error=False,
                ),
            }
        )
        component_report[spec.name] = report

    focus_indices = [
        index
        for index, spec in enumerate(specs)
        if spec.supports_focus_density
    ]
    if len(focus_indices) != 1:
        raise ValueError("exactly one focus-density component is required")
    focus_idx = focus_indices[0]
    focus_pairs = measurement_complete[:, focus_idx]
    if not bool(focus_pairs.any()):
        raise ValueError(
            "independent spatial validation has no bile-focus pairs"
        )
    focus_threshold = abundance_thresholds[focus_idx]
    best_focus: tuple[float, int] | None = None
    best_focus_errors: dict[int, float] | None = None
    for minimum_cells in [int(value) for value in focus_minimum_grid]:
        if minimum_cells <= 0:
            raise ValueError("focus minimum grid must be positive")
        errors = []
        pair_errors: dict[int, float] = {}
        for batch_idx in focus_pairs.nonzero().flatten().tolist():
            truth_count = _connected_component_count(
                area_bool[batch_idx, focus_idx],
                minimum_cells=1,
            )
            predicted_count = _connected_component_count(
                abundance_probability[batch_idx, focus_idx]
                >= focus_threshold,
                minimum_cells=minimum_cells,
            )
            errors.append(abs(predicted_count - truth_count))
            pair_errors[batch_idx] = float(
                abs(predicted_count - truth_count)
            )
        mae = sum(errors) / len(errors)
        candidate = (-mae, -minimum_cells)
        if best_focus is None or candidate > best_focus:
            best_focus = candidate
            best_focus_errors = pair_errors
    assert best_focus is not None
    assert best_focus_errors is not None
    minimum_focus_cells = -best_focus[1]
    component_report[names[focus_idx]].update(
        {
            "focus_pairs": int(focus_pairs.sum()),
            "focus_count_mae": float(-best_focus[0]),
            "focus_independent_slide_count": len(
                {
                    str(slide_ids[index])
                    for index in best_focus_errors
                }
            ),
        }
    )

    calibration = {
        "version": 1,
        "spatial_component_names": names,
        "instance_threshold": instance_thresholds,
        "abundance_threshold": abundance_thresholds,
        "nms_kernel": selected_kernels,
        "minimum_focus_cells": int(minimum_focus_cells),
        "spatial_output_stride": int(output_stride),
        "provenance": calibration_provenance,
    }
    report = {
        "protocol": {
            "split": "independent_spatial_validation",
            "tile_count": int(expected[0]),
            "point_tolerance_cells": int(point_tolerance_cells),
            "implicit_negative_weight": float(implicit_negative_weight),
            "brush_top_fraction": float(brush_top_fraction),
            "patient_level_results_emitted": False,
            "independent_slide_count": len(
                {str(value) for value in slide_ids}
            ),
        },
        "components": component_report,
    }
    return calibration, report
