from __future__ import annotations

import copy

import pytest
import torch

from hcc_sempath.spatial_schema import (
    DEFAULT_SPATIAL_COMPONENTS,
    spatial_component_specs,
)
from hcc_sempath.training.spatial_validation import (
    calibrate_spatial_decoder,
    evaluate_weak_spatial_supervision,
)


def _provenance() -> dict:
    return {
        "checkpoint_model_sha256": "0" * 64,
        "research_contract_sha256": "1" * 64,
        "validation_annotation_sha256": "2" * 64,
        "validation_protocol_sha256": "3" * 64,
        "validation_cohort_sha256": "4" * 64,
        "optimizer_visible_contract_sha256": "5" * 64,
        "supervision_assets_sha256": "6" * 64,
        "formal_asset_contract_sha256": "7" * 64,
        "source_tree_sha256": "8" * 64,
        "study_contract_sha256": "9" * 64,
        "selected_epoch": 1,
        "terminal_epoch": 1,
        "expected_epochs": 1,
        "selection_finalized": True,
    }


def _complete_validation_case() -> dict:
    component_count = len(DEFAULT_SPATIAL_COMPONENTS)
    specs = spatial_component_specs(DEFAULT_SPATIAL_COMPONENTS)
    shape = (2, component_count, 7, 7)
    instance = torch.zeros(shape)
    abundance = torch.zeros(shape)
    points = torch.zeros(shape)
    bags = torch.zeros(shape, dtype=torch.long)
    area = torch.zeros(shape, dtype=torch.bool)
    explicit = torch.zeros(shape, dtype=torch.bool)
    implicit = torch.ones(shape, dtype=torch.bool)
    count_complete = torch.zeros((2, component_count), dtype=torch.bool)
    measurement_complete = torch.ones(
        (2, component_count),
        dtype=torch.bool,
    )
    geometry_modes = [
        [tuple() for _ in range(component_count)]
        for _ in range(2)
    ]
    countable = tuple(
        index
        for index, spec in enumerate(specs)
        if spec.supports_instance_count
    )
    density_capable = tuple(
        index
        for index, spec in enumerate(specs)
        if spec.supports_density
    )
    area_capable = tuple(
        index
        for index, spec in enumerate(specs)
        if spec.supports_area
    )

    for component in countable:
        count_complete[:, component] = True
        points[0, component, 3, 3] = 1
        instance[0, component, 3, 3] = 0.9
        geometry_modes[0][component] = ("point",)
        geometry_modes[1][component] = ("negative",)
    for component in density_capable:
        abundance[0, component, 3, 3] = 0.9
    for component in area_capable:
        area[0, component, 2:5, 2:5] = True
        abundance[0, component, 2:5, 2:5] = 0.9
        geometry_modes[0][component] = (
            ("point", "brush")
            if component in countable
            else ("brush",)
        )
        geometry_modes[1][component] = ("negative",)
    implicit &= ~(points.bool() | area)

    return {
        "instance_probability": instance,
        "abundance_probability": abundance,
        "point_centers": points,
        "brush_bag_ids": bags,
        "area_positive": area,
        "explicit_negative": explicit,
        "implicit_negative": implicit,
        "count_complete": count_complete,
        "measurement_complete": measurement_complete,
        "geometry_modes": geometry_modes,
        "slide_ids": ["slide-positive", "slide-negative"],
        "calibration_provenance": _provenance(),
        "component_names": DEFAULT_SPATIAL_COMPONENTS,
        "output_stride": 7,
        "point_tolerance_cells": 1,
        "threshold_grid": [0.5],
        "nms_kernels": [3],
        "focus_minimum_grid": [1, 2],
    }


def test_weak_spatial_metrics_do_not_treat_unmarked_cells_as_false_positives() -> None:
    component_count = len(DEFAULT_SPATIAL_COMPONENTS)
    shape = (2, component_count, 5, 5)
    instance = torch.zeros(shape)
    abundance = torch.zeros(shape)
    points = torch.zeros(shape)
    bags = torch.zeros(shape, dtype=torch.long)
    area = torch.zeros(shape, dtype=torch.bool)
    explicit = torch.zeros(shape, dtype=torch.bool)
    implicit = torch.ones(shape, dtype=torch.bool)

    points[0, 0, 2, 2] = 1
    instance[0, 0, 2, 2] = 0.9
    # This unmarked response is unknown, not a false positive.
    instance[0, 0, 0, 0] = 0.8
    explicit[1, 0].fill_(True)
    area[0, 1, 1:3, 1:3] = True
    abundance[0, 1, 1:3, 1:3] = 0.9
    explicit[1, 1].fill_(True)
    implicit &= ~(points.bool() | area | explicit)

    _, report = evaluate_weak_spatial_supervision(
        instance_probability=instance,
        abundance_probability=abundance,
        point_centers=points,
        brush_bag_ids=bags,
        area_positive=area,
        explicit_negative=explicit,
        implicit_negative=implicit,
        threshold=0.5,
        point_tolerance_cells=1,
        nms_kernel=3,
    )

    point = report["components"][DEFAULT_SPATIAL_COMPONENTS[0]]
    region = report["components"][DEFAULT_SPATIAL_COMPONENTS[1]]
    assert point["point_hit_rate"] == 1.0
    assert point["instance_explicit_negative_fpr"] == 0.0
    assert point["tile_component_roc_auc"] == 1.0
    assert point["tile_component_f1"] == 1.0
    assert point["instance_nonassigned_high_response_rate"] > 0.0
    assert region["positive_area_recall"] == 1.0
    assert region["abundance_explicit_negative_fpr"] == 0.0


def test_weak_spatial_metrics_report_unknown_cells_without_implicit_masks() -> None:
    component_count = len(DEFAULT_SPATIAL_COMPONENTS)
    shape = (1, component_count, 3, 3)
    instance = torch.zeros(shape)
    abundance = torch.zeros(shape)
    points = torch.zeros(shape)
    bags = torch.zeros(shape, dtype=torch.long)
    area = torch.zeros(shape, dtype=torch.bool)
    explicit = torch.zeros(shape, dtype=torch.bool)
    implicit = torch.zeros(shape, dtype=torch.bool)

    points[0, 0, 1, 1] = 1
    instance[0, 0, 0, 0] = 0.8
    explicit[0, 0, 2, 2] = True

    _, report = evaluate_weak_spatial_supervision(
        instance_probability=instance,
        abundance_probability=abundance,
        point_centers=points,
        brush_bag_ids=bags,
        area_positive=area,
        explicit_negative=explicit,
        implicit_negative=implicit,
        threshold=0.5,
    )

    component = report["components"][DEFAULT_SPATIAL_COMPONENTS[0]]
    assert component["nonassigned_cells"] == 7
    assert component["instance_nonassigned_mean_response"] == pytest.approx(
        0.8 / 7.0
    )
    assert component["instance_nonassigned_high_response_rate"] == pytest.approx(
        1.0 / 7.0
    )
    assert report["protocol"]["nonassigned_region_definition"] == (
        "complement_of_positive_and_explicit_negative_support"
    )


def test_spatial_calibration_freezes_all_component_readouts() -> None:
    calibration, report = calibrate_spatial_decoder(
        **_complete_validation_case()
    )

    assert calibration["spatial_component_names"] == list(
        DEFAULT_SPATIAL_COMPONENTS
    )
    specs = spatial_component_specs(DEFAULT_SPATIAL_COMPONENTS)
    assert calibration["instance_threshold"] == [
        0.5 if spec.supports_instance_count else 1.0
        for spec in specs
    ]
    assert calibration["abundance_threshold"] == [
        0.5
    ] * len(DEFAULT_SPATIAL_COMPONENTS)
    assert calibration["nms_kernel"] == [
        3 if spec.supports_instance_count else 1
        for spec in specs
    ]
    assert calibration["provenance"] == _provenance()
    assert report["protocol"]["tile_count"] == 2
    assert report["protocol"]["independent_slide_count"] == 2
    assert all(
        item["measurement_weighted_f1"] == 1.0
        for item in report["components"].values()
    )
    assert (
        report["components"]["hepatocellular-parenchyma"][
            "geometry_strata"
        ]["point"]["independent_slide_count"]
        == 1
    )
    assert (
        report["components"]["hepatocellular-parenchyma"][
            "slide_macro_f1"
        ]
        == 1.0
    )
    assert (
        report["components"]["hepatocellular-parenchyma"][
            "geometry_strata"
        ]["negative"]["f1"]
        is None
    )
    structure_strata = report["components"][
        "steatosis-vacuolation"
    ]["geometry_strata"]
    assert "mixed" in structure_strata
    assert "point" not in structure_strata
    assert "brush" not in structure_strata


def test_unknown_structure_measurement_pair_cannot_shift_threshold() -> None:
    base = _complete_validation_case()
    structure_index = DEFAULT_SPATIAL_COMPONENTS.index(
        "steatosis-vacuolation"
    )
    base["threshold_grid"] = [0.5, 0.8]
    base["abundance_probability"][0, structure_index, 2:5, 2:5] = 0.7
    calibration, _ = calibrate_spatial_decoder(**base)

    extended = copy.deepcopy(base)
    for key in (
        "instance_probability",
        "abundance_probability",
        "point_centers",
        "brush_bag_ids",
        "area_positive",
        "explicit_negative",
        "implicit_negative",
    ):
        padding = torch.zeros_like(extended[key][:1])
        if key == "implicit_negative":
            padding.fill_(True)
        extended[key] = torch.cat([extended[key], padding], dim=0)
    for key in ("count_complete", "measurement_complete"):
        extended[key] = torch.cat(
            [
                extended[key],
                torch.zeros_like(extended[key][:1]),
            ],
            dim=0,
        )
    extended["point_centers"][2, structure_index, 3, 3] = 1
    extended["abundance_probability"][2, structure_index].fill_(0.7)
    extended["implicit_negative"][2, structure_index, 3, 3] = False
    extended["geometry_modes"].append(
        [
            tuple() if index != structure_index else ("point",)
            for index in range(len(DEFAULT_SPATIAL_COMPONENTS))
        ]
    )
    extended["slide_ids"].append("slide-unknown")
    extended_calibration, _ = calibrate_spatial_decoder(**extended)

    assert calibration["abundance_threshold"][structure_index] == 0.5
    assert extended_calibration["abundance_threshold"][structure_index] == 0.5


def test_completeness_only_measurement_negative_is_strong() -> None:
    case = _complete_validation_case()
    case["implicit_negative"][1, 1].zero_()
    case["explicit_negative"][1, 1].zero_()
    case["abundance_probability"][1, 1].fill_(0.9)

    _, report = calibrate_spatial_decoder(**case)

    assert (
        report["components"]["necrosis"][
            "measurement_weighted_fp"
        ]
        == 49.0
    )


def test_dense_brush_requires_full_contour_support() -> None:
    case = _complete_validation_case()
    for key in (
        "instance_probability",
        "abundance_probability",
        "point_centers",
        "brush_bag_ids",
        "area_positive",
        "explicit_negative",
        "implicit_negative",
    ):
        padding = torch.zeros_like(case[key][:1])
        if key == "implicit_negative":
            padding.fill_(True)
        case[key] = torch.cat([case[key], padding], dim=0)
    case["count_complete"] = torch.cat(
        [
            case["count_complete"],
            torch.zeros(
                (1, len(DEFAULT_SPATIAL_COMPONENTS)),
                dtype=torch.bool,
            ),
        ]
    )
    extra_measurement = torch.zeros(
        (1, len(DEFAULT_SPATIAL_COMPONENTS)),
        dtype=torch.bool,
    )
    extra_measurement[0, 0] = True
    case["measurement_complete"] = torch.cat(
        [case["measurement_complete"], extra_measurement]
    )
    case["brush_bag_ids"][2, 0, 1:5, 1:5] = 1
    case["abundance_probability"][2, 0, 1:2, 1:5] = 0.9
    case["implicit_negative"][2, 0, 1:5, 1:5] = False
    case["geometry_modes"].append(
        [
            ("brush",) if index == 0 else tuple()
            for index in range(len(DEFAULT_SPATIAL_COMPONENTS))
        ]
    )
    case["slide_ids"].append("slide-brush")

    _, report = calibrate_spatial_decoder(**case)

    # Only the first quarter of the painted support is predicted. Under the
    # contour-faithful formal contract, that brush remains a false negative.
    assert (
        report["components"]["hepatocellular-parenchyma"][
            "measurement_tp"
        ]
        == 1.0
    )
    assert (
        report["components"]["hepatocellular-parenchyma"][
            "measurement_fn"
        ]
        == 1.0
    )


def test_instance_count_mae_is_count_error_not_localization_error() -> None:
    case = _complete_validation_case()
    case["instance_probability"][0, 0].zero_()
    case["instance_probability"][0, 0, 0, 0] = 0.9

    _, report = calibrate_spatial_decoder(**case)
    component = report["components"][
        "hepatocellular-parenchyma"
    ]

    assert component["instance_f1"] == 0.0
    assert component["instance_fp"] == 1
    assert component["instance_fn"] == 1
    assert component["instance_count_mae"] == 0.0


def test_bile_focus_minimum_uses_complete_negative_tiles() -> None:
    case = _complete_validation_case()
    case["area_positive"][0, 3].zero_()
    case["area_positive"][0, 3, 3, 3:5] = True
    case["abundance_probability"][0, 3].zero_()
    case["abundance_probability"][0, 3, 3, 3:5] = 0.9
    case["implicit_negative"][0, 3].fill_(True)
    case["implicit_negative"][0, 3, 3, 3:5] = False
    case["abundance_probability"][1, 3, 0, 0] = 0.9

    calibration, report = calibrate_spatial_decoder(**case)

    assert calibration["minimum_focus_cells"] == 2
    assert (
        report["components"]["bile-pigment"][
            "focus_count_mae"
        ]
        == 0.0
    )
