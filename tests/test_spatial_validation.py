from __future__ import annotations

import copy

import torch

from hcc_sempath.spatial_schema import (
    DEFAULT_SPATIAL_COMPONENTS,
    spatial_component_specs,
)
from hcc_sempath.training.spatial_validation import (
    calibrate_spatial_decoder,
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
        "terminal_epoch": 1,
        "expected_epochs": 1,
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


def test_dense_brush_is_one_mil_positive_not_dense_pixel_truth() -> None:
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

    # One point pair plus one brush bag: the 16 brush cells are not promoted
    # to 16 exact positives.
    assert (
        report["components"]["hepatocellular-parenchyma"][
            "measurement_tp"
        ]
        == 2.0
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
