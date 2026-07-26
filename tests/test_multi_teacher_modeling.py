from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from hcc_sempath.modeling.models import (
    HCCSemPathModel,
    SpatialMorphometryHead,
    _depthwise_conv_fused,
    _pointwise_conv_as_linear,
    _sparse_connected_components_8,
    decode_spatial_morphometry,
    load_hcc_sempath_release,
    model_state_sha256,
    validate_spatial_decoder_calibration,
)
from hcc_sempath.spatial_schema import DEFAULT_SPATIAL_COMPONENTS
from hcc_sempath.modeling.prototypes import PrototypeRegistry
from hcc_sempath.training.losses import multi_teacher_distillation_loss
from hcc_sempath.training.engine import _objective_gradient_diagnostics


def _replace_spatial_prototypes(
    head: SpatialMorphometryHead,
    features: torch.Tensor,
    *,
    point_centers: torch.Tensor,
    brush_bag_ids: torch.Tensor,
    area_positive: torch.Tensor,
    explicit_negative: torch.Tensor,
    implicit_negative: torch.Tensor,
) -> None:
    head.replace_prototypes(
        head.prototype_observation_sums(
            features,
            point_centers=point_centers,
            brush_bag_ids=brush_bag_ids,
            area_positive=area_positive,
            explicit_negative=explicit_negative,
            implicit_negative=implicit_negative,
        )
    )


@pytest.mark.parametrize("route", ["pointwise", "depthwise"])
def test_spatial_convolution_routes_preserve_values_and_gradients(route: str) -> None:
    torch.manual_seed(5)
    features = torch.randn(2, 4, 9, 9, requires_grad=True)
    if route == "pointwise":
        reference_layer = torch.nn.Conv2d(4, 7, kernel_size=1)
        routed = _pointwise_conv_as_linear
    else:
        reference_layer = torch.nn.Conv2d(
            4,
            4,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=4,
        )
        routed = _depthwise_conv_fused
    routed_layer = copy.deepcopy(reference_layer)
    routed_features = features.detach().clone().requires_grad_(True)

    reference = reference_layer(features)
    actual = routed(routed_layer, routed_features)
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)

    reference.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(
        routed_features.grad,
        features.grad,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        routed_layer.weight.grad,
        reference_layer.weight.grad,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        routed_layer.bias.grad,
        reference_layer.bias.grad,
        rtol=1e-5,
        atol=1e-6,
    )


def test_hcc_sempath_model_returns_shared_embedding_and_teacher_outputs() -> None:
    model = HCCSemPathModel(
        backbone_name="vit_tiny_patch16_224",
        embedding_dim=11,
        teacher_dims={"teacher_a": 5, "teacher_b": 7},
        pretrained=False,
    )
    outputs = model(torch.randn(2, 3, 224, 224))

    assert outputs["embedding"].shape == (2, 11)
    assert outputs["embedding_norm"].shape == (2, 11)
    torch.testing.assert_close(outputs["embedding_norm"].norm(dim=-1), torch.ones(2), rtol=1e-5, atol=1e-5)
    assert outputs["teacher_outputs"]["teacher_a"].shape == (2, 5)
    assert outputs["teacher_outputs"]["teacher_b"].shape == (2, 7)


def test_spatial_model_exposes_instance_and_abundance_maps_without_changing_encode_contract() -> None:
    model = HCCSemPathModel(
        backbone_name="vit_tiny_patch16_224",
        embedding_dim=11,
        teacher_dims={},
        pretrained=False,
        l1_num_classes=4,
        spatial_num_components=3,
        spatial_dim=13,
    ).eval()
    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        direct = model.encode(images)
        outputs = model(images)

    torch.testing.assert_close(outputs["embedding"], direct)
    assert outputs["l1_logits"].shape == (2, 4)
    assert outputs["l2_instance_logits"].shape == (2, 3, 31, 31)
    assert outputs["l2_abundance_logits"].shape == (2, 3, 31, 31)


def test_spatial_model_only_materializes_selected_supervised_rows() -> None:
    model = HCCSemPathModel(
        backbone_name="vit_tiny_patch16_224",
        embedding_dim=11,
        teacher_dims={"teacher": 5},
        pretrained=False,
        spatial_num_components=3,
        spatial_dim=13,
    ).eval()
    images = torch.randn(3, 3, 224, 224)
    with torch.no_grad():
        outputs = model(
            images,
            spatial_sample_mask=torch.tensor([False, True, False]),
        )

    assert outputs["embedding"].shape == (3, 11)
    assert outputs["teacher_outputs"]["teacher"].shape == (3, 5)
    assert outputs["l2_instance_logits"].shape == (1, 3, 31, 31)
    assert outputs["l2_abundance_logits"].shape == (1, 3, 31, 31)


def test_spatial_detach_ablation_trains_head_without_shared_encoder() -> None:
    model = HCCSemPathModel(
        backbone_name="vit_tiny_patch16_224",
        embedding_dim=11,
        teacher_dims={},
        pretrained=False,
        spatial_num_components=2,
        spatial_dim=13,
    )
    outputs = model(torch.randn(1, 3, 224, 224), spatial_detach_backbone=True)
    (
        outputs["l2_instance_logits"].sum()
        + outputs["l2_abundance_logits"].sum()
    ).backward()

    assert any(parameter.grad is not None for parameter in model.spatial_head.parameters())
    assert all(parameter.grad is None for parameter in model.encoder.backbone.parameters())


def test_spatial_objective_reaches_shared_encoder_by_default() -> None:
    model = HCCSemPathModel(
        backbone_name="vit_tiny_patch16_224",
        embedding_dim=11,
        teacher_dims={},
        pretrained=False,
        spatial_num_components=2,
        spatial_dim=13,
    )
    outputs = model(torch.randn(1, 3, 224, 224))
    (
        outputs["l2_instance_logits"].sum()
        + outputs["l2_abundance_logits"].sum()
    ).backward()

    assert any(
        parameter.grad is not None
        for parameter in model.spatial_head.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in model.encoder.backbone.parameters()
    )


def test_fixed_nine_class_head_suppresses_non_countable_instance_channels() -> None:
    model = HCCSemPathModel(
        backbone_name="vit_tiny_patch16_224",
        embedding_dim=11,
        teacher_dims={},
        pretrained=False,
        spatial_num_components=9,
        spatial_dim=13,
    ).eval()
    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 224, 224))

    assert outputs["l2_instance_valid"].tolist() == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        True,
        True,
    ]
    invalid = ~outputs["l2_instance_valid"]
    assert torch.all(outputs["l2_instance_probabilities"][:, invalid] < 1e-8)


def test_spatial_gradient_diagnostic_uses_real_shared_backbone_tokens() -> None:
    model = HCCSemPathModel(
        backbone_name="vit_tiny_patch16_224",
        embedding_dim=11,
        teacher_dims={},
        pretrained=False,
        spatial_num_components=2,
        spatial_dim=13,
    )
    outputs = model(
        torch.randn(1, 3, 224, 224),
        return_spatial_features=True,
    )
    grid = outputs["l2_spatial_features"].shape[-2:]
    point_centers = torch.zeros((1, 2, *grid))
    point_centers[0, 0, grid[0] // 2, grid[1] // 2] = 1
    implicit_negative = torch.zeros((1, 2, *grid), dtype=torch.bool)
    implicit_negative[0, 0, 0, 0] = True
    _replace_spatial_prototypes(
        model.spatial_head,
        outputs["l2_spatial_features"],
        point_centers=point_centers,
        brush_bag_ids=torch.zeros((1, 2, *grid), dtype=torch.long),
        area_positive=torch.zeros((1, 2, *grid), dtype=torch.bool),
        explicit_negative=torch.zeros((1, 2, *grid), dtype=torch.bool),
        implicit_negative=implicit_negative,
    )
    _, abundance_logits = model.spatial_head.prototype_logits(
        outputs["l2_spatial_features"]
    )
    global_objective = outputs["embedding"].square().mean()
    spatial_objective = abundance_logits.square().mean()
    diagnostics = _objective_gradient_diagnostics(
        global_objective,
        spatial_objective,
        tuple(model.encoder.backbone.blocks[-1].parameters()),
    )

    assert diagnostics["gradient_global_norm"] > 0
    assert diagnostics["gradient_spatial_norm"] > 0
    (global_objective + spatial_objective).backward()
    assert any(parameter.grad is not None for parameter in model.encoder.backbone.parameters())


def test_spatial_prototypes_separate_positive_and_negative_local_patterns() -> None:
    head = SpatialMorphometryHead(
        student_dim=2,
        component_count=1,
        spatial_dim=2,
    )
    features = torch.tensor(
        [[[[1.0, -1.0]], [[0.0, 0.0]]]]
    )
    point = torch.tensor([[[[1.0, 0.0]]]])
    negative = torch.tensor([[[[False, True]]]])
    _replace_spatial_prototypes(
        head,
        features,
        point_centers=point,
        brush_bag_ids=torch.zeros((1, 1, 1, 2), dtype=torch.long),
        area_positive=torch.zeros((1, 1, 1, 2), dtype=torch.bool),
        explicit_negative=negative,
        implicit_negative=torch.zeros_like(negative),
    )

    instance, measurement = head.prototype_logits(features)

    assert instance[0, 0, 0, 0] > instance[0, 0, 0, 1]
    assert measurement[0, 0, 0, 0] > measurement[0, 0, 0, 1]


def test_vectorized_spatial_centroids_match_pair_balanced_reference() -> None:
    torch.manual_seed(17)
    features = torch.randn(3, 4, 3, 3)
    mask = torch.rand(3, 5, 3, 3) > 0.65
    mask[:, 4] = False

    actual_sums, actual_counts = (
        SpatialMorphometryHead._masked_pair_centroid_sums(
            features,
            mask,
        )
    )
    actual = torch.nn.functional.normalize(
        actual_sums / actual_counts.clamp_min(1).unsqueeze(-1),
        dim=1,
    )

    normalized = torch.nn.functional.normalize(features.float(), dim=1)
    reference = torch.zeros(5, 4)
    reference_counts = torch.zeros(5)
    for component_idx in range(5):
        pair_centroids = []
        for batch_idx in range(3):
            selected = mask[batch_idx, component_idx]
            if bool(selected.any()):
                pair_centroids.append(
                    normalized[batch_idx, :, selected].mean(dim=1)
                )
        if pair_centroids:
            reference[component_idx] = torch.nn.functional.normalize(
                torch.stack(pair_centroids).mean(dim=0),
                dim=0,
            )
            reference_counts[component_idx] = len(pair_centroids)

    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(actual_counts, reference_counts)


def test_spatial_prototype_replacement_uses_exact_bank_statistics() -> None:
    torch.manual_seed(23)
    head = SpatialMorphometryHead(
        student_dim=6,
        component_count=4,
        spatial_dim=6,
    )
    sums = torch.randn(4, 6)
    counts = torch.tensor([2.0, 4.0, 0.0, 1.0])
    observations = {
        name: (sums.clone(), counts.clone())
        for name in (
            "instance",
            "measurement",
            "instance_negative",
            "measurement_negative",
            "instance_implicit_negative",
            "measurement_implicit_negative",
        )
    }

    head.replace_prototypes(observations)

    expected = torch.nn.functional.normalize(
        sums / counts.clamp_min(1).unsqueeze(-1),
        dim=1,
    )
    expected[2] = 0
    torch.testing.assert_close(head.instance_prototypes, expected)
    torch.testing.assert_close(head.instance_prototype_counts, counts)


def test_structure_point_does_not_update_unknown_measurement_negative_prototype() -> None:
    head = SpatialMorphometryHead(
        student_dim=2,
        component_count=9,
        spatial_dim=2,
    )
    features = torch.randn(1, 2, 5, 5)
    point = torch.zeros((1, 9, 5, 5))
    point[0, 6, 2, 2] = 1
    implicit = torch.zeros((1, 9, 5, 5), dtype=torch.bool)
    implicit[0, 6] = True
    implicit[0, 6, 1:4, 1:4] = False

    _replace_spatial_prototypes(
        head,
        features,
        point_centers=point,
        brush_bag_ids=torch.zeros((1, 9, 5, 5), dtype=torch.long),
        area_positive=torch.zeros((1, 9, 5, 5), dtype=torch.bool),
        explicit_negative=torch.zeros((1, 9, 5, 5), dtype=torch.bool),
        implicit_negative=implicit,
    )

    assert head.instance_implicit_negative_prototype_counts[6].item() == 1
    assert head.measurement_implicit_negative_prototype_counts[6].item() == 0


def test_hcc_sempath_model_decodes_instances_and_uncalibrated_abundance() -> None:
    model = HCCSemPathModel(
        backbone_name="vit_tiny_patch16_224",
        embedding_dim=11,
        teacher_dims={},
        pretrained=False,
        l1_num_classes=4,
        spatial_num_components=3,
        spatial_dim=12,
    )
    outputs = model(torch.randn(2, 3, 224, 224))

    assert outputs["teacher_outputs"] == {}
    assert outputs["l1_probabilities"].shape == (2, 4)
    decoded = decode_spatial_morphometry(
        outputs,
        instance_threshold=0.0,
        abundance_threshold=0.0,
        output_stride=7,
        nms_kernel=3,
        minimum_focus_cells=1,
    )
    assert decoded["instance_counts"].shape == (2, 3)
    assert decoded["abundance_mass"].shape == (2, 3)
    assert decoded["mean_abundance"].shape == (2, 3)
    assert "dense_area_pixels" not in decoded
    assert len(decoded["instance_coordinates"]) == 2


def test_spatial_decoder_masks_invalid_counts_and_derives_bile_focus_density() -> None:
    instance = torch.zeros((1, 9, 5, 5))
    measurement = torch.zeros((1, 9, 5, 5))
    instance[0, 0, 2, 2] = 0.9
    instance[0, 1, 2, 2] = 0.9
    instance[0, 3, 2, 2] = 0.9
    measurement[0, 3, 1, 1] = 0.9
    measurement[0, 3, 3, 3] = 0.9
    decoded = decode_spatial_morphometry(
        {
            "l2_instance_probabilities": instance,
            "l2_abundance_probabilities": measurement,
        },
        instance_threshold=0.5,
        abundance_threshold=0.5,
        output_stride=7,
        nms_kernel=3,
        minimum_focus_cells=1,
    )

    assert decoded["instance_counts"][0, 0].item() == 1
    assert torch.isnan(decoded["instance_counts"][0, 1])
    assert torch.isnan(decoded["instance_counts"][0, 3])
    assert decoded["instance_count_valid"].tolist() == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        True,
        True,
    ]
    assert decoded["area_valid"].tolist() == [
        False,
        True,
        False,
        True,
        False,
        True,
        True,
        True,
        True,
    ]
    assert decoded["focus_counts"][0, 3].item() == 2
    assert torch.isnan(decoded["focus_counts"][0, 0])


def test_spatial_decoder_accepts_frozen_per_component_thresholds() -> None:
    instance = torch.zeros((1, 2, 5, 5))
    measurement = torch.zeros_like(instance)
    instance[0, :, 2, 2] = 0.6
    measurement[0, 0, 1, 1] = 0.4
    measurement[0, 1, 1, 1] = 0.8

    decoded = decode_spatial_morphometry(
        {
            "l2_instance_probabilities": instance,
            "l2_abundance_probabilities": measurement,
        },
        component_names=["synthetic-a", "synthetic-b"],
        instance_threshold=[0.5, 0.7],
        abundance_threshold=[0.5, 0.7],
        output_stride=7,
        nms_kernel=[3, 5],
        minimum_focus_cells=1,
    )

    assert decoded["instance_counts"].tolist() == [[1.0, 0.0]]
    assert decoded["high_abundance_fraction"][0, 0].item() == 0.0
    assert decoded["high_abundance_fraction"][0, 1].item() > 0.0


def test_spatial_decoder_collapses_a_flat_peak_to_one_instance() -> None:
    instance = torch.zeros((1, 1, 5, 5))
    measurement = torch.zeros_like(instance)
    instance[0, 0, 2:4, 2:4] = 0.9

    decoded = decode_spatial_morphometry(
        {
            "l2_instance_probabilities": instance,
            "l2_abundance_probabilities": measurement,
        },
        component_names=["synthetic"],
        instance_threshold=0.5,
        abundance_threshold=0.5,
        output_stride=7,
        nms_kernel=3,
        minimum_focus_cells=1,
    )

    assert decoded["instance_counts"].tolist() == [[1.0]]
    assert len(decoded["instance_coordinates"][0][0]) == 1
    assert decoded["instance_coordinates"][0][0][0][:2] == (17.5, 17.5)


def test_sparse_components_preserve_diagonal_eight_connectivity() -> None:
    mask = torch.zeros((6, 6), dtype=torch.bool)
    mask[0, 0] = True
    mask[1, 1] = True
    mask[2, 2] = True
    mask[5, 5] = True

    components = _sparse_connected_components_8(mask)

    assert [len(component) for component in components] == [3, 1]
    assert set(components[0]) == {(0, 0), (1, 1), (2, 2)}


def test_spatial_decoder_focus_minimum_uses_eight_connected_extent() -> None:
    instance = torch.zeros((1, 9, 5, 5))
    measurement = torch.zeros_like(instance)
    measurement[0, 3, 1, 1] = 0.9
    measurement[0, 3, 2, 2] = 0.9
    measurement[0, 3, 4, 4] = 0.9

    decoded = decode_spatial_morphometry(
        {
            "l2_instance_probabilities": instance,
            "l2_abundance_probabilities": measurement,
        },
        instance_threshold=0.5,
        abundance_threshold=0.5,
        output_stride=7,
        nms_kernel=3,
        minimum_focus_cells=2,
    )

    assert decoded["focus_counts"][0, 3].item() == 1


def test_spatial_decoder_requires_frozen_analysis_values() -> None:
    outputs = {
        "l2_instance_probabilities": torch.zeros((1, 1, 3, 3)),
        "l2_abundance_probabilities": torch.zeros((1, 1, 3, 3)),
    }

    with pytest.raises(ValueError, match="frozen decoder calibration"):
        decode_spatial_morphometry(
            outputs,
            component_names=["synthetic"],
        )


def _decoder_calibration(
    names: list[str] | tuple[str, ...],
    *,
    stride: int = 7,
) -> dict:
    return {
        "version": 1,
        "spatial_component_names": list(names),
        "instance_threshold": [0.5] * len(names),
        "abundance_threshold": [0.5] * len(names),
        "nms_kernel": [3] * len(names),
        "minimum_focus_cells": 1,
        "spatial_output_stride": stride,
        "provenance": {
            "checkpoint_model_sha256": "0" * 64,
            "research_contract_sha256": "1" * 64,
            "validation_annotation_sha256": "2" * 64,
            "validation_protocol_sha256": "3" * 64,
            "validation_cohort_sha256": "4" * 64,
            "optimizer_visible_contract_sha256": "5" * 64,
            "supervision_assets_sha256": "6" * 64,
            "terminal_epoch": 1,
            "expected_epochs": 1,
        },
    }


def test_spatial_decoder_calibration_contract_is_exact() -> None:
    calibration = _decoder_calibration(DEFAULT_SPATIAL_COMPONENTS)

    assert validate_spatial_decoder_calibration(
        calibration,
        DEFAULT_SPATIAL_COMPONENTS,
        expected_output_stride=7,
    ) == calibration

    changed = {**calibration, "spatial_component_names": ["wrong"]}
    with pytest.raises(ValueError, match="component order"):
        validate_spatial_decoder_calibration(
            changed,
            DEFAULT_SPATIAL_COMPONENTS,
        )
    with pytest.raises(ValueError, match="different checkpoint"):
        validate_spatial_decoder_calibration(
            calibration,
            DEFAULT_SPATIAL_COMPONENTS,
            expected_model_state_sha256="f" * 64,
        )
    with pytest.raises(ValueError, match="optimizer-visible"):
        validate_spatial_decoder_calibration(
            calibration,
            DEFAULT_SPATIAL_COMPONENTS,
            expected_optimizer_visible_contract_sha256="f" * 64,
        )
    with pytest.raises(ValueError, match="supervision-asset"):
        validate_spatial_decoder_calibration(
            calibration,
            DEFAULT_SPATIAL_COMPONENTS,
            expected_supervision_assets_sha256="f" * 64,
        )


def test_model_state_digest_normalizes_compile_prefix_and_scalars() -> None:
    state = {
        "weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "count": torch.tensor(3, dtype=torch.long),
    }
    compiled = {
        f"_orig_mod.{key}": value.clone()
        for key, value in state.items()
    }

    assert model_state_sha256(state) == model_state_sha256(compiled)


def test_release_loader_uses_checkpoint_backbone_configuration(tmp_path: Path) -> None:
    model = HCCSemPathModel(
        backbone_name="vit_tiny_patch16_224",
        embedding_dim=11,
        teacher_dims={},
        pretrained=False,
        l1_num_classes=4,
        spatial_num_components=3,
        spatial_dim=12,
    )
    checkpoint = tmp_path / "release.pt"
    config = tmp_path / "config.json"
    release_state = model.state_dict()
    torch.save(release_state, checkpoint)
    config.write_text(
        json.dumps({
            "format": "hcc-sempath-l1-spatial-state-dict",
            "version": 3,
            "model": {
                "backbone_name": "vit_tiny_patch16_224",
                "embedding_dim": 11,
                "projector_type": "linear",
                "l1_num_classes": 4,
                "spatial_num_components": 3,
                "spatial_dim": 12,
                "spatial_output_stride": 7,
            },
            "spatial_component_names": [
                "synthetic-0",
                "synthetic-1",
                "synthetic-2",
            ],
            "spatial_decoder_calibration": _decoder_calibration(
                ["synthetic-0", "synthetic-1", "synthetic-2"]
            ),
            "training_provenance": {
                "release_model_sha256": model_state_sha256(
                    release_state
                ),
            },
        }),
        encoding="utf-8",
    )

    loaded, _ = load_hcc_sempath_release(config, checkpoint)

    assert loaded.encoder.backbone.patch_embed.patch_size == (16, 16)
    torch.testing.assert_close(
        loaded.encoder.projector[1].weight,
        model.encoder.projector[1].weight,
    )


def test_release_loader_rejects_unsupported_classifier_format(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "release.pt"
    config = tmp_path / "config.json"
    torch.save({}, checkpoint)
    config.write_text(
        json.dumps(
            {
                "format": "hcc-sempath-classifier-state-dict",
                "version": 1,
                "model": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="version 3"):
        load_hcc_sempath_release(config, checkpoint)


def test_multi_teacher_distillation_loss_aggregates_named_heads() -> None:
    student_by_teacher = {
        "teacher_a": torch.randn(4, 5, requires_grad=True),
        "teacher_b": torch.randn(4, 7, requires_grad=True),
    }
    teacher_by_name = {
        "teacher_a": torch.randn(4, 5),
        "teacher_b": torch.randn(4, 7),
    }
    prototypes_by_teacher = {
        "teacher_a": PrototypeRegistry(
            prototypes=torch.randn(4, 5),
            names=["primary_tumor", "primary_non_tumor", "lymphocyte_rich", "fibrotic_stroma"],
            groups=["primary_state", "primary_state", "immune", "stroma"],
            levels=[1, 1, 2, 2],
            exclusive=[True, True, False, False],
        ),
        "teacher_b": PrototypeRegistry(
            prototypes=torch.randn(4, 7),
            names=["primary_tumor", "primary_non_tumor", "lymphocyte_rich", "fibrotic_stroma"],
            groups=["primary_state", "primary_state", "immune", "stroma"],
            levels=[1, 1, 2, 2],
            exclusive=[True, True, False, False],
        ),
    }

    loss, parts = multi_teacher_distillation_loss(
        student_by_teacher=student_by_teacher,
        teacher_by_name=teacher_by_name,
        prototypes_by_teacher=prototypes_by_teacher,
        relation_weight=0.25,
        semantic_weight=0.25,
        semantic_temperature=1.0,
    )

    assert loss.ndim == 0
    assert set(parts) == {
        "feature",
        "relation",
        "semantic",
        "teacher_a_feature_cosine",
        "teacher_b_feature_cosine",
    }
    assert all(
        -1.0 <= float(parts[key]) <= 1.0
        for key in ("teacher_a_feature_cosine", "teacher_b_feature_cosine")
    )
    loss.backward()
    assert student_by_teacher["teacher_a"].grad is not None
    assert student_by_teacher["teacher_b"].grad is not None
