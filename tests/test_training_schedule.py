from __future__ import annotations

import csv
import json

import pytest

from hcc_sempath.training.config import teacher_dims, teacher_names, validate_training_config
from hcc_sempath.training.engine import (
    _amp_enabled,
    _bucket_spatial_sample_mask,
    _classification_eval_metrics,
    _complete_bank_spatial_metrics,
    _eligible_selection_epoch_count,
    _selection_early_stop_requested,
    _normalized_selection_metrics,
    _normalize_uint8_images_fp16,
    _selection_start_step,
    _spatial_bucket_sizes,
    _teacher_retention_metrics,
    _spatial_global_targets_from_spatial,
    _objective_gradient_diagnostics,
    _optimizer_step,
    _scalar_epoch_metrics,
    _should_stop_for_alignment,
    _development_early_stop_state_from_csv,
    _truncate_csv_after_step,
    _update_development_early_stop_state,
    build_lr_scheduler,
    fit,
    run_epoch,
    scheduled_loss_config,
    StepMetricsWriter,
)
from hcc_sempath.training.losses import feature_distillation_loss_per_sample
from hcc_sempath.training.prototype_labels import DEFAULT_CLASSIFICATION_CLASSES
from hcc_sempath.training.spatial_losses import _mean_supervised_pair
from hcc_sempath.spatial_schema import DEFAULT_SPATIAL_COMPONENTS
from hcc_sempath.training.train import (
    _build_optimizer,
    _configure_default_compile_cache,
    _configure_compiled_training_for_gradient_diagnostics,
    _resume_contract,
    _resolve_configured_epochs,
)
from hcc_sempath.modeling.models import HCCSemPathModel
from hcc_sempath.modeling.prototypes import PrototypeRegistry
import torch


def test_classification_validation_metrics_are_class_balanced() -> None:
    logits = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )
    metrics = _classification_eval_metrics(
        {
            "prototype_mask": torch.ones(6, dtype=torch.bool),
            "prototype_classification": torch.tensor(
                [0, 0, 0, 0, 1, 2]
            ),
            "classification_logits": logits,
        }
    )

    assert metrics["classification_accuracy"] == pytest.approx(5 / 6)
    assert metrics["classification_balanced_accuracy"] == pytest.approx(
        2 / 3
    )
    assert metrics["classification_evaluated_classes"] == 3
    assert metrics["classification_total_classes"] == 3
    assert (
        metrics["classification_balanced_cross_entropy"]
        > metrics["classification_cross_entropy"]
    )


def test_joint_selection_metric_uses_fixed_normalized_components() -> None:
    metrics = _normalized_selection_metrics(
        {
            "teacher": 0.4,
            "classification": 0.6,
            "spatial": 0.9,
        },
        {
            "teacher": 0.8,
            "classification": 1.2,
            "spatial": 1.0,
        },
        {
            "teacher": 0.50,
            "classification": 0.25,
            "spatial": 0.25,
        },
    )

    assert metrics["selection_teacher_normalized"] == pytest.approx(0.5)
    assert metrics["selection_classification_normalized"] == pytest.approx(
        0.5
    )
    assert metrics["selection_spatial_normalized"] == pytest.approx(0.9)
    assert metrics["selection_loss"] == pytest.approx(0.6)


def test_configured_selection_baseline_is_positive_and_complete() -> None:
    cfg = {
        "data": {"teachers": ["teacher"]},
        "model": {"teacher_dims": {"teacher": 8}},
        "loss": {},
        "train": {
            "selection_metric_baseline": {
                "teacher": 0.8,
                "classification": 1.2,
                "spatial": 1.0,
            }
        },
    }

    validate_training_config(cfg, ["teacher"])
    cfg["train"]["selection_metric_baseline"]["spatial"] = 0.0
    with pytest.raises(ValueError, match="finite and positive"):
        validate_training_config(cfg, ["teacher"])


def test_teacher_retention_does_not_use_dynamic_population_loss() -> None:
    metrics = _teacher_retention_metrics(
        {
            "teacher_a_feature_cosine": 0.8,
            "teacher_a_relation_mse": 0.2,
            "teacher_b_feature_cosine": 0.6,
            "teacher_b_relation_mse": 0.4,
            "val_feature": 999.0,
        },
        relation_weight=0.05,
    )

    assert metrics["fixed_teacher_distance"] == pytest.approx(0.3)
    assert metrics["fixed_teacher_relation"] == pytest.approx(0.3)
    assert metrics["teacher_validation_loss"] == pytest.approx(0.315)


def test_selection_start_waits_for_every_active_supervision_ramp() -> None:
    assert _selection_start_step(
        {
            "loss": {
                "expert_supervision_start_step": 1000,
                "expert_supervision_ramp_steps": 1000,
                "prototype_filter_weight": 0.5,
                "prototype_filter_start_step": 1500,
                "prototype_filter_ramp_steps": 1000,
                "zhcc_response_weight": 0.15,
                "zhcc_response_start_step": 2000,
                "zhcc_response_ramp_steps": 750,
            }
        }
    ) == 2750


def test_selection_start_cannot_bypass_active_supervision_ramp() -> None:
    cfg = {
        "data": {"teachers": ["teacher"]},
        "model": {"teacher_dims": {"teacher": 8}},
        "loss": {
            "expert_supervision_start_step": 1000,
            "expert_supervision_ramp_steps": 1000,
            "prototype_filter_weight": 0.5,
            "prototype_filter_start_step": 1500,
            "prototype_filter_ramp_steps": 1000,
        },
        "train": {"selection_early_stop_start_step": 1},
    }

    assert _selection_start_step(cfg) == 2500
    with pytest.raises(ValueError, match="cannot precede"):
        validate_training_config(cfg, ["teacher"])


def test_selection_reachability_counts_post_ramp_epoch_ends() -> None:
    assert _eligible_selection_epoch_count(
        current_global_step=0,
        steps_per_epoch=209,
        start_epoch=1,
        expected_epochs=16,
        selection_start_step=2500,
    ) == 5
    assert _eligible_selection_epoch_count(
        current_global_step=0,
        steps_per_epoch=208,
        start_epoch=1,
        expected_epochs=16,
        selection_start_step=2500,
    ) == 4


def test_selection_reachability_handles_mid_epoch_resume() -> None:
    assert _eligible_selection_epoch_count(
        current_global_step=250,
        steps_per_epoch=100,
        start_epoch=3,
        expected_epochs=5,
        selection_start_step=400,
        resume_batch_in_epoch=50,
    ) == 2


def test_selection_early_stop_waits_for_minimum_eligible_epochs() -> None:
    assert not _selection_early_stop_requested(
        enabled=True,
        eligible=True,
        eligible_epochs=4,
        minimum_eligible_epochs=5,
        bad_epochs=4,
        patience=4,
    )
    assert _selection_early_stop_requested(
        enabled=True,
        eligible=True,
        eligible_epochs=5,
        minimum_eligible_epochs=5,
        bad_epochs=4,
        patience=4,
    )


def test_selection_early_stop_requires_eligibility_and_patience() -> None:
    assert not _selection_early_stop_requested(
        enabled=True,
        eligible=False,
        eligible_epochs=8,
        minimum_eligible_epochs=5,
        bad_epochs=8,
        patience=4,
    )
    assert not _selection_early_stop_requested(
        enabled=True,
        eligible=True,
        eligible_epochs=8,
        minimum_eligible_epochs=5,
        bad_epochs=3,
        patience=4,
    )


def test_complete_bank_spatial_reducer_is_batch_partition_invariant() -> None:
    complete = {
        "_spatial_eval_instance_point_sum": torch.tensor([3.0, 4.0]),
        "_spatial_eval_instance_point_count": torch.tensor([2.0, 4.0]),
        "_spatial_eval_measurement_positive_sum": torch.tensor([2.0, 0.0]),
        "_spatial_eval_measurement_positive_count": torch.tensor([4.0, 0.0]),
        "_spatial_eval_explicit_instance_sum": torch.tensor([1.0, 6.0]),
        "_spatial_eval_explicit_instance_count": torch.tensor([2.0, 3.0]),
        "_spatial_eval_explicit_abundance_sum": torch.tensor([2.0, 2.0]),
        "_spatial_eval_explicit_abundance_count": torch.tensor([4.0, 1.0]),
        "_spatial_eval_implicit_sum": torch.tensor([5.0, 1.0]),
        "_spatial_eval_implicit_count": torch.tensor([5.0, 2.0]),
    }
    first = {key: value * 0.25 for key, value in complete.items()}
    second = {
        key: complete[key] - first[key]
        for key in complete
    }
    merged = {
        key: first[key] + second[key]
        for key in complete
    }

    expected = _complete_bank_spatial_metrics(
        complete,
        explicit_negative_weight=1.0,
        implicit_negative_weight=0.05,
    )
    observed = _complete_bank_spatial_metrics(
        merged,
        explicit_negative_weight=1.0,
        implicit_negative_weight=0.05,
    )
    assert observed == pytest.approx(expected)


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        ([False, False, False], [False, False, False]),
        ([False, True, False], [False, True, False]),
        ([False, True, False, False, True], [False, True, False, False, True]),
        (
            [True, False, True, False, True, False, False, False],
            [True, True, True, False, True, False, False, False],
        ),
    ],
)
def test_spatial_compute_mask_uses_power_of_two_buckets(mask, expected) -> None:
    actual = _bucket_spatial_sample_mask(torch.tensor(mask, dtype=torch.bool))

    assert actual.tolist() == expected


@pytest.mark.parametrize(
    ("batch_size", "expected"),
    [
        (1, (1,)),
        (8, (1, 2, 4, 8)),
        (23, (1, 2, 4, 8, 16, 23)),
        (64, (1, 2, 4, 8, 16, 32, 64)),
    ],
)
def test_spatial_warmup_enumerates_every_reachable_bucket(
    batch_size,
    expected,
) -> None:
    assert _spatial_bucket_sizes(batch_size) == expected


def test_cuda_compile_uses_a_shared_default_cache(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    cfg = {
        "runtime": {"output_dir": str(tmp_path / "trial")},
        "train": {"compile": True},
    }

    cache_dir = _configure_default_compile_cache(
        cfg,
        torch.device("cuda"),
    )

    assert cache_dir == tmp_path / "torchinductor-cache"
    assert cfg["runtime"]["compile_cache_dir"] == str(cache_dir)
    assert cache_dir.is_dir()


def test_cuda_amp_precision_is_shared_by_training_and_evaluation() -> None:
    cfg = {"train": {"amp": True}}

    assert _amp_enabled(torch.device("cuda"), cfg)
    assert not _amp_enabled(torch.device("cpu"), cfg)
    assert not _amp_enabled(torch.device("cuda"), {"train": {"amp": False}})


def test_fused_image_normalization_matches_existing_amp_input() -> None:
    images = torch.arange(2 * 3 * 5 * 7, dtype=torch.uint8).reshape(2, 3, 5, 7)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    expected = (
        images.to(torch.float32).div_(255.0).sub_(mean).div_(std)
    ).to(torch.float16)

    actual = _normalize_uint8_images_fp16(images, mean, std)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_fused_optimizer_is_cuda_only(monkeypatch) -> None:
    captured: dict = {}

    def fake_adamw(parameters, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(torch.optim, "AdamW", fake_adamw)
    model = torch.nn.Linear(2, 2)
    cfg = {
        "train": {
            "lr": 1e-4,
            "weight_decay": 1e-2,
            "fused_optimizer": True,
        }
    }

    _build_optimizer(model, cfg, torch.device("cuda"))
    assert captured["fused"] is True

    captured.clear()
    _build_optimizer(model, cfg, torch.device("cpu"))
    assert "fused" not in captured


def test_step_metrics_writer_buffers_complete_optimizer_steps(tmp_path) -> None:
    path = tmp_path / "step_metrics.csv"
    writer = StepMetricsWriter(path, flush_steps=2)
    loss_cfg = {
        "semantic_weight": 0.1,
        "classification_weight": 0.2,
        "spatial_weight": 0.3,
        "prototype_filter_weight": 0.4,
        "zhcc_response_weight": 0.5,
    }
    for step in (1, 2):
        writer.append(
            epoch=1,
            global_step=step,
            spatial_supervised_step=step - 1,
            tiles_seen_in_epoch=step * 8,
            lr=1e-4,
            loss_cfg=loss_cfg,
            classification_active=step == 2,
            spatial_active=step == 2,
            loss=torch.tensor(float(step)),
            parts={"feature": torch.tensor(float(step) / 10)},
        )

    rows = path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3
    assert "global_step" in rows[0]
    assert rows[1].split(",")[1] == "1"
    assert rows[2].split(",")[1] == "2"


def test_compiled_training_disables_incompatible_buffer_donation() -> None:
    from torch._functorch import config as functorch_config

    previous = functorch_config.donated_buffer
    try:
        functorch_config.donated_buffer = True
        _configure_compiled_training_for_gradient_diagnostics()
        assert functorch_config.donated_buffer is False
    finally:
        functorch_config.donated_buffer = previous


class _CountingScaler:
    def __init__(self) -> None:
        self.step_count = 0

    def is_enabled(self) -> bool:
        return True

    def get_scale(self) -> float:
        return 8.0

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer) -> None:
        del optimizer

    def step(self, optimizer) -> None:
        self.step_count += 1
        optimizer.step()

    def update(self) -> None:
        return None


def test_scheduled_loss_config_warms_parallel_expert_terms_together() -> None:
    cfg = {
        "loss": {
            "relation_weight": 0.25,
            "semantic_weight": 0.4,
            "semantic_temperature": 1.0,
            "classification_weight": 1.0,
            "spatial_weight": 0.2,
            "expert_supervision_start_step": 100,
            "expert_supervision_ramp_steps": 100,
        }
    }

    teacher_only = scheduled_loss_config(cfg, epoch=1, global_step=99)
    ramping = scheduled_loss_config(cfg, epoch=4, global_step=150)
    active = scheduled_loss_config(cfg, epoch=4, global_step=200)

    assert teacher_only["semantic_weight"] == pytest.approx(0.0)
    assert teacher_only["classification_weight"] == pytest.approx(0.0)
    assert teacher_only["spatial_weight"] == pytest.approx(0.0)
    assert ramping["semantic_weight"] == pytest.approx(0.2)
    assert ramping["classification_weight"] == pytest.approx(0.5)
    assert ramping["spatial_weight"] == pytest.approx(0.1)
    assert active["semantic_weight"] == pytest.approx(0.4)
    assert active["classification_weight"] == pytest.approx(1.0)
    assert active["spatial_weight"] == pytest.approx(0.2)
    assert active["feature_loss_type"] == "cosine"
    assert active["spatial_point_tolerance_cells"] == 1
    assert active["spatial_abundance_point_weight"] == pytest.approx(0.5)
    assert active["spatial_brush_top_fraction"] == pytest.approx(1.0)
    assert active["spatial_implicit_negative_weight"] == pytest.approx(0.05)
    assert active["spatial_detach_backbone"] is False


def test_feature_loss_type_defaults_to_cosine() -> None:
    student = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    teacher = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    cosine = feature_distillation_loss_per_sample(student, teacher, loss_type="cosine")
    norm_mse = feature_distillation_loss_per_sample(student, teacher, loss_type="cosine_plus_norm_mse")

    assert cosine.tolist() == pytest.approx([0.0, 1.0])
    assert float(norm_mse[1]) > float(cosine[1])


def test_spatial_supervision_reaches_shared_encoder_during_common_ramp() -> None:
    cfg = {
        "loss": {
            "relation_weight": 0.0,
            "classification_weight": 1.0,
            "spatial_weight": 0.1,
            "expert_supervision_start_step": 100,
            "expert_supervision_ramp_steps": 100,
        }
    }
    teacher_only = scheduled_loss_config(cfg, epoch=1, global_step=50)
    joint = scheduled_loss_config(cfg, epoch=1, global_step=150)

    assert teacher_only["classification_weight"] == pytest.approx(0.0)
    assert teacher_only["spatial_weight"] == pytest.approx(0.0)
    assert joint["classification_weight"] == pytest.approx(0.5)
    assert joint["spatial_weight"] == pytest.approx(0.05)
    assert joint["spatial_detach_backbone"] is False


def test_detached_spatial_encoder_is_an_explicit_ablation_only() -> None:
    cfg = {
        "loss": {
            "spatial_weight": 0.2,
            "expert_supervision_start_step": 0,
            "expert_supervision_ramp_steps": 100,
            "spatial_detach_shared_encoder": True,
        }
    }

    scheduled = scheduled_loss_config(
        cfg,
        epoch=1,
        global_step=20,
    )

    assert scheduled["spatial_weight"] == pytest.approx(0.04)
    assert scheduled["spatial_detach_backbone"] is True


def test_cosine_scheduler_warms_up_and_decays() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    cfg = {"train": {"scheduler": "cosine", "epochs": 3, "lr_warmup_steps": 2, "min_lr": 0.01, "lr": 0.1}}
    scheduler = build_lr_scheduler(optimizer, cfg, steps_per_epoch=2)

    lrs = []
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    assert lrs[0] > 0.01
    assert max(lrs[:2]) <= 0.1
    assert lrs[-1] < lrs[2]


def test_amp_optimizer_helper_steps_exactly_once() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = _CountingScaler()

    stepped = _optimizer_step(
        loss=parameter.square(),
        model=torch.nn.ParameterList([parameter]),
        optimizer=optimizer,
        scaler=scaler,
        max_grad_norm=0.0,
    )

    assert stepped is True
    assert scaler.step_count == 1
    assert parameter.item() == pytest.approx(0.8)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_optimizer_helper_rejects_nonfinite_loss_without_step(
    value: float,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)

    with pytest.raises(FloatingPointError, match="finite scalar"):
        _optimizer_step(
            loss=parameter * value,
            model=torch.nn.ParameterList([parameter]),
            optimizer=optimizer,
            scaler=None,
            max_grad_norm=0.0,
        )

    assert parameter.item() == 1.0
    assert parameter.grad is None


def test_run_epoch_joint_classification_spatial_route_keeps_full_bank_prototypes_fixed() -> None:
    classification_count = len(DEFAULT_CLASSIFICATION_CLASSES)
    spatial_count = len(DEFAULT_SPATIAL_COMPONENTS)
    model = HCCSemPathModel(
        backbone_name="vit_tiny_patch16_224",
        embedding_dim=8,
        teacher_dims={"teacher": 4},
        pretrained=False,
        classification_num_classes=classification_count,
        spatial_num_components=spatial_count,
        spatial_dim=12,
        spatial_output_stride=7,
    )
    prototypes = {
        "teacher": PrototypeRegistry(
            prototypes=torch.randn(classification_count, 4),
            names=list(DEFAULT_CLASSIFICATION_CLASSES),
        )
    }
    grid = 31
    point_centers = torch.zeros((1, spatial_count, grid, grid))
    point_centers[0, 0, 10, 10] = 1
    implicit_negative = torch.zeros(
        (1, spatial_count, grid, grid),
        dtype=torch.bool,
    )
    implicit_negative[0, 0, 0, 0] = True
    zeros_bool = torch.zeros_like(implicit_negative)
    batch = {
        "tile_id": ["tile-0"],
        "images": torch.randn(1, 3, 224, 224),
        "teacher_features": {"teacher": torch.randn(1, 4)},
        "prototype_mask": torch.tensor([True]),
        "prototype_classification": torch.tensor([0]),
        "spatial_point_centers": point_centers,
        "spatial_brush_bag_ids": torch.zeros(
            (1, spatial_count, grid, grid),
            dtype=torch.long,
        ),
        "spatial_area_positive": zeros_bool,
        "spatial_explicit_negative": zeros_bool,
        "spatial_implicit_negative": implicit_negative,
        "spatial_supervised": torch.tensor(
            [[True] + [False] * (spatial_count - 1)]
        ),
    }
    model.replace_classification_prototypes(
        torch.randn(classification_count, 8),
        torch.ones(classification_count),
    )
    model.replace_global_spatial_prototypes(
        torch.randn(spatial_count, 8),
        torch.ones(spatial_count),
        {"teacher": torch.randn(spatial_count, 4)},
    )
    spatial_observations = {
        name: (torch.randn(spatial_count, 12), torch.ones(spatial_count))
        for name in (
            "instance",
            "measurement",
            "instance_negative",
            "measurement_negative",
            "instance_implicit_negative",
            "measurement_implicit_negative",
        )
    }
    model.spatial_head.replace_prototypes(spatial_observations)
    classification_before = model.classification_prototypes.clone()
    spatial_before = model.spatial_head.instance_prototypes.clone()
    cfg = {
        "runtime": {"device": "cpu", "seed": 13},
        "data": {
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
            "spatial_component_names": None,
        },
        "model": {"classification_class_names": list(DEFAULT_CLASSIFICATION_CLASSES)},
        "loss": {
            "teacher_weights": {"teacher": 1.0},
            "feature_loss_type": "cosine",
            "relation_weight": 0.0,
            "semantic_weight": 0.0,
            "prototype_filter_weight": 0.0,
            "zhcc_response_weight": 0.0,
            "classification_weight": 1.0,
            "spatial_weight": 0.1,
            "expert_supervision_start_step": 0,
            "expert_supervision_ramp_steps": 0,
            "spatial_detach_shared_encoder": False,
        },
        "train": {
            "log_interval": 0,
            "progress": False,
            "max_grad_norm": 0.0,
        },
    }
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    checkpoints = []
    callback_order = []

    def development_probe(*args):
        callback_order.append("probe")
        return True

    def step_checkpoint(*args):
        callback_order.append("checkpoint")
        checkpoints.append(args)

    result = run_epoch(
        model,
        [batch, batch],
        prototypes,
        optimizer,
        torch.device("cpu"),
        cfg,
        train=True,
        development_probe=development_probe,
        step_checkpoint=step_checkpoint,
    )

    assert result["global_step_end"] == 1
    assert result["spatial_supervised_step_end"] == 1
    assert result["batch_in_epoch_end"] == 1
    assert len(checkpoints) == 1
    assert callback_order == ["probe", "checkpoint"]
    assert checkpoints[0][:4] == (1, 1, 1, 1)
    assert checkpoints[0][4]["batches"] == 1
    assert checkpoints[0][4]["tiles"] == 1
    assert checkpoints[0][4]["totals"]["loss"] > 0
    torch.testing.assert_close(model.classification_prototypes, classification_before)
    torch.testing.assert_close(
        model.spatial_head.instance_prototypes,
        spatial_before,
    )


def test_spatial_pair_reduction_balances_active_components() -> None:
    values = torch.tensor(
        [[1.0, 10.0], [3.0, 0.0], [5.0, 0.0]],
        requires_grad=True,
    )
    supervised = torch.tensor(
        [[True, True], [True, False], [True, False]]
    )

    loss = _mean_supervised_pair(values, supervised, values)

    assert loss.item() == pytest.approx(6.5)
    loss.backward()
    assert values.grad is not None


def test_spatial_pair_reduction_empty_mask_returns_connected_zero() -> None:
    values = torch.randn(2, 3, requires_grad=True)

    loss = _mean_supervised_pair(
        values,
        torch.zeros_like(values, dtype=torch.bool),
        values,
    )

    assert loss.item() == 0.0
    loss.backward()
    assert values.grad is not None


def test_local_negative_is_not_promoted_to_global_component_absence() -> None:
    shape = (2, 1, 3, 3)
    point = torch.zeros(shape)
    point[0, 0, 1, 1] = 1
    explicit = torch.zeros(shape, dtype=torch.bool)
    explicit[0, 0, 0, 0] = True
    explicit[1, 0].fill_(True)
    batch = {
        "spatial_point_centers": point,
        "spatial_brush_bag_ids": torch.zeros(shape, dtype=torch.long),
        "spatial_area_positive": torch.zeros(shape, dtype=torch.bool),
        "spatial_explicit_negative": explicit,
    }

    positive, known = _spatial_global_targets_from_spatial(batch)

    assert positive[:, 0].tolist() == [True, False]
    assert known[:, 0].tolist() == [True, True]


def test_objective_gradient_diagnostics_measure_shared_signal() -> None:
    shared = torch.tensor([1.0, 2.0], requires_grad=True)
    diagnostics = _objective_gradient_diagnostics(
        (shared.square()).sum(),
        (3.0 * shared).sum(),
        (shared,),
    )

    assert diagnostics["gradient_global_norm"] > 0
    assert diagnostics["gradient_spatial_norm"] > 0
    assert 0 < diagnostics["gradient_spatial_share"] < 1
    assert -1 <= diagnostics["gradient_spatial_global_cosine"] <= 1


def test_alignment_early_stop_requires_small_global_and_per_teacher_gain() -> None:
    history = [
        {
            "epoch": 60.0,
            "teacher_alignment_score": 0.810,
            "gigapath_feature_cosine": 0.780,
            "uni2_h_feature_cosine": 0.760,
        },
        {
            "epoch": 80.0,
            "teacher_alignment_score": 0.811,
            "gigapath_feature_cosine": 0.781,
            "uni2_h_feature_cosine": 0.761,
        },
    ]
    assert _should_stop_for_alignment(history) is True

    history[-1]["uni2_h_feature_cosine"] = 0.764
    assert _should_stop_for_alignment(history) is False


def test_teacher_names_allows_h_optimus_1_teacher() -> None:
    cfg = {"data": {"teachers": ["gigapath", "h_optimus_1"]}, "model": {"teacher_dims": {"gigapath": 1536, "h_optimus_1": 1536}}}

    assert teacher_names(cfg) == ["gigapath", "h_optimus_1"]


def test_training_config_rejects_stale_teacher_entries() -> None:
    cfg = {
        "data": {
            "teachers": ["gigapath", "h_optimus_1", "uni2_h", "virchow2"],
            "prototype_paths": {
                "gigapath": "g.pt",
                "h_optimus_1": "h.pt",
                "uni2_h": "u.pt",
                "virchow2": "v.pt",
                "stale_teacher": "removed.pt",
            },
        },
        "model": {
            "teacher_dims": {
                "gigapath": 1536,
                "h_optimus_1": 1536,
                "uni2_h": 1536,
                "virchow2": 2560,
                "stale_teacher": 1536,
            }
        },
        "loss": {
            "teacher_weights": {
                "gigapath": 1.0,
                "h_optimus_1": 1.0,
                "uni2_h": 1.0,
                "virchow2": 1.0,
                "stale_teacher": 1.0,
            },
            "semantic_weight": 0.25,
            "prototype_filter_weight": 0.5,
        },
    }
    names = ["gigapath", "h_optimus_1", "uni2_h", "virchow2"]

    with pytest.raises(ValueError, match="model.teacher_dims contains unknown teacher entries"):
        validate_training_config(cfg, names)


def test_training_config_rejects_backbone_configuration() -> None:
    cfg = {
        "data": {"teachers": ["teacher"]},
        "model": {"teacher_dims": {"teacher": 8}, "pretrained": False},
        "loss": {},
    }

    with pytest.raises(ValueError, match="fixed pretrained DINOv2-S/14"):
        validate_training_config(cfg, ["teacher"])


def test_training_config_rejects_epoch_scaled_lr_warmup() -> None:
    cfg = {
        "data": {"teachers": ["teacher"]},
        "model": {"teacher_dims": {"teacher": 8}},
        "loss": {},
        "train": {"warmup_epochs": 1},
    }

    with pytest.raises(ValueError, match="lr_warmup_steps"):
        validate_training_config(cfg, ["teacher"])


def test_training_config_rejects_unsupported_exact_area_spatial_loss() -> None:
    cfg = {
        "data": {"teachers": ["teacher"]},
        "model": {"teacher_dims": {"teacher": 8}},
        "loss": {"spatial_region_dice_weight": 1.0},
    }

    with pytest.raises(ValueError, match="spatial_region_dice_weight"):
        validate_training_config(cfg, ["teacher"])


@pytest.mark.parametrize(
    "obsolete_key",
    (
        "semantic_warmup_epochs",
        "classification_start_step",
        "classification_ramp_steps",
        "spatial_start_step",
        "spatial_ramp_steps",
        "spatial_backbone_start_step",
    ),
)
def test_training_config_rejects_asynchronous_expert_schedules(
    obsolete_key: str,
) -> None:
    cfg = {
        "data": {"teachers": ["teacher"]},
        "model": {"teacher_dims": {"teacher": 8}},
        "loss": {obsolete_key: 1},
    }

    with pytest.raises(ValueError, match=obsolete_key):
        validate_training_config(cfg, ["teacher"])


def test_training_config_validates_brush_bag_fraction() -> None:
    cfg = {
        "data": {"teachers": ["teacher"]},
        "model": {"teacher_dims": {"teacher": 8}},
        "loss": {"spatial_brush_top_fraction": 0.0},
    }

    with pytest.raises(ValueError, match="spatial_brush_top_fraction"):
        validate_training_config(cfg, ["teacher"])


@pytest.mark.parametrize(
    "key",
    (
        "spatial_use_local_branch",
        "spatial_use_semantic_branch",
        "spatial_use_context",
    ),
)
def test_training_config_validates_spatial_ablation_booleans(
    key: str,
) -> None:
    cfg = {
        "data": {"teachers": ["teacher"]},
        "model": {"teacher_dims": {"teacher": 8}, key: 0},
        "loss": {},
    }

    with pytest.raises(ValueError, match=key):
        validate_training_config(cfg, ["teacher"])


def test_training_config_requires_one_spatial_observation_branch() -> None:
    cfg = {
        "data": {"teachers": ["teacher"]},
        "model": {
            "teacher_dims": {"teacher": 8},
            "spatial_use_local_branch": False,
            "spatial_use_semantic_branch": False,
        },
        "loss": {},
    }

    with pytest.raises(ValueError, match="cannot both be false"):
        validate_training_config(cfg, ["teacher"])


def test_training_config_rejects_unsupported_schedule_keys() -> None:
    cfg = {
        "data": {"teachers": ["teacher"]},
        "model": {"teacher_dims": {"teacher": 8}},
        "loss": {"filter_ramp_steps": 100},
    }

    with pytest.raises(ValueError, match="filter_ramp_steps"):
        validate_training_config(cfg, ["teacher"])


def test_training_config_accepts_active_pamtd_controls() -> None:
    cfg = {
        "data": {
            "teachers": ["teacher"],
            "prototype_path": "prototypes.pt",
            "expert_replay_interval_batches": 16,
            "expert_batch_size": 64,
        },
        "model": {"teacher_dims": {"teacher": 8}},
        "loss": {
            "semantic_weight": 0.02,
            "prototype_filter_weight": 0.5,
            "prototype_filter_alpha_min": 0.25,
            "prototype_consensus_weight": 0.4,
            "prototype_label_weight": 0.4,
            "prototype_student_weight": 0.2,
            "zhcc_response_weight": 0.15,
            "pamtd_classification_temperature": 0.1,
            "spatial_global_temperature": 0.1,
        },
        "train": {
            "dynamic_prototype_refresh_steps": 500,
            "dynamic_prototype_batch_size": 512,
            "dynamic_spatial_prototype_refresh_steps": 500,
        },
    }

    validate_training_config(cfg, ["teacher"])


def test_teacher_dims_require_configured_teacher_entries() -> None:
    cfg = {"model": {"teacher_dims": {"gigapath": 1536, "uni2_h": 1536}}}

    with pytest.raises(ValueError, match="model.teacher_dims missing teacher entries"):
        teacher_dims(cfg, ["gigapath", "uni2_h", "virchow2"])


def test_resume_contract_freezes_data_seed_and_preprocessing() -> None:
    cfg = {
        "runtime": {
            "device": "cuda",
            "output_dir": "run-a",
            "seed": 13,
        },
        "data": {
            "train_manifest_path": "manifest.yaml",
            "prototype_paths": {"teacher": "prototype.pt"},
            "mean": [0.1, 0.2, 0.3],
            "std": [0.4, 0.5, 0.6],
            "num_workers": 8,
            "prefetch_factor": 2,
        },
        "model": {"teacher_dims": {"teacher": 4}},
        "loss": {"teacher_weights": {"teacher": 1.0}},
        "train": {
            "epochs": 100,
            "max_grad_norm": 1.0,
            "log_interval": 10,
            "tensorboard": True,
        },
    }
    baseline = _resume_contract(cfg)
    host_only = {
        **cfg,
        "runtime": {
            **cfg["runtime"],
            "device": "cpu",
            "output_dir": "run-b",
        },
        "data": {**cfg["data"], "num_workers": 2, "prefetch_factor": 1},
        "train": {
            **cfg["train"],
            "epochs": 120,
            "log_interval": 0,
            "tensorboard": False,
        },
    }
    assert _resume_contract(host_only) == baseline

    for section, key, value in (
        ("runtime", "seed", 14),
        ("data", "mean", [0.2, 0.2, 0.3]),
        ("data", "prototype_paths", {"teacher": "other.pt"}),
        ("train", "max_grad_norm", 2.0),
    ):
        changed = {
            name: dict(payload)
            for name, payload in cfg.items()
            if isinstance(payload, dict)
        }
        changed[section][key] = value
        assert _resume_contract(changed) != baseline


def test_training_config_rejects_unsupported_profiling_controls() -> None:
    cfg = {
        "data": {"teachers": ["teacher"]},
        "model": {"teacher_dims": {"teacher": 8}},
        "loss": {},
        "train": {"pipeline_profile_interval": 10},
    }

    with pytest.raises(ValueError, match="unsupported profiling"):
        validate_training_config(cfg, ["teacher"])


def test_training_config_rejects_negative_checkpoint_interval() -> None:
    cfg = {
        "data": {"teachers": ["teacher"]},
        "model": {"teacher_dims": {"teacher": 8}},
        "loss": {},
        "train": {"checkpoint_interval_steps": -1},
    }

    with pytest.raises(ValueError, match="checkpoint_interval_steps"):
        validate_training_config(cfg, ["teacher"])


@pytest.mark.parametrize(
    "weights",
    [
        {
            "teacher": 0.50,
            "classification": 0.50,
        },
        {
            "teacher": 0.50,
            "classification": 0.25,
            "spatial": 0.20,
        },
        {
            "teacher": 0.75,
            "classification": 0.25,
            "spatial": 0.0,
        },
    ],
)
def test_training_config_rejects_invalid_selection_weights(weights) -> None:
    cfg = {
        "data": {"teachers": ["teacher"]},
        "model": {"teacher_dims": {"teacher": 8}},
        "loss": {},
        "train": {"selection_metric_weights": weights},
    }

    with pytest.raises(ValueError, match="selection_metric_weights"):
        validate_training_config(cfg, ["teacher"])


def test_training_config_rejects_degenerate_pamtd_reliability() -> None:
    cfg = {
        "data": {
            "teachers": ["teacher"],
            "prototype_paths": {"teacher": "prototype.pt"},
        },
        "model": {"teacher_dims": {"teacher": 8}},
        "loss": {
            "teacher_weights": {"teacher": 1.0},
            "prototype_filter_weight": 1.0,
            "prototype_filter_alpha_min": 0.0,
            "prototype_consensus_weight": 0.0,
            "prototype_label_weight": 1.0,
            "prototype_student_weight": 0.0,
        },
        "train": {},
    }

    with pytest.raises(ValueError, match="population tiles"):
        validate_training_config(cfg, ["teacher"])


def test_training_config_rejects_spatial_teacher_alignment_early_stop() -> None:
    cfg = {
        "data": {
            "teachers": ["teacher"],
            "spatial_manifest_path": "spatial.json",
        },
        "model": {"teacher_dims": {"teacher": 8}},
        "loss": {"teacher_weights": {"teacher": 1.0}},
        "train": {"early_stop_teacher_alignment": True},
    }

    with pytest.raises(ValueError, match="terminal epoch"):
        validate_training_config(cfg, ["teacher"])


def test_development_early_stop_cannot_precede_active_loss_ramps() -> None:
    cfg = {
        "data": {"teachers": ["teacher"]},
        "model": {"teacher_dims": {"teacher": 8}},
        "loss": {
            "teacher_weights": {"teacher": 1.0},
            "expert_supervision_start_step": 1000,
            "expert_supervision_ramp_steps": 1000,
            "prototype_filter_weight": 0.0,
            "zhcc_response_weight": 0.0,
        },
        "train": {
            "development_probe_interval_steps": 1000,
            "development_early_stop": True,
            "development_early_stop_min_step": 1500,
        },
    }

    with pytest.raises(ValueError, match="final active loss ramp"):
        validate_training_config(cfg, ["teacher"])


def test_spatial_fit_reports_terminal_epoch_and_sets_epoch_before_iter(
    tmp_path,
    monkeypatch,
) -> None:
    events: list[tuple[str, int]] = []

    class Loader:
        def __init__(self, label: str) -> None:
            self.label = label
            self.epoch = -1

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch
            events.append((f"{self.label}_set", epoch))

        def __iter__(self):
            events.append((f"{self.label}_iter", self.epoch))
            return iter(())

        def __len__(self) -> int:
            return 1

    train_loader = Loader("train")
    val_loader = Loader("val")
    val_losses = iter([0.1, 0.2])

    def fake_run_epoch(*args, train: bool, epoch: int, **kwargs):
        loader = args[1]
        iter(loader)
        if train:
            return {
                "loss": float(epoch),
                "global_step_end": epoch,
                "spatial_supervised_step_end": epoch,
            }
        metrics = {"loss": next(val_losses)}
        embeddings = (
            torch.zeros((1, 2)),
            {"teacher": torch.zeros((1, 2))},
            {"teacher": torch.zeros((1, 2))},
            {
                "prototype_mask": torch.zeros(1, dtype=torch.bool),
                "prototype_classification": torch.full((1,), -1),
                "classification_logits": torch.zeros((0, 0)),
            },
        )
        return metrics, embeddings

    monkeypatch.setattr(
        "hcc_sempath.training.engine.run_epoch",
        fake_run_epoch,
    )
    monkeypatch.setattr(
        "hcc_sempath.training.engine.evaluate_teacher_outputs",
        lambda *args, **kwargs: {"teacher_feature_cosine": 0.5},
    )
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = {
        "runtime": {"output_dir": str(tmp_path), "seed": 13},
        "data": {"spatial_manifest_path": "spatial.json"},
        "model": {},
        "loss": {
            "relation_weight": 0.0,
            "semantic_weight": 0.0,
            "classification_weight": 0.0,
            "spatial_weight": 0.0,
        },
        "train": {
            "epochs": 2,
            "amp": False,
            "topk": 1,
            "tensorboard": False,
            "early_stop_teacher_alignment": False,
        },
    }

    summary = fit(
        model,
        train_loader,
        val_loader,
        None,
        optimizer,
        torch.device("cpu"),
        cfg,
    )

    assert summary["epoch"] == 2
    assert json.loads((tmp_path / "summary.json").read_text())["epoch"] == 2
    checkpoint = torch.load(
        tmp_path / "checkpoints" / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["training_complete"] is True
    assert checkpoint["expected_epochs"] == 2
    assert checkpoint["optimizer_hyperparameters"][0]["lr"] == pytest.approx(0.1)
    assert checkpoint["scheduler_contract"]["name"] == "none"
    assert checkpoint["scheduler_contract"]["planned_epochs"] == 2
    assert events.index(("train_set", 0)) < events.index(("train_iter", 0))
    assert events.index(("train_set", 1)) < events.index(("train_iter", 1))


def test_resume_terminal_epoch_comes_only_from_config() -> None:
    cfg = {"train": {"epochs": 6}}
    completed = {"epoch": 3, "expected_epochs": 3}
    extended = {"epoch": 4, "expected_epochs": 6}

    assert _resolve_configured_epochs(cfg, completed) == 6
    assert _resolve_configured_epochs(cfg, extended) == 6

    with pytest.raises(ValueError, match="checkpoint epoch"):
        _resolve_configured_epochs({"train": {"epochs": 3}}, extended)


def test_fit_selects_checkpoint_from_independent_expert_validation(
    tmp_path,
    monkeypatch,
) -> None:
    train_loader = [object()]
    population_val_loader = [object()]
    classification_val_loader = [object()]
    spatial_val_loader = [object()]
    spatial_losses = iter([0.8, 0.8, 0.4, 0.7])
    raw_model = torch.nn.Linear(2, 2)

    class CompiledLikeModel(torch.nn.Module):
        def __init__(self, raw):
            super().__init__()
            self._orig_mod = raw

        def forward(self, inputs):
            return self._orig_mod(inputs)

    compiled_like_model = CompiledLikeModel(raw_model)

    def fake_run_epoch(
        model,
        loader,
        prototypes,
        optimizer,
        device,
        cfg,
        train,
        **kwargs,
    ):
        del prototypes, optimizer, device, cfg, kwargs
        if train:
            assert model is compiled_like_model
            epoch = fake_run_epoch.train_epoch
            fake_run_epoch.train_epoch += 1
            return {
                "loss": 1.0,
                "global_step_end": float(epoch),
                "spatial_supervised_step_end": float(epoch),
            }
        assert model is raw_model
        if loader is population_val_loader:
            embeddings = (
                torch.zeros((1, 2)),
                {"teacher": torch.zeros((1, 2))},
                {"teacher": torch.zeros((1, 2))},
                {
                    "prototype_mask": torch.zeros(
                        1,
                        dtype=torch.bool,
                    ),
                    "prototype_classification": torch.full(
                        (1,),
                        -1,
                    ),
                    "classification_logits": torch.zeros((0, 0)),
                },
            )
            return {"loss": 1.0, "feature": 0.1}, embeddings
        if loader is classification_val_loader:
            class_count = len(DEFAULT_CLASSIFICATION_CLASSES)
            embeddings = (
                torch.zeros((class_count, 2)),
                {},
                {},
                {
                    "prototype_mask": torch.ones(
                        class_count,
                        dtype=torch.bool,
                    ),
                    "prototype_classification": torch.arange(
                        class_count
                    ),
                    "classification_logits": (
                        torch.eye(class_count) * 5.0
                    ),
                },
            )
            return {}, embeddings
        if loader is spatial_val_loader:
            return {
                "spatial": next(spatial_losses),
                "spatial_explicit_negative_pairs": 7.0,
            }
        raise AssertionError("unexpected loader")

    fake_run_epoch.train_epoch = 1
    monkeypatch.setattr(
        "hcc_sempath.training.engine.run_epoch",
        fake_run_epoch,
    )
    monkeypatch.setattr(
        "hcc_sempath.training.engine.evaluate_teacher_outputs",
        lambda *args, **kwargs: {"teacher_feature_cosine": 0.5},
    )
    model = compiled_like_model
    optimizer = torch.optim.SGD(raw_model.parameters(), lr=0.1)
    cfg = {
        "runtime": {"output_dir": str(tmp_path), "seed": 13},
        "data": {"spatial_manifest_path": "spatial.json"},
        "model": {},
        "loss": {
            "relation_weight": 0.0,
            "semantic_weight": 0.0,
            "classification_weight": 1.0,
            "spatial_weight": 1.0,
        },
        "train": {
            "epochs": 3,
            "amp": False,
            "topk": 1,
            "tensorboard": False,
            "early_stop_teacher_alignment": False,
            "selection_early_stop": False,
        },
    }

    summary = fit(
        model,
        train_loader,
        population_val_loader,
        None,
        optimizer,
        torch.device("cpu"),
        cfg,
        expert_classification_val_loader=(
            classification_val_loader
        ),
        expert_spatial_val_loader=spatial_val_loader,
    )

    assert summary["epoch"] == 2
    assert summary["expert_val_spatial"] == pytest.approx(0.4)
    checkpoint = torch.load(
        tmp_path / "checkpoints" / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["epoch"] == 2
    assert checkpoint["best_selection_epoch"] == 2
    assert checkpoint["run_complete"] is True
    assert checkpoint["selection_finalized"] is True
    assert checkpoint["run_terminal_epoch"] == 3


def test_development_metrics_exclude_internal_continuation_state() -> None:
    metrics = {
        "loss": 0.25,
        "tiles": 512.0,
        "epoch_accumulator_end": {
            "totals": {"loss": 0.25},
            "batches": 1,
        },
    }

    assert _scalar_epoch_metrics(metrics) == {
        "loss": 0.25,
        "tiles": 512.0,
    }


def test_resume_truncates_metric_rows_after_checkpoint_step(tmp_path) -> None:
    path = tmp_path / "step_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["global_step", "loss"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"global_step": 3999, "loss": 0.3},
                {"global_step": 4000, "loss": 0.29},
                {"global_step": 4001, "loss": 0.28},
            ]
        )

    _truncate_csv_after_step(path, 4000)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["global_step"]) for row in rows] == [3999, 4000]


def test_early_stop_state_can_be_recovered_from_existing_probe_csv(
    tmp_path,
) -> None:
    path = tmp_path / "development_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["global_step", "loss"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"global_step": 3000, "loss": 0.3000},
                {"global_step": 4000, "loss": 0.2900},
                {"global_step": 5000, "loss": 0.2890},
                {"global_step": 6000, "loss": 0.2880},
            ]
        )

    state = _development_early_stop_state_from_csv(
        path,
        maximum_step=6000,
        minimum_step=4000,
        relative_delta=0.005,
    )

    assert state["last_probe_step"] == 6000
    assert state["previous_loss"] == pytest.approx(0.288)
    assert state["consecutive_low_gain"] == 2


def test_development_early_stop_requires_two_consecutive_small_gains() -> None:
    state = {
        "previous_loss": 0.300,
        "consecutive_low_gain": 0,
        "last_probe_step": 3000,
        "triggered": False,
    }

    _, stopped = _update_development_early_stop_state(
        state,
        step=4000,
        loss=0.290,
        enabled=True,
        minimum_step=4000,
        relative_delta=0.005,
        patience=2,
    )
    assert not stopped
    assert state["consecutive_low_gain"] == 0

    gain, stopped = _update_development_early_stop_state(
        state,
        step=5000,
        loss=0.289,
        enabled=True,
        minimum_step=4000,
        relative_delta=0.005,
        patience=2,
    )
    assert gain == pytest.approx((0.290 - 0.289) / 0.290)
    assert not stopped

    _, stopped = _update_development_early_stop_state(
        state,
        step=6000,
        loss=0.288,
        enabled=True,
        minimum_step=4000,
        relative_delta=0.005,
        patience=2,
    )
    assert stopped
    assert state["triggered"] is True


def test_fit_resumes_the_same_epoch_from_saved_batch_cursor(
    tmp_path,
    monkeypatch,
) -> None:
    events = []

    class Loader:
        def __len__(self):
            return 10

        def set_epoch(self, epoch):
            events.append(("epoch", int(epoch)))

        def set_batch_cursor(self, batch):
            events.append(("cursor", int(batch)))

        def __iter__(self):
            return iter(())

    train_calls = []

    def fake_run_epoch(*args, train: bool, epoch: int, **kwargs):
        if train:
            train_calls.append(
                (epoch, kwargs.get("resume_epoch_accumulator"))
            )
            return {
                "loss": 0.1,
                "global_step_end": kwargs["global_step"] + 1,
                "spatial_supervised_step_end": kwargs[
                    "spatial_supervised_step"
                ],
            }
        return (
            {"loss": 0.1},
            (
                torch.zeros((1, 2)),
                {"teacher": torch.zeros((1, 2))},
                {"teacher": torch.zeros((1, 2))},
                {
                    "prototype_mask": torch.zeros(1, dtype=torch.bool),
                    "prototype_classification": torch.full((1,), -1),
                    "classification_logits": torch.zeros((0, 0)),
                },
            ),
        )

    monkeypatch.setattr(
        "hcc_sempath.training.engine.run_epoch",
        fake_run_epoch,
    )
    monkeypatch.setattr(
        "hcc_sempath.training.engine.evaluate_teacher_outputs",
        lambda *args, **kwargs: {"teacher_feature_cosine": 0.5},
    )
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    accumulator = {
        "totals": {"loss": 3.0},
        "gradient_totals": {},
        "gradient_count": 0,
        "last_gradient_step": 1000,
        "batches": 3,
        "tiles": 1536,
        "seconds": 1.0,
    }
    cfg = {
        "runtime": {"output_dir": str(tmp_path), "seed": 13},
        "data": {"spatial_manifest_path": "spatial.json"},
        "model": {},
        "loss": {
            "relation_weight": 0.0,
            "semantic_weight": 0.0,
            "classification_weight": 0.0,
            "spatial_weight": 0.0,
        },
        "train": {
            "epochs": 3,
            "amp": False,
            "topk": 1,
            "tensorboard": False,
            "early_stop_teacher_alignment": False,
        },
    }

    fit(
        model,
        Loader(),
        Loader(),
        None,
        optimizer,
        torch.device("cpu"),
        cfg,
        resume_state={
            "epoch": 2,
            "batch_in_epoch": 3,
            "epoch_accumulator": accumulator,
            "expected_epochs": 3,
            "global_step": 2000,
        },
    )

    resume_epoch_position = events.index(("epoch", 1))
    assert events[resume_epoch_position : resume_epoch_position + 2] == [
        ("epoch", 1),
        ("cursor", 3),
    ]
    assert train_calls[0] == (2, accumulator)
    assert train_calls[1] == (3, None)
