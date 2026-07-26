from __future__ import annotations

import json

import pytest

from hcc_sempath.training.config import teacher_dims, teacher_names, validate_training_config
from hcc_sempath.training.engine import (
    _normalize_uint8_images_fp16,
    _objective_gradient_diagnostics,
    _optimizer_step,
    _should_stop_for_alignment,
    build_lr_scheduler,
    fit,
    run_epoch,
    scheduled_loss_config,
    StepMetricsWriter,
)
from hcc_sempath.training.losses import feature_distillation_loss_per_sample
from hcc_sempath.training.prototype_labels import DEFAULT_L1_CLASSES
from hcc_sempath.training.train import (
    _build_optimizer,
    _configure_compiled_training_for_gradient_diagnostics,
    _resume_contract,
)
from hcc_sempath.modeling.models import HCCSemPathModel
from hcc_sempath.modeling.prototypes import PrototypeRegistry
import torch


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
        "l1_weight": 0.2,
        "spatial_weight": 0.3,
        "prototype_filter_weight": 0.4,
        "zhcc_response_weight": 0.5,
    }
    for step in (1, 2):
        writer.append(
            epoch=1,
            global_step=step,
            l2_supervised_step=step - 1,
            tiles_seen_in_epoch=step * 8,
            lr=1e-4,
            loss_cfg=loss_cfg,
            l1_active=step == 2,
            l2_active=step == 2,
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
            "l1_weight": 1.0,
            "spatial_weight": 0.2,
            "expert_supervision_start_step": 100,
            "expert_supervision_ramp_steps": 100,
        }
    }

    teacher_only = scheduled_loss_config(cfg, epoch=1, global_step=99)
    ramping = scheduled_loss_config(cfg, epoch=4, global_step=150)
    active = scheduled_loss_config(cfg, epoch=4, global_step=200)

    assert teacher_only["semantic_weight"] == pytest.approx(0.0)
    assert teacher_only["l1_weight"] == pytest.approx(0.0)
    assert teacher_only["spatial_weight"] == pytest.approx(0.0)
    assert ramping["semantic_weight"] == pytest.approx(0.2)
    assert ramping["l1_weight"] == pytest.approx(0.5)
    assert ramping["spatial_weight"] == pytest.approx(0.1)
    assert active["semantic_weight"] == pytest.approx(0.4)
    assert active["l1_weight"] == pytest.approx(1.0)
    assert active["spatial_weight"] == pytest.approx(0.2)
    assert active["feature_loss_type"] == "cosine"
    assert active["spatial_point_tolerance_cells"] == 1
    assert active["spatial_abundance_point_weight"] == pytest.approx(0.5)
    assert active["spatial_brush_top_fraction"] == pytest.approx(0.25)
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
            "l1_weight": 1.0,
            "spatial_weight": 0.1,
            "expert_supervision_start_step": 100,
            "expert_supervision_ramp_steps": 100,
        }
    }
    teacher_only = scheduled_loss_config(cfg, epoch=1, global_step=50)
    joint = scheduled_loss_config(cfg, epoch=1, global_step=150)

    assert teacher_only["l1_weight"] == pytest.approx(0.0)
    assert teacher_only["spatial_weight"] == pytest.approx(0.0)
    assert joint["l1_weight"] == pytest.approx(0.5)
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


def test_run_epoch_joint_l1_l2_route_keeps_full_bank_prototypes_fixed() -> None:
    model = HCCSemPathModel(
        backbone_name="vit_tiny_patch16_224",
        embedding_dim=8,
        teacher_dims={"teacher": 4},
        pretrained=False,
        l1_num_classes=4,
        spatial_num_components=9,
        spatial_dim=12,
        spatial_output_stride=7,
    )
    prototypes = {
        "teacher": PrototypeRegistry(
            prototypes=torch.eye(4),
            names=list(DEFAULT_L1_CLASSES),
            groups=["l1"] * 4,
            levels=[1] * 4,
            exclusive=[True] * 4,
        )
    }
    grid = 31
    point_centers = torch.zeros((1, 9, grid, grid))
    point_centers[0, 0, 10, 10] = 1
    implicit_negative = torch.zeros(
        (1, 9, grid, grid),
        dtype=torch.bool,
    )
    implicit_negative[0, 0, 0, 0] = True
    zeros_bool = torch.zeros_like(implicit_negative)
    batch = {
        "tile_id": ["tile-0"],
        "images": torch.randn(1, 3, 224, 224),
        "teacher_features": {"teacher": torch.randn(1, 4)},
        "prototype_mask": torch.tensor([True]),
        "prototype_level1": torch.tensor([0]),
        "l2_point_centers": point_centers,
        "l2_brush_bag_ids": torch.zeros(
            (1, 9, grid, grid),
            dtype=torch.long,
        ),
        "l2_area_positive": zeros_bool,
        "l2_explicit_negative": zeros_bool,
        "l2_implicit_negative": implicit_negative,
        "l2_spatial_supervised": torch.tensor(
            [[True] + [False] * 8]
        ),
    }
    model.replace_l1_prototypes(
        torch.randn(4, 8),
        torch.ones(4),
    )
    model.replace_global_l2_prototypes(
        torch.randn(9, 8),
        torch.ones(9),
        {"teacher": torch.randn(9, 4)},
    )
    spatial_observations = {
        name: (torch.randn(9, 12), torch.ones(9))
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
    l1_before = model.l1_prototypes.clone()
    spatial_before = model.spatial_head.instance_prototypes.clone()
    cfg = {
        "runtime": {"device": "cpu", "seed": 13},
        "data": {
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
            "spatial_component_names": None,
        },
        "model": {"l1_class_names": list(DEFAULT_L1_CLASSES)},
        "loss": {
            "teacher_weights": {"teacher": 1.0},
            "feature_loss_type": "cosine",
            "relation_weight": 0.0,
            "semantic_weight": 0.0,
            "prototype_filter_weight": 0.0,
            "zhcc_response_weight": 0.0,
            "l1_weight": 1.0,
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

    result = run_epoch(
        model,
        [batch],
        prototypes,
        optimizer,
        torch.device("cpu"),
        cfg,
        train=True,
    )

    assert result["global_step_end"] == 1
    assert result["l2_supervised_step_end"] == 1
    torch.testing.assert_close(model.l1_prototypes, l1_before)
    torch.testing.assert_close(
        model.spatial_head.instance_prototypes,
        spatial_before,
    )


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
        "l1_start_step",
        "l1_ramp_steps",
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
            "pamtd_primary_temperature": 0.1,
            "l2_global_temperature": 0.1,
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


def test_resume_contract_freezes_data_seed_preprocessing_and_epochs() -> None:
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
            "log_interval": 0,
            "tensorboard": False,
        },
    }
    assert _resume_contract(host_only) == baseline

    for section, key, value in (
        ("runtime", "seed", 14),
        ("data", "mean", [0.2, 0.2, 0.3]),
        ("data", "prototype_paths", {"teacher": "other.pt"}),
        ("train", "epochs", 101),
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
                "l2_supervised_step_end": epoch,
            }
        metrics = {"loss": next(val_losses)}
        embeddings = (
            torch.zeros((1, 2)),
            {"teacher": torch.zeros((1, 2))},
            {"teacher": torch.zeros((1, 2))},
            {
                "prototype_mask": torch.zeros(1, dtype=torch.bool),
                "prototype_level1": torch.full((1,), -1),
                "l1_logits": torch.zeros((0, 0)),
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
            "l1_weight": 0.0,
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
    assert events.index(("train_set", 0)) < events.index(("train_iter", 0))
    assert events.index(("train_set", 1)) < events.index(("train_iter", 1))
