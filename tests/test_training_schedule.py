from __future__ import annotations

import pytest

from hcc_sempath.training.config import teacher_dims, teacher_names, validate_training_config
from hcc_sempath.training.engine import build_lr_scheduler, scheduled_loss_config
from hcc_sempath.training.losses import feature_distillation_loss_per_sample
import torch


def test_scheduled_loss_config_warms_prototype_terms() -> None:
    cfg = {
        "loss": {
            "relation_weight": 0.25,
            "semantic_weight": 0.4,
            "semantic_warmup_epochs": 4,
            "semantic_temperature": 1.0,
            "prototype_filter_weight": 0.8,
            "prototype_filter_warmup_epochs": 2,
            "prototype_filter_alpha_min": 0.3,
            "zhcc_proto_weight": 0.2,
            "zhcc_proto_warmup_epochs": 2,
        }
    }

    epoch_1 = scheduled_loss_config(cfg, epoch=1)
    epoch_4 = scheduled_loss_config(cfg, epoch=4)

    assert epoch_1["semantic_weight"] == pytest.approx(0.1)
    assert epoch_1["prototype_filter_weight"] == pytest.approx(0.4)
    assert epoch_1["prototype_filter_alpha_min"] == 0.3
    assert epoch_1["feature_loss_type"] == "cosine"
    assert epoch_1["zhcc_proto_weight"] == pytest.approx(0.1)
    assert epoch_4["semantic_weight"] == pytest.approx(0.4)
    assert epoch_4["prototype_filter_weight"] == pytest.approx(0.8)
    assert epoch_4["zhcc_proto_weight"] == pytest.approx(0.2)


def test_feature_loss_type_defaults_to_cosine() -> None:
    student = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    teacher = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    cosine = feature_distillation_loss_per_sample(student, teacher, loss_type="cosine")
    norm_mse = feature_distillation_loss_per_sample(student, teacher, loss_type="cosine_plus_norm_mse")

    assert cosine.tolist() == pytest.approx([0.0, 1.0])
    assert float(norm_mse[1]) > float(cosine[1])


def test_cosine_scheduler_warms_up_and_decays() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    cfg = {"train": {"scheduler": "cosine", "epochs": 3, "warmup_epochs": 1, "min_lr": 0.01, "lr": 0.1}}
    scheduler = build_lr_scheduler(optimizer, cfg, steps_per_epoch=2)

    lrs = []
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    assert lrs[0] > 0.01
    assert max(lrs[:2]) <= 0.1
    assert lrs[-1] < lrs[2]


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


def test_teacher_dims_require_configured_teacher_entries() -> None:
    cfg = {"model": {"teacher_dims": {"gigapath": 1536, "uni2_h": 1536}}}

    with pytest.raises(ValueError, match="model.teacher_dims missing teacher entries"):
        teacher_dims(cfg, ["gigapath", "uni2_h", "virchow2"])
