from __future__ import annotations

import pytest

from hcc_sempath.training.config import teacher_dims, teacher_names, validate_training_config
from hcc_sempath.training.engine import scheduled_loss_config


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
        }
    }

    epoch_1 = scheduled_loss_config(cfg, epoch=1)
    epoch_4 = scheduled_loss_config(cfg, epoch=4)

    assert epoch_1["semantic_weight"] == pytest.approx(0.1)
    assert epoch_1["prototype_filter_weight"] == pytest.approx(0.4)
    assert epoch_1["prototype_filter_alpha_min"] == 0.3
    assert epoch_4["semantic_weight"] == pytest.approx(0.4)
    assert epoch_4["prototype_filter_weight"] == pytest.approx(0.8)


def test_teacher_names_reject_removed_h1_teacher() -> None:
    cfg = {"data": {"teachers": ["gigapath", "h_optimus_1"]}, "model": {"teacher_dims": {"gigapath": 1536, "h_optimus_1": 1536}}}

    with pytest.raises(ValueError, match="excluded teacher"):
        teacher_names(cfg)


def test_training_config_rejects_stale_teacher_entries() -> None:
    cfg = {
        "data": {
            "teachers": ["gigapath", "uni2_h", "virchow2"],
            "prototype_paths": {
                "gigapath": "g.pt",
                "uni2_h": "u.pt",
                "virchow2": "v.pt",
                "h_optimus_1": "removed.pt",
            },
        },
        "model": {
            "teacher_dims": {
                "gigapath": 1536,
                "uni2_h": 1536,
                "virchow2": 2560,
                "h_optimus_1": 1536,
            }
        },
        "loss": {
            "teacher_weights": {
                "gigapath": 1.0,
                "uni2_h": 1.0,
                "virchow2": 1.0,
                "h_optimus_1": 1.0,
            },
            "semantic_weight": 0.25,
            "prototype_filter_weight": 0.5,
        },
    }
    names = ["gigapath", "uni2_h", "virchow2"]

    with pytest.raises(ValueError, match="model.teacher_dims contains unknown teacher entries"):
        validate_training_config(cfg, names)


def test_teacher_dims_require_configured_teacher_entries() -> None:
    cfg = {"model": {"teacher_dims": {"gigapath": 1536, "uni2_h": 1536}}}

    with pytest.raises(ValueError, match="model.teacher_dims missing teacher entries"):
        teacher_dims(cfg, ["gigapath", "uni2_h", "virchow2"])
