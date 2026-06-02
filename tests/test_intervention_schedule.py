from __future__ import annotations

import pytest

from hcc_sempath.training.engine import (
    default_schedule_state,
    scheduled_loss_config,
    update_plateau_schedule_state,
)


def _cfg() -> dict:
    return {
        "loss": {
            "relation_weight": 0.05,
            "semantic_weight": 0.0,
            "semantic_temperature": 1.0,
            "zhcc_proto_weight": 0.2,
            "prototype_filter_weight": 0.5,
            "zhcc_response_weight": 0.0,
            "prototype_ramp_steps": 10,
            "filter_ramp_steps": 10,
            "prototype_filter_alpha_min": 0.25,
            "teacher_prior_plateau_window_steps": 1,
            "teacher_prior_plateau_threshold": 0.01,
            "teacher_prior_plateau_patience": 2,
            "teacher_prior_ema_beta": 0.0,
            "min_teacher_warmup_steps": 0,
            "max_teacher_warmup_steps": 100,
            "proto_to_filter_delay_steps": 20,
        }
    }


def test_plateau_schedule_ramps_after_triggered_steps() -> None:
    cfg = _cfg()
    state = default_schedule_state()
    state["prototype_start_step"] = 10
    state["filter_start_step"] = 30

    before = scheduled_loss_config(cfg, epoch=1, global_step=9, schedule_state=state)
    proto_mid = scheduled_loss_config(cfg, epoch=1, global_step=15, schedule_state=state)
    proto_done = scheduled_loss_config(cfg, epoch=1, global_step=20, schedule_state=state)
    filter_before = scheduled_loss_config(cfg, epoch=1, global_step=29, schedule_state=state)
    filter_mid = scheduled_loss_config(cfg, epoch=1, global_step=35, schedule_state=state)
    filter_done = scheduled_loss_config(cfg, epoch=1, global_step=40, schedule_state=state)

    assert before["zhcc_proto_weight"] == 0.0
    assert proto_mid["zhcc_proto_weight"] == pytest.approx(0.1)
    assert proto_done["zhcc_proto_weight"] == pytest.approx(0.2)
    assert filter_before["prototype_filter_weight"] == 0.0
    assert filter_mid["prototype_filter_weight"] == pytest.approx(0.25)
    assert filter_done["prototype_filter_weight"] == pytest.approx(0.5)
    assert filter_done["intervention_stage"] == "pamtd_active"


def test_plateau_gate_sets_intervention_after_patience() -> None:
    cfg = _cfg()
    state = default_schedule_state()

    update_plateau_schedule_state(cfg, state, global_step=1, teacher_prior_loss=1.0)
    update_plateau_schedule_state(cfg, state, global_step=2, teacher_prior_loss=0.995)
    assert state["prototype_start_step"] is None
    update_plateau_schedule_state(cfg, state, global_step=3, teacher_prior_loss=0.994)

    assert state["prototype_start_step"] == 3
    assert state["filter_start_step"] == 23
    assert state["teacher_prior_plateau_count"] == 2


def test_plateau_gate_forces_intervention_at_max_warmup() -> None:
    cfg = _cfg()
    cfg["loss"]["max_teacher_warmup_steps"] = 2
    state = default_schedule_state()

    update_plateau_schedule_state(cfg, state, global_step=2, teacher_prior_loss=1.0)

    assert state["prototype_start_step"] == 2
    assert state["filter_start_step"] == 22


def test_schedule_state_resume_preserves_triggered_steps() -> None:
    cfg = _cfg()
    state = default_schedule_state()
    state["prototype_start_step"] = 11
    state["filter_start_step"] = 21
    resumed_state = dict(state)

    loss_cfg = scheduled_loss_config(cfg, epoch=2, global_step=21, schedule_state=resumed_state)

    assert loss_cfg["prototype_start_step"] == 11
    assert loss_cfg["filter_start_step"] == 21
    assert loss_cfg["intervention_stage"] == "filter_ramp"
