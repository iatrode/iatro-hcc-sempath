from __future__ import annotations

import pytest

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
