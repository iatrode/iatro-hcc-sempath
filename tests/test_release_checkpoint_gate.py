from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


def _script_module(name: str):
    path = REPO / "research" / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("script", "gate"),
    [
        ("calibrate_spatial_decoder.py", "_finalized_checkpoint"),
        ("export_release_sempath.py", "_require_finalized_checkpoint"),
    ],
)
def test_joint_selection_terminal_last_checkpoint_is_rejected(
    script: str,
    gate: str,
) -> None:
    module = _script_module(script)
    payload = {
        "epoch": 16,
        "expected_epochs": 16,
        "training_complete": True,
        "config": {
            "data": {"require_complete_expert_validation": True},
            "train": {"epochs": 16, "selection_early_stop": True},
        },
    }

    with pytest.raises(ValueError, match="finalized joint-selection"):
        getattr(module, gate)(payload)


@pytest.mark.parametrize(
    ("script", "gate"),
    [
        ("calibrate_spatial_decoder.py", "_finalized_checkpoint"),
        ("export_release_sempath.py", "_require_finalized_checkpoint"),
    ],
)
def test_finalized_selected_checkpoint_is_accepted(
    script: str,
    gate: str,
) -> None:
    module = _script_module(script)
    cfg = {
        "data": {"require_complete_expert_validation": True},
        "train": {"epochs": 16, "selection_early_stop": True},
    }
    payload = {
        "epoch": 11,
        "expected_epochs": 16,
        "best_selection_epoch": 11,
        "run_complete": True,
        "selection_finalized": True,
        "config": cfg,
    }

    assert getattr(module, gate)(payload) is cfg
