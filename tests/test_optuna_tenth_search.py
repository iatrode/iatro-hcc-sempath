from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


optuna = pytest.importorskip("optuna")


def _search_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "optuna_tenth_search.py"
    spec = importlib.util.spec_from_file_location("optuna_tenth_search", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trial_config_uses_matched_tenth_population_and_fixed_losses(
    tmp_path: Path,
) -> None:
    module = _search_module()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=13),
    )
    trial = study.ask(
        fixed_distributions={
            name: optuna.distributions.FloatDistribution(**distribution)
            for name, distribution in module.SEARCH_SPACE.items()
        }
    )
    losses = {
        "prototype_filter_weight": 0.5,
        "classification_weight": 1.0,
        "spatial_weight": 0.1,
    }
    cfg = module.trial_config(
        {
            "runtime": {"seed": 13},
            "data": {},
            "loss": losses,
            "train": {"batch_size": 512},
        },
        trial,
        tmp_path / "trial",
        epochs=3,
    )

    assert cfg["data"]["train_tile_fraction"] == pytest.approx(0.1)
    assert cfg["data"]["val_tile_fraction"] == pytest.approx(0.1)
    assert cfg["loss"] == losses
    assert cfg["train"]["epochs"] == 3
    assert set(trial.params) == {"lr", "weight_decay"}


def test_train_loss_objective_is_strict_and_directionally_correct() -> None:
    module = _search_module()

    assert module.score_row({"train_loss": "1.25"}, "train_loss") == -1.25
    with pytest.raises(ValueError, match="non-finite objective metric"):
        module.score_row({}, "train_loss")


def test_atomic_yaml_write_leaves_no_worker_temporary_file(
    tmp_path: Path,
) -> None:
    module = _search_module()
    output = tmp_path / "study_manifest.yaml"

    module.atomic_write_yaml(output, {"n_trials_requested": 5})

    assert module.load_yaml(output) == {"n_trials_requested": 5}
    assert list(tmp_path.glob(".*.tmp")) == []
