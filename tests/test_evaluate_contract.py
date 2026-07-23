from __future__ import annotations

import copy

import pytest

from hcc_sempath.training.evaluate import (
    _explicit_split_data_paths,
    _use_checkpoint_config,
)


def _config() -> dict:
    return {
        "runtime": {
            "device": "cuda",
            "output_dir": "run",
            "seed": 13,
        },
        "data": {
            "teachers": ["teacher"],
            "train_manifest_path": "manifest.yaml",
            "prototype_paths": {"teacher": "prototype.pt"},
            "mean": [0.1, 0.2, 0.3],
            "std": [0.4, 0.5, 0.6],
            "num_workers": 8,
            "prefetch_factor": 2,
        },
        "model": {
            "embedding_dim": 4,
            "teacher_dims": {"teacher": 4},
        },
        "loss": {
            "teacher_weights": {"teacher": 1.0},
            "semantic_weight": 0.0,
        },
        "train": {
            "batch_size": 2,
            "epochs": 3,
            "log_interval": 10,
            "tensorboard": True,
        },
    }


def test_evaluate_uses_checkpoint_semantics_and_host_overrides_only() -> None:
    saved = _config()
    requested = copy.deepcopy(saved)
    requested["runtime"]["device"] = "cpu"
    requested["runtime"]["output_dir"] = "eval"
    requested["data"]["num_workers"] = 0
    requested["data"]["prefetch_factor"] = 1
    requested["train"]["log_interval"] = 0
    requested["train"]["tensorboard"] = False

    resolved = _use_checkpoint_config(
        requested,
        {"config": saved},
    )

    assert resolved["runtime"]["device"] == "cpu"
    assert resolved["runtime"]["output_dir"] == "eval"
    assert resolved["data"]["num_workers"] == 0
    assert resolved["data"]["mean"] == saved["data"]["mean"]


def test_evaluate_rejects_preprocessing_or_data_contract_change() -> None:
    saved = _config()
    for section, key, value in (
        ("data", "mean", [0.2, 0.2, 0.3]),
        ("data", "train_manifest_path", "other.yaml"),
        ("model", "embedding_dim", 8),
    ):
        requested = copy.deepcopy(saved)
        requested[section][key] = value
        with pytest.raises(ValueError, match="differs"):
            _use_checkpoint_config(
                requested,
                {"config": saved},
            )


def test_evaluate_resolves_explicit_split_packages() -> None:
    cfg = _config()
    cfg["data"].update(
        {
            "val_image_tile_package_paths": ["val-a.iac", "val-b.iac"],
            "val_teacher_feature_package_paths": {
                "teacher": ["val-a.features.iac", "val-b.features.iac"],
            },
        }
    )

    tiles, teachers = _explicit_split_data_paths(cfg, "val")

    assert tiles == ["val-a.iac", "val-b.iac"]
    assert teachers == {
        "teacher": ["val-a.features.iac", "val-b.features.iac"]
    }


def test_evaluate_rejects_half_configured_explicit_split() -> None:
    cfg = _config()
    cfg["data"]["val_image_tile_package_paths"] = ["val.iac"]

    with pytest.raises(ValueError, match="configured together"):
        _explicit_split_data_paths(cfg, "val")
