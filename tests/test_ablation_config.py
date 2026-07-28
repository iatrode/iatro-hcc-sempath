from pathlib import Path

import pytest
import yaml

from experiments.ablation.scripts.resolve_ablation_config import (
    resolve_ablation_config,
)


CONFIG_ROOT = Path("experiments/ablation/configs")


def _local_base(tmp_path: Path) -> Path:
    path = tmp_path / "selected_a0.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "runtime": {"output_dir": "/runs/optuna/trial_0004", "seed": 13},
                "data": {
                    "train_tile_fraction": 0.1,
                    "val_tile_fraction": 0.1,
                    "teachers": [
                        "gigapath",
                        "h_optimus_1",
                        "uni2_h",
                        "virchow2",
                    ],
                    "prototype_paths": {
                        "gigapath": "/local/gigapath.pt",
                        "h_optimus_1": "/local/h1.pt",
                        "uni2_h": "/local/uni2.pt",
                        "virchow2": "/local/virchow2.pt",
                    },
                    "prototype_supervision_manifest_path": "/local/classification.csv",
                },
                "model": {
                    "teacher_dims": {
                        "gigapath": 1536,
                        "h_optimus_1": 1536,
                        "uni2_h": 1536,
                        "virchow2": 2560,
                    },
                    "spatial_use_local_branch": True,
                    "spatial_use_semantic_branch": True,
                    "spatial_use_context": True,
                },
                "loss": {
                    "teacher_weights": {
                        "gigapath": 1.0,
                        "h_optimus_1": 1.0,
                        "uni2_h": 1.0,
                        "virchow2": 1.0,
                    },
                    "semantic_weight": 0.02,
                    "prototype_filter_weight": 0.5,
                    "zhcc_response_weight": 0.15,
                    "spatial_brush_top_fraction": 0.25,
                    "expert_supervision_start_step": 1000,
                    "expert_supervision_ramp_steps": 1000,
                    "prototype_filter_start_step": 2000,
                    "prototype_filter_ramp_steps": 1000,
                    "zhcc_response_start_step": 2000,
                    "zhcc_response_ramp_steps": 1000,
                },
                "train": {
                    "epochs": 3,
                    "lr": 0.00018,
                    "weight_decay": 0.005,
                    "lr_warmup_steps": 1000,
                    "max_val_batches": 1,
                    "max_eval_batches": 1,
                    "dynamic_prototype_refresh_steps": 500,
                    "dynamic_spatial_prototype_refresh_steps": 500,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _resolve(tmp_path: Path, name: str) -> dict:
    condition = next(CONFIG_ROOT.glob(f"{name}_*.yaml"))
    return resolve_ablation_config(
        _local_base(tmp_path),
        condition,
        output_root=tmp_path / "runs",
    )


def test_all_conditions_reuse_the_selected_a0_budget_and_schedule(
    tmp_path: Path,
) -> None:
    conditions = sorted(CONFIG_ROOT.glob("a[1-9]_*.yaml")) + sorted(
        CONFIG_ROOT.glob("a1[0-2]_*.yaml")
    )
    assert len(conditions) == 12

    for condition in conditions:
        resolved = resolve_ablation_config(
            _local_base(tmp_path),
            condition,
            output_root=tmp_path / "runs",
        )
        assert resolved["data"]["train_tile_fraction"] == pytest.approx(0.1)
        assert resolved["data"]["val_tile_fraction"] == pytest.approx(0.1)
        assert resolved["train"]["epochs"] == 6
        assert resolved["train"]["development_probe_interval_steps"] == 1000
        assert resolved["train"]["development_early_stop"] is True
        assert resolved["train"]["development_early_stop_min_step"] == 4000
        assert resolved["train"]["development_early_stop_relative_delta"] == pytest.approx(0.005)
        assert resolved["train"]["development_early_stop_patience"] == 2
        assert resolved["runtime"]["seed"] == 13
        assert resolved["train"]["lr"] == pytest.approx(0.00018)
        assert resolved["train"]["weight_decay"] == pytest.approx(0.005)
        assert resolved["loss"]["expert_supervision_start_step"] == 1000
        assert resolved["loss"]["prototype_filter_start_step"] == 2000
        assert resolved["loss"]["zhcc_response_start_step"] == 2000
        assert Path(resolved["runtime"]["output_dir"]).parent == tmp_path / "runs"


def test_no_prototype_masks_classification_labels_but_preserves_replay_population(
    tmp_path: Path,
) -> None:
    resolved = _resolve(tmp_path, "a1")

    assert resolved["data"]["prototype_supervision_manifest_path"] is None
    assert resolved["data"]["expert_replay_prototype_manifest_path"] == "/local/classification.csv"
    assert resolved["data"]["prototype_paths"] is None
    assert resolved["loss"]["semantic_weight"] == 0.0
    assert resolved["loss"]["zhcc_response_weight"] == 0.0
    assert resolved["loss"]["prototype_filter_weight"] == 0.0


def test_single_teacher_pair_uses_the_same_local_teacher_asset(
    tmp_path: Path,
) -> None:
    without_prototype = _resolve(tmp_path, "a3")
    with_prototype = _resolve(tmp_path, "a4")

    for resolved in (without_prototype, with_prototype):
        assert resolved["data"]["teachers"] == ["virchow2"]
        assert resolved["model"]["teacher_dims"] == {"virchow2": 2560}
        assert resolved["loss"]["teacher_weights"] == {"virchow2": 1.0}
    assert without_prototype["data"]["prototype_paths"] is None
    assert with_prototype["data"]["prototype_paths"] == {
        "virchow2": "/local/virchow2.pt"
    }


def test_global_and_spatial_prototype_refresh_are_independent(
    tmp_path: Path,
) -> None:
    global_static = _resolve(tmp_path, "a5")
    spatial_static = _resolve(tmp_path, "a6")

    assert global_static["train"]["dynamic_prototype_refresh_steps"] == 0
    assert global_static["train"]["dynamic_spatial_prototype_refresh_steps"] == 500
    assert spatial_static["train"]["dynamic_prototype_refresh_steps"] == 500
    assert spatial_static["train"]["dynamic_spatial_prototype_refresh_steps"] == 0


def test_spatial_architecture_and_target_conditions_change_one_named_control(
    tmp_path: Path,
) -> None:
    semantic_only = _resolve(tmp_path, "a9")
    local_only = _resolve(tmp_path, "a10")
    no_context = _resolve(tmp_path, "a11")
    dense_brush = _resolve(tmp_path, "a12")

    assert semantic_only["model"]["spatial_use_local_branch"] is False
    assert semantic_only["model"]["spatial_use_semantic_branch"] is True
    assert local_only["model"]["spatial_use_local_branch"] is True
    assert local_only["model"]["spatial_use_semantic_branch"] is False
    assert no_context["model"]["spatial_use_context"] is False
    assert dense_brush["loss"]["spatial_brush_top_fraction"] == 1.0


@pytest.mark.parametrize(
    ("key_path", "value", "message"),
    [
        (("data", "train_tile_fraction"), 1.0, "train_tile_fraction=0.1"),
        (("data", "val_tile_fraction"), 1.0, "val_tile_fraction=0.1"),
        (("train", "epochs"), 4, "three-epoch"),
    ],
)
def test_resolver_rejects_a_nonmatching_a0_base(
    tmp_path: Path,
    key_path: tuple[str, str],
    value: float,
    message: str,
) -> None:
    base = yaml.safe_load(_local_base(tmp_path).read_text(encoding="utf-8"))
    base[key_path[0]][key_path[1]] = value
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(base), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        resolve_ablation_config(
            invalid,
            CONFIG_ROOT / "a2_no_adjudication.yaml",
        )
