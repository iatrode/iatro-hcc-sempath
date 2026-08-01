from pathlib import Path
import hashlib

import pytest
import torch
import yaml

from experiments.ablation.scripts.resolve_ablation_config import (
    _canonical_sha256,
    resolve_ablation_config,
    validate_ablation_resume_checkpoint,
)
from hcc_sempath.training.engine import _selection_start_step
from scripts.prepare_nvme_run import _prepare_full


CONFIG_ROOT = Path("experiments/ablation/configs")


def _local_base(tmp_path: Path) -> Path:
    teachers = [
        "gigapath",
        "h_optimus_1",
        "uni2_h",
        "virchow2",
    ]
    train_tile = str((tmp_path / "train.tiles.iac").resolve())
    val_tile = str((tmp_path / "val.tiles.iac").resolve())
    train_teacher = {
        teacher: [
            str(
                (
                    tmp_path
                    / f"train.{teacher}.features.iac"
                ).resolve()
            )
        ]
        for teacher in teachers
    }
    val_teacher = {
        teacher: [
            str(
                (
                    tmp_path
                    / f"val.{teacher}.features.iac"
                ).resolve()
            )
        ]
        for teacher in teachers
    }
    all_iac = {
        train_tile,
        val_tile,
        *(
            path
            for mapping in (train_teacher, val_teacher)
            for paths in mapping.values()
            for path in paths
        ),
    }
    payload = {
                "runtime": {"output_dir": "/runs/optuna/trial_0004", "seed": 13},
                "data": {
                    "train_tile_fraction": 0.1,
                    "val_tile_fraction": 0.1,
                    "require_complete_expert_validation": True,
                    "dynamic_package_sampling": True,
                    "package_multiprocessing": True,
                    "package_chunk_size": 64,
                    "package_buffer_batches": 4,
                    "teachers": teachers,
                    "train_image_tile_package_paths": [train_tile],
                    "val_image_tile_package_paths": [val_tile],
                    "train_teacher_feature_package_paths": train_teacher,
                    "val_teacher_feature_package_paths": val_teacher,
                    "prototype_paths": {
                        "gigapath": "/local/gigapath.pt",
                        "h_optimus_1": "/local/h1.pt",
                        "uni2_h": "/local/uni2.pt",
                        "virchow2": "/local/virchow2.pt",
                    },
                    "prototype_supervision_manifest_path": "/local/classification.csv",
                    "formal_study_contract_sha256": "a" * 64,
                    "formal_asset_sha256": {
                        "static_files": {
                            **{
                                f"prototype_{teacher}": "b" * 64
                                for teacher in teachers
                            },
                            "prototype_supervision_manifest_path": (
                                "c" * 64
                            ),
                        },
                        "iac_packages": {
                            path: "d" * 64
                            for path in all_iac
                        },
                        "student_pretrained": {
                            "path": "/local/dinov2.pth",
                            "sha256": "e" * 64,
                        },
                    },
                    "formal_source": {
                        "commit": "f" * 40,
                        "source_mode": "clean_git_commit",
                        "source_tree_sha256": "1" * 64,
                    },
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
                    "spatial_weight": 0.1,
                    "spatial_brush_top_fraction": 1.0,
                    "expert_supervision_start_step": 1000,
                    "expert_supervision_ramp_steps": 1000,
                    "prototype_filter_start_step": 2000,
                    "prototype_filter_ramp_steps": 1000,
                    "zhcc_response_start_step": 2000,
                    "zhcc_response_ramp_steps": 1000,
                },
                "train": {
                    "batch_size": 512,
                    "epochs": 16,
                    "lr": 0.00018,
                    "weight_decay": 0.005,
                    "lr_warmup_steps": 1000,
                    "max_val_batches": 1,
                    "max_eval_batches": 1,
                    "eval_pairwise_max_samples": 512,
                    "dynamic_prototype_refresh_steps": 500,
                    "dynamic_spatial_prototype_refresh_steps": 500,
                    "development_early_stop": False,
                    "selection_early_stop": True,
                    "selection_metric_weights": {
                        "teacher": 0.5,
                        "classification": 0.25,
                        "spatial": 0.25,
                    },
                    "selection_metric_baseline": {
                        "teacher": 0.8,
                        "classification": 1.2,
                        "spatial": 1.0,
                    },
                    "selection_early_stop_patience": 4,
                    "selection_early_stop_relative_delta": 0.005,
                    "selection_minimum_eligible_epochs": 5,
                    "early_stop_teacher_alignment": False,
                },
            }
    raw_config_sha256 = _canonical_sha256(payload)
    checkpoint = tmp_path / "a0_best.pt"
    checkpoint.write_bytes(b"selected-a0-checkpoint")
    payload["data"]["formal_a0_selection"] = {
        "study_complete": True,
        "study_name": "hcc_sempath_a0",
        "study_contract_sha256": "a" * 64,
        "total_trial_budget": 20,
        "executed_trial_records": 20,
        "selected_trial": 4,
        "selected_params": {
            "lr": 0.00018,
            "weight_decay": 0.005,
            "spatial_weight": 0.1,
        },
        "selection_loss": 0.72,
        "best_epoch": 9,
        "trial_config_sha256": raw_config_sha256,
        "best_checkpoint": str(checkpoint),
        "best_checkpoint_sha256": hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest(),
        "scheduler_contract": {
            "name": "cosine",
            "planned_total_steps": 20_576,
        },
    }
    path = tmp_path / "selected_a0.yaml"
    path.write_text(
        yaml.safe_dump(
            payload,
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


def test_full_run_translates_selection_to_fixed_global_steps(tmp_path) -> None:
    base = yaml.safe_load(_local_base(tmp_path).read_text())
    resolved = _prepare_full(base, tmp_path / "runs")

    assert resolved["data"]["train_tile_fraction"] == 1.0
    assert resolved["data"]["val_tile_fraction"] == 1.0
    assert resolved["train"]["selection_probe_interval_steps"] == 1000
    assert resolved["train"]["lr_total_steps"] == 20_576
    assert resolved["train"]["selection_minimum_eligible_probes"] == 8
    assert resolved["train"]["selection_early_stop_patience"] == 3
    assert resolved["train"]["development_probe_interval_steps"] == 0
    assert resolved["train"]["development_early_stop"] is False
    assert "selection_minimum_eligible_epochs" not in resolved["train"]


def test_condition_files_encode_only_the_prespecified_interventions() -> None:
    expected = {
        "a1": {
            "data": {"prototype_paths": None},
            "loss": {
                "classification_weight": 0.0,
                "semantic_weight": 0.0,
                "zhcc_response_weight": 0.0,
                "prototype_filter_weight": 0.0,
            },
        },
        "a2": {"loss": {"prototype_filter_weight": 0.0}},
        "a3": {
            "data": {"teachers": ["virchow2"], "prototype_paths": None},
            "model": {"teacher_dims": {"virchow2": 2560}},
            "loss": {
                "teacher_weights": {"virchow2": 1.0},
                "classification_weight": 0.0,
                "semantic_weight": 0.0,
                "zhcc_response_weight": 0.0,
                "prototype_filter_weight": 0.0,
            },
        },
        "a4": {
            "data": {
                "teachers": ["virchow2"],
                "prototype_paths": {
                    "virchow2": (
                        "artifacts/prototypes/"
                        "virchow2_hcc_semantic_prototypes.pt"
                    ),
                },
            },
            "model": {"teacher_dims": {"virchow2": 2560}},
            "loss": {"teacher_weights": {"virchow2": 1.0}},
        },
        "a5": {"train": {"dynamic_prototype_refresh_steps": 0}},
        "a6": {"train": {"dynamic_spatial_prototype_refresh_steps": 0}},
        "a7": {"loss": {"prototype_filter_weight": 1.0}},
        "a8": {"loss": {"spatial_detach_shared_encoder": True}},
        "a9": {"model": {"spatial_use_local_branch": False}},
        "a10": {"model": {"spatial_use_semantic_branch": False}},
        "a11": {"model": {"spatial_use_context": False}},
        "a12": {"loss": {"zhcc_response_weight": 0.0}},
    }
    conditions = sorted(CONFIG_ROOT.glob("a*.yaml"))
    assert len(conditions) == len(expected)
    for condition in conditions:
        condition_id = condition.name.split("_", 1)[0]
        payload = yaml.safe_load(condition.read_text(encoding="utf-8"))
        assert payload.pop("inherits") == "matched_tenth.yaml"
        runtime = payload.pop("runtime")
        assert set(runtime) == {"output_dir"}
        assert payload == expected[condition_id]


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
        assert resolved["train"]["epochs"] == 16
        assert resolved["train"]["development_early_stop"] is False
        assert resolved["train"]["selection_early_stop"] is True
        assert resolved["train"]["selection_metric_weights"] == {
            "teacher": 0.5,
            "classification": 0.25,
            "spatial": 0.25,
        }
        assert resolved["train"]["selection_early_stop_patience"] == 4
        assert resolved["train"]["batch_size"] == 512
        assert resolved["train"]["max_val_batches"] == 1
        assert resolved["train"]["max_eval_batches"] == 1
        assert resolved["train"]["eval_pairwise_max_samples"] == 512
        assert resolved["train"][
            "selection_early_stop_start_step"
        ] == 3000
        assert _selection_start_step(resolved) == 3000
        assert "selection_metric_baseline" not in resolved["train"]
        assert resolved["runtime"]["seed"] == 13
        assert resolved["train"]["lr"] == pytest.approx(0.00018)
        assert resolved["train"]["weight_decay"] == pytest.approx(0.005)
        assert resolved["loss"]["expert_supervision_start_step"] == 1000
        assert resolved["loss"]["prototype_filter_start_step"] == 2000
        assert resolved["loss"]["zhcc_response_start_step"] == 2000
        assert Path(resolved["runtime"]["output_dir"]).parent == tmp_path / "runs"


def test_no_global_expert_intervention_masks_labels_but_preserves_replay_population(
    tmp_path: Path,
) -> None:
    resolved = _resolve(tmp_path, "a1")

    assert resolved["data"]["expert_replay_prototype_manifest_path"] == (
        "/local/classification.csv"
    )
    assert resolved["data"]["prototype_paths"] is None
    assert resolved["loss"]["semantic_weight"] == 0.0
    assert resolved["loss"]["zhcc_response_weight"] == 0.0
    assert resolved["loss"]["prototype_filter_weight"] == 0.0
    assert resolved["data"]["prototype_supervision_manifest_path"] == (
        "/local/classification.csv"
    )
    assert resolved["loss"]["classification_weight"] == 0.0
    assert len(
        resolved["data"]["formal_asset_sha256"]["iac_packages"]
    ) == 10
    assert resolved["data"]["formal_source"][
        "source_tree_sha256"
    ] == "1" * 64
    assert "formal_study_contract_sha256" not in resolved["data"]
    assert resolved["data"][
        "ablation_parent_a0_study_contract_sha256"
    ] == "a" * 64
    assert len(
        resolved["data"]["formal_ablation_contract_sha256"]
    ) == 64
    assert "selection_metric_baseline" not in resolved["train"]


def test_single_teacher_pair_uses_the_same_local_teacher_asset(
    tmp_path: Path,
) -> None:
    without_prototype = _resolve(tmp_path, "a3")
    with_prototype = _resolve(tmp_path, "a4")

    for resolved in (without_prototype, with_prototype):
        assert resolved["data"]["teachers"] == ["virchow2"]
        assert resolved["model"]["teacher_dims"] == {"virchow2": 2560}
        assert resolved["loss"]["teacher_weights"] == {"virchow2": 1.0}
    assert without_prototype["loss"]["classification_weight"] == 0.0
    assert without_prototype["data"]["prototype_paths"] is None
    assert with_prototype["data"]["prototype_paths"] == {
        "virchow2": "/local/virchow2.pt"
    }
    assert len(
        without_prototype["data"]["formal_asset_sha256"][
            "iac_packages"
        ]
    ) == 4


def test_global_and_spatial_prototype_refresh_are_independent(
    tmp_path: Path,
) -> None:
    global_static = _resolve(tmp_path, "a5")
    spatial_static = _resolve(tmp_path, "a6")

    assert global_static["train"]["dynamic_prototype_refresh_steps"] == 0
    assert global_static["train"]["dynamic_spatial_prototype_refresh_steps"] == 500
    assert spatial_static["train"]["dynamic_prototype_refresh_steps"] == 500
    assert spatial_static["train"]["dynamic_spatial_prototype_refresh_steps"] == 0


def test_spatial_architecture_conditions_change_one_named_control(
    tmp_path: Path,
) -> None:
    semantic_only = _resolve(tmp_path, "a9")
    local_only = _resolve(tmp_path, "a10")
    no_context = _resolve(tmp_path, "a11")

    assert semantic_only["model"]["spatial_use_local_branch"] is False
    assert semantic_only["model"]["spatial_use_semantic_branch"] is True
    assert local_only["model"]["spatial_use_local_branch"] is True
    assert local_only["model"]["spatial_use_semantic_branch"] is False
    assert no_context["model"]["spatial_use_context"] is False


def test_no_student_response_changes_only_the_response_objective(
    tmp_path: Path,
) -> None:
    no_response = _resolve(tmp_path, "a12")

    assert no_response["loss"]["zhcc_response_weight"] == 0.0
    assert no_response["loss"]["prototype_filter_weight"] == pytest.approx(0.5)
    assert no_response["loss"]["semantic_weight"] == pytest.approx(0.02)
    assert no_response["data"]["prototype_paths"] is not None


@pytest.mark.parametrize(
    ("key_path", "value", "message"),
    [
        (("data", "train_tile_fraction"), 1.0, "train_tile_fraction=0.1"),
        (("data", "val_tile_fraction"), 1.0, "val_tile_fraction=0.1"),
        (("train", "epochs"), 5, "differs from the exported"),
        (
            ("train", "selection_early_stop"),
            False,
            "differs from the exported",
        ),
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


def test_ablation_cannot_change_the_teacher_probe_contract(
    tmp_path: Path,
) -> None:
    condition = tmp_path / "changed_probe.yaml"
    condition.write_text(
        yaml.safe_dump(
            {
                "inherits": str(
                    (CONFIG_ROOT / "matched_tenth.yaml").resolve()
                ),
                "runtime": {"output_dir": "outputs/changed_probe"},
                "train": {"max_eval_batches": 2},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="joint checkpoint-selection rule",
    ):
        resolve_ablation_config(
            _local_base(tmp_path),
            condition,
        )


def test_ablation_resume_rejects_a_different_original_epoch_plan(
    tmp_path: Path,
) -> None:
    resolved = _resolve(tmp_path, "a2")
    config_path = tmp_path / "resolved.yaml"
    config_path.write_text(
        yaml.safe_dump(resolved, sort_keys=False),
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "last.pt"
    torch.save({"expected_epochs": 6}, checkpoint_path)

    with pytest.raises(ValueError, match="different epoch plan"):
        validate_ablation_resume_checkpoint(
            config_path,
            checkpoint_path,
        )

    torch.save({"expected_epochs": 16}, checkpoint_path)
    validate_ablation_resume_checkpoint(
        config_path,
        checkpoint_path,
    )


def test_ablation_contract_binds_the_selected_trial_hyperparameters(
    tmp_path: Path,
) -> None:
    first_path = _local_base(tmp_path)
    first = resolve_ablation_config(
        first_path,
        CONFIG_ROOT / "a2_no_adjudication.yaml",
        output_root=tmp_path / "runs",
    )
    changed_payload = yaml.safe_load(
        first_path.read_text(encoding="utf-8")
    )
    selected = changed_payload["data"].pop("formal_a0_selection")
    changed_payload["train"]["lr"] = 0.00019
    selected["selected_params"]["lr"] = 0.00019
    selected["trial_config_sha256"] = _canonical_sha256(
        changed_payload
    )
    changed_payload["data"]["formal_a0_selection"] = selected
    changed_path = tmp_path / "selected_a0_changed.yaml"
    changed_path.write_text(
        yaml.safe_dump(changed_payload, sort_keys=False),
        encoding="utf-8",
    )
    second = resolve_ablation_config(
        changed_path,
        CONFIG_ROOT / "a2_no_adjudication.yaml",
        output_root=tmp_path / "runs",
    )

    assert first["train"]["lr"] != second["train"]["lr"]
    assert first["data"][
        "formal_ablation_contract_sha256"
    ] != second["data"]["formal_ablation_contract_sha256"]
