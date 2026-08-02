from __future__ import annotations

import csv
import importlib.util
import io
import json
from pathlib import Path

import pytest
import torch


optuna = pytest.importorskip("optuna")


def _search_module():
    path = Path(__file__).resolve().parents[1] / "research" / "scripts" / "optuna_a0_search.py"
    spec = importlib.util.spec_from_file_location("optuna_a0_search", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_selection_row() -> dict[str, str]:
    return {
        "epoch": "12",
        "selection_eligible": "true",
        "global_step": "2500",
        "selection_start_step": "2500",
        "selection_loss": "0.5",
        "selection_teacher_raw": "0.4",
        "selection_teacher_baseline": "0.8",
        "selection_teacher_normalized": "0.5",
        "selection_teacher_weight": "0.5",
        "selection_classification_raw": "0.6",
        "selection_classification_baseline": "1.2",
        "selection_classification_normalized": "0.5",
        "selection_classification_weight": "0.25",
        "selection_spatial_raw": "0.5",
        "selection_spatial_baseline": "1.0",
        "selection_spatial_normalized": "0.5",
        "selection_spatial_weight": "0.25",
        "teacher_validation_loss": "0.4",
        "expert_val_spatial": "0.5",
        "expert_val_classification_balanced_cross_entropy": "0.6",
        "expert_val_classification_evaluated_classes": "7",
        "expert_val_classification_total_classes": "7",
        "expert_val_spatial_explicit_negative_pairs": "4",
    }


def test_trial_config_uses_matched_tenth_population_and_fixed_losses(
    tmp_path: Path,
) -> None:
    module = _search_module()
    study = optuna.create_study(
        direction="minimize",
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
        selection_baseline={
            "teacher": 0.8,
            "classification": 1.2,
            "spatial": 1.0,
        },
        formal_asset_sha256={
            "static_files": {"spatial": "a" * 64}
        },
        formal_source={
            "commit": "b" * 40,
            "source_mode": "clean_git_commit",
            "source_tree_sha256": "c" * 64,
        },
        formal_study_contract_sha256="d" * 64,
        formal_population_validation={
            "selected_val_packages": 29,
            "probe_definition_sha256": "e" * 64,
        },
    )

    assert cfg["data"]["train_tile_fraction"] == pytest.approx(0.1)
    assert cfg["data"]["val_tile_fraction"] == pytest.approx(0.1)
    assert cfg["loss"]["prototype_filter_weight"] == 0.5
    assert cfg["loss"]["classification_weight"] == 1.0
    assert cfg["loss"]["spatial_weight"] == trial.params["spatial_weight"]
    assert cfg["train"]["epochs"] == 3
    assert cfg["runtime"]["seed"] == 13
    assert cfg["train"]["development_early_stop"] is False
    assert cfg["train"]["development_probe_interval_steps"] == 0
    assert cfg["train"]["gradient_diagnostic_interval_steps"] == 0
    assert cfg["data"]["persistent_workers"] is False
    assert cfg["data"]["val_persistent_workers"] is False
    assert cfg["train"]["selection_early_stop"] is True
    assert cfg["train"]["selection_metric_weights"] == {
        "teacher": 0.26,
        "classification": 0.28,
        "spatial": 0.46,
    }
    assert cfg["data"]["require_complete_expert_validation"] is True
    assert cfg["train"]["selection_metric_baseline"] == {
        "teacher": 0.8,
        "classification": 1.2,
        "spatial": 1.0,
    }
    assert cfg["data"]["formal_asset_sha256"] == {
        "static_files": {"spatial": "a" * 64}
    }
    assert cfg["data"]["formal_source"]["source_tree_sha256"] == "c" * 64
    assert cfg["data"]["formal_study_contract_sha256"] == "d" * 64
    assert cfg["data"]["formal_population_validation"] == {
        "selected_val_packages": 29,
        "probe_definition_sha256": "e" * 64,
    }
    assert set(trial.params) == {
        "lr",
        "weight_decay",
        "spatial_weight",
    }


def test_formal_preflight_binds_the_same_tenth_view_as_trials() -> None:
    module = _search_module()

    cfg = module._formal_base_config(
        {
            "runtime": {"seed": 13},
            "data": {
                "train_tile_fraction": 1.0,
                "val_tile_fraction": 1.0,
            },
            "train": {"epochs": 10},
        },
        epochs=16,
    )

    assert cfg["data"]["train_tile_fraction"] == pytest.approx(0.10)
    assert cfg["data"]["val_tile_fraction"] == pytest.approx(0.10)
    assert cfg["train"]["epochs"] == 16


def test_population_validation_contract_binds_fixed_teacher_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _search_module()
    package = tmp_path / "val.tiles.iac"
    package.touch()
    monkeypatch.setattr(
        "iatro.iac.read_header",
        lambda path: {"num_records": 200_000},
    )

    contract = module._population_validation_contract(
        {
            "data": {},
            "train": {
                "batch_size": 512,
                "max_val_batches": 256,
                "max_eval_batches": 128,
            },
        },
        [package],
        expert_tiles=3_095,
    )

    assert contract["selected_val_records"] == 200_000
    assert contract["population_val_records_lower_bound"] == 196_905
    assert contract["ordinary_validation_batches"] == 256
    assert contract["ordinary_validation_tiles"] == 131_072
    assert contract["teacher_retention_batches"] == 128
    assert contract["teacher_retention_tiles"] == 65_536
    assert contract["teacher_relation_tiles"] == 4_096


def test_population_validation_contract_rejects_hidden_probe_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _search_module()
    package = tmp_path / "val.tiles.iac"
    package.touch()
    monkeypatch.setattr(
        "iatro.iac.read_header",
        lambda path: {"num_records": 200_000},
    )

    with pytest.raises(
        ValueError,
        match="max_eval_batches cannot exceed max_val_batches",
    ):
        module._population_validation_contract(
            {
                "data": {},
                "train": {
                    "batch_size": 512,
                    "max_val_batches": 64,
                    "max_eval_batches": 128,
                },
            },
            [package],
            expert_tiles=0,
        )


def test_preflight_rejects_cross_modality_train_val_overlap(
    tmp_path: Path,
) -> None:
    module = _search_module()
    classification = tmp_path / "classification.csv"
    classification.write_text(
        "tile_id,source_split,adjudicated\n"
        "shared,train,true\n",
        encoding="utf-8",
    )
    spatial = tmp_path / "spatial.json"
    spatial.write_text(
        json.dumps(
            {
                "annotations": {
                    "shared": {
                        "tile_id": "shared",
                        "split": "val",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="expert tile overlap across L1/L2",
    ):
        module._expert_split_tile_counts(classification, spatial)


def test_preflight_explicit_packages_take_precedence_over_manifest(
    tmp_path: Path,
) -> None:
    module = _search_module()
    teachers = list(module.TEACHERS)
    cfg = {
        "data": {
            "teachers": teachers,
            "train_manifest_path": str(tmp_path / "must_not_be_read.yaml"),
            "train_image_tile_package_paths": [
                str(tmp_path / "train.tiles.iac")
            ],
            "val_image_tile_package_paths": [
                str(tmp_path / "val.tiles.iac")
            ],
            "train_teacher_feature_package_paths": {
                teacher: [str(tmp_path / f"train.{teacher}.iac")]
                for teacher in teachers
            },
            "val_teacher_feature_package_paths": {
                teacher: [str(tmp_path / f"val.{teacher}.iac")]
                for teacher in teachers
            },
        }
    }

    tiles, feature_paths = module._resolved_training_iac_paths(
        cfg,
        complete=True,
    )

    assert tiles == [
        (tmp_path / "train.tiles.iac").resolve(),
        (tmp_path / "val.tiles.iac").resolve(),
    ]
    assert all(len(feature_paths[teacher]) == 2 for teacher in teachers)

    train_tiles, train_feature_paths = (
        module._resolved_training_iac_paths(
            cfg,
            complete=True,
            splits=("train",),
        )
    )
    assert train_tiles == [(tmp_path / "train.tiles.iac").resolve()]
    assert all(
        paths == [(tmp_path / f"train.{teacher}.iac").resolve()]
        for teacher, paths in train_feature_paths.items()
    )


def test_selection_loss_objective_is_strict_and_directionally_correct() -> None:
    module = _search_module()
    row = {
        "selection_eligible": "true",
        "global_step": "2500",
        "selection_start_step": "2500",
        "selection_loss": "0.5",
        "selection_teacher_raw": "0.4",
        "selection_teacher_baseline": "0.8",
        "selection_teacher_normalized": "0.5",
        "selection_teacher_weight": "0.5",
        "selection_classification_raw": "0.6",
        "selection_classification_baseline": "1.2",
        "selection_classification_normalized": "0.5",
        "selection_classification_weight": "0.25",
        "selection_spatial_raw": "0.5",
        "selection_spatial_baseline": "1.0",
        "selection_spatial_normalized": "0.5",
        "selection_spatial_weight": "0.25",
        "teacher_validation_loss": "0.4",
        "expert_val_spatial": "0.5",
        "expert_val_classification_balanced_cross_entropy": "0.6",
        "expert_val_classification_evaluated_classes": "7",
        "expert_val_classification_total_classes": "7",
        "expert_val_spatial_explicit_negative_pairs": "4",
    }
    expected_weights = {
        "teacher": 0.5,
        "classification": 0.25,
        "spatial": 0.25,
    }
    expected_baseline = {
        "teacher": 0.8,
        "classification": 1.2,
        "spatial": 1.0,
    }
    assert module.score_row(
        row,
        expected_weights=expected_weights,
        expected_baseline=expected_baseline,
        expected_start_step=2500,
    ) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="precedes"):
        module.score_row({})
    with pytest.raises(ValueError, match="does not equal"):
        module.score_row({**row, "selection_loss": "0.49"})
    with pytest.raises(ValueError, match="normalization mismatch"):
        module.score_row(
            {
                **row,
                "selection_teacher_normalized": "0.4",
                "selection_loss": "0.45",
            }
        )
    with pytest.raises(ValueError, match="before its start step"):
        module.score_row(
            {
                **row,
                "global_step": "2499",
            }
        )
    with pytest.raises(ValueError, match="shared study baseline"):
        module.score_row(
            row,
            expected_baseline={
                **expected_baseline,
                "spatial": 2.0,
            },
        )


def test_metric_polling_ignores_a_concurrent_partial_csv_row(
    tmp_path: Path,
) -> None:
    module = _search_module()
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "epoch,selection_loss\n"
        "1,0.9\n"
        "2,0.",
        encoding="utf-8",
    )

    assert module.read_metric_rows(metrics) == [
        {"epoch": "1", "selection_loss": "0.9"}
    ]


def test_training_exit_drains_final_eligible_metric(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _search_module()
    output = tmp_path / "trial"
    (output / "checkpoints").mkdir(parents=True)
    row = _valid_selection_row()
    with (output / "metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    torch.save(
        {
            "epoch": 12,
            "run_complete": True,
            "selection_finalized": True,
            "best_selection_loss": 0.5,
        },
        output / "checkpoints" / "best.pt",
    )

    class FinishedProcess:
        returncode = 0
        stdout = io.StringIO("")

        def poll(self):
            return 0

    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *args, **kwargs: FinishedProcess(),
    )
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    score = module.train_with_pruning(
        trial=trial,
        cfg_path=tmp_path / "config.yaml",
        output_dir=output,
        python_bin="python",
        repo=tmp_path,
        poll_sec=20.0,
        expected_weights={
            "teacher": 0.5,
            "classification": 0.25,
            "spatial": 0.25,
        },
        expected_baseline={
            "teacher": 0.8,
            "classification": 1.2,
            "spatial": 1.0,
        },
        expected_start_step=2500,
    )
    study.tell(trial, score)

    assert study.trials[0].intermediate_values == {1: 0.5}
    assert study.trials[0].user_attrs[
        "eligible_step_1_epoch"
    ] == 12


def test_atomic_yaml_write_leaves_no_worker_temporary_file(
    tmp_path: Path,
) -> None:
    module = _search_module()
    output = tmp_path / "study_manifest.yaml"

    module.atomic_write_yaml(output, {"n_trials_requested": 5})

    assert module.load_yaml(output) == {"n_trials_requested": 5}
    assert list(tmp_path.glob(".*.tmp")) == []


def test_parallel_trial_devices_require_one_distinct_gpu_per_worker() -> None:
    module = _search_module()

    assert module.parse_cuda_devices(
        "0, 1,2,3",
        parallel_trials=4,
    ) == ("0", "1", "2", "3")
    assert module.parse_cuda_devices(
        "",
        parallel_trials=1,
    ) == (None,)
    with pytest.raises(ValueError, match="required"):
        module.parse_cuda_devices("", parallel_trials=4)
    with pytest.raises(ValueError, match="duplicates"):
        module.parse_cuda_devices("0,1,1,2", parallel_trials=4)
    with pytest.raises(ValueError, match="at least one"):
        module.parse_cuda_devices("0,1", parallel_trials=4)


def test_verified_preflight_reuses_only_matching_frozen_assets(
    tmp_path: Path,
) -> None:
    module = _search_module()
    source = {
        "commit": "a" * 40,
        "source_mode": "declared_archive",
        "source_tree_sha256": "b" * 64,
    }
    assets = {
        "formal_asset_sha256": {"static_files": {}},
        "population_schedule": {"selection_start_step": 2500},
        "population_validation": {"selected_val_packages": 3},
    }
    manifest = tmp_path / "preflight.yaml"
    module.write_yaml(
        manifest,
        {
            "source": source,
            "base_config_sha256": "c" * 64,
            "assets": assets,
        },
    )

    assert module.load_verified_preflight_assets(
        manifest,
        source=source,
        base_config_sha256="c" * 64,
    ) == assets
    with pytest.raises(RuntimeError, match="source differs"):
        module.load_verified_preflight_assets(
            manifest,
            source={**source, "commit": "d" * 40},
            base_config_sha256="c" * 64,
        )
    with pytest.raises(RuntimeError, match="base config differs"):
        module.load_verified_preflight_assets(
            manifest,
            source=source,
            base_config_sha256="e" * 64,
        )


def test_verified_iac_receipt_records_stat_guard(
    tmp_path: Path,
) -> None:
    module = _search_module()
    package = tmp_path / "population.iac"
    package.write_bytes(b"frozen")
    receipt_path = module.write_verified_iac_receipt(
        tmp_path / "receipt.json",
        {
            "iac_packages": {
                str(package.resolve()): "a" * 64,
            }
        },
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    entry = receipt["files"][str(package.resolve())]
    stat = package.stat()
    assert entry == {
        "sha256": "a" * 64,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def test_training_process_is_bound_to_requested_cuda_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _search_module()
    output = tmp_path / "trial"
    (output / "checkpoints").mkdir(parents=True)
    metrics = output / "metrics.csv"
    row = _valid_selection_row()
    row["epoch"] = "12"
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    torch.save(
        {
            "epoch": 12,
            "run_complete": True,
            "selection_finalized": True,
            "best_selection_loss": 0.5,
        },
        output / "checkpoints" / "best.pt",
    )

    captured_env = {}

    class FinishedProcess:
        returncode = 0
        stdout = io.StringIO("")

        def poll(self):
            return 0

    def fake_popen(*args, **kwargs):
        del args
        captured_env.update(kwargs["env"])
        return FinishedProcess()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    module.train_with_pruning(
        trial=trial,
        cfg_path=tmp_path / "config.yaml",
        output_dir=output,
        python_bin="python",
        repo=tmp_path,
        poll_sec=20.0,
        expected_weights={
            "teacher": 0.5,
            "classification": 0.25,
            "spatial": 0.25,
        },
        expected_baseline={
            "teacher": 0.8,
            "classification": 1.2,
            "spatial": 1.0,
        },
        expected_start_step=2500,
        cuda_visible_device="3",
    )

    assert captured_env["CUDA_VISIBLE_DEVICES"] == "3"


def test_declared_archive_requires_a_real_commit_shape(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _search_module()
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='test'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HCC_SEMPATH_SOURCE_COMMIT", "not-a-commit")

    with pytest.raises(RuntimeError, match="40-character"):
        module.source_state(tmp_path)


def test_study_baseline_is_bound_once_and_reused() -> None:
    module = _search_module()
    study = optuna.create_study(direction="minimize")
    baseline = {
        "teacher": 0.8,
        "classification": 1.2,
        "spatial": 1.0,
    }

    digest = module._bind_study_selection_baseline(study, baseline)

    assert module._study_selection_baseline(study) == baseline
    assert digest == study.user_attrs[
        "selection_metric_baseline_sha256"
    ]
    with pytest.raises(RuntimeError, match="differs"):
        module._bind_study_selection_baseline(
            study,
            {**baseline, "spatial": 2.0},
        )


def test_global_budget_counts_waiting_trials_as_future_executions() -> None:
    module = _search_module()
    study = optuna.create_study(direction="minimize")
    study.enqueue_trial(module.SEEDED_PARAMS[0])
    study.enqueue_trial(module.SEEDED_PARAMS[1])
    assert module._remaining_study_executions(study, 5) == 5

    trial = study.ask()
    study.tell(trial, 1.0)
    assert module._remaining_study_executions(study, 5) == 4


def test_sqlite_coordinator_lock_is_shared_across_output_roots(
    tmp_path: Path,
) -> None:
    module = _search_module()
    storage = module._normalize_storage(
        "sqlite:///runtime/a0.db",
        tmp_path,
    )

    first = module._coordinator_lock_path(
        storage,
        output_root=tmp_path / "first",
        study_name="formal/a0",
    )
    second = module._coordinator_lock_path(
        storage,
        output_root=tmp_path / "second",
        study_name="formal/a0",
    )

    assert first == second
    assert first.name == ".a0.db.formal_a0.coordinator.lock"


def test_trial_seeded_tpe_is_restart_reproducible(
    tmp_path: Path,
) -> None:
    module = _search_module()

    def sampler():
        return module.TrialSeededTPESampler(
            seed=13,
            n_startup_trials=2,
        )

    def objective(trial):
        first = trial.suggest_float("first", 1e-4, 1e-2, log=True)
        second = trial.suggest_float("second", 0.05, 1.0, log=True)
        return first + second

    continuous = optuna.create_study(
        study_name="continuous",
        storage=f"sqlite:///{tmp_path / 'continuous.db'}",
        direction="minimize",
        sampler=sampler(),
    )
    continuous.optimize(objective, n_trials=6)

    storage = f"sqlite:///{tmp_path / 'split.db'}"
    split = optuna.create_study(
        study_name="split",
        storage=storage,
        direction="minimize",
        sampler=sampler(),
    )
    split.optimize(objective, n_trials=2)
    resumed = optuna.create_study(
        study_name="split",
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=sampler(),
    )
    resumed.optimize(objective, n_trials=4)

    assert [trial.params for trial in continuous.trials] == [
        trial.params for trial in resumed.trials
    ]


@pytest.mark.parametrize(
    "state",
    [
        optuna.trial.TrialState.RUNNING,
        optuna.trial.TrialState.FAIL,
    ],
)
def test_formal_study_rejects_incomplete_trial_history(state) -> None:
    module = _search_module()
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    if state == optuna.trial.TrialState.FAIL:
        study.tell(trial, state=state)

    with pytest.raises(RuntimeError, match="cannot be resumed"):
        module._assert_resumable_study_states(study)


def test_selection_baseline_artifact_requires_all_three_terms(
    tmp_path: Path,
) -> None:
    module = _search_module()
    path = tmp_path / "selection_baseline.json"
    path.write_text(
        json.dumps(
            {
                "metrics": {
                    "teacher": 0.8,
                    "classification": 1.2,
                    "spatial": 1.0,
                }
            }
        ),
        encoding="utf-8",
    )
    assert module._selection_baseline_from_path(path) == {
        "teacher": 0.8,
        "classification": 1.2,
        "spatial": 1.0,
    }


def test_final_export_binds_selected_trial_config_and_checkpoint(
    tmp_path: Path,
) -> None:
    module = _search_module()
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    trial.suggest_float("lr", 1.5e-4, 1.5e-4)
    trial.suggest_float("weight_decay", 0.003, 0.003)
    trial.suggest_float("spatial_weight", 0.2, 0.2)
    trial_dir = tmp_path / "trial_0000"
    trial_dir.mkdir()
    config = {
        "runtime": {"seed": 13},
        "data": {
            "formal_study_contract_sha256": "a" * 64,
        },
        "train": {
            "lr": 1.5e-4,
            "weight_decay": 0.003,
        },
        "loss": {"spatial_weight": 0.2},
    }
    module.write_yaml(trial_dir / "config.yaml", config)
    checkpoint = trial_dir / "best.pt"
    torch.save(
        {
            "epoch": 7,
            "scheduler_contract": {
                "name": "cosine",
                "planned_total_steps": 20_576,
            },
        },
        checkpoint,
    )
    trial.set_user_attr("output_dir", str(trial_dir))
    trial.set_user_attr("best_checkpoint", str(checkpoint))
    trial.set_user_attr("best_epoch", 7)
    trial.set_user_attr(
        "config_sha256",
        module.config_digest(config),
    )
    study.tell(trial, 0.72)

    module.export_study_artifacts(
        study,
        output_root=tmp_path,
        manifest={
            "study_name": "hcc_sempath_a0",
            "study_contract_sha256": "a" * 64,
            "n_trials_requested": 1,
        },
        hash_best_checkpoint=True,
    )

    exported = module.load_yaml(tmp_path / "best_config.yaml")
    selected = exported["data"]["formal_a0_selection"]
    assert selected["study_complete"] is True
    assert selected["selected_trial"] == 0
    assert selected["selected_params"] == {
        "lr": pytest.approx(1.5e-4),
        "weight_decay": pytest.approx(0.003),
        "spatial_weight": pytest.approx(0.2),
    }
    assert selected["trial_config_sha256"] == module.config_digest(
        config
    )
    assert selected["best_checkpoint_sha256"] == module.file_sha256(
        checkpoint
    )
    assert selected["scheduler_contract"]["planned_total_steps"] == 20_576
