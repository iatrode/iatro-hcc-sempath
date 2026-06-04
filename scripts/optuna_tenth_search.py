#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any

import yaml

try:
    import optuna
except ImportError as exc:  # pragma: no cover
    raise SystemExit("optuna is required. Install with: pip install -e '.[search]'") from exc


TEACHERS = ("gigapath", "h_optimus_1", "uni2_h", "virchow2")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    parent = payload.get("inherits")
    if parent is None:
        return payload
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return deep_merge(load_yaml(parent_path), {key: value for key, value in payload.items() if key != "inherits"})


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def require_prototype_inputs(cfg: dict[str, Any]) -> None:
    data = cfg.get("data", {})
    missing = []
    prototype_paths = data.get("prototype_paths")
    if not isinstance(prototype_paths, dict):
        missing.append("data.prototype_paths")
    else:
        missing.extend(f"data.prototype_paths.{teacher}" for teacher in TEACHERS if teacher not in prototype_paths)
    for key in ("zhcc_prototype_image_path", "prototype_supervision_manifest_path"):
        if not data.get(key):
            missing.append(f"data.{key}")
    if missing:
        raise ValueError(
            "prototype search requires prototype inputs in the base config; missing: " + ", ".join(missing)
        )


def has_prototype_inputs(cfg: dict[str, Any]) -> bool:
    try:
        require_prototype_inputs(cfg)
    except ValueError:
        return False
    return True


def inject_prototype_assets(cfg: dict[str, Any], asset_dir: Path) -> dict[str, Any]:
    cfg = deep_merge({}, cfg)
    cfg.setdefault("data", {})
    cfg["data"]["prototype_paths"] = {
        "gigapath": str(asset_dir / "gigapath_hcc_semantic_prototypes.pt"),
        "h_optimus_1": str(asset_dir / "h_optimus_1_hcc_semantic_prototypes.pt"),
        "uni2_h": str(asset_dir / "uni2_h_hcc_semantic_prototypes.pt"),
        "virchow2": str(asset_dir / "virchow2_hcc_semantic_prototypes.pt"),
    }
    cfg["data"]["zhcc_prototype_image_path"] = str(asset_dir / "zhcc_hcc_prototype_images.pt")
    cfg["data"]["prototype_supervision_manifest_path"] = str(asset_dir / "hcc_prototype_supervision_manifest.csv")
    cfg["data"]["prototype_supervision_train_splits"] = ["train"]
    cfg["data"]["prototype_supervision_val_splits"] = ["val"]
    return cfg


def maybe_build_prototype_assets(
    *,
    cfg: dict[str, Any],
    annotation_json: str,
    asset_dir: Path,
    python_bin: str,
    repo: Path,
) -> dict[str, Any]:
    if has_prototype_inputs(cfg):
        return cfg
    if not annotation_json:
        require_prototype_inputs(cfg)
    asset_dir.mkdir(parents=True, exist_ok=True)
    required_assets = [
        asset_dir / "gigapath_hcc_semantic_prototypes.pt",
        asset_dir / "h_optimus_1_hcc_semantic_prototypes.pt",
        asset_dir / "uni2_h_hcc_semantic_prototypes.pt",
        asset_dir / "virchow2_hcc_semantic_prototypes.pt",
        asset_dir / "zhcc_hcc_prototype_images.pt",
        asset_dir / "hcc_prototype_supervision_manifest.csv",
    ]
    if any(not path.exists() for path in required_assets):
        manifest_path = Path(str(cfg["data"].get("train_manifest_path", "")))
        if not manifest_path.is_absolute():
            manifest_path = repo / manifest_path
        command = [
            python_bin,
            str(repo / "scripts" / "build_prototype_assets_from_annotations.py"),
            "--annotation-json",
            annotation_json,
            "--training-manifest",
            str(manifest_path),
            "--output-dir",
            str(asset_dir),
            "--embedding-dim",
            str(int(cfg.get("model", {}).get("embedding_dim", 1536))),
            "--source-split",
            "train",
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo / "src")
        subprocess.run(command, cwd=repo, env=env, check=True)
    built = inject_prototype_assets(cfg, asset_dir)
    require_prototype_inputs(built)
    return built


def trial_config(base_cfg: dict[str, Any], trial: optuna.Trial, output_dir: Path, epochs: int) -> dict[str, Any]:
    cfg = deep_merge({}, base_cfg)
    cfg.setdefault("runtime", {})
    cfg.setdefault("data", {})
    cfg.setdefault("loss", {})
    cfg.setdefault("train", {})
    cfg["runtime"]["output_dir"] = str(output_dir)

    cfg["data"]["train_tile_fraction"] = 0.10
    cfg["data"]["val_tile_fraction"] = 0.10
    cfg["data"]["dynamic_package_sampling"] = True
    cfg["data"]["tensor_collate"] = True
    cfg["data"]["package_buffer_batches"] = trial.suggest_categorical("package_buffer_batches", [4, 6, 8])
    cfg["data"]["package_chunk_size"] = trial.suggest_categorical("package_chunk_size", [32, 64, 96])

    prototype_label_share = trial.suggest_categorical("prototype_label_share", [0.35, 0.45, 0.55, 0.65])
    l2_agreement_share = trial.suggest_categorical("prototype_l2_agreement_share", [0.4, 0.5, 0.6])
    cfg["loss"]["relation_weight"] = trial.suggest_categorical("relation_weight", [0.02, 0.05, 0.08])
    cfg["loss"]["scale_relation_by_alpha"] = True
    cfg["loss"]["zhcc_proto_weight"] = trial.suggest_categorical("zhcc_proto_weight", [0.10, 0.20, 0.30])
    cfg["loss"]["zhcc_level2_weight"] = trial.suggest_categorical("zhcc_level2_weight", [0.35, 0.50, 0.65])
    cfg["loss"]["prototype_filter_weight"] = trial.suggest_categorical("prototype_filter_weight", [0.30, 0.50, 0.70])
    cfg["loss"]["prototype_filter_alpha_min"] = trial.suggest_categorical("prototype_filter_alpha_min", [0.15, 0.25, 0.35])
    cfg["loss"]["consensus_weight"] = round(1.0 - float(prototype_label_share), 6)
    cfg["loss"]["prototype_label_weight"] = float(prototype_label_share)
    cfg["loss"]["prototype_l1_agreement_weight"] = round(1.0 - float(l2_agreement_share), 6)
    cfg["loss"]["prototype_l2_agreement_weight"] = float(l2_agreement_share)
    cfg["loss"]["zhcc_response_weight"] = 0.0
    cfg["loss"]["zhcc_primary_temperature"] = trial.suggest_categorical("zhcc_primary_temperature", [0.07, 0.10, 0.15])
    cfg["loss"]["zhcc_attribute_temperature"] = trial.suggest_categorical("zhcc_attribute_temperature", [0.07, 0.10, 0.15])
    cfg["loss"]["min_teacher_warmup_steps"] = 1000
    cfg["loss"]["max_teacher_warmup_steps"] = 4000
    cfg["loss"]["teacher_prior_plateau_window_steps"] = 500
    cfg["loss"]["prototype_ramp_steps"] = 500
    cfg["loss"]["filter_ramp_steps"] = 500
    cfg["loss"]["proto_to_filter_delay_steps"] = 500

    cfg["train"]["batch_size"] = int(cfg["train"].get("batch_size", 512))
    cfg["train"]["epochs"] = int(epochs)
    cfg["train"]["warmup_epochs"] = 1
    cfg["train"]["lr"] = trial.suggest_categorical("lr", [5e-5, 1e-4, 2e-4])
    cfg["train"]["weight_decay"] = trial.suggest_categorical("weight_decay", [0.005, 0.01, 0.02])
    cfg["train"]["max_grad_norm"] = 1.0
    cfg["train"]["max_val_batches"] = 256
    cfg["train"]["max_eval_batches"] = 64
    cfg["train"]["log_interval"] = 0
    cfg["train"]["progress"] = "tqdm"
    cfg["train"]["tensorboard"] = True
    cfg["train"]["tensorboard_batch_interval"] = 50
    return cfg


def score_row(row: dict[str, str], objective: str) -> float:
    def value(key: str) -> float:
        try:
            return float(row.get(key, 0.0) or 0.0)
        except ValueError:
            return 0.0

    teacher_alignment = value("teacher_alignment_score")
    prototype_topk = value("prototype_bank_zhcc_prototype_topk_precision") or value("zhcc_prototype_topk_precision")
    l1_acc = value("prototype_bank_zhcc_level1_accuracy") or value("zhcc_level1_accuracy")
    l2_auc = value("prototype_bank_zhcc_level2_macro_auc") or value("zhcc_level2_macro_auc")
    if objective == "teacher_alignment":
        return teacher_alignment
    if objective == "prototype_qc":
        return 0.40 * prototype_topk + 0.30 * l1_acc + 0.30 * l2_auc
    return 0.70 * teacher_alignment + 0.10 * prototype_topk + 0.10 * l1_acc + 0.10 * l2_auc


def read_metric_rows(metrics_path: Path) -> list[dict[str, str]]:
    if not metrics_path.exists():
        return []
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stream_process(process: subprocess.Popen[str], log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()


def terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=30)


def run_command(command: list[str], env: dict[str, str], log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed returncode={completed.returncode}: {' '.join(command)}")


def train_with_pruning(
    *,
    trial: optuna.Trial,
    cfg_path: Path,
    output_dir: Path,
    python_bin: str,
    repo: Path,
    poll_sec: float,
    objective: str,
    preflight: bool,
) -> float:
    log_path = output_dir / "trial.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src")
    env["PYTHONNOUSERSITE"] = "1"
    if preflight:
        run_command(
            [
                python_bin,
                "-m",
                "hcc_sempath.cli.main",
                "preflight",
                "--config",
                str(cfg_path),
                "--max-records",
                "2048",
            ],
            env,
            log_path,
        )
    command = [python_bin, "-m", "hcc_sempath.cli.main", "train", "--config", str(cfg_path)]
    process = subprocess.Popen(
        command,
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        start_new_session=(os.name == "posix"),
    )
    thread = threading.Thread(target=stream_process, args=(process, log_path), daemon=True)
    thread.start()
    metrics_path = output_dir / "metrics.csv"
    reported_epochs: set[int] = set()
    best_score = float("-inf")
    while process.poll() is None:
        for row in read_metric_rows(metrics_path):
            epoch = int(float(row.get("epoch", "0") or 0))
            if epoch <= 0 or epoch in reported_epochs:
                continue
            score = score_row(row, objective)
            reported_epochs.add(epoch)
            best_score = max(best_score, score)
            trial.report(score, step=epoch)
            trial.set_user_attr(f"epoch_{epoch}_score", score)
            if trial.should_prune():
                terminate_process(process)
                raise optuna.TrialPruned(f"pruned at epoch={epoch} score={score:.6f}")
        time.sleep(float(poll_sec))
    thread.join(timeout=30)
    if process.returncode != 0:
        raise RuntimeError(f"training failed returncode={process.returncode}; see {log_path}")
    rows = read_metric_rows(metrics_path)
    if not rows:
        raise RuntimeError(f"training produced no metrics: {metrics_path}")
    final_score = score_row(rows[-1], objective)
    best_score = max(best_score, final_score)
    trial.set_user_attr("output_dir", str(output_dir))
    trial.set_user_attr("final_epoch", rows[-1].get("epoch"))
    trial.set_user_attr("final_score", final_score)
    trial.set_user_attr("best_observed_score", best_score)
    return best_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna 1/10 HCC-SemPath hyperparameter search.")
    parser.add_argument("--base-config", default="configs/local/server/train_tenth.yaml")
    parser.add_argument("--study-name", default="hcc_sempath_tenth_pamtd")
    parser.add_argument("--storage", default="sqlite:///runtime/runtime/optuna/hcc_sempath_tenth_pamtd.db")
    parser.add_argument("--output-root", default="runtime/runtime/optuna_runs")
    parser.add_argument("--annotation-json", default="")
    parser.add_argument("--prototype-asset-dir", default="runtime/prototypes/hcc_annotation_final_3000")
    parser.add_argument("--python", default="python")
    parser.add_argument("--n-trials", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--poll-sec", type=float, default=20.0)
    parser.add_argument("--objective", choices=["combined", "teacher_alignment", "prototype_qc"], default="combined")
    parser.add_argument("--no-preflight", action="store_true")
    parser.add_argument("--sampler-seed", type=int, default=13)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    base_config_path = Path(args.base_config)
    if not base_config_path.is_absolute():
        base_config_path = repo / base_config_path
    base_cfg = load_yaml(base_config_path)
    base_cfg = maybe_build_prototype_assets(
        cfg=base_cfg,
        annotation_json=str(args.annotation_json),
        asset_dir=Path(args.prototype_asset_dir),
        python_bin=str(args.python),
        repo=repo,
    )
    if int(args.n_trials) <= 0:
        print("optuna_search_ready n_trials=0")
        return

    storage_dir = Path(args.storage.removeprefix("sqlite:///")).parent if args.storage.startswith("sqlite:///") else None
    if storage_dir is not None:
        storage_dir.mkdir(parents=True, exist_ok=True)
    output_root = Path(args.output_root) / args.study_name
    output_root.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=int(args.sampler_seed), multivariate=True, group=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=4, interval_steps=1)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=args.storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    def objective(trial: optuna.Trial) -> float:
        trial_dir = output_root / f"trial_{trial.number:04d}"
        if trial_dir.exists():
            shutil.rmtree(trial_dir)
        trial_dir.mkdir(parents=True)
        cfg = trial_config(base_cfg, trial, trial_dir, epochs=int(args.epochs))
        cfg_path = trial_dir / "config.yaml"
        write_yaml(cfg_path, cfg)
        return train_with_pruning(
            trial=trial,
            cfg_path=cfg_path,
            output_dir=trial_dir,
            python_bin=str(args.python),
            repo=repo,
            poll_sec=float(args.poll_sec),
            objective=str(args.objective),
            preflight=not bool(args.no_preflight),
        )

    study.optimize(objective, n_trials=int(args.n_trials), gc_after_trial=True)
    best = study.best_trial
    print(f"best_trial={best.number} value={best.value:.6f}")
    print(f"best_params={best.params}")
    print(f"best_output_dir={best.user_attrs.get('output_dir')}")


if __name__ == "__main__":
    main()
