#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
from typing import Any

import yaml

try:
    import optuna
except ImportError as exc:  # pragma: no cover
    raise SystemExit("optuna is required. Install with: pip install -e '.[search]'") from exc


TEACHERS = ("gigapath", "h_optimus_1", "uni2_h", "virchow2")
BASELINE_PARAMS = {"lr": 1e-4, "weight_decay": 1e-2}
SEARCH_SPACE = {
    "lr": {"low": 3e-5, "high": 2e-4, "log": True},
    "weight_decay": {"low": 1e-3, "high": 3e-2, "log": True},
}
RESULT_METRICS = (
    "train_loss",
    "train_feature",
    "train_relation",
    "train_semantic",
    "train_pamtd_response",
    "train_l1",
    "train_l1_accuracy",
    "train_l2_spatial",
    "train_l2_instance_point",
    "train_l2_abundance_point",
    "train_l2_brush_bag",
    "train_l2_area_positive",
    "train_l2_explicit_negative",
    "train_l2_implicit_negative",
    "teacher_alignment_score",
    "train_tiles_per_sec",
    "global_step",
    "l2_supervised_step",
)


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


def config_digest(cfg: dict[str, Any]) -> str:
    payload = yaml.safe_dump(cfg, sort_keys=True, allow_unicode=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_commit(repo: Path) -> str:
    explicit = os.environ.get("HCC_SEMPATH_SOURCE_COMMIT", "").strip()
    if explicit:
        return explicit
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if result.returncode != 0 or not commit:
        raise RuntimeError(
            "source commit is unavailable; export HCC_SEMPATH_SOURCE_COMMIT "
            "when running from a source archive"
        )
    return commit


def require_prototype_inputs(cfg: dict[str, Any]) -> None:
    data = cfg.get("data", {})
    missing = []
    prototype_paths = data.get("prototype_paths")
    if not isinstance(prototype_paths, dict):
        missing.append("data.prototype_paths")
    else:
        missing.extend(f"data.prototype_paths.{teacher}" for teacher in TEACHERS if teacher not in prototype_paths)
    for key in ("prototype_supervision_manifest_path", "spatial_manifest_path"):
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

    # Optuna uses a deterministic one-tenth population view. The fixed L1/L2
    # expert banks remain complete and identical across trials.
    cfg["data"]["train_tile_fraction"] = 0.10
    cfg["data"]["val_tile_fraction"] = 0.10
    cfg["data"]["num_workers"] = int(cfg["data"].get("num_workers", 16))
    cfg["data"]["prefetch_factor"] = int(cfg["data"].get("prefetch_factor", 3))
    cfg["data"]["persistent_workers"] = bool(cfg["data"].get("persistent_workers", True))
    cfg["data"]["dynamic_package_sampling"] = bool(cfg["data"].get("dynamic_package_sampling", True))
    cfg["data"]["tensor_collate"] = bool(cfg["data"].get("tensor_collate", True))
    cfg["data"]["package_chunk_size"] = int(cfg["data"].get("package_chunk_size", 64))

    cfg["train"]["batch_size"] = int(cfg["train"].get("batch_size", 512))
    cfg["train"]["epochs"] = int(epochs)
    cfg["train"]["lr"] = trial.suggest_float("lr", **SEARCH_SPACE["lr"])
    cfg["train"]["weight_decay"] = trial.suggest_float(
        "weight_decay",
        **SEARCH_SPACE["weight_decay"],
    )
    cfg["train"]["max_grad_norm"] = 1.0
    # Hyperparameter selection is based on the population training objective.
    # Keep one diagnostic batch only because the generic trainer currently
    # finalizes its epoch row after that pass; it is not an Optuna validation
    # signal.
    cfg["train"]["max_val_batches"] = 1
    cfg["train"]["max_eval_batches"] = 1
    cfg["train"]["eval_pairwise_max_samples"] = 512
    cfg["train"]["log_interval"] = 200
    cfg["train"]["progress"] = "tqdm"
    cfg["train"]["tensorboard"] = False
    cfg["train"]["tensorboard_batch_interval"] = 0
    return cfg


def score_row(row: dict[str, str], objective: str) -> float:
    def value(key: str, *, required: bool = False) -> float:
        try:
            result = float(row.get(key, "") or "nan")
        except ValueError as exc:
            if required:
                raise ValueError(f"non-numeric objective metric {key}={row.get(key)!r}") from exc
            return 0.0
        if not math.isfinite(result):
            if required:
                raise ValueError(f"non-finite objective metric {key}={result}")
            return 0.0
        return result

    teacher_alignment = value("teacher_alignment_score")
    l1_acc = value("l1_accuracy")
    train_tiles_per_sec = value("train_tiles_per_sec")
    train_loss = value("train_loss", required=objective == "train_loss")
    if objective == "train_loss":
        return -train_loss
    if objective == "teacher_alignment":
        return teacher_alignment
    if objective == "speed":
        return train_tiles_per_sec
    if objective == "l1_accuracy":
        return l1_acc
    return teacher_alignment + 0.25 * l1_acc


def export_study_artifacts(
    study: optuna.Study,
    *,
    output_root: Path,
    manifest: dict[str, Any],
) -> None:
    rows: list[dict[str, Any]] = []
    for trial in study.get_trials(deepcopy=False):
        row = {
            "number": trial.number,
            "state": trial.state.name,
            "objective_score": trial.value,
            "lr": trial.params.get("lr"),
            "weight_decay": trial.params.get("weight_decay"),
        }
        row.update(
            {
                metric: trial.user_attrs.get(f"final_{metric}")
                for metric in RESULT_METRICS
            }
        )
        row["output_dir"] = trial.user_attrs.get("output_dir")
        row["failure_reason"] = trial.user_attrs.get("failure_reason")
        rows.append(row)

    summary_path = output_root / "study_summary.csv"
    temporary_summary = summary_path.with_suffix(".csv.tmp")
    fieldnames = list(rows[0]) if rows else [
        "number",
        "state",
        "objective_score",
        "lr",
        "weight_decay",
    ]
    with temporary_summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_summary.replace(summary_path)

    manifest_path = output_root / "study_manifest.yaml"
    write_yaml(manifest_path, manifest)
    if study.best_trials:
        best = study.best_trial
        best_config = Path(str(best.user_attrs.get("output_dir", ""))) / "config.yaml"
        if best_config.is_file():
            shutil.copy2(best_config, output_root / "best_config.yaml")


def read_metric_rows(metrics_path: Path) -> list[dict[str, str]]:
    if not metrics_path.exists():
        return []
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stream_process(process: subprocess.Popen[str], log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        assert process.stdout is not None
        while True:
            character = process.stdout.read(1)
            if not character:
                break
            flush = character in {"\r", "\n"}
            print(character, end="", flush=flush)
            log.write(character)
            if flush:
                log.flush()
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


def train_with_pruning(
    *,
    trial: optuna.Trial,
    cfg_path: Path,
    output_dir: Path,
    python_bin: str,
    repo: Path,
    poll_sec: float,
    objective: str,
) -> float:
    log_path = output_dir / "trial.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src")
    env["PYTHONNOUSERSITE"] = "1"
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
            trial.set_user_attr(f"epoch_{epoch}_train_tiles_per_sec", row.get("train_tiles_per_sec"))
            trial.set_user_attr(f"epoch_{epoch}_val_tiles_per_sec", row.get("val_tiles_per_sec"))
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
    for metric in RESULT_METRICS:
        trial.set_user_attr(f"final_{metric}", rows[-1].get(metric))
    trial.set_user_attr("final_val_tiles_per_sec", rows[-1].get("val_tiles_per_sec"))
    trial.set_user_attr("best_observed_score", best_score)
    return best_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna 1/10 HCC-SemPath hyperparameter search.")
    parser.add_argument("--base-config", default="configs/local/server/train_tenth.yaml")
    parser.add_argument("--study-name", default="hcc_sempath_tenth_spatial")
    parser.add_argument("--storage", default="sqlite:///runtime/optuna/hcc_sempath_tenth_spatial.db")
    parser.add_argument("--output-root", default="runtime/optuna_runs")
    parser.add_argument("--annotation-json", default="")
    parser.add_argument("--prototype-asset-dir", default="artifacts/prototypes/hcc_annotation_final_3000")
    parser.add_argument("--python", default="python")
    parser.add_argument("--n-trials", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--timeout-hours", type=float, default=0.0)
    parser.add_argument("--poll-sec", type=float, default=20.0)
    parser.add_argument(
        "--objective",
        choices=["train_loss", "combined", "teacher_alignment", "l1_accuracy", "speed"],
        default="train_loss",
    )
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

    sampler = optuna.samplers.TPESampler(
        seed=int(args.sampler_seed),
        n_startup_trials=4,
        multivariate=True,
        group=True,
    )
    pruner = (
        optuna.pruners.NopPruner()
        if int(args.epochs) <= 1
        else optuna.pruners.MedianPruner(
            n_startup_trials=2,
            n_warmup_steps=1,
            interval_steps=1,
        )
    )
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=args.storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )
    study.enqueue_trial(BASELINE_PARAMS, skip_if_exists=True)
    commit = source_commit(repo)
    manifest = {
        "study_name": args.study_name,
        "purpose": "new-manuscript one-tenth-population hyperparameter search",
        "source_commit": commit,
        "base_config": str(base_config_path),
        "base_config_sha256": config_digest(base_cfg),
        "objective": args.objective,
        "direction": "maximize",
        "objective_interpretation": (
            "negative train_loss; all task-loss definitions and weights are fixed across trials"
            if args.objective == "train_loss"
            else args.objective
        ),
        "epochs_per_trial": int(args.epochs),
        "population_fraction": 0.10,
        "complete_l1_l2_expert_banks": True,
        "n_trials_requested": int(args.n_trials),
        "timeout_hours": float(args.timeout_hours),
        "runtime_seed": int(base_cfg["runtime"]["seed"]),
        "sampler": "TPESampler",
        "sampler_seed": int(args.sampler_seed),
        "n_startup_trials": 4,
        "pruner": (
            "NopPruner"
            if int(args.epochs) <= 1
            else "MedianPruner(n_startup_trials=2,n_warmup_steps=1,interval_steps=1)"
        ),
        "baseline_params": BASELINE_PARAMS,
        "search_space": SEARCH_SPACE,
        "fixed_loss_config": base_cfg.get("loss", {}),
    }
    for key, value in manifest.items():
        if isinstance(value, (str, int, float, bool)):
            study.set_user_attr(key, value)
    export_study_artifacts(study, output_root=output_root, manifest=manifest)

    def objective(trial: optuna.Trial) -> float:
        trial_dir = output_root / f"trial_{trial.number:04d}"
        if trial_dir.exists():
            shutil.rmtree(trial_dir)
        trial_dir.mkdir(parents=True)
        cfg = trial_config(base_cfg, trial, trial_dir, epochs=int(args.epochs))
        cfg_path = trial_dir / "config.yaml"
        write_yaml(cfg_path, cfg)
        print(
            f"trial_start number={trial.number} "
            f"source_commit={commit} "
            f"lr={trial.params['lr']:.8g} "
            f"weight_decay={trial.params['weight_decay']:.8g}",
            flush=True,
        )
        trial.set_user_attr("source_commit", commit)
        trial.set_user_attr("config_sha256", config_digest(cfg))
        try:
            return train_with_pruning(
                trial=trial,
                cfg_path=cfg_path,
                output_dir=trial_dir,
                python_bin=str(args.python),
                repo=repo,
                poll_sec=float(args.poll_sec),
                objective=str(args.objective),
            )
        except Exception as exc:
            trial.set_user_attr("failure_reason", f"{type(exc).__name__}: {exc}")
            raise

    def export_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        del trial
        export_study_artifacts(study, output_root=output_root, manifest=manifest)

    timeout = (
        None
        if float(args.timeout_hours) <= 0
        else float(args.timeout_hours) * 3600.0
    )
    study.optimize(
        objective,
        n_trials=int(args.n_trials),
        timeout=timeout,
        gc_after_trial=True,
        callbacks=[export_callback],
        catch=(RuntimeError, ValueError),
    )
    export_study_artifacts(study, output_root=output_root, manifest=manifest)
    if not study.best_trials:
        print("optuna_search_complete completed_trials=0")
        return
    best = study.best_trial
    print(f"best_trial={best.number} value={best.value:.6f}")
    print(f"best_params={best.params}")
    print(f"best_output_dir={best.user_attrs.get('output_dir')}")


if __name__ == "__main__":
    main()
