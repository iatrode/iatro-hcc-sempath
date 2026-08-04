#!/usr/bin/env python
"""Reproducible A0 Optuna search on the fixed one-tenth population view."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any

import yaml
import torch

try:
    import optuna
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "optuna is required. Install with: pip install -e '.[search]'"
    ) from exc


TEACHERS = ("gigapath", "h_optimus_1", "uni2_h", "virchow2")
BASELINE_PARAMS = {
    "lr": 1.5088805358242106e-4,
    "weight_decay": 2.7941282807460287e-3,
    "spatial_weight": 0.1,
}
SEEDED_PARAMS = (
    BASELINE_PARAMS,
    {
        "lr": 1.30e-4,
        "weight_decay": 3.0e-3,
        "spatial_weight": 0.2,
    },
    {
        "lr": 1.70e-4,
        "weight_decay": 3.0e-3,
        "spatial_weight": 0.4,
    },
    {
        "lr": 1.50e-4,
        "weight_decay": 7.0e-3,
        "spatial_weight": 0.7,
    },
)
SEARCH_SPACE = {
    "lr": {"low": 9e-5, "high": 2.1e-4, "log": True},
    "weight_decay": {"low": 1e-3, "high": 1.2e-2, "log": True},
    "spatial_weight": {"low": 0.08, "high": 1.0, "log": True},
}
SELECTION_COMPONENTS = (
    "teacher",
    "classification",
    "spatial",
)
PRUNER_WARMUP_STEPS = 8
RESULT_METRICS = (
    "selection_loss",
    *(
        f"selection_{component}_{suffix}"
        for component in SELECTION_COMPONENTS
        for suffix in ("raw", "baseline", "normalized", "weight")
    ),
    "teacher_validation_loss",
    "fixed_teacher_distance",
    "fixed_teacher_relation",
    "expert_val_classification_macro_f1",
    "expert_val_classification_balanced_accuracy",
    "expert_val_classification_evaluated_tiles",
    "expert_val_classification_evaluated_classes",
    "expert_val_spatial_instance_point",
    "expert_val_spatial_measurement_positive",
    "expert_val_spatial_explicit_negative",
    "expert_val_spatial_implicit_negative",
    "teacher_alignment_score",
    "train_loss",
    "train_tiles_per_sec",
    "global_step",
    "spatial_supervised_step",
)


class TrialSeededTPESampler(optuna.samplers.BaseSampler):
    """TPE whose per-trial RNG is reproducible across process restarts."""

    def __init__(
        self,
        *,
        seed: int,
        n_startup_trials: int,
        constant_liar: bool = False,
    ) -> None:
        self.seed = int(seed)
        self.n_startup_trials = int(n_startup_trials)
        self.constant_liar = bool(constant_liar)
        self._delegates: dict[int, optuna.samplers.TPESampler] = {}
        self._delegates_lock = threading.Lock()

    def _delegate(
        self,
        trial: optuna.trial.FrozenTrial,
    ) -> optuna.samplers.TPESampler:
        with self._delegates_lock:
            if trial.number not in self._delegates:
                self._delegates[trial.number] = optuna.samplers.TPESampler(
                    seed=self.seed + int(trial.number),
                    n_startup_trials=self.n_startup_trials,
                    multivariate=True,
                    group=True,
                    constant_liar=self.constant_liar,
                )
            return self._delegates[trial.number]

    def before_trial(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
    ) -> None:
        self._delegate(trial).before_trial(study, trial)

    def infer_relative_search_space(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
    ) -> dict[str, optuna.distributions.BaseDistribution]:
        return self._delegate(trial).infer_relative_search_space(
            study,
            trial,
        )

    def sample_relative(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: dict[
            str,
            optuna.distributions.BaseDistribution,
        ],
    ) -> dict[str, Any]:
        return self._delegate(trial).sample_relative(
            study,
            trial,
            search_space,
        )

    def sample_independent(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution: optuna.distributions.BaseDistribution,
    ) -> Any:
        return self._delegate(trial).sample_independent(
            study,
            trial,
            param_name,
            param_distribution,
        )

    def after_trial(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        state: optuna.trial.TrialState,
        values: list[float] | None,
    ) -> None:
        with self._delegates_lock:
            delegate = self._delegates.pop(trial.number, None)
        if delegate is not None:
            delegate.after_trial(study, trial, state, values)


def deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
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
    return deep_merge(
        load_yaml(parent_path),
        {
            key: value
            for key, value in payload.items()
            if key != "inherits"
        },
    )


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def atomic_write_yaml(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    write_yaml(temporary, payload)
    temporary.replace(path)


def _canonical_digest(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def config_digest(cfg: dict[str, Any]) -> str:
    return _canonical_digest(cfg)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_tree_sha256(repo: Path) -> str:
    """Hash the executable/scientific source tree of a source archive."""

    roots = (
        repo / "pyproject.toml",
        repo / "README.md",
        repo / "CHANGELOG.md",
        repo / "configs",
        repo / "docs",
        repo / "experiments",
        repo / "scripts",
        repo / "src",
        repo / "tests",
    )
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix not in {".pyc", ".pyo"}
            )
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        relative = path.relative_to(repo).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def source_state(repo: Path) -> dict[str, str]:
    tree_digest = source_tree_sha256(repo)
    explicit = os.environ.get(
        "HCC_SEMPATH_SOURCE_COMMIT",
        "",
    ).strip()
    if explicit:
        if re.fullmatch(r"[0-9a-fA-F]{40}", explicit) is None:
            raise RuntimeError(
                "HCC_SEMPATH_SOURCE_COMMIT must be a full 40-character "
                "Git commit SHA"
            )
        declared_tree = os.environ.get(
            "HCC_SEMPATH_SOURCE_TREE_SHA256",
            "",
        ).strip()
        if declared_tree and declared_tree.lower() != tree_digest:
            raise RuntimeError(
                "declared source archive tree SHA-256 does not match "
                "the executable source tree"
            )
        return {
            "commit": explicit.lower(),
            "source_mode": "declared_archive",
            "source_tree_sha256": tree_digest,
        }
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
            "source commit is unavailable; export "
            "HCC_SEMPATH_SOURCE_COMMIT for a source archive"
        )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise RuntimeError("unable to verify source working tree")
    if status.stdout.strip():
        raise RuntimeError(
            "formal A0 search requires a clean committed source tree; "
            "commit or remove every staged, unstaged, and untracked change"
        )
    return {
        "commit": commit,
        "source_mode": "clean_git_commit",
        "source_tree_sha256": tree_digest,
    }


def _required_asset_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    data = cfg.get("data", {})
    paths: dict[str, Path] = {}
    prototype_paths = data.get("prototype_paths")
    if not isinstance(prototype_paths, dict):
        raise ValueError("data.prototype_paths is required")
    for teacher in TEACHERS:
        value = prototype_paths.get(teacher)
        if not value:
            raise ValueError(
                f"data.prototype_paths.{teacher} is required"
            )
        paths[f"prototype_{teacher}"] = Path(str(value)).resolve()
    for key in (
        "prototype_supervision_manifest_path",
        "spatial_manifest_path",
        "train_manifest_path",
    ):
        value = data.get(key)
        if not value:
            raise ValueError(f"data.{key} is required")
        paths[key] = Path(str(value)).resolve()
    missing = [
        f"{name}={path}"
        for name, path in paths.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "A0 search assets are missing: " + ", ".join(missing)
        )
    return paths


def _resolved_training_iac_paths(
    cfg: dict[str, Any],
    *,
    complete: bool,
    splits: tuple[str, ...] = ("train", "val"),
) -> tuple[list[Path], dict[str, list[Path]]]:
    """Resolve the exact tile/teacher packages behind the training manifest."""

    if not splits or any(split not in {"train", "val"} for split in splits):
        raise ValueError("splits must be a non-empty subset of train/val")
    from hcc_sempath.training.config import (
        image_tile_package_paths,
        manifest_data_paths,
        teacher_feature_package_paths,
    )
    from hcc_sempath.training.manifest import load_training_manifest

    resolved_cfg = deep_merge({}, cfg)
    if complete:
        resolved_cfg.setdefault("data", {})
        resolved_cfg["data"]["train_tile_fraction"] = 1.0
        resolved_cfg["data"]["val_tile_fraction"] = 1.0
    data = resolved_cfg.get("data", {})
    manifest_path = data.get("train_manifest_path")
    explicit_split_packages = "train_image_tile_package_paths" in data
    if explicit_split_packages:
        active_teachers = [
            str(value)
            for value in data.get("teachers", TEACHERS)
        ]
        tile_paths = []
        for split in splits:
            key = f"{split}_image_tile_package_paths"
            values = data.get(key)
            if values is None:
                raise ValueError(f"data.{key} is required")
            if isinstance(values, dict):
                tile_paths.extend(str(path) for path in values.values())
            elif isinstance(values, (list, tuple)):
                tile_paths.extend(str(path) for path in values)
            else:
                tile_paths.append(str(values))
        teacher_paths = {
            teacher: []
            for teacher in TEACHERS
        }
        for split in splits:
            key = f"{split}_teacher_feature_package_paths"
            mapping = data.get(key)
            if not isinstance(mapping, dict):
                raise ValueError(
                    f"data.{key} must be a teacher->paths mapping"
                )
            for teacher in active_teachers:
                values = mapping.get(teacher)
                if values is None:
                    raise ValueError(
                        f"data.{key} is missing teacher {teacher}"
                    )
                teacher_paths[teacher].extend(
                    str(path)
                    for path in (
                        values
                        if isinstance(values, (list, tuple))
                        else [values]
                    )
                )
    elif manifest_path:
        manifest = load_training_manifest(manifest_path)
        tile_paths: list[str] = []
        teacher_paths: dict[str, list[str]] = {
            teacher: [] for teacher in TEACHERS
        }
        for split in splits:
            split_tiles, split_teachers = manifest_data_paths(
                resolved_cfg,
                manifest,
                split,
            )
            tile_paths.extend(split_tiles)
            for teacher in TEACHERS:
                teacher_paths[teacher].extend(
                    split_teachers[teacher]
                )
    else:
        if set(splits) != {"train", "val"}:
            raise ValueError(
                "split-specific A0 path resolution requires a manifest or "
                "explicit train/val package lists"
            )
        tile_paths = image_tile_package_paths(resolved_cfg)
        teacher_paths = teacher_feature_package_paths(resolved_cfg)
    return (
        sorted({Path(path).resolve() for path in tile_paths}),
        {
            teacher: sorted(
                {
                    Path(path).resolve()
                    for path in teacher_paths.get(teacher, [])
                }
            )
            for teacher in TEACHERS
        },
    )


def _selection_start_step_from_config(cfg: dict[str, Any]) -> int:
    loss_cfg = cfg.get("loss", {})
    endpoints = [
        int(loss_cfg.get("expert_supervision_start_step", 0))
        + int(loss_cfg.get("expert_supervision_ramp_steps", 0))
    ]
    for weight_key, start_key, ramp_key in (
        (
            "prototype_filter_weight",
            "prototype_filter_start_step",
            "prototype_filter_ramp_steps",
        ),
        (
            "zhcc_response_weight",
            "zhcc_response_start_step",
            "zhcc_response_ramp_steps",
        ),
    ):
        if float(loss_cfg.get(weight_key, 0.0)) > 0.0:
            endpoints.append(
                int(loss_cfg.get(start_key, 0))
                + int(loss_cfg.get(ramp_key, 0))
            )
    configured = cfg.get("train", {}).get(
        "selection_early_stop_start_step"
    )
    return max(
        max(endpoints),
        int(configured) if configured is not None else 0,
    )


def _population_schedule_contract(
    cfg: dict[str, Any],
    selected_train_tiles: list[Path],
    *,
    classification_val_tiles: int,
    spatial_val_tiles: int,
) -> dict[str, int]:
    """Conservatively prove that post-ramp selection epochs are reachable."""

    from iatro.iac import read_header

    records = sum(
        int(read_header(path)["num_records"])
        for path in selected_train_tiles
    )
    # Classification/spatial banks may overlap. Subtracting both totals is conservative and therefore
    # cannot overstate the number of optimizer-visible rows.
    lower_bound = max(
        0,
        records - classification_val_tiles - spatial_val_tiles,
    )
    max_records = int(cfg.get("data", {}).get("max_train_records", 0))
    if max_records > 0:
        lower_bound = min(lower_bound, max_records)
    batch_size = int(cfg.get("train", {}).get("batch_size", 0))
    epochs = int(cfg.get("train", {}).get("epochs", 0))
    if batch_size <= 0 or epochs <= 0:
        raise ValueError("A0 schedule requires positive batch size and epochs")
    steps_per_epoch_lower_bound = (
        lower_bound + batch_size - 1
    ) // batch_size
    max_train_batches = cfg.get("train", {}).get("max_train_batches")
    if max_train_batches is not None:
        steps_per_epoch_lower_bound = min(
            steps_per_epoch_lower_bound,
            int(max_train_batches),
        )
    selection_start_step = _selection_start_step_from_config(cfg)
    from hcc_sempath.training.engine import (
        _eligible_selection_epoch_count,
    )

    eligible_epochs = _eligible_selection_epoch_count(
        current_global_step=0,
        steps_per_epoch=steps_per_epoch_lower_bound,
        start_epoch=1,
        expected_epochs=epochs,
        selection_start_step=selection_start_step,
    )
    required = int(
        cfg.get("train", {}).get(
            "selection_minimum_eligible_epochs",
            1,
        )
    )
    if eligible_epochs < required:
        raise ValueError(
            "A0 budget cannot reach enough post-ramp selection epochs: "
            f"records_lower_bound={lower_bound} "
            f"steps_per_epoch_lower_bound={steps_per_epoch_lower_bound} "
            f"selection_start_step={selection_start_step} "
            f"eligible_epochs={eligible_epochs} required={required}"
        )
    return {
        "selected_train_records": records,
        "optimizer_visible_records_lower_bound": lower_bound,
        "steps_per_epoch_lower_bound": steps_per_epoch_lower_bound,
        "selection_start_step": selection_start_step,
        "eligible_epochs_lower_bound": eligible_epochs,
        "required_eligible_epochs": required,
    }


def _population_validation_contract(
    cfg: dict[str, Any],
    selected_val_tiles: list[Path],
    *,
    expert_tiles: int,
    iac_sha256: dict[str, str] | None = None,
    expert_asset_sha256: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Bind the deterministic population-validation and teacher probe sizes."""

    from iatro.iac import read_header

    records = sum(
        int(read_header(path)["num_records"])
        for path in selected_val_tiles
    )
    max_records = int(cfg.get("data", {}).get("max_val_records", 0))
    if max_records > 0:
        records = min(records, max_records)
    # The teacher-retention probe excludes the complete classification/spatial train+validation
    # expert union. Subtracting that full union is conservative because many of
    # its rows may live outside the selected validation packages.
    lower_bound = max(0, records - int(expert_tiles))
    batch_size = int(cfg.get("train", {}).get("batch_size", 0))
    max_val_batches = int(
        cfg.get("train", {}).get("max_val_batches", 0)
    )
    max_eval_batches = int(
        cfg.get("train", {}).get("max_eval_batches", 0)
    )
    pairwise_samples = int(
        cfg.get("train", {}).get(
            "eval_pairwise_max_samples",
            4096,
        )
    )
    if batch_size <= 0:
        raise ValueError(
            "A0 population validation requires a positive batch size"
        )
    if max_val_batches <= 0 or max_eval_batches <= 0:
        raise ValueError(
            "A0 requires positive max_val_batches and max_eval_batches"
        )
    if max_eval_batches > max_val_batches:
        raise ValueError(
            "A0 max_eval_batches cannot exceed max_val_batches"
        )
    teacher_tiles = max_eval_batches * batch_size
    if pairwise_samples <= 0 or pairwise_samples > teacher_tiles:
        raise ValueError(
            "A0 eval_pairwise_max_samples must be positive and cannot "
            "exceed the fixed teacher-retention probe"
        )
    batches_lower_bound = (
        lower_bound + batch_size - 1
    ) // batch_size
    if batches_lower_bound < max_val_batches:
        raise ValueError(
            "selected population-validation view is too small for the "
            "prespecified fixed validation probe: "
            f"batches_lower_bound={batches_lower_bound} "
            f"required={max_val_batches}"
        )
    selected_package_contract = {
        str(path.resolve()): (
            str(iac_sha256[str(path.resolve())])
            if iac_sha256 is not None
            else file_sha256(path)
        )
        for path in selected_val_tiles
    }
    probe_definition = {
        "selected_val_tile_packages": selected_package_contract,
        "expert_exclusion_assets": dict(
            sorted((expert_asset_sha256 or {}).items())
        ),
        "runtime_seed": int(
            cfg.get("runtime", {}).get("seed", 13)
        ),
        "val_tile_fraction": float(
            cfg.get("data", {}).get("val_tile_fraction", 1.0)
        ),
        "dynamic_package_sampling": bool(
            cfg.get("data", {}).get(
                "dynamic_package_sampling",
                True,
            )
        ),
        "package_multiprocessing": bool(
            cfg.get("data", {}).get(
                "package_multiprocessing",
                False,
            )
        ),
        "package_chunk_size": int(
            cfg.get("data", {}).get("package_chunk_size", 64)
        ),
        "package_buffer_batches": int(
            cfg.get("data", {}).get("package_buffer_batches", 4)
        ),
        "batch_size": batch_size,
        "max_val_batches": max_val_batches,
        "max_eval_batches": max_eval_batches,
        "eval_pairwise_max_samples": pairwise_samples,
    }
    return {
        "selected_val_records": records,
        "selected_val_packages": len(selected_val_tiles),
        "population_val_records_lower_bound": lower_bound,
        "population_val_batches_lower_bound": batches_lower_bound,
        "ordinary_validation_batches": max_val_batches,
        "ordinary_validation_tiles": max_val_batches * batch_size,
        "teacher_retention_batches": max_eval_batches,
        "teacher_retention_tiles": teacher_tiles,
        "teacher_relation_tiles": pairwise_samples,
        "probe_definition_sha256": _canonical_digest(
            probe_definition
        ),
    }


def _classification_split_counts(
    path: Path,
    class_names: list[str],
) -> dict[str, dict[str, int]]:
    counts = {
        split: {name: 0 for name in class_names}
        for split in ("train", "val")
    }
    seen: set[str] = set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "tile_id",
            "classification_label",
            "source_split",
            "adjudicated",
            "iac",
            "row",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "prototype supervision manifest missing columns: "
                f"{sorted(missing)}"
            )
        for row in reader:
            tile_id = str(row["tile_id"]).strip()
            if tile_id in seen:
                raise ValueError(
                    f"duplicate prototype supervision tile_id: {tile_id}"
                )
            seen.add(tile_id)
            split = str(row["source_split"]).strip()
            if split not in counts:
                continue
            label = str(row["classification_label"]).strip()
            if label not in counts[split]:
                raise ValueError(
                    f"unknown classification label: {label}"
                )
            if not str(row["iac"]).strip() or not str(row["row"]).strip():
                raise ValueError(
                    f"classification provenance missing: {tile_id}"
                )
            if str(row["adjudicated"]).strip().lower() not in {
                "1",
                "true",
                "yes",
                "y",
                "adjudicated",
            }:
                continue
            counts[split][label] += 1
    for split, split_counts in counts.items():
        empty = [
            label
            for label, count in split_counts.items()
            if count <= 0
        ]
        if empty:
            raise ValueError(
                f"classification {split} split has empty classes: {empty}"
            )
    invalid_train_counts = {
        label: count
        for label, count in counts["train"].items()
        if count != 400
    }
    if invalid_train_counts:
        raise ValueError(
            "A0 training classification bank must contain exactly "
            f"400 tiles per class: {invalid_train_counts}"
        )
    return counts


def _spatial_split_counts(
    path: Path,
) -> dict[str, dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    annotations = (
        payload.get("annotations")
        if isinstance(payload, dict)
        else None
    )
    if not isinstance(annotations, dict):
        raise ValueError(
            "spatial manifest must contain an annotations object"
        )
    component_names = [
        str(value)
        for value in payload.get("spatial_prototypes", [])
    ]
    if not component_names:
        component_names = [
            str(item["name"])
            for item in payload.get(
                "label_definitions",
                {},
            ).get("spatial", [])
            if isinstance(item, dict) and item.get("name")
        ]
    if len(component_names) != 11 or len(set(component_names)) != 11:
        raise ValueError(
            "A0 requires the fixed eleven-component spatial contract"
        )
    counts = {
        split: {name: 0 for name in component_names}
        for split in ("train", "val")
    }
    seen: dict[str, str] = {}
    for raw in annotations.values():
        if not isinstance(raw, dict) or not raw.get("tile_id"):
            continue
        tile_id = str(raw["tile_id"])
        split = str(
            raw.get("split", raw.get("source_split", "train"))
        )
        if split not in counts:
            continue
        previous = seen.get(tile_id)
        if previous is not None and previous != split:
            raise ValueError(
                f"train/validation spatial tile overlap: {tile_id}"
            )
        seen[tile_id] = split
        if not bool(raw.get("roi_reviewed", False)):
            raise ValueError(
                f"spatial annotation is not physician-reviewed: {tile_id}"
            )
        if not str(raw.get("iac") or raw.get("iac_path") or "").strip():
            raise ValueError(
                f"spatial provenance is missing IAC: {tile_id}"
            )
        if raw.get("row") is None:
            raise ValueError(
                f"spatial provenance is missing row: {tile_id}"
            )
        tile_components = {
            str(roi.get("attribute"))
            for roi in raw.get("roi", [])
            if isinstance(roi, dict)
            and roi.get("attribute") in counts[split]
        }
        for component in tile_components:
            counts[split][component] += 1
    for split, split_counts in counts.items():
        empty = [
            component
            for component, count in split_counts.items()
            if count <= 0
        ]
        if empty:
            raise ValueError(
                f"spatial {split} split has empty components: {empty}"
            )
    return counts


def _spatial_split_tile_counts(path: Path) -> dict[str, int]:
    return {
        split: len(tile_ids)
        for split, tile_ids in _spatial_split_tile_ids(path).items()
    }


def _classification_split_tile_ids(
    path: Path,
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {"train": set(), "val": set()}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            split = str(row.get("source_split", "")).strip()
            tile_id = str(row.get("tile_id", "")).strip()
            adjudicated = str(row.get("adjudicated", "")).strip().lower()
            if (
                split in result
                and tile_id
                and adjudicated
                in {"1", "true", "yes", "y", "adjudicated"}
            ):
                result[split].add(tile_id)
    return result


def _spatial_split_tile_ids(path: Path) -> dict[str, set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    annotations = payload.get("annotations", {})
    result: dict[str, set[str]] = {"train": set(), "val": set()}
    for raw in annotations.values():
        if not isinstance(raw, dict) or not raw.get("tile_id"):
            continue
        split = str(
            raw.get("split", raw.get("source_split", "train"))
        )
        if split in result:
            result[split].add(str(raw["tile_id"]))
    return result


def _expert_split_tile_counts(
    classification_path: Path,
    spatial_path: Path,
) -> dict[str, int]:
    classification = _classification_split_tile_ids(
        classification_path
    )
    spatial = _spatial_split_tile_ids(spatial_path)
    train = classification["train"] | spatial["train"]
    val = classification["val"] | spatial["val"]
    overlap = train & val
    if overlap:
        raise ValueError(
            "train/validation expert tile overlap across classification/spatial supervision: "
            f"count={len(overlap)} sample={next(iter(sorted(overlap)))}"
        )
    return {
        "train": len(train),
        "val": len(val),
        "overlap": 0,
    }


def preflight_assets(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from hcc_sempath.modeling.models import (
        STUDENT_PRETRAINED_PATH,
        STUDENT_PRETRAINED_SHA256,
    )

    paths = _required_asset_paths(cfg)
    class_names = [
        str(value)
        for value in cfg.get("model", {}).get(
            "classification_class_names",
            [],
        )
    ]
    if len(class_names) != 7 or len(set(class_names)) != 7:
        raise ValueError(
            "A0 requires the fixed seven-class classification contract"
        )
    for teacher in TEACHERS:
        registry = torch.load(
            paths[f"prototype_{teacher}"],
            map_location="cpu",
            weights_only=False,
        )
        if [str(value) for value in registry.get("names", [])] != (
            class_names
        ):
            raise ValueError(
                f"{teacher} prototype class order is not the A0 contract"
            )
        if [int(value) for value in registry.get("counts", [])] != (
            [400] * len(class_names)
        ):
            raise ValueError(
                f"{teacher} prototypes were not built from the fixed "
                "400-tile-per-class training bank"
            )
    counts = _classification_split_counts(
        paths["prototype_supervision_manifest_path"],
        class_names,
    )
    spatial_counts = _spatial_split_counts(
        paths["spatial_manifest_path"],
    )
    spatial_tile_counts = _spatial_split_tile_counts(
        paths["spatial_manifest_path"],
    )
    expert_tile_counts = _expert_split_tile_counts(
        paths["prototype_supervision_manifest_path"],
        paths["spatial_manifest_path"],
    )
    if set(
        str(value)
        for value in cfg["data"].get(
            "prototype_supervision_train_splits",
            ["train"],
        )
    ) != {"train"}:
        raise ValueError(
            "prototype_supervision_train_splits must be [train]"
        )
    if set(
        str(value)
        for value in cfg["data"].get(
            "prototype_supervision_val_splits",
            ["val"],
        )
    ) != {"val"}:
        raise ValueError(
            "prototype_supervision_val_splits must be [val]"
        )
    if set(
        str(value)
        for value in cfg["data"].get(
            "spatial_train_splits",
            ["train"],
        )
    ) != {"train"}:
        raise ValueError("spatial_train_splits must be [train]")
    if set(
        str(value)
        for value in cfg["data"].get(
            "spatial_val_splits",
            ["val"],
        )
    ) != {"val"}:
        raise ValueError("spatial_val_splits must be [val]")
    complete_tiles, complete_teachers = _resolved_training_iac_paths(
        cfg,
        complete=True,
    )
    selected_train_tiles, _ = _resolved_training_iac_paths(
        cfg,
        complete=False,
        splits=("train",),
    )
    selected_val_tiles, _ = _resolved_training_iac_paths(
        cfg,
        complete=False,
        splits=("val",),
    )
    iac_paths = sorted(
        {
            *complete_tiles,
            *(
                path
                for teacher_paths in complete_teachers.values()
                for path in teacher_paths
            ),
        }
    )
    missing_iac = [
        str(path) for path in iac_paths if not path.is_file()
    ]
    if missing_iac:
        raise FileNotFoundError(
            "A0 training IAC assets are missing: "
            + ", ".join(missing_iac[:10])
        )
    if not STUDENT_PRETRAINED_PATH.is_file():
        raise FileNotFoundError(
            "fixed DINOv2 student initialization is missing: "
            f"{STUDENT_PRETRAINED_PATH}"
        )
    student_digest = file_sha256(STUDENT_PRETRAINED_PATH)
    if student_digest != STUDENT_PRETRAINED_SHA256:
        raise ValueError(
            "fixed DINOv2 student initialization SHA-256 mismatch"
        )
    static_sha256 = {
        name: file_sha256(path)
        for name, path in paths.items()
    }
    iac_sha256 = {
        str(path): file_sha256(path)
        for path in iac_paths
    }
    formal_asset_sha256 = {
        "static_files": static_sha256,
        "iac_packages": iac_sha256,
        "student_pretrained": {
            "path": str(STUDENT_PRETRAINED_PATH.resolve()),
            "sha256": student_digest,
        },
    }
    schedule = _population_schedule_contract(
        cfg,
        selected_train_tiles,
        classification_val_tiles=sum(counts["val"].values()),
        spatial_val_tiles=spatial_tile_counts["val"],
    )
    validation = _population_validation_contract(
        cfg,
        selected_val_tiles,
        expert_tiles=(
            expert_tile_counts["train"]
            + expert_tile_counts["val"]
        ),
        iac_sha256=iac_sha256,
        expert_asset_sha256={
            key: static_sha256[key]
            for key in (
                "prototype_supervision_manifest_path",
                "spatial_manifest_path",
            )
        },
    )
    return {
        "paths": {
            name: str(path)
            for name, path in paths.items()
        },
        "sha256": static_sha256,
        "iac_sha256": iac_sha256,
        "student_pretrained": formal_asset_sha256[
            "student_pretrained"
        ],
        "formal_asset_sha256": formal_asset_sha256,
        "classification_counts": counts,
        "spatial_counts": spatial_counts,
        "spatial_tile_counts": spatial_tile_counts,
        "expert_tile_counts": expert_tile_counts,
        "population_schedule": schedule,
        "population_validation": validation,
    }


def _formal_base_config(
    base_cfg: dict[str, Any],
    *,
    epochs: int,
) -> dict[str, Any]:
    """Freeze study-wide settings before preflight and contract hashing."""

    cfg = deep_merge({}, base_cfg)
    cfg.setdefault("runtime", {})
    cfg.setdefault("data", {})
    cfg.setdefault("train", {})
    if int(cfg["runtime"].get("seed", 13)) != 13:
        raise ValueError("formal A0 search requires runtime.seed=13")
    cfg["runtime"]["seed"] = 13
    # Trial configs repeat these values defensively, but the formal population
    # view must already be bound when its schedule and study digest are built.
    cfg["data"]["train_tile_fraction"] = 0.10
    cfg["data"]["val_tile_fraction"] = 0.10
    # Train and validation process pools coexist during validation. Keeping
    # either pool persistent exceeded the formal host's 90 GiB memory cgroup.
    cfg["data"]["persistent_workers"] = False
    cfg["data"]["val_persistent_workers"] = False
    cfg["train"]["epochs"] = int(epochs)
    # These diagnostics are useful in development runs but add an unrelated
    # 20–30 second probe and retained-gradient pass to every formal trial.
    cfg["train"]["development_probe_interval_steps"] = 0
    cfg["train"]["gradient_diagnostic_interval_steps"] = 0
    return cfg


def trial_config(
    base_cfg: dict[str, Any],
    trial: optuna.Trial,
    output_dir: Path,
    epochs: int,
    *,
    selection_baseline: dict[str, float] | None = None,
    formal_asset_sha256: dict[str, Any] | None = None,
    formal_source: dict[str, str] | None = None,
    formal_study_contract_sha256: str | None = None,
    formal_population_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = deep_merge({}, base_cfg)
    cfg.setdefault("runtime", {})
    cfg.setdefault("data", {})
    cfg.setdefault("loss", {})
    cfg.setdefault("train", {})
    cfg["runtime"]["output_dir"] = str(output_dir)
    cfg["runtime"]["seed"] = 13

    # Only the population view is reduced. Both human expert banks stay full.
    cfg["data"]["train_tile_fraction"] = 0.10
    cfg["data"]["val_tile_fraction"] = 0.10
    cfg["data"]["require_complete_expert_validation"] = True
    cfg["data"]["spatial_train_splits"] = ["train"]
    cfg["data"]["spatial_val_splits"] = ["val"]
    cfg["data"]["prototype_supervision_train_splits"] = ["train"]
    cfg["data"]["prototype_supervision_val_splits"] = ["val"]
    if formal_asset_sha256 is not None:
        cfg["data"]["formal_asset_sha256"] = deep_merge(
            {},
            formal_asset_sha256,
        )
    if formal_source is not None:
        cfg["data"]["formal_source"] = dict(formal_source)
    if formal_study_contract_sha256 is not None:
        cfg["data"]["formal_study_contract_sha256"] = str(
            formal_study_contract_sha256
        )
    if formal_population_validation is not None:
        cfg["data"]["formal_population_validation"] = deep_merge(
            {},
            formal_population_validation,
        )
    cfg["data"]["num_workers"] = int(
        cfg["data"].get("num_workers", 16)
    )
    cfg["data"]["prefetch_factor"] = int(
        cfg["data"].get("prefetch_factor", 3)
    )
    cfg["data"]["persistent_workers"] = bool(
        cfg["data"].get("persistent_workers", False)
    )
    cfg["data"]["val_persistent_workers"] = bool(
        cfg["data"].get("val_persistent_workers", False)
    )
    cfg["data"]["dynamic_package_sampling"] = bool(
        cfg["data"].get("dynamic_package_sampling", True)
    )
    cfg["data"]["tensor_collate"] = bool(
        cfg["data"].get("tensor_collate", True)
    )
    cfg["data"]["package_chunk_size"] = int(
        cfg["data"].get("package_chunk_size", 64)
    )

    cfg["train"]["batch_size"] = int(
        cfg["train"].get("batch_size", 512)
    )
    cfg["train"]["epochs"] = int(epochs)
    cfg["train"]["lr"] = trial.suggest_float(
        "lr",
        **SEARCH_SPACE["lr"],
    )
    cfg["train"]["weight_decay"] = trial.suggest_float(
        "weight_decay",
        **SEARCH_SPACE["weight_decay"],
    )
    cfg["loss"]["spatial_weight"] = trial.suggest_float(
        "spatial_weight",
        **SEARCH_SPACE["spatial_weight"],
    )
    cfg["train"]["max_grad_norm"] = 1.0
    cfg["train"]["max_val_batches"] = int(
        cfg["train"].get("max_val_batches", 128)
    )
    cfg["train"]["max_eval_batches"] = int(
        cfg["train"].get("max_eval_batches", 128)
    )
    cfg["train"]["eval_pairwise_max_samples"] = int(
        cfg["train"].get("eval_pairwise_max_samples", 4096)
    )
    # The old intra-epoch population-loss stop is not a selection signal.
    cfg["train"]["development_early_stop"] = False
    cfg["train"]["development_probe_interval_steps"] = 0
    cfg["train"]["gradient_diagnostic_interval_steps"] = 0
    cfg["train"]["selection_early_stop"] = True
    cfg["train"].setdefault(
        "selection_metric_weights",
        {
            "teacher": 0.26,
            "classification": 0.28,
            "spatial": 0.46,
        },
    )
    cfg["train"]["selection_minimum_eligible_epochs"] = int(
        cfg["train"].get("selection_minimum_eligible_epochs", 8)
    )
    if selection_baseline is not None:
        cfg["train"]["selection_metric_baseline"] = {
            name: float(selection_baseline[name])
            for name in SELECTION_COMPONENTS
        }
    cfg["train"]["selection_early_stop_patience"] = int(
        cfg["train"].get("selection_early_stop_patience", 3)
    )
    cfg["train"]["selection_early_stop_relative_delta"] = float(
        cfg["train"].get(
            "selection_early_stop_relative_delta",
            0.005,
        )
    )
    cfg["train"]["early_stop_teacher_alignment"] = False
    cfg["train"]["log_interval"] = 200
    cfg["train"]["progress"] = "tqdm"
    cfg["train"]["tensorboard"] = False
    cfg["train"]["tensorboard_batch_interval"] = 0
    return cfg


def _metric_value(
    row: dict[str, str],
    key: str,
) -> float:
    try:
        value = float(row.get(key, "") or "nan")
    except ValueError as exc:
        raise ValueError(
            f"non-numeric objective metric {key}={row.get(key)!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"non-finite objective metric {key}={value}"
        )
    return value


def score_row(
    row: dict[str, str],
    objective: str = "selection_loss",
    *,
    expected_weights: dict[str, float] | None = None,
    expected_baseline: dict[str, float] | None = None,
    expected_start_step: int | None = None,
) -> float:
    if objective != "selection_loss":
        raise ValueError(
            f"unsupported A0 objective: {objective}"
        )
    if str(row.get("selection_eligible", "")).strip().lower() not in {
        "1",
        "true",
        "yes",
    }:
        raise ValueError("selection row precedes the configured ramp end")
    global_step = int(_metric_value(row, "global_step"))
    row_start_step = int(_metric_value(row, "selection_start_step"))
    if global_step < row_start_step:
        raise ValueError(
            "selection row is marked eligible before its start step"
        )
    if (
        expected_start_step is not None
        and row_start_step != int(expected_start_step)
    ):
        raise ValueError(
            "selection row start step differs from the study contract"
        )
    reported = _metric_value(row, "selection_loss")
    weights = {
        component: _metric_value(
            row,
            f"selection_{component}_weight",
        )
        for component in SELECTION_COMPONENTS
    }
    if (
        any(value <= 0.0 for value in weights.values())
        or not math.isclose(
            sum(weights.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-8,
        )
    ):
        raise ValueError("selection weights must be positive and sum to one")
    if expected_weights is not None:
        for component in SELECTION_COMPONENTS:
            if not math.isclose(
                weights[component],
                float(expected_weights[component]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "selection row weight differs from the study contract: "
                    f"component={component}"
                )
    recomputed = 0.0
    for component in SELECTION_COMPONENTS:
        raw = _metric_value(row, f"selection_{component}_raw")
        baseline = _metric_value(
            row,
            f"selection_{component}_baseline",
        )
        normalized = _metric_value(
            row,
            f"selection_{component}_normalized",
        )
        if baseline <= 0.0 or not math.isclose(
            normalized,
            raw / baseline,
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            raise ValueError(
                "selection normalization mismatch: "
                f"component={component}"
            )
        if (
            expected_baseline is not None
            and not math.isclose(
                baseline,
                float(expected_baseline[component]),
                rel_tol=1e-6,
                abs_tol=1e-8,
            )
        ):
            raise ValueError(
                "selection row baseline differs from the shared study "
                f"baseline: component={component}"
            )
        recomputed += weights[component] * normalized
    if not math.isclose(
        reported,
        recomputed,
        rel_tol=1e-6,
        abs_tol=1e-7,
    ):
        raise ValueError(
            "selection_loss does not equal its normalized validation terms: "
            f"reported={reported} recomputed={recomputed}"
        )
    if not math.isclose(
        _metric_value(row, "selection_teacher_raw"),
        _metric_value(row, "teacher_validation_loss"),
        rel_tol=1e-6,
        abs_tol=1e-7,
    ):
        raise ValueError("selection teacher term is not fixed teacher loss")
    if not math.isclose(
        _metric_value(row, "selection_classification_raw"),
        _metric_value(
            row,
            "expert_val_classification_balanced_cross_entropy",
        ),
        rel_tol=1e-6,
        abs_tol=1e-7,
    ):
        raise ValueError("selection classification term mismatch")
    if not math.isclose(
        _metric_value(row, "selection_spatial_raw"),
        _metric_value(row, "expert_val_spatial"),
        rel_tol=1e-6,
        abs_tol=1e-7,
    ):
        raise ValueError("selection spatial term mismatch")
    evaluated_classes = _metric_value(
        row,
        "expert_val_classification_evaluated_classes",
    )
    total_classes = _metric_value(
        row,
        "expert_val_classification_total_classes",
    )
    if evaluated_classes != total_classes or total_classes != 7:
        raise ValueError(
            "selection row did not evaluate all seven classification classes"
        )
    if _metric_value(
        row,
        "expert_val_spatial_explicit_negative_pairs",
    ) <= 0:
        raise ValueError(
            "selection row has no explicit-negative spatial supervision"
        )
    return reported


def read_metric_rows(
    metrics_path: Path,
) -> list[dict[str, str]]:
    if not metrics_path.exists():
        return []
    payload = metrics_path.read_text(encoding="utf-8")
    # Training and the Optuna coordinator are separate processes. Ignore a
    # final row until its terminating newline is visible, so polling cannot
    # mistake a concurrent partial append for a malformed scientific metric.
    if payload and not payload.endswith(("\n", "\r")):
        payload = payload.rsplit("\n", 1)[0] + "\n"
    if not payload.strip():
        return []
    return list(csv.DictReader(io.StringIO(payload)))


def _selection_row_eligible(row: dict[str, str]) -> bool:
    return str(row.get("selection_eligible", "")).strip().lower() in {
        "1",
        "true",
        "yes",
    }


def stream_process(
    process: subprocess.Popen[str],
    log_path: Path,
) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        assert process.stdout is not None
        for character in iter(lambda: process.stdout.read(1), ""):
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


def parse_cuda_devices(
    raw_devices: str,
    *,
    parallel_trials: int,
) -> tuple[str | None, ...]:
    if int(parallel_trials) <= 0:
        raise ValueError("--parallel-trials must be positive")
    devices = tuple(
        device.strip()
        for device in str(raw_devices).split(",")
        if device.strip()
    )
    if not devices:
        if int(parallel_trials) != 1:
            raise ValueError(
                "--devices is required when --parallel-trials is greater "
                "than one"
            )
        return (None,)
    if len(set(devices)) != len(devices):
        raise ValueError("--devices must not contain duplicates")
    if len(devices) < int(parallel_trials):
        raise ValueError(
            "--devices must provide at least one distinct GPU per "
            "parallel trial"
        )
    return devices


def load_verified_preflight_assets(
    manifest_path: Path,
    *,
    source: dict[str, str],
    base_config_sha256: str,
) -> dict[str, Any]:
    manifest = load_yaml(manifest_path)
    if manifest.get("source") != source:
        raise RuntimeError(
            "verified preflight source differs from the executable "
            "scientific source"
        )
    if manifest.get("base_config_sha256") != base_config_sha256:
        raise RuntimeError(
            "verified preflight base config differs from the resolved "
            "formal config"
        )
    assets = manifest.get("assets")
    if not isinstance(assets, dict):
        raise RuntimeError("verified preflight manifest has no assets")
    for key in (
        "formal_asset_sha256",
        "population_schedule",
        "population_validation",
    ):
        if key not in assets:
            raise RuntimeError(
                f"verified preflight assets are missing {key}"
            )
    return assets


def write_verified_iac_receipt(
    path: Path,
    formal_asset_sha256: dict[str, Any],
) -> Path:
    iac_packages = formal_asset_sha256.get("iac_packages")
    if not isinstance(iac_packages, dict):
        raise RuntimeError(
            "formal asset contract has no IAC package mapping"
        )
    files: dict[str, dict[str, int | str]] = {}
    for raw_path, raw_digest in sorted(iac_packages.items()):
        resolved = Path(str(raw_path)).resolve()
        stat = resolved.stat()
        files[str(resolved)] = {
            "sha256": str(raw_digest),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "device": stat.st_dev,
            "inode": stat.st_ino,
        }
    payload = {
        "schema": "hcc-sempath-verified-asset-receipt-v1",
        "files": files,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def train_with_pruning(
    *,
    trial: optuna.Trial,
    cfg_path: Path,
    output_dir: Path,
    python_bin: str,
    repo: Path,
    poll_sec: float,
    expected_weights: dict[str, float],
    expected_baseline: dict[str, float] | None,
    expected_start_step: int,
    cuda_visible_device: str | None = None,
    verified_asset_receipt: Path | None = None,
    objective: str = "selection_loss",
) -> float:
    log_path = output_dir / "trial.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src")
    env["PYTHONNOUSERSITE"] = "1"
    if cuda_visible_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_device)
    if verified_asset_receipt is not None:
        env["HCC_SEMPATH_VERIFIED_ASSET_RECEIPT"] = str(
            verified_asset_receipt
        )
    command = [
        python_bin,
        "-m",
        "hcc_sempath.cli.main",
        "train",
        "--config",
        str(cfg_path),
    ]
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
    thread = threading.Thread(
        target=stream_process,
        args=(process, log_path),
        daemon=True,
    )
    thread.start()
    metrics_path = output_dir / "metrics.csv"
    reported_epochs: set[int] = set()
    reported_eligible_steps = 0
    best_score = float("inf")
    best_epoch = 0

    def drain_metrics() -> None:
        nonlocal reported_eligible_steps, best_score, best_epoch
        for row in read_metric_rows(metrics_path):
            epoch = int(float(row.get("epoch", "0") or 0))
            if epoch <= 0 or epoch in reported_epochs:
                continue
            reported_epochs.add(epoch)
            if not _selection_row_eligible(row):
                continue
            reported_eligible_steps += 1
            score = score_row(
                row,
                objective,
                expected_weights=expected_weights,
                expected_baseline=expected_baseline,
                expected_start_step=expected_start_step,
            )
            if score < best_score:
                best_score = score
                best_epoch = epoch
            # Optuna pruning is defined over post-ramp observations, not
            # absolute training epochs. This gives every trial the same
            # five-observation warm-up even when the ramp ends late.
            trial.report(score, step=reported_eligible_steps)
            trial.set_user_attr(f"epoch_{epoch}_score", score)
            trial.set_user_attr(
                f"eligible_step_{reported_eligible_steps}_epoch",
                epoch,
            )
            if trial.should_prune():
                raise optuna.TrialPruned(
                    "pruned at "
                    f"eligible_step={reported_eligible_steps} "
                    f"epoch={epoch} loss={score:.6f}"
                )

    try:
        while process.poll() is None:
            drain_metrics()
            time.sleep(float(poll_sec))
        if process.returncode != 0:
            raise RuntimeError(
                f"training failed returncode={process.returncode}; "
                f"see {log_path}"
            )
        # The process can finish one or more epochs between poll intervals.
        # Drain once after successful exit so the final eligible observations
        # participate in pruning and the persisted TPE history.
        drain_metrics()
        rows = [
            row
            for row in read_metric_rows(metrics_path)
            if _selection_row_eligible(row)
        ]
        if not rows:
            raise RuntimeError(
                "training produced no eligible selection metrics: "
                f"{metrics_path}"
            )
        best_row: dict[str, str] | None = None
        for row in rows:
            score = score_row(
                row,
                objective,
                expected_weights=expected_weights,
                expected_baseline=expected_baseline,
                expected_start_step=expected_start_step,
            )
            epoch = int(float(row["epoch"]))
            if score < best_score:
                best_score = score
                best_epoch = epoch
            if epoch == best_epoch:
                best_row = row
        if best_row is None:
            raise RuntimeError(
                "best Optuna epoch is absent from metrics.csv"
            )
        checkpoint = output_dir / "checkpoints" / "best.pt"
        if not checkpoint.is_file():
            raise RuntimeError(
                f"best selection checkpoint is missing: {checkpoint}"
            )
        import torch

        checkpoint_payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if (
            not bool(checkpoint_payload.get("run_complete", False))
            or not bool(
                checkpoint_payload.get("selection_finalized", False)
            )
        ):
            raise RuntimeError(
                "best checkpoint was not finalized by a completed run"
            )
        if int(checkpoint_payload.get("epoch", -1)) != best_epoch:
            raise RuntimeError(
                "best checkpoint epoch differs from metrics.csv: "
                f"checkpoint={checkpoint_payload.get('epoch')} "
                f"metrics={best_epoch}"
            )
        if not math.isclose(
            float(
                checkpoint_payload.get(
                    "best_selection_loss",
                    float("nan"),
                )
            ),
            best_score,
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            raise RuntimeError(
                "best checkpoint selection loss differs from metrics.csv"
            )
        trial.set_user_attr("output_dir", str(output_dir))
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("best_checkpoint", str(checkpoint))
        trial.set_user_attr("best_observed_score", best_score)
        for metric in RESULT_METRICS:
            trial.set_user_attr(
                f"best_{metric}",
                best_row.get(metric),
            )
        return best_score
    finally:
        if process.poll() is None:
            terminate_process(process)
        thread.join(timeout=30)


def _normalize_storage(storage: str, repo: Path) -> str:
    prefix = "sqlite:///"
    if not storage.startswith(prefix):
        return storage
    raw = storage.removeprefix(prefix)
    path = Path(raw)
    if not path.is_absolute():
        path = repo / path
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def _coordinator_lock_path(
    storage: str,
    *,
    output_root: Path,
    study_name: str,
) -> Path:
    safe_study_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", study_name)
    prefix = "sqlite:///"
    if storage.startswith(prefix):
        database = Path(storage.removeprefix(prefix)).resolve()
        return database.with_name(
            f".{database.name}.{safe_study_name}.coordinator.lock"
        )
    return output_root / f".{safe_study_name}.coordinator.lock"


def _study_contract(
    *,
    source: dict[str, str],
    base_config_sha256: str,
    asset_sha256: dict[str, Any],
    epochs: int,
    selection_weights: dict[str, float],
    sampler_seed: int,
    sampler_constant_liar: bool,
    total_trial_budget: int,
    population_validation: dict[str, Any],
) -> dict[str, Any]:
    sampler = {
        "name": "trial_seeded_tpe_v1",
        "optuna_version": optuna.__version__,
        "n_startup_trials": 6,
        "multivariate": True,
        "group": True,
    }
    if sampler_constant_liar:
        sampler["constant_liar"] = True
    return {
        "source": source,
        "base_config_sha256": base_config_sha256,
        "asset_sha256": asset_sha256,
        "objective": "selection_loss",
        "selection_formula": "sum_k weight_k * metric_k / epoch0_metric_k",
        "selection_components": list(SELECTION_COMPONENTS),
        "selection_weights": selection_weights,
        "selection_baseline_binding": (
            "first_trial_epoch0_then_reused_and_revalidated"
        ),
        "direction": "minimize",
        "epochs_per_trial": int(epochs),
        "population_fraction": 0.10,
        "population_validation": dict(population_validation),
        "search_space": SEARCH_SPACE,
        "sampler": sampler,
        "pruner": {
            "name": "MedianPruner",
            "n_startup_trials": 6,
            "n_warmup_post_ramp_observations": 5,
            "interval_steps": 1,
        },
        "fixed_random_seed": 13,
        "sampler_seed": int(sampler_seed),
        "total_trial_budget": int(total_trial_budget),
    }


def _bind_study_contract(
    study: optuna.Study,
    contract: dict[str, Any],
) -> str:
    digest = _canonical_digest(contract)
    previous = study.user_attrs.get("study_contract_sha256")
    if previous is not None and previous != digest:
        raise RuntimeError(
            "existing Optuna study has a different source/data/search "
            f"contract: existing={previous} requested={digest}"
        )
    study.set_user_attr("study_contract_sha256", digest)
    study.set_user_attr("objective", "selection_loss")
    study.set_user_attr("direction", "minimize")
    return digest


def _selection_baseline_from_path(
    path: Path,
) -> dict[str, float] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(
        SELECTION_COMPONENTS
    ):
        raise ValueError(f"invalid selection baseline artifact: {path}")
    result = {
        name: float(metrics[name])
        for name in SELECTION_COMPONENTS
    }
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in result.values()
    ):
        raise ValueError(f"invalid selection baseline values: {path}")
    return result


def _study_selection_baseline(
    study: optuna.Study,
) -> dict[str, float] | None:
    value = study.user_attrs.get("selection_metric_baseline")
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != set(
        SELECTION_COMPONENTS
    ):
        raise RuntimeError("study has an invalid shared selection baseline")
    return {
        name: float(value[name])
        for name in SELECTION_COMPONENTS
    }


def _bind_study_selection_baseline(
    study: optuna.Study,
    baseline: dict[str, float],
) -> str:
    normalized = {
        name: float(baseline[name])
        for name in SELECTION_COMPONENTS
    }
    previous = _study_selection_baseline(study)
    if previous is not None:
        for name in SELECTION_COMPONENTS:
            if not math.isclose(
                previous[name],
                normalized[name],
                rel_tol=1e-6,
                abs_tol=1e-8,
            ):
                raise RuntimeError(
                    "trial epoch-0 baseline differs from the shared study "
                    f"baseline: component={name}"
                )
        normalized = previous
    else:
        study.set_user_attr("selection_metric_baseline", normalized)
    digest = _canonical_digest(normalized)
    previous_digest = study.user_attrs.get(
        "selection_metric_baseline_sha256"
    )
    if previous_digest is not None and previous_digest != digest:
        raise RuntimeError("study selection baseline digest changed")
    study.set_user_attr("selection_metric_baseline_sha256", digest)
    rebound = _study_selection_baseline(study)
    if rebound is None:
        raise RuntimeError("failed to persist study selection baseline")
    return digest


def _remaining_study_executions(
    study: optuna.Study,
    total_trial_budget: int,
) -> int:
    non_waiting_trials = sum(
        trial.state != optuna.trial.TrialState.WAITING
        for trial in study.get_trials(deepcopy=False)
    )
    return max(0, int(total_trial_budget) - non_waiting_trials)


def _assert_resumable_study_states(study: optuna.Study) -> None:
    invalid = [
        trial
        for trial in study.get_trials(deepcopy=False)
        if trial.state
        in {
            optuna.trial.TrialState.RUNNING,
            optuna.trial.TrialState.FAIL,
        }
    ]
    if invalid:
        description = ", ".join(
            f"{trial.number}:{trial.state.name}"
            for trial in invalid
        )
        raise RuntimeError(
            "formal A0 study contains an interrupted or failed trial and "
            "cannot be resumed without changing its scientific budget: "
            f"{description}. Start a new study after fixing the cause."
        )


def export_study_artifacts(
    study: optuna.Study,
    *,
    output_root: Path,
    manifest: dict[str, Any],
    hash_best_checkpoint: bool = False,
) -> None:
    rows: list[dict[str, Any]] = []
    for trial in study.get_trials(deepcopy=False):
        row = {
            "number": trial.number,
            "state": trial.state.name,
            "selection_loss": trial.value,
            "lr": trial.params.get("lr"),
            "weight_decay": trial.params.get("weight_decay"),
            "spatial_weight": trial.params.get("spatial_weight"),
            "best_epoch": trial.user_attrs.get("best_epoch"),
        }
        row.update(
            {
                metric: trial.user_attrs.get(f"best_{metric}")
                for metric in RESULT_METRICS
            }
        )
        row["output_dir"] = trial.user_attrs.get("output_dir")
        row["failure_reason"] = trial.user_attrs.get("failure_reason")
        rows.append(row)

    summary_path = output_root / "study_summary.csv"
    temporary_summary = summary_path.with_name(
        f".{summary_path.name}.{os.getpid()}.tmp"
    )
    fieldnames = list(rows[0]) if rows else [
        "number",
        "state",
        "selection_loss",
        "lr",
        "weight_decay",
        "spatial_weight",
        "best_epoch",
    ]
    with temporary_summary.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_summary.replace(summary_path)

    trial_records = study.get_trials(deepcopy=False)
    total_trial_budget = int(manifest["n_trials_requested"])
    executed_trial_records = sum(
        trial.state != optuna.trial.TrialState.WAITING
        for trial in trial_records
    )
    study_complete = bool(
        executed_trial_records == total_trial_budget
        and all(
            trial.state
            in {
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.PRUNED,
            }
            for trial in trial_records
        )
    )
    manifest = {
        **manifest,
        "study_complete": study_complete,
        "executed_trial_records": executed_trial_records,
        "selection_metric_baseline": _study_selection_baseline(study),
        "selection_metric_baseline_sha256": study.user_attrs.get(
            "selection_metric_baseline_sha256"
        ),
    }
    atomic_write_yaml(output_root / "study_manifest.yaml", manifest)
    if not study.best_trials:
        return
    best = study.best_trial
    best_output = Path(str(best.user_attrs["output_dir"]))
    best_config = best_output / "config.yaml"
    checkpoint = Path(str(best.user_attrs["best_checkpoint"]))
    checkpoint_sha256 = (
        file_sha256(checkpoint)
        if hash_best_checkpoint and checkpoint.is_file()
        else None
    )
    raw_config_sha256 = best.user_attrs.get("config_sha256")
    if best_config.is_file():
        checkpoint_payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        scheduler_contract = checkpoint_payload.get(
            "scheduler_contract"
        )
        if not isinstance(scheduler_contract, dict):
            raise RuntimeError(
                "selected A0 checkpoint has no scheduler_contract"
            )
        selected_config = load_yaml(best_config)
        observed_config_sha256 = config_digest(selected_config)
        if (
            not isinstance(raw_config_sha256, str)
            or raw_config_sha256 != observed_config_sha256
        ):
            raise RuntimeError(
                "selected trial config differs from its Optuna provenance"
            )
        selected_config.setdefault("data", {})[
            "formal_a0_selection"
        ] = {
            "study_complete": study_complete,
            "study_name": str(manifest["study_name"]),
            "study_contract_sha256": str(
                manifest["study_contract_sha256"]
            ),
            "total_trial_budget": total_trial_budget,
            "executed_trial_records": executed_trial_records,
            "selected_trial": int(best.number),
            "selected_params": {
                str(name): float(value)
                for name, value in best.params.items()
            },
            "selection_loss": float(best.value),
            "best_epoch": int(best.user_attrs["best_epoch"]),
            "trial_config_sha256": raw_config_sha256,
            "best_checkpoint": str(checkpoint),
            "best_checkpoint_sha256": checkpoint_sha256,
            "scheduler_contract": dict(scheduler_contract),
        }
        atomic_write_yaml(
            output_root / "best_config.yaml",
            selected_config,
        )
    best_artifact = {
        "trial": best.number,
        "selection_loss": best.value,
        "params": best.params,
        "best_epoch": best.user_attrs.get("best_epoch"),
        "checkpoint": str(checkpoint),
        "checkpoint_size": (
            checkpoint.stat().st_size
            if checkpoint.is_file()
            else None
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "study_complete": study_complete,
        "trial_config_sha256": raw_config_sha256,
    }
    atomic_write_yaml(
        output_root / "best_artifact.yaml",
        best_artifact,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "A0 Optuna search using independent classification/spatial expert validation"
        )
    )
    parser.add_argument(
        "--base-config",
        default="experiments/configs/train_a0_optuna.example.yaml",
    )
    parser.add_argument(
        "--study-name",
        default="hcc_sempath_a0",
    )
    parser.add_argument(
        "--storage",
        default="sqlite:///runtime/optuna/hcc_sempath_a0.db",
    )
    parser.add_argument(
        "--output-root",
        default="runtime/optuna_runs",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument(
        "--study-trials",
        type=int,
        default=20,
        help="Strict total trial-record budget for this formal study.",
    )
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--timeout-hours", type=float, default=0.0)
    parser.add_argument("--poll-sec", type=float, default=20.0)
    parser.add_argument("--sampler-seed", type=int, default=13)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help=(
            "Scientific training source root. Defaults to the repository "
            "containing this coordinator."
        ),
    )
    parser.add_argument(
        "--verified-preflight-manifest",
        type=Path,
        default=None,
        help=(
            "Reuse a previously verified frozen-asset manifest instead "
            "of rehashing unchanged IAC, teacher, and tile assets."
        ),
    )
    parser.add_argument(
        "--parallel-trials",
        type=int,
        default=1,
        help=(
            "Number of independent single-GPU trials run concurrently "
            "by this coordinator. A free GPU immediately receives the "
            "next trial."
        ),
    )
    parser.add_argument(
        "--devices",
        default="",
        help=(
            "Comma-separated physical CUDA device identifiers. Required "
            "for parallel trials, for example: 0,1,2,3."
        ),
    )
    parser.add_argument(
        "--parallel-constant-liar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include in-flight configurations in parallel TPE proposals. "
            "Disable only when resuming a legacy sequential study whose "
            "contract predates this execution option."
        ),
    )
    args = parser.parse_args()

    if int(args.epochs) < 6:
        raise ValueError("A0 search requires at least 6 epochs per trial")
    if int(args.n_trials) < 0:
        raise ValueError("--n-trials must be non-negative")
    if int(args.study_trials) <= 0:
        raise ValueError("--study-trials must be positive")
    cuda_devices = parse_cuda_devices(
        str(args.devices),
        parallel_trials=int(args.parallel_trials),
    )

    coordinator_repo = Path(__file__).resolve().parents[2]
    repo = (
        coordinator_repo
        if args.source_root is None
        else args.source_root.resolve()
    )
    base_config_path = Path(args.base_config)
    if not base_config_path.is_absolute():
        base_config_path = repo / base_config_path
    base_config_path = base_config_path.resolve()
    base_cfg = _formal_base_config(
        load_yaml(base_config_path),
        epochs=int(args.epochs),
    )
    source = source_state(repo)
    resolved_base_config_sha256 = config_digest(base_cfg)
    assets = (
        preflight_assets(base_cfg)
        if args.verified_preflight_manifest is None
        else load_verified_preflight_assets(
            args.verified_preflight_manifest.resolve(),
            source=source,
            base_config_sha256=resolved_base_config_sha256,
        )
    )
    selection_weights = {
        name: float(
            base_cfg.get("train", {}).get(
                "selection_metric_weights",
                {
                    "teacher": 0.26,
                    "classification": 0.28,
                    "spatial": 0.46,
                },
            )[name]
        )
        for name in SELECTION_COMPONENTS
    }
    storage = _normalize_storage(str(args.storage), repo)
    output_root = (
        Path(args.output_root)
        if Path(args.output_root).is_absolute()
        else repo / args.output_root
    )
    output_root = output_root.resolve() / args.study_name
    output_root.mkdir(parents=True, exist_ok=True)
    verified_asset_receipt = (
        None
        if args.verified_preflight_manifest is None
        else write_verified_iac_receipt(
            output_root / "verified_asset_receipt.json",
            assets["formal_asset_sha256"],
        )
    )

    contract = _study_contract(
        source=source,
        base_config_sha256=resolved_base_config_sha256,
        asset_sha256=assets["formal_asset_sha256"],
        epochs=int(args.epochs),
        selection_weights=selection_weights,
        sampler_seed=int(args.sampler_seed),
        sampler_constant_liar=bool(args.parallel_constant_liar),
        total_trial_budget=int(args.study_trials),
        population_validation=assets["population_validation"],
    )
    contract_digest = _canonical_digest(contract)

    manifest = {
        "study_name": args.study_name,
        "purpose": "A0 baseline hyperparameter selection",
        "study_contract_sha256": contract_digest,
        "source": source,
        "base_config": str(base_config_path),
        "base_config_sha256": resolved_base_config_sha256,
        "assets": assets,
        "objective": "selection_loss",
        "direction": "minimize",
        "selection_formula": contract["selection_formula"],
        "selection_weights": selection_weights,
        "epochs_per_trial": int(args.epochs),
        "selection_early_stop": {
            "start_step": int(
                assets["population_schedule"][
                    "selection_start_step"
                ]
            ),
            "patience": int(
                base_cfg.get("train", {}).get(
                    "selection_early_stop_patience",
                    4,
                )
            ),
            "relative_delta": float(
                base_cfg.get("train", {}).get(
                    "selection_early_stop_relative_delta",
                    0.005,
                )
            ),
            "minimum_eligible_epochs": int(
                base_cfg.get("train", {}).get(
                    "selection_minimum_eligible_epochs",
                    1,
                )
            ),
        },
        "population_fraction": 0.10,
        "complete_classification_expert_bank": True,
        "complete_spatial_expert_bank": True,
        "n_trials_requested": int(args.study_trials),
        "timeout_hours": float(args.timeout_hours),
        "execution": {
            "parallel_trials": int(args.parallel_trials),
            "cuda_devices": [
                device
                for device in cuda_devices
                if device is not None
            ],
            "trial_gpu_mode": "one_independent_trial_per_device",
            "scheduling": "asynchronous_device_reuse",
            "constant_liar": bool(args.parallel_constant_liar),
            "coordinator_source_root": str(coordinator_repo),
        },
        "sampler": "TrialSeededTPESampler",
        "sampler_seed": int(args.sampler_seed),
        "n_startup_trials": 6,
        "pruner": (
            "MedianPruner(n_startup_trials=6,"
            f"n_warmup_steps={PRUNER_WARMUP_STEPS},interval_steps=1)"
        ),
        "baseline_params": BASELINE_PARAMS,
        "seeded_params": list(SEEDED_PARAMS),
        "search_space": SEARCH_SPACE,
        "fixed_loss_config": {
            key: value
            for key, value in base_cfg.get("loss", {}).items()
            if key != "spatial_weight"
        },
    }
    if int(args.n_trials) == 0:
        atomic_write_yaml(
            output_root / "preflight_manifest.yaml",
            manifest,
        )
        print(
            "optuna_preflight_complete "
            f"contract={contract_digest} output={output_root}"
        )
        return

    if os.name != "posix":
        raise RuntimeError(
            "formal A0 coordinator locking requires a POSIX host"
        )
    import fcntl

    coordinator_lock_path = _coordinator_lock_path(
        storage,
        output_root=output_root,
        study_name=args.study_name,
    )
    coordinator_lock_path.parent.mkdir(parents=True, exist_ok=True)
    coordinator_lock = coordinator_lock_path.open(
        "a+",
        encoding="utf-8",
    )
    try:
        fcntl.flock(
            coordinator_lock.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB,
        )
    except BlockingIOError as exc:
        raise RuntimeError(
            "another formal Optuna coordinator is already active for "
            f"storage={storage} study={args.study_name}"
        ) from exc
    coordinator_lock.seek(0)
    coordinator_lock.truncate()
    coordinator_lock.write(f"pid={os.getpid()}\n")
    coordinator_lock.flush()

    sampler = TrialSeededTPESampler(
        seed=int(args.sampler_seed),
        n_startup_trials=6,
        constant_liar=bool(args.parallel_constant_liar),
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=6,
        n_warmup_steps=PRUNER_WARMUP_STEPS,
        interval_steps=1,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )
    bound_contract_digest = _bind_study_contract(study, contract)
    if bound_contract_digest != contract_digest:
        raise RuntimeError("bound study contract digest changed")
    _assert_resumable_study_states(study)
    existing_trials = study.get_trials(deepcopy=False)
    if not existing_trials:
        for params in SEEDED_PARAMS[: int(args.study_trials)]:
            study.enqueue_trial(params)
    export_study_artifacts(
        study,
        output_root=output_root,
        manifest=manifest,
    )
    available_devices: queue.Queue[str | None] = queue.Queue()
    for cuda_device in cuda_devices:
        available_devices.put(cuda_device)
    baseline_lock = threading.Lock()
    export_lock = threading.Lock()

    def objective(trial: optuna.Trial) -> float:
        cuda_device = available_devices.get()
        try:
            current_source = source_state(repo)
            if current_source != source:
                raise RuntimeError(
                    "formal source changed after study preflight"
                )
            trial_dir = output_root / f"trial_{trial.number:04d}"
            if trial_dir.exists() and any(trial_dir.iterdir()):
                raise RuntimeError(
                    "refusing to overwrite an existing trial directory: "
                    f"{trial_dir}"
                )
            trial_dir.mkdir(parents=True, exist_ok=True)
            with baseline_lock:
                shared_baseline = _study_selection_baseline(study)
            cfg = trial_config(
                base_cfg,
                trial,
                trial_dir,
                epochs=int(args.epochs),
                selection_baseline=shared_baseline,
                formal_asset_sha256=assets["formal_asset_sha256"],
                formal_source=source,
                formal_study_contract_sha256=contract_digest,
                formal_population_validation=assets[
                    "population_validation"
                ],
            )
            cfg_path = trial_dir / "config.yaml"
            write_yaml(cfg_path, cfg)
            print(
                f"trial_start number={trial.number} "
                f"device={cuda_device if cuda_device is not None else 'env'} "
                f"lr={trial.params['lr']:.8g} "
                f"weight_decay={trial.params['weight_decay']:.8g} "
                f"spatial_weight={trial.params['spatial_weight']:.8g}",
                flush=True,
            )
            trial.set_user_attr(
                "study_contract_sha256",
                contract_digest,
            )
            trial.set_user_attr(
                "config_sha256",
                config_digest(cfg),
            )
            trial.set_user_attr(
                "cuda_visible_device",
                cuda_device if cuda_device is not None else "inherited",
            )
            try:
                return train_with_pruning(
                    trial=trial,
                    cfg_path=cfg_path,
                    output_dir=trial_dir,
                    python_bin=str(args.python),
                    repo=repo,
                    poll_sec=float(args.poll_sec),
                    expected_weights=selection_weights,
                    expected_baseline=shared_baseline,
                    expected_start_step=(
                        _selection_start_step_from_config(cfg)
                    ),
                    cuda_visible_device=cuda_device,
                    verified_asset_receipt=verified_asset_receipt,
                )
            except optuna.TrialPruned:
                raise
            except Exception as exc:
                trial.set_user_attr(
                    "failure_reason",
                    f"{type(exc).__name__}: {exc}",
                )
                raise
            finally:
                trial_baseline = _selection_baseline_from_path(
                    trial_dir / "selection_baseline.json"
                )
                if trial_baseline is not None:
                    with baseline_lock:
                        baseline_digest = _bind_study_selection_baseline(
                            study,
                            trial_baseline,
                        )
                    trial.set_user_attr(
                        "selection_metric_baseline_sha256",
                        baseline_digest,
                    )
        finally:
            available_devices.put(cuda_device)

    def export_callback(
        callback_study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
    ) -> None:
        del trial
        with export_lock:
            export_study_artifacts(
                callback_study,
                output_root=output_root,
                manifest=manifest,
            )

    timeout = (
        None
        if float(args.timeout_hours) <= 0
        else float(args.timeout_hours) * 3600.0
    )
    remaining_executions = _remaining_study_executions(
        study,
        int(args.study_trials),
    )
    invocation_trials = min(
        int(args.n_trials),
        remaining_executions,
    )
    if invocation_trials <= 0:
        print(
            "optuna_study_budget_exhausted "
            f"trial_records={len(study.get_trials(deepcopy=False))}"
        )
        export_study_artifacts(
            study,
            output_root=output_root,
            manifest=manifest,
            hash_best_checkpoint=True,
        )
        return
    study.optimize(
        objective,
        n_trials=invocation_trials,
        timeout=timeout,
        n_jobs=int(args.parallel_trials),
        gc_after_trial=True,
        callbacks=[export_callback],
    )
    export_study_artifacts(
        study,
        output_root=output_root,
        manifest=manifest,
        hash_best_checkpoint=True,
    )
    if not study.best_trials:
        print("optuna_search_complete completed_trials=0")
        return
    best = study.best_trial
    print(
        f"best_trial={best.number} "
        f"selection_loss={best.value:.6f}"
    )
    print(f"best_params={best.params}")
    print(
        "best_checkpoint="
        f"{best.user_attrs.get('best_checkpoint')}"
    )


if __name__ == "__main__":
    main()
