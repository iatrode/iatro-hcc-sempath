from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import yaml

from hcc_sempath.training.config import (
    _deep_merge,
    image_tile_package_paths,
    load_config,
    manifest_data_paths,
    teacher_feature_package_paths,
    teacher_names,
)
from hcc_sempath.training.engine import _selection_start_step
from hcc_sempath.training.manifest import load_training_manifest


def _raw_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return payload


def _selection_contract(config: dict) -> dict:
    train = config["train"]
    data = config["data"]
    runtime = config["runtime"]
    weights = {
        str(name): float(value)
        for name, value in train.get(
            "selection_metric_weights",
            {},
        ).items()
    }
    if set(weights) != {"teacher", "classification", "spatial"}:
        raise ValueError(
            "formal ablations require the A0 teacher/classification/spatial "
            "selection weights"
        )
    if any(value <= 0.0 for value in weights.values()) or abs(
        sum(weights.values()) - 1.0
    ) > 1e-9:
        raise ValueError(
            "formal ablations require positive A0 selection weights "
            "summing to one"
        )
    return {
        "epochs": int(train["epochs"]),
        "selection_early_stop": bool(
            train.get("selection_early_stop", False)
        ),
        "selection_metric_weights": weights,
        "selection_early_stop_start_step": train.get(
            "selection_early_stop_start_step"
        ),
        "selection_early_stop_patience": int(
            train.get("selection_early_stop_patience", 0)
        ),
        "selection_early_stop_relative_delta": float(
            train.get("selection_early_stop_relative_delta", 0.0)
        ),
        "selection_minimum_eligible_epochs": int(
            train.get("selection_minimum_eligible_epochs", 0)
        ),
        "development_early_stop": bool(
            train.get("development_early_stop", False)
        ),
        "early_stop_teacher_alignment": bool(
            train.get("early_stop_teacher_alignment", False)
        ),
        "teacher_retention_probe": {
            "runtime_seed": int(runtime["seed"]),
            "batch_size": int(train["batch_size"]),
            "max_val_batches": int(train["max_val_batches"]),
            "max_eval_batches": int(train["max_eval_batches"]),
            "eval_pairwise_max_samples": int(
                train["eval_pairwise_max_samples"]
            ),
            "dynamic_package_sampling": bool(
                data["dynamic_package_sampling"]
            ),
            "package_multiprocessing": bool(
                data.get("package_multiprocessing", False)
            ),
            "package_chunk_size": int(data["package_chunk_size"]),
            "package_buffer_batches": int(
                data.get("package_buffer_batches", 4)
            ),
        },
    }


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _selected_a0_provenance(config: dict) -> dict:
    data = config["data"]
    selected = data.get("formal_a0_selection")
    if not isinstance(selected, dict):
        raise ValueError(
            "ablation base must be the exported selected A0 best_config.yaml"
        )
    if not bool(selected.get("study_complete", False)):
        raise ValueError("selected A0 search has not exhausted its trial budget")
    total = int(selected.get("total_trial_budget", 0))
    executed = int(selected.get("executed_trial_records", -1))
    if total <= 0 or executed != total:
        raise ValueError("selected A0 trial-budget provenance is invalid")
    parent_study = str(data.get("formal_study_contract_sha256", ""))
    if str(selected.get("study_contract_sha256", "")) != parent_study:
        raise ValueError("selected A0 study digest does not match its config")
    trial_config_sha256 = str(
        selected.get("trial_config_sha256", "")
    )
    checkpoint_sha256 = str(
        selected.get("best_checkpoint_sha256", "")
    )
    if (
        len(trial_config_sha256) != 64
        or len(checkpoint_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in trial_config_sha256 + checkpoint_sha256
        )
    ):
        raise ValueError("selected A0 config/checkpoint digest is invalid")
    raw_config = copy.deepcopy(config)
    raw_config["data"].pop("formal_a0_selection", None)
    if _canonical_sha256(raw_config) != trial_config_sha256:
        raise ValueError(
            "ablation base differs from the exported selected A0 trial config"
        )
    params = selected.get("selected_params")
    if not isinstance(params, dict) or set(params) != {
        "lr",
        "weight_decay",
        "spatial_weight",
    }:
        raise ValueError("selected A0 search parameters are incomplete")
    expected = {
        "lr": float(config["train"]["lr"]),
        "weight_decay": float(config["train"]["weight_decay"]),
        "spatial_weight": float(config["loss"]["spatial_weight"]),
    }
    if any(
        not math.isclose(
            float(params[name]),
            expected[name],
            rel_tol=1e-12,
            abs_tol=0.0,
        )
        for name in expected
    ):
        raise ValueError(
            "ablation base hyperparameters differ from the selected A0 trial"
        )
    if (
        int(selected.get("selected_trial", -1)) < 0
        or int(selected.get("best_epoch", 0)) <= 0
        or not math.isfinite(
            float(selected.get("selection_loss", float("nan")))
        )
    ):
        raise ValueError("selected A0 trial identity is invalid")
    return dict(selected)


def _ablation_config_sha256(config: dict) -> str:
    payload = copy.deepcopy(config)
    payload["data"].pop("formal_ablation_contract_sha256", None)
    return _canonical_sha256(payload)


def _as_paths(value: object, *, key: str) -> list[str]:
    if value is None:
        raise ValueError(f"data.{key} is required")
    if isinstance(value, dict):
        return [str(path) for path in value.values()]
    if isinstance(value, (list, tuple)):
        return [str(path) for path in value]
    return [str(value)]


def _as_teacher_paths(
    value: object,
    *,
    key: str,
    active_teachers: list[str],
) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        raise ValueError(f"data.{key} must be a teacher->paths mapping")
    result: dict[str, list[str]] = {}
    for teacher in active_teachers:
        if teacher not in value:
            raise ValueError(f"data.{key} is missing teacher {teacher}")
        paths = value[teacher]
        result[teacher] = (
            [str(path) for path in paths]
            if isinstance(paths, (list, tuple))
            else [str(paths)]
        )
    return result


def _complete_iac_paths(
    config: dict,
) -> tuple[set[str], dict[str, set[str]]]:
    data = config["data"]
    active_teachers = teacher_names(config)
    if "train_image_tile_package_paths" in data:
        tile_paths = (
            _as_paths(
                data.get("train_image_tile_package_paths"),
                key="train_image_tile_package_paths",
            )
            + _as_paths(
                data.get("val_image_tile_package_paths"),
                key="val_image_tile_package_paths",
            )
        )
        train_teachers = _as_teacher_paths(
            data.get("train_teacher_feature_package_paths"),
            key="train_teacher_feature_package_paths",
            active_teachers=active_teachers,
        )
        val_teachers = _as_teacher_paths(
            data.get("val_teacher_feature_package_paths"),
            key="val_teacher_feature_package_paths",
            active_teachers=active_teachers,
        )
        teacher_paths = {
            teacher: {
                str(Path(path).resolve())
                for path in (
                    train_teachers[teacher] + val_teachers[teacher]
                )
            }
            for teacher in active_teachers
        }
    elif data.get("train_manifest_path"):
        full = copy.deepcopy(config)
        full["data"]["train_tile_fraction"] = 1.0
        full["data"]["val_tile_fraction"] = 1.0
        manifest = load_training_manifest(
            full["data"]["train_manifest_path"]
        )
        tile_paths = []
        teacher_paths = {
            teacher: set()
            for teacher in active_teachers
        }
        for split in ("train", "val"):
            split_tiles, split_teachers = manifest_data_paths(
                full,
                manifest,
                split,
            )
            tile_paths.extend(split_tiles)
            for teacher in active_teachers:
                teacher_paths[teacher].update(
                    str(Path(path).resolve())
                    for path in split_teachers[teacher]
                )
    else:
        tile_paths = image_tile_package_paths(config)
        raw_teacher_paths = teacher_feature_package_paths(config)
        teacher_paths = {
            teacher: {
                str(Path(path).resolve())
                for path in raw_teacher_paths[teacher]
            }
            for teacher in active_teachers
        }
    return (
        {
            str(Path(path).resolve())
            for path in tile_paths
        },
        teacher_paths,
    )


def _condition_formal_asset_contract(
    config: dict,
    parent: object,
) -> dict:
    if not isinstance(parent, dict):
        raise ValueError("selected A0 config has no complete asset contract")
    static_parent = parent.get("static_files")
    iac_parent = parent.get("iac_packages")
    student_parent = parent.get("student_pretrained")
    if (
        not isinstance(static_parent, dict)
        or not isinstance(iac_parent, dict)
        or not isinstance(student_parent, dict)
    ):
        raise ValueError("selected A0 config has no complete asset contract")
    static_keys: list[str] = []
    prototype_paths = config["data"].get("prototype_paths")
    if isinstance(prototype_paths, dict):
        static_keys.extend(
            f"prototype_{teacher}"
            for teacher in prototype_paths
        )
    static_keys.extend(
        key
        for key in (
            "prototype_supervision_manifest_path",
            "spatial_manifest_path",
            "train_manifest_path",
        )
        if config["data"].get(key)
    )
    missing_static = sorted(set(static_keys) - set(static_parent))
    if missing_static:
        raise ValueError(
            "ablation introduces static assets absent from A0: "
            f"{missing_static}"
        )
    tile_paths, teacher_paths = _complete_iac_paths(config)
    current_iac = set(tile_paths)
    current_iac.update(
        path
        for paths in teacher_paths.values()
        for path in paths
    )
    normalized_parent = {
        str(Path(str(path)).resolve()): str(digest)
        for path, digest in iac_parent.items()
    }
    missing_iac = sorted(current_iac - set(normalized_parent))
    if missing_iac:
        raise ValueError(
            "ablation introduces IAC assets absent from A0: "
            f"{missing_iac[:10]}"
        )
    return {
        "static_files": {
            key: str(static_parent[key])
            for key in sorted(static_keys)
        },
        "iac_packages": {
            path: normalized_parent[path]
            for path in sorted(current_iac)
        },
        "student_pretrained": dict(student_parent),
    }


def validate_ablation_resume_checkpoint(
    config_path: str | Path,
    checkpoint_path: str | Path,
) -> None:
    import torch

    config = load_config(config_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    planned = int(config["train"]["epochs"])
    previous = int(checkpoint.get("expected_epochs", -1))
    if previous != planned:
        raise ValueError(
            "refusing to extend an ablation checkpoint created under a "
            "different epoch plan: "
            f"checkpoint.expected_epochs={previous} configured={planned}"
        )


def resolve_ablation_config(
    base_path: str | Path,
    condition_path: str | Path,
    *,
    output_root: str | Path | None = None,
) -> dict:
    """Overlay one condition on the selected formal 1/10 Optuna A0 trial."""

    base = load_config(base_path)
    if float(base["data"].get("train_tile_fraction", 1.0)) != 0.1:
        raise ValueError(
            "formal ablations require the selected A0 trial with "
            "data.train_tile_fraction=0.1"
        )
    if float(base["data"].get("val_tile_fraction", 1.0)) != 0.1:
        raise ValueError(
            "formal ablations require the selected A0 trial with "
            "data.val_tile_fraction=0.1"
        )
    selected_a0 = _selected_a0_provenance(base)
    base["train"]["selection_early_stop_start_step"] = (
        _selection_start_step(base)
    )
    base_selection = _selection_contract(base)
    if base_selection["epochs"] < 6:
        raise ValueError(
            "formal ablations require an A0 maximum budget of at least "
            "six epochs"
        )
    if not base_selection["selection_early_stop"]:
        raise ValueError(
            "formal ablations require A0 joint selection early stopping"
        )
    if base_selection["development_early_stop"]:
        raise ValueError(
            "formal ablations cannot use population development-loss "
            "early stopping"
        )
    if base_selection["early_stop_teacher_alignment"]:
        raise ValueError(
            "formal ablations cannot use teacher-only early stopping"
        )
    if not bool(
        base["data"].get("require_complete_expert_validation", False)
    ):
        raise ValueError(
            "formal ablations require complete classification/spatial expert validation"
        )
    condition_path = Path(condition_path)
    condition = _raw_config(condition_path)
    parent = condition.get("inherits")
    if not parent:
        raise ValueError(
            "ablation condition must inherit the tracked tenth-duration base"
        )
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = condition_path.parent / parent_path
    parent_overlay = _raw_config(parent_path)
    parent_overlay.pop("inherits", None)
    condition.pop("inherits", None)

    base_prototype_manifest = base.get("data", {}).get(
        "expert_replay_prototype_manifest_path",
        base.get("data", {}).get("prototype_supervision_manifest_path"),
    )
    base_prototypes = base.get("data", {}).get("prototype_paths")
    resolved = _deep_merge(base, parent_overlay)
    resolved = _deep_merge(resolved, condition)
    # A single-teacher condition reuses that teacher's deployment asset. The
    # repository-relative path in the tracked overlay is only documentation.
    condition_prototypes = condition.get("data", {}).get(
        "prototype_paths",
        ...,
    )
    active_teachers = [str(name) for name in resolved["data"]["teachers"]]
    if condition_prototypes is not None and isinstance(base_prototypes, dict):
        resolved["data"]["prototype_paths"] = {
            name: base_prototypes[name] for name in active_teachers
        }

    # Every condition replays the same complete classification/spatial expert tile union.
    # A1/A3 mask classification labels from the objective but retain the classification images.
    resolved["data"]["expert_replay_prototype_manifest_path"] = base_prototype_manifest
    for key in ("train_tile_fraction", "val_tile_fraction"):
        if float(resolved["data"].get(key, 1.0)) != 0.1:
            raise ValueError(f"matched ablation requires data.{key}=0.1")
    if _selection_contract(resolved) != base_selection:
        raise ValueError(
            "matched ablation changed the selected A0 maximum budget or "
            "joint checkpoint-selection rule"
        )

    # The A0 search contract binds its exact four-teacher model. Derive a new
    # exact subset contract for the assets genuinely used by this condition.
    parent_study_digest = resolved["data"].get(
        "formal_study_contract_sha256"
    )
    if not parent_study_digest:
        raise ValueError("selected A0 config has no formal study digest")
    parent_asset_contract = resolved["data"].get("formal_asset_sha256")
    condition_asset_contract = _condition_formal_asset_contract(
        resolved,
        parent_asset_contract,
    )
    formal_source = resolved["data"].get("formal_source")
    if not isinstance(formal_source, dict):
        raise ValueError("selected A0 config has no formal source contract")
    resolved["data"]["formal_asset_sha256"] = condition_asset_contract
    resolved["data"].pop("formal_asset_contract_sha256", None)
    resolved["data"].pop("formal_study_contract_sha256", None)
    resolved["data"][
        "ablation_parent_a0_study_contract_sha256"
    ] = str(parent_study_digest)
    resolved["data"]["formal_a0_selection"] = selected_a0
    # Each intervention freezes its own shared-initialization denominators;
    # reusing an A0 denominator after changing a computation path is invalid.
    resolved["train"].pop("selection_metric_baseline", None)

    condition_name = Path(
        condition.get("runtime", {}).get(
            "output_dir",
            condition_path.stem,
        )
    ).name
    if output_root is None:
        output_root = Path(base["runtime"]["output_dir"]).parent / "formal_ablations"
    resolved["runtime"]["output_dir"] = str(Path(output_root) / condition_name)
    resolved["data"]["formal_ablation_contract_sha256"] = (
        _ablation_config_sha256(resolved)
    )
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay one tracked ablation condition on a local resolved base."
    )
    parser.add_argument("--base", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-root")
    args = parser.parse_args()

    resolved = resolve_ablation_config(
        args.base,
        args.condition,
        output_root=args.output_root,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False)


if __name__ == "__main__":
    main()
