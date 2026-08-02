from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.ablation.scripts.resolve_ablation_config import (
    _ablation_config_sha256,
    _raw_config,
    _selected_a0_provenance,
    _selection_contract,
)
from hcc_sempath.training.config import _deep_merge
from hcc_sempath.training.train import _file_sha256, _source_tree_sha256
from research.scripts.optuna_a0_search import (
    _expert_split_tile_counts,
    _population_validation_contract,
    _resolved_training_iac_paths,
    write_verified_iac_receipt,
)


PATH_REPLACEMENTS = (
    ("/autodl-fs/data/features/merged", "features"),
    ("/root/autodl-fs/features/merged", "features"),
    ("/root/data/features/merged", "features"),
    ("/autodl-fs/data/features", "features"),
    ("/root/autodl-fs/features", "features"),
    ("/root/autodl-tmp/tiles", "tiles"),
    ("/root/autodl-tmp/hcc-sempath-assets/final", "assets"),
    ("/root/data/hcc-sempath-assets/final", "assets"),
    ("/root/data/assets/final", "assets"),
    ("/root/autodl-tmp/hcc-sempath-assets", "assets"),
    ("/root/autodl-tmp/hcc-sempath-pretrained", "pretrained"),
    ("/root/data/hcc-sempath-assets", "assets"),
    ("/root/data/hcc-sempath-pretrained", "pretrained"),
)


def _remap_path(raw: str, data_root: Path) -> str:
    for old, relative in PATH_REPLACEMENTS:
        if raw == old or raw.startswith(old + "/"):
            suffix = raw[len(old) :].lstrip("/")
            return str((data_root / relative / suffix).resolve())
    return raw


def _remap_tree(value: object, data_root: Path) -> object:
    if isinstance(value, str):
        return _remap_path(value, data_root)
    if isinstance(value, list):
        return [_remap_tree(item, data_root) for item in value]
    if isinstance(value, dict):
        return {
            str(_remap_tree(key, data_root)): _remap_tree(item, data_root)
            for key, item in value.items()
        }
    return value


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _prepare_a11(base: dict, repo: Path, output_root: Path) -> dict:
    base = copy.deepcopy(base)
    selected = _selected_a0_provenance(base)
    base_selection = _selection_contract(base)
    condition_path = (
        repo
        / "experiments/ablation/configs/a11_no_student_response.yaml"
    )
    condition = _raw_config(condition_path)
    parent = Path(str(condition.pop("inherits")))
    if not parent.is_absolute():
        parent = condition_path.parent / parent
    parent_overlay = _raw_config(parent)
    parent_overlay.pop("inherits", None)
    resolved = _deep_merge(base, parent_overlay)
    resolved = _deep_merge(resolved, condition)
    if _selection_contract(resolved) != base_selection:
        raise ValueError("A11 changed the selected A0 checkpoint rule")
    resolved["train"].pop("selection_metric_baseline", None)
    study_digest = resolved["data"].pop(
        "formal_study_contract_sha256"
    )
    resolved["data"]["ablation_parent_a0_study_contract_sha256"] = (
        str(study_digest)
    )
    resolved["data"]["formal_a0_selection"] = selected
    resolved["runtime"]["output_dir"] = str(
        (output_root / "a11_no_student_response").resolve()
    )
    return resolved


def _prepare_full(base: dict, output_root: Path) -> dict:
    selected = _selected_a0_provenance(base)
    scheduler_contract = selected.get("scheduler_contract")
    if not isinstance(scheduler_contract, dict):
        raise ValueError(
            "selected A0 provenance has no scheduler_contract; re-export "
            "the study artifact before preparing full-population training"
        )
    a0_lr_total_steps = int(
        scheduler_contract.get("planned_total_steps", 0)
    )
    if a0_lr_total_steps <= 0:
        raise ValueError(
            "selected A0 scheduler_contract has no positive "
            "planned_total_steps"
        )
    resolved = copy.deepcopy(base)
    resolved["data"]["train_tile_fraction"] = 1.0
    resolved["data"]["val_tile_fraction"] = 1.0
    resolved["loss"].update(
        {
            "expert_supervision_start_step": 3000,
            "expert_supervision_ramp_steps": 1000,
            "prototype_filter_start_step": 4000,
            "prototype_filter_ramp_steps": 1000,
            "zhcc_response_start_step": 4000,
            "zhcc_response_ramp_steps": 1000,
        }
    )
    resolved["train"].update(
        {
            "checkpoint_interval_steps": 1000,
            "lr_total_steps": a0_lr_total_steps,
            "development_probe_interval_steps": 0,
            "development_early_stop": False,
            "selection_probe_interval_steps": 1000,
            "selection_early_stop": True,
            "selection_minimum_eligible_probes": 8,
            "selection_early_stop_patience": 3,
            "selection_early_stop_relative_delta": 0.005,
        }
    )
    resolved["train"].pop("selection_metric_baseline", None)
    resolved["train"].pop("selection_minimum_eligible_epochs", None)
    study_digest = resolved["data"].pop(
        "formal_study_contract_sha256"
    )
    resolved["data"]["full_training_parent_a0_study_contract_sha256"] = (
        str(study_digest)
    )
    resolved["runtime"]["output_dir"] = str(
        (output_root / "a0_full_population").resolve()
    )
    return resolved


def _relocate_assets(
    config: dict,
    *,
    data_root: Path,
    manifest_template: Path,
    manifest_output: Path,
) -> None:
    manifest = _remap_tree(
        yaml.safe_load(manifest_template.read_text(encoding="utf-8")),
        data_root,
    )
    if not isinstance(manifest, dict):
        raise ValueError("manifest template must be a YAML mapping")
    _write_yaml(manifest_output, manifest)

    data = config["data"]
    data["train_manifest_path"] = str(manifest_output.resolve())
    for key in (
        "prototype_paths",
        "prototype_supervision_manifest_path",
        "expert_replay_prototype_manifest_path",
        "spatial_manifest_path",
    ):
        if key in data:
            data[key] = _remap_tree(data[key], data_root)

    formal = data["formal_asset_sha256"]
    formal["iac_packages"] = {
        _remap_path(str(path), data_root): str(digest)
        for path, digest in formal["iac_packages"].items()
    }
    formal["student_pretrained"]["path"] = _remap_path(
        str(formal["student_pretrained"]["path"]),
        data_root,
    )
    formal["static_files"]["train_manifest_path"] = _file_sha256(
        manifest_output
    )


def _verify_static_assets(config: dict) -> None:
    data = config["data"]
    formal = data["formal_asset_sha256"]
    paths = {
        **{
            f"prototype_{name}": Path(path)
            for name, path in data["prototype_paths"].items()
        },
        "prototype_supervision_manifest_path": Path(
            data["prototype_supervision_manifest_path"]
        ),
        "spatial_manifest_path": Path(data["spatial_manifest_path"]),
        "train_manifest_path": Path(data["train_manifest_path"]),
    }
    for name, path in paths.items():
        expected = str(formal["static_files"][name])
        if not path.is_file() or _file_sha256(path) != expected:
            raise ValueError(f"formal static asset differs: {name}={path}")
    student = formal["student_pretrained"]
    student_path = Path(student["path"])
    if not student_path.is_file() or _file_sha256(student_path) != str(
        student["sha256"]
    ):
        raise ValueError(f"DINOv2 initialization differs: {student_path}")


def _refresh_population_validation(config: dict) -> None:
    selected_val, _ = _resolved_training_iac_paths(
        config,
        complete=False,
        splits=("val",),
    )
    counts = _expert_split_tile_counts(
        Path(config["data"]["prototype_supervision_manifest_path"]),
        Path(config["data"]["spatial_manifest_path"]),
    )
    formal = config["data"]["formal_asset_sha256"]
    config["data"]["formal_population_validation"] = (
        _population_validation_contract(
            config,
            selected_val,
            expert_tiles=counts["train"] + counts["val"],
            iac_sha256=formal["iac_packages"],
            expert_asset_sha256={
                name: formal["static_files"][name]
                for name in (
                    "prototype_supervision_manifest_path",
                    "spatial_manifest_path",
                )
            },
        )
    )


def prepare(args: argparse.Namespace) -> dict:
    repo = REPO_ROOT
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    base = yaml.safe_load(args.best_config.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError("best_config must be a YAML mapping")
    config = (
        _prepare_a11(base, repo, output_root)
        if args.mode == "a11"
        else _prepare_full(base, output_root)
    )
    _relocate_assets(
        config,
        data_root=data_root,
        manifest_template=args.manifest_template,
        manifest_output=args.manifest_output,
    )
    _verify_static_assets(config)
    _refresh_population_validation(config)
    source_commit = args.source_commit.strip().lower()
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef"
        for character in source_commit
    ):
        raise ValueError("--source-commit must be a 40-character Git SHA")
    config["data"]["formal_source"] = {
        "commit": source_commit,
        "source_mode": "declared_archive",
        "source_tree_sha256": _source_tree_sha256(repo),
    }
    if args.mode == "a11":
        config["data"].pop("formal_ablation_contract_sha256", None)
        config["data"]["formal_ablation_contract_sha256"] = (
            _ablation_config_sha256(config)
        )
    encoded = json.dumps(config, ensure_ascii=False)
    if "/autodl-fs" in encoded or "/root/autodl-fs" in encoded:
        raise ValueError("resolved config still references the network volume")
    _write_yaml(args.output_config, config)
    if args.verified_asset_receipt is not None:
        write_verified_iac_receipt(
            args.verified_asset_receipt.resolve(),
            config["data"]["formal_asset_sha256"],
        )
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare content-identical SemPath runs on local NVMe."
    )
    parser.add_argument("--mode", choices=("a11", "full"), required=True)
    parser.add_argument("--best-config", type=Path, required=True)
    parser.add_argument("--manifest-template", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/root/data"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/data/outputs"),
    )
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument(
        "--verified-asset-receipt",
        type=Path,
        default=None,
        help=(
            "Write a stat-bound receipt from the previously verified "
            "formal SHA-256 contract for fast local-NVMe restarts."
        ),
    )
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    prepared = prepare(parse_args())
    print(
        "prepared",
        prepared["runtime"]["output_dir"],
        "train_fraction",
        prepared["data"]["train_tile_fraction"],
        "val_fraction",
        prepared["data"]["val_tile_fraction"],
    )
