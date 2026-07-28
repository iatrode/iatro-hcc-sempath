from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from hcc_sempath.training.config import _deep_merge, load_config


def _raw_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return payload


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
    if int(base["train"]["epochs"]) != 3:
        raise ValueError("formal ablations require the selected three-epoch A0 trial")
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

    # Every condition replays the same complete L1/L2 expert tile union.
    # A1/A3 mask L1 labels from the objective but retain the L1 images.
    resolved["data"]["expert_replay_prototype_manifest_path"] = base_prototype_manifest
    for key in ("train_tile_fraction", "val_tile_fraction"):
        if float(resolved["data"].get(key, 1.0)) != 0.1:
            raise ValueError(f"matched ablation requires data.{key}=0.1")
    if int(resolved["train"]["epochs"]) != 3:
        raise ValueError("matched ablation requires train.epochs=3")

    condition_name = Path(
        condition.get("runtime", {}).get(
            "output_dir",
            condition_path.stem,
        )
    ).name
    if output_root is None:
        output_root = Path(base["runtime"]["output_dir"]).parent / "formal_ablations"
    resolved["runtime"]["output_dir"] = str(Path(output_root) / condition_name)
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
