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
) -> dict:
    """Overlay one matched full-population, tenth-duration ablation condition."""

    base = load_config(base_path)
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
    if (
        condition_prototypes is not None
        and isinstance(base_prototypes, dict)
    ):
        resolved["data"]["prototype_paths"] = {
            name: base_prototypes[name]
            for name in active_teachers
        }

    # All conditions use the complete population stream and replay the same
    # full L1/L2 expert tile union. A1/A3 mask L1 labels from the objective but
    # retain these images in the replay stream.
    resolved["data"][
        "expert_replay_prototype_manifest_path"
    ] = base_prototype_manifest
    for key in ("train_tile_fraction", "val_tile_fraction"):
        if float(resolved["data"].get(key, 1.0)) != 1.0:
            raise ValueError(
                f"matched ablation requires data.{key}=1.0"
            )
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay one tracked ablation condition on a local resolved base."
    )
    parser.add_argument("--base", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    resolved = resolve_ablation_config(args.base, args.condition)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False)


if __name__ == "__main__":
    main()
