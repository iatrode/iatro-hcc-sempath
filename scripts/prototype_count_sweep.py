from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
import yaml

from hcc_sempath.modeling.prototypes import PrototypeRegistry, load_prototype_registry
from hcc_sempath.training.config import load_config, teacher_names


def _deep_set(payload: dict, dotted_key: str, value: Any) -> None:
    current = payload
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _parse_int_list(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("expected at least one integer")
    return result


def _annotation_csv_path(input_path: str | Path) -> Path:
    path = Path(input_path)
    if path.is_file():
        return path
    candidate = path / "hcc_prototype_review.csv"
    if candidate.exists():
        return candidate
    matches = sorted(path.glob("*.csv"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"no annotation CSV found under input_path: {path}")
    raise ValueError(f"multiple CSV files under input_path; pass the CSV path explicitly: {matches}")


def _split_labels(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).replace("|", ";").split(";") if item.strip()]


def _load_label_map(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
    else:
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("label map must be a mapping")
    return {str(key): str(value) for key, value in payload.items()}


def _map_label(label: str, label_map: dict[str, str]) -> str:
    return label_map.get(label, label)


def load_annotations(input_path: str | Path, label_map: dict[str, str]) -> list[dict[str, Any]]:
    path = _annotation_csv_path(input_path)
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"tile_id", "l1", "l2"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"annotation CSV missing columns: {sorted(missing)}")
        for row in reader:
            l1 = _map_label(str(row["l1"]).strip(), label_map)
            l2 = [_map_label(name, label_map) for name in _split_labels(row.get("l2"))]
            rows.append(
                {
                    "tile_id": str(row["tile_id"]).strip(),
                    "slide": str(row.get("slide") or row.get("slide_id") or row["tile_id"]).strip(),
                    "l1": l1,
                    "l2": l2,
                    "source_row": dict(row),
                }
            )
    return rows


def _counter_for_level(rows: list[dict[str, Any]], names: list[str], level: str) -> Counter:
    allowed = set(names)
    counts: Counter = Counter()
    if level == "l1":
        for row in rows:
            if row["l1"] in allowed:
                counts[row["l1"]] += 1
        return counts
    for row in rows:
        for name in row["l2"]:
            if name in allowed:
                counts[name] += 1
    return counts


def select_prototypes(
    rows: list[dict[str, Any]],
    registry: PrototypeRegistry,
    *,
    prototype_count: int,
    primary_policy: str,
    min_primary: int,
) -> list[str]:
    primary_names = [registry.names[idx] for idx in registry.primary_indices]
    attribute_names = [registry.names[idx] for idx in registry.attribute_indices]
    primary_counts = _counter_for_level(rows, primary_names, "l1")
    attribute_counts = _counter_for_level(rows, attribute_names, "l2")

    if primary_policy == "all":
        selected_primary = list(primary_names)
        if prototype_count < len(selected_primary):
            raise ValueError(
                f"prototype_count={prototype_count} is smaller than all primary prototypes="
                f"{len(selected_primary)}; use --primary-policy top to sweep total counts below this"
            )
    elif primary_policy == "top":
        primary_limit = max(int(min_primary), min(len(primary_names), int(prototype_count)))
        selected_primary = sorted(primary_names, key=lambda name: (-primary_counts[name], name))[:primary_limit]
    else:
        raise ValueError(f"unsupported primary_policy: {primary_policy}")

    remaining = int(prototype_count) - len(selected_primary)
    if remaining < 0:
        raise ValueError(f"prototype_count={prototype_count} leaves no room after primary selection")
    selected_attributes = sorted(attribute_names, key=lambda name: (-attribute_counts[name], name))[:remaining]
    selected = set(selected_primary + selected_attributes)
    return [name for name in registry.names if name in selected]


def subset_registry(registry: PrototypeRegistry, selected_names: list[str], output_path: Path, source_note: dict[str, Any]) -> None:
    positions = {name: idx for idx, name in enumerate(registry.names)}
    missing = [name for name in selected_names if name not in positions]
    if missing:
        raise ValueError(f"prototype package missing selected names: {missing}")
    indices = [positions[name] for name in selected_names]
    selected_levels = [registry.levels[idx] for idx in indices]
    if selected_levels.count(1) < 2:
        raise ValueError("subset prototype package must retain at least two level-1 prototypes")
    payload = {
        "version": registry.version,
        "prototypes": registry.prototypes[indices].cpu(),
        "names": [registry.names[idx] for idx in indices],
        "groups": [registry.groups[idx] for idx in indices],
        "levels": selected_levels,
        "exclusive": [registry.exclusive[idx] for idx in indices],
        "source": {**(registry.source or {}), **source_note},
    }
    if registry.thresholds is not None:
        payload["thresholds"] = registry.thresholds[indices].cpu()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def _split_by_group(rows: list[dict[str, Any]], *, split_key: str, val_frac: float, seed: int) -> dict[str, str]:
    import random

    groups = sorted({str(row.get(split_key) or row["tile_id"]) for row in rows})
    rng = random.Random(seed)
    rng.shuffle(groups)
    val_count = int(round(len(groups) * float(val_frac)))
    if val_frac > 0 and groups:
        val_count = max(1, val_count)
    val_groups = set(groups[:val_count])
    return {group: ("val" if group in val_groups else "train") for group in groups}


def write_supervision_manifest(
    rows: list[dict[str, Any]],
    selected_registry: PrototypeRegistry,
    output_path: Path,
    *,
    split_key: str,
    val_frac: float,
    seed: int,
) -> dict[str, int]:
    primary_names = {selected_registry.names[idx] for idx in selected_registry.primary_indices}
    attribute_names = {selected_registry.names[idx] for idx in selected_registry.attribute_indices}
    split_by_group = _split_by_group(rows, split_key=split_key, val_frac=val_frac, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    dropped_l1 = 0
    duplicate_tile = 0
    seen_tiles: set[str] = set()
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["tile_id", "level1_label", "level2_labels", "source_split", "expert_a", "expert_b", "adjudicated"],
        )
        writer.writeheader()
        for row in rows:
            if row["l1"] not in primary_names:
                dropped_l1 += 1
                continue
            tile_id = str(row["tile_id"])
            if tile_id in seen_tiles:
                duplicate_tile += 1
                continue
            seen_tiles.add(tile_id)
            group = str(row.get(split_key) or tile_id)
            selected_l2 = [name for name in row["l2"] if name in attribute_names]
            writer.writerow(
                {
                    "tile_id": tile_id,
                    "level1_label": row["l1"],
                    "level2_labels": ";".join(selected_l2),
                    "source_split": split_by_group.get(group, "train"),
                    "expert_a": "annotation_csv",
                    "expert_b": "annotation_csv",
                    "adjudicated": "true",
                }
            )
            kept += 1
    return {"kept": kept, "dropped_l1": dropped_l1, "duplicate_tile": duplicate_tile}


def _apply_reduced_training_defaults(cfg: dict, args: argparse.Namespace, output_dir: Path) -> None:
    cfg["runtime"]["output_dir"] = str(output_dir)
    cfg["runtime"]["seed"] = int(args.current_seed)
    train_cfg = cfg.setdefault("train", {})
    data_cfg = cfg.setdefault("data", {})
    loss_cfg = cfg.setdefault("loss", {})
    train_cfg["epochs"] = int(args.epochs)
    train_cfg["max_train_batches"] = int(args.max_train_batches)
    train_cfg["max_val_batches"] = int(args.max_val_batches)
    train_cfg["max_eval_batches"] = int(args.max_eval_batches)
    data_cfg["max_train_records"] = int(args.max_train_records)
    data_cfg["max_val_records"] = int(args.max_val_records)
    loss_cfg["teacher_prior_plateau_window_steps"] = int(args.plateau_window_steps)
    loss_cfg["min_teacher_warmup_steps"] = int(args.min_teacher_warmup_steps)
    loss_cfg["max_teacher_warmup_steps"] = int(args.max_teacher_warmup_steps)
    loss_cfg["prototype_ramp_steps"] = int(args.prototype_ramp_steps)
    loss_cfg["proto_to_filter_delay_steps"] = int(args.proto_to_filter_delay_steps)
    loss_cfg["filter_ramp_steps"] = int(args.filter_ramp_steps)


def _set_prototype_paths(cfg: dict, zhcc_path: Path, teacher_paths: dict[str, Path]) -> None:
    data_cfg = cfg.setdefault("data", {})
    data_cfg["zhcc_prototype_path"] = str(zhcc_path)
    if teacher_paths:
        data_cfg["prototype_paths"] = {name: str(path) for name, path in sorted(teacher_paths.items())}


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _read_summary(output_dir: Path) -> dict[str, Any]:
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _experiment_dir(output_root: Path, count: int) -> Path:
    return output_root / f"count_{count:03d}"


def _training_command(config_path: Path) -> list[str]:
    return [sys.executable, "-m", "hcc_sempath.cli.main", "train", "--config", str(config_path)]


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_aggregate_csv(path: Path, rows: list[dict[str, Any]], metric_keys: set[str]) -> None:
    import math

    aggregate_rows = []
    counts = sorted({int(row["prototype_count_requested"]) for row in rows})
    for metric in sorted(metric_keys):
        previous_mean = None
        for count in counts:
            values = []
            for row in rows:
                if int(row["prototype_count_requested"]) != count or row.get("status") != "ok":
                    continue
                value = row.get(metric)
                if isinstance(value, bool) or value is None:
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(numeric):
                    values.append(numeric)
            if not values:
                continue
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
            delta = None if previous_mean is None else mean - previous_mean
            relative_delta = None if previous_mean in {None, 0.0} else delta / abs(previous_mean)
            aggregate_rows.append(
                {
                    "metric": metric,
                    "prototype_count": count,
                    "n": len(values),
                    "mean": mean,
                    "std": variance ** 0.5,
                    "delta_from_previous_count": "" if delta is None else delta,
                    "relative_delta_from_previous_count": "" if relative_delta is None else relative_delta,
                }
            )
            previous_mean = mean
    if not aggregate_rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)


def _run_command(command: list[str], log_path: Path) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            env=env,
        )
    return int(completed.returncode)


def _run_training(config_path: Path, log_path: Path) -> int:
    return _run_command(_training_command(config_path), log_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reduced main-training sweeps over prototype counts.")
    parser.add_argument("--base-config", default="configs/experiments/pamtd_full.yaml")
    parser.add_argument("--input-path", required=True, help="Annotation CSV path or annotation directory.")
    parser.add_argument("--output-root", default="outputs/prototype_count_sweep")
    parser.add_argument("--prototype-counts", default="10,12,14,16")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--label-map", default="", help="Optional YAML/JSON mapping from annotation labels to prototype names.")
    parser.add_argument("--primary-policy", choices=["all", "top"], default="all")
    parser.add_argument("--min-primary", type=int, default=2)
    parser.add_argument("--split-key", choices=["slide", "tile_id"], default="slide")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-batches", type=int, default=64)
    parser.add_argument("--max-val-batches", type=int, default=16)
    parser.add_argument("--max-eval-batches", type=int, default=16)
    parser.add_argument("--max-train-records", type=int, default=8192)
    parser.add_argument("--max-val-records", type=int, default=2048)
    parser.add_argument("--plateau-window-steps", type=int, default=20)
    parser.add_argument("--min-teacher-warmup-steps", type=int, default=20)
    parser.add_argument("--max-teacher-warmup-steps", type=int, default=80)
    parser.add_argument("--prototype-ramp-steps", type=int, default=20)
    parser.add_argument("--proto-to-filter-delay-steps", type=int, default=20)
    parser.add_argument("--filter-ramp-steps", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", help="Write configs/packages/manifests without launching training.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional config override as dotted.key=value; values are parsed as YAML scalars.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    counts = _parse_int_list(args.prototype_counts)
    seed = int(args.seed)
    output_root = Path(args.output_root)
    label_map = _load_label_map(args.label_map)
    annotations = load_annotations(args.input_path, label_map)
    base_cfg = load_config(args.base_config)
    teachers = teacher_names(base_cfg)
    zhcc_registry = load_prototype_registry(
        base_cfg["data"]["zhcc_prototype_path"],
        expected_dim=int(base_cfg["model"].get("embedding_dim", base_cfg["model"].get("teacher_dim", 256))),
    )
    teacher_registries: dict[str, PrototypeRegistry] = {}
    prototype_paths = base_cfg["data"].get("prototype_paths")
    if isinstance(prototype_paths, dict):
        dims = base_cfg["model"].get("teacher_dims", {})
        for teacher in teachers:
            expected_dim = int(dims[teacher]) if isinstance(dims, dict) and teacher in dims else None
            teacher_registries[teacher] = load_prototype_registry(prototype_paths[teacher], expected_dim=expected_dim)

    run_rows: list[dict[str, Any]] = []
    metric_keys: set[str] = set()
    for count in counts:
        selected_names = select_prototypes(
            annotations,
            zhcc_registry,
            prototype_count=count,
            primary_policy=args.primary_policy,
            min_primary=args.min_primary,
        )
        experiment_dir = _experiment_dir(output_root, count)
        prototype_dir = experiment_dir / "prototypes"
        zhcc_path = prototype_dir / "zhcc_prototypes.pt"
        source_note = {
            "sweep_prototype_count": count,
            "sweep_seed": seed,
            "sweep_selected_names": selected_names,
        }
        subset_registry(zhcc_registry, selected_names, zhcc_path, source_note)
        teacher_paths: dict[str, Path] = {}
        for teacher, registry in teacher_registries.items():
            teacher_path = prototype_dir / f"{teacher}_prototypes.pt"
            subset_registry(registry, selected_names, teacher_path, source_note)
            teacher_paths[teacher] = teacher_path
        selected_registry = load_prototype_registry(zhcc_path)
        supervision_path = experiment_dir / "prototype_supervision.csv"
        supervision_stats = write_supervision_manifest(
            annotations,
            selected_registry,
            supervision_path,
            split_key=args.split_key,
            val_frac=float(args.val_frac),
            seed=seed,
        )
        cfg = load_config(args.base_config)
        args.current_seed = seed
        run_output_dir = experiment_dir / "training"
        _apply_reduced_training_defaults(cfg, args, run_output_dir)
        _set_prototype_paths(cfg, zhcc_path, teacher_paths)
        cfg["data"]["prototype_supervision_manifest_path"] = str(supervision_path)
        cfg["data"]["prototype_supervision_train_splits"] = ["train"]
        cfg["data"]["prototype_supervision_val_splits"] = ["val"]
        for override in args.override:
            if "=" not in override:
                raise ValueError(f"override must be dotted.key=value: {override}")
            key, value = override.split("=", 1)
            _deep_set(cfg, key, yaml.safe_load(value))
        config_path = experiment_dir / "config.yaml"
        _write_yaml(config_path, cfg)
        log_path = run_output_dir / "train.log"
        status = "dry_run"
        returncode = 0
        if not args.dry_run:
            returncode = _run_training(config_path, log_path)
            status = "ok" if returncode == 0 else "failed"
        metrics = _read_summary(run_output_dir)
        metric_keys.update(metrics)
        run_rows.append(
            {
                "prototype_count_requested": count,
                "prototype_count_actual": len(selected_names),
                "seed": seed,
                "status": status,
                "returncode": returncode,
                **supervision_stats,
                **metrics,
                "config_path": str(config_path),
                "experiment_dir": str(experiment_dir),
                "output_dir": str(run_output_dir),
            }
        )
        if status != "ok" and not args.dry_run:
            print(f"run_failed count={count} seed={seed} log={log_path}", file=sys.stderr)
    _write_summary_csv(output_root / "sweep_summary.csv", run_rows)
    _write_aggregate_csv(output_root / "sweep_aggregate.csv", run_rows, metric_keys)
    print(f"sweep_ok output_root={output_root} runs={len(run_rows)} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
