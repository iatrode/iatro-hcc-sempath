#!/usr/bin/env python3
"""Select the fixed L1 prototype bank from four frozen teacher spaces."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_PATH = REPO_ROOT / "annotations" / "hcc_prototype_review.final_l1.json"
MANIFEST_PATH = REPO_ROOT / "configs" / "local" / "mac" / "manifest.yaml"
OUTPUT_JSON = REPO_ROOT / "annotations" / "hcc_prototype_bank.fixed.json"
OUTPUT_CSV = REPO_ROOT / "annotations" / "hcc_prototype_bank.fixed.csv"
REPORT_DIR = REPO_ROOT / "annotations" / "analysis" / "l1_fixed_prototype_bank"
TARGET_PER_CLASS = 400
# The sample-level separation gain reaches its practical plateau at 32; larger
# weights add negligible separation while increasing reliance on class margin.
SEPARATION_WEIGHT = 32.0
CHECKPOINTS = (25, 50, 75, 100, 150, 200, 250, 300, 350, 400)
TEACHERS = ("gigapath", "h_optimus_1", "uni2_h", "virchow2")


def _load_curve_module() -> Any:
    path = REPO_ROOT / "scripts" / "prototype_information_curve.py"
    spec = importlib.util.spec_from_file_location("prototype_information_curve_for_selection", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load feature reader from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _feature_packages() -> dict[str, list[Path]]:
    payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8")) or {}
    roots = payload.get("feature_roots")
    if not isinstance(roots, dict):
        raise ValueError(f"manifest has no feature_roots: {MANIFEST_PATH}")
    packages: dict[str, list[Path]] = {}
    for teacher in TEACHERS:
        root = Path(str(roots[teacher])).expanduser()
        matches = sorted(root.rglob("*.features.iac"))
        if not matches:
            raise FileNotFoundError(f"no {teacher} feature packages under {root}")
        packages[teacher] = matches
    return packages


def _normalized(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)


def _annotation_digest(annotations: dict[str, dict[str, Any]]) -> str:
    payload = json.dumps(annotations, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_features(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, list[Path]]]:
    module = _load_curve_module()
    packages = _feature_packages()
    store = module.FeatureStore(packages)
    tile_ids = [str(row["tile_id"]) for row in rows]
    result: dict[str, np.ndarray] = {}
    try:
        for teacher in TEACHERS:
            values: list[np.ndarray] = []
            for index, tile_id in enumerate(tile_ids, start=1):
                values.append(np.asarray(store.read(teacher, tile_id), dtype=np.float32).reshape(-1))
                if index == 1 or index % 500 == 0 or index == len(tile_ids):
                    print(f"[prototype-bank] {teacher}: {index}/{len(tile_ids)}", flush=True)
            result[teacher] = _normalized(np.stack(values))
    finally:
        store.close()
    return result, packages


def _facility_order(
    similarity: np.ndarray,
    count: int,
    *,
    margin_rank: np.ndarray | None = None,
) -> tuple[list[int], list[float]]:
    """Greedy coverage order, optionally favoring global class margin."""
    n = int(similarity.shape[0])
    if margin_rank is None:
        margin_rank = np.zeros(n, dtype=np.float32)
    covered = np.zeros(n, dtype=np.float32)
    selected = np.zeros(n, dtype=bool)
    order: list[int] = []
    gains: list[float] = []
    for _ in range(min(count, n)):
        gain_matrix = np.maximum(similarity - covered[:, None], 0.0)
        gain = gain_matrix.sum(axis=0, dtype=np.float64)
        score = gain * (1.0 + SEPARATION_WEIGHT * margin_rank)
        score[selected] = -np.inf
        chosen = int(np.argmax(score))
        order.append(chosen)
        gains.append(float(gain[chosen] / n))
        selected[chosen] = True
        covered = np.maximum(covered, similarity[:, chosen])
    return order, gains


def _global_class_margins(
    rows: list[dict[str, Any]],
    features: dict[str, np.ndarray],
    labels: list[str],
) -> np.ndarray:
    row_labels = np.asarray([str(row["l1"]) for row in rows])
    margin = np.zeros(len(rows), dtype=np.float32)
    for teacher in TEACHERS:
        centroids = []
        for label in labels:
            centroid = features[teacher][row_labels == label].mean(axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
            centroids.append(centroid)
        scores = features[teacher] @ np.stack(centroids).T
        for label_index, label in enumerate(labels):
            label_mask = row_labels == label
            competitor = np.max(
                np.delete(scores[label_mask], label_index, axis=1),
                axis=1,
            )
            margin[label_mask] += (
                scores[label_mask, label_index] - competitor
            ) / len(TEACHERS)
    return margin


def _rank_margin(
    indices: list[int],
    margin: np.ndarray,
) -> np.ndarray:
    values = margin[np.asarray(indices)]
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(indices), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(indices), dtype=np.float32)
    return ranks


def _leave_one_out_novelty(
    similarity: np.ndarray,
    selected_order: list[int],
    checkpoint: int,
) -> float:
    selected = np.asarray(selected_order[:checkpoint], dtype=np.int64)
    scores = similarity[:, selected].copy()
    positions = {sample_index: column for column, sample_index in enumerate(selected.tolist())}
    for sample_index, column in positions.items():
        scores[sample_index, column] = -np.inf
    covered = scores.max(axis=1)
    covered[~np.isfinite(covered)] = 0.0
    return float(np.mean(1.0 - covered))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "tile_id",
        "slide_id",
        "patient_id",
        "level1_label",
        "source_split",
        "adjudicated",
        "dataset",
        "iac",
        "row",
        "selection_rank_within_class",
        "four_teacher_marginal_gain",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _plot(curves: list[dict[str, Any]], path: Path) -> None:
    labels = sorted({str(row["label"]) for row in curves})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True)
    for axis, label in zip(axes.ravel(), labels):
        subset = [row for row in curves if row["label"] == label]
        for teacher in TEACHERS:
            rows = [row for row in subset if row["teacher"] == teacher]
            axis.plot(
                [row["count"] for row in rows],
                [row["remaining_novelty"] for row in rows],
                marker="o",
                linewidth=1.5,
                markersize=3,
                label=teacher,
            )
        aggregate = [row for row in subset if row["teacher"] == "four_teacher_mean"]
        axis.plot(
            [row["count"] for row in aggregate],
            [row["remaining_novelty"] for row in aggregate],
            color="black",
            linewidth=2.4,
            label="four-teacher mean",
        )
        axis.set_title(f"{label}\nN={max(row['count'] for row in aggregate)}")
        axis.grid(alpha=0.22)
        axis.set_xlabel("selected prototypes")
        axis.set_ylabel("leave-one-out remaining novelty")
    handles, names = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("L1 fixed prototype bank: four-teacher coverage by sample count")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_delta(curves: list[dict[str, Any]], path: Path) -> None:
    labels = sorted({str(row["label"]) for row in curves})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True)
    colors = {
        "gigapath": "tab:blue",
        "h_optimus_1": "tab:orange",
        "uni2_h": "tab:green",
        "virchow2": "tab:red",
        "four_teacher_mean": "black",
    }
    for axis, label in zip(axes.ravel(), labels):
        subset = [row for row in curves if row["label"] == label]
        for teacher in (*TEACHERS, "four_teacher_mean"):
            rows = sorted(
                (row for row in subset if row["teacher"] == teacher and int(row["count"]) <= 400),
                key=lambda row: int(row["count"]),
            )
            counts = [int(row["count"]) for row in rows]
            novelty = [float(row["remaining_novelty"]) for row in rows]
            marginal = [
                (novelty[index - 1] - novelty[index])
                / (counts[index] - counts[index - 1])
                for index in range(1, len(counts))
            ]
            baseline = marginal[0] if marginal and marginal[0] > 0 else 1.0
            ratio = [value / baseline for value in marginal]
            axis.plot(
                counts[1:],
                ratio,
                marker="o",
                linewidth=2.2 if teacher == "four_teacher_mean" else 1.4,
                markersize=3,
                color=colors[teacher],
                label=teacher,
            )
        axis.axhline(0.35, color="0.4", linestyle="--", linewidth=1.2)
        axis.axvspan(300, 400, color="0.5", alpha=0.08)
        axis.set_title(label)
        axis.grid(alpha=0.22)
        axis.set_xlabel("selected prototypes")
        axis.set_ylabel("normalized marginal gain per sample")
    handles, names = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(
        "L1 prototype Delta: final three count increments must remain below 0.35"
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    payload = json.loads(ANNOTATION_PATH.read_text(encoding="utf-8"))
    annotations = payload.get("annotations")
    if not isinstance(annotations, dict):
        raise ValueError(f"missing annotations object: {ANNOTATION_PATH}")
    keyed_rows = [
        (key, dict(value))
        for key, value in annotations.items()
        if value.get("tile_id") and value.get("l1")
    ]
    keyed_rows.sort(key=lambda item: str(item[1]["tile_id"]))
    keys = [key for key, _ in keyed_rows]
    rows = [row for _, row in keyed_rows]
    features, packages = _load_features(rows)

    labels = [str(label) for label in payload.get("l1_prototypes", [])]
    global_margin = _global_class_margins(rows, features, labels)
    selected_keys: list[str] = []
    selected_rows: list[dict[str, Any]] = []
    selection_report: dict[str, Any] = {}
    curve_rows: list[dict[str, Any]] = []

    for label in labels:
        indices = [index for index, row in enumerate(rows) if str(row["l1"]) == label]
        target = TARGET_PER_CLASS
        if len(indices) < target:
            raise ValueError(f"{label} has {len(indices)} samples, below target={target}")

        teacher_sims: dict[str, np.ndarray] = {}
        combined = np.zeros((len(indices), len(indices)), dtype=np.float32)
        for teacher in TEACHERS:
            matrix = features[teacher][indices]
            similarity = np.clip((matrix @ matrix.T + 1.0) * 0.5, 0.0, 1.0)
            teacher_sims[teacher] = similarity
            combined += similarity / len(TEACHERS)

        margin_rank = _rank_margin(
            indices,
            global_margin,
        )
        local_order, gains = _facility_order(
            combined,
            target,
            margin_rank=margin_rank,
        )
        chosen_global = [indices[index] for index in local_order]
        chosen = [rows[index] for index in chosen_global]
        for rank, (global_index, item, gain) in enumerate(
            zip(chosen_global, chosen, gains), start=1
        ):
            selected_keys.append(keys[global_index])
            selected_rows.append(
                {
                    "tile_id": item["tile_id"],
                    "slide_id": item.get("slide") or item["tile_id"],
                    "patient_id": item.get("patient_id") or item.get("slide") or item["tile_id"],
                    "level1_label": item["l1"],
                    "source_split": "train",
                    "adjudicated": "true",
                    "dataset": item.get("dataset", ""),
                    "iac": item.get("iac", ""),
                    "row": item.get("row", ""),
                    "selection_rank_within_class": rank,
                    "four_teacher_marginal_gain": f"{gain:.10g}",
                }
            )

        checkpoints = [count for count in CHECKPOINTS if count <= target]
        if target not in checkpoints:
            checkpoints.append(target)
        aggregate_novelty: list[float] = []
        teacher_novelty: dict[str, list[float]] = {}
        for teacher, similarity in teacher_sims.items():
            values = [
                _leave_one_out_novelty(similarity, local_order, checkpoint)
                for checkpoint in checkpoints
            ]
            teacher_novelty[teacher] = values
            for checkpoint, novelty in zip(checkpoints, values):
                curve_rows.append(
                    {
                        "label": label,
                        "teacher": teacher,
                        "count": checkpoint,
                        "remaining_novelty": novelty,
                    }
                )
        aggregate_novelty = [
            float(np.mean([teacher_novelty[teacher][index] for teacher in TEACHERS]))
            for index in range(len(checkpoints))
        ]
        for checkpoint, novelty in zip(checkpoints, aggregate_novelty):
            curve_rows.append(
                {
                    "label": label,
                    "teacher": "four_teacher_mean",
                    "count": checkpoint,
                    "remaining_novelty": novelty,
                }
            )
        standard_indices = [
            index for index, count in enumerate(checkpoints) if count <= TARGET_PER_CLASS
        ]
        standard_counts = [checkpoints[index] for index in standard_indices]
        teacher_tail_ratios: dict[str, list[float]] = {}
        for teacher in (*TEACHERS, "four_teacher_mean"):
            values = (
                aggregate_novelty
                if teacher == "four_teacher_mean"
                else teacher_novelty[teacher]
            )
            standard_values = [values[index] for index in standard_indices]
            marginal_per_sample = [
                (standard_values[index - 1] - standard_values[index])
                / (standard_counts[index] - standard_counts[index - 1])
                for index in range(1, len(standard_counts))
            ]
            early = marginal_per_sample[0] if marginal_per_sample else 0.0
            teacher_tail_ratios[teacher] = [
                value / early if early > 0 else 0.0
                for value in marginal_per_sample[-3:]
            ]
        tail_ratios = teacher_tail_ratios["four_teacher_mean"]
        teacher_pass = {
            teacher: bool(
                len(ratios) == 3 and all(value <= 0.35 for value in ratios)
            )
            for teacher, ratios in teacher_tail_ratios.items()
            if teacher != "four_teacher_mean"
        }
        selection_report[label] = {
            "available": len(indices),
            "selected": target,
            "four_teacher_tail_gain_ratios": tail_ratios,
            "tail_gain_ratios_by_teacher": teacher_tail_ratios,
            "teacher_pass": teacher_pass,
            "global_margin_weight": SEPARATION_WEIGHT,
            "mean_selected_global_margin": float(
                np.mean(global_margin[chosen_global])
            ),
            "three_consecutive_low_gain": bool(
                len(tail_ratios) == 3
                and all(value <= 0.35 for value in tail_ratios)
                and all(teacher_pass.values())
            ),
        }
        print(
            f"[prototype-bank] {label}: selected={target}/{len(indices)} "
            f"tail_ratios={','.join(f'{value:.3f}' for value in tail_ratios)}",
            flush=True,
        )

    selected_annotations = {key: annotations[key] for key in selected_keys}
    output_payload = {
        **{key: value for key, value in payload.items() if key != "annotations"},
        "annotations": selected_annotations,
        "prototype_bank": {
            "schema_version": 1,
            "selection": (
                "sample-level global class-margin-weighted greedy facility coverage"
            ),
            "teacher_aggregation": "equal contribution after per-teacher unit normalization",
            "global_margin_weight": SEPARATION_WEIGHT,
            "selection_axis": "sample count within each L1 label",
            "self_similarity_in_curve": "excluded",
            "target_per_class": TARGET_PER_CLASS,
            "source_annotation": str(ANNOTATION_PATH),
            "source_annotation_digest": _annotation_digest(annotations),
            "selected_count": len(selected_annotations),
            "class_counts": dict(Counter(row["l1"] for row in selected_annotations.values())),
        },
    }
    OUTPUT_JSON.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(OUTPUT_CSV, selected_rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORT_DIR / "coverage_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "teacher", "count", "remaining_novelty"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(curve_rows)
    report = {
        "source_annotation_digest": _annotation_digest(annotations),
        "teachers": list(TEACHERS),
        "teacher_feature_packages": {
            teacher: [str(path) for path in paths] for teacher, paths in packages.items()
        },
        "selection": selection_report,
        "all_classes_pass_three_increment_rule": all(
            value["three_consecutive_low_gain"] for value in selection_report.values()
        ),
    }
    (REPORT_DIR / "selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _plot(curve_rows, REPORT_DIR / "coverage_curve.png")
    _plot_delta(curve_rows, REPORT_DIR / "delta_curve.png")
    print(
        f"[prototype-bank] wrote {len(selected_annotations)} samples to {OUTPUT_JSON}",
        flush=True,
    )


if __name__ == "__main__":
    main()
