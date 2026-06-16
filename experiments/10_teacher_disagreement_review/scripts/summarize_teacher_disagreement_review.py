from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np


TEACHERS = ("gigapath", "h_optimus_1", "uni2_h", "virchow2")
GROUPS = ("random500", "top500")
MODEL_COLUMNS = ("pred_full", "pred_a1", "pred_a2", "pred_a3", "pred_a4", "pred_a6")
MODEL_BASE_DISPLAY = {
    "pred_full": "HCC-SemPath full",
    "pred_a1": "A1 no prototype",
    "pred_a2": "A2 no adjudication",
    "pred_a3": "A3 single teacher",
    "pred_a4": "A4 single teacher + prototype",
    "pred_a6": "A6 complete filter",
}


def _model_display(column: str) -> str:
    base = MODEL_BASE_DISPLAY[column]
    if column == "pred_full":
        return base
    return f"{base} (matched 10%-scale)"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else float("nan")


def _l2_columns(rows: list[dict[str, str]]) -> list[str]:
    return sorted(key for key in rows[0] if key.startswith("l2_"))


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _classification_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, float]:
    total = len(y_true)
    correct = sum(a == b for a, b in zip(y_true, y_pred))
    per_f1 = []
    per_recall = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        support = sum(a == label for a in y_true)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        if support:
            per_f1.append(f1)
            per_recall.append(recall)
    return {
        "accuracy": _safe_div(correct, total),
        "balanced_accuracy": float(mean(per_recall)) if per_recall else 0.0,
        "macro_f1": float(mean(per_f1)) if per_f1 else 0.0,
    }


def _per_class_metric_rows(
    rows: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
    labels: list[str],
) -> list[dict[str, object]]:
    pred_by_id = {row["review_id"]: row for row in prediction_rows}
    sources: dict[str, tuple[str, str, str]] = {
        "teacher_plurality": ("teacher_plurality", "review", "plurality_l1_name"),
    }
    for teacher in TEACHERS:
        sources[f"teacher_{teacher}"] = (teacher, "review", f"{teacher}_l1")
    for column in MODEL_COLUMNS:
        sources[column] = (_model_display(column), "prediction", column)

    output = []
    for group in (*GROUPS, "all"):
        group_rows = rows if group == "all" else [row for row in rows if row.get("source_group") == group]
        for source, (source_name, source_type, column) in sources.items():
            pairs = []
            for row in group_rows:
                if source_type == "prediction":
                    predicted = pred_by_id.get(row["review_id"], {}).get(column, "")
                    if predicted in {"", "N/A"}:
                        continue
                else:
                    predicted = row[column]
                pairs.append((row["l1"], predicted))
            for label in labels:
                tp = sum(truth == label and pred == label for truth, pred in pairs)
                fp = sum(truth != label and pred == label for truth, pred in pairs)
                fn = sum(truth == label and pred != label for truth, pred in pairs)
                support = sum(truth == label for truth, _ in pairs)
                precision = _safe_div(tp, tp + fp)
                recall = _safe_div(tp, tp + fn)
                f1 = _safe_div(2 * precision * recall, precision + recall)
                output.append({
                    "source_group": group,
                    "source": source,
                    "source_name": source_name,
                    "l1": label,
                    "support": support,
                    "precision": round(precision, 6),
                    "recall": round(recall, 6),
                    "f1": round(f1, 6),
                })
    return output


def _bootstrap_accuracy_ci(y_true: list[str], y_pred: list[str], seed: int = 13, rounds: int = 2000) -> tuple[float, float]:
    if not y_true:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    ok = np.asarray([a == b for a, b in zip(y_true, y_pred)], dtype=np.float32)
    values = []
    for _ in range(rounds):
        idx = rng.integers(0, len(ok), size=len(ok))
        values.append(float(ok[idx].mean()))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    # Average ranks for ties.
    sorted_scores = y_score[order]
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    pos_ranks = ranks[y_true == 1]
    return float((pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = int(y_true.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true)
    precision = tp / (np.arange(len(sorted_true)) + 1)
    return float((precision * sorted_true).sum() / positives)


def _summarize_group(rows: list[dict[str, str]], group: str) -> dict[str, object]:
    vals = [_float(row, "disagreement_score") for row in rows]
    reviewed = sum(row.get("reviewed") == "True" for row in rows)
    adjudicated = sum(row.get("adjudication_status") == "adjudicated" for row in rows)
    return {
        "source_group": group,
        "tiles": len(rows),
        "reviewed": reviewed,
        "adjudicated": adjudicated,
        "disagreement_mean": round(mean(vals), 6),
        "disagreement_median": round(median(vals), 6),
        "disagreement_min": round(min(vals), 6),
        "disagreement_max": round(max(vals), 6),
    }


def _label_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output = []
    for group in (*GROUPS, "all"):
        group_rows = rows if group == "all" else [row for row in rows if row.get("source_group") == group]
        counts = Counter(row.get("l1", "") for row in group_rows)
        for label, count in sorted(counts.items()):
            output.append({
                "source_group": group,
                "l1": label,
                "tiles": count,
                "fraction": round(count / len(group_rows), 6) if group_rows else 0.0,
            })
    return output


def _coverage_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output = []
    for group in (*GROUPS, "all"):
        group_rows = rows if group == "all" else [row for row in rows if row.get("source_group") == group]
        output.append({
            "source_group": group,
            "tiles": len(group_rows),
            "slides": len({row.get("slide_id", "") for row in group_rows}),
            "l1_classes": len({row.get("l1", "") for row in group_rows}),
            "reviewed": sum(row.get("reviewed") == "True" for row in group_rows),
            "adjudicated": sum(row.get("adjudication_status") == "adjudicated" for row in group_rows),
        })
    return output


def _teacher_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output = []
    for group in (*GROUPS, "all"):
        group_rows = rows if group == "all" else [row for row in rows if row.get("source_group") == group]
        for teacher in TEACHERS:
            matches = sum(row.get(f"{teacher}_l1") == row.get("l1") for row in group_rows)
            output.append({
                "source_group": group,
                "teacher": teacher,
                "tiles": len(group_rows),
                "expert_l1_matches": matches,
                "expert_l1_accuracy": round(matches / len(group_rows), 6) if group_rows else 0.0,
            })
    return output


def _queue_validation_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output = []
    for group in (*GROUPS, "all"):
        group_rows = rows if group == "all" else [row for row in rows if row.get("source_group") == group]
        diffs = []
        vote_entropy = []
        primary_pairwise = []
        attribute_pairwise = []
        primary_match = []
        for row in group_rows:
            reconstructed = (
                _float(row, "vote_entropy")
                + _float(row, "primary_pairwise_l1")
                + _float(row, "attribute_pairwise_l1")
            )
            diffs.append(abs(_float(row, "disagreement_score") - reconstructed))
            vote_entropy.append(_float(row, "vote_entropy"))
            primary_pairwise.append(_float(row, "primary_pairwise_l1"))
            attribute_pairwise.append(_float(row, "attribute_pairwise_l1"))
            primary_match.append(_float(row, "primary_match_fraction"))
        output.append({
            "source_group": group,
            "tiles": len(group_rows),
            "score_formula": "vote_entropy + primary_pairwise_l1 + attribute_pairwise_l1",
            "max_formula_abs_error": round(max(diffs), 8) if diffs else 0.0,
            "vote_entropy_mean": round(mean(vote_entropy), 6) if vote_entropy else 0.0,
            "primary_pairwise_l1_mean": round(mean(primary_pairwise), 6) if primary_pairwise else 0.0,
            "attribute_pairwise_l1_mean": round(mean(attribute_pairwise), 6) if attribute_pairwise else 0.0,
            "primary_match_fraction_mean": round(mean(primary_match), 6) if primary_match else 0.0,
        })
    return output


def _queue_provenance_rows() -> list[dict[str, object]]:
    return [
        {
            "source_group": "random500",
            "selection_basis": "uniform random sample from external-validation candidates",
            "ranking_key": "random seed",
            "expert_labels_used_for_selection": "False",
            "model_predictions_used_for_selection": "False",
            "teacher_outputs_used_for_selection": "False",
            "role": "population reference queue",
        },
        {
            "source_group": "top500",
            "selection_basis": "highest teacher-only disagreement among non-degenerate external-validation candidates",
            "ranking_key": "disagreement_score = vote_entropy + primary_pairwise_l1 + attribute_pairwise_l1",
            "expert_labels_used_for_selection": "False",
            "model_predictions_used_for_selection": "False",
            "teacher_outputs_used_for_selection": "True",
            "role": "teacher-conflict stress queue",
        },
    ]


def _plurality_rows(rows: list[dict[str, str]], labels: list[str]) -> list[dict[str, object]]:
    output = []
    for group in (*GROUPS, "all"):
        group_rows = rows if group == "all" else [row for row in rows if row.get("source_group") == group]
        y_true = [row["l1"] for row in group_rows]
        y_pred = [row["plurality_l1_name"] for row in group_rows]
        metrics = _classification_metrics(y_true, y_pred, labels)
        lo, hi = _bootstrap_accuracy_ci(y_true, y_pred)
        output.append({
            "source_group": group,
            "baseline": "teacher_plurality",
            "tiles": len(group_rows),
            "accuracy": round(metrics["accuracy"], 6),
            "accuracy_ci_low": round(lo, 6),
            "accuracy_ci_high": round(hi, 6),
            "balanced_accuracy": round(metrics["balanced_accuracy"], 6),
            "macro_f1": round(metrics["macro_f1"], 6),
        })
    return output


def _model_rows(
    rows: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
    labels: list[str],
) -> list[dict[str, object]]:
    review_by_id = {row["review_id"]: row for row in rows}
    output = []
    for group in (*GROUPS, "all"):
        group_review_ids = {
            row["review_id"]
            for row in rows
            if group == "all" or row.get("source_group") == group
        }
        group_pred_rows = [row for row in prediction_rows if row["review_id"] in group_review_ids]
        for column in MODEL_COLUMNS:
            valid = [row for row in group_pred_rows if row.get(column, "") not in {"", "N/A"}]
            y_true = [review_by_id[row["review_id"]]["l1"] for row in valid]
            y_pred = [row[column] for row in valid]
            metrics = _classification_metrics(y_true, y_pred, labels)
            lo, hi = _bootstrap_accuracy_ci(y_true, y_pred)
            output.append({
                "source_group": group,
                "model": column,
                "model_name": _model_display(column),
                "tiles": len(valid),
                "accuracy": round(metrics["accuracy"], 6),
                "accuracy_ci_low": round(lo, 6),
                "accuracy_ci_high": round(hi, 6),
                "balanced_accuracy": round(metrics["balanced_accuracy"], 6),
                "macro_f1": round(metrics["macro_f1"], 6),
            })
    return output


def _teacher_metric_rows(rows: list[dict[str, str]], labels: list[str]) -> list[dict[str, object]]:
    output = []
    for group in (*GROUPS, "all"):
        group_rows = rows if group == "all" else [row for row in rows if row.get("source_group") == group]
        y_true = [row["l1"] for row in group_rows]
        for teacher in TEACHERS:
            y_pred = [row[f"{teacher}_l1"] for row in group_rows]
            metrics = _classification_metrics(y_true, y_pred, labels)
            lo, hi = _bootstrap_accuracy_ci(y_true, y_pred)
            output.append({
                "source_group": group,
                "teacher": teacher,
                "tiles": len(group_rows),
                "accuracy": round(metrics["accuracy"], 6),
                "accuracy_ci_low": round(lo, 6),
                "accuracy_ci_high": round(hi, 6),
                "balanced_accuracy": round(metrics["balanced_accuracy"], 6),
                "macro_f1": round(metrics["macro_f1"], 6),
            })
    return output


def _confusion_rows(rows: list[dict[str, str]], prediction_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    review_by_id = {row["review_id"]: row for row in rows}
    sources = {f"teacher_{teacher}": (teacher, f"{teacher}_l1") for teacher in TEACHERS}
    sources["baseline_teacher_plurality"] = ("review", "plurality_l1_name")
    for column in MODEL_COLUMNS:
        sources[f"model_{column}"] = ("prediction", column)
    output = []
    for group in (*GROUPS, "all"):
        group_rows = rows if group == "all" else [row for row in rows if row.get("source_group") == group]
        group_ids = {row["review_id"] for row in group_rows}
        pred_by_id = {row["review_id"]: row for row in prediction_rows if row["review_id"] in group_ids}
        for source_name, (source_type, column) in sources.items():
            counts: Counter[tuple[str, str]] = Counter()
            for row in group_rows:
                if source_type == "prediction":
                    pred_row = pred_by_id.get(row["review_id"], {})
                    predicted = pred_row.get(column, "")
                    if predicted in {"", "N/A"}:
                        continue
                else:
                    predicted = row[column]
                counts[(row["l1"], predicted)] += 1
            for (expert_l1, predicted_l1), count in sorted(counts.items()):
                output.append({
                    "source_group": group,
                    "source": source_name,
                    "expert_l1": expert_l1,
                    "predicted_l1": predicted_l1,
                    "tiles": count,
                })
    return output


def _l2_metric_rows(rows: list[dict[str, str]], l2_npz_path: Path) -> list[dict[str, object]]:
    if not l2_npz_path.exists():
        return []
    l2_cols = _l2_columns(rows)
    row_by_id = {row["review_id"]: row for row in rows}
    payload = np.load(l2_npz_path, allow_pickle=True)
    review_ids = [str(x) for x in payload["review_ids"].tolist()]
    l2_names = [str(x) for x in payload["l2_names"].tolist()]
    output = []
    for group in (*GROUPS, "all"):
        group_ids = {
            row["review_id"]
            for row in rows
            if group == "all" or row.get("source_group") == group
        }
        idx = np.asarray([i for i, rid in enumerate(review_ids) if rid in group_ids], dtype=np.int64)
        if len(idx) == 0:
            continue
        y = np.asarray([
            [1 if row_by_id[review_ids[i]][f"l2_{name}"] == "True" else 0 for name in l2_names]
            for i in idx
        ], dtype=np.int8)
        for model in MODEL_COLUMNS:
            if model not in payload:
                continue
            scores = payload[model][idx]
            aucs = []
            aps = []
            f1s = []
            for j, name in enumerate(l2_names):
                truth = y[:, j]
                score = scores[:, j]
                auc = _binary_auc(truth, score)
                ap = _average_precision(truth, score)
                pred = (score >= 0.5).astype(np.int8)
                tp = int(((truth == 1) & (pred == 1)).sum())
                fp = int(((truth == 0) & (pred == 1)).sum())
                fn = int(((truth == 1) & (pred == 0)).sum())
                precision = _safe_div(tp, tp + fp)
                recall = _safe_div(tp, tp + fn)
                f1 = _safe_div(2 * precision * recall, precision + recall)
                if not np.isnan(auc):
                    aucs.append(auc)
                if not np.isnan(ap):
                    aps.append(ap)
                f1s.append(f1)
                output.append({
                    "source_group": group,
                    "model": model,
                    "model_name": _model_display(model),
                    "attribute": name,
                    "positives": int(truth.sum()),
                    "auc": round(auc, 6) if not np.isnan(auc) else "",
                    "average_precision": round(ap, 6) if not np.isnan(ap) else "",
                    "f1_at_0_5": round(f1, 6),
                })
            output.append({
                "source_group": group,
                "model": model,
                "model_name": _model_display(model),
                "attribute": "macro",
                "positives": int(y.sum()),
                "auc": round(float(mean(aucs)), 6) if aucs else "",
                "average_precision": round(float(mean(aps)), 6) if aps else "",
                "f1_at_0_5": round(float(mean(f1s)), 6) if f1s else "",
            })
    return output


def _topn_sensitivity_rows(
    rows: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
    labels: list[str],
) -> list[dict[str, object]]:
    review_by_id = {row["review_id"]: row for row in rows}
    pred_by_id = {row["review_id"]: row for row in prediction_rows}
    top_rows = [row for row in rows if row.get("source_group") == "top500"]
    output = []
    sources: dict[str, tuple[str, str]] = {
        "teacher_plurality": ("review", "plurality_l1_name"),
        "teacher_gigapath": ("review", "gigapath_l1"),
        "teacher_h_optimus_1": ("review", "h_optimus_1_l1"),
        "teacher_uni2_h": ("review", "uni2_h_l1"),
        "teacher_virchow2": ("review", "virchow2_l1"),
        "model_pred_full": ("prediction", "pred_full"),
        "model_pred_a0": ("prediction", "pred_a0"),
        "model_pred_a2": ("prediction", "pred_a2"),
    }
    for topn in (100, 250, 500):
        subset = [row for row in top_rows if int(row["rank"]) <= topn]
        for source, (source_type, column) in sources.items():
            y_true = []
            y_pred = []
            for row in subset:
                if source_type == "prediction":
                    predicted = pred_by_id.get(row["review_id"], {}).get(column, "")
                    if predicted in {"", "N/A"}:
                        continue
                else:
                    predicted = row[column]
                y_true.append(review_by_id[row["review_id"]]["l1"])
                y_pred.append(predicted)
            metrics = _classification_metrics(y_true, y_pred, labels)
            output.append({
                "topn": topn,
                "source": source,
                "tiles": len(y_true),
                "accuracy": round(metrics["accuracy"], 6),
                "balanced_accuracy": round(metrics["balanced_accuracy"], 6),
                "macro_f1": round(metrics["macro_f1"], 6),
            })
    return output


def _conflict_bin_rows(
    rows: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
    labels: list[str],
) -> list[dict[str, object]]:
    pred_by_id = {row["review_id"]: row for row in prediction_rows}
    sorted_rows = sorted(rows, key=lambda row: _float(row, "disagreement_score"))
    bins = np.array_split(np.asarray(sorted_rows, dtype=object), 4)
    sources: dict[str, tuple[str, str]] = {
        "teacher_plurality": ("review", "plurality_l1_name"),
        "teacher_gigapath": ("review", "gigapath_l1"),
        "teacher_h_optimus_1": ("review", "h_optimus_1_l1"),
        "teacher_uni2_h": ("review", "uni2_h_l1"),
        "teacher_virchow2": ("review", "virchow2_l1"),
        "model_pred_full": ("prediction", "pred_full"),
        "model_pred_a0": ("prediction", "pred_a0"),
        "model_pred_a2": ("prediction", "pred_a2"),
    }
    output = []
    for idx, bin_rows_array in enumerate(bins, start=1):
        bin_rows = [dict(row) for row in bin_rows_array.tolist()]
        scores = [_float(row, "disagreement_score") for row in bin_rows]
        top_fraction = _safe_div(sum(row.get("source_group") == "top500" for row in bin_rows), len(bin_rows))
        for source, (source_type, column) in sources.items():
            y_true = []
            y_pred = []
            for row in bin_rows:
                if source_type == "prediction":
                    predicted = pred_by_id.get(row["review_id"], {}).get(column, "")
                    if predicted in {"", "N/A"}:
                        continue
                else:
                    predicted = row[column]
                y_true.append(row["l1"])
                y_pred.append(predicted)
            metrics = _classification_metrics(y_true, y_pred, labels)
            output.append({
                "conflict_bin": f"Q{idx}",
                "tiles": len(y_true),
                "disagreement_min": round(min(scores), 6),
                "disagreement_max": round(max(scores), 6),
                "top500_fraction": round(top_fraction, 6),
                "source": source,
                "accuracy": round(metrics["accuracy"], 6),
                "balanced_accuracy": round(metrics["balanced_accuracy"], 6),
                "macro_f1": round(metrics["macro_f1"], 6),
            })
    return output


def _paired_delta_ci(
    y_true: list[str],
    pred_a: list[str],
    pred_b: list[str],
    seed: int = 13,
    rounds: int = 2000,
) -> tuple[float, float, float]:
    if not y_true:
        return 0.0, 0.0, 0.0
    a_ok = np.asarray([truth == pred for truth, pred in zip(y_true, pred_a)], dtype=np.float32)
    b_ok = np.asarray([truth == pred for truth, pred in zip(y_true, pred_b)], dtype=np.float32)
    delta = float((a_ok - b_ok).mean())
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(rounds):
        idx = rng.integers(0, len(a_ok), size=len(a_ok))
        values.append(float((a_ok[idx] - b_ok[idx]).mean()))
    return delta, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _paired_comparison_rows(
    rows: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    review_by_id = {row["review_id"]: row for row in rows}
    pred_by_id = {row["review_id"]: row for row in prediction_rows}
    comparison_sources: dict[str, tuple[str, str]] = {
        "teacher_plurality": ("review", "plurality_l1_name"),
        "teacher_gigapath": ("review", "gigapath_l1"),
        "teacher_h_optimus_1": ("review", "h_optimus_1_l1"),
        "teacher_uni2_h": ("review", "uni2_h_l1"),
        "teacher_virchow2": ("review", "virchow2_l1"),
        "pred_a1": ("prediction", "pred_a1"),
        "pred_a2": ("prediction", "pred_a2"),
        "pred_a3": ("prediction", "pred_a3"),
        "pred_a4": ("prediction", "pred_a4"),
        "pred_a6": ("prediction", "pred_a6"),
    }
    output = []
    for group in (*GROUPS, "all"):
        group_rows = rows if group == "all" else [row for row in rows if row.get("source_group") == group]
        for source_name, (source_type, column) in comparison_sources.items():
            y_true = []
            full_pred = []
            other_pred = []
            full_only = 0
            other_only = 0
            both_correct = 0
            both_wrong = 0
            for row in group_rows:
                pred_row = pred_by_id.get(row["review_id"], {})
                full = pred_row.get("pred_full", "")
                if full in {"", "N/A"}:
                    continue
                if source_type == "prediction":
                    other = pred_row.get(column, "")
                    if other in {"", "N/A"}:
                        continue
                else:
                    other = row[column]
                truth = row["l1"]
                full_ok = full == truth
                other_ok = other == truth
                both_correct += int(full_ok and other_ok)
                full_only += int(full_ok and not other_ok)
                other_only += int(other_ok and not full_ok)
                both_wrong += int(not full_ok and not other_ok)
                y_true.append(truth)
                full_pred.append(full)
                other_pred.append(other)
            delta, lo, hi = _paired_delta_ci(y_true, full_pred, other_pred)
            output.append({
                "source_group": group,
                "reference": "pred_full",
                "comparison": source_name,
                "tiles": len(y_true),
                "accuracy_delta": round(delta, 6),
                "delta_ci_low": round(lo, 6),
                "delta_ci_high": round(hi, 6),
                "full_only_correct": full_only,
                "comparison_only_correct": other_only,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
            })
    return output


def _paired_ablation_rows(
    rows: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    pred_by_id = {row["review_id"]: row for row in prediction_rows}
    comparisons = (
        ("pred_a1", "pred_a3"),
        ("pred_a2", "pred_a1"),
        ("pred_a4", "pred_a3"),
        ("pred_a6", "pred_a1"),
        ("pred_a6", "pred_a4"),
        ("pred_a6", "pred_a2"),
        ("pred_a0", "pred_a5"),
    )
    output = []
    for group in (*GROUPS, "all"):
        group_rows = rows if group == "all" else [row for row in rows if row.get("source_group") == group]
        for reference_column, comparison_column in comparisons:
            y_true = []
            reference_pred = []
            comparison_pred = []
            reference_only = 0
            comparison_only = 0
            both_correct = 0
            both_wrong = 0
            for row in group_rows:
                pred_row = pred_by_id.get(row["review_id"], {})
                reference = pred_row.get(reference_column, "")
                other = pred_row.get(comparison_column, "")
                if reference in {"", "N/A"} or other in {"", "N/A"}:
                    continue
                truth = row["l1"]
                reference_ok = reference == truth
                comparison_ok = other == truth
                both_correct += int(reference_ok and comparison_ok)
                reference_only += int(reference_ok and not comparison_ok)
                comparison_only += int(comparison_ok and not reference_ok)
                both_wrong += int(not reference_ok and not comparison_ok)
                y_true.append(truth)
                reference_pred.append(reference)
                comparison_pred.append(other)
            delta, lo, hi = _paired_delta_ci(y_true, reference_pred, comparison_pred)
            output.append({
                "source_group": group,
                "reference": reference_column,
                "comparison": comparison_column,
                "tiles": len(y_true),
                "accuracy_delta": round(delta, 6),
                "delta_ci_low": round(lo, 6),
                "delta_ci_high": round(hi, 6),
                "reference_only_correct": reference_only,
                "comparison_only_correct": comparison_only,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review",
        default="annotations/reviews/teacher_disagreement/exval_1000/review.csv",
    )
    parser.add_argument(
        "--predictions",
        default="artifacts/caches/local_cache/teacher_disagreement/teacher_disagreement_model_predictions.csv",
    )
    parser.add_argument(
        "--l2-probabilities",
        default="artifacts/caches/local_cache/teacher_disagreement/teacher_disagreement_l2_probabilities.npz",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/10_teacher_disagreement_review",
    )
    args = parser.parse_args()

    review_path = Path(args.review)
    output_dir = Path(args.output_dir)
    rows = _read_csv(review_path)
    if not rows:
        raise SystemExit(f"empty review file: {review_path}")
    prediction_path = Path(args.predictions)
    prediction_rows = _read_csv(prediction_path) if prediction_path.exists() else []
    labels = sorted({row["l1"] for row in rows})

    grouped = {group: [row for row in rows if row.get("source_group") == group] for group in GROUPS}
    if any(len(grouped[group]) == 0 for group in GROUPS):
        raise SystemExit("expected both random500 and top500 source groups")

    summary_rows = [_summarize_group(grouped[group], group) for group in GROUPS]
    summary_rows.append(_summarize_group(rows, "all"))
    queue_rows = _queue_validation_rows(rows)
    provenance_rows = _queue_provenance_rows()
    label_rows = _label_rows(rows)
    coverage_rows = _coverage_rows(rows)
    teacher_rows = _teacher_rows(rows)
    teacher_metric_rows = _teacher_metric_rows(rows, labels)
    plurality_rows = _plurality_rows(rows, labels)
    model_rows = _model_rows(rows, prediction_rows, labels) if prediction_rows else []
    per_class_rows = _per_class_metric_rows(rows, prediction_rows, labels) if prediction_rows else []
    confusion_rows = _confusion_rows(rows, prediction_rows) if prediction_rows else []
    l2_rows = _l2_metric_rows(rows, Path(args.l2_probabilities))
    topn_rows = _topn_sensitivity_rows(rows, prediction_rows, labels) if prediction_rows else []
    conflict_rows = _conflict_bin_rows(rows, prediction_rows, labels) if prediction_rows else []
    paired_rows = _paired_comparison_rows(rows, prediction_rows) if prediction_rows else []
    paired_ablation_rows = _paired_ablation_rows(rows, prediction_rows) if prediction_rows else []

    tables = output_dir / "tables"
    _write_csv(
        tables / "teacher_disagreement_review_summary.csv",
        summary_rows,
        [
            "source_group",
            "tiles",
            "reviewed",
            "adjudicated",
            "disagreement_mean",
            "disagreement_median",
            "disagreement_min",
            "disagreement_max",
        ],
    )
    _write_csv(
        tables / "high_conflict_queue_validation.csv",
        queue_rows,
        [
            "source_group",
            "tiles",
            "score_formula",
            "max_formula_abs_error",
            "vote_entropy_mean",
            "primary_pairwise_l1_mean",
            "attribute_pairwise_l1_mean",
            "primary_match_fraction_mean",
        ],
    )
    _write_csv(
        tables / "queue_construction_provenance.csv",
        provenance_rows,
        [
            "source_group",
            "selection_basis",
            "ranking_key",
            "expert_labels_used_for_selection",
            "model_predictions_used_for_selection",
            "teacher_outputs_used_for_selection",
            "role",
        ],
    )
    _write_csv(tables / "expert_l1_distribution.csv", label_rows, ["source_group", "l1", "tiles", "fraction"])
    _write_csv(
        tables / "expert_asset_coverage.csv",
        coverage_rows,
        ["source_group", "tiles", "slides", "l1_classes", "reviewed", "adjudicated"],
    )
    _write_csv(
        tables / "teacher_vs_expert_l1_accuracy.csv",
        teacher_rows,
        ["source_group", "teacher", "tiles", "expert_l1_matches", "expert_l1_accuracy"],
    )
    _write_csv(
        tables / "teacher_vs_expert_l1_metrics.csv",
        teacher_metric_rows,
        ["source_group", "teacher", "tiles", "accuracy", "accuracy_ci_low", "accuracy_ci_high", "balanced_accuracy", "macro_f1"],
    )
    _write_csv(
        tables / "teacher_plurality_vs_expert_l1_metrics.csv",
        plurality_rows,
        ["source_group", "baseline", "tiles", "accuracy", "accuracy_ci_low", "accuracy_ci_high", "balanced_accuracy", "macro_f1"],
    )
    if model_rows:
        _write_csv(
            tables / "pamtd_ablation_vs_expert_l1_metrics.csv",
            model_rows,
            ["source_group", "model", "model_name", "tiles", "accuracy", "accuracy_ci_low", "accuracy_ci_high", "balanced_accuracy", "macro_f1"],
        )
        _write_csv(
            tables / "l1_confusion_matrices.csv",
            confusion_rows,
            ["source_group", "source", "expert_l1", "predicted_l1", "tiles"],
        )
        _write_csv(
            tables / "l1_per_class_metrics.csv",
            per_class_rows,
            ["source_group", "source", "source_name", "l1", "support", "precision", "recall", "f1"],
        )
    if l2_rows:
        _write_csv(
            tables / "pamtd_ablation_vs_expert_l2_metrics.csv",
            l2_rows,
            ["source_group", "model", "model_name", "attribute", "positives", "auc", "average_precision", "f1_at_0_5"],
        )
    if topn_rows:
        _write_csv(
            tables / "high_conflict_topn_sensitivity.csv",
            topn_rows,
            ["topn", "source", "tiles", "accuracy", "balanced_accuracy", "macro_f1"],
        )
        _write_csv(
            tables / "conflict_quartile_l1_metrics.csv",
            conflict_rows,
            [
                "conflict_bin",
                "tiles",
                "disagreement_min",
                "disagreement_max",
                "top500_fraction",
                "source",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
            ],
        )
    if paired_rows:
        _write_csv(
            tables / "paired_l1_comparisons.csv",
            paired_rows,
            [
                "source_group",
                "reference",
                "comparison",
                "tiles",
                "accuracy_delta",
                "delta_ci_low",
                "delta_ci_high",
                "full_only_correct",
                "comparison_only_correct",
                "both_correct",
                "both_wrong",
            ],
        )
    if paired_ablation_rows:
        _write_csv(
            tables / "paired_ablation_l1_comparisons.csv",
            paired_ablation_rows,
            [
                "source_group",
                "reference",
                "comparison",
                "tiles",
                "accuracy_delta",
                "delta_ci_low",
                "delta_ci_high",
                "reference_only_correct",
                "comparison_only_correct",
                "both_correct",
                "both_wrong",
            ],
        )

    random_summary = summary_rows[0]
    top_summary = summary_rows[1]
    teacher_metric_by_group = {
        (row["source_group"], row["teacher"]): row for row in teacher_metric_rows
    }
    full_model_by_group = {
        row["source_group"]: row for row in model_rows if row["model"] == "pred_full"
    }
    a6_model_by_group = {
        row["source_group"]: row for row in model_rows if row["model"] == "pred_a6"
    }
    conflict_full = {
        row["conflict_bin"]: row for row in conflict_rows if row["source"] == "model_pred_full"
    }
    conflict_plurality = {
        row["conflict_bin"]: row for row in conflict_rows if row["source"] == "teacher_plurality"
    }
    macro_l2 = [
        row for row in l2_rows
        if row.get("attribute") == "macro" and row.get("model") in {"pred_full", "pred_a2", "pred_a6"}
    ]
    report = output_dir / "reports" / "teacher_disagreement_review_summary.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Teacher-Disagreement Expert Review Summary\n\n"
        "This is the current manuscript-level expert annotation asset for teacher-disagreement analysis.\n\n"
        "## Reviewed Queues\n\n"
        f"- `random500`: {random_summary['tiles']} fully random exval tiles; "
        f"{random_summary['adjudicated']} adjudicated.\n"
        f"- `top500`: {top_summary['tiles']} non-degenerate high teacher-conflict exval tiles; "
        f"{top_summary['adjudicated']} adjudicated.\n\n"
        "The queues were constructed before expert adjudication. `top500` is the deterministic "
        "top-ranked teacher-conflict stress queue; expert labels and HCC-SemPath predictions are "
        "not inputs to the selection rule.\n\n"
        "## Conflict Separation\n\n"
        "| source_group | tiles | disagreement_mean | disagreement_median | disagreement_min | disagreement_max |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {row['source_group']} | {row['tiles']} | {row['disagreement_mean']:.4f} | "
            f"{row['disagreement_median']:.4f} | {row['disagreement_min']:.4f} | "
            f"{row['disagreement_max']:.4f} |"
            for row in summary_rows
        )
        + "\n\n"
        "The disagreement score is exactly reconstructed as "
        "`vote_entropy + primary_pairwise_l1 + attribute_pairwise_l1`; "
        "the maximum absolute reconstruction error is "
        f"{max(float(row['max_formula_abs_error']) for row in queue_rows):.2e}.\n\n"
        "## Teacher Agreement With Expert L1\n\n"
        "| source_group | teacher | accuracy | balanced accuracy | macro F1 |\n"
        "|---|---|---:|---:|---:|\n"
        + "\n".join(
            f"| {row['source_group']} | {row['teacher']} | {row['accuracy']:.3f} | "
            f"{row['balanced_accuracy']:.3f} | {row['macro_f1']:.3f} |"
            for row in teacher_metric_rows
        )
        + "\n\n"
        "## PAMT-D / Ablation Agreement With Expert L1\n\n"
        "| source_group | model | accuracy | balanced accuracy | macro F1 |\n"
        "|---|---|---:|---:|---:|\n"
        + "\n".join(
            f"| {row['source_group']} | {row['model']} | {row['accuracy']:.3f} | "
            f"{row['balanced_accuracy']:.3f} | {row['macro_f1']:.3f} |"
            for row in model_rows
        )
        + "\n\n"
        "## Key L2 Macro Results\n\n"
        "| source_group | model | macro AUC | macro AP | macro F1@0.5 |\n"
        "|---|---|---:|---:|---:|\n"
        + "\n".join(
            f"| {row['source_group']} | {row['model']} | {row['auc']} | "
            f"{row['average_precision']} | {row['f1_at_0_5']} |"
            for row in macro_l2
        )
        + "\n\n"
        "## Paired L1 Accuracy Deltas\n\n"
        "| source_group | comparison | full-minus-comparison accuracy delta | 95% CI |\n"
        "|---|---|---:|---:|\n"
        + "\n".join(
            f"| {row['source_group']} | {row['comparison']} | {row['accuracy_delta']:.3f} | "
            f"[{row['delta_ci_low']:.3f}, {row['delta_ci_high']:.3f}] |"
            for row in paired_rows
            if row["comparison"] in {"teacher_plurality", "teacher_uni2_h", "pred_a0", "pred_a2"}
        )
        + "\n\n"
        "## Matched Ablation L1 Deltas\n\n"
        "| source_group | contrast | reference-minus-comparison accuracy delta | 95% CI |\n"
        "|---|---|---:|---:|\n"
        + "\n".join(
            f"| {row['source_group']} | {row['reference']} vs {row['comparison']} | "
            f"{row['accuracy_delta']:.3f} | "
            f"[{row['delta_ci_low']:.3f}, {row['delta_ci_high']:.3f}] |"
            for row in paired_ablation_rows
        )
        + "\n\n"
        "## Conflict Quartile Sensitivity\n\n"
        "| conflict bin | disagreement range | top500 fraction | full accuracy | plurality accuracy |\n"
        "|---|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {bin_name} | {row['disagreement_min']:.4f}-{row['disagreement_max']:.4f} | "
            f"{row['top500_fraction']:.3f} | {row['accuracy']:.3f} | "
            f"{conflict_plurality.get(bin_name, {}).get('accuracy', 'n/a')} |"
            for bin_name, row in conflict_full.items()
        )
        + "\n\n"
        "## Interpretation\n\n"
        f"- Random queue full-model L1 accuracy: {full_model_by_group.get('random500', {}).get('accuracy', 'n/a')}.\n"
        f"- High-conflict queue full-model L1 accuracy: {full_model_by_group.get('top500', {}).get('accuracy', 'n/a')}.\n"
        f"- High-conflict A6 matched 10%-scale L1 accuracy: {a6_model_by_group.get('top500', {}).get('accuracy', 'n/a')}.\n"
        "- The high-conflict queue is a teacher-defined stress set, not a population estimate; "
        "random500 and top500 should be reported separately.\n"
        "- `queue_construction_provenance.csv` records the selection basis and leakage boundary.\n"
        "- `l1_per_class_metrics.csv` and `conflict_quartile_l1_metrics.csv` provide class-level "
        "and conflict-graded checks for reviewer-facing tables.\n"
        "- `high_conflict_topn_sensitivity.csv` reports whether the high-conflict result "
        "is stable across top100/top250/top500 subsets.\n",
        encoding="utf-8",
    )
    print(f"teacher_disagreement_review_ok rows={len(rows)} output={output_dir}")


if __name__ == "__main__":
    main()
