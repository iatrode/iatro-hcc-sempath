from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


MODELS = ("pred_full", "pred_a0", "pred_a1", "pred_a2", "pred_a3", "pred_a4", "pred_a5", "pred_a6")
GROUPS = ("random500", "top500", "all")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _safe_div(num: int, den: int) -> float:
    return num / den if den else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", default="annotations/reviews/teacher_disagreement/exval_1000/review.csv")
    parser.add_argument(
        "--scores",
        default="artifacts/caches/local_cache/teacher_disagreement/teacher_disagreement_l2_probabilities.npz",
    )
    parser.add_argument(
        "--thresholds",
        default="artifacts/caches/local_cache/train_l2_thresholds/thresholds.json",
    )
    parser.add_argument(
        "--output",
        default="experiments/10_teacher_disagreement_review/tables/l2_prior_threshold_ablation_metrics.csv",
    )
    args = parser.parse_args()

    rows = _read_csv(Path(args.review))
    row_by_id = {row["review_id"]: row for row in rows}
    threshold_payload = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    thresholds = np.asarray(threshold_payload["thresholds"], dtype=np.float32)
    names = [str(name) for name in threshold_payload["l2_names"]]

    with np.load(args.scores, allow_pickle=True) as payload:
        review_ids = [str(value) for value in payload["review_ids"].tolist()]
        cached_names = [str(value) for value in payload["l2_names"].tolist()]
        if cached_names != names:
            raise RuntimeError("threshold and score L2 names differ")
        output = []
        for group in GROUPS:
            idx = np.asarray(
                [
                    i
                    for i, review_id in enumerate(review_ids)
                    if group == "all" or row_by_id[review_id]["source_group"] == group
                ],
                dtype=np.int64,
            )
            truth = np.asarray(
                [
                    [
                        row_by_id[review_ids[i]][f"l2_{name}"] == "True"
                        for name in names
                    ]
                    for i in idx
                ],
                dtype=bool,
            )
            for model in MODELS:
                key = f"raw_{model}"
                if key not in payload:
                    continue
                prediction = payload[key][idx] >= thresholds
                per_f1 = []
                per_precision = []
                per_recall = []
                for column in range(len(names)):
                    target = truth[:, column]
                    pred = prediction[:, column]
                    tp = int((target & pred).sum())
                    fp = int((~target & pred).sum())
                    fn = int((target & ~pred).sum())
                    precision = _safe_div(tp, tp + fp)
                    recall = _safe_div(tp, tp + fn)
                    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
                    per_precision.append(precision)
                    per_recall.append(recall)
                    per_f1.append(f1)
                output.append(
                    {
                        "source_group": group,
                        "model": model,
                        "tiles": len(idx),
                        "macro_precision": float(np.mean(per_precision)),
                        "macro_recall": float(np.mean(per_recall)),
                        "macro_f1": float(np.mean(per_f1)),
                        "hamming_accuracy": float((prediction == truth).mean()),
                        "exact_match": float((prediction == truth).all(axis=1).mean()),
                        "predicted_positive_rate": float(prediction.mean()),
                        "expert_positive_rate": float(truth.mean()),
                    }
                )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"l2_prior_threshold_ablation_ok rows={len(output)} output={output_path}")


if __name__ == "__main__":
    main()
