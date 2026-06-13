from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-items", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    reviews = {row["review_id"]: row for row in _read_csv(Path(args.review_items))}
    key_rows = _read_csv(Path(args.key))
    by_model = defaultdict(list)
    missing = 0
    for row in key_rows:
        score = reviews[row["review_id"]].get("morphology_relevance_score", "")
        if score == "":
            missing += 1
            continue
        by_model[row["model"]].append(float(score))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    for model, scores in sorted(by_model.items()):
        arr = np.asarray(scores, dtype=float)
        metrics_rows.append({
            "model": model,
            "reviewed_items": len(scores),
            "mean_relevance": float(arr.mean()),
            "precision_at_score_ge_2": float((arr >= 2).mean()),
        })
    with (out / "review_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "reviewed_items", "mean_relevance", "precision_at_score_ge_2"])
        writer.writeheader()
        writer.writerows(metrics_rows)
    with (out / "bootstrap_ci.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "metric", "lower", "upper"])
        writer.writeheader()
    report = Path("experiments/05_review_analysis/reports/review_analysis_summary.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    if missing:
        text = (
            "# Review Analysis Summary\n\n"
            f"Review scores are not yet populated for {missing} items. "
            "`review_metrics.csv` contains only completed scores.\n"
        )
    else:
        text = "# Review Analysis Summary\n\nAll available review scores were analyzed.\n"
    report.write_text(text, encoding="utf-8")
    print(f"review_analysis_ok scored={sum(len(v) for v in by_model.values())} missing={missing} output={out}")


if __name__ == "__main__":
    main()
