from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


KEYS = [
    "teacher_alignment_score",
    "scientific_score",
    "gigapath_feature_cosine",
    "h_optimus_1_feature_cosine",
    "uni2_h_feature_cosine",
    "virchow2_feature_cosine",
    "prototype_bank_zhcc_level1_accuracy",
    "prototype_bank_zhcc_level2_macro_auc",
    "prototype_bank_zhcc_prototype_topk_precision",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    rows = []
    metrics_by_key = {}
    for checkpoint in ("epoch61", "epoch100"):
        for split in ("val", "exval"):
            path = result_dir / f"{checkpoint}_{split}.json"
            with path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            metrics_by_key[(checkpoint, split)] = metrics
            for key in KEYS:
                rows.append({
                    "checkpoint": checkpoint,
                    "split": split,
                    "metric": key,
                    "value": metrics.get(key, ""),
                })
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["checkpoint", "split", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Checkpoint Comparison",
        "",
        "Protocol: same fixed-seed local sampled MPS evaluation as `00_local_eval`.",
        "",
        "| split | metric | epoch61 | epoch100 | delta |",
        "|---|---|---:|---:|---:|",
    ]
    deltas = []
    for split in ("val", "exval"):
        for key in KEYS:
            a = metrics_by_key[("epoch61", split)].get(key)
            b = metrics_by_key[("epoch100", split)].get(key)
            if a is None or b is None:
                continue
            delta = float(b) - float(a)
            if key in {"teacher_alignment_score", "scientific_score"}:
                deltas.append(delta)
            lines.append(f"| {split} | `{key}` | {float(a):.6f} | {float(b):.6f} | {delta:.6f} |")
    recommendation = "epoch100" if sum(deltas) >= 0 else "epoch61"
    lines.extend(["", f"Manuscript default under this sampled local protocol: `{recommendation}`."])
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"checkpoint_summary_ok csv={csv_path} report={report}")


if __name__ == "__main__":
    main()
