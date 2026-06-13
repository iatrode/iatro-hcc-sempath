from __future__ import annotations

import argparse
import json
from pathlib import Path


KEYS = [
    "teacher_alignment_score",
    "scientific_score",
    "gigapath_feature_cosine",
    "h_optimus_1_feature_cosine",
    "uni2_h_feature_cosine",
    "virchow2_feature_cosine",
    "gigapath_retrieval_overlap",
    "h_optimus_1_retrieval_overlap",
    "uni2_h_retrieval_overlap",
    "virchow2_retrieval_overlap",
    "prototype_bank_zhcc_level1_accuracy",
    "prototype_bank_zhcc_level2_macro_auc",
    "prototype_bank_zhcc_prototype_topk_precision",
]


def _load_input(item: str) -> tuple[str, dict]:
    label, path = item.split("=", 1)
    with Path(path).open("r", encoding="utf-8") as handle:
        return label, json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = [_load_input(item) for item in args.inputs]
    lines = [
        "# Local Evaluation Summary",
        "",
        "Protocol: fixed-seed local sampled evaluation on MPS using `max_eval_batches=4`, `batch_size=16`, and split tile fraction `0.001`.",
        "",
        "| metric | " + " | ".join(label for label, _ in rows) + " |",
        "|---|" + "|".join("---:" for _ in rows) + "|",
    ]
    for key in KEYS:
        values = []
        for _, metrics in rows:
            value = metrics.get(key)
            values.append("" if value is None else f"{float(value):.6f}")
        lines.append("| `" + key + "` | " + " | ".join(values) + " |")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary_ok output={out}")


if __name__ == "__main__":
    main()
