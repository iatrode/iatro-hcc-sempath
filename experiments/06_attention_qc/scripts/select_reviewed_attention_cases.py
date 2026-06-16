from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


MAIN_CLASSES = ("Background-liver", "HCC-tumor", "Inflammatory-stromal")


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
        "--output",
        default="experiments/06_attention_qc/configs/reviewed_attention_candidates.csv",
    )
    parser.add_argument("--per-stratum", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260615)
    args = parser.parse_args()

    review_rows = _read(Path(args.review))
    prediction_by_id = {row["review_id"]: row for row in _read(Path(args.predictions))}
    rows = [{**row, **prediction_by_id[row["review_id"]]} for row in review_rows]
    rng = random.Random(args.seed)
    selected: list[tuple[str, dict[str, str]]] = []
    for source_group in ("random500", "top500"):
        for label in MAIN_CLASSES:
            eligible = [
                row
                for row in rows
                if row["source_group"] == source_group and row["l1"] == label
            ]
            rng.shuffle(eligible)
            take = eligible[: min(args.per_stratum, len(eligible))]
            selected.extend((f"{source_group}: {label}", row) for row in take)

    fieldnames = (
        "stratum",
        "review_id",
        "tile_id",
        "package_idx",
        "package_path",
        "row_idx",
        "source_group",
        "disagreement_score",
        "expert_l1",
        "teacher_plurality",
        "full_prediction",
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for stratum, row in selected:
            writer.writerow({
                "stratum": stratum,
                "review_id": row["review_id"],
                "tile_id": row["tile_id"],
                "package_idx": row["package_idx"],
                "package_path": row["package_path"],
                "row_idx": row["row_idx"],
                "source_group": row["source_group"],
                "disagreement_score": row["disagreement_score"],
                "expert_l1": row["l1"],
                "teacher_plurality": row["plurality_l1_name"],
                "full_prediction": row["pred_full"],
            })
    print(
        f"reviewed_attention_candidates_ok rows={len(selected)} seed={args.seed} "
        f"output={output}"
    )


if __name__ == "__main__":
    main()
