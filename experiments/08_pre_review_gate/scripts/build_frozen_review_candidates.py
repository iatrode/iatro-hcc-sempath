from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", default="experiments/08_pre_review_gate/results/merged_query_results_for_review.csv")
    parser.add_argument("--output-dir", default="experiments/08_pre_review_gate/results/frozen_review_candidates")
    parser.add_argument("--items", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rows = _read_csv(Path(args.retrieval))
    rng = random.Random(args.seed)
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row["model"], row["rank"])].append(row)
    bucket_keys = sorted(buckets)
    per_bucket = max(1, args.items // len(bucket_keys))
    selected = []
    for key in bucket_keys:
        values = buckets[key][:]
        rng.shuffle(values)
        selected.extend(values[:per_bucket])
    if len(selected) < args.items:
        selected_keys = {
            (row["model"], row["query_tile_id"], row["neighbor_tile_id"], row["rank"])
            for row in selected
        }
        remaining = [
            row for row in rows
            if (row["model"], row["query_tile_id"], row["neighbor_tile_id"], row["rank"]) not in selected_keys
        ]
        rng.shuffle(remaining)
        selected.extend(remaining[: args.items - len(selected)])
    selected = selected[: args.items]
    rng.shuffle(selected)

    review_rows = []
    key_rows = []
    for idx, row in enumerate(selected, start=1):
        review_id = f"PRG-{idx:05d}"
        review_rows.append({
            "review_id": review_id,
            "query_tile_id": row["query_tile_id"],
            "neighbor_tile_id": row["neighbor_tile_id"],
            "rank": row["rank"],
            "morphology_relevance_score": "",
            "failure_reason": "",
            "dominant_morphology_note": "",
        })
        key_rows.append({"review_id": review_id, **row})

    out = Path(args.output_dir)
    _write_csv(
        out / "blinded_review_items.csv",
        review_rows,
        ["review_id", "query_tile_id", "neighbor_tile_id", "rank", "morphology_relevance_score", "failure_reason", "dominant_morphology_note"],
    )
    _write_csv(out / "hidden_answer_key.csv", key_rows, list(key_rows[0]))

    counts = defaultdict(int)
    for row in key_rows:
        counts[(row["model"], row["rank"])] += 1
    balance_rows = [
        {"model": model, "rank": rank, "items": value}
        for (model, rank), value in sorted(counts.items())
    ]
    _write_csv(out / "candidate_balance.csv", balance_rows, ["model", "rank", "items"])

    html_rows = "\n".join(
        f"<tr><td>{row['review_id']}</td><td>{row['query_tile_id']}</td><td>{row['neighbor_tile_id']}</td><td>{row['rank']}</td></tr>"
        for row in review_rows
    )
    (out / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Pre-review Candidates</title>"
        "<table><thead><tr><th>review_id</th><th>query_tile_id</th><th>neighbor_tile_id</th><th>rank</th></tr></thead>"
        f"<tbody>{html_rows}</tbody></table>",
        encoding="utf-8",
    )
    print(f"frozen_review_candidates_ok items={len(review_rows)} output={out}")


if __name__ == "__main__":
    main()
