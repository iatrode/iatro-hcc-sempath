from __future__ import annotations

import argparse
import csv
import random
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
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    rows = _read_csv(Path(args.retrieval))
    rng = random.Random(args.seed)
    keyed = []
    seen = set()
    for row in rows:
        pair_key = (row["query_tile_id"], row["neighbor_tile_id"], row["model"], row["rank"])
        if pair_key in seen:
            continue
        seen.add(pair_key)
        keyed.append(row)
    rng.shuffle(keyed)
    review_rows = []
    key_rows = []
    for idx, row in enumerate(keyed, start=1):
        review_id = f"BR-{idx:05d}"
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
    _write_csv(out / "blinded_review_key.csv", key_rows, list(key_rows[0]))
    package = out / "reviewer_package"
    package.mkdir(parents=True, exist_ok=True)
    html_rows = "\n".join(
        f"<tr><td>{r['review_id']}</td><td>{r['query_tile_id']}</td><td>{r['neighbor_tile_id']}</td><td>{r['rank']}</td></tr>"
        for r in review_rows
    )
    (package / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Blinded Review</title>"
        "<table><thead><tr><th>review_id</th><th>query_tile_id</th><th>neighbor_tile_id</th><th>rank</th></tr></thead>"
        f"<tbody>{html_rows}</tbody></table>",
        encoding="utf-8",
    )
    report = Path("experiments/04_blinded_review_package/reports/review_package_manifest.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Blinded Review Package Manifest\n\n"
        f"Review items: {len(review_rows)}.\n\n"
        "Reviewer-facing files: `blinded_review_items.csv`, `reviewer_package/index.html`.\n\n"
        "Hidden key: `blinded_review_key.csv`.\n",
        encoding="utf-8",
    )
    print(f"blinded_review_package_ok items={len(review_rows)} output={out}")


if __name__ == "__main__":
    main()
