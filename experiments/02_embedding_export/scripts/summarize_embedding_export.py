from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _count_metadata(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    lines = ["# Embedding Export Summary", "", "| split | tiles | student shape | provenance |", "|---|---:|---|---|"]
    for split in ("val", "exval"):
        arr = np.load(result_dir / f"student_embeddings_{split}.npz")["embedding_norm"]
        manifest = json.loads((result_dir / f"export_manifest_{split}.json").read_text(encoding="utf-8"))
        lines.append(
            f"| {split} | {_count_metadata(result_dir / f'tile_metadata_{split}.csv')} | {tuple(arr.shape)} | "
            f"`export_manifest_{split}.json` |"
        )
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"embedding_summary_ok report={report}")


if __name__ == "__main__":
    main()
