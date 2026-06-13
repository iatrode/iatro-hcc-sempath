from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    with (result_dir / "tile_attention_rows.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    saliency_count = len(list(result_dir.glob("*.saliency.png")))
    attention_count = len(list(result_dir.glob("*.attention.png")))
    lines = [
        "# Attention QC Summary",
        "",
        f"Sampled tiles: {len(rows)}.",
        f"Saliency overlays: {saliency_count}.",
        f"Attention rollout overlays: {attention_count}.",
        "",
        "Primary sheet: `tile_attention_sheet.png`.",
    ]
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"attention_summary_ok report={report}")


if __name__ == "__main__":
    main()
