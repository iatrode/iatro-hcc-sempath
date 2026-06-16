from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

from hcc_sempath.io.tile_package import TilePackageReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        default="experiments/06_attention_qc/configs/reviewed_attention_candidates.csv",
    )
    parser.add_argument(
        "--output",
        default="experiments/06_attention_qc/reports/reviewed_attention_candidates.png",
    )
    parser.add_argument("--columns", type=int, default=6)
    args = parser.parse_args()

    with Path(args.candidates).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    readers: dict[str, TilePackageReader] = {}
    images = []
    try:
        for row in rows:
            path = row["package_path"]
            if path not in readers:
                readers[path] = TilePackageReader(path)
            images.append(readers[path].read_image_at(int(row["row_idx"])).convert("RGB"))
    finally:
        for reader in readers.values():
            reader.close()

    columns = args.columns
    figure_rows = math.ceil(len(rows) / columns)
    fig, axes = plt.subplots(figure_rows, columns, figsize=(2.5 * columns, 2.8 * figure_rows), squeeze=False)
    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx >= len(rows):
            continue
        row = rows[idx]
        ax.imshow(images[idx])
        ax.set_title(
            f"{idx + 1:02d} | {row['source_group']} | {row['expert_l1']}",
            fontsize=8,
        )
    fig.suptitle(
        "Random blind-evaluation candidates for morphology-only selection",
        fontsize=12,
        weight="bold",
    )
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"attention_candidate_sheet_ok rows={len(rows)} output={output}")


if __name__ == "__main__":
    main()
