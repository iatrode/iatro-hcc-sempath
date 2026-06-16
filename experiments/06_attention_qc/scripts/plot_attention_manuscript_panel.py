from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from hcc_sempath.io.tile_package import TilePackageReader


PANEL_REVIEW_IDS = ("TD-0864", "TD-0477", "TD-0459", "TD-0363")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default="experiments/06_attention_qc/configs/reviewed_attention_cases.csv",
    )
    parser.add_argument(
        "--results",
        default="experiments/06_attention_qc/results",
    )
    parser.add_argument(
        "--output",
        default="experiments/06_attention_qc/reports/attention_manuscript_panel",
    )
    args = parser.parse_args()

    with Path(args.cases).open("r", newline="", encoding="utf-8") as handle:
        by_id = {row["review_id"]: row for row in csv.DictReader(handle)}
    rows = [by_id[review_id] for review_id in PANEL_REVIEW_IDS]
    result_dir = Path(args.results)
    with (result_dir / "tile_attention_scores.csv").open("r", newline="", encoding="utf-8") as handle:
        score_by_tile = {row["tile_id"]: row for row in csv.DictReader(handle)}
    readers: dict[str, TilePackageReader] = {}
    fig, axes = plt.subplots(len(rows), 2, figsize=(5.6, 9.4), squeeze=False)
    try:
        for idx, row in enumerate(rows):
            path = row["package_path"]
            if path not in readers:
                readers[path] = TilePackageReader(path)
            tile = readers[path].read_image_at(int(row["row_idx"])).convert("RGB")
            occlusion = Image.open(result_dir / f"{row['tile_id']}.occlusion.png").convert("RGB")
            for ax, image in zip(axes[idx], (tile, occlusion)):
                ax.imshow(image)
                ax.axis("off")
            axes[idx, 0].text(
                0.02,
                0.98,
                (
                    f"Predicted: {row['full_prediction']}\n"
                    f"Margin: {float(score_by_tile[row['tile_id']]['zhcc_student_l1_margin']):.3f}"
                ),
                transform=axes[idx, 0].transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.62, "edgecolor": "none", "pad": 2},
            )
        for ax, title in zip(axes[0], ("H&E tile", "Decision-margin occlusion")):
            ax.set_title(title, fontsize=10, weight="bold")
    finally:
        for reader in readers.values():
            reader.close()

    fig.suptitle("Morphology-focused responses of the final HCC-SemPath model", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"attention_manuscript_panel_ok output={output}")


if __name__ == "__main__":
    main()
