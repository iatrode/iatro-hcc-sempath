from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "full": "#0072B2",
    "plurality": "#4D4D4D",
    "teacher": "#A6A6A6",
    "prototype": "#009E73",
    "multiteacher": "#E69F00",
    "adjudication": "#CC79A7",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8, zorder=0)


def _plot_alignment_and_conflict(tables: Path, figures: Path) -> None:
    # 1. Load data
    teacher_rows = _read_csv(tables / "teacher_vs_expert_l1_metrics.csv")
    plurality_rows = _read_csv(tables / "teacher_plurality_vs_expert_l1_metrics.csv")
    model_rows = _read_csv(tables / "pamtd_ablation_vs_expert_l1_metrics.csv")
    quartile_rows = _read_csv(tables / "conflict_quartile_l1_metrics.csv")
    topn_rows = _read_csv(tables / "high_conflict_topn_sensitivity.csv")

    display = {
        "gigapath": "GigaPath",
        "h_optimus_1": "H-optimus-1",
        "uni2_h": "UNI2-h",
        "virchow2": "Virchow2",
        "teacher_plurality": "Teacher plurality",
        "pred_full": "HCC-SemPath",
    }
    groups = ("random500", "top500")
    records: dict[tuple[str, str], dict[str, str]] = {}
    for row in teacher_rows:
        records[(row["source_group"], row["teacher"])] = row
    for row in plurality_rows:
        records[(row["source_group"], row["baseline"])] = row
    for row in model_rows:
        if row["model"] == "pred_full":
            records[(row["source_group"], row["model"])] = row

    order = ("gigapath", "h_optimus_1", "uni2_h", "virchow2", "teacher_plurality", "pred_full")

    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.2), gridspec_kw={'hspace': 0.35, 'wspace': 0.28})

    # --- Top Row: Expert Level-1 Alignment (A, B) ---
    for col_idx, (group, title, panel_lbl) in enumerate(zip(groups, ("Random queue (n=500)", "High-conflict queue (n=500)"), ("A", "B"))):
        ax = axes[0, col_idx]
        y = np.arange(len(order))
        values = np.asarray([float(records[(group, source)]["accuracy"]) for source in order])
        lows = np.asarray([float(records[(group, source)]["accuracy_ci_low"]) for source in order])
        highs = np.asarray([float(records[(group, source)]["accuracy_ci_high"]) for source in order])
        colors = [
            COLORS["full"] if source == "pred_full"
            else COLORS["plurality"] if source == "teacher_plurality"
            else COLORS["teacher"]
            for source in order
        ]
        for value, low, high, ypos, color in zip(values, lows, highs, y, colors):
            ax.errorbar(
                value,
                ypos,
                xerr=np.asarray([[value - low], [high - value]]),
                fmt="o",
                color=color,
                markerfacecolor=color,
                markeredgecolor="white",
                markersize=8,
                elinewidth=2.6,
                capsize=5,
                capthick=2.2,
                zorder=3,
            )
            if high > 0.82:
                label_x = low - 0.018
                horizontal_alignment = "right"
            else:
                label_x = high + 0.018
                horizontal_alignment = "left"
            ax.text(
                label_x,
                ypos,
                f"{value:.3f} [{low:.3f}, {high:.3f}]",
                va="center",
                ha=horizontal_alignment,
                fontsize=8,
            )
        ax.set_title(f"({panel_lbl}) {title}", fontsize=11, weight="bold", loc="left")
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Accuracy against expert Level-1 label", fontsize=9)
        
        if col_idx == 0:
            ax.set_yticks(y, [display[source] for source in order])
        else:
            ax.set_yticks(y, [])
        ax.invert_yaxis()
        _style_axis(ax)

    # --- Bottom Row: Conflict Sensitivity (C, D) ---
    source_style = {
        "model_pred_full": ("HCC-SemPath", COLORS["full"], "o"),
        "teacher_plurality": ("Teacher plurality", COLORS["plurality"], "s"),
        "teacher_uni2_h": ("Best individual teacher", "#A6A6A6", "^"),
    }

    # Quartiles (C)
    ax_quartile = axes[1, 0]
    for source, (label, color, marker) in source_style.items():
        selected = [row for row in quartile_rows if row["source"] == source]
        selected.sort(key=lambda row: row["conflict_bin"])
        ax_quartile.plot(
            [row["conflict_bin"] for row in selected],
            [float(row["accuracy"]) for row in selected],
            color=color,
            marker=marker,
            linewidth=2,
            markersize=6,
            label=label,
        )
    ax_quartile.set_title("(C) Conflict quartiles", fontsize=11, weight="bold", loc="left")
    ax_quartile.set_xlabel("Increasing teacher disagreement", fontsize=9)
    ax_quartile.set_ylabel("Expert Level-1 accuracy", fontsize=9)
    ax_quartile.set_ylim(0, 1.02)
    ax_quartile.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax_quartile.spines["top"].set_visible(False)
    ax_quartile.spines["right"].set_visible(False)

    # Subsets (D)
    ax_subsets = axes[1, 1]
    for source, (label, color, marker) in source_style.items():
        selected = [row for row in topn_rows if row["source"] == source]
        selected.sort(key=lambda row: int(row["topn"]))
        ax_subsets.plot(
            [int(row["topn"]) for row in selected],
            [float(row["accuracy"]) for row in selected],
            color=color,
            marker=marker,
            linewidth=2,
            markersize=6,
            label=label,
        )
    ax_subsets.set_title("(D) Most-conflicted subsets", fontsize=11, weight="bold", loc="left")
    ax_subsets.set_xlabel("Number of highest-conflict reviewed tiles", fontsize=9)
    ax_subsets.set_ylim(0, 1.02)
    ax_subsets.set_xticks([100, 250, 500])
    ax_subsets.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax_subsets.spines["top"].set_visible(False)
    ax_subsets.spines["right"].set_visible(False)
    ax_subsets.legend(frameon=False, fontsize=8.5, loc="lower right")

    fig.suptitle("Expert alignment and robustness across teacher-conflict severity", fontsize=13, weight="bold", y=0.98)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.95])
    _save(fig, figures / "alignment_conflict")


def _plot_ablation_effects(tables: Path, figures: Path) -> None:
    rows = _read_csv(tables / "paired_ablation_l1_comparisons.csv")
    rows = [row for row in rows if row["source_group"] in {"random500", "top500"}]
    contrast_order = (
        ("pred_a1", "pred_a3", "A1--A3: multi-teacher, no prototypes"),
        ("pred_a0", "pred_a5", "A0--A5: dynamic prototype refresh"),
        ("pred_a2", "pred_a1", "A2--A1: prototype supervision"),
        ("pred_a4", "pred_a3", "A4--A3: prototype replication"),
        ("pred_a6", "pred_a2", "A0'--A2: complete filtering"),
    )
    lookup = {
        (row["source_group"], row["reference"], row["comparison"]): row
        for row in rows
    }

    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    base_y = np.arange(len(contrast_order))[::-1] * 1.25
    offsets = {"random500": 0.18, "top500": -0.18}
    group_labels = {"random500": "Random500", "top500": "High-conflict500"}
    group_markers = {"random500": "o", "top500": "s"}
    group_colors = {"random500": "#666666", "top500": "#0072B2"}

    for group in ("random500", "top500"):
        values = []
        lows = []
        highs = []
        ys = []
        for ypos, (reference, comparison, _) in zip(base_y, contrast_order):
            row = lookup[(group, reference, comparison)]
            values.append(float(row["accuracy_delta"]))
            lows.append(float(row["delta_ci_low"]))
            highs.append(float(row["delta_ci_high"]))
            ys.append(ypos + offsets[group])
        values_array = np.asarray(values)
        ax.errorbar(
            values_array,
            ys,
            xerr=np.vstack((values_array - np.asarray(lows), np.asarray(highs) - values_array)),
            fmt=group_markers[group],
            color=group_colors[group],
            markerfacecolor=group_colors[group],
            markeredgecolor="white",
            markersize=6,
            elinewidth=1.8,
            capsize=3,
            label=group_labels[group],
            zorder=3,
        )
        for value, low, high, ypos in zip(values, lows, highs, ys):
            if value >= 0.12:
                text_x = low - 0.006
                ha = "right"
            else:
                text_x = high + 0.006
                ha = "left"
            ax.text(
                text_x,
                ypos,
                f"{value:+.3f} [{low:+.3f}, {high:+.3f}]",
                va="center",
                ha=ha,
                fontsize=7.5,
                color=group_colors[group],
            )

    ax.axvline(0, color="#222222", linewidth=1)
    ax.set_yticks(base_y, [label for _, _, label in contrast_order])
    ax.set_xlabel("Paired accuracy difference (reference minus comparator)", fontsize=9)
    ax.set_title("Matched 10%-scale orthogonal mechanism contrasts", fontsize=12, weight="bold")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_xlim(-0.03, 0.205)
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, figures / "paired_ablation_effects")


# _plot_conflict_sensitivity was merged into _plot_alignment_and_conflict


def _plot_high_conflict_confusion(tables: Path, figures: Path) -> None:
    rows = _read_csv(tables / "l1_confusion_matrices.csv")
    labels = (
        "Background-liver",
        "Degenerative-material",
        "HCC-tumor",
        "Inflammatory-stromal",
    )
    display = ("Background", "Degenerative", "HCC", "Inflammatory /\nstromal")
    sources = (
        ("baseline_teacher_plurality", "Teacher plurality"),
        ("model_pred_full", "HCC-SemPath"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 4.2), sharex=True, sharey=True)
    for ax, (source, title) in zip(axes, sources):
        matrix = np.zeros((len(labels), len(labels)), dtype=float)
        for row in rows:
            if row["source_group"] != "top500" or row["source"] != source:
                continue
            matrix[labels.index(row["expert_l1"]), labels.index(row["predicted_l1"])] += int(row["tiles"])
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for i in range(len(labels)):
            for j in range(len(labels)):
                value = normalized[i, j]
                ax.text(
                    j,
                    i,
                    f"{int(matrix[i, j])}\n{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value > 0.55 else "#222222",
                )
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xticks(range(len(labels)), display, rotation=35, ha="right")
        ax.set_yticks(range(len(labels)), display)
        ax.set_xlabel("Predicted label", fontsize=9)
    axes[0].set_ylabel("Expert-adjudicated label", fontsize=9)
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.03, label="Row-normalized fraction")
    fig.suptitle("High-conflict error structure (n=500)", fontsize=12, weight="bold")
    fig.subplots_adjust(left=0.12, right=0.9, bottom=0.2, top=0.82, wspace=0.18)
    _save(fig, figures / "high_conflict_confusion")


def _plot_representation_audit(experiment_dir: Path, figures: Path) -> None:
    audit_tables = experiment_dir.parent / "09_representation_audit" / "tables"
    overlap_rows = _read_csv(audit_tables / "cross_model_overlap_summary.csv")
    agreement_rows = _read_csv(audit_tables / "model_teacher_agreement_summary.csv")
    names = {
        "z_hcc": "HCC-SemPath",
        "gigapath": "GigaPath",
        "h_optimus_1": "H-optimus-1",
        "uni2_h": "UNI2-h",
        "virchow2": "Virchow2",
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1))
    teachers = [row["comparison_model"] for row in overlap_rows]
    overlap = [float(row["mean_overlap_at_10"]) for row in overlap_rows]
    unique = [float(row["mean_z_hcc_unique_at_10"]) for row in overlap_rows]
    y = np.arange(len(teachers))
    axes[0].barh(y, unique, color=COLORS["full"], label="Unique to HCC-SemPath")
    axes[0].barh(y, overlap, left=unique, color="#BDBDBD", label="Shared with teacher")
    axes[0].set_yticks(y, [names[name] for name in teachers])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 10)
    axes[0].set_xlabel("Neighbors among top 10", fontsize=9)
    axes[0].set_title("Nearest-neighbor non-copy audit", fontsize=11, weight="bold")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    models = [row["model"] for row in agreement_rows]
    x = np.arange(len(models))
    cosine = [float(row["teacher_cosine_mean"]) for row in agreement_rows]
    prototype = [float(row["prototype_primary_match_mean"]) / 4.0 for row in agreement_rows]
    width = 0.36
    axes[1].bar(x - width / 2, cosine, width, color="#666666", label="Teacher cosine")
    axes[1].bar(x + width / 2, prototype, width, color=COLORS["prototype"], label="Prototype match / 4")
    axes[1].set_xticks(x, [names[name] for name in models], rotation=35, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Retained semantic structure", fontsize=11, weight="bold")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    axes[1].grid(axis="y", color="#E6E6E6", linewidth=0.8)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.suptitle("Representation is semantically aligned without copying one teacher", fontsize=12, weight="bold")
    fig.tight_layout()
    _save(fig, figures / "representation_audit")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        default="experiments/10_teacher_disagreement_review",
    )
    args = parser.parse_args()
    experiment_dir = Path(args.experiment_dir)
    tables = experiment_dir / "tables"
    figures = experiment_dir / "reports" / "figures"
    _plot_alignment_and_conflict(tables, figures)
    _plot_ablation_effects(tables, figures)
    _plot_high_conflict_confusion(tables, figures)
    _plot_representation_audit(experiment_dir, figures)
    print(f"manuscript_figures_ok output={figures}")

    # Automatically copy files to manuscript/figures/ with correct figure numbering
    repo_root = Path(__file__).resolve().parents[3]
    ms_figures = repo_root / "manuscript" / "figures"
    if ms_figures.exists():
        import shutil
        shutil.copyfile(figures / "alignment_conflict.pdf", ms_figures / "figure1_alignment_conflict.pdf")
        shutil.copyfile(figures / "paired_ablation_effects.pdf", ms_figures / "figure2_paired_ablation_effects.pdf")
        shutil.copyfile(figures / "representation_audit.pdf", ms_figures / "figure4_representation_audit.pdf")
        print(f"Automatically copied figures to {ms_figures}")


if __name__ == "__main__":
    main()
