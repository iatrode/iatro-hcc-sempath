from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import umap


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/select_l1_prototype_bank.py"
OUT = ROOT / "annotations/analysis/l1_fixed_prototype_bank"
WEIGHTS = (0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)

spec = importlib.util.spec_from_file_location("select_l1_prototype_bank_refresh", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

payload = json.loads(module.ANNOTATION_PATH.read_text(encoding="utf-8"))
keyed = sorted(
    (
        (key, dict(value))
        for key, value in payload["annotations"].items()
        if value.get("tile_id") and value.get("l1")
    ),
    key=lambda item: str(item[1]["tile_id"]),
)
keys = [key for key, _ in keyed]
rows = [row for _, row in keyed]
labels = [str(value) for value in payload["l1_prototypes"]]
features, _ = module._load_features(rows)
row_labels = np.asarray([str(row["l1"]) for row in rows])
margin = module._global_class_margins(rows, features, labels)

fused = np.concatenate([features[teacher] for teacher in module.TEACHERS], axis=1)
fused = module._normalized(fused)
pca = PCA(n_components=50, random_state=13)
pca_all = pca.fit_transform(fused).astype(np.float32)

class_data = {}
for label in labels:
    indices = np.flatnonzero(row_labels == label).tolist()
    teacher_sims = {}
    combined = np.zeros((len(indices), len(indices)), dtype=np.float32)
    for teacher in module.TEACHERS:
        matrix = features[teacher][indices]
        similarity = np.clip((matrix @ matrix.T + 1.0) * 0.5, 0.0, 1.0)
        teacher_sims[teacher] = similarity
        combined += similarity / len(module.TEACHERS)
    class_data[label] = (indices, teacher_sims, combined, module._rank_margin(indices, margin))


def select(weight: float):
    module.SEPARATION_WEIGHT = weight
    chosen = []
    local_orders = {}
    for label in labels:
        indices, _, combined, margin_rank = class_data[label]
        order, _ = module._facility_order(
            combined, module.TARGET_PER_CLASS, margin_rank=margin_rank
        )
        local_orders[label] = order
        chosen.extend(indices[index] for index in order)
    return np.asarray(chosen, dtype=np.int64), local_orders


def silhouette_metrics(chosen: np.ndarray):
    x = pca_all[chosen]
    y = row_labels[chosen]
    tumor = np.asarray([value.startswith("HCC-tumor-") for value in y])
    tumor_background = tumor | (y == "Background-liver")
    well_background = (y == "HCC-tumor-well-differentiated") | (
        y == "Background-liver"
    )
    return {
        "all_six": float(silhouette_score(x, y)),
        "tumor_grades": float(silhouette_score(x[tumor], y[tumor])),
        "tumor_plus_background": float(
            silhouette_score(x[tumor_background], y[tumor_background])
        ),
        "well_vs_background": float(
            silhouette_score(x[well_background], y[well_background])
        ),
    }


def final_coverage_and_delta(orders):
    coverage = []
    passes = []
    checkpoints = list(module.CHECKPOINTS)
    for label in labels:
        _, teacher_sims, _, _ = class_data[label]
        order = orders[label]
        for teacher in module.TEACHERS:
            values = [
                module._leave_one_out_novelty(teacher_sims[teacher], order, count)
                for count in checkpoints
            ]
            coverage.append(1.0 - values[-1])
            marginal = [
                (values[index - 1] - values[index])
                / (checkpoints[index] - checkpoints[index - 1])
                for index in range(1, len(checkpoints))
            ]
            early = marginal[0]
            ratios = [value / early if early > 0 else 0.0 for value in marginal[-3:]]
            passes.append(len(ratios) == 3 and all(value <= 0.35 for value in ratios))
    return float(np.mean(coverage)), bool(all(passes))


trial_rows = []
selected_by_weight = {}
orders_by_weight = {}
for weight in WEIGHTS:
    chosen, orders = select(weight)
    selected_by_weight[weight] = chosen
    orders_by_weight[weight] = orders
    metrics = silhouette_metrics(chosen)
    coverage, delta_pass = final_coverage_and_delta(orders)
    trial_rows.append(
        {
            "weight": weight,
            "delta_pass": delta_pass,
            "mean_coverage_at_400": coverage,
            "mean_global_margin": float(np.mean(margin[chosen])),
            **metrics,
        }
    )
    print(f"[figures] weight={weight:g} silhouette={metrics['all_six']:.5f}", flush=True)

baseline_coverage = trial_rows[0]["mean_coverage_at_400"]
for trial in trial_rows:
    trial["coverage_within_10pct_of_baseline"] = bool(
        trial["mean_coverage_at_400"] >= 0.9 * baseline_coverage
    )
    trial["eligible"] = bool(
        trial["delta_pass"] and trial["coverage_within_10pct_of_baseline"]
    )

selected = selected_by_weight[32.0]
selected_metrics = next(row for row in trial_rows if row["weight"] == 32.0)
baseline_metrics = trial_rows[0]

colors = {
    "HCC-tumor-well-differentiated": "#2ca02c",
    "HCC-tumor-moderately-differentiated": "#ff7f0e",
    "HCC-tumor-poorly-differentiated": "#d62728",
    "Background-liver": "#7f7f7f",
    "Inflammatory-stromal": "#1f77b4",
    "Degenerative-material": "#9467bd",
}


def scatter_panels(coords: np.ndarray, output: Path, method: str):
    y = row_labels[selected]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    panels = [
        (labels, "Six L1 prototype classes"),
        (
            [
                "Background-liver",
                "HCC-tumor-well-differentiated",
                "HCC-tumor-moderately-differentiated",
                "HCC-tumor-poorly-differentiated",
            ],
            "Differentiation continuum and background",
        ),
    ]
    for axis, (shown, title) in zip(axes, panels):
        for label in shown:
            mask = y == label
            axis.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=9,
                alpha=0.58,
                linewidths=0,
                color=colors[label],
                label=label,
            )
        axis.set_title(title)
        axis.set_xlabel(f"{method}-1")
        axis.set_ylabel(f"{method}-2")
        axis.grid(alpha=0.14)
        axis.legend(frameon=False, fontsize=8, markerscale=1.8)
    fig.suptitle(f"Fixed L1 prototype bank in fused four-teacher {method} space (N=2,400)")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


selected_pca = pca_all[selected, :2]
scatter_panels(selected_pca, OUT / "four_teacher_fused_pca.png", "PCA")

selected_pca50 = pca_all[selected]
reducer = umap.UMAP(
    n_neighbors=30,
    min_dist=0.25,
    metric="cosine",
    random_state=13,
)
selected_umap = reducer.fit_transform(selected_pca50).astype(np.float32)
scatter_panels(selected_umap, OUT / "four_teacher_fused_umap.png", "UMAP")

pca_summary = {
    "samples": 2400,
    "class_count": 6,
    "samples_per_class": 400,
    "teachers": list(module.TEACHERS),
    "fusion": "per-teacher L2 normalization, equal-norm concatenation, final L2 normalization",
    "pca_fit_population": "complete adjudicated L1 pool (N=3672)",
    "selection": "sample-level global class-margin-weighted greedy facility coverage",
    "global_margin_weight": 32.0,
    "coverage_only_baseline": {
        "silhouette_50d_all_six": baseline_metrics["all_six"],
        "silhouette_50d_tumor_grades": baseline_metrics["tumor_grades"],
        "silhouette_50d_tumor_grades_plus_background": baseline_metrics[
            "tumor_plus_background"
        ],
        "silhouette_50d_well_vs_background": baseline_metrics["well_vs_background"],
    },
    "selected_bank": {
        "silhouette_50d_all_six": selected_metrics["all_six"],
        "silhouette_50d_tumor_grades": selected_metrics["tumor_grades"],
        "silhouette_50d_tumor_grades_plus_background": selected_metrics[
            "tumor_plus_background"
        ],
        "silhouette_50d_well_vs_background": selected_metrics["well_vs_background"],
    },
    "all_teacher_delta_rules_pass": selected_metrics["delta_pass"],
    "coverage_within_10pct_of_baseline": selected_metrics[
        "coverage_within_10pct_of_baseline"
    ],
}
(OUT / "pca_summary.json").write_text(
    json.dumps(pca_summary, ensure_ascii=False, indent=2), encoding="utf-8"
)

comparison = {
    "selection_rule": "sample-level global class-margin-weighted greedy facility coverage",
    "selected_weight": 32.0,
    "weight_choice": "first practical plateau; higher weights add negligible separation",
    "baseline_metrics": baseline_metrics,
    "selected_metrics": selected_metrics,
    "trials": trial_rows,
}
(OUT / "separation_comparison.json").write_text(
    json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
)

umap_summary = {
    "samples": 2400,
    "input": "50-dimensional PCA of equally fused four-teacher features; PCA fit on complete adjudicated L1 pool",
    "n_neighbors": 30,
    "min_dist": 0.25,
    "metric": "cosine",
    "random_state": 13,
    "interpretation": "visualization only; quantitative separation is reported in PCA space",
}
(OUT / "umap_summary.json").write_text(
    json.dumps(umap_summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
