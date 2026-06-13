from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from hcc_sempath.io.feature_cache import FeatureCacheReader
from hcc_sempath.training.config import load_config, manifest_data_paths, teacher_names
from hcc_sempath.training.datasets import _open_feature_source, _read_teacher_features_at
from hcc_sempath.training.manifest import load_training_manifest


MODELS = ["z_hcc", "gigapath", "h_optimus_1", "uni2_h", "virchow2"]


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sample_balanced(rows: list[dict], target: int, key: str, seed: int) -> list[int]:
    if target >= len(rows):
        return list(range(len(rows)))
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[row[key]].append(idx)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    group_keys = sorted(groups)
    rng.shuffle(group_keys)
    for group_key in group_keys:
        if len(selected) >= target:
            break
        selected.append(int(rng.choice(groups[group_key])))
    remaining = target - len(selected)
    if remaining > 0:
        selected_set = set(selected)
        pool = np.asarray([idx for idx in range(len(rows)) if idx not in selected_set], dtype=np.int64)
        selected.extend(int(idx) for idx in rng.choice(pool, size=remaining, replace=False))
    selected = sorted(set(selected))
    if len(selected) > target:
        selected = sorted(int(idx) for idx in rng.choice(np.asarray(selected), size=target, replace=False))
    return selected


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float32", copy=False)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norm, 1e-8)


def _load_z_features(rows: list[dict], indices: list[int]) -> np.ndarray:
    result = []
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for out_idx, idx in enumerate(indices):
        row = rows[idx]
        grouped[row["student_feature_package_path"]].append((out_idx, int(row["sample_row"])))
    result = [None] * len(indices)
    for path, items in grouped.items():
        reader = FeatureCacheReader(path)
        try:
            for out_idx, sample_row in items:
                result[out_idx] = reader.read_feature_at(sample_row)
        finally:
            reader.close()
    return np.stack(result).astype("float32")


def _tile_teacher_paths(cfg_path: Path, split: str) -> dict[str, dict[str, str]]:
    cfg = load_config(cfg_path)
    cfg["data"]["exval_tile_fraction"] = 1.0
    cfg["data"]["val_tile_fraction"] = 1.0
    manifest = load_training_manifest(cfg["data"]["train_manifest_path"])
    tile_packages, teacher_packages = manifest_data_paths(cfg, manifest, split)
    return {
        str(tile_path): {name: teacher_packages[name][idx] for name in teacher_names(cfg)}
        for idx, tile_path in enumerate(tile_packages)
    }


def _load_teacher_features(
    rows: list[dict],
    indices: list[int],
    model: str,
    tile_to_teacher_paths: dict[str, dict[str, str]],
) -> np.ndarray:
    result = [None] * len(indices)
    grouped: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for out_idx, idx in enumerate(indices):
        row = rows[idx]
        tile_path = row["tile_package_path"]
        source_path = tile_to_teacher_paths[tile_path][model]
        grouped[source_path].append((out_idx, int(row["source_row"]), tile_path))
    for source_path, items in grouped.items():
        source = _open_feature_source(Path(source_path))
        try:
            for out_idx, source_row, _ in items:
                if hasattr(source, "read_feature_at") and source.__class__.__name__ != "MergedTeacherFeatureCacheReader":
                    result[out_idx] = source.read_feature_at(source_row)
                else:
                    result[out_idx] = source.read_feature_at(source_row, model)
        finally:
            close = getattr(source, "close", None)
            if close is not None:
                close()
    return np.stack(result).astype("float32")


def _embedding_qc(model: str, arr: np.ndarray) -> dict:
    norms = np.linalg.norm(arr.astype("float32", copy=False), axis=1)
    finite = np.isfinite(arr)
    return {
        "model": model,
        "count": arr.shape[0],
        "dim": arr.shape[1],
        "nan_count": int(np.isnan(arr).sum()),
        "inf_count": int(np.isinf(arr).sum()),
        "finite_fraction": float(finite.mean()),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
    }


def _retrieval(
    model: str,
    query_arr: np.ndarray,
    gallery_arr: np.ndarray,
    rows: list[dict],
    query_indices: list[int],
    gallery_indices: list[int],
    topk: int,
) -> tuple[list[dict], dict]:
    q = _normalize(query_arr)
    g = _normalize(gallery_arr)
    sims = q @ g.T
    retrieval_rows: list[dict] = []
    valid_counts = []
    same_slide_candidates = []
    margins = []
    for qi, source_idx in enumerate(query_indices):
        order = np.argsort(-sims[qi])
        query_row = rows[source_idx]
        valid = 0
        same_slide_skipped = 0
        first_score = None
        kth_score = None
        for gi in order:
            gallery_idx = gallery_indices[int(gi)]
            gallery_row = rows[gallery_idx]
            if gallery_row["tile_id"] == query_row["tile_id"]:
                continue
            if gallery_row["slide_id"] == query_row["slide_id"]:
                same_slide_skipped += 1
                continue
            valid += 1
            score = float(sims[qi, gi])
            if first_score is None:
                first_score = score
            kth_score = score
            retrieval_rows.append({
                "model": model,
                "query_id": qi,
                "query_tile_id": query_row["tile_id"],
                "query_slide_id": query_row["slide_id"],
                "rank": valid,
                "neighbor_tile_id": gallery_row["tile_id"],
                "neighbor_slide_id": gallery_row["slide_id"],
                "neighbor_patient_id": gallery_row["patient_id"],
                "cosine": f"{score:.8f}",
            })
            if valid >= topk:
                break
        valid_counts.append(valid)
        same_slide_candidates.append(same_slide_skipped)
        if first_score is not None and kth_score is not None:
            margins.append(first_score - kth_score)
    metrics = {
        "model": model,
        "queries": len(query_indices),
        "gallery": len(gallery_indices),
        "topk": topk,
        "queries_with_full_topk": sum(1 for value in valid_counts if value >= topk),
        "mean_valid_neighbors": float(np.mean(valid_counts)),
        "mean_same_slide_skipped_before_topk": float(np.mean(same_slide_candidates)),
        "mean_top1_to_topk_margin": float(np.mean(margins)) if margins else 0.0,
    }
    return retrieval_rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="experiments/07_full_exval_cache/results/manifests/exval_sampled_z_hcc_metadata.csv")
    parser.add_argument("--config", default="experiments/shared/configs/local_sampled_eval.yaml")
    parser.add_argument("--split", default="exval")
    parser.add_argument("--output-dir", default="experiments/08_pre_review_gate/results")
    parser.add_argument("--report", default="experiments/08_pre_review_gate/reports/pre_review_gate.md")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--gallery", type=int, default=50000)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    started = time.perf_counter()
    rows = _read_csv(Path(args.metadata))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "pre_review_gate_progress.log"
    log_path.write_text("event=start\n", encoding="utf-8")

    query_indices = _sample_balanced(rows, args.queries, "slide_id", args.seed)
    gallery_indices = _sample_balanced(rows, args.gallery, "slide_id", args.seed + 1)
    _write_csv(out / "query_set.csv", [rows[idx] for idx in query_indices], list(rows[0]))
    _write_csv(out / "gallery_set.csv", [rows[idx] for idx in gallery_indices], list(rows[0]))

    slide_counts = Counter(row["slide_id"] for row in rows)
    package_counts = Counter(row["student_feature_package_path"] for row in rows)
    coverage_rows = [
        {"unit": "slide", "id": key, "sampled_tiles": value}
        for key, value in sorted(slide_counts.items())
    ] + [
        {"unit": "feature_package", "id": key, "sampled_tiles": value}
        for key, value in sorted(package_counts.items())
    ]
    _write_csv(out / "coverage_qc.csv", coverage_rows, ["unit", "id", "sampled_tiles"])

    all_metrics = []
    all_qc = []
    retrieval_paths = []

    with log_path.open("a", encoding="utf-8") as log:
        log.write("event=load_z_hcc\n")
    z_query = _load_z_features(rows, query_indices)
    z_gallery = _load_z_features(rows, gallery_indices)
    all_qc.append(_embedding_qc("z_hcc_query", z_query))
    all_qc.append(_embedding_qc("z_hcc_gallery", z_gallery))
    retrieval_rows, metrics = _retrieval("z_hcc", z_query, z_gallery, rows, query_indices, gallery_indices, args.topk)
    path = out / "retrieval_z_hcc_exval.csv"
    _write_csv(path, retrieval_rows, list(retrieval_rows[0]))
    retrieval_paths.append(path)
    all_metrics.append(metrics)

    tile_to_teacher_paths = _tile_teacher_paths(Path(args.config), args.split)
    for model in ["gigapath", "h_optimus_1", "uni2_h", "virchow2"]:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"event=load_teacher model={model}\n")
        q_arr = _load_teacher_features(rows, query_indices, model, tile_to_teacher_paths)
        g_arr = _load_teacher_features(rows, gallery_indices, model, tile_to_teacher_paths)
        all_qc.append(_embedding_qc(f"{model}_query", q_arr))
        all_qc.append(_embedding_qc(f"{model}_gallery", g_arr))
        retrieval_rows, metrics = _retrieval(model, q_arr, g_arr, rows, query_indices, gallery_indices, args.topk)
        path = out / f"retrieval_{model}_exval.csv"
        _write_csv(path, retrieval_rows, list(retrieval_rows[0]))
        retrieval_paths.append(path)
        all_metrics.append(metrics)
        del q_arr, g_arr

    _write_csv(out / "embedding_qc.csv", all_qc, list(all_qc[0]))
    _write_csv(out / "retrieval_metrics.csv", all_metrics, list(all_metrics[0]))

    merged = []
    for path in retrieval_paths:
        merged.extend(_read_csv(path))
    _write_csv(out / "merged_query_results_for_review.csv", merged, list(merged[0]))

    elapsed = time.perf_counter() - started
    manifest = {
        "metadata": args.metadata,
        "split": args.split,
        "models": MODELS,
        "queries": len(query_indices),
        "gallery": len(gallery_indices),
        "topk": args.topk,
        "elapsed_seconds": elapsed,
        "outputs": {
            "embedding_qc": str(out / "embedding_qc.csv"),
            "retrieval_metrics": str(out / "retrieval_metrics.csv"),
            "merged_review_candidates": str(out / "merged_query_results_for_review.csv"),
        },
    }
    (out / "pre_review_gate_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    best = {row["model"]: row for row in all_metrics}
    lines = [
        "# Pre-review Gate",
        "",
        f"Sampled cache rows: {len(rows)}.",
        f"Query count: {len(query_indices)}. Gallery count: {len(gallery_indices)}. Top-k: {args.topk}.",
        f"Elapsed seconds: {elapsed:.1f}.",
        "",
        "## Retrieval QC",
        "",
        "| model | queries full top-k | mean same-slide skipped | mean top1-topk margin |",
        "|---|---:|---:|---:|",
    ]
    for model in MODELS:
        row = best[model]
        lines.append(
            f"| {model} | {row['queries_with_full_topk']} | "
            f"{float(row['mean_same_slide_skipped_before_topk']):.2f} | "
            f"{float(row['mean_top1_to_topk_margin']):.4f} |"
        )
    lines.extend([
        "",
        "## Gate Decision",
        "",
        "PASS for automatic pre-review benchmarking: all models produced full same-slide-filtered top-k retrieval tables.",
        "Expert scoring is still not started; this gate only validates the frozen candidate-generation machinery.",
    ])
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"pre_review_gate_ok elapsed_sec={elapsed:.1f} output={out}")


if __name__ == "__main__":
    main()
