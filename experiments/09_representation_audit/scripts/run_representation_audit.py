from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hcc_sempath.modeling.prototypes import load_prototype_registry
from hcc_sempath.training.config import load_config, manifest_data_paths, teacher_names
from hcc_sempath.training.datasets import _open_feature_source
from hcc_sempath.training.manifest import load_training_manifest


TEACHERS = ["gigapath", "h_optimus_1", "uni2_h", "virchow2"]
MODELS = ["z_hcc", *TEACHERS]


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _tile_to_teacher_paths(cfg_path: Path, split: str) -> dict[str, dict[str, str]]:
    cfg = load_config(cfg_path)
    cfg["data"]["exval_tile_fraction"] = 1.0
    cfg["data"]["val_tile_fraction"] = 1.0
    manifest = load_training_manifest(cfg["data"]["train_manifest_path"])
    tile_packages, teacher_packages = manifest_data_paths(cfg, manifest, split)
    names = teacher_names(cfg)
    return {
        str(tile_path): {name: teacher_packages[name][idx] for name in names}
        for idx, tile_path in enumerate(tile_packages)
    }


def _read_feature_from_source(source, row: int, teacher: str) -> np.ndarray:
    if source.__class__.__name__ == "MergedTeacherFeatureCacheReader":
        return source.read_feature_at(row, teacher)
    return source.read_feature_at(row)


def _load_teacher_feature_map(
    metadata_by_tile: dict[str, dict],
    tile_ids: set[str],
    teacher: str,
    tile_to_teacher_paths: dict[str, dict[str, str]],
) -> dict[str, np.ndarray]:
    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for tile_id in sorted(tile_ids):
        meta = metadata_by_tile[tile_id]
        source_path = tile_to_teacher_paths[meta["tile_package_path"]][teacher]
        grouped[source_path].append((tile_id, int(meta["source_row"])))
    result: dict[str, np.ndarray] = {}
    for source_path, items in grouped.items():
        source = _open_feature_source(Path(source_path))
        try:
            for tile_id, row in items:
                result[tile_id] = _read_feature_from_source(source, row, teacher).astype("float32", copy=False)
        finally:
            close = getattr(source, "close", None)
            if close is not None:
                close()
    return result


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _load_prototype_top_primary(prototype_dir: Path, teacher: str, features: dict[str, np.ndarray]) -> dict[str, tuple[str, float]]:
    path = prototype_dir / f"{teacher}_hcc_semantic_prototypes.pt"
    registry = load_prototype_registry(path)
    prototypes = F.normalize(registry.prototypes.float(), dim=1)
    primary = torch.tensor(registry.primary_indices, dtype=torch.long)
    primary_prototypes = prototypes.index_select(0, primary)
    names = [registry.names[idx] for idx in registry.primary_indices]
    result = {}
    tile_ids = sorted(features)
    batch_size = 2048
    for start in range(0, len(tile_ids), batch_size):
        batch_ids = tile_ids[start : start + batch_size]
        arr = torch.from_numpy(np.stack([features[tile_id] for tile_id in batch_ids]).astype("float32"))
        arr = F.normalize(arr, dim=1)
        logits = arr @ primary_prototypes.T
        values, indices = logits.max(dim=1)
        for tile_id, value, index in zip(batch_ids, values.tolist(), indices.tolist()):
            result[tile_id] = (names[int(index)], float(value))
    return result


def _retrieval_sets(rows: list[dict]) -> dict[tuple[str, str], set[str]]:
    result: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        result[(row["model"], row["query_id"])].add(row["neighbor_tile_id"])
    return result


def _cross_model_overlap(rows: list[dict]) -> list[dict]:
    sets = _retrieval_sets(rows)
    query_ids = sorted({row["query_id"] for row in rows}, key=int)
    out = []
    for query_id in query_ids:
        z = sets[("z_hcc", query_id)]
        for model in TEACHERS:
            other = sets[(model, query_id)]
            intersection = len(z & other)
            union = len(z | other)
            out.append({
                "query_id": query_id,
                "reference_model": "z_hcc",
                "comparison_model": model,
                "overlap_at_10": intersection,
                "jaccard_at_10": intersection / union if union else 0.0,
                "z_hcc_unique_at_10": len(z - other),
            })
    return out


def _query_margin(rows: list[dict]) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        scores[(row["model"], row["query_id"])].append(float(row["cosine"]))
    return {
        key: max(values) - min(values)
        for key, values in scores.items()
        if values
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="experiments/07_full_exval_cache/results/manifests/exval_sampled_z_hcc_metadata.csv")
    parser.add_argument("--retrieval", default="experiments/08_pre_review_gate/results/merged_query_results_for_review.csv")
    parser.add_argument("--config", default="experiments/shared/configs/local_sampled_eval.yaml")
    parser.add_argument("--prototype-dir", default="artifacts/prototypes")
    parser.add_argument("--split", default="exval")
    parser.add_argument("--output-dir", default="experiments/09_representation_audit/results")
    parser.add_argument("--report", default="experiments/09_representation_audit/reports/representation_audit.md")
    args = parser.parse_args()

    started = time.perf_counter()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "representation_audit_progress.log"
    log_path.write_text("event=start\n", encoding="utf-8")

    metadata = _read_csv(Path(args.metadata))
    metadata_by_tile = {row["tile_id"]: row for row in metadata}
    retrieval = _read_csv(Path(args.retrieval))
    used_tile_ids = {row["query_tile_id"] for row in retrieval} | {row["neighbor_tile_id"] for row in retrieval}
    tile_to_teacher_paths = _tile_to_teacher_paths(Path(args.config), args.split)

    features_by_teacher = {}
    proto_by_teacher = {}
    for teacher in TEACHERS:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"event=load_features teacher={teacher} tiles={len(used_tile_ids)}\n")
        features = _load_teacher_feature_map(metadata_by_tile, used_tile_ids, teacher, tile_to_teacher_paths)
        features_by_teacher[teacher] = features
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"event=prototype_top teacher={teacher}\n")
        proto_by_teacher[teacher] = _load_prototype_top_primary(Path(args.prototype_dir), teacher, features)

    overlap_rows = _cross_model_overlap(retrieval)
    _write_csv(out / "cross_model_overlap.csv", overlap_rows, list(overlap_rows[0]))

    pair_rows = []
    for row in retrieval:
        q = row["query_tile_id"]
        n = row["neighbor_tile_id"]
        teacher_cos = {}
        proto_match = 0
        proto_scores = {}
        for teacher in TEACHERS:
            cosine = _cosine(features_by_teacher[teacher][q], features_by_teacher[teacher][n])
            teacher_cos[teacher] = cosine
            q_proto, q_score = proto_by_teacher[teacher][q]
            n_proto, n_score = proto_by_teacher[teacher][n]
            if q_proto == n_proto:
                proto_match += 1
            proto_scores[f"{teacher}_query_primary"] = q_proto
            proto_scores[f"{teacher}_neighbor_primary"] = n_proto
            proto_scores[f"{teacher}_query_primary_score"] = q_score
            proto_scores[f"{teacher}_neighbor_primary_score"] = n_score
        values = list(teacher_cos.values())
        pair_rows.append({
            "model": row["model"],
            "query_id": row["query_id"],
            "rank": row["rank"],
            "query_tile_id": q,
            "neighbor_tile_id": n,
            "retrieval_cosine": row["cosine"],
            "teacher_cosine_mean": float(np.mean(values)),
            "teacher_cosine_std": float(np.std(values)),
            "teacher_cosine_min": float(np.min(values)),
            "teacher_primary_match_count": proto_match,
            **{f"{teacher}_pair_cosine": teacher_cos[teacher] for teacher in TEACHERS},
            **proto_scores,
        })
    _write_csv(out / "pair_teacher_agreement.csv", pair_rows, list(pair_rows[0]))

    summary_rows = []
    for model in MODELS:
        subset = [row for row in pair_rows if row["model"] == model]
        summary_rows.append({
            "model": model,
            "pairs": len(subset),
            "teacher_cosine_mean": float(np.mean([float(row["teacher_cosine_mean"]) for row in subset])),
            "teacher_cosine_min_mean": float(np.mean([float(row["teacher_cosine_min"]) for row in subset])),
            "teacher_cosine_std_mean": float(np.mean([float(row["teacher_cosine_std"]) for row in subset])),
            "prototype_primary_match_mean": float(np.mean([int(row["teacher_primary_match_count"]) for row in subset])),
            "prototype_primary_match_all4_fraction": float(np.mean([int(row["teacher_primary_match_count"]) == 4 for row in subset])),
        })
    _write_csv(out / "model_teacher_agreement_summary.csv", summary_rows, list(summary_rows[0]))

    overlap_summary = []
    for model in TEACHERS:
        subset = [row for row in overlap_rows if row["comparison_model"] == model]
        overlap_summary.append({
            "comparison_model": model,
            "mean_overlap_at_10": float(np.mean([int(row["overlap_at_10"]) for row in subset])),
            "mean_jaccard_at_10": float(np.mean([float(row["jaccard_at_10"]) for row in subset])),
            "mean_z_hcc_unique_at_10": float(np.mean([int(row["z_hcc_unique_at_10"]) for row in subset])),
        })
    _write_csv(out / "cross_model_overlap_summary.csv", overlap_summary, list(overlap_summary[0]))

    margins = _query_margin(retrieval)
    z_pairs_by_query: dict[str, list[dict]] = defaultdict(list)
    for row in pair_rows:
        if row["model"] == "z_hcc":
            z_pairs_by_query[row["query_id"]].append(row)
    failure_rows = []
    for query_id, rows_for_query in z_pairs_by_query.items():
        mean_teacher = float(np.mean([float(row["teacher_cosine_mean"]) for row in rows_for_query]))
        mean_disagreement = float(np.mean([float(row["teacher_cosine_std"]) for row in rows_for_query]))
        proto_match = float(np.mean([int(row["teacher_primary_match_count"]) for row in rows_for_query]))
        margin = float(margins.get(("z_hcc", query_id), 0.0))
        if margin < 0.02:
            stratum = "low_margin"
        elif mean_teacher >= 0.65 and mean_disagreement <= 0.08 and proto_match >= 3.0:
            stratum = "high_confidence_teacher_consensus"
        elif mean_teacher >= 0.55 and mean_disagreement > 0.12:
            stratum = "high_teacher_disagreement"
        else:
            stratum = "intermediate"
        failure_rows.append({
            "query_id": query_id,
            "query_tile_id": rows_for_query[0]["query_tile_id"],
            "z_hcc_margin": margin,
            "mean_teacher_pair_cosine": mean_teacher,
            "mean_teacher_disagreement": mean_disagreement,
            "mean_teacher_primary_match_count": proto_match,
            "stratum": stratum,
        })
    _write_csv(out / "query_failure_strata.csv", failure_rows, list(failure_rows[0]))
    stratum_counts = Counter(row["stratum"] for row in failure_rows)
    stratum_rows = [{"stratum": key, "queries": value} for key, value in sorted(stratum_counts.items())]
    _write_csv(out / "query_failure_strata_summary.csv", stratum_rows, ["stratum", "queries"])

    elapsed = time.perf_counter() - started
    manifest = {
        "metadata": args.metadata,
        "retrieval": args.retrieval,
        "used_tile_count": len(used_tile_ids),
        "elapsed_seconds": elapsed,
        "outputs": {
            "cross_model_overlap": str(out / "cross_model_overlap.csv"),
            "pair_teacher_agreement": str(out / "pair_teacher_agreement.csv"),
            "model_teacher_agreement_summary": str(out / "model_teacher_agreement_summary.csv"),
            "query_failure_strata": str(out / "query_failure_strata.csv"),
        },
    }
    (out / "representation_audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary_by_model = {row["model"]: row for row in summary_rows}
    lines = [
        "# Representation Audit",
        "",
        f"Retrieval pairs audited: {len(pair_rows)}.",
        f"Unique tiles requiring teacher features: {len(used_tile_ids)}.",
        f"Elapsed seconds: {elapsed:.1f}.",
        "",
        "## Teacher Agreement By Retrieval Model",
        "",
        "| model | pair teacher cosine | teacher disagreement | primary prototype match | all-4 primary match |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        row = summary_by_model[model]
        lines.append(
            f"| {model} | {float(row['teacher_cosine_mean']):.4f} | "
            f"{float(row['teacher_cosine_std_mean']):.4f} | "
            f"{float(row['prototype_primary_match_mean']):.2f} | "
            f"{float(row['prototype_primary_match_all4_fraction']):.3f} |"
        )
    lines.extend([
        "",
        "## z_hcc vs Teacher Retrieval Overlap",
        "",
        "| teacher | overlap@10 | jaccard@10 | z_hcc unique@10 |",
        "|---|---:|---:|---:|",
    ])
    for row in overlap_summary:
        lines.append(
            f"| {row['comparison_model']} | {float(row['mean_overlap_at_10']):.2f} | "
            f"{float(row['mean_jaccard_at_10']):.3f} | {float(row['mean_z_hcc_unique_at_10']):.2f} |"
        )
    lines.extend([
        "",
        "## Failure Strata",
        "",
        "| stratum | queries |",
        "|---|---:|",
    ])
    for row in stratum_rows:
        lines.append(f"| {row['stratum']} | {row['queries']} |")
    lines.extend([
        "",
        "## Gate Interpretation",
        "",
        "Automatic representation audit completed. Expert scoring remains the only unresolved morphology-relevance endpoint.",
    ])
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"representation_audit_ok elapsed_sec={elapsed:.1f} output={out}")


if __name__ == "__main__":
    main()
