from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


MODELS = ["student", "gigapath", "h_optimus_1", "uni2_h", "virchow2"]


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_embeddings(result_dir: Path, model: str, split: str) -> np.ndarray:
    if model == "student":
        path = result_dir / f"student_embeddings_{split}.npz"
    else:
        path = result_dir / f"teacher_embeddings_{model}_{split}.npz"
    arr = np.load(path)["embedding_norm"].astype("float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-8)


def _select_indices(size: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = min(count, size)
    return np.sort(rng.choice(size, size=count, replace=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="exval")
    parser.add_argument("--queries", type=int, default=16)
    parser.add_argument("--gallery", type=int, default=128)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    embedding_dir = Path(args.embedding_dir)
    output_dir = Path(args.output_dir)
    metadata = _read_csv(embedding_dir / f"tile_metadata_{args.split}.csv")
    query_idx = _select_indices(len(metadata), args.queries, args.seed)
    gallery_idx = _select_indices(len(metadata), args.gallery, args.seed + 1)
    _write_csv(output_dir / "query_set.csv", [metadata[int(i)] for i in query_idx], list(metadata[0]))
    _write_csv(output_dir / "gallery_set.csv", [metadata[int(i)] for i in gallery_idx], list(metadata[0]))

    merged_rows = []
    for model in MODELS:
        emb = _load_embeddings(embedding_dir, model, args.split)
        q = emb[query_idx]
        g = emb[gallery_idx]
        sims = q @ g.T
        rows = []
        for qi, source_idx in enumerate(query_idx):
            order = np.argsort(-sims[qi])
            rank = 0
            for gi in order:
                if int(gallery_idx[gi]) == int(source_idx):
                    continue
                rank += 1
                row = {
                    "model": model,
                    "split": args.split,
                    "query_rank": qi,
                    "query_tile_id": metadata[int(source_idx)]["tile_id"],
                    "query_slide_id": metadata[int(source_idx)]["slide_id"],
                    "rank": rank,
                    "neighbor_tile_id": metadata[int(gallery_idx[gi])]["tile_id"],
                    "neighbor_slide_id": metadata[int(gallery_idx[gi])]["slide_id"],
                    "cosine": f"{float(sims[qi, gi]):.8f}",
                }
                rows.append(row)
                merged_rows.append(row)
                if rank >= args.topk:
                    break
        _write_csv(
            output_dir / f"retrieval_{model}_{args.split}.csv",
            rows,
            ["model", "split", "query_rank", "query_tile_id", "query_slide_id", "rank", "neighbor_tile_id", "neighbor_slide_id", "cosine"],
        )
    _write_csv(
        output_dir / "merged_query_results_for_review.csv",
        merged_rows,
        ["model", "split", "query_rank", "query_tile_id", "query_slide_id", "rank", "neighbor_tile_id", "neighbor_slide_id", "cosine"],
    )
    manifest = {
        "split": args.split,
        "models": MODELS,
        "queries": int(len(query_idx)),
        "gallery": int(len(gallery_idx)),
        "topk": int(args.topk),
        "seed": int(args.seed),
    }
    (output_dir / "retrieval_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = [
        "# Retrieval Benchmark Summary",
        "",
        f"Split: `{args.split}`. Fixed query count: {len(query_idx)}. Fixed gallery count: {len(gallery_idx)}. Top-k: {args.topk}.",
        "",
        "Outputs include one top-k table per model and `merged_query_results_for_review.csv`.",
    ]
    report = Path("experiments/03_retrieval_benchmark/reports/retrieval_benchmark_summary.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"retrieval_ok output={output_dir}")


if __name__ == "__main__":
    main()
