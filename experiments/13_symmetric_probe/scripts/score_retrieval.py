"""Score blinded retrieval ratings into per-model precision@k / mean-relevance@k /
NDCG@k, with paired bootstrap CIs (z_HCC vs each teacher).

Input: the expert ratings (CSV or JSON) with adjudicated relevance in {0,1,2} per
(query, neighbor) pair. Multiple raters are averaged per pair (consensus). The
model->pair provenance (pair_provenance.json) maps each rated pair back to which
models retrieved it and at what rank, so one expert score is reused across every
model that retrieved the pair.

Usage:
  python score_retrieval.py expert_ratings.csv
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "retrieval"
MODELS = ["z_hcc", "gigapath", "h_optimus_1", "uni2_h", "virchow2"]
TEACHERS = MODELS[1:]
TOPK = 5
SEED = 13
N_BOOT = 2000


def _stable_key(q: str, n: str) -> str:
    import hashlib
    return hashlib.sha1(f"{q}|{n}".encode()).hexdigest()[:12]


def load_consensus(rating_paths: list[str]) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for p in rating_paths:
        if p.endswith(".csv"):
            import csv as _csv
            with open(p, newline="") as fh:
                for r in _csv.DictReader(fh):
                    v = r.get("rating")
                    if v not in (None, ""):
                        acc[r["pair_id"]].append(float(v))
            continue
        for r in json.loads(Path(p).read_text()):
            if r.get("value") is not None:
                acc[r["pair_id"]].append(float(r["value"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def dcg(rels: list[float]) -> float:
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(rels))


def score(provenance: dict, consensus: dict[str, float]) -> dict:
    """Rebuild per-model ranked relevance lists per query, compute metrics."""
    # model -> query_tile -> list of (rank, relevance)
    per = {m: defaultdict(list) for m in MODELS}
    for key, e in provenance.items():
        rel = consensus.get(key)
        if rel is None:
            continue
        q = e["query_tile_id"]
        for m, rank in e["retrieved_by"].items():
            per[m][q].append((rank, rel))

    results = {}
    # per-query, per-model metric vectors (aligned across models by query)
    queries = sorted({e["query_tile_id"] for e in provenance.values()})
    metric_by_model = {m: {"p_at_k": [], "mean_rel": [], "ndcg": []} for m in MODELS}
    for q in queries:
        ideal = None
        for m in MODELS:
            ranked = sorted(per[m].get(q, []), key=lambda x: x[0])[:TOPK]
            rels = [r for _, r in ranked]
            if not rels:
                continue
            p_at_k = np.mean([1.0 if r >= 1.5 else 0.0 for r in rels])  # rel>=2 counts as relevant
            mean_rel = np.mean(rels)
            if ideal is None:
                ideal = sorted(rels, reverse=True)
            idcg = dcg(sorted(rels, reverse=True)) or 1.0
            ndcg = dcg(rels) / idcg
            metric_by_model[m]["p_at_k"].append(p_at_k)
            metric_by_model[m]["mean_rel"].append(mean_rel)
            metric_by_model[m]["ndcg"].append(ndcg)

    for m in MODELS:
        results[m] = {k: round(float(np.mean(v)), 4) if v else None for k, v in metric_by_model[m].items()}
        results[m]["n_queries"] = len(metric_by_model[m]["p_at_k"])

    # paired bootstrap: z_hcc minus each teacher on mean_rel (per query, paired)
    rng = np.random.default_rng(SEED)
    paired = {}
    z = np.array(metric_by_model["z_hcc"]["mean_rel"])
    for t in TEACHERS:
        tv = np.array(metric_by_model[t]["mean_rel"])
        n = min(len(z), len(tv))
        if n == 0:
            continue
        diffs = z[:n] - tv[:n]
        boot = [diffs[rng.choice(n, n, replace=True)].mean() for _ in range(N_BOOT)]
        lo, md, hi = np.percentile(boot, [2.5, 50, 97.5])
        paired[t] = {"delta_mean_rel": round(float(md), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
                     "significant": bool(lo > 0)}
    return {"per_model": results, "z_hcc_vs_teacher_paired": paired, "n_rated_pairs": len(consensus)}


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python score_retrieval.py rating1.json [rating2.json ...]")
        sys.exit(1)
    provenance = json.loads((OUT / "pair_provenance.json").read_text())
    # provenance keys are stable hashes; ensure pair_id matches
    prov_by_key = {}
    for key, e in provenance.items():
        prov_by_key[_stable_key(e["query_tile_id"], e["neighbor_tile_id"])] = e
    consensus = load_consensus(sys.argv[1:])
    report = score(prov_by_key, consensus)
    (OUT / "retrieval_scores.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
