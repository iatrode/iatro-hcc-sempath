"""Generate the blinded morphology-retrieval annotation set.

Uses the 1000-tile expert-labeled asset as both query pool and gallery (every
tile already carries adjudicated L1/L2 labels). For each query, take top-k cosine
neighbors per model (z_HCC + 4 teachers), then DEDUPLICATE (query, neighbor) pairs
across models — experts score each unique pair once, and that score is shared by
every model that retrieved it. Pairs are emitted in a blinded, shuffled order with
model identity hidden, for the web annotation tool.

Outputs (results/retrieval/):
  pairs.json        - unique (query, neighbor) pairs, blinded, shuffled (blinded review input)
  pair_provenance.json - which models retrieved each pair + at what rank (NOT shown to expert)
  queries.json      - query tile list with metadata
  tiles/<tile_id>.jpg  - exported images referenced for review
Also prints the exact deduped pair count.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import probe_data as P

MODELS = ["z_hcc", *P.TEACHERS]
RESULTS = Path(__file__).resolve().parents[1] / "results"
CACHE = RESULTS / "cache"
OUT = RESULTS / "retrieval"
TILES = OUT / "tiles"

N_TOP_QUERIES = 30      # from Top500 (high-conflict)
N_RANDOM_QUERIES = 30   # from Random500 (population reference); 30:30 balanced
TOPK = 5                # retrieval depth; @5 is standard, keeps annotation < prototype set
SEED = 13


def _l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def _stable_key(query_tile: str, neighbor_tile: str) -> str:
    return hashlib.sha1(f"{query_tile}|{neighbor_tile}".encode()).hexdigest()[:12]


def select_queries(labels: dict, rng: np.random.Generator) -> list[int]:
    top_idx = np.where(labels["top500"])[0]
    rnd_idx = np.where(labels["random500"])[0]
    q_top = rng.choice(top_idx, min(N_TOP_QUERIES, len(top_idx)), replace=False)
    q_rnd = rng.choice(rnd_idx, min(N_RANDOM_QUERIES, len(rnd_idx)), replace=False)
    return sorted(int(i) for i in np.concatenate([q_top, q_rnd]))


def build(rng: np.random.Generator) -> dict:
    rows = P.load_review_rows()
    labels = P.build_labels(rows)
    feats = {m: _l2norm(P.load_or_cache_features(rows, CACHE)[m].astype("float64")) for m in MODELS}
    query_indices = select_queries(labels, rng)

    # per-model top-k neighbors for each query (exclude self and same-slide to avoid trivial near-dupes)
    slide = np.array([r["slide_id"] for r in rows])
    provenance: dict[str, dict] = {}
    for m in MODELS:
        sim = feats[m] @ feats[m].T
        np.fill_diagonal(sim, -np.inf)
        for qi in query_indices:
            mask_same_slide = slide == slide[qi]
            order = np.argsort(-sim[qi])
            picked = [int(j) for j in order if not mask_same_slide[j]][:TOPK]
            for rank, nb in enumerate(picked, 1):
                key = _stable_key(rows[qi]["tile_id"], rows[nb]["tile_id"])
                entry = provenance.setdefault(key, {
                    "query_idx": qi, "neighbor_idx": nb,
                    "query_tile_id": rows[qi]["tile_id"], "neighbor_tile_id": rows[nb]["tile_id"],
                    "retrieved_by": {},
                })
                entry["retrieved_by"][m] = rank

    # blinded pair list (no model identity, shuffled)
    pair_keys = list(provenance.keys())
    rng.shuffle(pair_keys)
    pairs = []
    for key in pair_keys:
        e = provenance[key]
        pairs.append({
            "pair_id": key,
            "query_tile_id": e["query_tile_id"],
            "query_image": f"tiles/{e['query_tile_id']}.jpg",
            "neighbor_tile_id": e["neighbor_tile_id"],
            "neighbor_image": f"tiles/{e['neighbor_tile_id']}.jpg",
        })

    queries = [{
        "query_idx": qi, "tile_id": rows[qi]["tile_id"], "slide_id": rows[qi]["slide_id"],
        "queue": "top500" if labels["top500"][qi] else "random500",
        "expert_l1": rows[qi]["l1"],
    } for qi in query_indices]

    return {
        "rows": rows, "labels": labels, "query_indices": query_indices,
        "provenance": provenance, "pairs": pairs, "queries": queries,
    }


def export_images(rows: list[dict], tile_indices: set[int]) -> int:
    from hcc_sempath.io.tile_package import TilePackageReader
    TILES.mkdir(parents=True, exist_ok=True)
    readers: dict[str, "TilePackageReader"] = {}
    n = 0
    for idx in sorted(tile_indices):
        r = rows[idx]
        dst = TILES / f"{r['tile_id']}.jpg"
        if dst.exists():
            n += 1
            continue
        pkg = r["package_path"]
        reader = readers.get(pkg) or readers.setdefault(pkg, TilePackageReader(pkg))
        img = reader.read_image(r["tile_id"]).convert("RGB")
        img.save(dst, "JPEG", quality=90)
        n += 1
    for reader in readers.values():
        close = getattr(reader, "close", None)
        if close is not None:
            close()
    return n


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    data = build(rng)

    # images needed = all queries + all neighbors
    needed = set(data["query_indices"])
    for e in data["provenance"].values():
        needed.add(e["neighbor_idx"])
    n_img = export_images(data["rows"], needed)

    (OUT / "pairs.json").write_text(json.dumps(data["pairs"], indent=2), encoding="utf-8")
    (OUT / "queries.json").write_text(json.dumps(data["queries"], indent=2), encoding="utf-8")
    # provenance: drop numpy ints for json
    prov = {k: {"query_tile_id": v["query_tile_id"], "neighbor_tile_id": v["neighbor_tile_id"],
                "retrieved_by": v["retrieved_by"]} for k, v in data["provenance"].items()}
    (OUT / "pair_provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")

    manifest = {
        "n_queries": len(data["query_indices"]),
        "n_top500_queries": N_TOP_QUERIES, "n_random500_queries": N_RANDOM_QUERIES,
        "topk": TOPK, "seed": SEED, "models": MODELS,
        "n_unique_pairs": len(data["pairs"]),
        "n_images_exported": n_img,
        "naive_pairs_without_dedup": len(data["query_indices"]) * TOPK * len(MODELS),
    }
    (OUT / "retrieval_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
