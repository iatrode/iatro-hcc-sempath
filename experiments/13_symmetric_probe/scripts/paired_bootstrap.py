"""Paired bootstrap CIs for the symmetric probe: z_HCC vs the best teacher per
queue, under both LR and kNN readouts. Writes paired_bootstrap.json.

Pairing is on the same tiles (same out-of-fold prediction indices), so the CI
reflects per-tile correctness differences, matching the paper's Table 1 paired
bootstrap convention.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import probe_data as P
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

MODELS = ["z_hcc", *P.TEACHERS]
RESULTS = Path(__file__).resolve().parents[1] / "results"
SEED = 13
N_BOOT = 2000


def _oof_lr(feats, y):
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = np.full(len(y), -1)
    for tr, te in skf.split(feats, y):
        sc = StandardScaler().fit(feats[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(feats[tr]), y[tr])
        oof[te] = clf.predict(sc.transform(feats[te]))
    return oof


def _oof_knn(feats, y, k=10):
    fn = feats / np.clip(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12, None)
    sim = fn @ fn.T
    np.fill_diagonal(sim, -np.inf)
    nn = np.argsort(-sim, axis=1)[:, :k]
    return np.array([np.bincount(y[nn[i]], minlength=int(y.max()) + 1).argmax() for i in range(len(y))])


def _paired(a, b, mask, rng):
    idx = np.where(mask)[0]
    d = [a[s].mean() - b[s].mean() for s in (rng.choice(idx, len(idx), replace=True) for _ in range(N_BOOT))]
    return np.percentile(d, [2.5, 50, 97.5]).tolist()


def main():
    rows = P.load_review_rows()
    lab = P.build_labels(rows)
    y = lab["l1"]
    feats = P.load_or_cache_features(rows, RESULTS / "cache")
    masks = {"Random500": lab["random500"], "Top500": lab["top500"], "All": np.ones(len(y), bool)}
    rng = np.random.default_rng(SEED)

    pred = {
        "LR": {m: _oof_lr(feats[m], y) for m in MODELS},
        "kNN": {m: _oof_knn(feats[m], y) for m in MODELS},
    }
    report = {}
    for proto in ("LR", "kNN"):
        report[proto] = {}
        for q, m in masks.items():
            za = (pred[proto]["z_hcc"] == y).astype(float)
            tb = max(P.TEACHERS, key=lambda t: (pred[proto][t] == y)[m].mean())
            tba = (pred[proto][tb] == y).astype(float)
            lo, md, hi = _paired(za, tba, m, rng)
            report[proto][q] = {
                "best_teacher": tb,
                "z_hcc_acc": round(float(za[m].mean()), 4),
                "best_teacher_acc": round(float(tba[m].mean()), 4),
                "delta_median": round(md, 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "significant": bool(lo > 0),
            }
    (RESULTS / "paired_bootstrap.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
