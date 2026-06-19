"""Symmetric representation-quality probes across z_HCC and the four teachers.

All five models are evaluated under IDENTICAL readout protocols on the 1000-tile
expert-adjudicated asset, so the comparison is method-symmetric (unlike Table 1,
where the student uses a trained prototype readout and teachers use zero-shot
nearest-centroid). Protocols:

  * Linear probe  : slide-grouped stratified k-fold CV, logistic regression on
                    standardized features (L1: multinomial; L2: per-attribute
                    one-vs-rest).
  * kNN           : cosine kNN (k=10) excluding all same-slide tiles for L1.
  * Neighborhood  : reuse training _neighborhood_purity (L1 + L2 Jaccard).

Reported per queue: Random500, Top500, All. The probe is fit by CV ON the 1000
eval tiles (unseen by any representation training; the student's prototype
supervision used a SEPARATE 3000-tile set), so no model trains a probe on its own
test data and z_HCC gets no special advantage.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import probe_data as P
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

MODELS = ["z_hcc", *P.TEACHERS]
RESULTS = Path(__file__).resolve().parents[1] / "results"
CACHE = RESULTS / "cache"
N_FOLDS = 5
KNN_K = 10
SEED = 13


def _l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def _masks(labels: dict) -> dict[str, np.ndarray]:
    n = len(labels["l1"])
    return {"All": np.ones(n, bool), "Random500": labels["random500"], "Top500": labels["top500"]}


def linear_probe_l1(feats: np.ndarray, y: np.ndarray, groups: np.ndarray, masks: dict) -> dict:
    """Slide-grouped stratified CV LR; score out-of-fold predictions per queue."""
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(len(y), -1, dtype=np.int64)
    for tr, te in skf.split(feats, y, groups):
        scaler = StandardScaler().fit(feats[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(scaler.transform(feats[tr]), y[tr])
        oof[te] = clf.predict(scaler.transform(feats[te]))
    out = {}
    for q, m in masks.items():
        out[q] = {
            "accuracy": round(float((oof[m] == y[m]).mean()), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y[m], oof[m])), 4),
            "macro_f1": round(float(f1_score(y[m], oof[m], average="macro")), 4),
        }
    return out


def linear_probe_l2(feats: np.ndarray, Y: np.ndarray, groups: np.ndarray, masks: dict) -> dict:
    """Per-attribute one-vs-rest LR via CV; out-of-fold probabilities -> macro AP/AUC."""
    n, k = Y.shape
    oof_prob = np.zeros((n, k), dtype=np.float64)
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for j in range(k):
        yj = Y[:, j]
        positive_groups = len(np.unique(groups[yj == 1]))
        negative_groups = len(np.unique(groups[yj == 0]))
        if positive_groups < N_FOLDS or negative_groups < N_FOLDS:
            oof_prob[:, j] = yj.mean()  # too few positives to CV; constant prior
            continue
        for tr, te in skf.split(feats, yj, groups):
            scaler = StandardScaler().fit(feats[tr])
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
            clf.fit(scaler.transform(feats[tr]), yj[tr])
            oof_prob[te, j] = clf.predict_proba(scaler.transform(feats[te]))[:, 1]
    out = {}
    for q, m in masks.items():
        aps, aucs = [], []
        for j in range(k):
            yj = Y[m, j]
            if yj.sum() == 0 or yj.sum() == len(yj):
                continue
            aps.append(average_precision_score(yj, oof_prob[m, j]))
            aucs.append(roc_auc_score(yj, oof_prob[m, j]))
        out[q] = {
            "macro_ap": round(float(np.mean(aps)), 4) if aps else None,
            "macro_auc": round(float(np.mean(aucs)), 4) if aucs else None,
            "n_attrs_scored": len(aps),
        }
    return out


def knn_l1(feats: np.ndarray, y: np.ndarray, groups: np.ndarray, masks: dict, k: int = KNN_K) -> dict:
    """Cosine kNN majority vote after excluding every same-slide tile."""
    f = _l2norm(feats.astype(np.float64))
    sim = f @ f.T
    sim[groups[:, None] == groups[None, :]] = -np.inf
    nn = np.argsort(-sim, axis=1)[:, :k]
    pred = np.array([np.bincount(y[nn[i]], minlength=int(y.max()) + 1).argmax() for i in range(len(y))])
    return {q: {"accuracy": round(float((pred[m] == y[m]).mean()), 4)} for q, m in masks.items()}


def neighborhood_purity(feats: np.ndarray, y: np.ndarray, Y: np.ndarray, masks: dict, k: int = KNN_K) -> dict:
    """Reuse the training-time purity routine for an apples-to-apples intrinsic metric."""
    import torch
    from hcc_sempath.training.zhcc_metrics import _neighborhood_purity

    # purity is computed over the full set's neighbor graph; report global value
    l1p, l2p = _neighborhood_purity(
        torch.from_numpy(_l2norm(feats).astype("float32")),
        torch.from_numpy(y.astype("int64")),
        torch.from_numpy(Y.astype("int64")),
        k,
    )
    return {"All": {"l1_purity": round(float(l1p), 4), "l2_purity": round(float(l2p), 4)}}


def main() -> None:
    rows = P.load_review_rows()
    labels = P.build_labels(rows)
    feats = P.load_or_cache_features(rows, CACHE)
    masks = _masks(labels)
    y, Y = labels["l1"], labels["l2"]
    groups = np.asarray([row["slide_id"] for row in rows])

    report = {"protocol": {"folds": N_FOLDS, "knn_k": KNN_K, "seed": SEED,
                           "n_tiles": len(rows), "cv_group": "slide_id",
                           "knn_exclusion": "same_slide"}, "models": {}}
    for name in MODELS:
        f = feats[name]
        report["models"][name] = {
            "feature_dim": int(f.shape[1]),
            "linear_probe_l1": linear_probe_l1(f, y, groups, masks),
            "knn_l1": knn_l1(f, y, groups, masks),
            "linear_probe_l2": linear_probe_l2(f, Y, groups, masks),
            "neighborhood_purity": neighborhood_purity(f, y, Y, masks),
        }
        print(f"done: {name}")

    out_path = RESULTS / "symmetric_probe_results.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    # compact L1 table
    print("\n=== Linear-probe L1 accuracy ===")
    print(f"{'model':14} {'Random500':>10} {'Top500':>10} {'All':>10}")
    for name in MODELS:
        lp = report["models"][name]["linear_probe_l1"]
        print(f"{name:14} {lp['Random500']['accuracy']:>10} {lp['Top500']['accuracy']:>10} {lp['All']['accuracy']:>10}")
    print("\n=== kNN L1 accuracy ===")
    print(f"{'model':14} {'Random500':>10} {'Top500':>10} {'All':>10}")
    for name in MODELS:
        kn = report["models"][name]["knn_l1"]
        print(f"{name:14} {kn['Random500']['accuracy']:>10} {kn['Top500']['accuracy']:>10} {kn['All']['accuracy']:>10}")


if __name__ == "__main__":
    main()
