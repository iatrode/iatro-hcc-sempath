"""Slide-grouped metric panel for the symmetric probe: accuracy, balanced accuracy,
macro-F1, and macro one-vs-rest AUC for L1, per model per queue, with paired
bootstrap CIs (z_HCC vs best teacher) on EACH metric.

Threshold-free / imbalance-aware metrics (balanced acc, macro-AUC) matter here
because L1 is heavily imbalanced (603/182/206/9). Raw accuracy is dominated by
the HCC-tumor majority; the other metrics can tell a different story.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import probe_data as P
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

MODELS = ["z_hcc", *P.TEACHERS]
RESULTS = Path(__file__).resolve().parents[1] / "results"
CACHE = RESULTS / "cache"
N_FOLDS, SEED, N_BOOT = 5, 13, 2000


def oof_predict(feats, y, groups):
    """Return out-of-fold hard preds and class-probability matrix."""
    skf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
    classes = np.unique(y)
    proba = np.zeros((len(y), len(classes)))
    pred = np.full(len(y), -1)
    for tr, te in skf.split(feats, y, groups):
        sc = StandardScaler().fit(feats[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(feats[tr]), y[tr])
        proba[te] = clf.predict_proba(sc.transform(feats[te]))
        pred[te] = clf.predict(sc.transform(feats[te]))
    return pred, proba, classes


def metrics(y, pred, proba, classes, mask):
    ym, pm, prm = y[mask], pred[mask], proba[mask]
    present = np.unique(ym)
    # macro OVR AUC over classes present in this mask with both pos/neg
    aucs = []
    for ci, c in enumerate(classes):
        yc = (ym == c).astype(int)
        if yc.sum() == 0 or yc.sum() == len(yc):
            continue
        aucs.append(roc_auc_score(yc, prm[:, ci]))
    return {
        "accuracy": float((pm == ym).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(ym, pm)),
        "macro_f1": float(f1_score(ym, pm, average="macro")),
        "macro_auc": float(np.mean(aucs)) if aucs else None,
        "n_classes_in_auc": len(aucs),
    }


def paired_ci(metric_fn, y, predA, prA, predB, prB, classes, mask, rng):
    """Bootstrap CI of metric(A)-metric(B) on the same resampled tiles."""
    idx = np.where(mask)[0]
    diffs = []
    for _ in range(N_BOOT):
        s = rng.choice(idx, len(idx), replace=True)
        m = np.zeros(len(y), bool);
        # build boolean mask via index multiset: use take instead
        a = metric_fn(y[s], predA[s], prA[s], classes)
        b = metric_fn(y[s], predB[s], prB[s], classes)
        if a is None or b is None:
            continue
        diffs.append(a - b)
    lo, md, hi = np.percentile(diffs, [2.5, 50, 97.5])
    return round(float(md), 4), [round(float(lo), 4), round(float(hi), 4)], bool(lo > 0)


def _bacc(y, p, pr, c):
    """Mean recall over classes present in a bootstrap resample, without warnings."""
    present = np.unique(y)
    return float(np.mean([(p[y == cls] == cls).mean() for cls in present]))
def _mf1(y, p, pr, c): return f1_score(y, p, average="macro")
def _auc(y, p, pr, c):
    aucs = []
    for ci, cls in enumerate(c):
        yc = (y == cls).astype(int)
        if 0 < yc.sum() < len(yc):
            aucs.append(roc_auc_score(yc, pr[:, ci]))
    return float(np.mean(aucs)) if aucs else None


def main():
    rows = P.load_review_rows()
    lab = P.build_labels(rows)
    y = lab["l1"]
    groups = np.asarray([row["slide_id"] for row in rows])
    feats = P.load_or_cache_features(rows, CACHE)
    masks = {"Random500": lab["random500"], "Top500": lab["top500"], "All": np.ones(len(y), bool)}

    oof = {m: oof_predict(feats[m], y, groups) for m in MODELS}
    panel = {q: {} for q in masks}
    for q, mk in masks.items():
        for m in MODELS:
            pred, proba, classes = oof[m]
            panel[q][m] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics(y, pred, proba, classes, mk).items()}

    # paired CI vs best teacher per (metric, queue)
    rng = np.random.default_rng(SEED)
    classes = oof["z_hcc"][2]
    pA, prA = oof["z_hcc"][0], oof["z_hcc"][1]
    contrasts = {}
    for q, mk in masks.items():
        contrasts[q] = {}
        for mname, fn in [("balanced_accuracy", _bacc), ("macro_f1", _mf1), ("macro_auc", _auc)]:
            best_t = max(P.TEACHERS, key=lambda t: panel[q][t][mname] or 0)
            pB, prB = oof[best_t][0], oof[best_t][1]
            md, ci, sig = paired_ci(fn, y, pA, prA, pB, prB, classes, mk, rng)
            contrasts[q][mname] = {"vs": best_t, "delta": md, "ci95": ci, "significant": sig,
                                   "z_hcc": panel[q]["z_hcc"][mname], "best_teacher": panel[q][best_t][mname]}

    out = {"panel": panel, "z_hcc_vs_best_teacher": contrasts}
    (RESULTS / "metric_panel.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    for q in masks:
        print(f"\n=== {q} (LR probe) ===")
        print(f"{'model':14}{'acc':>8}{'bal_acc':>9}{'macroF1':>9}{'macroAUC':>10}")
        for m in MODELS:
            d = panel[q][m]
            print(f"{m:14}{d['accuracy']:>8}{d['balanced_accuracy']:>9}{d['macro_f1']:>9}{str(d['macro_auc']):>10}")
        print("  z_hcc vs best teacher:")
        for mname, c in contrasts[q].items():
            print(f"    {mname:18} Δ={c['delta']:+.4f} {c['ci95']} vs {c['vs']} {'SIG' if c['significant'] else ''}")


if __name__ == "__main__":
    main()
