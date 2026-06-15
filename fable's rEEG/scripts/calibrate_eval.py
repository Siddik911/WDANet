#!/usr/bin/env python3
"""Leakage-free operating point + calibration for saved LOSO scores.

We never promise an accuracy. We compute it honestly:

    for each held-out subject i (of the 53):
        fit threshold / calibrator on the OTHER 52 subjects' scores+labels only
        apply it to subject i
    aggregate the 53 held-out decisions -> balanced acc / acc / sens / spec / MCC

The held-out subject's LABEL never touches its own threshold. (Caveat: the 52
scores are out-of-fold scores from models that each saw subject i's *data*; a
*fully* nested version would re-train inner models that never see i at all — that
needs re-running LOSO, not just post-processing. Stated, not hidden.)

Three leakage-free operating points, plus calibration:
  * prior-matched   : cut so predicted-positive rate = the 52-subject MDD prior
  * inner-bacc      : cut that maximises balanced accuracy on the 52
  * Platt           : logistic calibration P=sigmoid(a*score+b) fit on the 52,
                      decide at 0.5 (also yields calibrated probs -> ECE)

For reference only (NOT honest): default 0.5, and test-set Youden (leaky).
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


def confusion(y, p):
    y = np.asarray(y); p = np.asarray(p)
    tp = int(((y == 1) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    denom = np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    mcc = (tp*tn - fp*fn) / denom if denom > 0 else 0.0
    return dict(acc=(tp+tn)/len(y), bacc=0.5*(sens+spec), sens=sens, spec=spec,
                f1=f1, mcc=mcc, tp=tp, tn=tn, fp=fp, fn=fn)


def bacc_threshold(y, s):
    """Threshold maximising balanced accuracy on (y, s)."""
    cand = np.unique(s)
    grid = np.concatenate([[cand.min() - 1e-9],
                           (cand[:-1] + cand[1:]) / 2 if len(cand) > 1 else cand,
                           [cand.max() + 1e-9]])
    best_t, best_v = 0.5, -1.0
    for t in grid:
        c = confusion(y, (s >= t).astype(int))
        if c["bacc"] > best_v:
            best_v, best_t = c["bacc"], t
    return best_t


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    order = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[order] = np.arange(1, len(s)+1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, r); r = (sums/cnt)[inv]
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    return (r[y == 1].sum() - n1*(n1+1)/2) / (n1*n0) if n1 and n0 else 0.5


def ece(probs, y, bins=10):
    probs = np.asarray(probs); y = np.asarray(y)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for b in range(bins):
        m = (probs > edges[b]) & (probs <= edges[b+1]) if b else (probs >= 0) & (probs <= edges[1])
        if m.sum() == 0:
            continue
        e += (m.sum()/len(y)) * abs(y[m].mean() - probs[m].mean())
    return e


def loo_predict(y, s, method):
    """Leave-one-subject-out: fit operating point on the other N-1, apply to i."""
    y = np.asarray(y); s = np.asarray(s); n = len(y)
    pred = np.zeros(n, int); cal = np.full(n, np.nan)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        ytr, str_ = y[m], s[m]
        if method == "prior":
            t = np.quantile(str_, 1.0 - ytr.mean())
            pred[i] = int(s[i] >= t)
        elif method == "inner_bacc":
            pred[i] = int(s[i] >= bacc_threshold(ytr, str_))
        elif method == "platt":
            lr = LogisticRegression(C=1e6, solver="lbfgs").fit(str_[:, None], ytr)
            cal[i] = lr.predict_proba(s[i:i+1, None])[0, 1]
            pred[i] = int(cal[i] >= 0.5)
    return pred, cal


def report(path):
    d = json.load(open(path))
    pf = d["per_fold"]
    y = np.array([e["label"] for e in pf]); s = np.array([e["score"] for e in pf])
    name = Path(path).stem
    a = auc(y, s)
    print(f"\n=== {name}   (AUC {a:.4f}, threshold-free) ===")
    hdr = f"{'operating point':<34}{'acc':>7}{'bacc':>7}{'sens':>7}{'spec':>7}{'mcc':>7}"
    print(hdr); print("-" * len(hdr))

    # honest, leakage-free
    rows = []
    for label, meth in [("prior-matched  (honest)", "prior"),
                        ("inner-bacc     (honest)", "inner_bacc"),
                        ("Platt-calibrated (honest)", "platt")]:
        pred, cal = loo_predict(y, s, meth)
        c = confusion(y, pred)
        extra = f"  ECE={ece(cal, y):.3f}" if meth == "platt" else ""
        rows.append((label, c, extra))
    # reference only (not honest)
    c05 = confusion(y, (s >= 0.5).astype(int))
    t_leak = bacc_threshold(y, s); cleak = confusion(y, (s >= t_leak).astype(int))
    rows += [("default 0.5  (ref)", c05, ""),
             ("test-set best  (LEAKY ref)", cleak, "")]
    for label, c, extra in rows:
        print(f"{label:<34}{c['acc']:>7.3f}{c['bacc']:>7.3f}"
              f"{c['sens']:>7.3f}{c['spec']:>7.3f}{c['mcc']:>7.3f}{extra}")
    return name, a, rows[0][1], rows[1][1], rows[2][1]


if __name__ == "__main__":
    args = sys.argv[1:] or ["results/nr_k3.json", "results/nr_k5.json",
                            "results/s2_loso.json"]
    print("Honest = threshold/calibrator fit on the OTHER 52 subjects, applied to the held-out one.")
    print("WDANet MODMA reference: accuracy 70.94%  (their own protocol).")
    best = []
    for p in args:
        if Path(p).exists():
            best.append(report(p))
        else:
            print(f"  (skip {p} — not found)")
    print("\n#### honest leaderboard — prior-matched balanced accuracy ####")
    for name, a, prior, ibacc, platt in sorted(best, key=lambda r: r[2]["bacc"], reverse=True):
        print(f"  {name:<12} AUC {a:.4f} | prior bacc {prior['bacc']*100:5.2f} "
              f"acc {prior['acc']*100:5.2f} | Platt bacc {platt['bacc']*100:5.2f} "
              f"acc {platt['acc']*100:5.2f}")
