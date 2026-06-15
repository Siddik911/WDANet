#!/usr/bin/env python3
"""Honest, leakage-free accuracy from saved LOSO scores.

The LOSO harness stores per-subject scores in results/<stage>_*.json under
`per_fold` = [{sid,label,score,n_windows}, ...]. Reporting accuracy at a FIXED
0.5 cut understates a model whose scores are well-ranked (high AUC) but shifted;
reporting accuracy at the test-set Youden cut OVERSTATES it (threshold tuned on
test labels = leakage).

The fair operating point: pick the threshold on the *other 52* subjects only
(inner-LOSO), apply it to the held-out subject. No test label ever touches the
threshold. We report plain accuracy (to match WDANet's "Accuracy"), balanced
accuracy, sensitivity, specificity, F1.

Usage:
    python threshold_eval.py results/nr_k3.json results/cr_k2.json ...
    python threshold_eval.py            # defaults to a standard set
"""
import json
import sys
from pathlib import Path

import numpy as np


def _confusion(y, p):
    y = np.asarray(y); p = np.asarray(p)
    tp = int(((y == 1) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0     # recall on MDD (positive)
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    acc = (tp + tn) / len(y)                          # plain accuracy (WDANet metric)
    bacc = 0.5 * (sens + spec)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    return dict(acc=acc, bacc=bacc, sens=sens, spec=spec, f1=f1,
                tp=tp, tn=tn, fp=fp, fn=fn)


def _best_threshold(labels, scores, objective="youden"):
    """Threshold maximising the objective on (labels, scores)."""
    labels = np.asarray(labels); scores = np.asarray(scores)
    cand = np.unique(scores)
    # midpoints + open ends so every split is reachable
    mids = (cand[:-1] + cand[1:]) / 2 if len(cand) > 1 else cand
    grid = np.concatenate([[cand.min() - 1e-6], mids, [cand.max() + 1e-6]])
    best_t, best_v = 0.5, -np.inf
    for t in grid:
        p = (scores >= t).astype(int)
        c = _confusion(labels, p)
        v = (c["sens"] + c["spec"] - 1) if objective == "youden" else c["bacc"]
        if v > best_v:
            best_v, best_t = v, t
    return best_t


def inner_loso_predict(pf, objective="youden"):
    """Leave-one-subject-out threshold: cut chosen on the other N-1 subjects."""
    sids = [e["sid"] for e in pf]
    y = np.array([e["label"] for e in pf])
    s = np.array([e["score"] for e in pf])
    preds = np.zeros(len(pf), dtype=int)
    for i in range(len(pf)):
        mask = np.ones(len(pf), bool); mask[i] = False
        t = _best_threshold(y[mask], s[mask], objective)
        preds[i] = int(s[i] >= t)
    return y, preds


def prior_threshold_predict(pf):
    """Per-fold prior-matched cut: predicted-positive rate = training MDD prior.
    Threshold = the (1 - prior) quantile of the other N-1 subjects' scores."""
    y = np.array([e["label"] for e in pf])
    s = np.array([e["score"] for e in pf])
    preds = np.zeros(len(pf), dtype=int)
    for i in range(len(pf)):
        mask = np.ones(len(pf), bool); mask[i] = False
        prior = y[mask].mean()                       # fraction MDD in train fold
        t = np.quantile(s[mask], 1.0 - prior)
        preds[i] = int(s[i] >= t)
    return y, preds


def auc(pf):
    y = np.array([e["label"] for e in pf]); s = np.array([e["score"] for e in pf])
    # Mann-Whitney U / rank AUC
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n1 = (y == 1).sum(); n0 = (y == 0).sum()
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def report(path):
    d = json.load(open(path))
    pf = d["per_fold"]
    name = Path(path).stem
    a = auc(pf)
    # honest operating points (no test labels touch the threshold)
    yj, pj = inner_loso_predict(pf, "youden");   cj = _confusion(yj, pj)
    yb, pb = inner_loso_predict(pf, "bacc");      cb = _confusion(yb, pb)
    yp, pp = prior_threshold_predict(pf);          cp = _confusion(yp, pp)
    # default + (leaky) global-Youden, for reference
    y = np.array([e["label"] for e in pf]); s = np.array([e["score"] for e in pf])
    c05 = _confusion(y, (s >= 0.5).astype(int))
    t_leak = _best_threshold(y, s, "youden"); cleak = _confusion(y, (s >= t_leak).astype(int))
    print(f"\n=== {name}   (AUC {a:.4f}) ===")
    hdr = f"{'operating point':<28}{'acc':>7}{'bacc':>7}{'sens':>7}{'spec':>7}{'f1':>7}"
    print(hdr); print("-" * len(hdr))
    rows = [
        ("default cut 0.5", c05),
        ("inner-LOSO Youden  (HONEST)", cj),
        ("inner-LOSO bal-acc (HONEST)", cb),
        ("inner-LOSO prior   (HONEST)", cp),
        ("global Youden  (LEAKY ref)", cleak),
    ]
    for label, c in rows:
        print(f"{label:<28}{c['acc']:>7.3f}{c['bacc']:>7.3f}"
              f"{c['sens']:>7.3f}{c['spec']:>7.3f}{c['f1']:>7.3f}")
    return name, a, cj, cb, cp


if __name__ == "__main__":
    args = sys.argv[1:] or [
        "results/s2_loso.json",
        "results/nr_k3.json", "results/nr_k5.json",
        "results/cr_k1.json", "results/cr_k2.json", "results/cr_k3.json",
    ]
    print("WDANet MODMA (Case 5): Accuracy 70.94 | Sens 81.88 | Spec 60.00 | F1 73.98")
    print("(their cut = softmax argmax ~ a single fixed operating point)")
    best = []
    for p in args:
        if Path(p).exists():
            best.append(report(p))
    # leaderboard on the honest inner-LOSO Youden accuracy
    print("\n\n#### HONEST accuracy leaderboard (inner-LOSO Youden) vs WDANet 70.94 ####")
    best.sort(key=lambda r: r[2]["acc"], reverse=True)
    for name, a, cj, cb, cp in best:
        flag = "  <-- beats WDANet" if cj["acc"] > 0.7094 else ""
        print(f"  {name:<12} AUC {a:.4f} | acc {cj['acc']*100:5.2f} | "
              f"bacc {cj['bacc']*100:5.2f} | sens {cj['sens']*100:5.1f} | "
              f"spec {cj['spec']*100:5.1f}{flag}")
