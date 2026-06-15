"""Fuse LODO models (nrde spectral backbone + deep ds + SPD-DANN) and report AUC and
balanced accuracy per fold and on average. Reads whatever per-fold result JSONs exist.

Score fusion = mean of per-subject P(MDD) across the chosen models (matched by sid).
We report AUC (threshold-free, primary) and balanced accuracy at thr=0.5 and at the
per-fold Youden threshold (optimistic upper bound on accuracy).
"""
import json
import itertools
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

FOLDS = ["modma", "mumtaz", "opennuero"]
MODELS = {
    "nrde": "results/lodo_{f}_nrde_harm.json",
    "nrdestd": "results/lodo_{f}_nrde_harm_destd.json",
    "ds": "results/lodo_{f}_ds_harm.json",
    "spddannC": "results/lodo_{f}_spddannC_harm.json",   # connectivity-only SPD-DANN
    "spddann": "results/lodo_{f}_spddann_harm.json",      # spectral-augmented SPD-DANN
}


def load(model, fold):
    try:
        d = json.load(open(MODELS[model].format(f=fold)))
        return {p["sid"]: (p["label"], p["score"]) for p in d["per_fold"]}
    except FileNotFoundError:
        return None


def youden_bacc(y, s):
    order = np.argsort(-s)
    ys = y[order]
    P, N = (y == 1).sum(), (y == 0).sum()
    tp, fp = np.cumsum(ys == 1), np.cumsum(ys == 0)
    j = tp / max(P, 1) - fp / max(N, 1)
    thr = s[order][int(np.argmax(j))]
    return balanced_accuracy_score(y, (s >= thr).astype(int))


def evaluate(models):
    aucs, bacc05, baccY = [], [], []
    per = {}
    for fold in FOLDS:
        loaded = [load(m, fold) for m in models]
        if any(l is None for l in loaded):
            return None
        sids = sorted(set.intersection(*[set(l) for l in loaded]))
        y = np.array([loaded[0][s][0] for s in sids])
        scores = np.mean([[l[s][1] for s in sids] for l in loaded], axis=0)
        a = roc_auc_score(y, scores)
        b0 = balanced_accuracy_score(y, (scores >= 0.5).astype(int))
        bY = youden_bacc(y, scores)
        aucs.append(a); bacc05.append(b0); baccY.append(bY)
        per[fold] = (a, b0, bY)
    return per, np.mean(aucs), np.mean(bacc05), np.mean(baccY)


combos = [("nrde",), ("ds",), ("spddannC",), ("spddann",),
          ("nrde", "spddann"), ("nrde", "ds"), ("spddann", "ds"),
          ("nrde", "spddann", "ds"), ("nrde", "spddannC")]

print(f"{'config':22s} {'AUC':>7s} | {'mod':>6s} {'mum':>6s} {'opn':>6s} | "
      f"{'bAcc.5':>7s} {'bAccY':>7s}")
print("-" * 72)
for combo in combos:
    res = evaluate(list(combo))
    if res is None:
        print(f"{'+'.join(combo):22s}  (missing results)")
        continue
    per, mauc, mb0, mbY = res
    pf = " ".join(f"{per[f][0]:6.3f}" for f in FOLDS)
    print(f"{'+'.join(combo):22s} {mauc:7.4f} | {pf} | {mb0:7.4f} {mbY:7.4f}")
print("-" * 72)
print("AUC primary (threshold-free). bAcc.5 = balanced acc @0.5; bAccY = @Youden (optimistic).")
print("goal: mean AUC > 0.70")
