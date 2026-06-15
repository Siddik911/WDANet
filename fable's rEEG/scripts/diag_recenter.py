"""Diagnose WHY per-dataset Riemannian recentering hurts LODO.

Hypothesis under test: recentering each dataset to its own Frechet mean changes the
*direction* of the MDD/HC class axis differently per dataset, so a classifier trained on
one dataset points the wrong way on another (cross-dataset class-axis MISALIGNMENT).

We work at the subject-mean correlation representation (one matrix per subject) — a clean,
fast proxy for the LODO transfer. For recenter in {off, on} we measure:
  (1) cross-dataset transfer AUC: train LR on dataset A's subject vectors, score dataset B;
  (2) cosine alignment of each dataset's class axis (mean_MDD - mean_HC) in the tangent space;
  (3) within-dataset class separation (self AUC) — does recentering wash out class signal?
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import os as _os, sys as _sys  # bootstrap: repo root on path (script moved to scripts/)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from triad import config as C, data as D
from triad.models import ShallowStage, _recenter
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space

DATASETS = ["modma", "mumtaz", "opennuero"]

# --- load harmonized per-subject mean correlation matrices ---
st = ShallowStage(rep="corr", n_nuisance=0)
means, labels = {}, {}
for ds in DATASETS:
    C.set_active_dataset(ds)
    subs = D.load_subjects(harmonize=True)
    means[ds] = np.stack([st._features(s)["mean"] for s in subs])
    labels[ds] = np.array([s.label for s in subs])
    print(f"{ds:10s}: {len(subs)} subjects, mats {means[ds].shape}")


def tangent_by_dataset(recenter: bool):
    if recenter:
        dmean = {ds: mean_riemann(means[ds]) for ds in DATASETS}
        rc = {ds: _recenter(means[ds], dmean[ds]) for ds in DATASETS}
    else:
        rc = {ds: means[ds] for ds in DATASETS}
    ref = mean_riemann(np.concatenate([rc[ds] for ds in DATASETS]))
    return {ds: (tangent_space(rc[ds], ref), labels[ds]) for ds in DATASETS}, rc


for recenter in [False, True]:
    P, rc = tangent_by_dataset(recenter)
    print("\n" + "=" * 60)
    print(f"RECENTER = {recenter}")
    print("-- cross-dataset transfer AUC (rows=train, cols=test) --")
    transfer = []
    for A in DATASETS:
        XA, yA = P[A]
        clf = LogisticRegression(class_weight="balanced", max_iter=3000).fit(XA, yA)
        row = []
        for B in DATASETS:
            XB, yB = P[B]
            auc = roc_auc_score(yB, clf.predict_proba(XB)[:, 1])
            row.append(auc)
            if A != B:
                transfer.append(auc)
        print(f"  train {A:10s}: " + "  ".join(f"{b[:4]}={v:.3f}" for b, v in zip(DATASETS, row)))
    print(f"  >> mean OFF-DIAGONAL (cross-dataset) transfer AUC = {np.mean(transfer):.4f}")

    # class axes + cosine alignment
    axes = {ds: (P[ds][0][P[ds][1] == 1].mean(0) - P[ds][0][P[ds][1] == 0].mean(0))
            for ds in DATASETS}
    print("-- class-axis cosine alignment across datasets --")
    cs = []
    for i, a in enumerate(DATASETS):
        for b in DATASETS[i + 1:]:
            ca, cb = axes[a] / np.linalg.norm(axes[a]), axes[b] / np.linalg.norm(axes[b])
            c = float(ca @ cb)
            cs.append(c)
            print(f"  cos({a[:4]}, {b[:4]}) = {c:+.3f}")
    print(f"  >> mean cross-dataset class-axis cosine = {np.mean(cs):+.4f}")
