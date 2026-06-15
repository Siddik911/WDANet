#!/usr/bin/env python3
"""Issue #1 audit: do the REMOVED 'identity' PCs actually carry diagnosis?

The nr / ds identity-removal assumes the top-K between-subject PCs = identity, not
diagnosis. Univariate PC-AUC ≈ 0.5 is NECESSARY but NOT SUFFICIENT — a subspace can
carry class info multivariately while each axis looks null. So for the removed
subspace V_K (top-K PCs of the subject-mean cloud) we test, for K = 0..10:

  * univariate  : LOO-AUC of each removed PC               (what we had)
  * multivariate: leave-one-subject-out logistic-regression AUC on the K-dim
                  projection onto V_K                       (the reviewer's ask)
  * permutation : p-value of that multivariate AUC vs label shuffles

If the multivariate LR-AUC on the removed subspace is ~0.5 with p>0.05, removing it
does NOT discard diagnosis. If it is high with small p, identity removal at that K
is deleting class signal — and we should stop before it.

Diagnostic only (uses all 53 subjects to characterise the SUBSPACE structure — a
property of the data, not a performance claim).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.linear_model import LogisticRegression
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space

import os as _os, sys as _sys  # bootstrap: repo root on path (script moved to scripts/)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from triad import data as D
from triad.models import ShallowStage


def loo_auc(scores, y):
    o = np.argsort(scores, kind="mergesort"); r = np.empty(len(scores)); r[o] = np.arange(1, len(scores)+1)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    return (r[y == 1].sum() - n1*(n1+1)/2) / (n1*n0) if n1 and n0 else 0.5


def loo_lr_auc(X, y):
    """Leave-one-subject-out logistic-regression AUC on a low-dim projection."""
    s = np.zeros(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        if len(set(y[m].tolist())) < 2:
            s[i] = 0.5; continue
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(X[m], y[m])
        s[i] = lr.predict_proba(X[i:i+1])[0, 1]
    return loo_auc(s, y)


def main():
    subs = D.load_subjects()
    print(f"{len(subs)} subjects — computing correlation tangent features ...")
    st = ShallowStage(rep="corr", max_windows=120)
    feats = [st._features(s) for s in subs]
    ref = mean_riemann(np.stack([f["mean"] for f in feats]))
    Xs = [tangent_space(f["mats"], ref) for f in feats]
    subj_means = np.stack([x.mean(axis=0) for x in Xs])           # (53, d)
    y = np.array([f["label"] for f in feats])
    centred = subj_means - subj_means.mean(axis=0)
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)        # rows = PCs

    rng = np.random.default_rng(0); n_perm = 500
    print(f"\nremoved-subspace class audit  (d={subj_means.shape[1]}, {len(y)} subjects, "
          f"{n_perm} perms)\n")
    print(f"{'K':>2} {'uni PC-AUCs (each removed PC)':<34} {'multivar LR-AUC':>15} {'perm p':>8}  verdict")
    print("-" * 80)
    for K in range(1, 11):
        V = Vt[:K]
        proj = subj_means @ V.T                                   # (53, K) removed component
        uni = [round(float(loo_auc(proj[:, j], y)), 2) for j in range(K)]
        mv = loo_lr_auc(proj, y)
        perm = np.array([loo_lr_auc(proj, rng.permutation(y)) for _ in range(n_perm)])
        p = (1 + int((perm >= mv).sum())) / (1 + n_perm)
        verdict = ("IDENTITY (safe to remove)" if p > 0.05
                   else "carries DIAGNOSIS — removing hurts")
        ut = str(uni) if K <= 6 else f"[{uni[0]}..{uni[-1]}] ({K} PCs)"
        print(f"{K:>2} {ut:<34} {mv:>15.3f} {p:>8.3f}  {verdict}")

    print("\nRead: nr uses K=3. If K≤3 says IDENTITY (p>0.05) and a larger K flips to "
          "DIAGNOSIS, that\nlocates exactly where removal starts eating class signal — "
          "the honest basis for choosing K.")


if __name__ == "__main__":
    main()
