#!/usr/bin/env python3
"""Transductive Riemannian Procrustes Analysis (RPA) for LODO depression transfer.

Tests the one literature-backed lever that directly attacks this project's diagnosed
root cause: the MDD/HC *connectivity* axis is near-orthogonal across datasets (a
ROTATION the per-dataset recentering could not fix). RPA = recenter (TLCenter) +
rotate source class-means onto target class-means (TLRotate, Rodrigues 2019). The
target is unlabeled (LODO), so its class means come from TRANSDUCTIVE pseudo-labels.

Subject-level: each subject = Riemannian mean of its window correlation matrices.
Compares, per LODO fold:
  conn-baseline : tangent+LR on pooled source, no alignment  (= `nr` connectivity)
  conn-RPA      : TLCenter + pseudo-label + TLRotate, then tangent+LR
  de            : differential-entropy stream alone (the part that already transfers)
  nrde-RPA      : decision-level fusion of conn-RPA + de
"""
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pyriemann.estimation import Covariances
from pyriemann.utils.tangentspace import tangent_space
from pyriemann.utils.mean import mean_riemann
from pyriemann.transfer import TLCenter, TLRotate, encode_domains

import os as _os, sys as _sys  # bootstrap: repo root on path (script moved to scripts/)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from triad import data as D, config as C
from triad.models import _cov_to_corr, _compute_de
from triad.data import epoch

MAXW = 100
ALL = ["modma", "mumtaz", "opennuero"]


def _rank_auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else 0.5


def subj_features(sub):
    """One subject -> (mean correlation matrix 17x17, mean DE vector)."""
    w = epoch(sub.data, max_windows=MAXW)
    covs = Covariances(estimator="lwf").transform(w).astype(np.float64)
    corr = _cov_to_corr(covs)
    M = mean_riemann(corr)
    de = _compute_de(w).mean(axis=0)
    return M, de


def load_all():
    out = {}
    for ds in ALL:
        C.set_active_dataset(ds)
        subs = D.load_subjects(harmonize=True)
        rows = [(s.sid, s.label, *subj_features(s)) for s in subs]
        out[ds] = rows
        print(f"  [{ds}] {len(rows)} subjects", flush=True)
    return out


def tan(M, ref):
    return tangent_space(M, ref)


def fit_lr(X, y):
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                              solver="lbfgs", random_state=42).fit(X, y)


def run(target, n_iter=1):
    sources = [d for d in ALL if d != target]
    data = load_all()
    # --- assemble subject-level arrays ---
    Ms = np.array([r[2] for d in sources for r in data[d]])          # source matrices
    ys = np.array([r[1] for d in sources for r in data[d]])
    doms = np.array([d for d in sources for r in data[d]])
    DEs = np.array([r[3] for d in sources for r in data[d]])
    Mt = np.array([r[2] for r in data[target]])                     # target matrices
    yt = np.array([r[1] for r in data[target]])
    DEt = np.array([r[3] for r in data[target]])
    domt = np.array([target] * len(Mt))

    # ===== connectivity baseline (no alignment) =====
    ref = mean_riemann(Ms)
    lr = fit_lr(tan(Ms, ref), ys)
    s_base = lr.predict_proba(tan(Mt, ref))[:, 1]
    auc_base = _rank_auc(yt, s_base)

    # ===== DE stream (already transfers) =====
    sc = StandardScaler().fit(DEs)
    lr_de = fit_lr(sc.transform(DEs), ys)
    s_de = lr_de.predict_proba(sc.transform(DEt))[:, 1]
    auc_de = _rank_auc(yt, s_de)

    # ===== RPA: TLCenter + transductive TLRotate =====
    X_all = np.concatenate([Ms, Mt])
    dom_all = np.concatenate([doms, domt])
    # 1) recenter each domain to identity (labels unused by TLCenter)
    y_place = np.concatenate([ys, np.zeros(len(Mt), int)])
    _, ye = encode_domains(X_all, y_place, dom_all)
    Xc = TLCenter(target_domain=target, metric="riemann").fit_transform(X_all, ye)
    Xc_s, Xc_t = Xc[:len(Ms)], Xc[len(Ms):]
    I = np.eye(Ms.shape[-1])

    # 2) transductive pseudo-labels for the target from centered source
    lr_c = fit_lr(tan(Xc_s, I), ys)
    pt = (lr_c.predict_proba(tan(Xc_t, I))[:, 1] >= 0.5).astype(int)

    auc_rpa = auc_base
    for it in range(n_iter):
        # 3) rotate sources to match target class means (pseudo-labeled)
        y_rot = np.concatenate([ys, pt])
        _, yer = encode_domains(Xc, y_rot, dom_all)
        try:
            Xr = TLRotate(target_domain=target, metric="euclid").fit_transform(Xc, yer)
        except Exception as e:
            print(f"    [TLRotate failed: {e!r}]"); break
        Xr_s, Xr_t = Xr[:len(Ms)], Xr[len(Ms):]
        # 4) retrain on rotated source, score target
        lr_r = fit_lr(tan(Xr_s, I), ys)
        s_rpa = lr_r.predict_proba(tan(Xr_t, I))[:, 1]
        auc_rpa = _rank_auc(yt, s_rpa)
        new_pt = (s_rpa >= 0.5).astype(int)
        print(f"    iter {it+1}: conn-RPA AUC={auc_rpa:.3f}  "
              f"(pseudo-label flips={int((new_pt!=pt).sum())})", flush=True)
        if (new_pt == pt).all():
            break
        pt = new_pt

    # ===== fusion: conn-RPA + DE =====
    auc_fuse = _rank_auc(yt, (s_rpa + s_de) / 2.0) if 's_rpa' in dir() else auc_base
    nrde_fuse = _rank_auc(yt, (s_base + s_de) / 2.0)  # no-RPA fusion (= nrde baseline)

    print(f"\n  target={target:10s}  conn-base={auc_base:.3f}  conn-RPA={auc_rpa:.3f}  "
          f"de={auc_de:.3f}  nrde(noRPA)={nrde_fuse:.3f}  nrde-RPA={auc_fuse:.3f}")
    return dict(target=target, conn_base=auc_base, conn_rpa=auc_rpa, de=auc_de,
                nrde_norpa=nrde_fuse, nrde_rpa=auc_fuse)


if __name__ == "__main__":
    targets = sys.argv[1:] or ALL
    print("=== Transductive RPA LODO (subject-level, 17-ch harmonized) ===")
    res = [run(t) for t in targets]
    print("\n#### SUMMARY (subject-level AUC) ####")
    print(f"{'target':10s} {'conn-base':>9} {'conn-RPA':>9} {'de':>6} "
          f"{'nrde-noRPA':>10} {'nrde-RPA':>9}")
    for r in res:
        print(f"{r['target']:10s} {r['conn_base']:>9.3f} {r['conn_rpa']:>9.3f} "
              f"{r['de']:>6.3f} {r['nrde_norpa']:>10.3f} {r['nrde_rpa']:>9.3f}")
    if len(res) > 1:
        import numpy as _np
        for k in ['conn_base', 'conn_rpa', 'de', 'nrde_norpa', 'nrde_rpa']:
            print(f"  mean {k:12s} = {_np.mean([r[k] for r in res]):.3f}")
