"""Complement to diag_recenter.py: do the SAME class-axis-alignment test on the SPECTRAL
(Differential Entropy) features, to show WHY spectral transfers but connectivity doesn't.

If the depression class axis is better aligned across datasets in DE space than in
correlation space, that single fact explains: (a) nrde transfers, nr doesn't; (b) recentering
(which only re-aligns the correlation MEAN, not the axis) can't help.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import os as _os, sys as _sys  # bootstrap: repo root on path (script moved to scripts/)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from triad import config as C, data as D
from triad.models import ShallowStage

DATASETS = ["modma", "mumtaz", "opennuero"]

st = ShallowStage(rep="corr", n_nuisance=0, use_de=True)
DE, labels = {}, {}
for ds in DATASETS:
    C.set_active_dataset(ds)
    subs = D.load_subjects(harmonize=True)
    DE[ds] = np.stack([st._features(s)["de"].mean(0) for s in subs])   # subject-mean DE (85,)
    labels[ds] = np.array([s.label for s in subs])
    print(f"{ds:10s}: {len(subs)} subjects, DE {DE[ds].shape}")

# z-score on the pooled set so all datasets are commensurate (as nrde does)
scaler = StandardScaler().fit(np.concatenate([DE[ds] for ds in DATASETS]))
X = {ds: scaler.transform(DE[ds]) for ds in DATASETS}

print("\n-- SPECTRAL (DE) cross-dataset transfer AUC (rows=train, cols=test) --")
transfer = []
for A in DATASETS:
    clf = LogisticRegression(class_weight="balanced", max_iter=3000).fit(X[A], labels[A])
    row = []
    for B in DATASETS:
        auc = roc_auc_score(labels[B], clf.predict_proba(X[B])[:, 1])
        row.append(auc)
        if A != B:
            transfer.append(auc)
    print(f"  train {A:10s}: " + "  ".join(f"{b[:4]}={v:.3f}" for b, v in zip(DATASETS, row)))
print(f"  >> mean OFF-DIAGONAL (cross-dataset) transfer AUC = {np.mean(transfer):.4f}")

axes = {ds: (X[ds][labels[ds] == 1].mean(0) - X[ds][labels[ds] == 0].mean(0)) for ds in DATASETS}
print("-- SPECTRAL (DE) class-axis cosine alignment across datasets --")
cs = []
for i, a in enumerate(DATASETS):
    for b in DATASETS[i + 1:]:
        ca, cb = axes[a] / np.linalg.norm(axes[a]), axes[b] / np.linalg.norm(axes[b])
        c = float(ca @ cb)
        cs.append(c)
        print(f"  cos({a[:4]}, {b[:4]}) = {c:+.3f}")
print(f"  >> mean cross-dataset class-axis cosine = {np.mean(cs):+.4f}")
print("\n(compare: CONNECTIVITY/correlation mean cosine was +0.224, transfer AUC 0.527)")
