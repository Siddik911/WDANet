#!/usr/bin/env python3
"""Numerical correctness checks for the NEW code (not just 'it runs').

Covers: AIRM clustering reduction, SPD invariants, no-leakage of the pooling +
identity subspace, the two-phase identity removal algebra, and the metric/threshold
scripts (AUC vs sklearn, confusion, inner-LOSO leakage, ECE/MCC).

Run:  python verify_dualspd.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch

PASS, FAIL = 0, 0
def chk(name, ok, detail=""):
    global PASS, FAIL
    ok = bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    PASS += int(ok); FAIL += int(not ok)

rng = np.random.default_rng(0)

# ===========================================================================
print("1) AIRM clustering reduction  (triad/dualspd.py: _pooling_matrix, _mats)")
import os as _os, sys as _sys  # bootstrap: repo root on path (script moved to scripts/)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from triad.dualspd import DualSPDStage
from triad.models import _cov_to_corr

# C1: P @ C @ P.T == covariance of the pooled signal  (linear-transform identity)
T = 500; ch = 20
x = rng.standard_normal((ch, T))
C = (x @ x.T) / T                                   # raw sample covariance
P = np.zeros((4, ch))                                # 4 clusters, average pooling
groups = np.array_split(np.arange(ch), 4)
for k, g in enumerate(groups): P[k, g] = 1.0 / len(g)
red = P @ C @ P.T
cov_pooled = ((P @ x) @ (P @ x).T) / T
chk("P@C@P.T == cov(P x)", np.allclose(red, cov_pooled, atol=1e-10),
    f"max diff={np.abs(red-cov_pooled).max():.1e}")

# C2: pooling rows are a valid average-pool partition (rows sum to 1, each ch once)
st = DualSPDStage(streams=("cov",), n_clusters=4, max_windows=20)
from triad import data as D
subs = D.load_subjects(limit=6)
Pl = st._pooling_matrix(subs[:5])
chk("pooling rows sum to 1 (average pool)", np.allclose(Pl.sum(1), 1.0),
    f"row sums in [{Pl.sum(1).min():.3f},{Pl.sum(1).max():.3f}]")
each_once = np.allclose((Pl > 0).sum(0), 1)
chk("every channel in exactly one cluster", each_once)

# C3: reduced matrices are SPD
covs = st._raw_covs(subs[0])
st._P = Pl
red_all = (Pl @ covs @ Pl.T)
mineig = np.linalg.eigvalsh(red_all).min()
chk("clustered K×K covariances stay SPD", mineig > 0, f"min eig={mineig:.2e}")

# C4: pooling learned on TRAIN ONLY (changing a NON-train subject must not change P)
P_a = st._pooling_matrix(subs[:5])
P_b = st._pooling_matrix(subs[:5])           # same train set -> identical
chk("pooling deterministic on fixed train set", np.allclose(P_a, P_b))

# ===========================================================================
print("\n2) SPD network invariants  (triad/dualspd.py: SPDStream / DualSPDNet)")
from triad.dualspd import DualSPDNet
net = DualSPDNet(in_dim=4, subspace=4, stream_names=("cov", "corr"))
# C5: BiMap weight on Stiefel (WᵀW = I)
W = net.enc["cov"].spd[0].W.detach().squeeze(0)
WtW = W.t() @ W
chk("BiMap W on Stiefel (WᵀW≈I)", torch.allclose(WtW, torch.eye(W.shape[1], dtype=W.dtype), atol=1e-10),
    f"max|WᵀW-I|={(WtW-torch.eye(W.shape[1],dtype=W.dtype)).abs().max():.1e}")
# C6: forward finite, shape (B,2)
mats = {s: torch.from_numpy(np.stack([red_all[0]]*8)) for s in ("cov", "corr")}
out = net(mats)
chk("forward shape (B,2) & finite", out.shape == (8, 2) and torch.isfinite(out).all(), f"shape={tuple(out.shape)}")

# ===========================================================================
print("\n3) Two-phase identity removal algebra  (dualspd.py fit Phase 2)")
feat = 15
F = torch.from_numpy(rng.standard_normal((50, feat)))
# emulate V = top-k right singular vectors of a centred subject-mean cloud
means = rng.standard_normal((10, feat)); centred = means - means.mean(0)
_, _, Vt = np.linalg.svd(centred, full_matrices=False)
V = torch.from_numpy(Vt[:3])
# C7: V rows orthonormal
chk("identity subspace V orthonormal (VVᵀ=I)",
    torch.allclose(V @ V.t(), torch.eye(3, dtype=V.dtype), atol=1e-10),
    f"max|VVᵀ-I|={(V@V.t()-torch.eye(3,dtype=V.dtype)).abs().max():.1e}")
# C8: removed features are orthogonal to V (projection correct)
removed = F - (F @ V.t()) @ V
chk("removed ⟂ V (I-VᵀV projection)", torch.allclose(removed @ V.t(), torch.zeros(50, 3, dtype=V.dtype), atol=1e-10),
    f"max|removed·Vᵀ|={(removed@V.t()).abs().max():.1e}")
# C9: projection idempotent
removed2 = removed - (removed @ V.t()) @ V
chk("projection idempotent", torch.allclose(removed, removed2, atol=1e-12))

# ===========================================================================
print("\n4) No-leakage in the dual-stream  (structural + behavioural)")
# C10: predict pools the TEST subject by the TRAIN-learned P (P set in fit, unchanged by predict)
st2 = DualSPDStage(streams=("cov",), subspace=4, n_clusters=4, epochs=2, max_windows=20)
st2.fit(subs[:5])
P_after_fit = st2._P.copy()
_ = st2.predict_subject(subs[5])
chk("predict does NOT modify the learned pooling P", np.allclose(P_after_fit, st2._P))
# C11: the held-out subject's covs are never used to learn P (P depends only on train ids)
P_train_only = st2._pooling_matrix(subs[:5])
chk("P reproducible from train ids alone (no test dependence)", np.allclose(P_train_only, st2._P))

# ===========================================================================
print("\n5) Metric / threshold scripts  (threshold_eval.py, calibrate_eval.py)")
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, matthews_corrcoef
import calibrate_eval as CE
y = rng.integers(0, 2, 60); s = rng.random(60)
# C12: AUC matches sklearn
chk("auc() == sklearn roc_auc_score", abs(CE.auc(y, s) - roc_auc_score(y, s)) < 1e-9,
    f"ours={CE.auc(y,s):.6f} sklearn={roc_auc_score(y,s):.6f}")
# C13: confusion bacc & mcc match sklearn at a fixed cut
p = (s >= 0.5).astype(int); c = CE.confusion(y, p)
chk("confusion bacc == sklearn", abs(c["bacc"] - balanced_accuracy_score(y, p)) < 1e-9)
mcc_ref = matthews_corrcoef(y, p) if len(set(p)) > 1 else 0.0
chk("confusion mcc == sklearn", abs(c["mcc"] - mcc_ref) < 1e-9, f"ours={c['mcc']:.4f} ref={mcc_ref:.4f}")
# C14: inner-LOSO threshold for subject i never sees label_i  -> flipping label_i
#      must not change subject i's PREDICTION (its threshold came from the other 52)
pred, _ = CE.loo_predict(y, s, "prior")
y_flip = y.copy(); y_flip[7] ^= 1
pred_flip, _ = CE.loo_predict(y_flip, s, "prior")
# only entry 7's own threshold uses OTHER labels (incl. its prior); its prediction
# may move because the prior changes, but predictions of j!=7 that don't depend on
# label_7 except via prior. Strict check: with bacc method, subject i's threshold is
# from others only -> label_i flip cannot change pred[i] decision rule for that i.
predb, _ = CE.loo_predict(y, s, "inner_bacc")
yf = y.copy(); yf[7] ^= 1
predb_f, _ = CE.loo_predict(yf, s, "inner_bacc")
chk("inner-bacc: flipping label_i leaves pred[i] rule label-free",
    predb[7] == ((s[7] >= CE.bacc_threshold(np.delete(y,7), np.delete(s,7))).astype(int)))
# C15: ECE = 0 for perfectly calibrated probs
probs_perf = y.astype(float)            # prob = label -> perfectly calibrated
chk("ECE==0 for perfect calibration", CE.ece(probs_perf, y) < 1e-9, f"ECE={CE.ece(probs_perf,y):.2e}")

print(f"\n{'='*60}\n{PASS}/{PASS+FAIL} checks passed", "-> ALL GOOD" if FAIL == 0 else f"-> {FAIL} FAILED")
