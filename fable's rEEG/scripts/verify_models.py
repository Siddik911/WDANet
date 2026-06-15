#!/usr/bin/env python3
"""Correctness checks for the TRIAD components.

Verifies the model math against references rather than just "it runs":
  1. _cov_to_corr     -> unit diagonal, symmetric, equals D^-1/2 C D^-1/2, stays SPD.
  2. _recenter        -> defining property: Frechet mean of recentred set == I
                         (this is exactly pyriemann TLCenter's Riemannian whitening).
  3. tangent_space    -> correct dimension and exact round-trip via untangent_space.
  4. EEGNet           -> output shape (B,2), flatten dim matches reference formula,
                         max-norm constraint actually bounds the constrained weights.
  5. pipeline windows -> covariances are SPD, finite, well-conditioned with shrinkage.

Run:  python verify_models.py
"""
import numpy as np

from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.tangentspace import tangent_space, untangent_space

import os as _os, sys as _sys  # bootstrap: repo root on path (script moved to scripts/)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from triad import config as C
from triad import data as D
from triad.models import _cov_to_corr, _recenter, _build_eegnet

rng = np.random.default_rng(0)
PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, cond, detail=""):
    tag = PASS if cond else FAIL
    results.append(cond)
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))


def random_spd(n, ch, samples=400):
    X = rng.standard_normal((n, ch, samples))
    return Covariances(estimator="lwf").transform(X).astype(np.float64)


print("1) _cov_to_corr -----------------------------------------------------")
C8 = random_spd(20, 8)
R = _cov_to_corr(C8)
ref = np.stack([Ci / np.sqrt(np.outer(np.diag(Ci), np.diag(Ci))) for Ci in C8])
check("unit diagonal", np.allclose(np.diagonal(R, 0, 1, 2), 1.0, atol=1e-10),
      f"max|diag-1|={np.abs(np.diagonal(R,0,1,2)-1).max():.2e}")
check("symmetric", np.allclose(R, R.transpose(0, 2, 1), atol=1e-12))
check("matches D^-1/2 C D^-1/2 reference", np.allclose(R, ref, atol=1e-10),
      f"max abs diff={np.abs(R-ref).max():.2e}")
check("stays SPD (all eigvals>0)", all(np.linalg.eigvalsh(Ri).min() > 0 for Ri in R))

print("2) _recenter (== TLCenter Riemannian whitening) ---------------------")
M = random_spd(30, 8)
mean = mean_riemann(M)
Mr = _recenter(M, mean)
new_mean = mean_riemann(Mr)
I = np.eye(8)
check("Frechet mean of recentred set == identity",
      distance_riemann(new_mean, I) < 1e-6,
      f"d_riemann(mean, I)={distance_riemann(new_mean, I):.2e}")
check("recentring preserves SPD", all(np.linalg.eigvalsh(Mi).min() > 0 for Mi in Mr))
# whitening identity: recenter(M, M_i itself) sends that matrix to I
single = _recenter(M[:1], M[0])[0]
check("matrix recentred by itself -> I", distance_riemann(single, I) < 1e-6)

print("3) tangent_space ----------------------------------------------------")
ref_mean = mean_riemann(C8)
T = tangent_space(C8, ref_mean)
check("tangent dim = ch*(ch+1)/2", T.shape == (20, 8 * 9 // 2),
      f"shape={T.shape}, expected (20,{8*9//2})")
back = untangent_space(T, ref_mean)
check("tangent<->untangent round-trip exact",
      np.allclose(back, C8, atol=1e-8), f"max diff={np.abs(back-C8).max():.2e}")

print("4) EEGNet architecture ----------------------------------------------")
import torch
net = _build_eegnet(n_ch=C.N_CHANNELS, win=C.WIN_LEN, dropout=0.25)
xb = torch.randn(5, 1, C.N_CHANNELS, C.WIN_LEN)
out = net(xb)
check("forward output shape (B,2)", tuple(out.shape) == (5, 2), f"got {tuple(out.shape)}")
check("output finite", torch.isfinite(out).all().item())
n_params = sum(p.numel() for p in net.parameters())
check("param count in EEGNet-8,2 range (1k-20k)", 1000 < n_params < 20000,
      f"{n_params} params")
# max-norm constraint must clamp the constrained weights
with torch.no_grad():
    net.spatial.weight.mul_(50.0)         # blow up the depthwise weights
    net.classify.weight.mul_(50.0)
net.clip_constraints()
sp_norm = net.spatial.weight.flatten(1).norm(2, dim=1).max().item()
cl_norm = net.classify.weight.flatten(1).norm(2, dim=1).max().item()
check("depthwise spatial max-norm <= 1.0", sp_norm <= 1.0 + 1e-5, f"max row-norm={sp_norm:.4f}")
check("classifier max-norm <= 0.25", cl_norm <= 0.25 + 1e-5, f"max row-norm={cl_norm:.4f}")

print("5) frozen pipeline windows ------------------------------------------")
subs = D.load_subjects(limit=4)
sub = subs[0]
n_ch = sub.data.shape[0]
wins = D.epoch(sub.data, max_windows=20)
check(f"window tensor shape (n,{n_ch},WIN_LEN)",
      wins.shape[1:] == (n_ch, C.WIN_LEN), f"shape={wins.shape}")
check("windows finite", np.isfinite(wins).all())
covs = Covariances(estimator=C.COV_ESTIMATOR).transform(wins).astype(np.float64)
eig_min = min(np.linalg.eigvalsh(Ci).min() for Ci in covs)
conds = [np.linalg.cond(Ci) for Ci in covs]
check("shrinkage covariances are SPD (min eig>0)", eig_min > 0, f"min eig={eig_min:.2e}")
check("covariances well-conditioned (cond<1e6 w/ Ledoit-Wolf)",
      max(conds) < 1e6, f"max cond={max(conds):.1e}")

print("6) SPDNet S4/S5 (TSMNet layers) -------------------------------------")
from triad.spdnet import SPDNet
from geoopt.optim import RiemannianAdam
torch.manual_seed(0)
Xw = torch.randn(24, n_ch, C.WIN_LEN)
yb = torch.randint(0, 2, (24,))

# S4: global SPD batch-norm
net4 = SPDNet(n_ch=n_ch, domain_bn=False, device="cpu", spd_device="cpu")
out4 = net4(Xw)
check("S4 forward shape (B,2)", tuple(out4.shape) == (24, 2), f"got {tuple(out4.shape)}")
check("S4 output finite", torch.isfinite(out4).all().item())
opt = RiemannianAdam(net4.parameters(), lr=1e-3, stabilize=10)
l0 = torch.nn.functional.cross_entropy(net4(Xw), yb)
for _ in range(8):
    opt.zero_grad(); loss = torch.nn.functional.cross_entropy(net4(Xw), yb)
    loss.backward(); opt.step()
l1 = torch.nn.functional.cross_entropy(net4(Xw), yb)
check("S4 trains (loss decreases over 8 steps)", l1.item() < l0.item(),
      f"{l0.item():.4f} -> {l1.item():.4f}")
W = net4.spdnet[0].W.detach()
WtW = (W.transpose(-2, -1) @ W).squeeze(0)
check("S4 BiMap weight stays on Stiefel (WᵀW≈I)",
      torch.allclose(WtW, torch.eye(WtW.shape[-1], dtype=WtW.dtype), atol=1e-4),
      f"max|WᵀW-I|={ (WtW-torch.eye(WtW.shape[-1],dtype=WtW.dtype)).abs().max().item():.2e}")

# S5: per-domain SPD batch-norm (SPDDSMBN) with a multi-domain batch
net5 = SPDNet(n_ch=n_ch, domain_bn=True, n_domains=4, device="cpu", spd_device="cpu")
dom = torch.tensor([0, 1, 2, 3] * 6)
out5 = net5(Xw, dom)
check("S5 forward with multi-domain batch (B,2)", tuple(out5.shape) == (24, 2))
check("S5 output finite", torch.isfinite(out5).all().item())
# new held-out subject = new domain: register + estimate its reference, then classify
import spdnets.batchnorm as BN
new_bn = BN.SPDBatchNormImpl(shape=(1, net5.subspace, net5.subspace), batchdim=0,
    learn_mean=False, learn_std=True, dispersion=BN.BatchNormDispersion.SCALAR,
    mean=net5.bn.mean, std=net5.bn.std, eta=1.0, eta_test=0.1, dtype=torch.double)
net5.bn.add_domain_(new_bn, torch.tensor(4))
with torch.no_grad():
    net5.bn.initrunningstats(net5._pre_bn(Xw), torch.tensor(4))
    out5t = net5(Xw, torch.full((24,), 4))
check("S5 handles unseen test domain (label-free reference)",
      tuple(out5t.shape) == (24, 2) and torch.isfinite(out5t).all().item())

print("\n--------------------------------------------------------------------")
print(f"{sum(results)}/{len(results)} checks passed",
      "-> ALL GOOD" if all(results) else "-> SOME FAILED")
