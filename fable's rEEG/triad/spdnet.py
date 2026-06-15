"""S4 / S5 — deep SPD-manifold encoder, built on the TSMNet reference layers.

Architecture (Kobler et al. 2022, TSMNet), identical for S4 and S5 except the SPD
batch-norm:

    window (B, n_ch, T)
      → temporal Conv2d + spatial Conv2d  (learned, reduces n_ch → spatial_filters)
      → CovariancePool                     → (B, F, F) SPD
      → BiMap (F → subspace) → ReEig       → (B, d, d) SPD
      → SPD batch-norm:
            S4: AdaMomSPDBatchNorm          (global Fréchet-mean normalisation — safe)
            S5: AdaMomDomainSPDBatchNorm    (per-SUBJECT normalisation = SPDDSMBN)
      → LogEig → Flatten → Linear → 2 logits

The ONLY difference S4→S5 is global vs per-domain SPD batch-norm, so Δ(S5−S4) isolates
the value (or harm) of per-subject normalisation.

⚠ Scientific caveat (the S3 litmus test): SPDDSMBN with ``learn_mean=False`` recentres
every subject's Fréchet mean to the identity. For a *resting-state, label-per-subject*
task this risks deleting the between-subject mean — the same failure mode that collapsed
S3 — unless the learned BiMap routes class signal into the dispersion. S5 is the test of
whether per-subject normalisation survives a subject-level label here.

The SPD layers run in float64 on ``spd_device`` (CPU by default, matching the reference
for eigen-decomposition stability); the conv front-end runs on ``device`` (GPU).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .data import Subject, epoch

# Make the TSMNet reference package importable and reuse its (correct) SPD layers.
_TSMNET = C.PROJECT_ROOT / "example code" / "TSMNet"
if str(_TSMNET) not in sys.path:
    sys.path.insert(0, str(_TSMNET))
import spdnets.modules as M          # noqa: E402  BiMap, ReEig, LogEig, CovariancePool
import spdnets.batchnorm as BN       # noqa: E402  AdaMom(Domain)SPDBatchNorm
from geoopt.optim import RiemannianAdam  # noqa: E402


# ===========================================================================
# Network
# ===========================================================================
class SPDNet(nn.Module):
    def __init__(self, n_ch: int, temporal_filters: int = 4, spatial_filters: int = 40,
                 subspace: int = 20, temp_kernel: int = 25, domain_bn: bool = False,
                 n_domains: int = 0, device="cuda", spd_device="cpu"):
        super().__init__()
        self.device_ = torch.device(device)
        self.spd_device_ = torch.device(spd_device)
        self.domain_bn = domain_bn
        self.subspace = subspace

        self.cnn = nn.Sequential(
            nn.Conv2d(1, temporal_filters, (1, temp_kernel),
                      padding="same", padding_mode="reflect"),
            nn.Conv2d(temporal_filters, spatial_filters, (n_ch, 1)),
            nn.Flatten(start_dim=2),
        ).to(self.device_)

        self.cov = M.CovariancePool()
        self.spdnet = nn.Sequential(
            M.BiMap((1, spatial_filters, subspace), dtype=torch.double, device=self.spd_device_),
            M.ReEig(threshold=1e-4),
        )

        bn_shape = (1, subspace, subspace)
        bn_kw = dict(batchdim=0, learn_mean=False, learn_std=True,
                     dispersion=BN.BatchNormDispersion.SCALAR, eta=1.0, eta_test=0.1,
                     dtype=torch.double, device=self.spd_device_)
        if domain_bn:
            self.bn = BN.AdaMomDomainSPDBatchNorm(
                bn_shape, domains=torch.arange(n_domains), **bn_kw)
        else:
            self.bn = BN.AdaMomSPDBatchNorm(bn_shape, **bn_kw)

        self.logeig = nn.Sequential(M.LogEig(subspace), nn.Flatten(start_dim=1))
        tsdim = subspace * (subspace + 1) // 2
        self.classifier = nn.Linear(tsdim, 2).double().to(self.spd_device_)

    def _pre_bn(self, x: torch.Tensor) -> torch.Tensor:
        h = self.cnn(x.to(self.device_)[:, None, ...])
        cc = self.cov(h).to(device=self.spd_device_, dtype=torch.double)
        return self.spdnet(cc)                                # (B, d, d) SPD

    def _head(self, l: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.logeig(l))

    def forward(self, x: torch.Tensor, d: torch.Tensor | None = None) -> torch.Tensor:
        l = self._pre_bn(x)
        l = self.bn(l, d.to(self.spd_device_)) if self.domain_bn else self.bn(l)
        return self._head(l)


# ===========================================================================
# Stage wrapper (S4 / S5)
# ===========================================================================
class SPDNetStage:
    """Deep SPD encoder trained source-only; per-window softmax averaged per subject.

    domain_bn=False → S4 (global SPD batch-norm).
    domain_bn=True  → S5 (SPDDSMBN, per-subject). The held-out subject is a new domain;
                      its batch-norm reference is estimated from its OWN unlabelled
                      windows at test time (label-free, one-shot — as in TSMNet LOSO).
    """

    def __init__(self, domain_bn: bool, epochs: int = 50, batch_size: int = 128,
                 lr: float = 1e-3, weight_decay: float = 1e-4, max_windows: int | None = None,
                 temporal_filters: int = 4, spatial_filters: int = 40, subspace: int = 20,
                 temp_kernel: int = 25, domains_per_batch: int = 8,
                 device: str | None = None, seed: int = C.SEED, verbose: bool = False):
        self.domain_bn = domain_bn
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_windows = max_windows
        self.hp = dict(temporal_filters=temporal_filters, spatial_filters=spatial_filters,
                       subspace=subspace, temp_kernel=temp_kernel)
        self.domains_per_batch = domains_per_batch
        self.seed = seed
        self.verbose = verbose
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- per-recording standardisation (leakage-free) ----
    def _windows(self, sub: Subject) -> np.ndarray:
        w = epoch(sub.data, max_windows=self.max_windows)
        mu = w.mean(axis=(0, 2), keepdims=True)
        sd = w.std(axis=(0, 2), keepdims=True) + 1e-6
        return ((w - mu) / sd).astype(np.float32)

    # ---- domain-balanced batches (needed so SPDDSMBN can estimate a per-domain mean) ----
    def _domain_batches(self, dom_idx: np.ndarray, rng: np.random.Generator):
        by_dom = {d: np.where(dom_idx == d)[0] for d in np.unique(dom_idx)}
        for idxs in by_dom.values():
            rng.shuffle(idxs)
        n_dom = max(1, min(self.domains_per_batch, len(by_dom)))
        per_dom = max(1, self.batch_size // n_dom)
        n_batches = int(np.ceil(len(dom_idx) / self.batch_size))
        doms = list(by_dom)
        cursor = {d: 0 for d in doms}
        for _ in range(n_batches):
            chosen = rng.choice(doms, size=n_dom, replace=False)
            batch = []
            for d in chosen:
                pool = by_dom[d]
                c = cursor[d]
                take = pool[c:c + per_dom]
                if len(take) < per_dom:                       # wrap around
                    take = np.concatenate([take, pool[:per_dom - len(take)]])
                cursor[d] = (c + per_dom) % max(1, len(pool))
                batch.append(take)
            yield np.concatenate(batch)

    def fit(self, train_subjects: list[Subject]) -> "SPDNetStage":
        t0 = time.time()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        Xs, ys, ds = [], [], []
        for di, s in enumerate(train_subjects):
            w = self._windows(s)
            Xs.append(w)
            ys.append(np.full(len(w), s.label))
            ds.append(np.full(len(w), di))
        X = np.concatenate(Xs)                                # (N,ch,T); forward adds chan dim
        y = np.concatenate(ys)
        dom = np.concatenate(ds)
        self.n_domains = len(train_subjects)
        n_ch = X.shape[1]

        counts = np.bincount(y, minlength=2).astype(np.float64)
        w_cls = counts.sum() / (2.0 * np.clip(counts, 1, None))
        weight = torch.tensor(w_cls, dtype=torch.double, device=self.net_spd_device())

        self.net = SPDNet(n_ch=n_ch, domain_bn=self.domain_bn, n_domains=self.n_domains,
                          device=self.device, **self.hp)
        opt = RiemannianAdam(self.net.parameters(), lr=self.lr,
                             weight_decay=self.weight_decay, stabilize=10)
        loss_fn = nn.CrossEntropyLoss(weight=weight)

        Xt = torch.from_numpy(X)
        yt = torch.from_numpy(y).long()
        dt = torch.from_numpy(dom).long()

        self.net.train()
        for ep in range(self.epochs):
            if self.domain_bn:
                batches = list(self._domain_batches(dom, rng))
            else:
                order = rng.permutation(len(y))
                batches = [order[i:i + self.batch_size]
                           for i in range(0, len(y), self.batch_size)]
            tot = 0.0
            for bi in batches:
                bi = torch.from_numpy(np.asarray(bi)).long()
                opt.zero_grad()
                logits = self.net(Xt[bi], dt[bi] if self.domain_bn else None)
                loss = loss_fn(logits, yt[bi].to(logits.device))
                loss.backward()
                opt.step()
                tot += loss.item() * len(bi)
            if self.verbose and (ep + 1) % 10 == 0:
                print(f"      epoch {ep+1:>3}/{self.epochs}  loss={tot/len(y):.4f}")
        self.fit_s = time.time() - t0
        return self

    def net_spd_device(self):
        return torch.device("cpu")

    @torch.no_grad()
    def predict_subject(self, sub: Subject) -> tuple[float, int]:
        self.net.eval()
        w = self._windows(sub)                                # (n,ch,T); forward adds chan dim
        Xt = torch.from_numpy(w)

        if self.domain_bn:
            # held-out subject is a NEW domain: estimate its SPD-BN reference from its
            # own unlabelled windows (label-free), then classify with it.
            di = self.n_domains
            new_bn = BN.SPDBatchNormImpl(
                shape=(1, self.net.subspace, self.net.subspace), batchdim=0,
                learn_mean=False, learn_std=True, dispersion=BN.BatchNormDispersion.SCALAR,
                mean=self.net.bn.mean, std=self.net.bn.std, eta=1.0, eta_test=0.1,
                dtype=torch.double, device=self.net.spd_device_)
            self.net.bn.add_domain_(new_bn, torch.tensor(di))
            l_pre = self.net._pre_bn(Xt)
            self.net.bn.initrunningstats(l_pre, torch.tensor(di))
            d = torch.full((len(w),), di, dtype=torch.long)
            logits = self.net(Xt, d)
        else:
            logits = self.net(Xt)

        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        return float(probs.mean()), len(probs)


def build_spdnet(stage: str, **kw) -> SPDNetStage:
    return SPDNetStage(domain_bn=(stage == "s5"), **kw)
