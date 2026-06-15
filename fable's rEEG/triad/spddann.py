"""SPD-DANN — domain-adversarial SPD-manifold encoder for LODO (Cheng et al., Neural
Networks 2026), specialised for cross-DATASET depression transfer.

Why this model (vs. the shallow nrde backbone): the shallow diagnostics showed the
cross-dataset MDD/HC *connectivity* axis is near-orthogonal across datasets — fixed
features can't be re-aligned. SPD-DANN can, because it *learns* the representation under
two transductive objectives that target exactly that:
  * adversarial domain-invariance (GRL domain discriminator) — marginal alignment;
  * SPD class-prototype pair loss with entropy-filtered pseudo-labels — CONDITIONAL
    alignment that pulls same-class source/target prototypes together (the axis-rotation
    the shallow recentring could not do).
It is transductive: the held-out dataset's UNLABELLED windows are used at fit time
(no labels). Set them via set_target() before fit() — the LODO harness does this.

Architecture (paper Fig. 2, simplified — single covariance pool, no manifold attention,
to limit overfitting at clinical N): conv(temporal)→conv(spatial)→CovariancePool→BiMap→
ReEig → SPD feature S. Classifier and domain discriminator both consume logm(S) (tangent).
Loss: L_C (source CE) + GRL·L_D (domain) + L_M (LEM mean align) + λ_P·L_P (prototype pair).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .data import Subject, epoch

_TSMNET = C.PROJECT_ROOT / "example code" / "TSMNet"
if str(_TSMNET) not in sys.path:
    sys.path.insert(0, str(_TSMNET))
import spdnets.modules as M               # noqa: E402  BiMap, ReEig, CovariancePool
from spdnets.batchnorm import SPDBatchNorm  # noqa: E402  Riemannian (Fréchet-mean) BN
from geoopt.optim import RiemannianAdam   # noqa: E402


# ---------------------------------------------------------------------------
# Gradient Reversal Layer (Ganin & Lempitsky 2015)
# ---------------------------------------------------------------------------
class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None


def grad_reverse(x, lambd):
    return _GRL.apply(x, lambd)


def _logm_sym(S: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Matrix log of a batch of symmetric PD matrices via eigh (B,d,d) -> (B,d,d)."""
    S = 0.5 * (S + S.transpose(-1, -2))
    w, U = torch.linalg.eigh(S)
    w = torch.clamp(w, min=eps)
    return U @ torch.diag_embed(torch.log(w)) @ U.transpose(-1, -2)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class SPDDANN(nn.Module):
    def __init__(self, n_ch: int, temporal_filters: int = 4, spatial_filters: int = 24,
                 subspace: int = 12, temp_kernel: int = 25, disc_hidden: int = 64,
                 use_de: bool = True, de_dim: int = 0, de_hidden: int = 64,
                 feat_dropout: float = 0.0, spd_input: str = "conv", spd_bn: bool = False,
                 nr_dim: int = 0, nr_proj_dim: int = 0, device="cuda", spd_device="cuda"):
        super().__init__()
        self.device_ = torch.device(device)
        self.spd_device_ = torch.device(spd_device)
        self.subspace = subspace
        self.use_de = use_de
        self.feat_dropout = feat_dropout
        # spd_input: 'conv' = learned temporal+spatial conv -> covariance pool (paper backbone);
        # 'corr' = feed the RAW channel correlation matrix straight into BiMap (a *learnable nr*:
        # supervised SPD dim-reduction of the true correlation manifold, the representation that
        # wins within-dataset at clinical N where the conv encoder overfits).
        self.spd_input = spd_input
        # identity-subspace projection (I - VᵀV) on the fused tangent feature, the nr
        # mechanism that drives the shallow within-dataset win: strips the top-K
        # between-subject identity directions the encoder would otherwise memorise. V is
        # re-estimated each epoch by the stage (SVD of the train subject-mean cloud) and
        # set as a buffer here; None = off (preserves the original LODO behaviour).
        self.register_buffer("V", None, persistent=False)
        if spd_input == "conv":
            self.cnn = nn.Sequential(
                nn.Conv2d(1, temporal_filters, (1, temp_kernel),
                          padding="same", padding_mode="reflect"),
                nn.Conv2d(temporal_filters, spatial_filters, (n_ch, 1)),
                nn.Flatten(start_dim=2),
            ).to(self.device_)
            bimap_in = spatial_filters
        else:                                    # 'corr' — no conv; BiMap reduces n_ch -> subspace
            self.cnn = None
            bimap_in = n_ch
        self.cov = M.CovariancePool()
        spd_layers = [M.BiMap((1, bimap_in, subspace), dtype=torch.double,
                              device=self.spd_device_)]
        if spd_bn:
            # Riemannian (Fréchet-mean) batch norm [Kobler 2022] — recentres each batch's
            # SPD mean to a learned reference + scalar dispersion. Regularises the encoder
            # (within-dataset) and learns a Fréchet-mean alignment (the LEARNED analog of the
            # fixed recentering that hurt). NeurIPS-2022's documented cross-dataset gain driver.
            spd_layers.append(SPDBatchNorm((subspace, subspace), batchdim=0,
                                           dtype=torch.double, device=self.spd_device_))
        spd_layers.append(M.ReEig(threshold=1e-4))
        self.spdnet = nn.Sequential(*spd_layers)
        feat_dim = subspace * subspace
        if use_de:
            # spectral branch (the cross-dataset-transferable signal) fused end-to-end so
            # the adversarial + prototype objectives shape BOTH streams jointly.
            self.de_mlp = nn.Sequential(
                nn.Linear(de_dim, de_hidden), nn.ReLU(),
                nn.Linear(de_hidden, de_hidden), nn.ReLU(),
            ).double().to(self.spd_device_)
            feat_dim += de_hidden
        # fixed-nrde connectivity stream: the full identity-removed correlation tangent
        # (computed outside the net, like DE) fed RAW into the linear classifier — within
        # a single dataset this lets the classifier behave like nrde's logistic regression
        # (the within-dataset winner the learned encoder can't match at clinical N). For
        # LODO the learned+adversarial streams carry transfer (this fixed block transfers
        # at chance, so the classifier simply down-weights it).
        self.nr_dim = nr_dim
        self.nr_proj = None
        if nr_dim:
            if nr_proj_dim > 0:
                # learnable linear projection of the fixed nr tangent. It feeds BOTH the
                # classifier and (via GRL) the domain discriminator, so adversarial training
                # can suppress the non-transferable connectivity directions for LODO, while
                # within-dataset (GRL off) it just learns the class-discriminative projection
                # (≈ nrde). No activation -> stays linear, preserving nrde's separability.
                self.nr_proj = nn.Linear(nr_dim, nr_proj_dim).double().to(self.spd_device_)
                feat_dim += nr_proj_dim
            else:
                feat_dim += nr_dim                # raw concat (not GRL-suppressible)
        self.classifier = nn.Linear(feat_dim, 2).double().to(self.spd_device_)
        self.discriminator = nn.Sequential(
            nn.Linear(feat_dim, disc_hidden), nn.ReLU(),
            nn.Linear(disc_hidden, disc_hidden), nn.ReLU(),
            nn.Linear(disc_hidden, 2),
        ).double().to(self.spd_device_)

    def features_logm(self, x: torch.Tensor) -> torch.Tensor:
        """raw windows (B, n_ch, T) -> matrix-form logm of the SPD feature (B,d,d)."""
        if self.spd_input == "conv":
            h = self.cnn(x.to(self.device_)[:, None, ...])
            cc = self.cov(h).to(device=self.spd_device_, dtype=torch.double)
        else:                                    # 'corr' — raw channel correlation manifold
            xt = x.to(device=self.spd_device_, dtype=torch.double)   # (B, n_ch, T)
            cc = self.cov(xt)                                        # (B, n_ch, n_ch) covariance
            dg = torch.sqrt(torch.diagonal(cc, dim1=-2, dim2=-1).clamp_min(1e-8))
            cc = cc / (dg[:, :, None] * dg[:, None, :])             # -> correlation (scale-inv)
        # trace-normalise + diagonal loading (SPD-DANN Eq. 8) so the eigendecomposition
        # in ReEig is well-conditioned (avoids the eigh non-convergence on degenerate cov).
        d = cc.shape[-1]
        tr = cc.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-6)
        cc = cc / tr[:, None, None] * d
        cc = cc + 1e-3 * torch.eye(d, dtype=cc.dtype, device=cc.device)
        S = self.spdnet(cc)                                   # (B, d, d) SPD
        return _logm_sym(S)                                   # (B, d, d) tangent

    def raw_features(self, x, de=None):
        """Fused tangent feature BEFORE identity projection (used to estimate V)."""
        L = self.features_logm(x)                             # (B,d,d) connectivity tangent
        v = L.flatten(1)                                      # (B, d*d)
        if self.use_de:
            de_emb = self.de_mlp(de.to(device=self.spd_device_, dtype=torch.double))
            v = torch.cat([v, de_emb], dim=1)                # fuse spectral + connectivity
        return L, v

    def project_identity(self, v):
        """Remove the learned between-subject identity directions: v(I - VᵀV)."""
        if self.V is None:
            return v
        return v - (v @ self.V.t()) @ self.V

    def forward(self, x, de=None, nr=None, lambd: float = 0.0):
        L, v = self.raw_features(x, de)
        v = self.project_identity(v)                          # nr-style identity removal
        if self.nr_dim:                                       # append fixed-nrde tangent stream
            nr_t = nr.to(device=self.spd_device_, dtype=torch.double)
            if self.nr_proj is not None:                      # GRL-exposed learnable projection
                nr_t = self.nr_proj(nr_t)
            v = torch.cat([v, nr_t], dim=1)
        v_in = nn.functional.dropout(v, p=self.feat_dropout, training=self.training)
        logits = self.classifier(v_in)
        dlogits = self.discriminator(grad_reverse(v_in, lambd))
        return logits, dlogits, L, v


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def _lem_mean_align(Ls: torch.Tensor, Lt: torch.Tensor) -> torch.Tensor:
    """L_M: Frobenius distance between source/target log-Euclidean means (logm already)."""
    return torch.linalg.norm(Ls.mean(0) - Lt.mean(0))


def _prototype_pair_loss(Ls, ys, Lt, logits_t, margin=0.1, ent_keep=0.5):
    """L_P (paper Eq. 15) in the log domain: pull same-class source/target prototypes
    together, push different-class apart, using entropy-filtered target pseudo-labels."""
    n_cls = 2
    protos_s = []
    for k in range(n_cls):
        m = ys == k
        if m.sum() == 0:
            return Ls.new_zeros(())
        protos_s.append(Ls[m].mean(0))
    # entropy-filtered target pseudo-labels (top ent_keep most confident)
    p = torch.softmax(logits_t, dim=1)
    ent = -(p * torch.log(p + 1e-8)).sum(1)
    yhat = p.argmax(1)
    keep = max(1, int(ent_keep * len(ent)))
    idx = torch.argsort(ent)[:keep]
    Lt_c, yhat_c = Lt[idx], yhat[idx]
    protos_t = []
    for k in range(n_cls):
        m = yhat_c == k
        protos_t.append(Lt_c[m].mean(0) if m.sum() > 0 else None)
    loss, cnt = Ls.new_zeros(()), 0
    for i in range(n_cls):
        if protos_t[i] is None:
            continue
        d_si_ti = torch.linalg.norm(protos_s[i] - protos_t[i])
        for j in range(n_cls):
            if j == i:
                continue
            d_ti_tj = (torch.linalg.norm(protos_t[i] - protos_t[j])
                       if protos_t[j] is not None else d_si_ti + margin)
            d_ti_sj = torch.linalg.norm(protos_t[i] - protos_s[j])
            loss = loss + torch.clamp(d_si_ti - d_ti_tj + margin, min=0) \
                        + torch.clamp(d_si_ti - d_ti_sj + margin, min=0)
            cnt += 1
    return loss / max(1, cnt)


def _mmd(xs: torch.Tensor, xt: torch.Tensor,
         mul=(0.25, 0.5, 1.0, 2.0, 4.0)) -> torch.Tensor:
    """Multi-bandwidth Gaussian MMD^2 between feature sets xs (n,d), xt (m,d).

    Bandwidth = median pairwise squared distance (median heuristic), scaled by `mul`.
    Stable, closed-form (no critic) — the WDANet global/local discrepancy, MMD variant."""
    z = torch.cat([xs, xt], dim=0)
    d2 = torch.cdist(z, z).pow(2)
    n = xs.shape[0]
    with torch.no_grad():
        med = d2[d2 > 0].median().clamp_min(1e-6)
    K = sum(torch.exp(-d2 / (2.0 * med * m)) for m in mul)
    Kss = K[:n, :n].mean()
    Ktt = K[n:, n:].mean()
    Kst = K[:n, n:].mean()
    return (Kss + Ktt - 2.0 * Kst).clamp_min(0.0)


# ---------------------------------------------------------------------------
# Stage wrapper (LODO; transductive)
# ---------------------------------------------------------------------------
class SPDDANNStage:
    def __init__(self, epochs: int = 60, batch_size: int = 128, lr: float = 1e-2,
                 weight_decay: float = 1e-3, subspace: int = 12, spatial_filters: int = 24,
                 temporal_filters: int = 4, temp_kernel: int = 25, disc_hidden: int = 64,
                 use_de: bool = True, de_hidden: int = 64, domain_adapt: bool = True,
                 lambda_p: float = 0.1, margin: float = 0.1, ent_keep: float = 0.5,
                 identity_removal: int = 0, feat_dropout: float = 0.0, v_stride: int = 1,
                 spd_input: str = "conv", spd_bn: bool = False,
                 nr_stream: bool = False, nr_k: int = 3, nr_proj_dim: int = 0,
                 nr_domain_removal: int = 0, nr_mmd: float = 0.0,
                 wdann: bool = False, align_weight: float = 1.0,
                 max_windows: int | None = 100, device: str | None = None,
                 spd_device: str | None = None, seed: int = C.SEED, verbose: bool = False):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_de = use_de
        # domain_adapt=False -> supervised two-stream classifier only (L_C); no GRL/L_M/L_P
        # and no target needed. Use this WITHIN-dataset (no domain gap to bridge); the DA
        # losses are meant for a real cross-dataset shift and destabilise CV folds otherwise.
        self.domain_adapt = domain_adapt
        # identity_removal=K>0 -> strip the top-K between-subject identity PCs from the fused
        # tangent feature each epoch (the nr mechanism behind the shallow within-dataset win).
        # v_stride re-estimates V every v_stride epochs (1 = every epoch).
        self.identity_removal = identity_removal
        self.v_stride = max(1, v_stride)
        # nr_stream=True -> append the fixed identity-removed correlation tangent (= nrde's
        # connectivity feature) to the fused vector, so within a single dataset the linear
        # classifier matches nrde. nr_k = identity PCs removed (the shallow default is 3).
        self.nr_stream = nr_stream
        self.nr_k = nr_k
        self.nr_proj_dim = nr_proj_dim    # >0: GRL-exposed learnable projection of v_nr (#2)
        # nr_domain_removal>0: strip top-D between-DATASET directions from the nr tangent
        # (dataset-level analog of identity removal, class-axis protected) — removes the
        # source-specific connectivity that causes the MODMA-LODO negative transfer. No-op
        # within-dataset (one domain). nr_mmd>0: add an MMD penalty aligning source/target
        # nr-proj embeddings (explicit distribution alignment of the fixed stream).
        self.nr_domain_removal = nr_domain_removal
        self.nr_mmd = nr_mmd
        # WDANet-style alignment (MMD variant): replace the CE discriminator + prototype loss
        # with global MMD + local class-conditional MMD (LMMD), balanced by a dynamic factor ω.
        self.wdann = wdann
        self.align_weight = align_weight
        self._nr_head = None
        self._nr_scaler = None
        self._nr_domW = None              # stored between-dataset removal basis (predict-time)
        self.hp = dict(subspace=subspace, spatial_filters=spatial_filters,
                       temporal_filters=temporal_filters, temp_kernel=temp_kernel,
                       disc_hidden=disc_hidden, use_de=use_de, de_hidden=de_hidden,
                       feat_dropout=feat_dropout, spd_input=spd_input, spd_bn=spd_bn)
        self.lambda_p = lambda_p
        self.margin = margin
        self.ent_keep = ent_keep
        self.max_windows = max_windows
        self.seed = seed
        self.verbose = verbose
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.spd_device = spd_device or self.device
        self._target = None
        self._de_mu = self._de_sd = None      # pooled DE standardiser (fit on source)

    def set_target(self, subjects) -> None:
        """Stash the held-out dataset's UNLABELLED subjects (used transductively)."""
        self._target = list(subjects)

    @torch.no_grad()
    def _estimate_V(self, Xs_t, DEs_t, bounds, k, chunk=256):
        """Re-estimate the identity subspace V from the current encoder: top-K right
        singular vectors of the centred TRAIN subject-mean cloud of raw fused features.
        Mirrors nr's identity-PC removal but on the learned (evolving) representation."""
        was_training = self.net.training
        self.net.eval()
        means = []
        for a, b in bounds:                       # one (start, end) per training subject
            vs = []
            for c in range(a, b, chunk):
                e = min(b, c + chunk)
                de = DEs_t[c:e] if self.use_de else None
                vs.append(self.net.raw_features(Xs_t[c:e], de)[1])
            means.append(torch.cat(vs, 0).mean(0))
        cloud = torch.stack(means)                # (n_subj, feat_dim)
        cloud = cloud - cloud.mean(0)
        Vt = torch.linalg.svd(cloud, full_matrices=False)[2]
        self.net.V = Vt[:k].detach().contiguous()
        if was_training:
            self.net.train()

    def _prep(self, sub: Subject):
        """Return (z-scored windows for conv, standardisable DE features or None)."""
        w = epoch(sub.data, max_windows=self.max_windows)        # (n, ch, T) raw
        de = None
        if self.use_de:
            from .models import _compute_de
            de = _compute_de(w).astype(np.float32)               # (n, ch*5) from raw windows
        mu = w.mean(axis=(0, 2), keepdims=True)
        sd = w.std(axis=(0, 2), keepdims=True) + 1e-6
        wz = ((w - mu) / sd).astype(np.float32)
        return wz, de

    def fit(self, train_subjects: list[Subject]) -> "SPDDANNStage":
        if self.domain_adapt:
            assert self._target is not None, "call set_target(unlabeled target subjects) first"
        t0 = time.time()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        sp = [self._prep(s) for s in train_subjects]
        Xs = np.concatenate([w for w, _ in sp])
        ys = np.concatenate([np.full(len(w), s.label) for s, (w, _) in zip(train_subjects, sp)])
        n_ch = Xs.shape[1]
        # per-subject window spans (for the identity-subspace V estimate)
        _counts = [len(w) for w, _ in sp]
        _ends = np.cumsum(_counts)
        bounds = list(zip((_ends - _counts).tolist(), _ends.tolist()))
        if self.domain_adapt:
            tp = [self._prep(s) for s in self._target]
            Xt = np.concatenate([w for w, _ in tp])

        de_dim = 0
        if self.use_de:
            DEs = np.concatenate([d for _, d in sp])
            self._de_mu = DEs.mean(0, keepdims=True)
            self._de_sd = DEs.std(0, keepdims=True) + 1e-6
            DEs = ((DEs - self._de_mu) / self._de_sd).astype(np.float32)
            de_dim = DEs.shape[1]
            if self.domain_adapt:
                DEt = np.concatenate([d for _, d in tp])
                DEt = ((DEt - self._de_mu) / self._de_sd).astype(np.float32)

        # ---- fixed-nrde connectivity stream (identity-removed correlation tangent) ----
        nr_dim = 0
        if self.nr_stream:
            from collections import defaultdict
            from .models import make_shallow
            from pyriemann.utils.tangentspace import tangent_space as _ts
            from sklearn.preprocessing import StandardScaler
            self._nr_head = make_shallow("nr", n_nuisance=self.nr_k,
                                         max_windows=self.max_windows)
            self._nr_head.fit(train_subjects)               # train-only ref + identity PCs
            def _nrfeat(sub):                                # same epoch() windows as _prep
                f = self._nr_head._features(sub)
                return self._nr_head._project(_ts(f["mats"], self._nr_head._ref)).astype(np.float32)
            src_feats = [_nrfeat(s) for s in train_subjects]
            self._nr_scaler = StandardScaler().fit(np.vstack(src_feats))
            src_feats = [self._nr_scaler.transform(x).astype(np.float32) for x in src_feats]
            tgt_feats = ([self._nr_scaler.transform(_nrfeat(s)).astype(np.float32)
                          for s in self._target] if self.domain_adapt else [])
            # ---- between-DATASET direction removal (class-protected) ----
            if self.nr_domain_removal > 0 and self.domain_adapt:
                by_ds = defaultdict(list)
                for s, x in zip(train_subjects, src_feats):
                    by_ds[s.dataset].append(x.mean(0))
                dmeans = [np.mean(ms, axis=0) for ms in by_ds.values()]
                dmeans.append(np.vstack(tgt_feats).mean(0))   # transductive target domain
                cloud = np.stack(dmeans); cloud = cloud - cloud.mean(0)
                D = min(self.nr_domain_removal, cloud.shape[0] - 1)
                W0 = np.linalg.svd(cloud, full_matrices=False)[2][:D]
                allsrc = np.vstack(src_feats)                  # protect the class axis
                ylab = np.concatenate([np.full(len(x), s.label)
                                       for s, x in zip(train_subjects, src_feats)])
                c = allsrc[ylab == 1].mean(0) - allsrc[ylab == 0].mean(0)
                c = c / (np.linalg.norm(c) + 1e-12)
                W0 = W0 - np.outer(W0 @ c, c)                  # strip class component
                Q, _ = np.linalg.qr(W0.T)
                self._nr_domW = Q.T[:D].astype(np.float32)
                _rm = lambda x: (x - (x @ self._nr_domW.T) @ self._nr_domW).astype(np.float32)
                src_feats = [_rm(x) for x in src_feats]
                tgt_feats = [_rm(x) for x in tgt_feats]
                print(f"      [nr] removed {D} between-dataset direction(s) from nr stream",
                      flush=True)
            NRs = np.vstack(src_feats)
            nr_dim = NRs.shape[1]
            if self.domain_adapt:
                NRt = np.vstack(tgt_feats)

        counts = np.bincount(ys, minlength=2).astype(np.float64)
        w_cls = counts.sum() / (2.0 * np.clip(counts, 1, None))
        dev = torch.device(self.spd_device)
        weight = torch.tensor(w_cls, dtype=torch.double, device=dev)

        self.net = SPDDANN(n_ch=n_ch, de_dim=de_dim, nr_dim=nr_dim,
                           nr_proj_dim=self.nr_proj_dim, device=self.device,
                           spd_device=self.spd_device, **self.hp)
        opt = RiemannianAdam(self.net.parameters(), lr=self.lr,
                             weight_decay=self.weight_decay, stabilize=10)
        ce = nn.CrossEntropyLoss(weight=weight)
        ce_dom = nn.CrossEntropyLoss()

        Xs_t, ys_t = torch.from_numpy(Xs), torch.from_numpy(ys).long()
        DEs_t = torch.from_numpy(DEs) if self.use_de else None
        NRs_t = torch.from_numpy(NRs) if self.nr_stream else None
        n_s = len(ys)
        if self.domain_adapt:
            Xt_t = torch.from_numpy(Xt)
            DEt_t = torch.from_numpy(DEt) if self.use_de else None
            NRt_t = torch.from_numpy(NRt) if self.nr_stream else None
            n_t = len(Xt)
        steps = int(np.ceil(n_s / self.batch_size))
        total = self.epochs * steps

        k_id = min(self.identity_removal, len(train_subjects) - 1) if self.identity_removal else 0

        self.net.train()
        gstep = 0
        n_skip = 0
        for ep in range(self.epochs):
            if k_id > 0 and ep % self.v_stride == 0:
                self._estimate_V(Xs_t, DEs_t, bounds, k_id)   # refresh identity subspace
            order = rng.permutation(n_s)
            tot = 0.0
            for si in range(0, n_s, self.batch_size):
                bs = order[si:si + self.batch_size]
                p = gstep / max(1, total)
                lambd = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0       # DANN schedule (= λ_D)
                gstep += 1

                opt.zero_grad()
                bsi = torch.from_numpy(bs).long()
                xs = Xs_t[bsi]
                des = DEs_t[bsi] if self.use_de else None
                nrs = NRs_t[bsi] if self.nr_stream else None
                yb = ys_t[bsi].to(dev)
                try:
                    logit_s, dlog_s, Ls, vs = self.net(xs, des, nrs, lambd)
                    L_C = ce(logit_s, yb)
                    if self.domain_adapt:
                        bti = torch.from_numpy(rng.choice(n_t, size=len(bs),
                                                          replace=len(bs) > n_t)).long()
                        xt = Xt_t[bti]
                        det = DEt_t[bti] if self.use_de else None
                        nrt = NRt_t[bti] if self.nr_stream else None
                        logit_t, dlog_t, Lt, vt = self.net(xt, det, nrt, lambd)
                        L_M = _lem_mean_align(Ls, Lt)
                        if self.wdann:
                            # global + local(class-conditional) MMD with dynamic ω (WDANet).
                            # Align ONLY the connectivity tangent (where the cross-dataset gap
                            # is) — NOT the fused vector, so the transferable spectral/DE stream
                            # is left undistorted (aligning it craters spectral-driven Mumtaz).
                            cs, ct = Ls.flatten(1), Lt.flatten(1)
                            pt = torch.softmax(logit_t.detach(), 1).argmax(1)
                            mmd_g = _mmd(cs, ct)
                            mmd_l, nc = cs.new_zeros(()), 0
                            for k in (0, 1):
                                ms, mt = yb == k, pt == k
                                if int(ms.sum()) > 1 and int(mt.sum()) > 1:
                                    mmd_l = mmd_l + _mmd(cs[ms], ct[mt]); nc += 1
                            mmd_l = mmd_l / max(1, nc)
                            dg, dl = mmd_g.detach(), mmd_l.detach()
                            om = dg / (dg + dl + 1e-8)         # dynamic ω: weight the worse-aligned
                            align = om * mmd_g + (1.0 - om) * mmd_l
                            loss = L_C + L_M + lambd * self.align_weight * align
                        else:
                            dom = torch.cat([torch.ones(len(bs), dtype=torch.long),
                                             torch.zeros(len(bti), dtype=torch.long)]).to(dev)
                            L_D = ce_dom(torch.cat([dlog_s, dlog_t]), dom)
                            L_P = _prototype_pair_loss(vs, yb, vt, logit_t.detach(),
                                                       self.margin, self.ent_keep)
                            loss = L_C + L_D + L_M + self.lambda_p * L_P
                    else:
                        loss = L_C                                   # supervised-only (no gap)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                    opt.step()
                except torch.linalg.LinAlgError:
                    # ill-conditioned eigh (forward) or singular Stiefel retraction (step):
                    # skip this batch (params stay at the last valid point).
                    opt.zero_grad(set_to_none=True)
                    n_skip += 1
                    continue
                tot += L_C.item() * len(bs)
            if self.verbose and (ep + 1) % 10 == 0:
                mode = f"lambd={lambd:.2f}" if self.domain_adapt else "supervised"
                sk = f"  skipped={n_skip}" if n_skip else ""
                print(f"      epoch {ep+1:>3}/{self.epochs}  L_C={tot/n_s:.4f}  {mode}{sk}",
                      flush=True)
        self.fit_s = time.time() - t0
        return self

    @torch.no_grad()
    def predict_subject(self, sub: Subject) -> tuple[float, int]:
        self.net.eval()
        wz, de = self._prep(sub)
        x = torch.from_numpy(wz)
        det = None
        if self.use_de:
            de = ((de - self._de_mu) / self._de_sd).astype(np.float32)
            det = torch.from_numpy(de)
        nrt = None
        if self.nr_stream:
            from pyriemann.utils.tangentspace import tangent_space as _ts
            f = self._nr_head._features(sub)
            nr = self._nr_head._project(_ts(f["mats"], self._nr_head._ref)).astype(np.float32)
            nr = self._nr_scaler.transform(nr).astype(np.float32)
            if self._nr_domW is not None:                     # between-dataset removal
                nr = (nr - (nr @ self._nr_domW.T) @ self._nr_domW).astype(np.float32)
            nrt = torch.from_numpy(nr)
        logits, _, _, _ = self.net(x, det, nrt, 0.0)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        return float(probs.mean()), len(probs)
