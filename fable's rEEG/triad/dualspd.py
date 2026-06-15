"""TRIAD-Dual — learned low-rank dual-manifold SPD encoder (no conv front-end).

Scientific question (NOT "beat nr"):
    Can learned low-rank dual-manifold SPD geometry recover the cov+corr information
    that shallow `bd` lost to dimensionality, while preserving the between-subject
    class axis?

Design, with every hard-won lesson baked in:
  * Direct 125x125 cov / corr matrices into the BiMaps — NO temporal-conv front-end,
    so the question stays "learned vs handcrafted GEOMETRY", uncontaminated by
    "learned spectral filters" (keeps the two scientific questions separate).
  * One stream per manifold (cov, corr); they share nothing until the flat layer
    (the principled "both", unlike the dimensionality-drowned shallow `bd`).
  * GLOBAL SPD batch-norm only (AdaMomSPDBatchNorm). HYPOTHESIS (not axiom): the
    per-domain variant (SPDDSMBN) recentres each subject and may delete the
    between-subject mean — our experiments are *consistent* with this (S3 recentring
    = 0.500, feature-space transduction tnr = 0.61), but it is not proven here.
    Global recentring is a single congruence (an AIRM isometry), so relative
    between-subject structure — the class axis — should survive.
  * Identity removal is a TWO-PHASE, frozen operation, never a per-epoch PCA:
        Phase 1: train encoder + head, NO removal -> freeze encoder.
        Phase 2: estimate V ONCE on the frozen LogEig subject-means, verify the
                 removed PCs are class-free (PCA-AUC, as for shallow nr), apply
                 I-VVt, refit ONLY the linear head.
    This kills the weights->V->projection feedback loop and turns identity removal
    into a clean controlled ablation (Phase 1 vs Phase 2 IS the test).

Ablation ladder (each rung = one question), via CLI flags --streams / --identity-removal:
    rung 1  --streams cov                       does a learned SPD encoder match/collapse?
    rung 2  --streams cov  --identity-removal    does removal help in the LEARNED space?
    rung 3  --streams corr --identity-removal    cov vs corr in the deep space
    rung 4  --streams cov,corr --identity-removal does fusing manifolds recover bd's loss?

Everything runs float64 on CPU (SPD eigen-ops want double for stability; no conv =
no GPU benefit).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch
import torch.nn as nn
from pyriemann.estimation import Covariances
from sklearn.cluster import AgglomerativeClustering

from . import config as C
from .data import Subject, epoch
from .models import _cov_to_corr, _bandpass, _BANDS

_TSMNET = C.PROJECT_ROOT / "example code" / "TSMNet"
if str(_TSMNET) not in sys.path:
    sys.path.insert(0, str(_TSMNET))
import spdnets.modules as M          # noqa: E402  BiMap, ReEig, LogEig
import spdnets.batchnorm as BN       # noqa: E402  AdaMomSPDBatchNorm
from geoopt.optim import RiemannianAdam  # noqa: E402

_DTYPE = torch.double


# ===========================================================================
# Network
# ===========================================================================
class SPDStream(nn.Module):
    """One manifold: BiMap(in->d) -> ReEig -> GLOBAL SPD-BN -> LogEig -> tangent vec."""

    def __init__(self, in_dim: int, subspace: int):
        super().__init__()
        self.spd = nn.Sequential(
            M.BiMap((1, in_dim, subspace), dtype=_DTYPE, device=torch.device("cpu")),
            M.ReEig(threshold=1e-4),
        )
        self.bn = BN.AdaMomSPDBatchNorm(
            (1, subspace, subspace), batchdim=0, learn_mean=False, learn_std=True,
            dispersion=BN.BatchNormDispersion.SCALAR, eta=1.0, eta_test=0.1,
            dtype=_DTYPE, device=torch.device("cpu"))
        self.logeig = M.LogEig(subspace)                  # -> (B, d(d+1)/2)
        self.tsdim = subspace * (subspace + 1) // 2

    def forward(self, X: torch.Tensor) -> torch.Tensor:   # X: (B, in, in) SPD double
        return self.logeig(self.bn(self.spd(X)))


class DualSPDNet(nn.Module):
    """One or two SPD streams -> concat -> [frozen I-VVt] -> dropout -> Linear(2)."""

    def __init__(self, in_dim: int, subspace: int, stream_names: tuple[str, ...],
                 dropout: float = 0.5):
        super().__init__()
        self.stream_names = tuple(stream_names)
        self.enc = nn.ModuleDict({s: SPDStream(in_dim, subspace) for s in stream_names})
        self.feat_dim = sum(self.enc[s].tsdim for s in stream_names)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.feat_dim, 2).to(_DTYPE)
        self._V: torch.Tensor | None = None               # frozen identity subspace (k, feat)

    def features(self, mats: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([self.enc[s](mats[s]) for s in self.stream_names], dim=1)

    def remove_identity(self, F: torch.Tensor) -> torch.Tensor:
        if self._V is None:
            return F
        return F - (F @ self._V.t()) @ self._V

    def forward(self, mats: dict[str, torch.Tensor]) -> torch.Tensor:
        F = self.remove_identity(self.features(mats))
        return self.classifier(self.dropout(F))


# ===========================================================================
# Stage wrapper (two-phase)
# ===========================================================================
def _loo_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank AUC of a 1-D subject-level score vs binary labels (Mann-Whitney)."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    n1 = int((labels == 1).sum()); n0 = int((labels == 0).sum())
    if n1 == 0 or n0 == 0:
        return 0.5
    return (ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


class DualSPDStage:
    """Two-phase dual-manifold SPD stage for the ablation ladder.

    streams           : ("cov",) | ("corr",) | ("cov","corr")
    identity_removal  : if True, run Phase 2 (freeze encoder, remove top-K identity
                        PCs of the learned LogEig subject-means, refit the head only).
    n_nuisance        : K identity directions removed in Phase 2.
    """

    _COV_CACHE: dict[tuple, np.ndarray] = {}

    def __init__(self, streams=("cov", "corr"), subspace: int = 16,
                 identity_removal: bool = False, n_nuisance: int = 3,
                 n_clusters: int = 0, band: tuple | None = None,
                 epochs: int = 80, head_epochs: int = 120, batch_size: int = 128,
                 lr: float = 1e-3, head_lr: float = 5e-3, weight_decay: float = 1e-4,
                 dropout: float = 0.5, val_frac: float = 0.15, patience: int = 12,
                 max_windows: int | None = 120,
                 device: str | None = None, seed: int = C.SEED, verbose: bool = False):
        self.streams = tuple(streams)
        self.subspace = subspace
        self.n_clusters = n_clusters          # 0 = full 125x125; K>0 = AIRM-cluster 125->K
        self._P = None                        # learned pooling matrix (K, 125), per fold
        self.band = band                      # (lo, hi) Hz for bandpass, or None = broadband
        self.identity_removal = identity_removal
        self.n_nuisance = n_nuisance
        self.epochs = epochs
        self.head_epochs = head_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.head_lr = head_lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.val_frac = val_frac
        self.patience = patience
        self.max_windows = max_windows
        self.seed = seed
        self.verbose = verbose
        self._estimator = Covariances(estimator=C.COV_ESTIMATOR)

    # -- raw 125x125 covariances per subject (fold-independent -> cache once) --
    def _raw_covs(self, sub: Subject) -> np.ndarray:
        key = (sub.sid, self.max_windows, self.band)
        c = self._COV_CACHE.get(key)
        if c is None:
            wins = epoch(sub.data, max_windows=self.max_windows)
            if self.band is not None:
                wins = _bandpass(wins, self.band[0], self.band[1])
            c = self._estimator.transform(wins).astype(np.float64)       # (n,125,125)
            self._COV_CACHE[key] = c
        return c

    # -- functional-connectivity channel clustering 125->K (TRAIN subjects only) --
    #    NOTE: this clusters channels by CORRELATION affinity (1 - mean corr), an
    #    approximation of EEG-RCformer's clustering — it is NOT the AIRM geodesic
    #    distance d(C1,C2)=||log(C1^-1/2 C2 C1^-1/2)||_F between SPD matrices.
    def _pooling_matrix(self, train_subjects: list[Subject]) -> np.ndarray:
        R = np.mean([_cov_to_corr(self._raw_covs(s)).mean(axis=0) for s in train_subjects],
                    axis=0)                                              # mean connectivity
        dist = 1.0 - R; np.fill_diagonal(dist, 0.0)                     # positive-corr -> close
        labels = AgglomerativeClustering(n_clusters=self.n_clusters, metric="precomputed",
                                         linkage="average").fit_predict(dist)
        P = np.zeros((self.n_clusters, R.shape[0]))
        for k in range(self.n_clusters):
            idx = np.where(labels == k)[0]
            P[k, idx] = 1.0 / max(1, len(idx))                         # average-pool the region
        return P

    # -- stream matrices (optionally cluster-reduced to KxK via the fold's P) --
    def _mats(self, sub: Subject) -> dict[str, torch.Tensor]:
        covs = self._raw_covs(sub)
        if self.n_clusters > 0 and self._P is not None:
            covs = self._P @ covs @ self._P.T                          # (n, K, K)
        out = {}
        if "cov" in self.streams:
            out["cov"] = torch.from_numpy(np.ascontiguousarray(covs))
        if "corr" in self.streams:
            out["corr"] = torch.from_numpy(_cov_to_corr(covs))
        return out

    def _class_weight(self, y: np.ndarray) -> torch.Tensor:
        counts = np.bincount(y, minlength=2).astype(np.float64)
        w = counts.sum() / (2.0 * np.clip(counts, 1, None))
        return torch.tensor(w, dtype=_DTYPE)

    def fit(self, train_subjects: list[Subject]) -> "DualSPDStage":
        t0 = time.time()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        # learn the (correlation-affinity) channel-clustering pooling, train-only,
        # BEFORE building matrices
        self._P = self._pooling_matrix(train_subjects) if self.n_clusters > 0 else None

        # stack all training windows per stream + labels + subject index
        per_sub = [self._mats(s) for s in train_subjects]
        mats_all = {s: torch.cat([m[s] for m in per_sub], dim=0) for s in self.streams}
        y = np.concatenate([np.full(len(per_sub[i][self.streams[0]]),
                                    train_subjects[i].label) for i in range(len(train_subjects))])
        sub_idx = np.concatenate([np.full(len(per_sub[i][self.streams[0]]), i)
                                  for i in range(len(train_subjects))])
        N = len(y)
        in_dim = mats_all[self.streams[0]].shape[-1]
        eff_subspace = min(self.subspace, in_dim)      # BiMap needs subspace <= in_dim (KxK)
        yt = torch.from_numpy(y).long()
        weight = self._class_weight(y)

        self.net = DualSPDNet(in_dim, eff_subspace, self.streams, dropout=self.dropout)
        loss_fn = nn.CrossEntropyLoss(weight=weight)

        # inner-validation split (stratified by label) for early stopping — the encoder
        # memorises ~50 subjects in a few epochs, so we keep the best-generalising state.
        labs = np.array([s.label for s in train_subjects])
        val_subj = set()
        if self.val_frac > 0 and len(train_subjects) >= 6:
            for c in (0, 1):
                ci = np.where(labs == c)[0]; rng.shuffle(ci)
                nv = max(1, int(round(self.val_frac * len(ci))))
                val_subj.update(ci[:nv].tolist())
        val_mask = np.isin(sub_idx, list(val_subj))
        tr_idx = np.where(~val_mask)[0]
        va_idx = np.where(val_mask)[0]

        # ---------- Phase 1: train encoder + head, NO identity removal ----------
        opt = RiemannianAdam(self.net.parameters(), lr=self.lr,
                             weight_decay=self.weight_decay, stabilize=10)
        best_val, best_state, bad = float("inf"), None, 0
        for ep in range(self.epochs):
            self.net.train()
            order = rng.permutation(tr_idx)
            tot = 0.0
            for i in range(0, len(order), self.batch_size):
                bi = torch.from_numpy(order[i:i + self.batch_size]).long()
                mb = {s: mats_all[s][bi] for s in self.streams}
                opt.zero_grad()
                loss = loss_fn(self.net(mb), yt[bi])
                loss.backward()
                opt.step()
                tot += loss.item() * len(bi)
            # early-stopping check on the held-out inner-val subjects
            vloss = float("nan")
            if len(va_idx):
                self.net.eval()
                with torch.no_grad():
                    vl = torch.cat([self.net({s: mats_all[s][va_idx[j:j + 512]]
                                              for s in self.streams})
                                    for j in range(0, len(va_idx), 512)], dim=0)
                    vloss = loss_fn(vl, yt[va_idx]).item()
                if vloss < best_val - 1e-4:
                    best_val, bad = vloss, 0
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
            if self.verbose and (ep + 1) % 20 == 0:
                print(f"      [phase1] epoch {ep+1:>3}/{self.epochs}  "
                      f"train={tot/max(1,len(tr_idx)):.4f}  val={vloss:.4f}  best={best_val:.4f}")
            if len(va_idx) and bad >= self.patience:
                if self.verbose:
                    print(f"      [phase1] early stop @ epoch {ep+1} (best val={best_val:.4f})")
                break
        if best_state is not None:                              # restore best-generalising state
            self.net.load_state_dict(best_state)

        # ---------- Phase 2: frozen identity removal + head refit ----------
        if self.identity_removal:
            self.net.eval()
            for p in self.net.enc.parameters():           # freeze the encoder
                p.requires_grad_(False)
            with torch.no_grad():                          # learned features (frozen enc)
                F = torch.cat([self.net.features({s: mats_all[s][i:i + 512]
                                                  for s in self.streams})
                               for i in range(0, N, 512)], dim=0)        # (N, feat)
            Fnp = F.numpy()
            subj_means = np.stack([Fnp[sub_idx == i].mean(axis=0)
                                   for i in range(len(train_subjects))])  # (n_subj, feat)
            subj_lab = np.array([s.label for s in train_subjects])
            centred = subj_means - subj_means.mean(axis=0)
            k = min(self.n_nuisance, centred.shape[0] - 1)
            _, _, Vt = np.linalg.svd(centred, full_matrices=False)

            # verify the removed PCs are class-free in the LEARNED space (as for nr)
            pcs = subj_means @ Vt.T                          # (n_subj, n_pc)
            aucs = [round(_loo_auc(pcs[:, j], subj_lab), 3) for j in range(min(k + 3, pcs.shape[1]))]
            print(f"      [phase2] removed PC class-AUC (PC1..{len(aucs)}): {aucs}  "
                  f"(want first {k} ~0.5 = identity, not diagnosis)", flush=True)

            self.net._V = torch.from_numpy(Vt[:k]).to(_DTYPE)            # frozen (k, feat)
            removed = F - (F @ self.net._V.t()) @ self.net._V

            # refit ONLY the linear head on the identity-removed frozen features
            head = nn.Linear(self.net.feat_dim, 2).to(_DTYPE)
            self.net.classifier = head
            hopt = torch.optim.Adam(head.parameters(), lr=self.head_lr,
                                    weight_decay=self.weight_decay)
            for ep in range(self.head_epochs):
                order = rng.permutation(N)
                for i in range(0, N, self.batch_size):
                    bi = torch.from_numpy(order[i:i + self.batch_size]).long()
                    hopt.zero_grad()
                    # functional dropout with training=True (net.eval() disables the module)
                    fin = nn.functional.dropout(removed[bi], p=self.dropout, training=True)
                    loss = loss_fn(fin @ head.weight.t() + head.bias, yt[bi])
                    loss.backward()
                    hopt.step()

        self.fit_s = time.time() - t0
        return self

    @torch.no_grad()
    def predict_subject(self, sub: Subject) -> tuple[float, int]:
        self.net.eval()
        mats = self._mats(sub)
        logits = self.net(mats)                            # (n, 2)
        probs = torch.softmax(logits, dim=1)[:, 1]
        meanp = float(probs.mean())
        # aggregation diagnostic (the reviewer's #4): compare mean-prob / median / mean-logit
        medp = float(probs.median())
        meanlogit = float((logits[:, 1] - logits[:, 0]).mean())
        print(f"      [ds sid={sub.sid} y={sub.label}] meanp={meanp:.3f} "
              f"medp={medp:.3f} meanlogit={meanlogit:+.3f}", flush=True)
        return meanp, len(probs)


def build_dualspd(stage: str, **kw) -> DualSPDStage:
    return DualSPDStage(**kw)


# ===========================================================================
# Multiband dual-SPD ensemble — per-band DualSPDStage, decision-level fusion
# ===========================================================================
class MultibandDualSPDStage:
    """5 per-band DualSPD heads (no channel clustering), scores averaged.

    Each band runs a full DualSPDStage with early stopping independently.
    Decision-level mean fusion — avoids the dimensionality explosion of
    concatenating 5× learned feature vectors (same lesson as shallow mb).
    """

    def __init__(self, bands=("delta", "theta", "alpha", "beta", "gamma"),
                 streams=("cov", "corr"), n_nuisance: int = 3, subspace: int = 16,
                 epochs: int = 80, max_windows: int | None = 120,
                 lr: float = 1e-3, device: str | None = None,
                 verbose: bool = False, seed: int = C.SEED):
        self.band_names = list(bands)
        self.heads = [
            DualSPDStage(
                streams=streams, n_nuisance=n_nuisance, subspace=subspace,
                n_clusters=0,
                band=(_BANDS[b] if isinstance(b, str) else tuple(b)),
                epochs=epochs, max_windows=max_windows, lr=lr,
                device=device, verbose=verbose, seed=seed,
            )
            for b in bands
        ]

    def fit(self, train_subjects: list) -> "MultibandDualSPDStage":
        t0 = time.time()
        for name, h in zip(self.band_names, self.heads):
            print(f"  [mbds] fitting band={name} ...", flush=True)
            h.fit(train_subjects)
        self.fit_s = time.time() - t0
        return self

    def predict_subject(self, sub) -> tuple[float, int]:
        out = [h.predict_subject(sub) for h in self.heads]
        scores = np.array([o[0] for o in out])
        print(f"      [mbds sid={sub.sid} y={sub.label}] " +
              " ".join(f"{n}={s:.2f}" for n, s in zip(self.band_names, scores)) +
              f"  mean={scores.mean():.3f}", flush=True)
        return float(scores.mean()), out[0][1]


def build_mbds(**kw) -> MultibandDualSPDStage:
    return MultibandDualSPDStage(**kw)
