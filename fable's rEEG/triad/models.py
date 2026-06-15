"""Stage models. Each exposes the same minimal interface so the harness is shared:

    model.fit(train_subjects)            # list[Subject] with labels
    score = model.predict_subject(sub)   # -> P(MDD) in [0, 1] for one held-out subject

Stages implemented here (roadmap S0..S3):
  S0  EEGNetStage   — Euclidean CNN floor (torch). 1 loss (class-weighted CE).
  S1  ShallowStage(rep='cov',  recenter=False)  — covariance + tangent-space LR.
  S2  ShallowStage(rep='corr', recenter=False)  — correlation manifold (scale-invariant).
  S3  ShallowStage(rep='corr', recenter=True)   — + per-subject Frechet-mean recentring
                                                  (cheapest, one-shot transduction).

The shallow stages share precomputed per-subject covariances so that only the
fold-dependent work (tangent reference + logistic regression) is repeated per fold.
Per-subject recentring is fold-independent (a subject is centred by its OWN mean), so
it too is precomputed once.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.mean import mean_riemann as _mean_riemann
def mean_riemann(X, **kw): return _mean_riemann(X, maxiter=500, **kw)
from pyriemann.utils.tangentspace import tangent_space
from scipy.signal import butter, detrend as _detrend, filtfilt, welch
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from . import config as C
from .data import Subject, epoch


# ===========================================================================
# Shallow Riemannian stages (S1 / S2 / S3)
# ===========================================================================
def _cov_to_corr(covs: np.ndarray) -> np.ndarray:
    """Normalise each SPD covariance to a correlation matrix (scale-invariant).

    Re-symmetrised after the division: mathematically R == Rᵀ already, but float
    rounding can introduce tiny asymmetry, and downstream SPD eigen-ops (ReEig/
    LogEig/tangent) assume strict symmetry.
    """
    d = np.sqrt(np.diagonal(covs, axis1=1, axis2=2))      # (n, ch)
    denom = d[:, :, None] * d[:, None, :]
    R = covs / np.clip(denom, 1e-12, None)
    return 0.5 * (R + np.swapaxes(R, -1, -2))


def _recenter(mats: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Whiten each matrix by a reference: M^{-1/2} X M^{-1/2}."""
    p = invsqrtm(ref)
    return p @ mats @ p


def _bandpass(wins: np.ndarray, lo: float, hi: float, fs: float = C.FS,
              order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the time axis of (n, ch, T) windows."""
    b, a = butter(order, [lo, hi], btype="band", fs=fs)
    return filtfilt(b, a, wins, axis=-1).astype(np.float32)


_DE_BANDS = [(1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)]


def _compute_de(wins: np.ndarray) -> np.ndarray:
    """Differential Entropy per channel per band from broadband windows.

    DE of a Gaussian signal = 0.5 * log(2*pi*e * variance).
    wins : (N, C, T) broadband EEG windows (float32/64)
    Returns : (N, C*5) float64  — delta/theta/alpha/beta/gamma stacked
    """
    parts = []
    for lo, hi in _DE_BANDS:
        filtered = _bandpass(wins, lo, hi)                      # (N, C, T)
        var = np.var(filtered.astype(np.float64), axis=-1)     # (N, C)
        de = 0.5 * np.log(2.0 * np.pi * np.e * np.clip(var, 1e-10, None))
        parts.append(de)
    return np.concatenate(parts, axis=-1)                       # (N, C*5)


def _compute_spectral(wins: np.ndarray) -> np.ndarray:
    """Per-window spectral feature vector = aperiodic slope + relative + absolute band power.

    Replicates SPARC-Net's proven spectral branch (model/spectra/spectra_data.py): the
    per-channel aperiodic 1/f exponent is the documented ds003478 marker (Cohen d=0.506,
    p=0.003 — an E/I-balance signal invisible to the scale-invariant correlation manifold),
    while relative/absolute band power carries the Mumtaz/ds003478 power axis. The aperiodic
    slope is an OLS fit of log10(PSD) vs log10(f) over 2-40 Hz (negated -> positive exponent).

    wins : (N, C, T) broadband windows at C.FS.
    Returns : (N, C*(1 + 5 + 5)) float64 — [slope | relative bp (5) | abs log bp (5)] per ch.
    """
    fs = C.FS
    nperseg = min(wins.shape[-1], int(fs))
    f, P = welch(wins.astype(np.float64), fs=fs, nperseg=nperseg, axis=-1)   # (N, C, nf)
    # --- aperiodic exponent: OLS slope of log-log PSD over 2-40 Hz (vectorised) ---
    m = (f >= 2.0) & (f <= 40.0)
    lf = np.log10(f[m] + 1e-12)                                 # (nfm,)
    lP = np.log10(P[:, :, m] + 1e-24)                           # (N, C, nfm)
    lf_c = lf - lf.mean()
    slope = -((lP - lP.mean(-1, keepdims=True)) * lf_c).sum(-1) / (lf_c ** 2).sum()  # (N, C)
    # --- band power: relative + absolute (log) ---
    bp = np.stack([P[:, :, (f >= lo) & (f < hi)].sum(-1) for lo, hi in _DE_BANDS], axis=-1)
    rel = bp / (bp.sum(-1, keepdims=True) + 1e-24)              # (N, C, 5) relative
    logabs = np.log10(bp + 1e-12)                              # (N, C, 5) absolute log power
    N, Cc = slope.shape
    return np.concatenate([slope[:, :, None], rel, logabs], axis=-1).reshape(N, Cc * 11)


def _rank_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney rank AUC of subject scores s vs binary labels y."""
    y = np.asarray(y); s = np.asarray(s)
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else 0.5


class ShallowStage:
    """Hand-crafted SPD / correlation features + tangent-space logistic regression.

    Parameters
    ----------
    rep        : 'cov' (S1) or 'corr' (S2/S3).
    recenter   : if True (S3), each subject's matrices are recentred by that subject's
                 own Frechet mean before tangent projection — including the unlabelled
                 held-out subject (one-shot transduction).
    detrend    : if True (Option 1, 'n1'), linearly detrend each window per channel before
                 the covariance — stacks slow-drift nuisance removal on top of correlation.
    clf        : 'lr'       — tangent-space logistic regression (S1/S2/S3); or
                 'lda_subj' — Option 2 ('cc'): PCA then *subject-level* shrinkage-LDA.
                 LDA's within-class scatter is estimated from the per-SUBJECT mean tangent
                 vectors, so it suppresses between-subject within-class variance (individual
                 identity) while projecting onto the MDD/HC axis. Learned from training only
                 and applied identically to the held-out subject -> no train/test mismatch.
    n_nuisance : if > 0 ('nr'), project every tangent vector orthogonal to the top-K
                 between-subject identity directions before the LR. The directions are the
                 leading PCs of the TRAINING subject-mean cloud (estimated on train only).
                 modma_analysis showed PC1-5 of that cloud carry no class signal (LOO-AUC
                 <=0.44) yet dominate variance, so removing them strips identity nuisance
                 while preserving the low-variance MDD/HC axis the classifier needs.
    add_logpower : if True ('cr'), append per-channel log-power log(diag(C)) to the
                 correlation tangent vector. corr = cov with the diagonal normalised out, so
                 log(diag(C)) is *exactly* the information corr discards (per-channel power,
                 an amplitude biomarker). It is z-scored on TRAIN windows so the two blocks
                 are commensurate. Power is also the dominant identity confound, so this is
                 meant to run WITH n_nuisance>0: identity removal strips the nuisance part of
                 power, keeping any class-discriminative power direction.
    max_windows: cap windows per subject (speed); None uses all.
    """

    # Shared across model instances so per-subject features (which are fold-independent)
    # are computed once, not once per LOSO fold.
    _FEATURE_CACHE: dict[tuple, dict] = {}

    def __init__(self, rep: str = "cov", recenter: bool = False, detrend: bool = False,
                 clf: str = "lr", n_pca: int = 50, n_nuisance: int = 0,
                 k_grid: list | None = None,
                 add_logpower: bool = False, standardize: bool = False,
                 both_tangent: bool = False, band: tuple | None = None,
                 use_de: bool = False,
                 transductive: bool = False, n_iter: int = 0,
                 protect_class: bool = False, control: str | None = None,
                 dataset_recenter: bool = False, de_dataset_standardize: bool = False,
                 max_windows: int | None = 120, C_reg: float = 1.0, seed: int = C.SEED):
        assert rep in ("cov", "corr")
        assert clf in ("lr", "lda_subj")
        assert control in (None, "inductive", "random_inject")
        self.rep = rep
        self.recenter = recenter
        self.detrend = detrend
        self.clf = clf
        self.n_pca = n_pca
        self.n_nuisance = n_nuisance
        self.k_grid = k_grid                  # if set: pick n_nuisance by inner-CV per fold
        self.add_logpower = add_logpower
        self.standardize = standardize
        self.both_tangent = both_tangent
        self.band = band                      # (lo, hi) Hz band-pass before cov; None=broadband
        self.use_de = use_de                  # append DE (125ch × 5bands) after identity removal
        # --- per-DATASET DE standardisation (LODO spectral alignment) ---
        # z-score the DE block per source dataset (not pooled); at test, use the TARGET
        # dataset's own scaler (fit transductively on its unlabelled DE via set_target).
        # Removes per-dataset amplifier-gain band-power offsets — the spectral analog of
        # recentring, but on the stream that actually transfers cross-dataset.
        self.de_dataset_standardize = de_dataset_standardize
        self._de_scalers: dict = {}
        self._de_target_scaler = None
        # --- transductive nuisance-subspace adaptation (stage 'tnr') ---
        self.transductive = transductive      # use held-out subject's UNLABELLED windows in V
        self.n_iter = n_iter                  # refinement iterations (0 = init-only, the base)
        self.protect_class = protect_class    # orthogonalise V against the train class axis
        self.control = control                # None | 'inductive' | 'random_inject' (ablations)
        # --- per-DATASET Riemannian recentring (LODO alignment, stage flag) ---
        # Map each dataset's Frechet mean to the identity before the shared tangent
        # projection (the universal cross-domain primitive). Source means come from train;
        # the target dataset mean is set transductively from the held-out dataset's
        # UNLABELLED windows via set_target_reference(). No labels are used.
        self.dataset_recenter = dataset_recenter
        self._ds_means: dict = {}             # source dataset -> Frechet mean (set in fit)
        self._target_mean = None              # target dataset mean (set before predict)
        self.max_windows = max_windows
        self.C_reg = C_reg
        self.seed = seed
        self._nuis = None
        self._feat_mu = self._feat_sd = None
        self._ref_cov = self._ref_corr = None
        self._de_scaler = None                # StandardScaler for DE features (train-fit only)
        # transductive state (set in fit, consumed in predict_subject)
        self._train_means = self._train_X = self._train_y = self._class_dir = None
        self._estimator = Covariances(estimator=C.COV_ESTIMATOR)

    # -- per-subject feature construction (computed once, reused across folds) --
    def _features(self, sub: Subject) -> dict:
        # Key by the subject's OWN dataset (not the globally-active one) + its channel count,
        # so pooled cross-dataset (LODO) fits never collide on a shared sid or confuse a
        # harmonised 17-ch signal with a native-montage one.
        ds = getattr(sub, "dataset", "") or C.DATASET
        key = (ds, sub.data.shape[0], sub.sid, self.rep, self.recenter, self.detrend,
               self.max_windows, self.both_tangent, self.band, self.use_de)
        cached = self._FEATURE_CACHE.get(key)
        if cached is not None:
            return cached
        wins = epoch(sub.data, max_windows=self.max_windows)      # (n, ch, T)
        wins_broad = wins                                         # keep broadband for DE
        if self.band is not None:                                 # frequency-band sub-filter
            wins = _bandpass(wins, self.band[0], self.band[1])
        if self.detrend:
            wins = _detrend(wins, axis=2, type="linear").astype(np.float32)
        covs = self._estimator.transform(wins).astype(np.float64)  # (n, ch, ch)
        mats = _cov_to_corr(covs) if self.rep == "corr" else covs
        if self.recenter:
            subj_mean = mean_riemann(mats)
            mats = _recenter(mats, subj_mean)
        # per-channel log-power = exactly the info corr drops vs cov (diag of covariance)
        logpow = np.log(np.clip(np.diagonal(covs, axis1=1, axis2=2), 1e-12, None))
        feat = {"mats": mats, "label": sub.label, "mean": mean_riemann(mats),
                "logpow": logpow,
                "de": _compute_de(wins_broad) if self.use_de else None}
        if self.both_tangent:
            # keep BOTH SPD views for the block-diagonal product-manifold tangent
            # (cross-block is exactly 0 under AIRM, so they combine without mixing)
            corr = mats if self.rep == "corr" else _cov_to_corr(covs)
            feat["cov"] = covs
            feat["corr"] = corr
            feat["cov_mean"] = feat["mean"] if self.rep == "cov" else mean_riemann(covs)
            feat["corr_mean"] = feat["mean"] if self.rep == "corr" else mean_riemann(corr)
        self._FEATURE_CACHE[key] = feat
        return feat

    def fit(self, train_subjects: list[Subject]) -> "ShallowStage":
        t0 = time.time()
        feats = [self._features(s) for s in train_subjects]
        if self.dataset_recenter:
            assert self.clf == "lr" and not self.both_tangent and not self.transductive, \
                "dataset_recenter supports the plain LR path (nr/nrde) only"
            mats_list, mean_list = self._fit_dataset_recenter(train_subjects, feats)
        else:
            mats_list = [f["mats"] for f in feats]
            mean_list = [f["mean"] for f in feats]
        # tangent reference = Riemannian mean of training subjects' (recentred) means
        self._ref = mean_riemann(np.stack(mean_list))
        Xs = [tangent_space(m, self._ref) for m in mats_list]

        if self.both_tangent:
            # block-diagonal product manifold: each block tangent-mapped at its OWN
            # reference (so the two views are commensurate by construction, no z-score)
            self._ref_cov = mean_riemann(np.stack([f["cov_mean"] for f in feats]))
            self._ref_corr = mean_riemann(np.stack([f["corr_mean"] for f in feats]))
            Xs = [np.concatenate([tangent_space(f["cov"], self._ref_cov),
                                  tangent_space(f["corr"], self._ref_corr)], axis=1)
                  for f in feats]

        if self.add_logpower:
            Xs = [np.concatenate([x, f["logpow"]], axis=1) for x, f in zip(Xs, feats)]
        if self.standardize:
            # per-dimension z-score on TRAIN windows so all feature blocks are commensurate
            # (so the identity-subspace PCA and LR weight them on equal footing)
            allX = np.concatenate(Xs, axis=0)
            self._feat_mu = allX.mean(axis=0)
            self._feat_sd = allX.std(axis=0) + 1e-8
            Xs = [(x - self._feat_mu) / self._feat_sd for x in Xs]

        if self.transductive:
            # Transductive path: V (and therefore the LR) is re-estimated per held-out
            # subject in predict_subject(), folding in that subject's UNLABELLED windows.
            # Here we only stash the train-only state it needs. No test label is ever used.
            self._train_means = np.stack([x.mean(axis=0) for x in Xs])          # (n_subj, d)
            self._train_X = np.concatenate(Xs, axis=0).astype(np.float32)        # all windows
            self._train_y = np.concatenate(
                [np.full(len(f["mats"]), f["label"]) for f in feats])
            labs = np.array([f["label"] for f in feats])
            # class axis from TRAIN labels only (leakage-free) — V is kept orthogonal to it
            self._class_dir = (self._train_means[labs == 1].mean(axis=0)
                               - self._train_means[labs == 0].mean(axis=0))
            self.fit_s = time.time() - t0
            return self

        if self.k_grid is not None and self.clf == "lr":
            # choose n_nuisance by inner-CV on the training subjects (leakage-free, #6)
            labels = np.array([f["label"] for f in feats])
            self.n_nuisance = self._select_k_inner(Xs, labels)

        if self.clf == "lr":
            if self.n_nuisance > 0:
                # Identity-subspace removal: leading PCs of the TRAINING subject-mean
                # cloud are the between-subject identity axes. Project them out of every
                # window so the LR sees only the residual where the class signal lives.
                subj_means = np.stack([x.mean(axis=0) for x in Xs])     # (n_subj, d)
                centred = subj_means - subj_means.mean(axis=0)
                k = min(self.n_nuisance, centred.shape[0] - 1)
                _, _, Vt = np.linalg.svd(centred, full_matrices=False)
                self._nuis = Vt[:k]                                     # (k, d) orthonormal
                Xs = [self._project(x) for x in Xs]
            if self.use_de:
                # Append DE features AFTER identity removal (V estimated from tangent only).
                from sklearn.preprocessing import StandardScaler
                self._de_scaler = StandardScaler().fit(np.vstack([f["de"] for f in feats]))
                if self.de_dataset_standardize:
                    # per-source-dataset scaler; transform each subject by ITS dataset's
                    by_ds: dict = {}
                    for s, f in zip(train_subjects, feats):
                        by_ds.setdefault(s.dataset, []).append(f["de"])
                    self._de_scalers = {ds: StandardScaler().fit(np.vstack(v))
                                        for ds, v in by_ds.items()}
                    de_blocks = [self._de_scalers[s.dataset].transform(f["de"])
                                 for s, f in zip(train_subjects, feats)]
                else:
                    de_blocks = [self._de_scaler.transform(f["de"]) for f in feats]
                Xs = [np.hstack([x, de]) for x, de in zip(Xs, de_blocks)]
            X = np.concatenate(Xs, axis=0)
            y = np.concatenate([np.full(len(f["mats"]), f["label"]) for f in feats])
            self._clf = LogisticRegression(
                C=self.C_reg, class_weight="balanced", max_iter=2000,
                solver="lbfgs", random_state=self.seed,
            ).fit(X, y)
        else:  # 'lda_subj' — class-conditional, identity-suppressing (Option 2)
            Xall = np.concatenate(Xs, axis=0)
            k = min(self.n_pca, Xall.shape[0], Xall.shape[1])
            self._pca = PCA(n_components=k, svd_solver="randomized",
                            random_state=self.seed).fit(Xall)
            subj_vecs = np.stack([self._pca.transform(x).mean(axis=0) for x in Xs])
            subj_lab = np.array([f["label"] for f in feats])
            self._lda = LinearDiscriminantAnalysis(
                solver="lsqr", shrinkage="auto").fit(subj_vecs, subj_lab)
        self.fit_s = time.time() - t0
        return self

    def _project(self, X: np.ndarray) -> np.ndarray:
        """Remove the components along the learned identity directions (I - VVᵀ)."""
        if self._nuis is None:
            return X
        return X - (X @ self._nuis.T) @ self._nuis

    def _fit_dataset_recenter(self, subjects, feats):
        """Recentre each source dataset to its own Frechet mean (-> identity).

        Returns (mats_list, mean_list): per-subject recentred correlation matrices and
        their means, for the shared tangent projection. The dataset mean is the Frechet
        mean of that dataset's per-subject means; because the AIRM Frechet mean is
        congruence-equivariant, recentring a subject's mean equals the mean of its
        recentred matrices, so no per-subject Frechet mean is recomputed.
        """
        from collections import defaultdict
        by_ds = defaultdict(list)
        for s, f in zip(subjects, feats):
            by_ds[s.dataset].append(f["mean"])
        self._ds_means = {ds: mean_riemann(np.stack(ms)) for ds, ms in by_ds.items()}
        mats_list, mean_list = [], []
        for s, f in zip(subjects, feats):
            M = self._ds_means[s.dataset]
            mats_list.append(_recenter(f["mats"], M))
            mean_list.append(_recenter(f["mean"][None], M)[0])   # equivariance (cheap)
        return mats_list, mean_list

    def set_target(self, test_subjects) -> None:
        """Transductive target adaptation for the held-out dataset's UNLABELLED windows.
        Under de_dataset_standardize, fit the DE scaler on the target's own DE so its
        band-power offset is removed the same way each source dataset's was (no labels)."""
        if self.use_de and self.de_dataset_standardize:
            from sklearn.preprocessing import StandardScaler
            de = np.vstack([self._features(s)["de"] for s in test_subjects])
            self._de_target_scaler = StandardScaler().fit(de)

    def set_target_reference(self, test_subjects) -> None:
        """Transductively set the target dataset Frechet mean from the held-out dataset's
        UNLABELLED windows (no labels used). Consumed by predict_subject under
        dataset_recenter. Called once by the LODO harness before prediction."""
        if not self.dataset_recenter:
            return
        feats = [self._features(s) for s in test_subjects]
        self._target_mean = mean_riemann(np.stack([f["mean"] for f in feats]))

    def _select_k_inner(self, Xs: list, labels: np.ndarray, n_inner: int = 5) -> int:
        """Leakage-free n_nuisance selection: inner stratified CV over the TRAIN subjects
        only. For each candidate K, estimate V on inner-train means, project, fit LR,
        score inner-val subjects; pick K with best pooled inner-val AUC. The held-out
        (outer) subject is never involved, so K is not tuned on test."""
        n = len(Xs)
        rng = np.random.default_rng(self.seed)
        fold_of = np.empty(n, int)                       # subject-level stratified folds
        for c in (0, 1):
            idx = np.where(labels == c)[0]; rng.shuffle(idx)
            fold_of[idx] = np.arange(len(idx)) % n_inner
        results = {}
        for K in self.k_grid:
            scores = np.full(n, np.nan)
            for f in range(n_inner):
                tr = np.where(fold_of != f)[0]; va = np.where(fold_of == f)[0]
                if len(va) == 0 or len(set(labels[tr].tolist())) < 2:
                    continue
                if K > 0:
                    sm = np.stack([Xs[i].mean(axis=0) for i in tr])
                    _, _, Vt = np.linalg.svd(sm - sm.mean(axis=0), full_matrices=False)
                    V = Vt[:min(K, len(tr) - 1)]
                    prj = lambda x: x - (x @ V.T) @ V
                else:
                    prj = lambda x: x
                Xtr = np.concatenate([prj(Xs[i]) for i in tr])
                ytr = np.concatenate([np.full(len(Xs[i]), labels[i]) for i in tr])
                lr = LogisticRegression(C=self.C_reg, class_weight="balanced",
                                        max_iter=2000, solver="lbfgs",
                                        random_state=self.seed).fit(Xtr, ytr)
                for i in va:
                    scores[i] = lr.predict_proba(prj(Xs[i]))[:, 1].mean()
            m = ~np.isnan(scores)
            results[K] = _rank_auc(labels[m], scores[m])
        best_k = max(results, key=results.get)
        print("      [nested-K] inner-CV AUC " +
              " ".join(f"K{k}={v:.3f}" for k, v in results.items()) +
              f"  -> K*={best_k}", flush=True)
        return best_k

    # ---- transductive nuisance-subspace adaptation (stage 'tnr') -------------
    @staticmethod
    def _orth_out(V: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Return an orthonormal basis spanning row-space(V) with the c-direction removed.

        Guarantees every returned row ⊥ c, so projecting data onto (I - VᵀV) can NOT
        remove the class axis c. c is the train-estimated MDD-HC direction. This is the
        provable safeguard against the nuisance subspace absorbing diagnosis."""
        c = c / (np.linalg.norm(c) + 1e-12)
        V = V - np.outer(V @ c, c)                 # strip c-component from each row
        # re-orthonormalise the rows (their span is still ⊥ c, since each row is)
        Q, _ = np.linalg.qr(V.T)                   # (d, k) orthonormal columns
        return Q.T[: V.shape[0]]

    def _estimate_V(self, cloud: np.ndarray) -> np.ndarray:
        """Top-K between-subject directions of a subject-descriptor cloud (n, d).

        n_iter=0 = a single SVD of the augmented cloud (the defensible base case).
        n_iter>0 re-projects the ORIGINAL cloud by the latest V and re-estimates, which
        refines *which* K directions are removed without compounding residuals."""
        k = min(self.n_nuisance, cloud.shape[0] - 1)
        desc, V = cloud, None
        for _ in range(self.n_iter + 1):
            centred = desc - desc.mean(axis=0)
            _, _, Vt = np.linalg.svd(centred, full_matrices=False)
            V = Vt[:k]
            if self.protect_class and self._class_dir is not None:
                V = self._orth_out(V, self._class_dir)
            desc = cloud - (cloud @ V.T) @ V
        return V

    @staticmethod
    def _mean_principal_angle(A: np.ndarray, B: np.ndarray) -> float:
        """Mean principal angle (radians) between two row-orthonormal subspaces."""
        s = np.linalg.svd(A @ B.T, compute_uv=False)
        return float(np.arccos(np.clip(s, 0.0, 1.0)).mean())

    def _predict_transductive(self, sub: Subject, X_test: np.ndarray) -> tuple[float, int]:
        # subject descriptor that gets folded into the nuisance-subspace estimate
        if self.control == "inductive":
            extra = None                                        # train-only V (= nr + protect)
        elif self.control == "random_inject":
            rng = np.random.default_rng(self.seed + int(sub.sid))
            extra = self._train_means[rng.integers(len(self._train_means))]
        else:
            extra = X_test.mean(axis=0)                          # the real held-out subject
        cloud = (self._train_means if extra is None
                 else np.vstack([self._train_means, extra[None]]))
        V = self._estimate_V(cloud)

        # ablation D: how far did the target pull V vs the train-only subspace?
        V_ind = self._estimate_V(self._train_means)
        drift = self._mean_principal_angle(V, V_ind)

        # project train + test onto (I - VᵀV), refit LR, score the held-out windows
        Xtr = self._train_X - (self._train_X @ V.T) @ V
        Xte = X_test - (X_test @ V.T) @ V
        clf = LogisticRegression(
            C=self.C_reg, class_weight="balanced", max_iter=2000,
            solver="lbfgs", random_state=self.seed,
        ).fit(Xtr, self._train_y)
        probs = clf.predict_proba(Xte)[:, 1]
        print(f"      [tnr sid={sub.sid} ctrl={self.control or 'transductive'}] "
              f"Vdrift={drift:.3f}rad  score={probs.mean():.3f}", flush=True)
        return float(probs.mean()), len(probs)

    def predict_subject(self, sub: Subject) -> tuple[float, int]:
        feat = self._features(sub)
        mats = feat["mats"]
        if self.dataset_recenter and self._target_mean is not None:
            # transductive: recentre the target dataset to identity (same as sources)
            mats = _recenter(mats, self._target_mean)
        X = tangent_space(mats, self._ref)
        if self.transductive:
            return self._predict_transductive(sub, X)
        if self.both_tangent:
            X = np.concatenate([tangent_space(feat["cov"], self._ref_cov),
                                tangent_space(feat["corr"], self._ref_corr)], axis=1)
        if self.add_logpower:
            X = np.concatenate([X, feat["logpow"]], axis=1)
        if self.standardize:
            X = (X - self._feat_mu) / self._feat_sd
        if self.clf == "lr":
            Xp = self._project(X)
            if self.use_de:
                de_scaler = (self._de_target_scaler
                             if (self.de_dataset_standardize and self._de_target_scaler is not None)
                             else self._de_scaler)
                Xp = np.hstack([Xp, de_scaler.transform(feat["de"])])
            probs = self._clf.predict_proba(Xp)[:, 1]
        else:
            probs = self._lda.predict_proba(self._pca.transform(X))[:, 1]
        return float(probs.mean()), len(probs)


def make_shallow(stage: str, **kw) -> ShallowStage:
    cfg = {"s1": dict(rep="cov", recenter=False),
           "s2": dict(rep="corr", recenter=False),
           "s3": dict(rep="corr", recenter=True),
           "n1": dict(rep="corr", recenter=False, detrend=True),            # Option 1
           "cc": dict(rep="corr", recenter=False, clf="lda_subj"),         # Option 2
           "nr": dict(rep="corr", recenter=False, n_nuisance=3),           # identity removal
           "nc": dict(rep="cov", recenter=False, n_nuisance=3),            # cov (= both) + removal
           "cr": dict(rep="corr", recenter=False, n_nuisance=3,            # corr + log-power
                      add_logpower=True, standardize=True),
           "bd": dict(rep="corr", recenter=False, n_nuisance=3,            # block-diag cov⊕corr
                      both_tangent=True),
           "tnr": dict(rep="corr", recenter=False, n_nuisance=3,           # transductive nr
                       transductive=True, protect_class=False, n_iter=0),
           "ncv":  dict(rep="corr", recenter=False,                        # nr + nested-K
                       k_grid=[0, 1, 2, 3, 5]),                            # audited-safe range
           "nrde": dict(rep="corr", recenter=False, n_nuisance=3,          # nr + DE power
                        use_de=True)}[stage]
    cfg.update(kw)
    return ShallowStage(**cfg)


# ===========================================================================
# Multiband ensemble ('mb') — per-band nr, decision-level score fusion
# ===========================================================================
_BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
          "beta": (13.0, 30.0), "gamma": (30.0, 45.0)}


class MultibandStage:
    """Per-frequency-band nr, fused at the decision level (mean of subject scores).

    Depression markers are band-specific (alpha asymmetry, theta elevation), but a
    single broadband correlation integrates them. Here we run an INDEPENDENT nr
    (correlation + identity-subspace removal) per band and AVERAGE the per-subject
    scores. Decision-level fusion (P5) deliberately avoids the 5x7875-dim feature
    concatenation that would overfit 52 subjects — the failure mode of bd.

    bands      : names from {delta,theta,alpha,beta,gamma} or explicit (lo,hi) tuples.
    n_nuisance : identity PCs removed per band (audited safe for K<=5).
    k_grid     : if set, each band selects its own K by inner-CV (nested, leakage-free).
    """

    def __init__(self, bands=("delta", "theta", "alpha", "beta", "gamma"),
                 n_nuisance: int = 3, k_grid: list | None = None,
                 max_windows: int | None = 120, C_reg: float = 1.0, seed: int = C.SEED):
        self.band_names = list(bands)
        self.bands = [_BANDS[b] if isinstance(b, str) else tuple(b) for b in bands]
        self.heads = [ShallowStage(rep="corr", recenter=False, n_nuisance=n_nuisance,
                                   k_grid=k_grid, band=bd, max_windows=max_windows,
                                   C_reg=C_reg, seed=seed) for bd in self.bands]

    def fit(self, train_subjects: list[Subject]) -> "MultibandStage":
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=2) as ex:
            list(ex.map(lambda h: h.fit(train_subjects), self.heads))
        self.fit_s = time.time() - t0
        return self

    def predict_subject(self, sub: Subject) -> tuple[float, int]:
        out = [h.predict_subject(sub) for h in self.heads]
        scores = np.array([o[0] for o in out])
        print(f"      [mb sid={sub.sid} y={sub.label}] " +
              " ".join(f"{n}={s:.2f}" for n, s in zip(self.band_names, scores)) +
              f"  mean={scores.mean():.3f}", flush=True)
        return float(scores.mean()), out[0][1]


# ===========================================================================
# DE-only classifier with identity removal
# ===========================================================================
class DEStage:
    """Differential Entropy (band-power) classifier with identity removal.

    Features: 125 ch × 5 bands = 625-dim DE per window (Euclidean, not SPD).
    Identity removal: same SVD approach as nr but estimated on DE subject means.
    Each stream is independent — safe to fuse with nr at decision level.
    """

    _DE_CACHE: dict[tuple, np.ndarray] = {}

    def __init__(self, n_nuisance: int = 3, max_windows: int | None = 120,
                 C_reg: float = 1.0, seed: int = C.SEED):
        self.n_nuisance = n_nuisance
        self.max_windows = max_windows
        self.C_reg = C_reg
        self.seed = seed
        self._nuis = None
        self._de_scaler = None
        self._clf = None

    def _get_de(self, sub: Subject) -> np.ndarray:
        key = (C.DATASET, sub.sid, self.max_windows)
        if key not in self._DE_CACHE:
            wins = epoch(sub.data, max_windows=self.max_windows)
            self._DE_CACHE[key] = _compute_de(wins)
        return self._DE_CACHE[key]

    def _project(self, X: np.ndarray) -> np.ndarray:
        if self._nuis is None:
            return X
        return X - (X @ self._nuis.T) @ self._nuis

    def fit(self, train_subjects: list[Subject]) -> "DEStage":
        from sklearn.preprocessing import StandardScaler
        t0 = time.time()
        de_list = [self._get_de(s) for s in train_subjects]
        labels = [s.label for s in train_subjects]
        all_de = np.vstack(de_list)
        self._de_scaler = StandardScaler().fit(all_de)
        de_list = [self._de_scaler.transform(d) for d in de_list]
        subj_means = np.stack([d.mean(axis=0) for d in de_list])
        centred = subj_means - subj_means.mean(axis=0)
        k = min(self.n_nuisance, centred.shape[0] - 1)
        _, _, Vt = np.linalg.svd(centred, full_matrices=False)
        self._nuis = Vt[:k]
        de_proj = [self._project(d) for d in de_list]
        X = np.vstack(de_proj)
        y = np.concatenate([np.full(len(d), lab) for d, lab in zip(de_proj, labels)])
        self._clf = LogisticRegression(
            C=self.C_reg, class_weight="balanced", max_iter=2000,
            solver="lbfgs", random_state=self.seed).fit(X, y)
        self.fit_s = time.time() - t0
        return self

    def predict_subject(self, sub: Subject) -> tuple[float, int]:
        de = self._de_scaler.transform(self._get_de(sub))
        probs = self._clf.predict_proba(self._project(de))[:, 1]
        return float(probs.mean()), len(probs)


class SpectralStage(DEStage):
    """Full spectral classifier (SPARC-Net spectral-branch analog).

    Same standardise → identity-removal → LR pipeline as DEStage, but the per-window
    feature is the richer spectral vector from ``_compute_spectral``: per-channel
    aperiodic 1/f slope + relative band power + absolute log band power. The aperiodic
    exponent is the documented ds003478 depression marker that pure band power (DE) and
    the scale-invariant correlation manifold both miss.
    """

    _DE_CACHE: dict[tuple, np.ndarray] = {}        # separate cache from DE band-power feats

    def _get_de(self, sub: Subject) -> np.ndarray:
        key = (C.DATASET, sub.sid, self.max_windows)
        if key not in self._DE_CACHE:
            wins = epoch(sub.data, max_windows=self.max_windows)
            self._DE_CACHE[key] = _compute_spectral(wins)
        return self._DE_CACHE[key]


# ===========================================================================
# Decision-level fusion: nr (connectivity) + DE (power) — each with own ID removal
# ===========================================================================
class DEFusionStage:
    """Fuse nr and DE scores at decision level.

    Stream 1 (nr):  correlation tangent + K=3 identity removal → LR → p_nr
    Stream 2 (de):  DE band-power      + K=3 identity removal → LR → p_de
    Final score  :  (p_nr + p_de) / 2

    Each stream has its own independently estimated identity subspace so DE
    nuisance (amplitude differences) and tangent nuisance are removed separately.
    """

    def __init__(self, n_nuisance: int = 3, max_windows: int | None = 120,
                 C_reg: float = 1.0, seed: int = C.SEED):
        self.nr_head = ShallowStage(rep="corr", recenter=False, n_nuisance=n_nuisance,
                                    max_windows=max_windows, C_reg=C_reg, seed=seed)
        self.de_head = DEStage(n_nuisance=n_nuisance, max_windows=max_windows,
                               C_reg=C_reg, seed=seed)

    def fit(self, train_subjects: list[Subject]) -> "DEFusionStage":
        t0 = time.time()
        self.nr_head.fit(train_subjects)
        self.de_head.fit(train_subjects)
        self.fit_s = time.time() - t0
        return self

    def predict_subject(self, sub: Subject) -> tuple[float, int]:
        nr_score, n_win = self.nr_head.predict_subject(sub)
        de_score, _ = self.de_head.predict_subject(sub)
        fused = (nr_score + de_score) / 2.0
        print(f"      [defu sid={sub.sid} y={sub.label}] "
              f"nr={nr_score:.3f}  de={de_score:.3f}  fused={fused:.3f}", flush=True)
        return fused, n_win


class SpectralFusionStage:
    """Decision-level fusion of nr connectivity + full spectral (SPARC-Net hybrid analog).

    Stream 1 (nr):    correlation tangent + identity removal → LR → p_nr
    Stream 2 (spec):  aperiodic 1/f + band power + identity removal → LR → p_spec
    Final score   :   (p_nr + p_spec) / 2

    The hybrid of connectivity (MODMA's signal) and spectral power/aperiodic (Mumtaz &
    ds003478's signal) is the representation that travels best across all three cohorts.
    """

    def __init__(self, n_nuisance: int = 3, max_windows: int | None = 120,
                 C_reg: float = 1.0, seed: int = C.SEED):
        self.nr_head = ShallowStage(rep="corr", recenter=False, n_nuisance=n_nuisance,
                                    max_windows=max_windows, C_reg=C_reg, seed=seed)
        self.sp_head = SpectralStage(n_nuisance=n_nuisance, max_windows=max_windows,
                                     C_reg=C_reg, seed=seed)

    def fit(self, train_subjects: list[Subject]) -> "SpectralFusionStage":
        t0 = time.time()
        self.nr_head.fit(train_subjects)
        self.sp_head.fit(train_subjects)
        self.fit_s = time.time() - t0
        return self

    def predict_subject(self, sub: Subject) -> tuple[float, int]:
        nr_score, n_win = self.nr_head.predict_subject(sub)
        sp_score, _ = self.sp_head.predict_subject(sub)
        fused = (nr_score + sp_score) / 2.0
        print(f"      [spfu sid={sub.sid} y={sub.label}] "
              f"nr={nr_score:.3f}  spec={sp_score:.3f}  fused={fused:.3f}", flush=True)
        return fused, n_win


# ===========================================================================
# MLP on nr tangent features  ('mlp')
# ===========================================================================
class MLPStage:
    """Small MLP classifier on top of the proven nr (correlation + identity removal) features.

    Pipeline:
      1. nr preprocessing: correlation tangent → identity removal (I−VVᵀ, K=n_nuisance)
      2. PCA reduction: 7875 → pca_dim  (regularises the high-dim input)
      3. MLP: pca_dim → hidden[0] → ... → 2  (window-level, early-stopped)
      4. Subject score: mean softmax probability over ~120 windows

    Subject-balanced early stopping: held-out inner validation subjects (stratified,
    val_frac=0.15) prevent memorising the ~52 training subjects — same as DualSPD.
    """

    def __init__(self, hidden: tuple = (64, 32), pca_dim: int = 64,
                 dropout: float = 0.5, lr: float = 1e-3, epochs: int = 150,
                 weight_decay: float = 1e-3, n_nuisance: int = 3,
                 max_windows: int | None = 120, val_frac: float = 0.15,
                 patience: int = 20, batch_size: int = 256, seed: int = C.SEED):
        self._prep = ShallowStage(rep="corr", recenter=False, n_nuisance=n_nuisance,
                                   max_windows=max_windows, seed=seed)
        self.hidden = tuple(hidden)
        self.pca_dim = pca_dim
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.val_frac = val_frac
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed
        self._pca = None
        self.net = None
        self.fit_s = 0.0

    def fit(self, train_subjects: list[Subject]) -> "MLPStage":
        import torch
        import torch.nn as nn
        from sklearn.decomposition import PCA

        t0 = time.time()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        # Fit nr preprocessing (reference mean + identity subspace V)
        self._prep.fit(train_subjects)

        # Get nr-projected tangent features for all training windows
        feats = [self._prep._features(s) for s in train_subjects]
        Xs = [self._prep._project(tangent_space(f["mats"], self._prep._ref))
              for f in feats]

        X_all = np.concatenate(Xs, axis=0).astype(np.float32)
        y_all = np.concatenate([np.full(len(f["mats"]), f["label"])
                                 for f in feats]).astype(np.int64)
        sub_idx = np.concatenate([np.full(len(Xs[i]), i)
                                   for i in range(len(train_subjects))])

        # PCA: 7875 → pca_dim  (reduces parameters, controls overfitting)
        k = min(self.pca_dim, X_all.shape[1], X_all.shape[0] - 1)
        self._pca = PCA(n_components=k, svd_solver="randomized",
                        random_state=self.seed).fit(X_all)
        X_all = self._pca.transform(X_all).astype(np.float32)

        # Stratified inner-validation split (by subject label, not window)
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

        # Build MLP
        in_dim = X_all.shape[1]
        dims = [in_dim] + list(self.hidden) + [2]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
        self.net = nn.Sequential(*layers).float()

        counts = np.bincount(y_all, minlength=2).astype(float)
        w = counts.sum() / (2.0 * np.clip(counts, 1, None))
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))

        Xt = torch.from_numpy(X_all)
        yt = torch.from_numpy(y_all)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        best_val, best_state, bad = float("inf"), None, 0

        for ep in range(self.epochs):
            self.net.train()
            order = rng.permutation(tr_idx)
            for i in range(0, len(order), self.batch_size):
                bi = torch.from_numpy(order[i:i + self.batch_size]).long()
                opt.zero_grad()
                loss_fn(self.net(Xt[bi]), yt[bi]).backward()
                opt.step()
            if len(va_idx):
                self.net.eval()
                with torch.no_grad():
                    vloss = loss_fn(self.net(Xt[va_idx]), yt[va_idx]).item()
                if vloss < best_val - 1e-4:
                    best_val, bad = vloss, 0
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
                if bad >= self.patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.fit_s = time.time() - t0
        return self

    def predict_subject(self, sub: Subject) -> tuple[float, int]:
        import torch
        feat = self._prep._features(sub)
        X = self._prep._project(tangent_space(feat["mats"], self._prep._ref))
        X = torch.from_numpy(self._pca.transform(X.astype(np.float32)))
        self.net.eval()
        with torch.no_grad():
            probs = torch.softmax(self.net(X), dim=1)[:, 1]
        return float(probs.mean()), len(probs)


# ===========================================================================
# Window attention  ('wattn') — multiple-instance learning on subject bags
# ===========================================================================
def _build_attention_pool(in_dim: int, attn_dim: int):
    """Returns an additive attention pooling module (lazy torch import)."""
    import torch
    import torch.nn as nn

    class _Pool(nn.Module):
        def __init__(self):
            super().__init__()
            self.W = nn.Linear(in_dim, attn_dim)
            self.v = nn.Linear(attn_dim, 1, bias=False)

        def forward(self, H):                      # H: (N_windows, in_dim)
            e = self.v(torch.tanh(self.W(H)))      # (N, 1)
            a = torch.softmax(e, dim=0)            # (N, 1) — over windows
            return (a * H).sum(dim=0), a.squeeze(1)

    return _Pool()


class WindowAttnStage:
    """Multiple-instance learning: attention pooling of window tangent features.

    Trains at the SUBJECT level (bag of windows → one label), not window level.
    The attention gate learns which windows are most diagnostic for MDD/HC.

    Pipeline:
      1. nr preprocessing  →  7875-dim projected tangent per window
      2. PCA               →  pca_dim-dim
      3. AttentionPool      →  weighted subject embedding (pca_dim-dim)
      4. Linear(pca_dim→2) →  subject-level classification

    Compared to MLP (window-level training), this correctly frames the problem
    as bag-level prediction: no window pseudo-labels, fewer parameters, and
    training signal comes from 52 subject labels (not 6240 window labels).
    """

    def __init__(self, attn_dim: int = 32, pca_dim: int = 64,
                 hidden_dim: int = 0,
                 dropout: float = 0.3, lr: float = 5e-4, epochs: int = 300,
                 weight_decay: float = 1e-3, n_nuisance: int = 3,
                 max_windows: int | None = 120, val_frac: float = 0.2,
                 patience: int = 30, batch_size: int = 8, seed: int = C.SEED):
        self._prep = ShallowStage(rep="corr", recenter=False, n_nuisance=n_nuisance,
                                   max_windows=max_windows, seed=seed)
        self.attn_dim = attn_dim
        self.pca_dim = pca_dim
        self.hidden_dim = hidden_dim   # 0 = direct linear(pca_dim→2); >0 adds hidden layer
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.val_frac = val_frac
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed
        self._pca = None
        self._pool = None
        self._clf_head = None
        self.fit_s = 0.0

    def fit(self, train_subjects: list[Subject]) -> "WindowAttnStage":
        import torch
        import torch.nn as nn
        from sklearn.decomposition import PCA

        t0 = time.time()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        # nr preprocessing
        self._prep.fit(train_subjects)
        feats = [self._prep._features(s) for s in train_subjects]
        Xs = [self._prep._project(tangent_space(f["mats"], self._prep._ref))
              for f in feats]

        # PCA fit on all training windows
        X_cat = np.concatenate(Xs, axis=0).astype(np.float32)
        k = min(self.pca_dim, X_cat.shape[1], X_cat.shape[0] - 1)
        self._pca = PCA(n_components=k, svd_solver="randomized",
                        random_state=self.seed).fit(X_cat)

        # Per-subject PCA-projected tensors (variable N_windows each)
        bags = [torch.from_numpy(self._pca.transform(x.astype(np.float32)))
                for x in Xs]                                         # list of (N_i, k)
        labels = torch.tensor([f["label"] for f in feats], dtype=torch.long)
        in_dim = k

        # Build attention pool + classifier head (optionally with hidden layer)
        self._pool = _build_attention_pool(in_dim, self.attn_dim)
        if self.hidden_dim > 0:
            self._clf_head = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, 2),
            )
        else:
            self._clf_head = nn.Linear(in_dim, 2)
        params = list(self._pool.parameters()) + list(self._clf_head.parameters())

        counts = np.bincount(labels.numpy(), minlength=2).astype(float)
        w = counts.sum() / (2.0 * np.clip(counts, 1, None))
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        # Inner-validation split (by subject, stratified)
        labs_np = labels.numpy()
        val_subj = set()
        if self.val_frac > 0 and len(train_subjects) >= 6:
            for c in (0, 1):
                ci = np.where(labs_np == c)[0]; rng.shuffle(ci)
                nv = max(1, int(round(self.val_frac * len(ci))))
                val_subj.update(ci[:nv].tolist())
        tr_idx = [i for i in range(len(train_subjects)) if i not in val_subj]
        va_idx = list(val_subj)

        best_val, best_pool, best_head, bad = float("inf"), None, None, 0

        for ep in range(self.epochs):
            # --- train ---
            self._pool.train(); self._clf_head.train()
            rng.shuffle(tr_idx)
            for i in range(0, len(tr_idx), self.batch_size):
                batch = tr_idx[i:i + self.batch_size]
                opt.zero_grad()
                zs = torch.stack([self._pool(bags[j])[0] for j in batch])
                zs = nn.functional.dropout(zs, p=self.dropout, training=True)
                loss_fn(self._clf_head(zs),
                        labels[batch]).backward()
                opt.step()

            # --- validate ---
            if va_idx:
                self._pool.eval(); self._clf_head.eval()
                with torch.no_grad():
                    zv = torch.stack([self._pool(bags[j])[0] for j in va_idx])
                    vloss = loss_fn(self._clf_head(zv), labels[va_idx]).item()
                if vloss < best_val - 1e-4:
                    best_val, bad = vloss, 0
                    best_pool = {k: v.clone() for k, v in self._pool.state_dict().items()}
                    best_head = {k: v.clone() for k, v in self._clf_head.state_dict().items()}
                else:
                    bad += 1
                if bad >= self.patience:
                    break

        if best_pool:
            self._pool.load_state_dict(best_pool)
            self._clf_head.load_state_dict(best_head)
        self.fit_s = time.time() - t0
        return self

    def predict_subject(self, sub: Subject) -> tuple[float, int]:
        import torch
        feat = self._prep._features(sub)
        X = self._prep._project(tangent_space(feat["mats"], self._prep._ref))
        bag = torch.from_numpy(self._pca.transform(X.astype(np.float32)))
        self._pool.eval(); self._clf_head.eval()
        with torch.no_grad():
            z, attn = self._pool(bag)
            prob = float(torch.softmax(self._clf_head(z.unsqueeze(0)), dim=1)[0, 1])
        top3 = attn.topk(min(3, len(attn))).indices.tolist()
        print(f"      [wattn sid={sub.sid} y={sub.label}] "
              f"score={prob:.3f}  top-attn windows={top3}", flush=True)
        return prob, len(attn)


# ===========================================================================
# AIRM channel clustering ('cl') — reduce 125 channels -> K region tokens
# ===========================================================================
class ClusterStage:
    """Functional channel clustering on the correlation manifold (EEG-RCformer idea).

    Channels are grouped into K clusters by functional connectivity (learned from the
    TRAINING subjects only — leakage-free), average-pooled into K region "tokens", and
    each window's 125x125 covariance is reduced to a K x K region covariance via the
    pooling matrix:  C_K = P C Pᵀ  (since cov(P x) = P cov(x) Pᵀ).  The reduced matrix is
    normalised to a correlation, tangent-mapped, and classified.

    On MODMA cross-subject this clustering is the documented driver of EEG-RCformer's
    gain (+8.2% AUC over no clustering, Paper 8). It also removes the over-fitting cause:
    a K x K (K~=5..20) matrix cannot memorise 125-channel individual identity.
    """

    # per-subject 125x125 covariances are fold-independent -> cache once
    _COV_CACHE: dict[tuple, dict] = {}

    def __init__(self, n_clusters: int = 5, max_windows: int | None = 120,
                 C_reg: float = 1.0, seed: int = C.SEED):
        self.n_clusters = n_clusters
        self.max_windows = max_windows
        self.C_reg = C_reg
        self.seed = seed
        self._estimator = Covariances(estimator=C.COV_ESTIMATOR)

    def _covs(self, sub: Subject) -> dict:
        key = (C.DATASET, sub.sid, self.max_windows)
        cached = self._COV_CACHE.get(key)
        if cached is not None:
            return cached
        wins = epoch(sub.data, max_windows=self.max_windows)
        covs = self._estimator.transform(wins).astype(np.float64)      # (n,125,125)
        mean_corr = _cov_to_corr(covs).mean(axis=0)                    # (125,125) affinity
        feat = {"covs": covs, "mean_corr": mean_corr, "label": sub.label}
        self._COV_CACHE[key] = feat
        return feat

    def _pooling_matrix(self, train_feats: list[dict]) -> np.ndarray:
        # average connectivity across training subjects -> cluster channels by POSITIVE
        # correlation (1 - corr, not 1 - |corr|): grouping anti-correlated channels and
        # then averaging their raw signals would cancel the region token to ~0.
        R = np.mean([f["mean_corr"] for f in train_feats], axis=0)
        dist = 1.0 - R                              # in [0, 2]; positive corr -> small dist
        np.fill_diagonal(dist, 0.0)
        labels = AgglomerativeClustering(
            n_clusters=self.n_clusters, metric="precomputed", linkage="average"
        ).fit_predict(dist)
        n_ch = R.shape[0]
        P = np.zeros((self.n_clusters, n_ch))
        for k in range(self.n_clusters):
            idx = np.where(labels == k)[0]
            P[k, idx] = 1.0 / len(idx)
        return P

    def _reduce(self, covs: np.ndarray) -> np.ndarray:
        red = self._P @ covs @ self._P.T            # (n,K,K) region covariance
        return _cov_to_corr(red)                    # K x K correlation

    def fit(self, train_subjects: list[Subject]) -> "ClusterStage":
        t0 = time.time()
        feats = [self._covs(s) for s in train_subjects]
        self._P = self._pooling_matrix(feats)
        reduced = [self._reduce(f["covs"]) for f in feats]
        self._ref = mean_riemann(np.stack([mean_riemann(r) for r in reduced]))
        X = np.concatenate([tangent_space(r, self._ref) for r in reduced], axis=0)
        y = np.concatenate([np.full(len(r), f["label"]) for r, f in zip(reduced, feats)])
        self._clf = LogisticRegression(
            C=self.C_reg, class_weight="balanced", max_iter=2000,
            solver="lbfgs", random_state=self.seed,
        ).fit(X, y)
        self.fit_s = time.time() - t0
        return self

    def predict_subject(self, sub: Subject) -> tuple[float, int]:
        r = self._reduce(self._covs(sub)["covs"])
        probs = self._clf.predict_proba(tangent_space(r, self._ref))[:, 1]
        return float(probs.mean()), len(probs)


# ===========================================================================
# S0 — EEGNet Euclidean floor (torch)
# ===========================================================================
def _build_eegnet(n_ch: int, win: int, dropout: float):
    """Canonical EEGNet-8,2 (Lawhern et al. 2018), matching the TSMNet `EEGNetv4`
    reference in ``example code/TSMNet/spdnets/models/eegnet.py``:

    - temporal kernel = srate//2 (=62 at 125 Hz), separable kernel = srate//8 (=15);
    - BatchNorm momentum=0.01, eps=1e-3;
    - max-norm weight constraints (1.0 on the depthwise spatial conv, 0.25 on the
      classifier) applied after each optimiser step via ``clip_constraints()``.
    """
    import torch
    import torch.nn as nn

    F1, D, F2 = 8, 2, 16
    k1 = C.FS // 2               # temporal kernel ~0.5 s
    k2 = max(1, (C.FS // 2) // 4)  # separable temporal kernel
    bn_kw = dict(momentum=0.01, eps=1e-3)

    class EEGNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(1, F1, (1, k1), padding="same", bias=False),
                nn.BatchNorm2d(F1, **bn_kw),
            )
            self.spatial = nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False)
            self.depthwise = nn.Sequential(
                nn.BatchNorm2d(F1 * D, **bn_kw), nn.ELU(),
                nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
            )
            self.separable = nn.Sequential(
                nn.Conv2d(F1 * D, F1 * D, (1, k2), padding="same",
                          groups=F1 * D, bias=False),
                nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
                nn.BatchNorm2d(F2, **bn_kw), nn.ELU(),
                nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
            )
            feat = (win // 4 // 8) * F2
            self.classify = nn.Linear(feat, 2)

        def forward(self, x):                 # x: (B, 1, n_ch, win)
            x = self.block1(x)
            x = self.spatial(x)
            x = self.depthwise(x)
            x = self.separable(x)
            x = x.flatten(1)
            return self.classify(x)

        @torch.no_grad()
        def clip_constraints(self):
            for layer, mx in ((self.spatial, 1.0), (self.classify, 0.25)):
                w = layer.weight
                norm = w.flatten(1).norm(2, dim=1, keepdim=True)
                scale = (norm.clamp(max=mx) / (norm + 1e-8))
                w.mul_(scale.view(-1, *([1] * (w.dim() - 1))))

    return EEGNet()


class EEGNetStage:
    """Compact CNN trained source-only; per-window probabilities averaged per subject."""

    def __init__(self, epochs: int = 30, batch_size: int = 128, lr: float = 1e-3,
                 dropout: float = 0.25, max_windows: int | None = None,
                 device: str | None = None, seed: int = C.SEED, verbose: bool = False):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.max_windows = max_windows
        self.seed = seed
        self.verbose = verbose
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _windows(self, sub: Subject):
        wins = epoch(sub.data, max_windows=self.max_windows)
        # per-recording, per-channel standardisation (leakage-free; uses own stats)
        mu = wins.mean(axis=(0, 2), keepdims=True)
        sd = wins.std(axis=(0, 2), keepdims=True) + 1e-6
        return ((wins - mu) / sd).astype(np.float32)

    def fit(self, train_subjects: list[Subject]) -> "EEGNetStage":
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        t0 = time.time()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        Xs, ys = [], []
        for s in train_subjects:
            w = self._windows(s)
            Xs.append(w)
            ys.append(np.full(len(w), s.label))
        X = np.concatenate(Xs)[:, None]                # (N, 1, ch, win)
        y = np.concatenate(ys)
        n_ch, win = X.shape[2], X.shape[3]

        # class weights for imbalance
        counts = np.bincount(y, minlength=2).astype(np.float64)
        w_cls = counts.sum() / (2.0 * np.clip(counts, 1, None))
        weight = torch.tensor(w_cls, dtype=torch.float32, device=self.device)

        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long())
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.net = _build_eegnet(n_ch, win, self.dropout).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

        self.net.train()
        for ep in range(self.epochs):
            tot = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                opt.step()
                self.net.clip_constraints()       # EEGNet max-norm weight constraint
                tot += loss.item() * len(xb)
            if self.verbose and (ep + 1) % 10 == 0:
                print(f"      epoch {ep+1:>3}/{self.epochs}  loss={tot/len(ds):.4f}")
        self.fit_s = time.time() - t0
        return self

    @property
    def fit_s_(self):
        return getattr(self, "fit_s", 0.0)

    def predict_subject(self, sub: Subject) -> tuple[float, int]:
        import torch
        self.net.eval()
        w = self._windows(sub)[:, None]
        probs = []
        with torch.no_grad():
            for i in range(0, len(w), 512):
                xb = torch.from_numpy(w[i:i + 512]).to(self.device)
                p = torch.softmax(self.net(xb), dim=1)[:, 1]
                probs.append(p.cpu().numpy())
        probs = np.concatenate(probs)
        return float(probs.mean()), len(probs)


def build_model(stage: str, **kw):
    stage = stage.lower()
    if stage == "s0":
        return EEGNetStage(**kw)
    if stage in ("s1", "s2", "s3", "n1", "cc", "nr", "nc", "cr", "bd", "tnr", "ncv", "nrde"):
        return make_shallow(stage, **kw)
    if stage == "mb":
        return MultibandStage(**kw)
    if stage == "de":
        return DEStage(**kw)
    if stage == "spec":
        return SpectralStage(**kw)
    if stage == "defu":
        return DEFusionStage(**kw)
    if stage == "spfu":
        return SpectralFusionStage(**kw)
    if stage == "mlp":
        return MLPStage(**kw)
    if stage == "wattn":
        return WindowAttnStage(**kw)
    if stage == "cl":
        return ClusterStage(**kw)
    if stage in ("s4", "s5"):
        from .spdnet import build_spdnet
        return build_spdnet(stage, **kw)
    if stage == "ds":
        from .dualspd import build_dualspd
        return build_dualspd(stage, **kw)
    if stage == "mbds":
        from .dualspd import build_mbds
        return build_mbds(**kw)
    if stage == "spddann":
        from .spddann import SPDDANNStage
        return SPDDANNStage(**kw)
    raise ValueError(f"unknown stage {stage!r} (use s0|s1|s2|s3|s4|s5|ds|mbds|spddann)")
