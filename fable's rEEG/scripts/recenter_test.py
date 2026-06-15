"""Diagnostic: WHY does per-subject Riemannian recentring (S3 / rebiasing / S-RCT) fail
on MODMA depression LODO?

Thesis to test empirically on real MODMA subjects (= domains):
  Per-subject recentring whitens every subject by its OWN Frechet mean, sending each
  subject's mean correlation matrix to the identity I. That deletes the BETWEEN-subject
  differences in the mean. In motor-imagery the label varies WITHIN a subject, so the
  class signal lives in within-subject variation and survives recentring. In MODMA the
  label is CONSTANT per subject (resting state, one label per recording), so the class
  signal lives ENTIRELY in the between-subject mean -> recentring deletes exactly the
  signal and the classifier collapses to AUC 0.5.

We verify this with four measurements on selected MODMA subjects/domains:
  [A] Recentring really sends each domain mean to I (per-subject check).
  [B] Where the depression signal lives: subject-level mean signal vs within-subject
      residual signal (LOSO AUC of each component).
  [C] The S2 (no recenter) vs S3 (recenter) classifier head, mini-LOSO, same subjects.
  [D] Synthetic control: an injected WITHIN-subject label -> recentring HELPS.
      (Proves the operation is correct, just mismatched to a per-subject-label task.)
"""
from __future__ import annotations

import warnings

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.utils.base import invsqrtm, logm
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import os as _os, sys as _sys  # bootstrap: repo root on path (script moved to scripts/)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from triad import config as C
from triad.data import load_subjects, epoch
from triad.models import _cov_to_corr, _recenter

warnings.filterwarnings("ignore")
rng = np.random.default_rng(C.SEED)

MAX_WIN = 60          # windows per subject (speed)
N_PER_CLASS = 10      # selective subset size per class for the classifier tests


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
def corr_mats(sub, max_win=MAX_WIN):
    wins = epoch(sub.data, max_windows=max_win)
    covs = Covariances(estimator=C.COV_ESTIMATOR).transform(wins).astype(np.float64)
    return _cov_to_corr(covs)                       # (n_win, ch, ch) SPD correlation


def frechet(mats):
    return mean_riemann(mats)


def spd_norm_to_I(M):
    """Riemannian distance of an SPD matrix to the identity = ||logm(M)||_F."""
    return float(np.linalg.norm(logm(M)))


# ---------------------------------------------------------------------------
def main():
    print("Loading MODMA subjects (cached) ...")
    subs = load_subjects()
    mdd = [s for s in subs if s.label == 1]
    hc = [s for s in subs if s.label == 0]
    print(f"  {len(subs)} subjects total: {len(mdd)} MDD / {len(hc)} HC, "
          f"{subs[0].data.shape[0]} channels, {MAX_WIN} windows/subj\n")

    # Selective, balanced subset of subjects = "domains" for the detailed tests
    sel = sorted(mdd[:N_PER_CLASS] + hc[:N_PER_CLASS], key=lambda s: s.sid)
    print("Selected domains (subjects) for detailed recenter test:")
    print("  MDD:", [s.sid for s in mdd[:N_PER_CLASS]])
    print("  HC :", [s.sid for s in hc[:N_PER_CLASS]], "\n")

    # Precompute per-subject correlation matrices + means for the selected subset
    feats = {}
    for s in sel:
        m = corr_mats(s)
        feats[s.sid] = dict(label=s.label, mats=m, mean=frechet(m))

    # =====================================================================
    # [A] Does recentring really push every DOMAIN mean to the identity?
    # =====================================================================
    print("=" * 72)
    print("[A] Per-domain recenter check:  dist(mean, I)  before -> after")
    print("=" * 72)
    print(f"  {'sid':>8} {'cls':>4} {'dist_to_I (orig)':>18} {'dist_to_I (recentred)':>22}")
    before_I, after_I = [], []
    for s in sel:
        f = feats[s.sid]
        M = f["mean"]
        Mr = frechet(_recenter(f["mats"], M))        # mean after whitening by own mean
        d0, d1 = spd_norm_to_I(M), spd_norm_to_I(Mr)
        before_I.append(d0); after_I.append(d1)
        print(f"  {s.sid:>8} {C.LABEL_NAMES[s.label]:>4} {d0:>18.4f} {d1:>22.2e}")
    print(f"\n  mean dist(mean, I):  orig = {np.mean(before_I):.4f}   "
          f"recentred = {np.mean(after_I):.2e}")
    print("  => recentring collapses every domain's mean onto I (signal in the mean is erased).\n")

    # =====================================================================
    # [B] WHERE does the depression signal live?
    #     subject-level mean  vs  within-subject residual
    # =====================================================================
    print("=" * 72)
    print("[B] Decompose the signal: between-subject mean  vs  within-subject residual")
    print("=" * 72)
    # population reference = Frechet mean of all selected subject means
    G = mean_riemann(np.stack([feats[s.sid]["mean"] for s in sel]))

    # subject-level vector = subject mean projected to tangent at G  (one per domain)
    Vmean = {s.sid: tangent_space(feats[s.sid]["mean"][None], G)[0] for s in sel}
    # window-level vectors at G, and the within-subject residual (= what recentring keeps)
    Vwin, Vres, Ywin, Sidwin = [], [], [], []
    for s in sel:
        T = tangent_space(feats[s.sid]["mats"], G)   # (n_win, d) window tangent vectors
        Vwin.append(T)
        Vres.append(T - Vmean[s.sid])                # remove subject mean -> residual
        Ywin.append(np.full(len(T), s.label))
        Sidwin.append(np.full(len(T), s.sid))
    Vwin = np.vstack(Vwin); Vres = np.vstack(Vres)
    Ywin = np.concatenate(Ywin); Sidwin = np.concatenate(Sidwin)

    def loso_auc_subjectlevel(vecs: dict):
        """LOSO AUC using one vector per subject (the between-subject mean signal)."""
        sids = [s.sid for s in sel]
        y = np.array([feats[i]["label"] for i in sids])
        X = np.array([vecs[i] for i in sids])
        scores = np.zeros(len(sids))
        for k in range(len(sids)):
            tr = np.arange(len(sids)) != k
            clf = LogisticRegression(C=1.0, class_weight="balanced",
                                     max_iter=2000).fit(X[tr], y[tr])
            scores[k] = clf.decision_function(X[k:k + 1])[0]
        return roc_auc_score(y, scores)

    def loso_auc_windowlevel(X, y, sid):
        """LOSO AUC at the window level, aggregating window scores per subject."""
        usids = np.unique(sid)
        ys = np.array([y[sid == u][0] for u in usids])
        agg = np.zeros(len(usids))
        for k, u in enumerate(usids):
            tr = sid != u
            clf = LogisticRegression(C=1.0, class_weight="balanced",
                                     max_iter=2000).fit(X[tr], y[tr])
            agg[k] = clf.decision_function(X[sid == u]).mean()
        return roc_auc_score(ys, agg)

    auc_mean = loso_auc_subjectlevel(Vmean)
    auc_win = loso_auc_windowlevel(Vwin, Ywin, Sidwin)
    auc_res = loso_auc_windowlevel(Vres, Ywin, Sidwin)
    # mean Riemannian distance between/within class (subject means)
    d_between, d_within = [], []
    for i, a in enumerate(sel):
        for b in sel[i + 1:]:
            d = distance_riemann(feats[a.sid]["mean"], feats[b.sid]["mean"])
            (d_between if a.label != b.label else d_within).append(d)
    print(f"  subject-mean Riemannian distance:  between-class = {np.mean(d_between):.4f}"
          f"   within-class = {np.mean(d_within):.4f}")
    print(f"\n  LOSO AUC from the SUBJECT MEAN only (between-subject signal) : {auc_mean:.3f}")
    print(f"  LOSO AUC from FULL window tangent vectors (mean + residual)  : {auc_win:.3f}")
    print(f"  LOSO AUC from WITHIN-subject RESIDUAL only (recentring keeps): {auc_res:.3f}")
    print("  => the depression signal lives in the subject MEAN; the within-subject")
    print("     residual that survives recentring is ~chance.\n")

    # =====================================================================
    # [C] The real S2 vs S3 head on the SAME selected domains (mini-LOSO)
    # =====================================================================
    print("=" * 72)
    print("[C] Actual tangent-space LR head, mini-LOSO on selected domains")
    print("=" * 72)
    from triad.models import ShallowStage

    def mini_loso2(recenter: bool):
        # NB: do NOT clear the feature cache per fold — features are fold-independent
        # (each subject is centred by its own mean), and the cache key already includes
        # `recenter`, so S2 and S3 never collide. Clearing per fold would recompute every
        # training subject's covariance on every fold.
        y, score, wnorms = [], [], []
        for held in sel:
            train = [s for s in sel if s.sid != held.sid]
            m = ShallowStage(rep="corr", recenter=recenter, max_windows=MAX_WIN)
            m.fit(train)
            p, _ = m.predict_subject(held)
            y.append(held.label); score.append(p)
            wnorms.append(float(np.linalg.norm(m._clf.coef_)))
        return roc_auc_score(y, score), np.mean(wnorms)

    auc_s2, w_s2 = mini_loso2(recenter=False)
    print("  [S2 done]", flush=True)
    auc_s3, w_s3 = mini_loso2(recenter=True)
    print(f"  S2  corr, NO recenter : AUC = {auc_s2:.3f}   mean||LR weights|| = {w_s2:.3e}")
    print(f"  S3  corr, recenter    : AUC = {auc_s3:.3f}   mean||LR weights|| = {w_s3:.3e}")
    print("  => after recentring the LR has nothing to fit (weights ~0) -> AUC ~0.5.\n")

    # =====================================================================
    # [D] Synthetic control: inject a WITHIN-subject label. Now recentring HELPS.
    # =====================================================================
    print("=" * 72)
    print("[D] Control: synthetic WITHIN-subject label (same shift in every domain)")
    print("=" * 72)
    # Build a fixed SPD 'class-B' transform applied to half of EACH subject's windows.
    ch = subs[0].data.shape[0]
    A = rng.standard_normal((ch, ch)) * 0.06
    Wsq = np.eye(ch) + (A + A.T) / 2.0
    # ensure SPD-ish congruence factor
    Wsq = Wsq @ Wsq.T
    Wsq = Wsq / np.trace(Wsq) * ch

    def synth_features(recenter: bool):
        X, y, sid = [], [], []
        for s in sel:
            m = feats[s.sid]["mats"].copy()
            n = len(m)
            half = n // 2
            lab = np.zeros(n, dtype=int)
            lab[half:] = 1
            # apply the SAME congruence to the 'class-1' windows of every subject
            m[half:] = Wsq @ m[half:] @ Wsq.T
            if recenter:
                m = _recenter(m, mean_riemann(m))   # whiten by subject's own mean
            ref = mean_riemann(m)
            X.append(tangent_space(m, ref if recenter else G_syn))
            y.append(lab); sid.append(np.full(n, s.sid))
        return np.vstack(X), np.concatenate(y), np.concatenate(sid)

    # shared reference for the no-recenter synthetic case
    allm = []
    for s in sel:
        m = feats[s.sid]["mats"].copy(); half = len(m) // 2
        m[half:] = Wsq @ m[half:] @ Wsq.T
        allm.append(mean_riemann(m))
    G_syn = mean_riemann(np.stack(allm))

    def synth_auc(recenter: bool):
        X, y, sid = synth_features(recenter)
        # within-subject CV is irrelevant; label is shared, so pooled CV across subjects:
        # train on all-but-one subject, test held-out subject's windows
        usids = np.unique(sid)
        aucs = []
        for u in usids:
            tr = sid != u
            clf = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], y[tr])
            sc = clf.decision_function(X[sid == u])
            aucs.append(roc_auc_score(y[sid == u], sc))
        return np.mean(aucs)

    auc_syn_no = synth_auc(False)
    auc_syn_rc = synth_auc(True)
    print(f"  synthetic within-subject label, NO recenter : mean AUC = {auc_syn_no:.3f}")
    print(f"  synthetic within-subject label, recenter    : mean AUC = {auc_syn_rc:.3f}")
    print("  => when the label is WITHIN-subject, recentring removes the nuisance mean")
    print("     and IMPROVES (or preserves) separability. The operation is correct;")
    print("     it is the per-subject-constant depression label that makes it fail.\n")

    # ---- one-line verdict ----
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  depression signal in subject mean (AUC {auc_mean:.2f}) vs residual "
          f"(AUC {auc_res:.2f}).")
    print(f"  recentring zeros the mean (dist_to_I {np.mean(after_I):.0e}) -> S3 AUC "
          f"{auc_s3:.2f} ~ chance,")
    print(f"  while S2 (no recenter) keeps it -> AUC {auc_s2:.2f}.")
    print(f"  same recentring on a WITHIN-subject label HELPS (AUC {auc_syn_no:.2f} -> "
          f"{auc_syn_rc:.2f}).")


if __name__ == "__main__":
    main()
