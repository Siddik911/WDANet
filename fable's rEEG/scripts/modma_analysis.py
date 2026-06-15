#!/usr/bin/env python3
"""
MODMA Dataset Analysis — Why is LODO generalisation so hard?
=============================================================
Eight analyses that systematically dissect the failure modes.

Outputs -> modma_analysis/
  A2  Inter-subject Riemannian distance matrix (53×53 heatmap)
  B   Variance decomposition: identity vs class signal vs noise
  C   PCA / scree + LOO-AUC vs #PCs (how much information per dimension)
  D   Band-specific class separation delta/theta/alpha/beta
  E   Within-subject EEG stability vs S2 classifier confidence
  F   Per-fold S2 score analysis (who is hardest to classify?)
  G   Correlation matrix gallery (visual inter-subject variability)
  H   Learning curve: AUC vs n_training_subjects (does more data help?)

Run:
  python -u modma_analysis.py 2>&1 | tee modma_analysis/run.log
  (~25-40 min, no GPU needed)
"""
import sys, json, warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import signal as ss
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
from pyriemann.tangentspace import TangentSpace

warnings.filterwarnings('ignore')
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import os as _os, sys as _sys  # bootstrap: repo root on path (script moved to scripts/)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from triad import data as D, config as C

OUT    = Path('modma_analysis')
OUT.mkdir(exist_ok=True)
MAXW   = 60             # windows per subject (broadband)
MAXW_B = 40             # windows for band analysis (faster)
FS     = C.FS           # 125 Hz after decimation
BANDS  = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
C_MDD  = '#E53935'
C_HC   = '#1E88E5'
summary: dict = {}


# ── helpers ───────────────────────────────────────────────────────────────────
def _bandpass(x, lo, hi, fs=FS, order=4):
    """Zero-phase Butterworth bandpass. x: (..., T)."""
    nyq = fs / 2.0
    b, a = ss.butter(order, [lo / nyq, hi / nyq], btype='band')
    return ss.filtfilt(b, a, x, axis=-1).astype(np.float64)


def _corr_stack(wins):
    """(W, ch, T) → (W, ch, ch) LWF correlation matrices."""
    covs = Covariances(estimator='lwf').fit_transform(wins.astype(np.float64))
    d    = np.sqrt(np.einsum('...ii->...i', covs))     # (W, ch) diagonal sds
    d    = np.clip(d, 1e-10, None)
    return covs / np.einsum('...i,...j->...ij', d, d)  # normalise to correlation


def _riem_mean(corrs):
    return mean_riemann(corrs.astype(np.float64), maxiter=60)


def _save(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    → modma_analysis/{name}")


def _loo_auc(X, y):
    """Fast subject-level LOO AUC on a small (N, d) feature matrix."""
    N     = len(y)
    lr    = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
    sc    = np.zeros(N)
    for i in range(N):
        tr = list(range(N)); tr.pop(i)
        lr.fit(X[tr], y[tr])
        sc[i] = lr.predict_proba(X[[i]])[0, 1]
    return roc_auc_score(y, sc), sc


# ══════════════════════════════════════════════════════════════════════════════
print("=" * 66)
print("  MODMA Dataset Analysis")
print("=" * 66)

# ── Load ──────────────────────────────────────────────────────────────────────
print("\nLoading subjects from cache ...")
subs   = D.load_subjects()
print(D.summary(subs))
N      = len(subs)
labels = np.array([s.label for s in subs])
sids   = [s.sid for s in subs]
mdd_i  = np.where(labels == 1)[0]
hc_i   = np.where(labels == 0)[0]
n_mdd, n_hc = len(mdd_i), len(hc_i)

print(f"\nEpoching {N} subjects (max {MAXW} windows each) ...")
wins_list = []
for i, s in enumerate(subs):
    w = D.epoch(s.data, max_windows=MAXW)
    wins_list.append(w)
    if (i + 1) % 10 == 0 or i == N - 1:
        print(f"  {i+1}/{N}", flush=True)
n_wins = np.array([len(w) for w in wins_list])
print(f"  median {np.median(n_wins):.0f}  "
      f"min {n_wins.min()}  max {n_wins.max()}  total {n_wins.sum()}")
summary.update({'N': N, 'n_mdd': int(n_mdd), 'n_hc': int(n_hc),
                'median_wins': int(np.median(n_wins)),
                'total_wins': int(n_wins.sum())})


# ── A. Broadband subject means ────────────────────────────────────────────────
print(f"\n[A] LWF correlation matrices + Riemannian means (broadband) ...")
subj_corrs = []
subj_means = []
for i, (s, w) in enumerate(zip(subs, wins_list)):
    corrs = _corr_stack(w)
    mu    = _riem_mean(corrs)
    subj_corrs.append(corrs)
    subj_means.append(mu)
    if (i + 1) % 10 == 0 or i == N - 1:
        print(f"  {i+1}/{N}", flush=True)

means_arr = np.array(subj_means)   # (N, ch, ch)

# shared tangent space (reference = grand Riemannian mean of subject means)
print("  Fitting shared tangent space ...")
ts_global = TangentSpace(metric='riemann').fit(means_arr)
subj_tang = ts_global.transform(means_arr)   # (N, d)  d = ch*(ch+1)/2
d_tang    = subj_tang.shape[1]
print(f"  Tangent feature dim: {d_tang}")


# ── A2. Inter-subject distance matrix ─────────────────────────────────────────
print("\n[A2] Pairwise AIRM distance matrix ...")
Dmat = np.zeros((N, N))
for i in range(N):
    for j in range(i + 1, N):
        v = distance_riemann(means_arr[i], means_arr[j])
        Dmat[i, j] = Dmat[j, i] = v
    if (i + 1) % 10 == 0 or i == N - 1:
        print(f"  row {i+1}/{N}", flush=True)

sort_idx = np.concatenate([mdd_i, hc_i])
Ds = Dmat[np.ix_(sort_idx, sort_idx)]

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(Ds, cmap='plasma', aspect='auto')
plt.colorbar(im, ax=ax, label='AIRM distance')
ax.axvline(n_mdd - .5, color='white', lw=1.5, ls='--')
ax.axhline(n_mdd - .5, color='white', lw=1.5, ls='--')
ax.set_title('Inter-subject Riemannian distance  (MDD | HC)', fontsize=12)
ax.set_xlabel('Subject index  (MDD → HC)')
ax.set_ylabel('Subject index  (MDD → HC)')
ax.text(n_mdd / 2, -2, 'MDD', ha='center', color=C_MDD, fontweight='bold')
ax.text(n_mdd + n_hc / 2, -2, 'HC', ha='center', color=C_HC, fontweight='bold')
_save('A2_distance_matrix.png')

wm = Dmat[np.ix_(mdd_i, mdd_i)].copy(); np.fill_diagonal(wm, np.nan)
wh = Dmat[np.ix_(hc_i,  hc_i)].copy();  np.fill_diagonal(wh, np.nan)
xc = Dmat[np.ix_(mdd_i, hc_i)]
d_wm, d_wh, d_xc = np.nanmean(wm), np.nanmean(wh), xc.mean()
sep = d_xc / ((d_wm + d_wh) / 2)
print(f"  within-MDD : {d_wm:.4f}")
print(f"  within-HC  : {d_wh:.4f}")
print(f"  MDD ↔ HC   : {d_xc:.4f}")
print(f"  sep-ratio  : {sep:.4f}  (<1.05 = class buried in within-class spread)")
summary.update({'dist_within_mdd': round(d_wm, 4), 'dist_within_hc': round(d_wh, 4),
                'dist_cross': round(d_xc, 4), 'class_sep_ratio': round(sep, 4)})


# ── B. Variance decomposition ─────────────────────────────────────────────────
print("\n[B] Variance decomposition: identity vs class vs noise ...")
grand_t = subj_tang.mean(0)
mdd_t   = subj_tang[mdd_i].mean(0)
hc_t    = subj_tang[hc_i].mean(0)

# Between-subject: how much spread do subjects have around the grand mean?
var_btwn_subj  = float(np.sum((subj_tang - grand_t) ** 2) / (N - 1))

# Between-class: L2 norm of (MDD_mean - HC_mean) relative to grand mean
# (signed direction of the depression effect)
class_diff     = mdd_t - hc_t
class_signal   = float(np.linalg.norm(class_diff))
subj_spreads   = np.linalg.norm(subj_tang - grand_t, axis=1)  # (N,) per-subject dist
identity_mean  = float(subj_spreads.mean())

ratio_sig_id   = class_signal / identity_mean  # key diagnostic

# Within-subject: average variance of windows around each subject's mean
print("  Per-window tangent projections ...")
win_tangs = []
for i, corrs in enumerate(subj_corrs):
    win_tangs.append(ts_global.transform(corrs))   # (W, d)
    if (i + 1) % 10 == 0 or i == N - 1:
        print(f"    {i+1}/{N}", flush=True)
var_within = float(np.mean([np.var(t, ddof=1, axis=0).mean() for t in win_tangs]))

print(f"  class signal (||MDD_mean - HC_mean||) : {class_signal:.4f}")
print(f"  mean identity spread (||subj - grand||): {identity_mean:.4f}")
print(f"  signal/identity ratio                  : {ratio_sig_id:.4f}  "
      f"(<0.5 = small needle in large haystack)")
print(f"  within-subject window variance          : {var_within:.4f}")
summary.update({'class_signal_norm': round(class_signal, 4),
                'identity_spread': round(identity_mean, 4),
                'signal_identity_ratio': round(ratio_sig_id, 4),
                'var_within_subj': round(var_within, 4)})

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.bar(['Identity\n(between-subj)', 'Class signal\n(MDD − HC)', 'Noise\n(within-subj)'],
       [identity_mean, class_signal, np.sqrt(var_within)],
       color=['#78909C', '#EF5350', '#42A5F5'], width=0.55,
       edgecolor='white', linewidth=1.5)
ax.set_ylabel('L2 norm in tangent space (log scale)')
ax.set_yscale('log')
ax.set_title('Signal size on the manifold\n(why class signal is drowned out)', fontsize=11)
ax.text(1, class_signal * 1.1,
        f'ratio = {ratio_sig_id:.3f}\n(ideal ≫ 1)',
        ha='center', va='bottom', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray'))

ax = axes[1]
ax.scatter(subj_spreads[hc_i],  np.zeros(n_hc)  + 0, c=C_HC,  s=70,
           alpha=0.8, edgecolors='white', lw=0.5, label='HC', zorder=3)
ax.scatter(subj_spreads[mdd_i], np.zeros(n_mdd) + 1, c=C_MDD, s=70,
           alpha=0.8, edgecolors='white', lw=0.5, label='MDD', zorder=3)
ax.axvline(class_signal, color='black', ls='--', lw=1.5,
           label=f'class-signal size ({class_signal:.1f})')
ax.set_xlabel('Distance from grand mean (identity spread per subject)')
ax.set_ylabel('Group')
ax.set_yticks([0, 1]); ax.set_yticklabels(['HC', 'MDD'])
ax.set_title('Each dot = one subject\'s distance from grand mean\nvs the MDD−HC signal size (dashed)', fontsize=10)
ax.legend(fontsize=9)
_save('B_variance_decomposition.png')


# ── C. PCA visualization ──────────────────────────────────────────────────────
print("\n[C] PCA visualization ...")
n_pc  = min(30, N - 1)
pca   = PCA(n_components=n_pc).fit(subj_tang)
Xpca  = pca.transform(subj_tang)
vexp  = pca.explained_variance_ratio_

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for lv, col, nm in [(1, C_MDD, 'MDD'), (0, C_HC, 'HC')]:
    idx = np.where(labels == lv)[0]
    ax.scatter(Xpca[idx, 0], Xpca[idx, 1], c=col, label=nm, s=85,
               alpha=0.85, edgecolors='white', linewidths=0.5)
ax.set_xlabel(f'PC1 ({vexp[0]*100:.1f}%)', fontsize=11)
ax.set_ylabel(f'PC2 ({vexp[1]*100:.1f}%)', fontsize=11)
ax.set_title('Subject mean correlations in PC1-PC2 space', fontsize=11)
ax.legend(fontsize=10)

ax = axes[1]
k_sc = min(15, n_pc)
ax.bar(range(1, k_sc + 1), vexp[:k_sc] * 100,
       color='steelblue', alpha=0.8, edgecolor='white', linewidth=0.5)
ax2 = ax.twinx()
ax2.plot(range(1, k_sc + 1), np.cumsum(vexp[:k_sc]) * 100,
         'o-', color='tomato', lw=2, ms=5)
ax2.axhline(80, ls='--', color='tomato', lw=1, alpha=0.4)
ax2.set_ylabel('Cumulative variance (%)', color='tomato')
ax.set_xlabel('Principal component')
ax.set_ylabel('Explained variance (%)')
ax.set_title('Scree plot', fontsize=11)
_save('C_pca.png')

# LOO-AUC vs number of PCs kept
print("  LOO-AUC vs #PCs ...")
pc_ks = [k for k in [1, 2, 3, 5, 8, 10, 15, 20, 30] if k <= n_pc]
pca_aucs = []
for k in pc_ks:
    auc, _ = _loo_auc(Xpca[:, :k], labels)
    pca_aucs.append(auc)
    print(f"    top-{k:2d} PCs: LOO-AUC = {auc:.3f}")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(pc_ks, pca_aucs, 'o-', color='steelblue', lw=2, ms=7)
ax.axhline(0.684, color='green',   ls='--', lw=1.5, label='S2 broadband LOSO (0.684)')
ax.axhline(0.715, color='darkred', ls='--', lw=1.5, label='SOTA EEG-RCformer (0.715)')
ax.axhline(0.5,   color='gray',    ls=':',  lw=1)
ax.set_xlabel('Number of PCA components used', fontsize=11)
ax.set_ylabel('LOO subject-level AUC', fontsize=11)
ax.set_title('Information per dimension:\nLOO-AUC vs #PCs', fontsize=11)
ax.legend(fontsize=9); ax.set_xticks(pc_ks)
_save('C2_pca_auc.png')
summary['pca_aucs'] = {str(k): round(a, 3) for k, a in zip(pc_ks, pca_aucs)}


# ── D. Band-specific class separation ─────────────────────────────────────────
print(f"\n[D] Band-specific analysis ({MAXW_B} windows/subject) ...")
band_results = {}
band_tang    = {}   # store for composite figure

for bname, (lo, hi) in BANDS.items():
    print(f"  {bname} ({lo}-{hi} Hz) ...", flush=True)
    b_means = []
    failed  = False
    for s, w in zip(subs, wins_list):
        try:
            x_f  = _bandpass(s.data, lo, hi)
            w_f  = D.epoch(x_f, max_windows=MAXW_B)
            if len(w_f) < 5:
                raise RuntimeError(f"only {len(w_f)} windows after bandpass")
            corrs_b = _corr_stack(w_f)
            b_means.append(_riem_mean(corrs_b))
        except Exception as ex:
            print(f"    FAILED on {s.sid}: {ex}")
            failed = True; break
    if failed or len(b_means) != N:
        band_results[bname] = {'auc': 0.5, 'status': 'FAILED'}
        continue

    bm_arr = np.array(b_means)
    try:
        ts_b   = TangentSpace(metric='riemann').fit(bm_arr)
        tang_b = ts_b.transform(bm_arr)
    except Exception as ex:
        print(f"    Tangent space failed: {ex}")
        band_results[bname] = {'auc': 0.5, 'status': 'FAILED'}
        continue

    auc_b, sc_b = _loo_auc(tang_b, labels)

    # effect size: |mu_MDD - mu_HC| / pooled_std (averaged over dims)
    diff  = tang_b[mdd_i].mean(0) - tang_b[hc_i].mean(0)
    s_p   = np.sqrt((tang_b[mdd_i].var(0) + tang_b[hc_i].var(0)) / 2 + 1e-10)
    eff   = float(np.mean(np.abs(diff) / s_p))

    print(f"    AUC = {auc_b:.3f}  effect-size d' = {eff:.4f}")
    band_results[bname] = {'auc': round(auc_b, 3), 'dprime': round(eff, 4), 'status': 'ok'}
    band_tang[bname]    = tang_b

summary['band_results'] = band_results

bnames  = list(BANDS.keys())
aucs_b  = [band_results.get(b, {}).get('auc', 0.5) for b in bnames]
dps_b   = [band_results.get(b, {}).get('dprime', 0.0) for b in bnames]
bcolors = ['#5C6BC0', '#26A69A', '#FFA726', '#EF5350']

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
bars = ax.bar(bnames, aucs_b, color=bcolors, alpha=0.85, edgecolor='white', lw=1.5, width=0.55)
ax.axhline(0.684, color='green',   ls='--', lw=1.5, label='S2 broadband (0.684)')
ax.axhline(0.715, color='darkred', ls='--', lw=1.5, label='SOTA (0.715)')
ax.axhline(0.5,   color='gray',    ls=':',  lw=1)
for b, v in zip(bars, aucs_b):
    ax.text(b.get_x() + b.get_width()/2, v + 0.005, f'{v:.3f}',
            ha='center', va='bottom', fontsize=10)
ax.set_ylim(0.35, 0.82)
ax.set_ylabel('LOO Subject-level AUC', fontsize=11)
ax.set_title('AUC per frequency band', fontsize=11)
ax.legend(fontsize=9)

ax = axes[1]
bars = ax.bar(bnames, dps_b, color=bcolors, alpha=0.85, edgecolor='white', lw=1.5, width=0.55)
for b, v in zip(bars, dps_b):
    ax.text(b.get_x() + b.get_width()/2, v + 0.0005, f'{v:.3f}',
            ha='center', va='bottom', fontsize=10)
ax.set_ylabel("Effect size d'", fontsize=11)
ax.set_title("Class separation per frequency band", fontsize=11)
_save('D_band_analysis.png')

# D2. Band PCA scatter (2×2 grid)
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, bname in zip(axes.flat, bnames):
    if bname not in band_tang:
        ax.set_title(f'{bname} — FAILED'); continue
    Xb2 = PCA(n_components=2).fit_transform(band_tang[bname])
    for lv, col, nm in [(1, C_MDD, 'MDD'), (0, C_HC, 'HC')]:
        idx = np.where(labels == lv)[0]
        ax.scatter(Xb2[idx, 0], Xb2[idx, 1], c=col, label=nm, s=60, alpha=0.8,
                   edgecolors='white', lw=0.4)
    auc_txt = band_results.get(bname, {}).get('auc', '?')
    ax.set_title(f'{bname}  ({lo}-{hi} Hz)  AUC={auc_txt}', fontsize=10)
    ax.legend(fontsize=8)
plt.suptitle('Band-specific PCA scatter (PC1 vs PC2)', fontsize=13, y=1.01)
_save('D2_band_pca.png')


# ── E. Within-subject stability ────────────────────────────────────────────────
print("\n[E] Within-subject window stability ...")
win_vars = np.array([np.var(t, ddof=1, axis=0).mean() for t in win_tangs])
t_stat, p_val = ttest_ind(win_vars[mdd_i], win_vars[hc_i])
print(f"  MDD mean var: {win_vars[mdd_i].mean():.4f}  "
      f"HC mean var: {win_vars[hc_i].mean():.4f}  "
      f"t={t_stat:.2f}  p={p_val:.3f}")
summary.update({'within_var_mdd': round(win_vars[mdd_i].mean(), 4),
                'within_var_hc':  round(win_vars[hc_i].mean(), 4),
                'within_var_pval': round(p_val, 3)})

# Load S2 per-fold scores for confidence correlation
s2_path = ROOT / 'results' / 's2_loso.json'
has_s2  = s2_path.exists()
s2_scores = None
if has_s2:
    pf_map    = {x['sid']: x['score'] for x in json.loads(s2_path.read_text())['per_fold']}
    s2_scores = np.array([pf_map.get(sid, np.nan) for sid in sids])
    has_s2    = not np.isnan(s2_scores).any()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
vp = ax.violinplot([win_vars[hc_i], win_vars[mdd_i]], positions=[0, 1],
                   showmedians=True, showextrema=True)
for pc, col in zip(vp['bodies'], [C_HC, C_MDD]):
    pc.set_facecolor(col); pc.set_alpha(0.65)
ax.set_xticks([0, 1]); ax.set_xticklabels(['HC', 'MDD'], fontsize=11)
ax.set_ylabel('Mean per-window variance in tangent space')
ax.set_title(f'Within-subject EEG variability\n(t={t_stat:.2f},  p={p_val:.3f})', fontsize=11)

ax = axes[1]
if has_s2:
    margin = np.abs(s2_scores - 0.5)
    ax.scatter(win_vars[hc_i],  margin[hc_i],  c=C_HC,  s=65, alpha=0.75,
               label='HC',  edgecolors='white', lw=0.5)
    ax.scatter(win_vars[mdd_i], margin[mdd_i], c=C_MDD, s=65, alpha=0.75,
               label='MDD', edgecolors='white', lw=0.5)
    r = np.corrcoef(win_vars, margin)[0, 1]
    ax.set_xlabel('Within-subject window variance (EEG noise)')
    ax.set_ylabel('|S2 score − 0.5|  (classifier confidence)')
    ax.set_title(f'Variability vs classifier confidence\n(r = {r:.3f})', fontsize=11)
    ax.legend()
    summary['var_vs_confidence_r'] = round(r, 3)
else:
    ax.text(0.5, 0.5, 'S2 results not found', ha='center', va='center',
            transform=ax.transAxes, color='gray')
_save('E_stability.png')


# ── F. Per-fold S2 score analysis ─────────────────────────────────────────────
if has_s2:
    print("\n[F] Per-fold S2 analysis ...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(s2_scores[hc_i],  bins=14, alpha=0.65, color=C_HC,
            label='HC',  density=True, edgecolor='white')
    ax.hist(s2_scores[mdd_i], bins=14, alpha=0.65, color=C_MDD,
            label='MDD', density=True, edgecolor='white')
    ax.axvline(0.5, color='black', ls='--', lw=1.5)
    ax.set_xlabel('S2 MDD probability score', fontsize=11)
    ax.set_ylabel('Density')
    ax.set_title('Score distributions  (ideal: HC left, MDD right)', fontsize=11)
    ax.legend()

    ax = axes[1]
    order = np.argsort(s2_scores)
    bar_colors = []
    for idx in order:
        correct = (labels[idx] == 1 and s2_scores[idx] >= 0.5) or \
                  (labels[idx] == 0 and s2_scores[idx] < 0.5)
        bar_colors.append((C_MDD if labels[idx] == 1 else C_HC) if correct else '#BDBDBD')
    ax.bar(range(N), s2_scores[order], color=bar_colors, width=0.85)
    ax.axhline(0.5, color='black', ls='--', lw=1.5)
    ax.set_xlabel('Subject (sorted by score)', fontsize=11)
    ax.set_ylabel('S2 MDD probability score', fontsize=11)
    ax.set_title('Per-subject S2 scores\n(gray = misclassified @ threshold 0.5)', fontsize=11)
    handles = [mpatches.Patch(color=C_MDD,    label='MDD correct'),
               mpatches.Patch(color=C_HC,     label='HC correct'),
               mpatches.Patch(color='#BDBDBD', label='wrong')]
    ax.legend(handles=handles, fontsize=9)
    n_correct = sum(1 for i in range(N) if (labels[i] == 1) == (s2_scores[i] >= 0.5))
    print(f"  Accuracy @0.5: {n_correct}/{N} = {n_correct/N*100:.1f}%")
    summary['s2_accuracy_at_thresh_05'] = round(n_correct / N, 3)
    _save('F_per_fold_scores.png')

    # F2. Score margin vs subject distance from class mean
    fig, ax = plt.subplots(figsize=(8, 5))
    mdd_centroid = subj_tang[mdd_i].mean(0)
    hc_centroid  = subj_tang[hc_i].mean(0)
    discriminant = subj_tang @ (mdd_centroid - hc_centroid)  # projection onto class axis
    discriminant = (discriminant - discriminant.min()) / (discriminant.max() - discriminant.min())
    for lv, col, nm in [(1, C_MDD, 'MDD'), (0, C_HC, 'HC')]:
        idx = np.where(labels == lv)[0]
        ax.scatter(discriminant[idx], s2_scores[idx], c=col, s=70,
                   alpha=0.8, label=nm, edgecolors='white', lw=0.5)
    r_dc = np.corrcoef(discriminant, s2_scores)[0, 1]
    ax.set_xlabel('Projection onto MDD−HC direction (tangent space)')
    ax.set_ylabel('S2 probability score')
    ax.set_title(f'How well does tangent-space geometry predict the S2 score?\n(r = {r_dc:.3f})', fontsize=11)
    ax.legend()
    summary['discriminant_vs_s2_r'] = round(r_dc, 3)
    _save('F2_discriminant_vs_score.png')


# ── G. Correlation matrix gallery ─────────────────────────────────────────────
print("\n[G] Correlation matrix gallery ...")
n_show = 3
fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
for row, (lv, col, nm, idx_grp) in enumerate([
        (1, C_MDD, 'MDD', mdd_i), (0, C_HC, 'HC', hc_i)]):
    for c_i, si in enumerate(idx_grp[:n_show]):
        ax = axes[row, c_i]
        im = ax.imshow(subj_means[si], cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_title(f'{nm}  {sids[si]}', color=col, fontsize=9, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.suptitle('Per-subject mean correlation matrices — sample subjects\n'
             '(each is the Riemannian mean of their ~60 windows)',
             fontsize=12, y=1.02)
_save('G_corr_gallery.png')

# G2. Grand mean matrices per class (averaged over subjects)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
gm_mdd  = mean_riemann(means_arr[mdd_i])
gm_hc   = mean_riemann(means_arr[hc_i])
gm_all  = mean_riemann(means_arr)
diff_mat = gm_mdd - gm_hc

for ax, mat, title, cmap, vr in [
        (axes[0], gm_mdd,  'MDD group mean',    'RdBu_r', (-1, 1)),
        (axes[1], gm_hc,   'HC group mean',     'RdBu_r', (-1, 1)),
        (axes[2], diff_mat, 'MDD − HC difference', 'PiYG',  (None, None))]:
    vm = max(abs(mat.min()), abs(mat.max())) if vr[0] is None else None
    kw = dict(cmap=cmap, aspect='auto',
              vmin=-vm if vm else vr[0], vmax=vm if vm else vr[1])
    im = ax.imshow(mat, **kw)
    ax.set_title(title, fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.suptitle('Group-level mean correlation matrices + MDD−HC difference', fontsize=12)
_save('G2_group_means.png')


# ── H. Learning curve ─────────────────────────────────────────────────────────
print("\n[H] Learning curve: AUC vs n_training_subjects ...")
rng_lc = np.random.default_rng(42)
ns_lc  = [6, 10, 14, 18, 24, 32, 40, 48, 52]
n_rep  = 30
lc_all = {}

for n_tr in ns_lc:
    n_mdd_tr = min(n_tr // 2, n_mdd - 1)
    n_hc_tr  = min(n_tr - n_mdd_tr, n_hc - 1)
    reps = []
    lr_lc = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
    for _ in range(n_rep):
        mdd_tr = rng_lc.choice(mdd_i, n_mdd_tr, replace=False)
        hc_tr  = rng_lc.choice(hc_i,  n_hc_tr,  replace=False)
        tr_idx = np.concatenate([mdd_tr, hc_tr])
        te_idx = np.array([i for i in range(N) if i not in set(tr_idx)])
        if len(np.unique(labels[te_idx])) < 2:
            continue
        lr_lc.fit(subj_tang[tr_idx], labels[tr_idx])
        sc  = lr_lc.predict_proba(subj_tang[te_idx])[:, 1]
        reps.append(roc_auc_score(labels[te_idx], sc))
    m, s = np.mean(reps), np.std(reps)
    lc_all[n_tr] = {'mean': round(m, 3), 'std': round(s, 3)}
    print(f"  n_train={n_tr:2d}: AUC = {m:.3f} ± {s:.3f}")

summary['learning_curve'] = {str(k): v for k, v in lc_all.items()}

lc_ms = [lc_all[n]['mean'] for n in ns_lc]
lc_ss = [lc_all[n]['std']  for n in ns_lc]

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(ns_lc, lc_ms, 'o-', color='steelblue', lw=2, ms=7, zorder=4)
ax.fill_between(ns_lc,
                [m - s for m, s in zip(lc_ms, lc_ss)],
                [m + s for m, s in zip(lc_ms, lc_ss)],
                alpha=0.25, color='steelblue', label='±1 std  (30 repeats)')
ax.axhline(0.684, color='green',   ls='--', lw=1.5, label='S2 full LOSO 53-fold (0.684)')
ax.axhline(0.715, color='darkred', ls='--', lw=1.5, label='SOTA EEG-RCformer (0.715)')
ax.axhline(0.5,   color='gray',    ls=':',  lw=1)
ax.set_xlabel('Number of training subjects', fontsize=11)
ax.set_ylabel('AUC on held-out subjects (±1 std)', fontsize=11)
ax.set_title('Learning curve: does more training data help?\n'
             '(tangent-space LR, 30 random balanced splits per n)', fontsize=11)
ax.legend(fontsize=9); ax.set_xticks(ns_lc)
_save('H_learning_curve.png')


# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 66)
print("SUMMARY")
print("=" * 66)
print(json.dumps(summary, indent=2))
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
print(f"\nAll outputs -> {OUT}/")
print("Files written:")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name}")
