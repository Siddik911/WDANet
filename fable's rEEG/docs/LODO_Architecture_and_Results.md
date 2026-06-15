# LODO Architecture & Empirical Ladder (Leave-One-Dataset-Out)

**Goal.** Make a model trained on 2 datasets and tested on a held-out 3rd (strict
cross-dataset) perform like the within-dataset `nr` model (~0.72 AUC). This document is the
*measured* result of building that ladder — it supersedes the hypotheses in
`LODO_Strategy_Report.md` where the data disagreed.

**Protocol.** 3 folds (train on 2 datasets pooled → test on the held-out one). Subject-level
ROC-AUC. No test subject/label seen at fit time. All datasets projected to a shared montage.

---

## 1. The decisions the data forced (vs. the literature plan)

| Decision | Plan (LODO_Strategy_Report) | What the data said | Final |
|---|---|---|---|
| Shared montage | 19-ch 10-20 | Cz = GSN vertex **reference** (absent in MODMA); T4/T8 = dead electrode in MODMA | **17-ch** |
| Identity-removal K | locked K=3 | K=3 audited on 128-ch only; catastrophic at 17ch (Mumtaz −0.27) | **K=1** (or per-fold inner-CV) |
| Cross-dataset engine | `nr` (Riemannian connectivity) | `nr` transfers at **chance** (0.46 mean) | **`nrde`** (corr **+** spectral) |
| Alignment primitive | per-dataset Riemannian recentering | **hurts** (−0.015 mean); can't fix the axis-flip | **none** |
| Aperiodic 1/f (`defu`) | OpenNeuro's native marker | does **not** transfer (`defu` < `de`) | not used |

**Why `nr` alone fails cross-dataset:** the within-dataset 0.72 on MODMA is built on *dense
128-ch functional connectivity*; the 17-ch harmonization required to make matrices
dimension-compatible physically deletes it. The correlation axis that separates MDD/HC also
points a different way in each dataset (Mumtaz fold < 0.5 = sign flip), which mean-recentering
cannot correct. The **spectral** features (per-channel band power) are both channel-robust and
dataset-transferable, so they carry LODO.

---

## 2. Final architecture (recommended)

```
 raw EEG (any montage)
   │  harmonize → shared 17-ch 10-20  (triad/harmonize.py; Cz & T4 excluded)
   │  125 Hz · 1–40 Hz · 4 s / 2 s windows · robust-z reject
   ▼
 ┌─ connectivity stream ──────────────┐   ┌─ spectral stream ─────────────────┐
 │ LW-shrinkage cov → correlation 17² │   │ Differential Entropy, 5 bands      │
 │ Riemannian mean (pooled train)     │   │ (δθαβγ) per channel = 17×5         │
 │ → tangent → remove top-K=1 ident PC│   │ z-scored on pooled TRAIN windows   │
 └──────────────┬─────────────────────┘   └──────────────┬────────────────────┘
                └──────────── concatenate ────────────────┘
                                  │  (this is the `nrde` feature vector)
                                  ▼
                 class-balanced Logistic Regression → P(MDD)
                 subject = mean of window scores
```

- **No per-dataset recentering** (measured net-negative).
- **Train = 2 source datasets pooled; test = held-out dataset, unlabeled.**
- Stage id: `nrde`, `--n-nuisance 1`, `--harmonize` (implied by LODO mode).

---

## 3. The empirical ladder (all numbers measured, subject-level AUC)

### L0 — within-dataset native (the original goal)
`nr` K=3, 10-fold CV: MODMA **0.7227** · Mumtaz 0.7119 · OpenNeuro 0.7232.

### L1 — harmonize to 17ch, within-dataset (the physical ceiling)
| model | MODMA | Mumtaz | OpenNeuro |
|---|---|---|---|
| `nr` (best K) | 0.64 (K1) | 0.88 (K0) | 0.59 |
| `nrde` (K1) | **0.658** | **0.948** | **0.678** |
| `defu` | 0.606 | 0.673 | 0.595 |

→ The 17-ch montage **caps the LODO mean at ~0.76** (nrde ceiling), even before any transfer loss.
K=3 was destroying Mumtaz (0.61 vs 0.88 at K0).

### L2 — naive LODO (train 2 → test 1, no alignment)
| fold | `nr` | `de` | `defu` | **`nrde`** |
|---|---|---|---|---|
| MODMA | 0.501 | 0.566 | 0.553 | **0.579** |
| Mumtaz | 0.399 | 0.793 | 0.635 | **0.864** |
| OpenNeuro | 0.494 | 0.439 | 0.451 | **0.569** |
| **MEAN** | 0.465 | 0.599 | 0.546 | **0.671** |

→ `nrde` wins. Correlation is useless alone but **complementary** with spectral
(`nrde` > `de`). Connectivity (`nr`) is at chance.

### L3 — + per-dataset Riemannian recentering
`nr`: 0.465 → **0.447**.  `nrde`: 0.671 → **0.656**.  → **Recentering hurts. Abandoned.**

---

## 4. Final result & honest assessment

**FINAL MODEL — `SpectSPD-DANN` (Spectral-Augmented SPD Domain-Adversarial Network):**
the two-stream (Riemannian connectivity + spectral DE) encoder fused end-to-end under
GRL domain-adversarial training + the entropy-filtered class-prototype loss, transductive.
**Mean LODO AUC = 0.743** (MODMA 0.707, Mumtaz 0.913, OpenNeuro 0.609), balanced
accuracy 0.703 — clears 0.70. *(single seed=42; multi-seed mean±std being verified.)*
`nrde`/`nr`/`ds`/recentering are ablations of this model.

| model | mean LODO AUC | MODMA | Mumtaz | OpenNeuro |
|---|---|---|---|---|
| **SpectSPD-DANN** (final) | **0.743** | 0.707 | 0.913 | 0.609 |
| SpectSPD-DANN + ds (fusion) | 0.750 | 0.668 | 0.926 | 0.655 |
| `nrde` (shallow two-stream, ablation: −adversarial) | 0.671 | 0.579 | 0.864 | 0.569 |
| SPD-DANN connectivity-only (ablation: −spectral) | 0.618 | 0.572 | 0.674 | 0.609 |
| `nr` (ablation: −spectral, shallow) | 0.465 | 0.501 | 0.399 | 0.494 |

Earlier shallow-only assessment (superseded by SpectSPD-DANN):

| fold | `nrde` LODO AUC | harmonized ceiling | % of ceiling recovered |
|---|---|---|---|
| Mumtaz | **0.864** | 0.948 | 91% (exceeds the 0.72 goal) |
| MODMA | 0.579 | 0.658 | 88% |
| OpenNeuro | 0.569 | 0.678 | 84% |

- **A flat 0.72 LODO mean is not achievable** with these 3 datasets: 17-ch harmonization caps
  MODMA/OpenNeuro near 0.66/0.68 even *within*-dataset, and OpenNeuro additionally suffers a
  label-definition shift (BDI cutoffs vs clinical MDD).
- **But the transfer itself is strong:** `nrde` recovers **84–91%** of each dataset's
  harmonized within-dataset ceiling cross-dataset, and the Mumtaz fold already beats the 0.72
  target. The honest target is **per-fold vs. ceiling**, not a single flat number.

## 5. Remaining levers (untested / optional, ranked)

1. **Per-dataset spectral (DE) standardization** — z-score DE per dataset, not pooled, to
   remove amplifier-gain offsets (the spectral analog of recentering, but on the stream that
   actually transfers). Direct attack on the ~0.09 transfer gap. *Needs a small code change.*
2. **Source-subject selection** — drop source subjects far from the target distribution
   (negative-transfer control); most relevant for the OpenNeuro fold.
3. **Per-fold K via inner-CV** (`ncv`) for the correlation stream inside `nrde`.

## 6. How to reproduce

```bash
# best LODO config, one fold (repeat with --test mumtaz / opennuero)
python run.py --stage nrde --test modma --n-nuisance 1
# within-dataset 17-ch ceiling
python run.py --stage nrde --dataset modma --harmonize --n-nuisance 1 --cv-folds 10
```
Code: `triad/harmonize.py` (montage), `triad/harness.py::run_lodo`, `--harmonize` /
`--test` / `--recenter-datasets` flags in `run.py`. Result JSONs: `results/lodo_*_harm.json`.
