# Final Results — SpectSPD-DANN

**Model.** **SpectSPD-DANN** (Spectral-Augmented SPD Domain-Adversarial Network): a two-stream
encoder — Riemannian functional-connectivity (SPD) + spectral differential-entropy — fused
**end-to-end** inside a gradient-reversal domain-adversarial framework and an entropy-filtered
class-prototype alignment loss. Transductive (uses the held-out dataset's *unlabeled* windows at
train time; never its labels). Architecture diagram: `SpectSPD-DANN_architecture.drawio`.

**Two evaluation protocols.** (1) Within-dataset 10-fold stratified CV (native montage);
(2) strict Leave-One-Dataset-Out (LODO): train on 2 datasets pooled → test on the held-out 3rd,
all projected to a shared **17-channel 10-20** montage (Cz = GSN reference and T4/T8 = dead in
MODMA are excluded). Datasets: MODMA (53), Mumtaz (58), OpenNeuro ds003478 (121). Primary metric:
subject-level ROC-AUC.

---

## 1. Headline

| Setting | Model | Result |
|---|---|---|
| **Cross-dataset (LODO)** | **SpectSPD-DANN, 4-seed ensemble** | **mean AUC 0.723** (MODMA 0.764, Mumtaz 0.811, OpenNeuro 0.594) |
| Within-dataset (10-fold CV) | `nrde` (two-stream, no DA needed — single domain) | **mean AUC 0.788** (MODMA 0.707, Mumtaz 0.951, OpenNeuro 0.705) |

SpectSPD-DANN is the cross-dataset model; within a single dataset it reduces to the two-stream
classifier `nrde` (the domain-adversarial machinery is inactive with only one domain).

---

## 2. Within-dataset, 10-fold CV (native montage)

| Model | MODMA | Mumtaz | OpenNeuro | **Mean** |
|---|---|---|---|---|
| `nr`  (connectivity only) | 0.723 | 0.712 | 0.723 | 0.719 |
| **`nrde`** (connectivity + spectral) | 0.707 | **0.951** | 0.705 | **0.788** |
| `defu` (spectral + aperiodic) | 0.659 | 0.742 | **0.749** | 0.717 |

Per-dataset peak if a different model is allowed per dataset: MODMA `nr` 0.723, Mumtaz `nrde`
0.951, OpenNeuro `defu` 0.749. `nrde` is the best **single** model across all three.

SOTA reference (MODMA, subject-independent): EEG-RCformer 0.7154 — `nr`/`nrde` are competitive.

---

## 3. Cross-dataset LODO — full comparison (harmonized 17-ch)

| Model | MODMA | Mumtaz | OpenNeuro | **Mean** | note |
|---|---|---|---|---|---|
| `nr` (connectivity, auto-K) | 0.501 | 0.399 | 0.494 | 0.465 | connectivity does **not** transfer (chance) |
| `ds` (deep dual-SPD) | 0.513 | 0.496 | 0.669 | 0.559 | depth alone helps only OpenNeuro |
| `defu` (spectral+aperiodic) | 0.553 | 0.635 | 0.451 | 0.546 | aperiodic 1/f does not transfer |
| `de` (pure spectral) | 0.566 | 0.793 | 0.439 | 0.599 | |
| SPD-DANN (connectivity-only) | 0.572 | 0.674 | 0.609 | 0.618 | adversarial+prototype lift connectivity |
| `nrde` (shallow two-stream) | 0.579 | 0.864 | 0.569 | 0.671 | strong shallow baseline |
| **SpectSPD-DANN (single-seed mean ± std)** | 0.717 ±.02 | 0.763 ±.14 | 0.577 ±.03 | 0.686 ± 0.049 | |
| **SpectSPD-DANN (4-seed ensemble)** | **0.764** | **0.811** | 0.594 | **0.723** | **final result** |

Best fusion observed (single-seed): SpectSPD-DANN + `ds` = 0.750; `nrde` + SpectSPD-DANN = 0.731
(report with the same multi-seed caveat as above).

---

## 4. SpectSPD-DANN — seed analysis (the honest robustness picture)

Deep model at clinical N is seed-sensitive; **report the ensemble, not a single seed.**

| Seed | MODMA | Mumtaz | OpenNeuro | Mean |
|---|---|---|---|---|
| 42 | 0.707 | 0.913 | 0.609 | 0.743 |
| 1 | 0.727 | 0.820 | 0.597 | 0.715 |
| 2 | 0.697 | 0.776 | 0.552 | 0.675 |
| 3 | 0.739 | 0.543 | 0.552 | 0.611 |
| **single-seed mean** | 0.717 ± 0.016 | 0.763 ± 0.136 | 0.577 ± 0.026 | **0.686 ± 0.049** |
| **4-seed ensemble** | **0.764** | **0.811** | **0.594** | **0.723** |

- **MODMA fold is the robust win:** every seed 0.70–0.74 (±0.016); ensemble 0.764 — vs shallow
  `nrde` 0.579. The adversarial + prototype training reliably fixes the hardest
  (cross-signal-type) fold.
- **Mumtaz is the variance source** (single seeds 0.54–0.91); ensembling stabilizes it to 0.811.
- **OpenNeuro stays ~0.59** across seeds (its BDI-label + aperiodic-signal ceiling).
- Balanced accuracy (ensemble) = **0.650 @ threshold 0.50** (higher at a tuned/Youden threshold).
  AUC clears 0.70; if the target is balanced-accuracy ≥ 0.70, threshold calibration is still needed.

---

## 5. Ablations (LODO, harmonized 17-ch)

| Variant | Mean AUC | vs base | Conclusion |
|---|---|---|---|
| `nrde` (base) | 0.671 | — | spectral is essential (vs `nr` 0.465) |
| `nrde` − spectral (= `nr`) | 0.465 | −0.206 | connectivity alone fails cross-dataset |
| `nrde` + per-dataset recentering | 0.656 | −0.015 | recentering **hurts** |
| `nrde` + per-dataset DE-standardization | 0.644 | −0.027 | spectral alignment **hurts** too |
| `nr` + recentering | 0.447 | −0.018 | recentering can't fix axis-flip |
| SPD-DANN connectivity-only (− spectral branch) | 0.618 | — | spectral fusion adds +0.10 to the deep model |

**Law observed here:** any *per-dataset alignment of a fixed feature* (Riemannian recentering or
spectral z-scoring) hurts, because it removes class signal on the dataset where it is strongest
(Mumtaz). Alignment must be *learned* (adversarial), not imposed.

---

## 6. Why harmonization caps LODO — within-dataset 17-ch ceilings

The shared 17-ch montage is mandatory (montages differ) but caps each fold:

| Model (17-ch, within-dataset 10-fold) | MODMA | Mumtaz | OpenNeuro | Mean |
|---|---|---|---|---|
| `nrde` ceiling | 0.658 | 0.948 | 0.678 | 0.761 |
| `defu` ceiling | 0.606 | 0.673 | 0.595 | 0.625 |

→ The LODO mean is bounded by ~0.76. SpectSPD-DANN's ensemble (0.723) recovers **95%** of this
ceiling, and on MODMA it actually **exceeds** the shallow within-dataset 17-ch ceiling (0.764 vs
0.658) because it learns a cross-dataset-invariant representation.

---

## 7. Key scientific findings

1. **Connectivity does not transfer across datasets; spectral does.** The MDD/HC *connectivity*
   axis is near-orthogonal across datasets (cosine 0.12–0.32; transfer AUC 0.53 ≈ chance),
   whereas spectral band-power transfers (AUC 0.67). Diagnosed in `diag_recenter.py` /
   `diag_axes_de.py`.
2. **The standard Riemannian alignment primitive (per-dataset recentering) hurts** here — it
   corrects a mean shift, but the gap is an *axis* misalignment. Negative result with mechanism.
3. **Learned (adversarial) alignment + spectral fusion is what works:** end-to-end fusion under
   the domain-adversarial + class-prototype objectives lifts the connectivity-dominated MODMA fold
   from 0.58 (shallow) to 0.76 (ensemble), robustly across seeds.

---

## 8. Reproduction

```bash
# Within-dataset 10-fold CV (native) — model nrde
python run.py --stage nrde --dataset modma     --cv-folds 10      # 0.707
python run.py --stage nrde --dataset mumtaz    --cv-folds 10      # 0.951
python run.py --stage nrde --dataset opennuero --cv-folds 10      # 0.705

# Cross-dataset LODO — SpectSPD-DANN (repeat seeds for the ensemble)
for S in 42 1 2 3; do for T in modma mumtaz opennuero; do
  python run.py --stage spddann --test $T --epochs 80 --max-windows 100 --subspace 12 --lr 1e-2 --seed $S
done; done
# then average per-subject scores across seeds per fold (deep ensemble) -> 0.723
```

Runtime: ~6 min per LODO fold (RTX 3060); full 4-seed sweep ≈ 75 min. Result JSONs under
`results/lodo_*_spddann_s{seed}_harm.json`. Code: `triad/spddann.py`, `triad/harmonize.py`,
`triad/harness.py::run_lodo`.

---

## 9. Honest caveats (for the paper)

- **Report the seed-ensemble (0.723), not the single best seed (0.743).** Single-seed mean is
  0.686 ± 0.049; the 0.743 was the best of 4. The ensemble is robust and standard practice.
- **AUC ≥ 0.70 is met; balanced accuracy @0.5 is 0.65** — needs threshold calibration if a 0.70
  *accuracy* bar is required.
- **OpenNeuro fold (~0.59) is the limiting fold** — capped by its BDI-vs-clinical label definition
  and aperiodic-dominant signal, not by the method.
- 3 datasets = high-variance estimates; report per-fold CIs and the within-dataset 17-ch ceiling
  alongside each LODO fold.
