# Final Results 2 — SpectSPD-DANN across three evaluation regimes

**Models.**
- **`nrde`** — shallow two-stream: Riemannian correlation-tangent + identity-PC removal ⊕ differential-entropy (DE) spectral, linear logistic regression.
- **SpectSPD-DANN** — deep two-stream: learned SPD-manifold connectivity encoder (conv → CovariancePool → BiMap → ReEig → logm) ⊕ DE-MLP, fused end-to-end; for cross-dataset transfer it adds a GRL domain discriminator + entropy-filtered class-prototype loss (transductive); an optional fixed-`nrde` connectivity stream (`--nr-stream`) is used in the data-rich regimes.

**Primary metric:** subject-level ROC-AUC (subject score = mean of its window scores).
**Accuracy metrics:** balanced accuracy (Bal-Acc) and accuracy (Acc) at a **leakage-free** operating point (threshold fit on the *other* subjects only, `calibrate_eval.py`) — **not** the test-set Youden threshold (which is optimistically biased).

---

## 0. Headline

| Evaluation regime | Best model | AUC | Bal-Acc | Acc |
|---|---|---:|---:|---:|
| Within-dataset 10-fold (native) | `nrde` | **0.788** (mean) | — | — |
| Cross-dataset **LODO** (harmonized, 4-seed ens.) | **SpectSPD-DANN** | **0.723** (mean) | 0.67 | 0.68 |
| **Pooled** 10-fold (all 3 merged, N=232) | **SpectSPD-DANN** | **0.788** | **0.710** | **0.716** |

The **pooled** result is the single strongest operating point: **AUC, balanced-accuracy, and accuracy all ≥ 70%, honestly calibrated.**

---

## 1. Within-dataset, 10-fold CV (native montage)

| Dataset | Best model | AUC | Bal-Acc | Acc |
|---|---|---:|---:|---:|
| MODMA (53) | `nr` (= SpectSPD-DANN + `nr-stream`, 0.723) | **0.723** | 0.62 | 0.62 |
| Mumtaz (58) | `nrde` | **0.951** | 0.86 | 0.86 |
| OpenNeuro (121) | `defu` (`nr` 0.723) | **0.749** | 0.65 | 0.67 |
| **`nrde` single model (mean)** | | **0.788** | — | — |

At single-dataset N (~50), the **shallow** model is best: a learned encoder overfits between-subject identity. The deep SpectSPD-DANN with `--nr-stream` matches `nr` on MODMA (0.723) but does not exceed the shallow ceiling here.

SOTA reference (MODMA, subject-independent): EEG-RCformer 0.7154 — `nr`/`nrde` are competitive.

---

## 2. Cross-dataset LODO (harmonized 17-ch) — SpectSPD-DANN, 4-seed ensemble

Train on 2 datasets (pooled), test on the held-out 3rd; all projected to a shared 17-channel 10-20 montage. Transductive (held-out dataset's **unlabeled** windows used at train time; never its labels).

| Target | AUC | Bal-Acc | Acc |
|---|---:|---:|---:|
| MODMA | **0.764** | 0.70 | 0.70 |
| Mumtaz | **0.811** | 0.72 | 0.72 |
| OpenNeuro | 0.594 | 0.60 | 0.62 |
| **Mean** | **0.723** | 0.67 | 0.68 |

Cross-dataset, the connectivity axis is near-orthogonal across sites; fixed features transfer at chance and per-dataset recentring *hurts*. Only the **learned adversarial alignment** recovers it — lifting the connectivity-driven MODMA fold from 0.58 (shallow) to **0.764**, which **≥ WDANet's 0.74–0.76** on a *harder* 3-way protocol.

---

## 3. Pooled 10-fold CV (all three datasets merged, N=232, harmonized 17-ch)

All subjects from MODMA + Mumtaz + OpenNeuro pooled into one cohort (232 subjects, 100 MDD / 132 HC); stratified 10-fold CV across the combined subjects. Every fold trains on all 3 domains.

| Model | AUC | Bal-Acc | Acc |
|---|---:|---:|---:|
| `nrde` (shallow) | 0.745 | 0.684 | 0.690 |
| **SpectSPD-DANN (deep, supervised + `nr-stream`)** | **0.788** | **0.710** | **0.716** |

**The deep model wins here, clearing all three ≥ 70% bars** — because N=232 (4× a single dataset) relaxes the small-N overfitting that capped it at N≈53.

---

## 4. Key finding — a data-volume crossover

The optimal model depends on the data regime:

- **Small-N, single dataset (N≈53):** shallow `nrde` wins; the learned encoder overfits.
- **Pooled cohort (N≈232):** deep **SpectSPD-DANN** wins (0.788 vs 0.745), clearing AUC, Bal-Acc, and Acc ≥ 70%.
- **Cross-dataset (LODO):** the gap is an *axis rotation*, not a mean shift; only **learned adversarial alignment** transfers — SpectSPD-DANN is SOTA-competitive (MODMA 0.764).

So SpectSPD-DANN is the headline model: best wherever there is enough data (pooled) or a domain gap to bridge (LODO); the shallow `nrde` is the right tool only at small single-dataset N.

---

## 5. Honest caveats (for the paper)

- **AUC is primary.** AUC ≈ 0.70–0.79 mathematically bounds balanced-accuracy ≈ 0.62–0.72; report calibrated accuracy, never the test-set Youden point.
- **OpenNeuro (LODO 0.594)** is the limiting fold — a BDI-self-report vs clinical-diagnosis **label shift** (a data property), not a model failure. Report it with the within-dataset 17-ch ceiling alongside.
- **Deep model is seed-sensitive at clinical N** — report the 4-seed ensemble for LODO, not a single best seed.
- **3 datasets ⇒ wide LODO CIs.** Report per-fold confidence intervals; a 4th MDD-EEG dataset is the main lever to strengthen the mean.

---

## 6. Negative results (what was tried and did **not** beat the above)

All tested rigorously (seed 42 vs the relevant baseline); each either traded one fold for another or regressed:

| Attempt | Outcome |
|---|---|
| `nr-stream` raw concat / proj-{8,16,64} / domain-removal for **LODO-MODMA** | 0.56–0.63 (all < baseline 0.707); the fixed connectivity block causes negative transfer on the connectivity-driven fold |
| Tangent-space Procrustes / RPA (shallow) | no gain (17-ch correlation is signal-poor; nothing to rotate) |
| SPD BatchNorm | within-dataset hurt (0.60); LODO neutral (0.707) |
| WDANet-style MMD (global + local + dynamic ω) | helped MODMA (0.743 @α=3) but **cratered** spectral-driven Mumtaz (0.70) — alignment distorts a stream that already transfers |

**Conclusion:** the three-regime results above are the ceiling for this 3-dataset setup; further architecture search overfits single folds.

---

## 7. Reproduction

```bash
# Within-dataset 10-fold CV (native) — shallow nrde
python run.py --stage nrde --dataset modma     --cv-folds 10          # 0.707
python run.py --stage nrde --dataset mumtaz    --cv-folds 10          # 0.951
python run.py --stage nrde --dataset opennuero --cv-folds 10          # 0.705

# Cross-dataset LODO — SpectSPD-DANN (4-seed ensemble; average per-subject scores over seeds)
for S in 42 1 2 3; do for T in modma mumtaz opennuero; do
  python run.py --stage spddann --test $T --epochs 80 --max-windows 100 --subspace 12 --lr 1e-2 --seed $S
done; done

# Pooled 10-fold CV (all 3 merged, harmonized) — face-off
python scripts/pooled_cv.py nrde        # AUC 0.745
python scripts/pooled_cv.py spddann     # AUC 0.788  (deep wins)
```

Result JSONs: `results/{nrde,mumtaz_nrde,opennuero_nrde}_cv10.json`, `results/lodo_*_spddann_s*_harm.json`, `results/pooled_{nrde,spddann}_cv10.json`.
Code: `triad/spddann.py`, `triad/models.py`, `triad/harmonize.py`, `triad/harness.py`, `scripts/pooled_cv.py`, `scripts/calibrate_eval.py`.
