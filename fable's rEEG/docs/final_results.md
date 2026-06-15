# TRIAD Final Results

**Final model: `nr` (Nuisance Removal) — K=3, 10-fold stratified CV**

## 3×3 Comparison: {nr, nrde, defu} × {MODMA, Mumtaz, OpenNeuro}

| Dataset       | Subjects | nr (K=3) | nrde (K=3) | defu (K=3) |
|---------------|----------|-----------|------------|------------|
| MODMA         | 53       | **0.7227**| 0.7069     | 0.6595     |
| Mumtaz        | 58       | 0.7119    | **0.9512** | 0.7417     |
| OpenNeuro     | 121      | 0.7232    | 0.7046     | **0.7493** |
| **Mean AUC**  |          | 0.7193    | 0.7876     | 0.7168     |

All runs: `--cv-folds 10 --n-nuisance 3`, random_state=42.

## Final Model: `nr` (K=3)

**Pipeline (frozen):**
1. Resample to 125 Hz (250→125 decimate; 256→125 resample_poly; 500→125 decimate)
2. Bandpass 1–40 Hz (4th-order Butterworth)
3. Epoch: 4 s windows, 2 s stride, robust-z artifact rejection (threshold=8)
4. Ledoit–Wolf shrinkage covariance → correlation (scale-invariant)
5. Riemannian mean (training set) → tangent-space projection
6. Identity-subspace removal: project out top-**K=3** between-subject PCs (train-only, leakage-free)
7. Logistic regression (C=1.0, max_iter=1000)

**Why K=3:** Multivariate audit (`audit_identity.py`) confirms the K=3 removed subspace carries zero class information (LR-AUC=0.450, permutation p=0.443). K=8 removes diagnostic signal (p=0.022), explaining the crater at K=8.

## Per-Dataset Signal Source

| Dataset   | Primary signal          | Why `nr` works                                             |
|-----------|-------------------------|------------------------------------------------------------|
| MODMA     | Functional connectivity | Scale-invariant correlation + identity removal isolates it |
| Mumtaz    | Spectral power / 1/f    | `nrde` (0.951) dominates; `nr` still competitive (0.712)  |
| OpenNeuro | Aperiodic 1/f exponent  | `defu` (0.749) edges out; `nr` consistent (0.723)         |

## SOTA Comparison (MODMA, strict LOSO)

| Method                        | AUC    | Protocol  |
|-------------------------------|--------|-----------|
| EEG-RCformer (Zhao 2024)      | 0.7154 | 10-fold   |
| **TRIAD `nr` K=3**            | **0.7227** | **10-fold** |
| TRIAD `nr` K=3 (strict LOSO)  | 0.7256 | LOSO      |
| TRIAD `nr` K=5 (strict LOSO)  | 0.7227 | LOSO      |

## Result Files

```
results/nr_cv10.json          — MODMA nr K=3 (AUC 0.7227)
results/nrde_cv10.json        — MODMA nrde K=3 (AUC 0.7069)
results/defu_cv10.json        — MODMA defu K=3 (AUC 0.6595)
results/mumtaz_nr_cv10.json   — Mumtaz nr K=3 (AUC 0.7119)
results/mumtaz_nrde_cv10.json — Mumtaz nrde K=3 (AUC 0.9512)
results/mumtaz_defu_cv10.json — Mumtaz defu K=3 (AUC 0.7417)
results/opennuero_nr_cv10.json   — OpenNeuro nr K=3 (AUC 0.7232)
results/opennuero_nrde_cv10.json — OpenNeuro nrde K=3 (AUC 0.7046)
results/opennuero_defu_cv10.json — OpenNeuro defu K=3 (AUC 0.7493)
```
