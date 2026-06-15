# A Novel Cross-Subject Transductive Learning Framework for Leave-One-Subject-Out (LODO) Depression Detection from EEG

**Status:** Research proposal / paper outline
**Scope:** Synthesised from a 98-paper literature review (`Paper Summary.md`) spanning Riemannian/SPD geometry, cross-subject EEG decoding, domain adaptation/generalisation, transfer learning, hyperbolic learning, and EEG depression detection.
**Primary benchmark target:** MODMA (53-subject, 128-channel resting-state EEG, MDD vs. HC) under strict Leave-One-Subject-Out.

> **Reading convention.** Throughout this document, claims grounded in the surveyed literature are tagged **[LIT]** with the originating paper number(s) (e.g., **[LIT P22]**), and newly proposed ideas are tagged **[NEW]**. This separation is maintained deliberately so the reader can always distinguish *observation* from *proposal*.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Common Patterns Identified](#2-common-patterns-identified)
3. [Strong Baselines Identified](#3-strong-baselines-identified)
4. [Proposed Novel Architecture — TRIAD](#4-proposed-novel-architecture--triad)
5. [Novel Training Pipeline](#5-novel-training-pipeline)
6. [Finalised LODO Protocol](#6-finalised-lodo-protocol)
7. [Ablation Study Plan](#7-ablation-study-plan)
8. [Final Research Hypothesis](#8-final-research-hypothesis)

---

# 1. Executive Summary

## 1.1 Main Findings from the Literature

The review yields five convergent, mutually reinforcing findings, each supported by multiple independent papers:

**F1 — Riemannian (SPD-manifold) geometry beats Euclidean processing for cross-subject EEG.** This is the single most strongly corroborated finding. The cleanest causal evidence is SPD-DANN's disentangling ablation **[LIT P98]**: on BCI-IV-2a, a Euclidean DANN reaches 33.4%; adding the *same* alignment losses raises it to 37.3%; switching geometry alone (SPD, no custom losses) reaches 39.8% — i.e., *geometry alone beats an optimised Euclidean pipeline*. EEG-RCformer **[LIT P8]** independently shows AIRM-based processing improves subject-*independent* MODMA depression AUC by **+9.05%** over a Euclidean variant. HybridRDG **[LIT P46]** concludes that "geometric covariance structure, not merely Euclidean feature alignment, is central to zero-shot EEG generalisation." MDRM/TSLDA's Euclidean control **[LIT P9]** (56.2% vs. 70.2%) established the same point in 2012.

**F2 — For *cross-subject* settings specifically, scale-invariant *correlation* matrices outperform raw covariance.** Three independent lines converge: CorSW (Sliced-Wasserstein on the correlation manifold) **[LIT P4]**, the Off-log/correlation-manifold geometry work **[LIT P15]**, and the systematic correlation-manifold deep network **[LIT P68]** all report that removing channel-wise amplitude scaling — the dominant inter-subject confound — improves generalisation. RHOP's quotient-Gaussian normalisation **[LIT P3]** is the same insight applied at the pooling head.

**F3 — Per-subject Riemannian re-centring is the universal cross-domain alignment primitive.** The identical operation — map each domain's Fréchet mean to the identity, $C \mapsto M^{-1/2} C M^{-1/2}$ — is rediscovered under at least five names: rebiasing **[LIT P6, P7]**, S-RCT session centring **[LIT P21]**, parallel transport **[LIT P14, P23]**, tangent-space alignment (TSA, of which rebiasing is a special case) **[LIT P24]**, and the momentum Fréchet-mean tracking inside SPDDSMBN **[LIT P22]**. Its appearance across motor imagery, biometrics, emotion, ADHD/fMRI harmonisation, and neurodegenerative diagnosis marks it as a robust, near-universal tool.

**F4 — Transductive use of unlabelled target data beats pure domain generalisation.** The cross-subject DL survey **[LIT P54]** states this explicitly; the SPD-neuroimaging survey **[LIT P69]** — authored by the SPDDSMBN/TSMNet group — names *test-subject Fréchet-mean adaptation* as an under-explored direction. Empirically, adaptive-reference RK-SVM (ARK) **[LIT P6]**, PT+moments alignment **[LIT P23]**, and test-time normalisation (PSDNorm **[LIT P50]**, SPDDSMBN target statistics **[LIT P22]**) all exploit unlabelled target data and outperform source-only baselines.

**F5 — End-to-end SPD domain-invariant learning is mature, but stops at *training-time* source alignment.** TSMNet + SPDDSMBN **[LIT P22]** (NeurIPS 2022) is the de-facto state of the art for cross-subject SPD UDA, and SPD-DANN **[LIT P98]** (2026) is the first adversarial alignment *on the manifold*. Both align *source* subjects during training; neither performs an explicit, iterative *test-time transductive* refinement against the held-out subject.

## 1.2 Key Limitations of Existing Approaches

| # | Limitation | Evidence | Consequence |
|---|------------|----------|-------------|
| L1 | **Depression-specific cross-subject work is mostly Euclidean.** | WDANet **[LIT P44]** (the strongest depression-DA paper, 10/10) uses a Euclidean CNN; it explicitly *lacks* Riemannian geometry. | Leaves F1's +9% geometric gain unclaimed on the depression task. |
| L2 | **The strongest SPD UDA methods are not transductive.** | SPDDSMBN **[LIT P22]**, SPD-DANN **[LIT P98]** align sources only; SPD-DANN is single-source, train-time-only. | Unlabelled held-out-subject EEG (freely available at test time) is discarded. |
| L3 | **Evaluation is rarely strict LODO with proper metrics.** | EEG-RCformer **[LIT P8]** uses 5 held-out subjects, not full LOSO; WDANet **[LIT P44]** reports accuracy only; SPD-DANN **[LIT P98]** reports accuracy even on an *imbalanced* set. | Inflated/unreliable numbers; no AUC/balanced-accuracy on imbalanced MDD:HC. |
| L4 | **Negative transfer from heterogeneous subjects is unhandled.** | Selective-transfer **[LIT P31]** and SPD-DANN's subject-similarity result **[LIT P98]** (A1↔A3 ≈68% vs. A2↔A3 ≈30%) show some sources hurt. | Pooling all source subjects naively underperforms. |
| L5 | **Class-imbalance corrupts the alignment primitive itself.** | Fixation-based recentring **[LIT P52]**: task-based mean estimation injects class bias under imbalance (AUC +56%/+34% when corrected). | Naive per-subject recentring on imbalanced depression data is biased. |
| L6 | **Covariance estimation is fragile in the small-N clinical regime.** | Ledoit-Wolf shrinkage essential for <50-subject cohorts **[LIT P46]**; MODMA has 53 subjects, 128 channels (128×128 matrices). | Unregularised sample covariance is ill-conditioned; AIRM is $O(C^3)$. |

## 1.3 Motivation for a Transductive Cross-Subject Framework

The LODO depression setting has a structural property that current methods leave on the table: **at inference time, the entire unlabelled resting-state recording of the held-out subject is available.** Depression screening is not a low-latency streaming task — a clinician acquires a multi-minute resting EEG and then asks for a prediction. This means we are not in the strict *inductive* domain-generalisation regime (no target data) but in the *transductive* regime (abundant unlabelled target data, zero target labels).

Three facts make transduction the right paradigm here:

1. **The dominant nuisance is geometric and individually estimable.** Inter-subject shift is, to first order, a Fréchet-mean translation plus a second-order shape change on the SPD/correlation manifold **[LIT P21, P22, P23]**. Both can be estimated *without labels* from the target's own unlabelled covariances.
2. **The literature already validates every ingredient in isolation** — adaptive reference points **[LIT P6]**, parallel transport + moment matching **[LIT P23]**, manifold-safe pseudo-labelling **[LIT P98]**, test-time normalisation **[LIT P50]** — but **no paper combines them into an end-to-end, depression-specific, strict-LODO transductive pipeline.** This is the gap (see §3.2).
3. **The field's own leaders flag it as open** **[LIT P69]**.

## 1.4 Expected Scientific Contribution

We propose **TRIAD** (**Tr**ansductive **Ri**emannian **Ad**aptation for cross-subject Depression detection): a framework that (i) represents EEG on the *scale-invariant correlation manifold*, (ii) learns a *subject-invariant* encoder via momentum Fréchet-mean normalisation and supervised manifold-contrastive prototype learning across source subjects, and (iii) — the novel core — performs **Episodic Transductive Prototype Refinement (ETPR)** at test time: a label-free, manifold-safe inner loop that re-anchors the geometry to the held-out subject and refines class prototypes from confidence-filtered pseudo-labels. Crucially, TRIAD is **meta-trained to be transduced**: the test-time adaptation loop is *simulated on held-out source subjects during training*, so the encoder learns representations on which a few unlabelled adaptation steps measurably help.

Anticipated contributions (detailed in §8.3):

- **Methodological:** the first framework to unify training-time source-invariance (SPDDSMBN-style) with an explicit, *meta-learned*, test-time transductive prototype-refinement loop on the **correlation** manifold, specialised for depression.
- **Algorithmic:** ETPR — a manifold-safe, class-imbalance-robust, entropy-filtered transductive label-propagation procedure with closed-form prototype updates.
- **Experimental:** the first *strict-LODO*, AUC/balanced-accuracy, statistically-validated MODMA depression benchmark that re-runs strong baselines (EEG-RCformer, WDANet-Riemannianised, SPDDSMBN, SPD-DANN) under identical folds, cleanly separating the value of *geometry* from the value of *transduction*.
- **Practical:** a calibrated (ECE-controlled), privacy-compatible (source-free-capable) screening model deployable per new subject with zero labelled calibration data.

---

# 2. Common Patterns Identified

This section abstracts recurring choices across the 98 papers. Each pattern is presented with *evidence*, *why researchers use it*, and *its limitations*.

## 2.1 Frequently Used Datasets

| Dataset | Paradigm | Subjects / size | Used by (examples) | Role in our work |
|---------|----------|-----------------|--------------------|------------------|
| **MODMA** | Resting-state, MDD vs. HC | 53 subj, 128-ch | **P8, P44** | **Primary benchmark** |
| Self-collected depression sets | Resting-state, MDD vs. HC | up to 170 subj | **P44** | Cross-dataset validation |
| Subclinical depression sets | Resting-state | 3 datasets | **P45** | Out-of-distribution check |
| SEED / SEED-IV / SEED-V | Stimulus emotion | 15–45 subj, 62-ch | P5, P8, P12, P25 | Affect transfer prior |
| BCI Competition IV 2a/2b | Motor imagery | 9 subj, 22/3-ch | P2, P6, P9, P10, P13, P22, P98 | Method development / sanity |
| PhysioNet MI | Motor imagery | 103–109 subj | P7, P17 | Large-N alignment tests |
| TUAB / TUEV / TUSZ | Clinical EEG events | 100K+ segments | P3, P13 | Foundation-model context |
| Sleep staging (10 sets) | Polysomnography | 10K subj | P50 | Test-time-norm precedent |
| ADHD-200 / ABIDE / ADNI (fMRI) | Clinical connectivity | 1000s | P15, P48, P49, P71 | Harmonisation precedent |

**Why these.** MODMA is the *only* widely-shared, high-density, resting-state MDD-vs-HC EEG dataset with published cross-subject numbers **[LIT P8, P44]**, making it the natural anchor. MI datasets dominate methods papers because they are standardised and the SPD pipeline matured there.

**Limitations.** (i) The depression benchmark base is *thin* — essentially MODMA plus private sets, raising over-fitting and reproducibility concerns. (ii) MI-derived conclusions do not automatically transfer: resting-state depression EEG has higher inter-subject variance and weaker, oscillatory (not event-locked) signals **[LIT P8, P25]**. (iii) Small N (53) with 128 channels is a high-dimension/low-sample regime **[LIT P46]**.

## 2.2 Common Feature Representations

| Representation | Evidence | Why used | Limitation |
|----------------|----------|----------|------------|
| **Sample covariance (SPD)** | P6, P9, P22, P65 (foundational) | Universal 2nd-order EEG descriptor; manifold structure exploitable | Scale-dependent → inter-subject amplitude confound; ill-conditioned at high C |
| **Correlation matrices (quotient/scale-invariant)** | **P3, P4, P15, P68** | Removes amplitude scaling — the main inter-subject nuisance (F2) | Distinct quotient manifold; fewer mature tools |
| **Functional connectivity (PLV/PCC/COH/MI → SPD)** | P5, P53 | Captures brain-network coupling; PCC/PLV stable across subjects | Not natively SPD (needs Laplacian+max trick); MI noisy |
| **PSD + Differential Entropy (band features)** | **P8**, P45 | Well-validated spectral depression markers; subject-stable | First-order only; discards covariance structure unless paired |
| **Wavelet multi-scale + Riemannian** | **P45 (WRFE)** | Joint temporal-dynamics + spatial connectivity | Heavier; diffusion-trained variant data-hungry |
| **DCCA / detrended cross-correlation** | P7 | Removes slow drifts before covariance (resting-state fatigue/drowsiness) | Modest gains; scale tuning |

**Convergent signal.** Across P3, P4, P15, P68 the field is migrating *covariance → correlation* for cross-subject tasks, and across P8, P45 it pairs *spectral first-order* with *Riemannian second-order* features. **[NEW]** TRIAD adopts **multi-band correlation matrices with Ledoit-Wolf shrinkage** as the primary representation (scale-invariant + small-N-robust), optionally fused with PSD+DE first-order context.

## 2.3 Typical Model Architectures

```
Archetype A  (shallow geometric):   Cov/Corr → (rebias/TSA) → Tangent space → LDA/SVM/MDM
                                     P6, P9, P21, P23, P24

Archetype B  (deep SPD network):    Cov → BiMap → ReEig → (SPD-BN) → LogEig → linear
                                     P22, P65, P66, P68

Archetype C  (SPD + graph/attention): Cov tokens → manifold attention / Graph-BiMap → pool
                                     P3, P8, P13, P16, P17, P25, P72, P73

Archetype D  (Euclidean DL + DA):    raw EEG → CNN/Transformer → adversarial/MMD DA
                                     P10, P29, P44, P74, P77

Archetype E  (hyperbolic):           EEG → Lorentz/Poincaré embedding → hyperbolic attention
                                     P80, P81, P82
```

| Architecture | Why used | Limitation |
|--------------|----------|------------|
| A — shallow geometric | Strong, interpretable, small-data-friendly; foundational | No representation learning; alignment hand-crafted |
| B — deep SPD | End-to-end, geometry-preserving; SoA UDA via SPD-BN | Cubic cost at high C; needs Stiefel/pseudo-AI reduction |
| C — SPD + attention/graph | Captures channel structure + non-stationarity; token reduction | Heavier; clustering can be unstable |
| D — Euclidean DL+DA | Mature DA toolbox (adversarial, Wasserstein); depression SoA (WDANet) | Discards manifold geometry (F1) |
| E — hyperbolic | Models hierarchical subject/brain structure; EEG shown δ-hyperbolic | Numerically delicate; complementary not core |

**[NEW]** TRIAD is an **Archetype B/C hybrid on the correlation manifold**: correlation-manifold BiMap/ReEig blocks + AIRM channel clustering (128→K tokens) + manifold attention, with **Stiefel/pseudo-AI dimensionality reduction** **[LIT P11, P17]** for 128-channel feasibility.

## 2.4 Training Paradigms

| Paradigm | Evidence | Why | Limitation |
|----------|----------|-----|------------|
| Supervised within-subject | P1, P2, P5, P9, P13, P16 | Simplest; high accuracy | No cross-subject generalisation; often leaky CV |
| Source-only Domain Generalisation | P4 (CorSW), P36, P46 | No target data needed; deployable cold | Ignores available unlabelled target (F4) |
| Unsupervised Domain Adaptation (train-time) | **P22, P23, P24, P25, P33, P44, P98** | Uses unlabelled target distribution; strong | Target seen only at train time; no per-subject test refinement |
| Source-Free DA | P28, P67 | Privacy-preserving; model-only transfer | Pseudo-label noise; instability |
| Meta-learning / few-shot | P57, P95 | Fast adaptation to new subjects | Usually needs *labelled* few-shot target |
| Test-time adaptation / transduction | **P6 (ARK), P50 (PSDNorm)** | Per-subject, label-free, at inference | Under-explored on SPD for depression (the gap) |

## 2.5 Evaluation Protocols

| Protocol | Rigour | Used by | Note |
|----------|--------|---------|------|
| Within-subject CV | Low | P1, P2, P5, P13, P16, P19, P20 | Risk of temporal leakage (P16's 96% suspicious) |
| Session-to-session | Low–Med | P6, P7, P9, P10, P21 | Cross-session ≠ cross-subject |
| 5-held-out / k-fold subject | Med | **P8** | Not full LOSO |
| **Strict LOSO / LODO** | **High** | **P17, P22 (≈), P24, P31, P45, P46** | Gold standard (F: P54) |
| Cross-dataset | Highest | P22, P24, P44, P50 | Hardest; rare for depression |

**Metric usage.** Accuracy dominates (often *only* metric: P44, P98); AUC used where imbalance is acknowledged (P8, P52); balanced accuracy + Cohen's κ in clinical/imbalanced sets (P3, P22, P46); calibration (ECE) rare but valuable (P17). **[NEW]** TRIAD reports the *full* panel (§6.2) including AUC, balanced accuracy, MCC, and ECE.

## 2.6 Recurrent Strengths and Weaknesses

**Recurrent strengths (what the field gets right).**
- **S1.** Manifold geometry is load-bearing for cross-subject transfer (F1) — verified by ablations in P9, P8, P98.
- **S2.** Fréchet-mean recentring reliably reduces domain shift (F3) — verified across 6+ task domains.
- **S3.** Scale-invariance (correlation/quotient-Gaussian) consistently helps cross-subject (F2).
- **S4.** Source selection matters — similar subjects transfer far better (P31, P35, P98).

**Recurrent weaknesses (the openings).**
- **W1.** Depression work ignores geometry (L1); geometry work ignores depression.
- **W2.** Unlabelled *test-subject* data is wasted by train-time-only UDA (L2, F4).
- **W3.** Evaluation under-rigorous: rarely strict LODO, rarely AUC, rarely statistically tested (L3).
- **W4.** Class-imbalance both *un-metricised* (accuracy-only) and *un-corrected* in recentring (L3, L5).
- **W5.** Pseudo-labels are geometrically dangerous on the manifold but used naively outside P98.

---

# 3. Strong Baselines Identified

## 3.1 Existing State-of-the-Art Baselines

### 3.1.1 Traditional ML / shallow geometric

| Method | Core Idea | Advantages | Weaknesses | LODO Suitability |
|--------|-----------|-----------|------------|------------------|
| **MDRM** (P9) | Nearest Riemannian class-mean on SPD | Parameter-free, interpretable, small-N robust | No representation learning; no alignment | **Baseline-tier** — strong only with recentring |
| **TSLDA** (P9) | Tangent-space LDA at global Fréchet mean | Linear, fast, ANOVA feature selection | Within-subject by default; needs alignment | **Baseline-tier** with TSA/rebias |
| **RK-SVM / ARK** (P6) | Riemannian-kernel SVM; adaptive reference | First *adaptive* (transductive) reference point | SVM scaling; cross-session not cross-subject | **Good** — ARK is a transductive proto-baseline |
| **RRSLVQ-LEML** (P12) | Prototype LVQ on SPD + metric learning | Few-shot robust; learnable prototypes | Within-subject; LEM not AIRM | Medium |
| **FC + SPDNet** (P5) | PLV/PCC→Laplacian→SPD→SPDNet | Brain-network features | Subject-dependent only | Low without DA |

### 3.1.2 Deep learning baselines

| Method | Core Idea | Advantages | Weaknesses | LODO Suitability |
|--------|-----------|-----------|------------|------------------|
| **SPDNet** (P65) | BiMap/ReEig/LogEig deep SPD net | Foundational end-to-end Riemannian | No DA module; cubic cost | **Backbone**, not standalone |
| **EEG-RCformer** (P8) | AIRM channel clustering + Transformer | **MODMA depression SoA**, AUC 0.7154 (sub-indep.) | 5-held-out not LOSO; no explicit DA | **Primary baseline to beat** |
| **WRFE / ReTa-Diffusion** (P45) | Wavelet + Riemannian + diffusion | Depression LOSO; strong features | Needs task-state EEG at train; multiclass | High (feature module) |
| **HybridRDG** (P46) | EEGNet ⊕ Riemannian, Ledoit-Wolf, zero-shot | Clinical, <50 subj, geometry-first | ASD not depression; LEM | High (architecture template) |
| **Stiefel-SPD GCN** (P17) | Stiefel→Graph-SPD→Log, full geometry | LOSO, calibrated (ECE≤0.04), 1 HP set | MI only; no transduction | High (reduction + calibration) |

### 3.1.3 Graph-based methods

| Method | Core Idea | Advantages | Weaknesses | LODO Suitability |
|--------|-----------|-----------|------------|------------------|
| **RepSPD** (P13) | SPD ⊕ dynamic-graph cross-attention | Captures non-stationary connectivity | Within-subject; expensive | Medium (fusion idea) |
| **RST-GNN / Graph-BiMap** (P16) | Manifold-preserving graph convolution | SPD-consistent aggregation | Within-subject; leaky CV | Medium (component) |
| **RGT-GAA** (P25) | Riemannian graph transformer + geodesic adversarial | Cross-subject emotion SoA | SEED stimulus-evoked | High (geodesic-adv loss) |

### 3.1.4 Transformer-based methods

| Method | Core Idea | Advantages | Weaknesses | LODO Suitability |
|--------|-----------|-----------|------------|------------------|
| **RHOP head** (P3) | Quotient-Gaussian 2nd-order pooling | Scale-invariant, ~1K params, plug-in | No DA; population-level eval | High (classifier head) |
| **SPD-Token Transformer** (P72) | BWSPD token embeddings | Better gradients at high C (128-ch) | Within-subject | Medium (embedding choice) |
| **GyroAtt** (P73) | Gyrovector attention on matrix manifolds | Geometry-preserving attention | General, not depression | Medium (attention module) |
| **Spatiotemporal/cross-subj. Transformer** (P29) | Zero-calibration cross-subject attention | No target data needed | Euclidean | Medium |

### 3.1.5 Domain adaptation methods

| Method | Core Idea | Advantages | Weaknesses | LODO Suitability |
|--------|-----------|-----------|------------|------------------|
| **TSMNet + SPDDSMBN** (P22) | Per-domain momentum Fréchet-mean SPD BatchNorm | **SoA SPD UDA**, end-to-end, proven convergence | Source-align only; needs domain ID; MI/ERP | **Core component & key baseline** |
| **SPD-DANN** (P98) | Adversarial (GRL) DA *on* SPD + entropy-filtered prototype loss | First manifold-adversarial; manifold-safe pseudo-labels | Single-source, train-time only; cov+LEM; MI | **Strong baseline & loss source** |
| **WDANet** (P44) | Global+local+Wasserstein discriminators, dynamic factor | **Depression cross-domain SoA** | Euclidean; not strict LODO; acc-only | **Primary non-geometric baseline** |
| **CorSW** (P4) | Sliced-Wasserstein DG on correlation manifold | Scale-invariant, zero inference cost | Source-only DG | High (training regulariser) |
| **PT + Moments** (P23) | Parallel transport ⊕ tangent moment match | Transductive; mean+shape alignment | Shallow; small-scale | High (transductive primitive) |
| **TSA** (P24) | Tangent-space affine source→target alignment | 18-DB validated; cross-channel-count | Shallow SVM; needs target mean | High (alignment) |
| **BARN-DA / R-MMD** (P33) | Band-aware + Riemannian MMD | Band-specific alignment | Session-level; MI | Medium |
| **Geodesic Adversarial (GAA)** (P25) | Adversarial loss in AIRM metric | Geometry-faithful adversary | Emotion only | Medium |

### 3.1.6 Subject-generalisation methods

| Method | Core Idea | Advantages | Weaknesses | LODO Suitability |
|--------|-----------|-----------|------------|------------------|
| **DGFEL** (P36) | Covariance-centroid align + adversarial DG | Calibration-free, true DG | ERP-specific (xDAWN) | Medium |
| **Selective transfer** (P31) | Pick K most-similar source subjects (tangent MMD) | Avoids negative transfer | Small-N MI | High (source selection) |
| **Rotation-based selection** (P35) | Eigenvector-orientation source similarity | Complements AIRM selection | MI | Medium |
| **Stiefel routing** (P26) | K-expert subspace routing, anti-collapse conditions | Subject-adaptive subspaces | Overkill small-N | Medium |
| **Hyperbolic (LAtte/HEEGNet)** (P80, P81) | Hierarchical subject embedding | EEG shown δ-hyperbolic; weighted selection | Numerically delicate | Medium (subject weighting) |

### 3.1.7 Baseline ladder (what we will actually run)

**[NEW]** We define a *monotone difficulty ladder* so each rung isolates one factor (geometry, alignment, transduction):

```
R0  Euclidean CNN (EEGNet)                         ── no geometry, no DA      (floor)
R1  MDRM + per-subject rebias (P6/P9/P21)          ── geometry + recentring
R2  TSLDA + TSA (P9/P24)                            ── + tangent alignment
R3  SPDNet (P65)                                    ── deep geometry, no DA
R4  TSMNet + SPDDSMBN (P22)                         ── SoA source-invariant SPD UDA
R5  SPD-DANN, multi-source re-run (P98)            ── manifold-adversarial UDA
R6  EEG-RCformer (P8)                               ── MODMA depression SoA (AUC 0.7154*)
R7  WDANet, Riemannian-ised (P44 + SPD features)   ── depression DA + geometry
────────────────────────────────────────────────────────────────────────────────
R8  TRIAD (ours)                                    ── R4-class invariance + ETPR transduction
```
*\*EEG-RCformer's 0.7154 is on 5-held-out, not full LOSO; we re-run it under identical strict-LODO folds for a fair comparison.*

## 3.2 Research Gaps Identified

For each gap: **(1) what is missing, (2) why it matters, (3) how TRIAD addresses it.**

### 3.2.1 Cross-Subject Challenges

**G1 — Subject heterogeneity (between-subject variance ≫ between-class variance).**
1. *Missing:* Depression methods either pool all subjects (ignoring heterogeneity) or use a single global alignment; MDD subtypes form distinct clusters that a single class prototype cannot capture.
2. *Why it matters:* EEG-RCformer's sub-independent AUC (0.715) is far below its sub-dependent AUC (0.980) **[LIT P8]** — a 26-point generalisation gap that is *the* problem. Subject-similarity drives transfer **[LIT P98]**.
3. *TRIAD:* **multicentric class prototypes** (multiple Riemannian means per class, P28-style but manifold-native) + **source-similarity-weighted** training (P31/P35/P81) + per-subject normalisation (P22).

**G2 — Distribution shift (geometric domain shift on the manifold).**
1. *Missing:* shift is decomposed only partially — most methods correct the *mean* (rebias) but not higher-order *shape*; P23 corrects both but is shallow and train-time.
2. *Why it matters:* "PT alone fails when source/target share a mean but differ in distribution" **[LIT P23]**; structural shift persists after recentring.
3. *TRIAD:* SPDDSMBN handles source mean shift in-training; **ETPR adds a test-time target Fréchet update *plus* tangent second-moment matching** (P23 made transductive and end-to-end).

**G3 — Label imbalance (MDD:HC not 1:1).**
1. *Missing:* depression DA reports accuracy only (P44, P98) and recentres using all trials, injecting class bias (P52).
2. *Why it matters:* accuracy is misleading under imbalance; biased recentring degrades the decision margin asymmetrically (AUC swing of +34–56% when corrected, P52).
3. *TRIAD:* **class-agnostic baseline-period recentring** (P52) + **balanced pseudo-label selection** (P28) + full imbalanced-aware metric panel (AUC, balanced acc, MCC).

**G4 — Domain mismatch (channel/montage/site variability).**
1. *Missing:* most SPD methods assume fixed channels; cross-dataset depression transfer is rare.
2. *Why it matters:* deploying MODMA-trained models on other montages requires channel-count-agnostic alignment.
3. *TRIAD:* tangent-space alignment is channel-count-agnostic (P24); AIRM clustering maps any montage to K region-tokens (P8); optional source-space projection (P43).

### 3.2.2 Methodological Gaps

**G5 — Lack of transductive learning on the SPD/correlation manifold for depression.** *(the central gap)*
1. *Missing:* No paper combines (a) correlation-manifold representation, (b) end-to-end source-invariant SPD training, and (c) an explicit *iterative test-time transductive* refinement against the held-out subject — for depression, under strict LODO. SPDDSMBN **[LIT P22]** stops at (b); P23 has (c) but shallow; P6's ARK has a *one-shot* reference update only; the SPD survey **[LIT P69]** names this exact gap.
2. *Why it matters:* discards the freely-available unlabelled target recording (F4); transduction is shown to beat DG across the board (P54).
3. *TRIAD:* **ETPR** is precisely this missing loop, and is **meta-trained** so the encoder is *optimised to be transduced*.

**G6 — Weak personalisation.**
1. *Missing:* one global model per fold; no per-subject parameters.
2. *Why it matters:* subjects occupy different manifold regions (P26); a single subspace under-fits.
3. *TRIAD:* per-subject **transductive prototypes** and (optional) **Stiefel expert routing** (P26) give lightweight, label-free personalisation.

**G7 — Limited adaptation to unseen subjects.** Addressed by G5/G6 jointly.

**G8 — Insufficient utilisation of unlabelled test-subject data.**
1. *Missing:* only P6, P50 use it at test time; neither is depression+SPD+iterative.
2. *Why it matters:* this is the highest-leverage unused signal in LODO.
3. *TRIAD:* ETPR consumes the *entire* unlabelled target recording for mean/shape alignment + pseudo-label prototype refinement + entropy minimisation.

**G9 — Over-reliance on subject-independent (source-only) training.**
1. *Missing:* DG methods (P4, P36, P46) deliberately avoid target data.
2. *Why it matters:* leaves accuracy on the table where target data exists.
3. *TRIAD:* uses DG (CorSW) as a *regulariser/initialisation* and then *transduces* — strictly dominates source-only when unlabelled target is present (testable: §7).

### 3.2.3 Evaluation Gaps

**G10 — Incomplete LODO evaluation.** *(missing: full LOSO; matters: 5-held-out inflates/destabilises; TRIAD: strict 53-fold nested LODO, §6).*
**G11 — Small sample sizes.** *(missing: 53 subjects → high variance; matters: single splits unreliable; TRIAD: subject-level bootstrap CIs, ≥5 seeds, permutation tests, §6.3).*
**G12 — Poor robustness analysis.** *(missing: no sensitivity to imbalance, artifacts, HP; matters: clinical deployment needs it; TRIAD: robustness suite — imbalance sweep, artifact stress, calibration/ECE, source-count ablation, §7).*

> **Gap → contribution map.** G5 (+G8, G9) is the *primary* novelty target (ETPR + meta-transduction). G1–G4 are addressed by design choices that are individually literature-backed but **newly combined**. G10–G12 are addressed by the protocol (§6) and are themselves a contribution given L3/W3.

---

# 4. Proposed Novel Architecture — TRIAD

## 4.1 High-Level Concept

> **Core intuition [NEW].** *Inter-subject variation in resting-state EEG is, to good approximation, a smooth geometric transport on the correlation manifold plus a label-free re-clustering of class prototypes. If we (i) represent EEG where the dominant nuisance (amplitude scaling) is quotiented out, (ii) train an encoder that is explicitly invariant to source-subject identity, and (iii) — at test time — let the held-out subject's own unlabelled data **re-anchor the geometry and re-estimate the class prototypes**, then a model meta-trained to benefit from this re-anchoring will generalise to unseen subjects far better than any source-only or train-time-only-UDA model.*

TRIAD therefore has two operating regimes that *share weights*:

- **Inductive backbone** (trained on source subjects): a correlation-manifold encoder + subject-invariant normalisation + multicentric prototype head.
- **Transductive wrapper** (ETPR, run per held-out subject at test time): label-free re-anchoring + pseudo-label prototype refinement, **simulated during training** on held-out source subjects so the backbone *learns to be adapted*.

The design is the union of the strongest, mutually-compatible literature components, with **one genuinely new mechanism (meta-trained ETPR)** binding them:

| Need (gap) | Borrowed enabling idea | Source |
|------------|------------------------|--------|
| Scale-invariant features | correlation manifold + quotient-Gaussian | P3, P4, P15, P68 |
| Small-N robust covariance | Ledoit-Wolf shrinkage | P46 |
| 128-ch tractability | AIRM clustering + Stiefel/pseudo-AI reduction | P8, P11, P17 |
| Source-subject invariance | momentum Fréchet-mean SPD-BN (SPDDSMBN) | P22 |
| Subject-invariant discrimination | supervised manifold contrastive + prototypes | P28, P32, P98 |
| **Test-time transduction** | **adaptive reference + PT/moments + entropy-filtered prototypes** | **P6, P23, P98, P50** |
| Imbalance-safe recentring | class-agnostic baseline-period mean | P52 |
| Calibrated clinical output | LogEig + temperature scaling, ECE | P17 |

## 4.2 Architecture Components

Notation: $\mathcal{S}_{++}^C$ = SPD manifold of $C\times C$ matrices; $\mathcal{C}or_C$ = full-rank correlation manifold; $\delta_R(\cdot,\cdot)$ = AIRM geodesic distance; $\mathrm{Log}_M$, $\mathrm{Exp}_M$ = Riemannian log/exp at base $M$; $\mathfrak{M}(\{C_i\})=\arg\min_M\sum_i \delta_R^2(M,C_i)$ = Fréchet mean.

### Component 1 — Feature Encoder $f_\theta$

- **Inputs:** raw multi-channel EEG of one subject, segmented into overlapping windows; band-pass into $B$ physiological bands $\{\delta,\theta,\alpha,\beta,(\gamma)\}$ (depression biomarkers are band-specific — alpha asymmetry, theta elevation **[LIT P33, P21]**).
- **Process:**
  1. Per band $b$, per window $w$: shrinkage covariance $\hat\Sigma_{b,w}=(1-\rho)\,S_{b,w}+\rho\,\nu I$ (Ledoit-Wolf) **[LIT P46]**; optional DCCA detrending for slow drifts **[LIT P7]**.
  2. Normalise to **correlation** $R_{b,w}=D^{-1/2}\hat\Sigma_{b,w}D^{-1/2}$, $D=\mathrm{diag}(\hat\Sigma_{b,w})$ — quotients out amplitude scaling **[LIT P3, P4, P68]**.
  3. **AIRM channel clustering** (128→K region tokens, K≈5) via farthest-point sampling on the manifold + EMA-stable spatial centres **[LIT P8]**, yielding per-token reduced correlation matrices.
- **Outputs:** a set $\{R^{(t)}\}_{t=1}^{K\cdot B\cdot W}$ of reduced correlation matrices ("SPD/correlation tokens").
- **Mathematical role:** maps raw EEG to points on $\mathcal{C}or_{d}$ ($d\ll C$) where inter-subject amplitude shift is removed and dimensionality is tractable.
- **Expected benefit:** addresses F2 (scale-invariance), L6 (shrinkage), 128-ch cost (clustering), G4 (montage-agnostic tokens).

### Component 2 — Subject-Invariant Representation Module $g_\phi$

- **Inputs:** correlation tokens $\{R^{(t)}\}$ plus *source-subject identity* $s$ (train only).
- **Process:** stack of **correlation-manifold BiMap/ReEig blocks** **[LIT P68]** (scale-invariant analogues of SPDNet **[LIT P65]**), interleaved with **SPDDSMBN** **[LIT P22]**: maintain per-subject momentum Fréchet means $\mu_s$ and transport features to a common identity reference,
$$\Gamma_s(R)=\mu_s^{-1/2}\,R\,\mu_s^{-1/2},\qquad \mu_s \leftarrow \mathfrak{M}\big(\mu_s, \bar R_{\text{batch},s}\big)\ \text{(momentum)} .$$
Optional **Stiefel projection** $B\in\mathrm{St}(d,d')$ for further reduction **[LIT P17]**, with anti-collapse routing if multiple experts **[LIT P26]**.
- **Outputs:** subject-invariant manifold features $\{Z^{(t)}\}$ (still on $\mathcal{C}or_{d'}$).
- **Mathematical role:** removes the *first-order* (Fréchet-mean) component of inter-subject shift inside the network, with proven convergence of $\mu_s\!\to\!$ true Fréchet mean **[LIT P22]**.
- **Expected benefit:** addresses G1/G2 (heterogeneity, mean shift); this is the strongest *training-time* invariance mechanism in the literature.

### Component 3 — Subject Prototype Memory $\mathcal{P}$

- **Inputs:** invariant features $\{Z^{(t)}\}$ with source labels $y\in\{\text{MDD},\text{HC}\}$.
- **Process:** maintain a manifold **memory bank** of:
  - *Global* class prototypes $P_k^{\text{glob}}=\mathfrak{M}(\{Z : y=k\})$;
  - *Multicentric* sub-prototypes $\{P_{k,m}\}_{m=1}^{M}$ per class via Riemannian K-means (captures MDD subtypes / HC variability) **[LIT P28]**;
  - *Per-source-subject* class means $P_k^{(s)}$ (for source-similarity weighting & negative-transfer control) **[LIT P31, P98]**.
  Updated by momentum on the manifold (Fréchet averaging).
- **Outputs:** prototype set used by the prediction head and, crucially, the **initialisation** for test-time ETPR.
- **Mathematical role:** non-parametric, geometry-respecting class model; prototypes are points on $\mathcal{C}or_{d'}$ whose geodesic distances define the classifier.
- **Expected benefit:** addresses G1 (subtype structure), G6 (personalisation hook), and makes the classifier *adaptable without gradient steps* (prototypes can be re-estimated transductively).

### Component 4 — Graph / Attention Component $a_\psi$ (beneficial, ablatable)

- **Inputs:** token features $\{Z^{(t)}\}$ across bands/regions/windows.
- **Process:** **manifold attention** — RHOP quotient-Gaussian second-order pooling **[LIT P3]** and/or GyroAtt gyrovector attention **[LIT P73]** — over tokens, with a **geodesic-distance graph** (edges by AIRM proximity) **[LIT P25]** so functionally-similar tokens (not just spatially adjacent) attend to each other. Captures temporal non-stationarity across windows (resting-state drift) **[LIT P13, P71]**.
- **Outputs:** a single fused manifold descriptor $\bar Z$ per subject-segment.
- **Mathematical role:** geometry-preserving aggregation of first- + second-order statistics across the token set.
- **Expected benefit:** addresses non-stationarity; RHOP adds ~1K params (low overfit risk in small-N) **[LIT P3]**. Ablated in §7 to quantify marginal value.

### Component 5 — Transductive Adaptation Module (ETPR) — **the novel core [NEW]**

- **Inputs (test time):** the held-out subject's **unlabelled** correlation tokens $\{R_*^{(t)}\}$ (whole recording), the trained $f_\theta,g_\phi,a_\psi$, and prototype memory $\mathcal{P}$.
- **Process (one ETPR pass; iterate $n_{\text{adapt}}$ times):**
  1. **Class-agnostic re-anchoring [LIT P52 + P22, NEW combination].** Estimate the target Fréchet mean $\mu_*$ from *baseline/rest* segments only (avoids class-imbalance bias), then set the SPDDSMBN target reference to $\mu_*$: $\Gamma_*(R)=\mu_*^{-1/2}R\,\mu_*^{-1/2}$.
  2. **Second-moment (shape) matching [LIT P23, NEW transductive form].** In the tangent space at $\mu_*$, match target token covariance to the source-aggregate covariance (closed-form whitening), correcting the structural shift PT-alone misses.
  3. **Pseudo-label generation [LIT P9 MDM-style].** For each target token, soft-assign $q_k(R_*) \propto \exp(-\delta_R^2(\,\Gamma_*(R_*),\,P_k\,)/\tau)$ to the (multicentric) prototypes.
  4. **Confidence filtering [LIT P98, NEW on correlation manifold].** Keep only tokens with low predictive entropy $H(q)<\tau_H$ **and** balanced per-class quotas **[LIT P28, P52]** — because a wrong pseudo-label distorts a Fréchet mean *non-linearly* on the manifold (error amplification is geometry-specific) **[LIT P98]**.
  5. **Transductive prototype refinement [NEW].** Re-estimate *target-specific* prototypes from confident tokens and **geodesically interpolate** with the source prototype (controls negative transfer):
$$ P_k^{*} \;=\; \mathrm{Exp}_{P_k^{\text{glob}}}\!\Big(\alpha \,\mathrm{Log}_{P_k^{\text{glob}}}\big(\mathfrak{M}\{\Gamma_*(R_*^{(t)}) : \hat y=k,\,H<\tau_H\}\big)\Big),\quad \alpha\in[0,1]. $$
  6. **(Optional) information-maximisation step:** a few gradient steps minimising target prediction entropy while keeping batch class-marginal near the source prior **[LIT P50, P28]**.
- **Outputs:** target-adapted reference $\mu_*$, shape transform, and prototypes $\{P_k^*\}$ used for final prediction.
- **Mathematical role:** a label-free fixed-point refinement of (reference geometry, class prototypes) on the manifold; $\alpha$ and $\tau_H$ bound how far adaptation may move from source-validated estimates.
- **Expected benefit:** the direct answer to G5/G8/G9 — exploits the entire unlabelled target recording; manifold-safe by construction; imbalance-robust.

### Component 6 — Depression Prediction Head $h_\omega$

- **Inputs:** adapted descriptor $\Gamma_*(\bar Z_*)$ and prototypes $\{P_k^*\}$.
- **Process:** **LogEig** to tangent space at the (adapted) reference → concatenate with RHOP second-order vector → linear layer → softmax; **temperature scaling** calibrated on source validation folds for clinical ECE control **[LIT P17]**. Prediction can be either prototype-distance (parameter-free, fully transductive) or the linear head (fixed weights) — **ensembled** for robustness **[LIT P47]**.
- **Outputs:** calibrated $p(\text{MDD}\mid \text{subject})$ at subject level (token votes aggregated).
- **Mathematical role:** linearises the manifold decision at the subject-specific reference; prototype-distance branch inherits ETPR adaptation directly.
- **Expected benefit:** calibrated, interpretable (tangent-space ANOVA variable importance **[LIT P9, P47]**), and imbalance-aware (subject-level aggregation + AUC).

## 4.3 Architecture Diagram (ASCII)

```
                              ┌──────────────────────────── TRAINING (source subjects s=1..N-1) ───────────────────────────┐
 raw EEG (subject s)          │                                                                                            │
   │  band-pass {δθαβ}        │   ┌─────────── meta-transduction: hold out one SOURCE subject per episode ──────────┐      │
   ▼                          │   │                  (simulate §4.2-C5 ETPR on it, backprop through it)            │      │
┌───────────────────────────┐ │   │                                                                               │      │
│ C1  FEATURE ENCODER  f_θ   │ │   ▼                                                                               │      │
│  Ledoit-Wolf shrink cov    │ │  ╔═══════════════════════════════════════════════════════════════════════════╗  │      │
│  → CORRELATION  R_{b,w}    │─┼─►║ C2 SUBJECT-INVARIANT MODULE  g_φ                                            ║  │      │
│  (scale-invariant, P3/4/68)│ │  ║  corr-manifold BiMap/ReEig (P68)  +  SPDDSMBN per-subj Fréchet mean (P22)   ║  │      │
│  AIRM cluster 128→K (P8)   │ │  ║  Γ_s(R)=μ_s^{-1/2} R μ_s^{-1/2}   [+ Stiefel reduce (P17), routing (P26)]   ║  │      │
└───────────────────────────┘ │  ╚═══════════════════════════════════════════════════════════════════════════╝  │      │
                              │            │ invariant tokens {Z^t} on Cor_d'                                    │      │
                              │            ▼                                                                     │      │
                              │  ┌───────────────────────────┐      ┌──────────────────────────────────────┐    │      │
                              │  │ C4 MANIFOLD ATTENTION a_ψ │◄────►│ C3 SUBJECT PROTOTYPE MEMORY  𝒫        │    │      │
                              │  │ RHOP 2nd-order pool (P3)   │      │ global P_k^glob, multicentric P_{k,m} │    │      │
                              │  │ GyroAtt (P73) + geo-graph │      │ per-source P_k^(s)  (Fréchet, P28/31) │    │      │
                              │  │ (P25); non-stationarity   │      └──────────────────────────────────────┘    │      │
                              │  └───────────────────────────┘                    │                              │      │
                              │            │ fused Z̄                              │ proto init                  │      │
                              │            ▼                                       ▼                              │      │
                              │  ┌──────────────────────────────────────────────────────────────────────────┐   │      │
                              │  │ LOSSES:  CE  +  supervised manifold-contrastive (P32/98)  +  prototype     │   │      │
                              │  │ compactness  +  CorSW domain-align (P4)  +  [adversarial GRL (P98) opt.]   │   │      │
                              │  │           +  META-TRANSDUCTION loss on held-out source (───────────────────┼───┘      │
                              │  └──────────────────────────────────────────────────────────────────────────┘          │
                              └────────────────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────────────────── TEST (held-out subject *, UNLABELLED) ──────────────────────┐
 raw EEG (subject *)          │   f_θ ─► correlation tokens {R_*^t}  ─►  g_φ (weights frozen)                            │
   │                          │                                          │                                              │
   ▼                          │                                          ▼                                              │
   └──────────────────────────┼──►  ╔══════════════ C5  ETPR  (label-free, iterate n_adapt) ══════════════╗            │
                              │      ║ (1) re-anchor μ_*  from REST segments only (class-agnostic, P52)     ║            │
                              │      ║ (2) tangent 2nd-moment / shape match  (P23, transductive)            ║            │
                              │      ║ (3) pseudo-label  q_k ∝ exp(-δ_R²(Γ_*(R_*),P_k)/τ)   (MDM, P9)       ║            │
                              │      ║ (4) entropy + class-balance filter   (P98 / P28)                     ║            │
                              │      ║ (5) prototype refine  P_k* = Exp_{P_k^glob}(α·Log(target Fréchet))    ║  ◄─ NEW    │
                              │      ║ (6) optional info-max entropy-min steps  (P50/P28)                   ║            │
                              │      ╚══════════════════════════════════════════════════════════════════════╝            │
                              │                                          │ μ_*, shape, {P_k*}                            │
                              │                                          ▼                                              │
                              │   C6 HEAD h_ω:  LogEig@μ_*  ⊕ RHOP  → {linear head ‖ prototype-distance}              │
                              │                 → temperature scale (ECE, P17) → subject-level vote                    │
                              │                                          │                                              │
                              │                                          ▼   calibrated  p(MDD | subject *)            │
                              └────────────────────────────────────────────────────────────────────────────────────────┘
```

## 4.4 Novelty Analysis

**What is genuinely novel [NEW]:**

- **N1 — Meta-trained test-time transduction on the correlation manifold.** The ETPR loop (re-anchor → shape-match → manifold-safe pseudo-label → geodesic prototype refinement) is *simulated on held-out source subjects during training* (episodic meta-transduction), so the encoder is explicitly optimised so that a few *label-free* steps improve the held-out subject. No surveyed paper meta-trains a manifold transductive loop; SPDDSMBN **[LIT P22]** is train-time only, ARK **[LIT P6]** is one-shot and shallow, P23 is shallow/train-time, and meta-learning EEG papers **[LIT P57, P95]** require *labelled* few-shot targets.
- **N2 — Geodesic, negative-transfer-bounded prototype interpolation.** The $\alpha$-controlled $\mathrm{Exp}/\mathrm{Log}$ interpolation between source-validated and target-estimated prototypes is a new, principled control on how far transduction may stray — directly targeting the negative-transfer evidence **[LIT P31, P98]**.
- **N3 — Imbalance-safe manifold transduction.** Combining *class-agnostic baseline-period* recentring **[LIT P52]** with *balanced, entropy-filtered* pseudo-labels **[LIT P28, P98]** *on the correlation manifold* is a new synthesis that none of the source papers performs together (P52 is within-subject MI; P98 is covariance/LEM single-source).
- **N4 — Depression-specific, strict-LODO, geometry-vs-transduction-disentangled benchmark.** The R0–R8 ladder (§3.1.7) is constructed so that the experimental result *attributes* improvement to geometry vs. transduction — answering a question the depression literature (P44, P8) has never posed.

**Why it differs from the closest prior work:**

| Closest prior | What they do | What TRIAD adds |
|---------------|--------------|-----------------|
| SPDDSMBN/TSMNet (P22) | per-domain Fréchet-mean SPD-BN, *source* alignment, covariance | + correlation manifold, + **test-time** target Fréchet/shape adaptation, + **meta-trained** transduction, + depression/LODO |
| SPD-DANN (P98) | adversarial SPD DA, entropy-filtered prototype loss, *train-time*, single-source, covariance/LEM | + multi-source LODO, + **test-time** ETPR loop, + correlation/AIRM, + geodesic α-bounded prototypes, + AUC |
| WDANet (P44) | Euclidean 3-discriminator depression DA | + full **Riemannian geometry** (F1's +9%), + transduction, + strict LODO + AUC |
| EEG-RCformer (P8) | AIRM clustering + Transformer, 5-held-out | + invariance module + **ETPR transduction**, + strict LOSO |
| PT+moments (P23) | transductive PT + tangent moments, *shallow* | + end-to-end deep encoder, + **meta-training**, + prototype refinement + pseudo-labels |

**Why reviewers may find it valuable:**
- It *closes a gap the field's own survey names* **[LIT P69]** (credible, well-motivated).
- It produces a *clean scientific result* (geometry vs. transduction disentangled), not just a leaderboard bump.
- Every component is independently literature-validated, so the risk is low and the ablations are informative.
- It is *clinically meaningful*: calibrated, label-free per-subject deployment, privacy-compatible.

---

# 5. Novel Training Pipeline

TRIAD trains in three stages on the $N\!-\!1$ source subjects of each LODO fold. Let $\mathcal{D}_s=\{(R_i,y_i)\}$ be subject $s$'s correlation tokens; $\Theta=\{\theta,\phi,\psi,\omega\}$ the parameters; $\mathcal{P}$ the prototype memory.

## Stage 1 — Representation Learning (subject-invariant encoder)

**Objective:** learn $f_\theta,g_\phi$ producing class-discriminative, subject-invariant manifold features.

**Losses:**
- *Supervised classification* (subject-level aggregated, class-weighted for imbalance):
$$\mathcal{L}_{\text{CE}}=-\sum_{k}\beta_k\, y_k \log p_k,\qquad \beta_k=\tfrac{N}{N_k}\ \text{(inverse-frequency)}.$$
- *SPD domain-specific normalisation* is structural (SPDDSMBN layers, **[LIT P22]**) — no explicit loss, but it makes the encoder's batch statistics subject-referenced.
- *Manifold geometry regularisation* (keep features well-conditioned): ReEig eigenvalue floor + BWSPD embedding for stable gradients at high $d$ **[LIT P72]**.

**Subject-invariant learning strategy:** momentum Fréchet-mean tracking per source subject (Component 2); whitening interpretation = per-subject tangent re-centring **[LIT P6, P21]**.

## Stage 2 — Cross-Subject Generalisation

**Objective:** force features from *different subjects, same class* together and align source distributions.

- **Domain alignment (distributional):** **CorSW** sliced-Wasserstein DG loss aligning each source subject's correlation distribution to a shared manifold reference (zero inference cost) **[LIT P4]**; optionally a **geodesic adversarial / GRL** term on the SPD features **[LIT P25, P98]** (ablated):
$$\mathcal{L}_{\text{align}}=\sum_{s}\text{CorSW}\big(\{R\}_s,\ \mathcal{N}_{\text{ref}}\big)\;[+\;\lambda_{\text{adv}}\mathcal{L}_{\text{GRL}}].$$
- **Contrastive learning (instance-level):** supervised manifold InfoNCE on tangent vectors $z=\mathrm{Log}_{\mu}(R)$ — pull same-class/different-subject pairs, push different-class **[LIT P32, P38, P98]**:
$$\mathcal{L}_{\text{con}}=-\sum_i \log \frac{\sum_{j\in\text{pos}(i)} e^{\,\mathrm{sim}(z_i,z_j)/\tau}}{\sum_{j\neq i} e^{\,\mathrm{sim}(z_i,z_j)/\tau}}.$$
- **Prototype learning:** compactness around (multicentric) class Fréchet means + margin between classes **[LIT P28, P98, P12]**:
$$\mathcal{L}_{\text{proto}}=\sum_i \delta_R^2\!\big(R_i,P_{y_i}^{\text{near}}\big)\;-\;\gamma \min_{k\neq y_i}\delta_R^2\!\big(R_i,P_{k}\big).$$
- **Source-similarity weighting / selection:** down-weight or drop dissimilar source subjects (tangent-MMD / rotation metric / hyperbolic distance) to curb negative transfer **[LIT P31, P35, P81]**.
- **Regularisation:** Ledoit-Wolf shrinkage (feature side) **[LIT P46]**, dropout in attention **[LIT P8]**, $\alpha$-/temperature priors.

**Stage-2 total:** $\mathcal{L}_2=\mathcal{L}_{\text{CE}}+\lambda_1\mathcal{L}_{\text{align}}+\lambda_2\mathcal{L}_{\text{con}}+\lambda_3\mathcal{L}_{\text{proto}}$.

## Stage 3 — Test-Time Transductive Adaptation (and its meta-training)

This stage has **two faces**: (a) at *test* time it is the ETPR procedure of §4.2-C5; (b) at *train* time we **simulate** it on held-out source subjects so the encoder learns to be transduced.

**Adaptation mechanism (test time, label-free):** run ETPR steps (1)–(6) on the held-out subject's unlabelled tokens; freeze $\theta,\phi,\psi,\omega$ (or allow only BN/affine + temperature to move), update *reference geometry* $\mu_*$, *shape transform*, and *prototypes* $\{P_k^*\}$.

**Pseudo-label generation:** soft manifold assignment to multicentric prototypes (Eq. in C5-3); aggregate token votes to subject-level.

**Confidence filtering:** keep token $t$ iff predictive entropy $H(q_t)<\tau_H$ **and** within per-class balanced quota; this is essential because Fréchet means are non-linearly sensitive to label noise on the manifold **[LIT P98]**.

**Prototype refinement:** geodesic α-interpolation (Eq. in C5-5) with $\alpha$ chosen on source meta-validation.

**Meta-training objective [NEW] (the mechanism that makes Stage 3 learnable):** For each mini-epoch, sample an *episode*: hold out one source subject $s^\star$ as a *pseudo-target*, hide its labels, run ETPR on it, and minimise its (now revealed) supervised loss after adaptation:
$$\mathcal{L}_{\text{meta}}=\mathbb{E}_{s^\star}\Big[\ \mathcal{L}_{\text{CE}}\big(\text{ETPR}_{s^\star}(\Theta),\ y_{s^\star}\big)\ \Big].$$
Back-propagating through (or with a first-order/Reptile approximation of) the ETPR steps shapes $f_\theta,g_\phi$ so that *label-free* adaptation provably helps on unseen subjects — operationalising F4 and directly differentiating TRIAD from P22/P98 (train-time-only) and P57/P95 (label-required).

**Full objective:** $\boxed{\ \mathcal{L}=\mathcal{L}_{\text{CE}}+\lambda_1\mathcal{L}_{\text{align}}+\lambda_2\mathcal{L}_{\text{con}}+\lambda_3\mathcal{L}_{\text{proto}}+\lambda_4\mathcal{L}_{\text{meta}}\ }$

**Optimisation notes:** Riemannian SGD/Adam for BiMap/Stiefel layers (QR retraction) **[LIT P17, P65]**; pseudo-AI metric to keep AIRM operations $O(d^2)$ on 128→K reduced tokens **[LIT P11]**; SPDLearn library for layer implementations **[LIT P75]**.

---

# 6. Finalised LODO Protocol

> Designed to *fix* the evaluation weaknesses L3/W3/G10–G12: strict LOSO, nested validation, full metric panel, subject-level statistics, and apples-to-apples baseline re-runs.

## 6.1 Dataset Splits

**Primary:** MODMA, 53 subjects, 128-channel resting-state, MDD vs. HC **[LIT P8, P44]** (note the documented MDD:HC class imbalance — handled explicitly, not ignored).

- **Outer loop — strict LODO (LOSO):** 53 folds. Fold $i$ trains on 52 subjects, tests on **all** tokens of subject $i$. *Every* token of a subject is either entirely in train or entirely in test — **no subject, session, or window crosses the split** (prevents the leakage that inflates within-subject CV, cf. P16).
- **Inner loop — nested validation for HP/early-stopping:** within each outer fold, **leave-one-source-subject-out** (or 5-fold over the 52 source subjects) to choose hyperparameters $(\lambda_{1..4},\tau,\tau_H,\alpha,K,M,n_{\text{adapt}})$ **[LIT P17 "single HP set", P31 K-selection]**. Hyperparameters are **frozen before** the held-out subject is touched — the test subject is *never* used for selection.
- **Transductive data discipline:** ETPR may use the held-out subject's **unlabelled** EEG only. Labels of the held-out subject are used **exclusively** for final scoring. We report both:
  - *Inductive TRIAD* (ETPR off) and *Transductive TRIAD* (ETPR on) — so the transductive gain is explicit and the comparison to inductive baselines is fair.

**Secondary / external validity:**
- **Cross-dataset:** train on MODMA, test on an independent depression EEG set (e.g., the self-collected cohorts used by WDANet **[LIT P44]**) with tangent-space channel-count-agnostic alignment **[LIT P24]** — leave-one-*dataset*-out.
- **Robustness splits:** imbalance sweep (artificially vary MDD:HC), artifact-stress, and reduced-channel (montage) conditions (§7).

## 6.2 Evaluation Metrics

Computed **per held-out subject**, then aggregated (mean ± CI over 53 subjects). Because the unit of generalisation is the *subject*, the **subject** is the unit of analysis.

| Metric | Why included | Primary? |
|--------|--------------|----------|
| **ROC-AUC** | Threshold-free, imbalance-robust; comparable to P8 (0.7154) | **Primary** |
| **Balanced Accuracy** | Corrects majority-class inflation (L3/G3) | **Primary** |
| **F1 (MDD-positive)** | Clinically relevant (miss cost) | Yes |
| **Precision / Recall (Sensitivity)** | Screening trade-off transparency | Yes |
| **Specificity** | False-positive cost | Yes |
| **MCC** | Single balanced summary under imbalance | Yes |
| **Cohen's κ** | Comparability to clinical EEG work (P3, P22) | Secondary |
| **ECE / reliability** | Calibration for clinical use (P17) | Yes |
| Accuracy | Comparability to P8/P44/P98 only | Reported, not relied upon |

Aggregation: report **subject-level macro-average** (each subject equal weight) as primary; also report **pooled** token-level for reference. Subject-level AUC handled via subject-mean predicted probability + global ROC over 53 subject-scores.

## 6.3 Statistical Validation

- **Confidence intervals:** 95% CIs via **subject-level bootstrap** (resample the 53 subjects with replacement, $B\!=\!10{,}000$) for every metric — the honest CI given $N=53$ (G11).
- **Significance testing:** TRIAD vs. each baseline on paired per-subject scores using **Wilcoxon signed-rank** (non-parametric, 53 paired deltas) and the **corrected resampled $t$-test** (Nadeau–Bengio) to account for train-set overlap across folds. Report effect sizes (Cliff's δ / Cohen's $d$) **[LIT P8 reports effect sizes]**.
- **Multiple-comparison control:** Holm–Bonferroni across the R0–R8 ladder and across ablations.
- **Multiple-run averaging:** ≥5 random seeds; report mean ± std of the subject-macro metrics; the *headline* number is mean-over-seeds of subject-macro-AUC with its bootstrap CI.
- **Permutation sanity:** label-permutation test to confirm above-chance generalisation isn't an artifact (G12).

## 6.4 Fair-Comparison Rules

1. **Identical folds, preprocessing, channels, epoching, shrinkage** for *all* methods (one frozen data pipeline; only the model differs).
2. **Re-run, don't cite.** EEG-RCformer (P8, 5-held-out→re-run as full LOSO), WDANet (P44, +Riemannian-ised variant), SPDDSMBN (P22), SPD-DANN (P98, re-run multi-source LODO) are all executed under our protocol — quoted numbers from papers with *different* splits are not used for claims.
3. **Match the information regime.** Transductive baselines (ARK P6, PT+moments P23, PSDNorm P50) are compared to *Transductive TRIAD*; inductive/DG baselines (CorSW P4, HybridRDG P46) are compared to *Inductive TRIAD*. Cross-regime comparisons are clearly flagged, never hidden.
4. **Equal tuning budget.** Same nested-CV HP-search budget for every method; report the search space.
5. **No test-subject leakage of any kind** — labels of the held-out subject touch only the scoring function.
6. **Report compute** (params, FLOPs, per-fold train + ETPR time) so gains are weighed against cost (P3/P11 emphasise efficiency).

---

# 7. Ablation Study Plan

**Purpose:** attribute performance to each mechanism and validate the central claim that *transduction adds value beyond geometry*. All ablations run under the identical strict-LODO protocol (§6), reporting subject-macro AUC + balanced accuracy with bootstrap CIs.

## 7.1 Core component ablations

| # | Configuration | Hypothesis | Expected outcome | Interpretation if confirmed / if not |
|---|---------------|-----------|------------------|--------------------------------------|
| A1 | **Full TRIAD** | Reference | Best AUC | — |
| A2 | **− ETPR transduction** (inductive only) | Transduction is the largest single contributor (G5/G8) | Largest drop (e.g., −3 to −6 AUC pts) | Confirms novelty value / if small ⇒ transduction not the story → pivot claim |
| A3 | **− meta-training $\mathcal{L}_{\text{meta}}$** (ETPR at test only, not meta-trained) | Meta-training makes the encoder *adaptable* (N1) | Moderate drop; ETPR helps less | Confirms meta-training necessity / if no drop ⇒ ETPR works without it (simpler story) |
| A4 | **− Subject Prototype Memory** (single global mean, no multicentric) | Subtype structure matters (G1) | Drop on heterogeneous subjects; higher variance | Confirms multicentric value / if not ⇒ classes uni-modal |
| A5 | **− Domain alignment** (no SPDDSMBN / CorSW) | Source invariance is necessary (F3) | Clear drop; worse calibration | Confirms alignment / if not ⇒ contrastive+proto suffice |
| A6 | **− Contrastive loss** | Instance-level invariance helps (P32/P38) | Small–moderate drop | Confirms / if not ⇒ prototype loss subsumes it |
| A7 | **− Graph/attention (C4)** (mean-pool instead of RHOP/GyroAtt) | Non-stationarity modelling helps (P13/P3) | Small drop; ~1K fewer params | Confirms marginal value / if not ⇒ drop for simplicity |
| A8 | **Covariance instead of correlation** | Scale-invariance matters cross-subject (F2) | Drop, esp. cross-dataset | Confirms F2 on depression / if not ⇒ amplitude not dominant here |

## 7.2 Encoder / geometry choices

| # | Variant | Hypothesis | Expected | Interpretation |
|---|---------|-----------|----------|----------------|
| B1 | Backbone: SPDNet (P65) vs. Corr-manifold net (P68) vs. EEGNet (Euclidean) | Geometry > Euclidean (F1); correlation ≥ SPD (F2) | Corr ≥ SPD ≫ Euclidean | Replicates P98-Table-10 logic on depression |
| B2 | Metric: AIRM vs. LEM vs. OLM/Off-log (P15) vs. BWSPD (P72) | AIRM/OLM best; BWSPD best-conditioned at 128-ch | OLM closed-form competitive; BWSPD stable | Guides metric choice at high $C$ |
| B3 | Reduction: AIRM-cluster (P8) vs. Stiefel (P17) vs. pseudo-AI (P11) | Reduction needed for 128-ch; minimal acc loss | All ≈ full AIRM, much faster | Justifies tractability choices |
| B4 | + Hyperbolic subject-embedding weighting (P81) | EEG δ-hyperbolic ⇒ better source weighting | Small gain on heterogeneous folds | Tests optional hyperbolic module |

## 7.3 Transduction-mechanism ablations (zoom into ETPR)

| # | Variant | Hypothesis | Expected | Interpretation |
|---|---------|-----------|----------|----------------|
| C1 | Re-anchor: all-trials mean vs. **class-agnostic baseline-period** (P52) | Baseline-period avoids imbalance bias (G3/L5) | Baseline-period higher AUC under imbalance | Validates N3 |
| C2 | Pseudo-labels: **no filter** vs. entropy-filter (P98) vs. entropy+balanced (P28) | Manifold pseudo-labels need filtering | Unfiltered hurts; balanced best | Validates manifold-safety (W5) |
| C3 | Prototype update: hard-replace ($\alpha\!=\!1$) vs. **geodesic α-interp** vs. source-only ($\alpha\!=\!0$) | α-bounded controls negative transfer (N2) | Interior $\alpha^\star$ optimal | Validates N2; reveals $\alpha^\star$ |
| C4 | Shape matching on/off (P23 transductive) | 2nd-order shift persists after mean align (G2) | On > off when source/target variance differ | Validates G2 treatment |
| C5 | $n_{\text{adapt}}$ sweep (0,1,2,5,10) | Few steps suffice; over-adaptation drifts | Plateau then slight decline | Sets safe default; flags drift risk |
| C6 | Info-max step on/off (P50/P28) | Entropy-min adds marginal gain | Small gain; risk of collapse | Decide inclusion |

## 7.4 Robustness / clinical ablations

| # | Stress | Hypothesis | Expected | Interpretation |
|---|--------|-----------|----------|----------------|
| D1 | Imbalance sweep (MDD:HC from 1:1→1:4) | TRIAD's imbalance handling holds AUC | Flatter AUC vs. accuracy-only baselines | Confirms G3/L5 fix |
| D2 | Source-count sweep (10→52 source subjects) | More sources help; saturating | Monotone, saturating | Sample-efficiency profile |
| D3 | Source selection on/off (P31/P35) | Dropping dissimilar sources curbs neg. transfer | Gain when outlier subjects present | Validates G1/S4 |
| D4 | Reduced channels (128→64→19) | Token clustering degrades gracefully | Gentle decline | Deployability on cheap montages |
| D5 | Calibration: temperature scaling on/off (P17) | ECE improves without AUC loss | Lower ECE, same AUC | Clinical readiness |
| D6 | Artifact stress (inject EOG/EMG) | Shrinkage + robust Fréchet mean (P97) resist | Smaller drop than naive cov | Robustness evidence |

## 7.5 Ablation table template (to be filled)

```
Config | AUC (95% CI) | Bal.Acc | F1 | MCC | ECE | Δ vs Full | p (Wilcoxon, Holm) | params | train+ETPR time
-------+--------------+---------+----+-----+-----+-----------+--------------------+--------+-----------------
A1 Full|  .   (. , .) |   .     | .  |  .  |  .  |    —      |        —           |   .    |       .
A2 −ETPR
A3 −meta
A4 −proto
A5 −align
...
R6 EEG-RCformer (re-run)
R4 SPDDSMBN
R7 WDANet-Riem.
R0 EEGNet (floor)
```

---

# 8. Final Research Hypothesis

## 8.1 Main Hypothesis

> **H_main [NEW].** *For leave-one-subject-out depression detection from resting-state EEG, an encoder that (i) represents signals on the scale-invariant correlation manifold and is rendered subject-invariant by momentum Fréchet-mean normalisation, and (ii) is **meta-trained so that a label-free, manifold-safe transductive refinement (ETPR) of its reference geometry and class prototypes against the held-out subject's own unlabelled recording improves its predictions**, will achieve significantly higher subject-level ROC-AUC and balanced accuracy than (a) Euclidean cross-domain depression models (e.g., WDANet), (b) state-of-the-art train-time-only SPD domain-adaptation models (e.g., TSMNet/SPDDSMBN, SPD-DANN), and (c) the same encoder without transduction — with the **transductive component contributing a statistically significant increment beyond geometry alone** on the MODMA benchmark under a strict, nested-LODO protocol.*

The boldface clause is the *falsifiable core*: if ablation **A2** (−ETPR) shows no significant drop, H_main is refuted regardless of leaderboard position.

## 8.2 Secondary Hypotheses

- **H1 — Geometry dominance (F1, replicating P98-Table-10 on depression).** Correlation/SPD-manifold encoders outperform an optimised Euclidean encoder *before* any adaptation; geometry alone explains a large share of the cross-subject gain (test: B1, A8).
- **H2 — Scale-invariance (F2).** Correlation matrices beat covariance for cross-subject and especially cross-dataset depression transfer (test: A8, cross-dataset split).
- **H3 — Meta-transduction necessity (N1).** Meta-training the ETPR loop yields a larger transductive gain than applying ETPR to a non-meta-trained encoder (test: A3) — i.e., the encoder must be *taught to be adapted*.
- **H4 — Negative-transfer bounding (N2/S4).** An interior geodesic-interpolation weight $\alpha^\star\!\in\!(0,1)$ outperforms both hard target replacement and source-only prototypes, and source-similarity selection helps most when subject heterogeneity is high (test: C3, D3).
- **H5 — Imbalance robustness (N3/G3).** Class-agnostic baseline-period recentring + balanced entropy-filtered pseudo-labels preserve AUC under increasing MDD:HC imbalance, where accuracy-only baselines degrade (test: C1, C2, D1).

## 8.3 Expected Contributions

**Methodological contribution.**
The first framework to *unify* training-time source-subject invariance (momentum Fréchet-mean SPD normalisation, **[LIT P22]**) with an explicit, **meta-learned, test-time transductive** refinement loop on the **correlation** manifold, specialised for resting-state depression — directly closing the gap named by the SPD-neuroimaging survey **[LIT P69]** and unaddressed by the strongest depression (Euclidean, **[LIT P44]**) and SPD-UDA (train-time-only, **[LIT P22, P98]**) baselines.

**Algorithmic contribution.**
**ETPR** — a manifold-safe transductive algorithm comprising: class-agnostic baseline-period re-anchoring (imbalance-robust, **[LIT P52]**), transductive tangent-space second-moment matching (**[LIT P23]** made end-to-end), entropy- + class-balance-filtered manifold pseudo-labelling (**[LIT P98, P28]**), and **geodesic $\alpha$-bounded prototype interpolation** (novel negative-transfer control). Plus the **episodic meta-transduction** objective $\mathcal{L}_{\text{meta}}$ that trains the encoder to be adaptable.

**Experimental contribution.**
The first **strict-LODO, AUC/balanced-accuracy, statistically-validated** MODMA depression benchmark in which strong baselines (EEG-RCformer **[LIT P8]**, WDANet-Riemannianised **[LIT P44]**, TSMNet/SPDDSMBN **[LIT P22]**, SPD-DANN **[LIT P98]**) are **re-run under identical folds**, with a ladder (R0–R8) and ablation suite designed to **disentangle the contribution of geometry from that of transduction** — a question the depression literature has not previously isolated.

**Practical contribution.**
A **calibrated** (ECE-controlled, **[LIT P17]**), **privacy-compatible** (source-free-capable, **[LIT P28]**), **montage-flexible** (**[LIT P24]**) depression-screening model that deploys to a new subject with **zero labelled calibration data**, using only their own unlabelled resting-state recording — the realistic clinical screening scenario.

---

## Appendix A — Mapping of borrowed components to source papers (traceability)

| TRIAD element | Primary sources | Secondary / supporting |
|---------------|-----------------|------------------------|
| Correlation-manifold representation | P68, P4, P15 | P3 (quotient-Gaussian) |
| Ledoit-Wolf shrinkage | P46 | P9 (regularisation need) |
| AIRM channel clustering (128→K) | P8 | P16 (graph-BiMap) |
| Stiefel / pseudo-AI reduction | P17, P11 | P26 (routing) |
| SPDDSMBN momentum Fréchet-mean norm | P22 | P66 (ARMAGNAC), P21/P6 (recentring) |
| Manifold contrastive / adversarial | P32, P98 | P25 (geodesic-adv), P38 |
| CorSW DG regulariser | P4 | P33 (R-MMD), P27 (Siegel) |
| Multicentric prototypes | P28 | P12 (LVQ), P47 (ensemble) |
| RHOP / GyroAtt pooling | P3, P73 | P72 (BWSPD) |
| **ETPR re-anchor (baseline-period)** | **P52** | P21, P6 |
| **ETPR shape matching** | **P23** | P27 |
| **ETPR pseudo-label + filter** | **P98, P28** | P50 (info-max) |
| **ETPR geodesic α-prototype** | **[NEW]** | P31/P35 (selection rationale) |
| **Meta-transduction $\mathcal{L}_{\text{meta}}$** | **[NEW]** | P57/P95 (meta-learning contrast) |
| Calibration (temperature/ECE) | P17 | — |
| Strict-LODO + stats protocol | P54, P69 (gold-standard rationale) | P8 (effect sizes) |
| Implementation library | P75 (SPDLearn) | P60 (derivations) |

## Appendix B — Notes, risks, and where consulting the raw PDFs would help

- **Open implementation questions best resolved from the `Other Papers/` PDFs:** exact SPDDSMBN momentum/update equations and convergence conditions (P22), the precise correlation-manifold BiMap/ReEig definitions across the five geometries (P68), EEG-RCformer's exact MODMA split and channel handling for a faithful re-run (P8), and SPD-DANN's GRL schedule + entropy-filter threshold (P98). These do not change the design but matter for reproduction.
- **Primary risk — small N (53).** Mitigations: shrinkage, ~1K-param heads (RHOP), heavy regularisation, augmentation via RGP-VAE/DiFFEoCFM (P14/P49), and subject-level bootstrap CIs that *report* the uncertainty honestly rather than hiding it.
- **Secondary risk — pseudo-label collapse in ETPR.** Mitigations: entropy + class-balance filtering, geodesic $\alpha$-bounding, small $n_{\text{adapt}}$, and the info-max marginal prior (C6).
- **Generalisability beyond MODMA.** The cross-dataset split (train MODMA → test external depression cohort) and channel-count-agnostic alignment (P24) are the external-validity test; positive transfer there is the strongest possible evidence for the framework.
