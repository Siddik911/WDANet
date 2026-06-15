# A Progressive, Ablation-Driven Research Roadmap for Cross-Subject Transductive LODO Depression Detection (TRIAD)

**Status:** Implementation roadmap (companion to `Novel_Cross_Subject_Transductive_LODO_Depression_Detection.md`).
**Purpose:** Re-cast the full TRIAD architecture as an **incremental build-up** in which each stage adds exactly one mechanism, tests it, and earns (or fails to earn) its place before the next is introduced.
**Primary benchmark:** MODMA (53-subject, 128-channel resting-state EEG, MDD vs. HC) under strict Leave-One-Subject-Out (LOSO/LODO).

> **Why this document exists.** The companion design doc specifies the *final* model (6 components, 5 losses, meta-training) up front. That is the right *destination* but the wrong *development order*: a model with many interacting losses cannot be debugged, and it cannot tell you *which* component bought the gain. This roadmap rebuilds the same destination one component at a time, so that every reported improvement is attributable and every stage is independently runnable.

> **Reading convention.** Claims grounded in the surveyed literature are tagged **[LIT Pxx]** (paper numbers match `Paper Summary.md`); newly proposed elements are tagged **[NEW]**.

> **Naming.** TRIAD (this roadmap) ≡ T-CoRRiN in earlier notes — the same framework. The roadmap below treats **TRIAD** as the final assembled model and each stage as a named checkpoint (S0–S8).

---

## Table of Contents

1. [Research Roadmap Overview](#1-research-roadmap-overview)
2. [Stage-by-Stage Plan](#2-stage-by-stage-plan)
3. [Ablation Study Plan](#3-ablation-study-plan)
4. [Evaluation Protocol](#4-evaluation-protocol)
5. [Risk Analysis](#5-risk-analysis)
6. [Final Recommended Architecture](#6-final-recommended-architecture)

---

# 1. Research Roadmap Overview

## 1.1 Rules of engagement (the discipline)

1. **One new mechanism per stage.** Each stage differs from its predecessor by *exactly one* component or loss. Nothing else changes — not the data pipeline, not the folds, not the metric.
2. **The data pipeline is frozen at Stage 0 and never touched again.** Identical preprocessing, epoching, channels, shrinkage, and 53 LOSO folds for *every* stage. Only the model changes. (This is the single most important rule: it is what makes cross-stage deltas interpretable instead of confounded.)
3. **A stage must beat its predecessor on the held-out primary metric to be kept.** If it does not, it is either fixed (one debugging pass) or *abandoned*, and we record *why* — a negative result is a result.
4. **Losses are added late and sparingly.** Stages 0–6 use at most **one** loss (cross-entropy, often none for the shallow stages). The only additional training loss in the entire spine is the meta-transduction loss, added **last** (Stage 8). The 5-loss "kitchen sink" of the companion doc is deliberately deferred to an *optional enrichment pool* (Stage 9), entered only if the spine plateaus.
5. **Every stage is independently runnable and debuggable in isolation.** Shallow stages (S1–S3) need no GPU and run in minutes; they exist partly to validate the harness before any deep learning is involved.
6. **Cheap evaluation during development, full statistical machinery only at the end.** Per-stage gating uses subject-level AUC + balanced accuracy + a confusion matrix + a t-SNE/UMAP sanity check (§4.1). The full bootstrap-CI / Wilcoxon / Holm protocol (§4.2) is run once, on the final spine, for the paper.

## 1.2 The build-up at a glance

| Stage | Checkpoint name | One new mechanism | New loss? | Tests which finding | Maps to companion-doc |
|------:|-----------------|-------------------|-----------|---------------------|------------------------|
| **S0** | Euclidean floor | The frozen LOSO harness + EEGNet baseline | CE | leakage-free pipeline exists | R0 |
| **S1** | Shallow Riemannian | Covariance (SPD) + tangent-space classifier | none / CE | **F1** geometry helps | R1/R2, C1 |
| **S2** | Correlation manifold | Correlation instead of covariance | none / CE | **F2** scale-invariance | A8 |
| **S3** | Per-subject recentring | Fréchet-mean rebias (incl. target's own mean) | none / CE | **F3** + cheapest transduction | C1 of ETPR |
| **S4** | Deep encoder | Learned SPD/corr BiMap–ReEig–LogEig net | CE | representation learning > hand-crafted | B1 |
| **S5** | Subject-invariant norm | SPDDSMBN momentum Fréchet-mean BN | CE | **F5** matches SoA (TSMNet) | R4, A5 |
| **S6** | Test-time recentring | Iterative target reference update at inference | none | **F4** transduction > train-time-only | **A6 − A4 (core)** |
| **S7** | Prototype refinement | ETPR: filtered pseudo-labels + geodesic α-protos | none / +info-max | prototype-level transduction adds | A2 internals, C2/C3 |
| **S8** | Meta-transduction | Episodic L_meta (train the encoder to be adapted) | + L_meta | **N1** "taught to be adapted" | A3 |
| **S9** | Enrichment pool | (optional, one at a time) attention, multicentric protos, CorSW, contrastive, source selection | ≤1 each | marginal gains | A4/A6/A7, B/D |

## 1.3 Dependency and gating diagram

```
            FROZEN DATA PIPELINE  (set once at S0, never changed)
                              │
  S0 EEGNet floor ──► S1 SPD+tangent ──► S2 correlation ──► S3 recentring
   (harness OK?)        (geometry>Eucl?)   (corr>cov?)       (big shallow jump?)
                                                                  │
                                                                  ▼
                            S4 deep corr-encoder  ◄── gate: deep ≥ shallow at N=53?
                                                                  │     │ (no → stay shallow,
                                                                  │     │  regularise, revisit)
                                                                  ▼
                            S5 SPDDSMBN (train-time invariance)  ── gate: matches/≈ SoA?
                                                                  │
                                                                  ▼
              ┌──────────  S6 TEST-TIME RECENTRING  ── ★ CENTRAL GATE ★
              │            (transductive gain over S5 = the paper's thesis)
              │                 │ significant +AUC?           │ null/negative?
              │                 ▼                              ▼
              │            S7 ETPR prototype refine      PIVOT: report "geometry, not
              │                 │ adds over S6?           transduction, is the story"
              │                 ▼                         (still publishable; see §5)
              │            S8 meta-transduction (L_meta)
              │                 │ amplifies S6/S7 gain?
              │                 ▼
              └──────────► S9 enrichment pool (only if spine plateaus below target)
                                │
                                ▼
                       FINAL TRIAD = spine pruned to components that passed their gate
```

## 1.4 The sharper scientific question this ordering exposes **[NEW]**

Per-subject recentring at **S3** already uses the held-out subject's *own unlabelled* mean — that is transduction in its cheapest, one-shot form (cf. ARK's adaptive reference **[LIT P6]**). So by S3 we will likely already know that *using target data helps at all*. The real, sharper question the later stages isolate is therefore:

> **Does iterative, prototype-level, meta-trained transduction (S6–S8) add a statistically significant increment *beyond* simple one-shot recentring (S3) and train-time-only invariance (S5)?**

Framing it this way is more honest and more falsifiable than "does transduction help": S3 nearly guarantees a yes to the weak question, so the burden of novelty falls squarely on S6–S8 clearing the S5 bar. This is the falsifiable core of the whole programme.

---

# 2. Stage-by-Stage Plan

Each stage is specified with the eight required fields: **Goal · Model · New Component · Hypothesis · Training Objective · Evaluation Protocol · Decision Rule · Expected Outcome**, followed by **Literature positioning**.

---

## Stage 0 — Euclidean Floor & the Frozen Harness

**1. Goal.** Stand up the one artifact every later stage depends on: a strict, leakage-free 53-fold LOSO harness with a fixed data pipeline, and establish a Euclidean performance *floor*. This stage is 80% infrastructure, 20% model.

**2. Model.** EEGNet (compact CNN) **[LIT P98 R0-tier]** on band-pass-filtered EEG windows, trained source-only on the 52 training subjects, evaluated on all windows of the held-out subject, aggregated to a subject-level score.

**3. New component introduced.** The **frozen pipeline + LOSO harness itself**: Ledoit-Wolf-ready epoching, band-pass into {δ, θ, α, β}, window/overlap, subject-level aggregation, and the metric/plotting code. (This is the "component" because it is what S1–S9 reuse unchanged.)

**4. Hypothesis.** A Euclidean source-only model will generalise poorly across subjects (subject-independent AUC far below subject-dependent), giving a low but *honest* floor and confirming the harness has no leakage (no suspiciously high AUC à la P16's 96%).

**5. Training objective.** Cross-entropy only, class-weighted by inverse frequency for MDD:HC imbalance.
`L = L_CE`. **(1 loss.)**

**6. Evaluation protocol.** Subject-level ROC-AUC (primary), balanced accuracy, macro-F1, confusion matrix. Plus a **leakage audit**: shuffle-label permutation test (AUC must collapse to ≈0.5) and a check that no subject's windows appear in both train and test of any fold.

**7. Decision rule.**
- If permutation AUC ≈ 0.5 **and** real AUC is a plausible floor (≈0.50–0.62): harness is trustworthy → **proceed to S1**.
- If real AUC is implausibly high (>0.85 sub-independent): **stop — you have leakage.** Fix the split before doing anything else.
- If AUC ≈ 0.5 even with correct labels: check preprocessing/feature scaling before proceeding.

**8. Expected outcome.** A reproducible, audited harness and a documented Euclidean floor number that every subsequent ΔAUC is measured against.

**Literature positioning.** Equivalent to the R0 rung. EEGNet is the standard Euclidean reference in cross-subject EEG; WDANet **[LIT P44]** is a stronger Euclidean depression model but it is a *domain-adaptation* method and belongs at the S5/S6 comparison point, not the floor.

---

## Stage 1 — Shallow Riemannian Baseline (geometry, zero learning)

**1. Goal.** Test the single most strongly corroborated finding in the review — **F1: manifold geometry beats Euclidean for cross-subject EEG** — with *no learnable representation*, so the result is unambiguous.

**2. Model.** Per-window **Ledoit-Wolf shrinkage covariance** **[LIT P46]** → project to the tangent space at the global Fréchet mean → **Tangent-Space LDA (TSLDA)**, with **MDM** (minimum-distance-to-Riemannian-mean) **[LIT P9]** as a parameter-free cross-check.

**3. New component introduced.** The **SPD covariance representation + Riemannian tangent-space classifier** (replaces EEGNet entirely).

**4. Hypothesis.** Geometry alone — with *no* representation learning, *no* alignment — beats the Euclidean floor of S0, because second-order channel structure on the SPD manifold is more subject-transferable than raw Euclidean features (F1; SPD-DANN's ablation: geometry alone 39.8% vs. optimised Euclidean 37.3% **[LIT P98]**; EEG-RCformer +9.05% AUC from AIRM **[LIT P8]**).

**5. Training objective.** None for MDM (parameter-free); closed-form LDA for TSLDA. Optionally CE on tangent features with a linear head. **(0–1 loss.)**

**6. Evaluation protocol.** Same panel as S0. Add a **t-SNE/UMAP of tangent vectors** coloured by subject and by class — the qualitative check that classes are at least partially separable and subjects form visible clusters (motivating later alignment).

**7. Decision rule.**
- If Riemannian AUC > Euclidean floor by a clear margin: **F1 confirmed on depression → commit to the manifold, proceed to S2.**
- If Riemannian ≈ Euclidean: inspect covariance conditioning (128×128 at N=53 is ill-posed — is shrinkage on?) and montage; do *not* proceed until geometry shows an edge, or the whole premise is in doubt.

**8. Expected outcome.** A first, clean quantification of "what geometry buys" on depression LOSO, and a working shallow pipeline that is the substrate for S2–S3.

**Literature positioning.** R1/R2 rungs. MDRM/TSLDA **[LIT P9]** are the foundational shallow geometric baselines; their 2012 Euclidean control (56.2% vs. 70.2%) is the original F1 evidence.

---

## Stage 2 — Correlation Manifold (scale-invariance)

**1. Goal.** Test **F2: scale-invariant correlation matrices outperform raw covariance for cross-subject EEG**, isolating amplitude-scaling (the dominant inter-subject nuisance) from everything else.

**2. Model.** Identical to S1, but each shrinkage covariance `Σ̂` is normalised to a **correlation matrix** `R = D^{-1/2} Σ̂ D^{-1/2}`, `D = diag(Σ̂)`, and processed on the correlation manifold **[LIT P3, P4, P15, P68]**.

**3. New component introduced.** The **covariance → correlation** representation swap (quotients out per-channel amplitude scaling). Exactly one change from S1.

**4. Hypothesis.** Correlation > covariance on cross-subject AUC — most visibly where amplitude differs across subjects/sites — because amplitude scaling is the main inter-subject confound and correlation removes it (F2).

**5. Training objective.** Same as S1 (0–1 loss). **(No new loss.)**

**6. Evaluation protocol.** Same panel. This stage *is* companion-doc ablation **A8** run early; record the covariance-vs-correlation ΔAUC explicitly.

**7. Decision rule.**
- If correlation ≥ covariance: **keep correlation as the canonical representation for all later stages.**
- If covariance wins on MODMA specifically: keep covariance as the spine but **retain correlation for the cross-dataset split** (F2's gain is strongest cross-site), and note the finding.

**8. Expected outcome.** A decision on the canonical second-order representation, with the scale-invariance contribution quantified in isolation — a result the depression literature has never reported (WDANet is Euclidean; EEG-RCformer uses covariance).

**Literature positioning.** Directly tests the field's covariance→correlation migration (P3/P4/P15/P68) on depression for the first time.

---

## Stage 3 — Per-Subject Riemannian Recentring (the universal alignment primitive)

**1. Goal.** Add **F3: per-subject Fréchet-mean recentring**, the alignment operation rediscovered under ≥5 names (rebias, S-RCT, parallel transport, TSA, SPDDSMBN momentum). This is also the **cheapest possible transduction**: the held-out subject is recentred using *its own unlabelled* Fréchet mean.

**2. Model.** S2 + recentring: map every subject's (source *and* held-out) data so its Fréchet mean goes to the identity, `R ↦ M_s^{-1/2} R M_s^{-1/2}`, before the tangent-space classifier **[LIT P6, P21, P23, P24]**.

**3. New component introduced.** **Fréchet-mean recentring (rebias/TSA)** per subject, including the unlabelled held-out subject. One change from S2.

**4. Hypothesis.** Recentring produces the *largest single shallow jump* of the whole spine, because inter-subject shift is to first order a Fréchet-mean translation, and removing it directly attacks the 26-point subject-dependent→independent gap (EEG-RCformer 0.980→0.715 **[LIT P8]**).

**5. Training objective.** Same as S2 (recentring is a structural transform, not a loss). **(No new loss.)**

**6. Evaluation protocol.** Same panel. Re-draw the t-SNE/UMAP **after** recentring: subject clusters should visibly collapse together while class structure persists — the qualitative signature that alignment worked. Also run the **class-agnostic baseline-period vs. all-trials** recentring variant here (companion ablation **C1**, **[LIT P52]**) to pre-empt class-imbalance bias in the mean estimate.

**7. Decision rule.**
- If recentring gives a large, significant jump: **F3 confirmed; this is now your strong shallow baseline and the reference point for "does deep/learned transduction beat one-shot recentring?"** Proceed to S4.
- If the jump is small: subjects are not primarily mean-shifted (unlikely) — inspect whether shrinkage/montage is corrupting the mean estimate before proceeding.
- Adopt class-agnostic baseline-period recentring **iff** it ≥ all-trials recentring under imbalance.

**8. Expected outcome.** A strong, transduction-lite shallow baseline (no deep learning yet). **Critically, this becomes the bar that S6–S8 must clear** to justify the paper's central novelty: iterative/learned transduction must beat one-shot recentring, not merely beat a non-transductive model.

**Literature positioning.** ARK **[LIT P6]** (adaptive reference, one-shot), TSA **[LIT P24]**, PT+moments **[LIT P23]**. S3 reproduces their *mean-alignment* core; S6–S8 extend it from one-shot to iterative/learned.

---

## Stage 4 — Deep Correlation-Manifold Encoder (representation learning)

**1. Goal.** Introduce *learned* manifold representation and ask whether it beats the hand-crafted tangent features of S1–S3 at this small sample size. This is the first GPU stage and the first real overfitting risk gate.

**2. Model.** Replace the fixed tangent projection with a **deep correlation-manifold network**: BiMap → ReEig → (LogEig) blocks **[LIT P65, P68]**, with **AIRM channel clustering 128→K≈5 region tokens** **[LIT P8]** and **Stiefel / pseudo-AI reduction** **[LIT P11, P17]** for 128-channel tractability. Recentring from S3 is retained.

**3. New component introduced.** The **learned deep SPD/correlation encoder** (BiMap/ReEig + clustering/reduction). One conceptual change: hand-crafted → learned features.

**4. Hypothesis.** A learned manifold encoder ≥ hand-crafted tangent features, *provided* dimensionality reduction and shrinkage keep it from overfitting 53 subjects; geometry-preserving depth captures channel structure a single global tangent map cannot.

**5. Training objective.** Cross-entropy with Riemannian SGD/Adam (QR retraction for BiMap/Stiefel) **[LIT P17, P65]**. **(1 loss.)**

**6. Evaluation protocol.** Same panel. **Overfitting watch:** report train vs. held-out AUC gap per fold; plot learning curves; compare params/FLOPs vs. S3. t-SNE of the *learned* embedding vs. the S3 hand-crafted one.

**7. Decision rule.**
- If deep ≥ shallow at acceptable train/test gap and compute: **proceed to S5.**
- If deep < shallow (classic small-N overfit): **do not proceed.** Reduce depth/K/d, increase shrinkage/dropout, add augmentation (RGP-VAE/DiFFEoCFM **[LIT P14, P49]**), or **keep the shallow S3 spine** and route the deep encoder to the optional pool. The roadmap explicitly permits a shallow final model if depth never earns its keep at N=53.

**8. Expected outcome.** A decision on whether representation learning is worth its overfitting risk here — and if so, a learnable encoder ready to receive in-network invariance (S5) and test-time adaptation (S6).

**Literature positioning.** SPDNet **[LIT P65]**, correlation-manifold deep net **[LIT P68]**, EEG-RCformer's clustering **[LIT P8]**, Stiefel-SPD-GCN reduction **[LIT P17]**. B1 ablation (SPDNet vs. corr-net vs. EEGNet) lives here.

---

## Stage 5 — Subject-Invariant Normalisation (SPDDSMBN, train-time UDA)

**1. Goal.** Add the strongest *training-time* invariance mechanism in the literature and reach the **current state of the art for source-aligned SPD UDA** — the bar the novelty must exceed.

**2. Model.** S4 encoder + **SPDDSMBN** layers: per-source-subject momentum Fréchet-mean batch normalisation, transporting each subject's features to a common identity reference, with proven convergence of the running mean to the true Fréchet mean **[LIT P22]**.

**3. New component introduced.** **SPDDSMBN momentum-Fréchet-mean SPD batch norm** (in-network, per-subject). One change from S4.

**4. Hypothesis.** In-network per-subject normalisation > static S3 recentring + plain deep encoder, because momentum tracking removes first-order shift *adaptively during training* and improves calibration (F5); this should match TSMNet-class SoA.

**5. Training objective.** Cross-entropy only — **SPDDSMBN is structural, it adds no loss.** **(1 loss.)**

**6. Evaluation protocol.** Same panel + **ECE/reliability** (SPDDSMBN should improve calibration). This is the **R4 reference point**: re-run TSMNet/SPDDSMBN **[LIT P22]** and EEG-RCformer **[LIT P8]** under the identical frozen folds here for an apples-to-apples SoA comparison (companion §6.4 "re-run, don't cite").

**7. Decision rule.**
- If S5 matches or approaches re-run SoA: **the inductive backbone is SoA-grade → proceed to S6 (the novelty).**
- If S5 < SoA: debug SPDDSMBN (momentum schedule, per-subject batch construction, domain-ID wiring **[LIT P22]**) before adding any transduction — you cannot claim a transductive gain on top of a sub-SoA backbone.

**8. Expected outcome.** A SoA-grade, train-time-only inductive backbone (`Inductive TRIAD`). Everything after this point is the contribution: turning a SoA *inductive* model into a *transductive* one.

**Literature positioning.** TSMNet+SPDDSMBN **[LIT P22]** (NeurIPS 2022, SoA SPD UDA) and SPD-DANN **[LIT P98]** (first manifold-adversarial). Both align *sources only* — S6 is where TRIAD departs.

---

## Stage 6 — Test-Time Transductive Recentring ★ the central gate ★

**1. Goal.** The pivotal stage: turn the train-time-only S5 backbone into a transductive one by re-anchoring its reference geometry to the **held-out subject's own unlabelled recording at inference**, and measure whether this beats SoA. **This single delta (S6 − S5) is the paper's thesis** (companion ablation **A6 − A4**).

**2. Model.** S5 with weights frozen at test time; the SPDDSMBN target reference is **re-estimated from the held-out subject's unlabelled tokens** (target Fréchet mean μ\*, class-agnostic baseline-period estimate **[LIT P52]**), iterated `n_adapt` times. No prototype refinement yet — recentring only.

**3. New component introduced.** The **test-time target-reference update loop** (label-free). One change from S5: the reference geometry is now subject-specific at inference, not fixed from training.

**4. Hypothesis (F4).** Using the freely-available unlabelled held-out recording to re-anchor the geometry **beats train-time-only invariance** — and, the sharper test, beats the one-shot shallow recentring of S3 because it operates on the *learned, invariant* features rather than raw tangent vectors.

**5. Training objective.** None — this is inference-time, label-free adaptation. The backbone is unchanged. **(0 new losses.)**

**6. Evaluation protocol.** Report **both** `Inductive TRIAD` (S5, loop off) and `Transductive TRIAD` (S6, loop on) on identical folds, so the transductive increment is explicit and fairly compared to inductive baselines (companion §6.4 rule 3). Sweep `n_adapt ∈ {0,1,2,5,10}` (companion **C5**) to find the safe plateau and flag over-adaptation drift.

**7. Decision rule — the make-or-break gate.**
- If S6 − S5 is a **positive, significant** subject-level AUC increment (and S6 > S3): **the central hypothesis is supported → proceed to S7 to strengthen it.**
- If S6 ≈ S5 (no transductive gain): **this is the falsifiable core failing.** Before pivoting, verify μ\* estimation and `n_adapt`. If still null, **pivot the paper's claim to "geometry, not transduction, is the cross-subject story on depression"** — which, given S1–S5, is itself a clean, publishable result (see §5, R5).
- If S6 < S5 (negative transfer): the target reference is being corrupted (imbalance bias, artifacts) — fix recentring (baseline-period, robust Fréchet mean **[LIT P97]**) before continuing.

**8. Expected outcome.** The first direct evidence for or against the framework's reason to exist. A positive result here is the headline; a null result redirects the paper but does not waste it.

**Literature positioning.** No surveyed paper does this for depression+SPD+iterative. PSDNorm **[LIT P50]** and SPDDSMBN target statistics **[LIT P22]** do test-time normalisation in adjacent settings; the SPD survey **[LIT P69]** names test-subject Fréchet adaptation as the open direction. S6 is that direction, minimally instantiated.

---

## Stage 7 — ETPR Prototype Refinement (transduction beyond recentring)

**1. Goal.** Test whether refining the *class prototypes* (not just the reference geometry) with the held-out subject's unlabelled data adds a further, manifold-safe increment — assembling the ETPR core piece by piece.

**2. Model.** S6 + a **prototype memory** and the ETPR refinement steps: soft manifold pseudo-labelling to class prototypes (MDM-style **[LIT P9]**), entropy + class-balance filtering of pseudo-labels **[LIT P98, P28]**, and **geodesic α-bounded interpolation** between source-validated and target-estimated prototypes **[NEW]**. Optional transductive tangent second-moment (shape) matching **[LIT P23]**.

**3. New component introduced.** **ETPR prototype refinement** (filtered pseudo-labels + geodesic α-interpolated prototypes). Added on top of S6's reference update.

**4. Hypothesis.** Refining prototypes adds a measurable increment over reference-only recentring, **provided** pseudo-labels are filtered (a wrong pseudo-label distorts a Fréchet mean non-linearly **[LIT P98]**) and α ∈ (0,1) bounds how far adaptation strays (negative-transfer control, **N2**).

**5. Training objective.** Still primarily label-free at test time; **optionally one** info-maximisation entropy-min step **[LIT P50, P28]**. **(0–1 new losses — added only if C6 shows it helps.)**

**6. Evaluation protocol.** Same panel. Run the ETPR-internal ablations *here, locally*, since this is the only place they matter: pseudo-label filtering (**C2**: none vs. entropy vs. entropy+balanced), prototype update (**C3**: hard-replace α=1 vs. interior α vs. source-only α=0), shape-matching on/off (**C4**), info-max on/off (**C6**).

**7. Decision rule.**
- If prototype refinement adds over S6 with stable behaviour: **keep it; record the optimal α\* and filter.**
- If it destabilises (pseudo-label collapse, AUC variance up): **keep S6 reference-only recentring as the transductive mechanism and drop prototype refinement** — the simpler transductive story still stands.
- Adopt info-max/shape-matching **only** if C6/C4 show a positive, stable increment.

**8. Expected outcome.** A validated, minimal ETPR (only the sub-steps that earned their place), with α\* and the filter rule fixed — or a principled decision that reference-only transduction (S6) is the right stopping point.

**Literature positioning.** Entropy-filtered SPD prototype loss **[LIT P98]**, multicentric/balanced pseudo-labels **[LIT P28]**, PT+moments shape **[LIT P23]**. The geodesic α-bounded interpolation is the **[NEW]** negative-transfer control (companion N2).

---

## Stage 8 — Episodic Meta-Transduction (train the encoder to be adapted)

**1. Goal.** Test the framework's headline novelty (**N1**): that the encoder must be *meta-trained* so that label-free test-time adaptation provably helps — i.e., "taught to be adapted." Added **last**, because it only makes sense once a working ETPR loop (S6/S7) exists to simulate.

**2. Model.** S7 + **episodic meta-transduction**: in each training episode, hold out one *source* subject as a pseudo-target, hide its labels, run ETPR on it, then minimise its (revealed) supervised loss after adaptation — back-propagating through the ETPR steps (or a first-order/Reptile approximation) **[NEW]**.

**3. New component introduced.** The **meta-transduction loss** `L_meta` and the episodic training loop. This is the *only* additional training loss in the entire spine.

**4. Hypothesis (N1).** Meta-training yields a *larger* transductive gain than applying ETPR to a non-meta-trained encoder (companion **A3**): the encoder learns representations on which a few unlabelled steps measurably help, differentiating TRIAD from train-time-only UDA **[LIT P22, P98]** and from label-requiring meta-learning **[LIT P57, P95]**.

**5. Training objective.** `L = L_CE + λ_meta · L_meta`. **(2 losses — the maximum the spine ever reaches.)** Begin with first-order/Reptile to avoid second-order instability; switch to full backprop-through-ETPR only if first-order plateaus.

**6. Evaluation protocol.** Same panel. The decisive comparison is **A3**: `{ETPR + meta}` vs. `{ETPR, no meta}` (= S7) — the *increase in the transductive gain* attributable to meta-training, not just absolute AUC.

**7. Decision rule.**
- If meta-training increases the transductive gain: **N1 confirmed — this is the strongest, most defensible novelty; include it.**
- If meta-training adds nothing over S7: **drop it.** "ETPR helps without meta-training" is a *simpler, still-novel* story; do not carry a complex, hard-to-tune meta-loop that buys nothing. (This decision rule is itself the antidote to the original over-engineering.)
- If meta-training is unstable: stay first-order, reduce λ_meta and the inner-loop step count, or defer.

**8. Expected outcome.** A final verdict on whether the spine ends at S7 (transductive but not meta-trained) or S8 (meta-transductive). Either is a complete paper; S8 is the more novel if it earns its place.

**Literature positioning.** Meta-learning EEG **[LIT P57, P95]** requires labelled targets; SPDDSMBN/SPD-DANN **[LIT P22, P98]** are train-time only. Meta-training a *label-free manifold transductive loop* is unclaimed in the review — the central **[NEW]** contribution.

---

## Stage 9 — Optional Enrichment Pool (only if the spine plateaus below target)

Entered **only** if the S6/S7/S8 spine has not reached the target (beat re-run EEG-RCformer 0.7154 under strict LOSO, companion H_main). Add **one** at a time, keep only on a significant increment:

| Add (one at a time) | Mechanism | Companion ref | Keep iff |
|---|---|---|---|
| Manifold attention / RHOP pool | non-stationarity across windows **[LIT P3, P73, P25]** | A7 | +AUC over mean-pool at ~1K params |
| Multicentric prototypes | MDD-subtype structure **[LIT P28]** | A4 | helps on heterogeneous folds |
| CorSW domain-alignment loss | sliced-Wasserstein DG, 0 inference cost **[LIT P4]** | A5 | +AUC, esp. cross-dataset |
| Supervised manifold contrastive | same-class/diff-subject pulling **[LIT P32, P98]** | A6 | +AUC and proto loss doesn't subsume it |
| Source-subject selection/weighting | curb negative transfer **[LIT P31, P35, P81]** | D3 | +AUC when outlier subjects present |
| Hyperbolic subject weighting | δ-hyperbolic source weighting **[LIT P81]** | B4 | small gain on heterogeneous folds |

Each carries **at most one** new loss, added in isolation, gated as above. This is where the companion doc's "5 losses" may *partially* reappear — but only the subset that demonstrably earns its place, and never all at once.

---

# 3. Ablation Study Plan

## 3.1 The roadmap *is* the spine ablation

Because each stage changes exactly one thing against a frozen pipeline, **the sequence of checkpoints S0→S8 is itself the core ablation table** — no separate ablation run is needed for the spine. Each row is a stage delta:

```
Δ(S1−S0) = value of GEOMETRY            (F1; companion B1/A-geom)
Δ(S2−S1) = value of SCALE-INVARIANCE    (F2; companion A8)
Δ(S3−S2) = value of RECENTRING          (F3; one-shot transduction)
Δ(S4−S3) = value of LEARNED REPRESENTATION
Δ(S5−S4) = value of TRAIN-TIME INVARIANCE (SPDDSMBN; companion A5)
Δ(S6−S5) = value of TEST-TIME TRANSDUCTION  ★ headline = companion A6−A4 ★
Δ(S7−S6) = value of PROTOTYPE REFINEMENT
Δ(S8−S7) = value of META-TRAINING          (companion A3, hypothesis N1)
```

A reviewer reading this column learns *exactly* how much each idea contributed — the attribution the monolithic TRIAD could never provide.

## 3.2 Local (zoom-in) ablations — run only inside the owning stage

Sub-mechanism ablations are run **at the stage that introduces the mechanism**, not globally, so they stay cheap and interpretable:

- **At S3:** C1 — all-trials vs. class-agnostic baseline-period recentring (imbalance bias).
- **At S4:** B1 — SPDNet vs. correlation-net vs. EEGNet backbone; B3 — reduction method (AIRM-cluster vs. Stiefel vs. pseudo-AI).
- **At S6:** C5 — `n_adapt` sweep {0,1,2,5,10}.
- **At S7:** C2 — pseudo-label filter (none/entropy/entropy+balanced); C3 — prototype update (α=1 / interior α / α=0); C4 — shape-matching; C6 — info-max step.
- **At S8:** A3 — meta vs. no-meta transductive gain.

## 3.3 Robustness suite — run once, on the final spine only

Deferred to the end (these stress-test the *finished* model, they don't gate development): imbalance sweep MDD:HC 1:1→1:4 (**D1**), source-count sweep 10→52 (**D2**), source-selection on/off (**D3**), reduced channels 128→64→19 (**D4**), temperature-scaling calibration (**D5**), artifact stress (**D6**).

## 3.4 Baseline ladder (re-run, don't cite)

Run under the identical frozen folds, slotted next to their natural stage: EEGNet (S0/R0), MDRM/TSLDA (S1/R1-2), SPDNet (S4/R3), TSMNet-SPDDSMBN (S5/R4), SPD-DANN multi-source (R5), EEG-RCformer (S5-compare/R6, the number to beat), WDANet ± Riemannian (S5-compare/R7). Quoted numbers from papers with different splits are **never** used for claims (companion §6.4).

---

# 4. Evaluation Protocol

Two tiers, by design: a **cheap tier for per-stage gating** (so development stays fast and debuggable) and the **full statistical tier run once** on the final spine (for the paper). This split is itself a deliberate anti-over-engineering choice.

## 4.1 Development tier (every stage, every fold)

Computed **per held-out subject**, aggregated as a subject-level macro-average (each subject equal weight — the subject is the unit of generalisation):

| Metric | Role | Why |
|---|---|---|
| **Subject-level ROC-AUC** | **Primary** | Threshold-free, imbalance-robust; directly comparable to EEG-RCformer 0.7154 **[LIT P8]**. (AUC, not accuracy, is the correct primary metric under MDD:HC imbalance.) |
| **Balanced accuracy** | **Primary** | Corrects majority-class inflation. |
| **Macro-F1** | Standard | Requested standard metric; clinically relevant (miss cost). |
| **Confusion matrix** | Standard | Requested; shows the error structure per fold. |
| **t-SNE / UMAP** | Qualitative | Requested; subject/class embedding sanity check — especially before vs. after recentring (S3) and learned vs. hand-crafted (S4). |

No bootstrap, no significance tests at this tier — just the point estimates and the embedding plot, so a stage can be accepted/rejected in one run. This keeps the standard, widely-reported metric set (AUC, balanced acc, macro-F1, confusion matrix, t-SNE) and nothing heavier.

## 4.2 Publication tier (final spine only, once)

The full companion-doc §6 machinery, run after the spine is frozen: subject-level **bootstrap 95% CIs** (resample 53 subjects, B=10,000); **Wilcoxon signed-rank** + Nadeau–Bengio corrected resampled t-test on paired per-subject deltas vs. each baseline; **Holm–Bonferroni** across the ladder and ablations; **≥5 seeds**; **label-permutation** sanity test; **ECE/reliability** and Cohen's κ for clinical comparability. Report compute (params, FLOPs, per-fold train + adaptation time).

## 4.3 Fair-comparison invariants (all tiers)

Identical folds/preprocessing/channels/epoching/shrinkage for every model; transductive methods compared to `Transductive TRIAD`, inductive/DG methods to `Inductive TRIAD`, cross-regime comparisons flagged not hidden; equal HP-tuning budget; **no held-out-subject label ever touches anything but the final scoring function** (companion §6.4).

---

# 5. Risk Analysis

| # | Risk | Where it bites | Early-warning signal | Mitigation | If it fires |
|---|------|----------------|----------------------|------------|-------------|
| **K1** | **Data leakage inflates everything** | S0 | Permutation AUC ≫ 0.5; sub-independent AUC implausibly high | Strict subject-disjoint folds; permutation test gate at S0 | Halt all stages until fixed — every later number is void otherwise |
| **K2** | **Small N=53 overfitting** | S4 onward | Train≫held-out AUC gap; high seed variance | Shrinkage, ~1K-param heads, dimensionality reduction, augmentation (RGP-VAE/DiFFEoCFM **[LIT P14, P49]**), heavy regularisation | Keep the shallow S3 spine; route depth to optional pool — a shallow final model is allowed |
| **K3** | **No transductive gain (S6 = S5)** | S6 ★ | Δ(S6−S5) ≈ 0 across seeds | Verify μ\* estimation, `n_adapt`, baseline-period recentring | **Pivot** to the "geometry is the story" paper (S1–S5 already give a clean, publishable result) — the staged design makes this a graceful fallback, not a dead end |
| **K4** | **Pseudo-label collapse in ETPR** | S7 | AUC variance spikes; prototypes drift to one class | Entropy + class-balance filter **[LIT P98, P28]**, geodesic α-bounding, small `n_adapt`, info-max marginal prior | Fall back to reference-only transduction (S6) — still a complete contribution |
| **K5** | **Negative transfer from heterogeneous subjects** | S5/S6 | Some folds drop below S3 | α-bounded prototypes (**N2**), source selection/weighting **[LIT P31, P35]** | Enable source selection (S9-D3); report per-fold breakdown |
| **K6** | **Meta-training instability / no benefit** | S8 | L_meta diverges, or A3 shows no gain | First-order/Reptile first; reduce λ_meta and inner steps | Drop meta-training; ship S7 (simpler, still novel) |
| **K7** | **Class-imbalance corrupts the alignment primitive** | S3/S6 | AUC swings with MDD:HC ratio | Class-agnostic baseline-period recentring **[LIT P52]**; imbalanced-aware metrics throughout | Make baseline-period recentring the default (validated at C1/D1) |
| **K8** | **Ill-conditioned 128×128 covariance at N=53** | S1 onward | Numerical instability; geometry shows no edge at S1 | Ledoit-Wolf shrinkage **[LIT P46]**, AIRM clustering 128→K, pseudo-AI O(d²) ops **[LIT P11]** | Increase shrinkage; reduce K/d before proceeding |
| **K9** | **Thin depression benchmark (MODMA + few)** | external validity | Gains don't survive cross-dataset | Cross-dataset split (train MODMA → external cohort) with channel-agnostic TSA **[LIT P24]** | Report MODMA-only honestly; frame cross-dataset as the external-validity test |
| **K10** | **Over-engineering relapse** | S7–S9 | Many losses creeping back, tuning explodes | One mechanism + ≤1 loss per stage; gate hard; drop anything that doesn't pass | Enforce the §1.1 rules; prefer the simpler passing spine |

> **The roadmap's structural mitigation:** because every stage is a single, gated, reversible step against a frozen pipeline, *any* risk that fires localises to one stage and the spine can fall back to the last passing checkpoint. The monolithic TRIAD has no such fallback — that is precisely the failure mode this roadmap is built to prevent.

---

# 6. Final Recommended Architecture

**Do not build this until the gates pass.** The final model is *not* predetermined — it is whatever subset of components cleared its decision rule. The roadmap yields one of a small number of principled end-states:

```
                         which gates passed?
S6 (test-time transduction) significant?
        │ NO ──────────────► FINAL = S5  : SoA inductive SPD model on the
        │                                  correlation manifold. Paper claim:
        │                                  "geometry + train-time invariance is
        │                                  the cross-subject depression story."
        │ YES
        ▼
S7 (prototype refinement) adds & stable?
        │ NO ──────────────► FINAL = S6  : Transductive recentring TRIAD.
        │                                  "Test-time reference adaptation beats
        │                                  train-time-only UDA." (simplest novel)
        │ YES
        ▼
S8 (meta-transduction) amplifies gain?
        │ NO ──────────────► FINAL = S7  : ETPR-TRIAD without meta-training.
        │                                  "Manifold-safe transductive prototype
        │                                  refinement helps." (strong, simpler)
        │ YES
        ▼
   FINAL = S8 (+ any S9 components that passed)  : full meta-transductive TRIAD.
```

## 6.1 The maximal end-state (only if every gate passes)

If S6, S7, S8 all clear their gates, the recommended architecture is the **pruned TRIAD spine** — the companion doc's design *minus any component that did not earn its place*:

```
raw EEG ─► band-pass {δθαβ} ─► Ledoit-Wolf shrinkage cov ─► CORRELATION R
        ─► AIRM cluster 128→K + Stiefel/pseudo-AI reduce
        ─► deep corr-manifold BiMap/ReEig encoder
        ─► SPDDSMBN per-subject momentum Fréchet-mean norm        ── (S5)
        ─► [optional, only if passed: manifold attention / multicentric protos / CorSW / contrastive / source selection]   ── (S9)
        ─► LogEig head + temperature scaling
   TEST TIME (label-free, per held-out subject):
        ─► ETPR: target reference μ* (baseline-period)            ── (S6)
                 + filtered pseudo-labels + geodesic α-protos     ── (S7)
   TRAINING: L = L_CE + λ_meta·L_meta  (+ ≤1 enrichment loss each, only if passed)  ── (S8/S9)
```

## 6.2 Loss budget — the discipline made concrete

| Spine through | Training losses | Test-time |
|---|---|---|
| S0–S3 | 0–1 (CE or closed-form) | recentring is label-free |
| S4–S6 | 1 (CE) | S6 transduction label-free |
| S7 | 1 (+ ≤1 optional info-max) | ETPR label-free |
| S8 (maximal spine) | **2** (CE + L_meta) | ETPR label-free |
| S9 enrichments | +1 each, only the passing subset | — |

The spine **never** trains with 5 losses; the companion doc's full loss set is reached *only* if every optional enrichment independently passes — which the roadmap predicts is unlikely at N=53. **The expected final model is leaner than the companion TRIAD**, and you will be able to name exactly why each surviving component is there.

## 6.3 Headline claim the final model supports

Whichever end-state is reached, the staged evidence supports a precise, attributable claim of the form: *"On strict-LOSO MODMA depression detection, geometry contributes Δ(S1−S0)+Δ(S2−S1)+Δ(S3−S2) AUC, train-time invariance adds Δ(S5−S4), and test-time (meta-)transduction adds Δ(S6−S5)[+Δ(S7−S6)+Δ(S8−S7)] beyond SoA — with the transductive increment being [significant/not] under a nested-LOSO protocol with subject-level bootstrap CIs."* That decomposition — not a single leaderboard number — is the contribution the incremental method makes possible.

---

## Appendix — Stage → Companion-Doc → Literature crosswalk

| Stage | Companion §/component | Companion ablation | Key papers |
|---|---|---|---|
| S0 | R0 ladder | — | EEGNet; P44 (Euclidean DA, compared later) |
| S1 | R1/R2, C1-arch | B1, A-geom | P9 (MDRM/TSLDA), P98 (geom-alone ablation), P8 (+9% AIRM) |
| S2 | §2.2, C1-feat | A8 | P3, P4, P15, P68 (correlation manifold) |
| S3 | C5 step (1), §2.3-A | C1 | P6 (ARK), P21, P23, P24 (recentring/TSA), P52 (baseline-period) |
| S4 | C1+C2 encoder | B1, B3 | P65 (SPDNet), P68 (corr-net), P8 (clustering), P11/P17 (reduction) |
| S5 | C2 (SPDDSMBN) | A5 | P22 (TSMNet/SPDDSMBN), P98 (SPD-DANN) |
| S6 | C5 steps (1)-(2) | **A6−A4** | P50 (PSDNorm), P22 (target stats), P69 (named gap) |
| S7 | C5 steps (3)-(6), C3,C5 | A2, C2, C3, C4, C6 | P9, P98, P28, P23 (ETPR ingredients); **[NEW]** α-protos |
| S8 | §5 Stage 3 L_meta | A3 | **[NEW]** meta-transduction; contrast P57/P95 (label-required) |
| S9 | C3,C4 + §5 Stage 2 | A4, A6, A7, B4, D3 | P3/P73/P25, P28, P4, P32, P31/P35/P81 |
