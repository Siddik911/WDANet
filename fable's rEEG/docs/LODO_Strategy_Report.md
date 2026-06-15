# LODO Strategy Report (Revised)
## Leave‑One‑Dataset‑Out Depression Detection: Best Approaches from the Literature

**Setup**: MODMA (53 subj, 128‑ch, 250 Hz) | Mumtaz (58 subj, 19‑ch, 256 Hz) | OpenNeuro ds003478 (121 subj, 64‑ch, 500 Hz)
**Protocol**: Train on 2 datasets → test on 1, strict subject‑independent; all resampled to 125 Hz
**Current model**: `nr` K=3 (within‑dataset AUC ≈ 0.72 on each dataset; own result, not a literature value) locked as final within‑dataset model

> **Revision note (2026‑06‑13).** This report was audited claim‑by‑claim against the 96 papers in `Other Papers/` (summarized as P1–P98 in `Paper Summary.md`); see `Verification_Report.md` and `Synthesis_Matrix.md`. Corrections applied in this version: (1) WDANet's published numbers are **accuracy/F1, not AUC** — they are no longer presented as an "AUC baseline"; (2) the §1.2 "survey quote" was **not verbatim in Paper 54** and has been replaced with a sourced paraphrase; (3) the SPD‑DANN ablation delta was misattributed and is now stated correctly; (4) Papers 21 and 98 **have no PDF in this corpus** and are flagged; (5) projected AUCs are labeled as **author hypotheses, not literature values**; (6) "LODO" (dataset‑level) is now disambiguated from the "LOSO" (subject‑level) evidence most papers actually provide.

> **Terminology (read first).** In this report **LODO = Leave‑One‑*Dataset*‑Out** (generalize to a held‑out *dataset*). Almost every cited paper instead evaluates **LOSO = Leave‑One‑*Subject*‑Out** (generalize to a held‑out *subject* within one dataset). LODO is the **stricter** test. When a paper's result is imported below, the protocol it actually used is named explicitly, because subject‑level evidence is an *optimistic proxy* for dataset‑level generalization.

> **Maturity caveat.** Much of the strongest‑sounding 2026 evidence (P3, P4, P44, P45, P46, P54, P68, P69, P98, P21) is **pre‑print or under review**; P69's "SOTA" endorsement of P22 is by the **same research group**. Treat single‑paper SOTA claims as provisional.

---

## 1. The Core LODO Problem

### 1.1 What Makes LODO Hard Here

| Challenge | Description | Severity |
|-----------|-------------|----------|
| **Channel mismatch** | 19 vs 64 vs 128 channels — SPD/tangent methods need a common matrix dimension, so no model transfers across montages without harmonization | Blocker |
| **Double dissociation** | MODMA signal ≈ functional connectivity; Mumtaz/OpenNeuro signal ≈ spectral 1/f + band power (own observation — see §4) | High |
| **Distribution shift** | Different labs, equipment, populations → covariance‑manifold Fréchet means are far apart (premise of P6, P21, P22, P23) | High |
| **Dataset‑label shift** | BDI cut‑offs differ; Mumtaz filename labels vs MODMA xlsx | Medium |
| **Class imbalance** | MODMA 24 MDD/29 HC; Mumtaz 30 MDD/28 HC; ON 46 MDD/75 HC | Medium |
| **Small N / few folds** | 3 datasets ⇒ only 3 LODO folds, each a high‑variance point estimate (§"Statistical considerations") | High |

### 1.2 What the Survey Literature Actually Says (corrected)

The cross‑subject DL survey (**P54**) does **not** contain the words "transductive" or "gold standard" (verified against the PDF). What it *does* establish, and what is genuinely usable here:

- **Subject‑dependent or mixed‑subject evaluation inflates accuracy** — P54 reports models dropping from **>95 % to ~60 %** when moved to unseen subjects, attributing it to subject‑identity **data leakage**. Therefore only out‑of‑distribution evaluation is honest, and **cross‑dataset LODO is the strongest such test**.
- **Subject‑independent protocols (LOSO, group‑k‑fold, hold‑out) are required**; P54 lists LOSO as one valid implementation, not as a uniquely "gold" one.
- The **transductive thesis** of this report — *use the unlabeled target dataset's EEG to align* — is supported not by P54's wording but by the **methods** papers that do exactly this and win: P6 (adaptive reference), P22 (SPD‑BN), P23 (parallel transport), P50 (test‑time OT), P98 (manifold pseudo‑labels). That is the correct evidentiary basis and is used throughout below.

Your `nr` model (Riemannian correlation, scale‑invariant) is already in the right family. The open question is how to **bridge datasets**, evaluated honestly.

---

## 2. Channel Harmonization — The Mandatory First Step

Every SPD/tangent‑space alignment method below operates on matrices of a **fixed dimension**; you therefore need a shared feature space *before* any LODO method applies. Three options, ranked by difficulty:

### Option A: 19‑channel 10‑20 harmonization ⭐ Recommended
Project all datasets onto the 19‑channel 10‑20 montage intersection.
- **Mumtaz**: already 19‑ch 10‑20 — no change.
- **OpenNeuro**: 64‑ch → select the 19 named 10‑20 channels.
- **MODMA**: 128‑ch GSN → map to 10‑20 using the electrode‑position table.

**Why best**: preserves spatial structure, smallest common set, yields identical 19×19 correlation matrices across datasets so `nr` runs directly.
**Implementation note**: the SPARC‑Net `prepare_xdataset.py` mapping can be reused — this is an *external code reference*, not one of the 98 papers.

### Option B: Spectral features only (channel‑count independent)
Use `nrde`/`spec` band‑power features and dataset‑level spectral profiles (average across channels per band) → a low‑dimensional dataset‑agnostic feature.
**Why limited**: destroys spatial connectivity — loses the `nr` signal on MODMA. (Design reasoning + own results, not a literature finding.)

### Option C: Shared latent space via domain adaptation
Per‑dataset encoder mapping different channel counts to a common d‑dim embedding, then distribution alignment — the TSMNet/SPDDSMBN path (**P22**). More model complexity.

**Bottom line**: Do Option A first; it is the only approach that lets you directly reuse `nr`.

---

## 3. Best Approaches from the 96 Papers

### 3.1 Tier 1: Must‑Consider

#### P44 — WDANet (Relevance 10/10) — *primary architectural competitor, not an AUC baseline*
**"Wasserstein Distribution Inspired Dynamic Adversarial Network for Cross‑Domain Depression Recognition"** — *IEEE Trans. Affective Computing 2026*

The only paper in the corpus that directly addresses **cross‑dataset** depression and uses MODMA. Three discriminators:
1. **Global** — aligns marginal distributions (dataset ↔ dataset);
2. **Local** — aligns class‑conditional distributions (MDD features across datasets);
3. **Wasserstein** — models the *intermediate* distribution‑transformation states, not just endpoints.

A **dynamic adversarial factor** (from the A‑distance between discriminator losses) auto‑balances global vs local each iteration.

**Reported performance (corrected):** WDANet reports **classification accuracy** of **83.3 / 75.5 / 73.9 / 76.0 / 70.9 %** across its cross‑subject and cross‑dataset cases, and cross‑dataset **F1 = 77.8 / 87.3 %**. **It reports no AUC** (verified against the 14‑page PDF). The 83.3 % is *within‑dataset cross‑subject* accuracy, not a MODMA AUC and not strict LODO. **Do not compare it directly to `nr`'s 0.7227 AUC** — accuracy and AUC are not interchangeable, and the datasets differ.

**What to borrow**: the 3‑discriminator architecture + dynamic weighting.
**What to improve**: WDANet uses Euclidean CNN features. Replacing them with Riemannian correlation features + SPDDSMBN (P22), and adding a *test‑time* transductive step (WDANet adapts at train time only), is the contribution.

---

#### P22 — TSMNet + SPDDSMBN (Relevance 9/10)
**"SPD Domain‑Specific Batch Normalization …"** — *NeurIPS 2022*

SPDDSMBN tracks each domain's Fréchet mean on the SPD manifold via an **exponential moving average during training**; features are transported to the tangent space at that mean → domain shift is absorbed geometrically. TSMNet = temporal filter → spatial filter → SPDDSMBN → LogEig → linear. P69's review calls SPDDSMBN current SOTA for cross‑subject EEG UDA (note: P69 is by the same group, arXiv).

**LODO application**:
```
Training: source datasets 1+2 → SPDDSMBN tracks their separate Fréchet means
          → transports each to identity tangent space → domain-invariant features
Test:     target dataset 3 → one forward pass to initialize its Fréchet mean → classify
```
**Transductive extension (your novel contribution)**: at test time, feed unlabeled target trials to *update* the target Fréchet mean. P22 does **not** do this (train‑time only); P69 lists exactly this as an open gap.

---

#### P98 — SPD‑DANN (Relevance 8/10) — 🔎 *no PDF in this corpus; numbers are summary‑sourced*
**"SPD‑manifold unsupervised domain adaptation for cross‑subject MI EEG"** — *Neural Networks 2026*

First adversarial training (GRL) done entirely on the SPD manifold, plus:
1. **SPD domain alignment loss** L_M — log‑Euclidean distance between source/target mean SPD features;
2. **SPD class‑prototype pair loss** L_P — triplet pulling same‑class source/target prototypes together, using **entropy‑filtered pseudo‑labels** (top 50 % most confident), because a wrong pseudo‑label distorts the log‑Euclidean mean non‑linearly.

**Ablation (Table 10), stated correctly:** Euclidean DANN **33.4 %** → Euclidean DANN + the same losses (*optimized* Euclidean) **37.3 %** → SPD geometry‑only **39.8 %** → full SPD‑DANN **44.9 %**. So **geometry alone beats *plain* Euclidean DANN by 6.4 points, and beats the *optimized* Euclidean by 2.5 points** (the original report's "6.4 % over optimized Euclidean" conflated the two).

**What to borrow**: the SPD class‑prototype pair loss + entropy‑filtered pseudo‑labels — a manifold‑safe transductive alignment objective for MODMA LODO.

---

#### P23 — Parallel Transport + Moments Alignment (Relevance 8/10)
**"Domain Adaptation Using Riemannian Geometry of SPD Matrices"** — *ICASSP 2019*

Two‑step unsupervised DA using the target's unlabeled EEG:
1. **Parallel transport**: per source dataset, compute Riemannian mean M_s → transport along the geodesic to target mean M_t via `P → M_t^{1/2} M_s^{-1/2} P M_s^{-1/2} M_t^{1/2}`.
2. **Moments alignment**: in the tangent space at M_t, match variance/skewness of source and target.

Key finding: **PT alone fails when domains share a mean but differ in structure** — moments alignment is then essential. This is precisely the regime expected for lab‑different MODMA/Mumtaz/OpenNeuro.

```python
# Test time, unlabeled target dataset available
M_target = riemannian_mean(target_covs)
for src in sources:
    M_source = riemannian_mean(src.covs)
    P_aligned = M_target**0.5 @ M_source**-0.5 @ P @ M_source**-0.5 @ M_target**0.5
    match_variance(tangent(P_aligned, M_target), tangent(target, M_target))
# classify aligned source + target in the common tangent space
```

---

#### P24 — TSA: Tangent Space Alignment (Relevance 8/10)
**"Tangent Space Alignment: Transfer Learning for BCI"** — *Frontiers in Human Neuroscience 2022* (18 databases, 349 subjects)

TSA maps source covariances to the tangent space at the target's Riemannian mean and applies a full affine alignment (not just mean recentering). **+2.7 % accuracy over RPA** (previous best Riemannian TL) across 18 databases; subsumes rebiasing as a special case.

**Channel‑count caveat (corrected):** TSA aligns *distributions* in symmetric‑matrix (tangent) space, but the matrices must already be the **same dimension** for a common tangent space to exist. TSA does **not** remove the need for channel harmonization (§2); the original "handles different channel counts naturally" phrasing overstated this. Use TSA *after* the 19‑ch projection.

**Implementation**: `pyriemann.TangentSpace` with `reference_point` set to the target dataset's geometric mean.

---

#### P4 — CorSW (Relevance 8/10)
**"A Sliced‑Wasserstein Framework on Correlation Matrices for EEG Decoding"** — *KDD 2026*

CorSW‑LSM adds a Sliced‑Wasserstein distribution‑alignment loss (training‑only, **zero inference overhead**) that pulls each source's correlation‑matrix distribution toward a shared Gaussian reference on the correlation manifold. *Source‑only DG; no depression or cross‑dataset evaluation* — so the LODO use below is a **proposed extension**, not a validated result.

**Proposed LODO use**: during training on the two source datasets, regularize both toward a common reference, reducing the source‑source gap before any test‑time correction:
```
Loss_total = Loss_cls + λ·CorSW_LSM(Dataset1, ref) + λ·CorSW_LSM(Dataset2, ref)
```

### 3.2 Tier 2: High Value

- **P6 — RK‑SVM / Adaptive Reference (7/10).** Re‑estimating the Riemannian mean from the target's unlabeled data (ARK) is the **simplest transductive baseline** and should be run first. (Evidence is cross‑*session*; treat as a baseline, not proof for cross‑dataset.)
- **P21 — S‑RCT (5/10) 🔎 no PDF; preprint.** Per‑dataset Riemannian centering to identity; mathematically identical to P6 rebiasing applied per session. **Convergent** (not "universal") evidence from a biometric task. Combine with the class‑imbalance caution below.
  - *Caution (P52):* centering using **all** trials is biased under class imbalance; estimate the reference from **resting/baseline windows** rather than all windows. (P52 uses fixation periods; the specific "first 30 s of each recording" is a *design choice here*, not a P52 result.)
- **P3 — RHOP (7/10).** Quotient‑Gaussian embedding normalizes per‑token covariance to correlation form, removing amplitude/scale confounds across datasets; ~1K params, plug‑and‑play head. (P3 itself was **not** evaluated cross‑subject/cross‑dataset — cross‑dataset benefit is plausible but untested.)
- **P46 — HybridRDG (7/10).** Strongest endorsement that *"geometric covariance structure is central to zero‑shot EEG generalization,"* on **<50‑subject** clinical cohorts. **Ledoit‑Wolf shrinkage is essential at these N** (MODMA N=53). (ASD, not depression; balanced accuracy, no AUC.)
- **P68 — Correlation‑Manifold Networks (8/10).** Deep‑learning implementation of the correlation manifold (OLM/LSM BiMap layers) — outperforms covariance‑SPD nets for cross‑subject EEG. Architecture backbone if going deep. (arXiv.)
- **P31 + P35 — Source Selection.** **Negative transfer is real** (P31: K≈5–7 similar subjects beat using all N‑1). Within each LODO fold, optionally filter source *subjects* whose covariance distribution is far from the target via Riemannian tangent‑space MMD (P31) or rotation‑based SPD distance (P35).
- **P50 — PSDNorm (6/10).** Test‑time normalization via Monge‑OT mapping aligns the target distribution to the source at inference with no labels — a drop‑in layer after the Riemannian backbone. Validated cross‑dataset (10 sleep datasets), though feature‑space not SPD.

### 3.3 Tier 3: Good Ideas, Lower Priority

| Paper | Technique | Value |
|-------|-----------|-------|
| P14 RGP‑VAE | Parallel‑transport VAE augmentation | Low‑N augmentation |
| P32 Contrastive DA | Pull same‑class features across datasets in tangent space | Cross‑dataset class alignment |
| P33 BARN‑DA | Riemannian MMD as alignment objective | Alternative to CorSW |
| P25 RGT‑GAA | Geodesic (AIRM) adversarial loss | More geometry‑faithful than Euclidean GAN |
| P26 Stiefel Routing | K Stiefel experts + per‑dataset routing | Handles dataset heterogeneity |
| P45 ReTa‑Diffusion / WRFE | Wavelet+Riemannian fusion; covariance augmentation | Multi‑scale fusion, minority‑class augmentation |
| P28 SFDA pseudo‑labeling | Multicentric class prototypes from pseudo‑labels | When source data can't be shared |

---

## 4. The Double‑Dissociation Problem

This is the hardest part of *this* LODO setup, and **no paper in the corpus addresses it directly** (the observation and the numbers below are your own results, not literature).

- **MODMA**: signal ≈ functional connectivity (correlation structure across channels). `nr` removes identity, revealing the between‑subject connectivity axis.
- **Mumtaz + OpenNeuro**: signal ≈ spectral power + aperiodic 1/f exponent; `nrde` and `defu` dominate because they add band power.

**Consequences for LODO**:
- Train on Mumtaz+OpenNeuro (spectral) → test on MODMA (connectivity) is **cross‑signal‑type transfer** — the hardest direction.
- Train on MODMA+Mumtaz → test on OpenNeuro has both signal types in training → more feasible.

**Mitigations adapted from the corpus** (proposals, not validated for this setting):
1. **Multi‑stream architecture** — a connectivity stream (correlation + `nr`) and a spectral stream (DE/band power) fused at decision level (`defu`‑style). Supported in spirit by P5 (decision‑level FC fusion), P45 (WRFE wavelet+Riemannian fusion), P68 (correlation‑manifold net). *(The original report cited P13 here; P45/P5 are the better fusion references.)*
2. **Align the connectivity stream only** with CorSW; leave the spectral stream (already partly dataset‑invariant) unaligned.
3. **Dataset‑conditional stream weighting** — a lightweight gate that, from quick unlabeled statistics, weights connectivity vs spectral per target dataset.

---

## 5. Recommended LODO Pipeline

### 5.1 Preprocessing (frozen; own pipeline + Ledoit‑Wolf per P46)
```
19-channel 10-20 harmonization → all datasets
  MODMA: 128ch GSN → 10-20    Mumtaz: already 19-ch    OpenNeuro: 64ch → 19 named 10-20
125 Hz | 4 s/2 s windows | 1-40 Hz bandpass | Ledoit-Wolf shrinkage cov
  → correlation matrix (19×19) → Riemannian mean → tangent space
```

### 5.2 Training (two source datasets) — *composition is the contribution, not an established result*
**Stage 1 — dataset‑level alignment:** per‑dataset Riemannian mean; S‑RCT center each dataset to identity (use **baseline windows** to avoid class‑imbalance bias, P52); add CorSW‑LSM to keep the two source distributions close (P4).
**Stage 2 — identity removal:** `nr` K=3, projecting out the top‑3 between‑subject PCs computed from the **pooled** source subjects (so the projection is dataset‑agnostic).
**Stage 3 — classification:** logistic regression on aligned, identity‑removed tangent vectors; or SPDDSMBN → LogEig → linear if going deep.

### 5.3 Test Time (one target dataset, no labels) — transductive
1. Compute the target's Riemannian mean from all unlabeled test windows (baseline windows for the reference).
2. **Parallel transport** source training samples from their mean to the target mean (P23).
3. **Moments alignment** in the tangent space at the target mean (P23) — adds robustness when source/target structure differs.
4. Re‑apply identity removal (PCs recomputed on the aligned source).
5. Classify.

This sequential **TSA + PT + moments** (P24 + P23) on top of dataset‑level SPDDSMBN is your transductive contribution — **no paper in the corpus does all of this for multi‑source, dataset‑level depression LODO**.

### 5.4 Evaluation Protocol (3 LODO folds)

| Fold | Train | Test |
|------|-------|------|
| LODO‑1 | Mumtaz + OpenNeuro | MODMA |
| LODO‑2 | MODMA + OpenNeuro | Mumtaz |
| LODO‑3 | MODMA + Mumtaz | OpenNeuro |

Primary metric **subject‑level ROC‑AUC**; secondary **balanced accuracy** (imbalanced ON). Report a **95 % CI per fold** and the **across‑fold mean ± spread** (see §11). Tune only via **nested** CV (inner LOSO on the two source datasets) to avoid peeking at the held‑out dataset.

---

## 6. Baselines to Position Against (metrics kept honest)

| Method | Source | Reported number | Protocol & metric | Comparable to `nr` AUC? |
|--------|--------|-----------------|-------------------|--------------------------|
| `nr` K=3 (no alignment) | This work | **AUC 0.7227** | Within‑dataset, MODMA | — (own reference) |
| EEG‑RCformer | P8 | **AUC 0.7154 ± 0.061** | MODMA, subject‑indep (5 held‑out) | **Yes** — same metric/dataset; closest honest baseline |
| WDANet | P44 | **Acc 83.3/75.5/73.9/76.0/70.9 %; F1 77.8/87.3 %** | Cross‑subject + cross‑dataset depression; **no AUC** | **No** — accuracy/F1, different datasets; architectural competitor only |
| SPDDSMBN (TSMNet) | P22 | Balanced acc (MI/ERP) | Multi‑dataset UDA; not MODMA | No — method template, not a depression number |
| Zero‑shot DG | P4, P46 | task‑specific acc | Source‑only DG; not depression LODO | No |
| Your LODO (proposed) | This work | **TBD** | Dataset‑level LODO, AUC | the number to actually produce |

**Honest competitor to beat:** P8's **AUC 0.7154** on MODMA subject‑independent. WDANet is an *architecture* to learn from, not an AUC target.

---

## 7. Implementation Roadmap

**Step 1 — Channel harmonization (unblock LODO).** `triad/harmonize.py`: MODMA 128→19, OpenNeuro 64→19, Mumtaz unchanged; `N_CHANNELS_HARMONIZED = 19`; `--harmonize` flag; rebuild caches → 19×19 correlation matrices everywhere.

**Step 2 — Baseline: `nr` on harmonized 19‑ch (LODO).**
```bash
python run.py --stage nr --dataset lodo --train modma,mumtaz   --test opennuero --n-nuisance 3 --cv-folds 10 --harmonize
python run.py --stage nr --dataset lodo --train modma,opennuero --test mumtaz    --n-nuisance 3 --harmonize
python run.py --stage nr --dataset lodo --train mumtaz,opennuero --test modma    --n-nuisance 3 --harmonize
```
Establishes the **naive cross‑dataset floor** before any adaptation. *Expect a drop from the within‑dataset 0.72 — this is the gap to close, not a failure.*

**Step 3 — S‑RCT alignment** in `harness.py`: `_rebias(mats, ref_mean) = ref_mean^{-1/2} C ref_mean^{-1/2}`; source means from training, target mean from unlabeled test (baseline windows).

**Step 4 — Parallel transport** at test time (P23): transport source samples to the target tangent space; add moments matching if structure differs.

**Step 5 — CorSW regularizer** (optional, if Step 4 insufficient): push the two source datasets together during training.

> Any "+X % AUC" written next to these steps would be a **hypothesis**, not a literature value (P24's +2.7 % is *accuracy on BCI MI*, not depression LODO AUC). Measure each step's effect empirically with CIs.

---

## 8. Key Design Decisions (with evidence grade)

- **8.1 Correlation > covariance for cross‑dataset — ✅ strongly supported.** P3, P4, P15, P68 converge: scale‑invariant correlation matrices generalize better across subjects/sites. `nr`'s `rep="corr"` is the right call. *Best‑supported decision in this report.*
- **8.2 Transductive > source‑only DG — 🟡 supported by methods, not by a survey quote.** Using the unlabeled target dataset wins in P6, P22, P23, P50, P98. This is **not data leakage** (no labels used). Avoid the unqualified "across ALL categories" claim — it is not in P54.
- **8.3 Ledoit‑Wolf shrinkage — ✅ for small N.** Essential at **N < 50** (P46); MODMA's N=53 is in this regime. Keep it. *(The earlier "N<100" threshold overstated P46.)*
- **8.4 The double dissociation is a framing of your own result**, not a literature finding. State it as the central scientific question LODO tests here, and label it as observation/interpretation.
- **8.5 Do NOT use all subjects as sources — ✅.** Negative transfer is real (P31, P35, P39); filter dissimilar source subjects per fold.

---

## 9. Why LODO Is Used (and What It Buys You)

LODO (Leave‑One‑*Dataset*‑Out) is the strongest practical test of **deployment generalization** for clinical EEG:
- It directly simulates the real failure mode — a model meeting a **new site, montage, and population** — that LOSO does not (LOSO holds the dataset fixed).
- It is the cleanest guard against the two dominant sources of optimistic bias in EEG ML: **subject‑identity leakage** (P54: 95→60 % when leakage is removed) and **dataset‑specific artifacts** (P37 documents cross‑dataset variability directly).
- It is **feasible**: P22 (6 datasets), P24 (18 databases), P50 (10 datasets, leave‑dataset‑out), P48/P15 (multi‑site harmonization) show methods can be built and evaluated across many datasets/sites.
- With three datasets you get three independent "new deployment" tests — a far more credible generalization claim than any single within‑dataset CV number.

## 10. When LODO Fails (and How to Tell)

LODO can produce **misleading or unfairly low** numbers when:
1. **The shift is too large to bridge.** P23 (PT alone fails when structure differs), P31 (negative transfer), and P98 (transfer accuracy collapses to ~30 % for dissimilar subject pairs) show alignment can fail or reverse. A near‑chance LODO fold may reflect an *ill‑posed transfer*, not a bad model.
2. **Signal‑type mismatch (the double dissociation).** Training on spectral‑dominant datasets and testing on a connectivity‑dominant one (or vice‑versa) asks the model to bridge a gap no current method closes — LODO then measures the *dataset gap*, not the *method*.
3. **Too few datasets.** With 3 folds, one anomalous dataset dominates the mean; a single LODO number is a high‑variance point estimate (§11).
4. **Reference estimated on imbalanced data.** Centering on all (class‑imbalanced) target trials injects bias (P52) that can *look* like poor generalization.
5. **Hyper‑parameters tuned on the held‑out dataset.** Any peeking turns LODO into an optimistic in‑distribution estimate.

**Diagnosis kit:** always report the **within‑dataset LOSO ceiling** alongside each LODO fold (so you can attribute the drop to transfer vs model); inspect per‑dataset Riemannian‑mean distances (large distance ⇒ expect a hard fold); and check whether the failing direction is the cross‑signal‑type one.

## 11. Statistical Considerations

- **Confidence intervals dominate.** At N≈53–121 subjects, a subject‑level AUC has a 95 % CI of roughly **±0.07–0.10** (DeLong/bootstrap). Several inter‑fold differences this report once treated as meaningful are **inside the noise**. Report CIs; do not chase third‑decimal AUC differences.
- **Three folds is not a sample for significance testing.** Avoid t‑tests across 3 LODO folds. Prefer per‑fold bootstrap CIs and, for method comparisons, **paired subject‑level bootstrap within each fold**.
- **Class imbalance** (ON 46/75) ⇒ use **AUC + balanced accuracy**, never raw accuracy (the trap in P44/P98 reporting). P8 and P52 both default to AUC for this reason.
- **Multiple comparisons.** Each added alignment step is a hypothesis; correct for the number of pipeline variants tried (or pre‑register the pipeline) to avoid selecting a lucky fold.
- **Effect sizes.** Where you do claim an improvement, report an effect size as P8 did (Cohen's d / AUC delta with CI), not just a point gain.

## 12. Common Pitfalls (checklist)

- ❌ Comparing **accuracy to AUC** across methods (the WDANet error). Keep metric, dataset, and protocol fixed before comparing.
- ❌ Importing a paper's **LOSO** number as if it were **LODO** evidence.
- ❌ Treating **pre‑print SOTA** as established (most 2026 backbone papers).
- ❌ Estimating the alignment reference from **class‑imbalanced** trials (P52).
- ❌ **AIRM‑everywhere** assumptions — Log‑Euclidean is sometimes chosen deliberately for stable closed‑form means (P98, P46).
- ❌ Trusting **pseudo‑labels** on the manifold without confidence filtering — wrong labels distort log‑Euclidean means non‑linearly (P98).
- ❌ Tuning on the held‑out dataset (use nested CV).
- ❌ Reporting a single LODO number without a CI or the within‑dataset ceiling.

## 13. Best Practices (consensus distilled from the corpus)

1. **Harmonize channels first**; all SPD/tangent methods need equal dimension (§2).
2. **Use scale‑invariant correlation matrices** (P3/P4/P68) + **Ledoit‑Wolf** shrinkage at small N (P46).
3. **Align means *and* structure** — recentering alone is insufficient (P23/P24/P98).
4. **Exploit unlabeled target data** transductively when available (P6/P22/P23/P50) — it is not leakage.
5. **Filter source data** to avoid negative transfer (P31/P35).
6. **Report AUC + balanced accuracy with CIs**, plus the within‑dataset LOSO ceiling.
7. **Ablate the geometric component** explicitly (as P8/P22/P98 do) so gains are attributable.
8. **Pre‑specify the pipeline / use nested CV**; disclose pre‑print status of cited methods.

## 14. Generalization Implications

- **Subject‑level → dataset‑level is a real ladder, not a single rung.** Strong LOSO numbers (P8 0.72 on MODMA) are an *upper bound* on what to expect at the dataset level; the report's value is in honestly measuring the *gap*.
- **Geometry is the most transferable inductive bias** found in the corpus — it helps precisely because it removes subject/dataset‑specific scale and reference confounds (P3/P6/P8/P46/P98). It is the right foundation for LODO.
- **There is a transfer ceiling set by signal‑type compatibility.** When datasets encode different physiology (connectivity vs spectral), generalization is bounded regardless of alignment sophistication — this is the scientific finding the LODO experiment is positioned to demonstrate.
- **The open research gap (P69) is exactly your contribution:** test‑time Fréchet‑mean / transductive adaptation for multi‑source, dataset‑level clinical EEG. No corpus paper occupies this cell.

---

## 15. Research Gaps & Future Directions

1. **No prior strict dataset‑level LODO depression benchmark with subject‑level AUC** — establishing one (the 3‑fold protocol here) is itself a contribution.
2. **Test‑time transductive Fréchet adaptation** on the SPD/correlation manifold for clinical EEG (gap named by P69) — the core novelty.
3. **Cross‑signal‑type transfer** (connectivity ↔ spectral) is unaddressed by any paper — a multi‑stream + gating approach is an open design space (§4).
4. **Correlation‑manifold + transductive DA** has not been combined (P4 is source‑only; P68 has no DA) — a natural fusion.
5. **Honest small‑N evaluation** (CIs, nested CV, within‑dataset ceilings) is rare in the corpus — adopting it is a methodological contribution.

---

## 16. Paper Reference Summary (priorities, corrected)

| Paper | Method | Contribution to LODO | Priority |
|-------|--------|----------------------|----------|
| P44 WDANet | Wasserstein 3‑discriminator DA (depression, **accuracy/F1, not AUC**) | Architecture blueprint; *not* an AUC baseline | Critical |
| P22 TSMNet/SPDDSMBN | End‑to‑end SPD UDA via momentum Fréchet mean | Train‑side alignment; **transductive extension = novel** | Critical |
| P23 PT+Moments | Parallel transport + moments matching | Test‑time transductive step | Critical |
| P24 TSA | Tangent‑space alignment | Test‑time alignment (needs equal dim) | Critical |
| P4 CorSW | Sliced‑Wasserstein on correlation manifold | Training regularizer (proposed LODO extension) | High |
| P98 SPD‑DANN 🔎 | Adversarial DA on SPD; manifold‑safe pseudo‑labels | Class‑prototype loss for test‑time alignment | High |
| P6 RK‑SVM/ARK | Adaptive reference (geometric mean) | Simplest transductive baseline | High |
| P21 S‑RCT 🔎 | Per‑session Riemannian centering | Per‑dataset centering (convergent, preprint) | Medium |
| P8 EEG‑RCformer | AIRM clustering + Transformer | **Honest AUC baseline (0.7154) on MODMA** | High |
| P46 HybridRDG | Hybrid EEGNet+Riemannian DG | Ledoit‑Wolf + "geometry is central" | Medium |
| P68 Corr Manifold | DL on correlation manifold | Deep backbone if needed | Medium |
| P3 RHOP | Scale‑invariant SPD pooling head | Cross‑dataset scale normalization | Medium |
| P50 PSDNorm | OT test‑time normalization | Drop‑in test‑time layer (cross‑dataset‑validated) | Medium |
| P31/P35 Source selection | Riemannian similarity selection | Avoid negative transfer | Low |
| P52 Recenter | Class‑agnostic baseline recentering | Imbalance‑safe reference estimation | Low |
| P37 / P48 / P15 | Cross‑dataset variability; multi‑site harmonization | **Direct LODO‑relevant evidence (add to related work)** | Background |
| P54 Survey | Cross‑subject DL survey | Positions contribution; leakage/inflation evidence | Background |
| P69 Survey | SPD neuroimaging review | Names transductive test‑time adaptation as the open gap | Background |

---

*Companion documents:* `Verification_Report.md` (claim‑by‑claim audit & corrections) · `Synthesis_Matrix.md` (cross‑paper evidence matrix, agreements/contradictions, when‑LODO‑helps) · `Paper Summary.md` (the 98 individual paper summaries — Deliverable 1).
