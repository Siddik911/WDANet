# Verification Report — Audit of `LODO_Strategy_Report.md`

**Audited document:** `LODO_Strategy_Report.md` (379 lines, version dated 2026‑06‑13)
**Evidence base:** `Paper Summary.md` (98 paper summaries) + the 96 source PDFs in `Other Papers/`, with primary‑source spot‑checks of the load‑bearing claims.
**Auditor method:** Each substantive statement in the report was classified **Supported / Partially Supported / Unsupported / Contradicted** against the cited paper(s). Where a claim carries a number or a direct quotation, the underlying PDF (not only the summary) was opened and searched. Section/line references point into `LODO_Strategy_Report.md`; paper references use the `Paper Summary.md` numbering (P1–P98).

> **Legend** — ✅ Supported · 🟡 Partially Supported · ⚠️ Unsupported (no evidence found) · ❌ Contradicted (evidence says otherwise) · 🔎 Unverifiable (cited paper has no PDF in the corpus / claim is the author's own result)

---

## Part A — Corpus Coverage Verification (Phase 5 / Quality Control)

**Result: all 96 PDFs in `Other Papers/` are already summarized in `Paper Summary.md`.** No source PDF is missing a summary. The mapping was confirmed by matching journal/article codes and titles (e.g. `s40708`→Brain Informatics = P19, `fphy‑2021`→Frontiers in Physics = P42, `s11571`→Cognitive Neurodynamics = P36, `s10489`→Applied Intelligence = P97, `s41598`→Scientific Reports = P53, the two `Tesi_dottorato_*` = P60/P64, `PhDThesis.pdf` = P59, `Pan_2026_J._Neural_Eng.` = P10, `TR_2025_01` = P63).

**Counts:** 96 PDFs · 98 summaries · 7 subfolders.

**Two summaries have no PDF in the corpus** (they were summarized from external sources and therefore cannot be checked against a primary document here):

| Paper | Title | Status | Consequence for the report |
|-------|-------|--------|----------------------------|
| **P21 — S‑RCT** | Robust Brainprint Recognition via Session‑wise Riemannian Alignment | 🔎 No folder PDF; Research Square **preprint** (not peer‑reviewed); biometric task | Report cites P21 (§3.2, §5.2, §7‑Step 3) for "per‑dataset Riemannian centering." Technique is independently supported by P6, but the specific P21 evidence ("validates universally") is unverifiable here. |
| **P98 — SPD‑DANN** | SPD‑DANN: An SPD manifold UDA method for cross‑subject MI EEG | 🔎 No folder PDF (the `Other Papers/.../SPD‑DANN... .pdf` path in the summary does not exist on disk) | Report cites P98 (§3.1, §10) with a numeric ablation. The number cannot be checked against the paper; an internal arithmetic inconsistency is documented in Part B. |

> The PDF `DOMAIN ADAPTATION USING RIEMANNIAN GEOMETRY OF SPD MATRICES.pdf` corresponds to **P23** (Yair et al., ICASSP 2019), **not** P98 — do not confuse the two adversarial/geometry DA papers.

**One matcher caveat (not an error in the corpus):** the file `A Riemannian Network for SPD Matrix Learnin.pdf` is **P65 (SPDNet, AAAI 2017)**, not P6.

**Terminology flag affecting the whole document:** `Paper Summary.md` uses **"LODO" to mean Leave‑One‑*Subject*‑Out (LOSO)** in most entries (e.g. P8 "LODO: Yes (SEED LOSO)", P31 "LODO: Yes (9‑subject LOSO)"). `LODO_Strategy_Report.md` uses **"LODO" to mean Leave‑One‑*Dataset*‑Out**. These are different protocols. When the report imports a paper's "LODO/LOSO" result as evidence for *dataset‑level* LODO, it is generally importing *subject‑level* evidence. This is the single most pervasive interpretive slippage in the document (see Part D‑1).

---

## Part B — Claim‑by‑Claim Audit

### §Header & §1 — Problem framing

| # | Report claim (location) | Verdict | Evidence / Note |
|---|--------------------------|---------|-----------------|
| 1 | `nr` K=3 AUC ≈ 0.72 on all three datasets; nr K=3 locked as final model (header, line 6) | 🔎 | **Author's own experimental result**, not from any paper. Traceable to `results/*.json`, not to the literature. Correct to present as own work; should be labeled as such. |
| 2 | Channel mismatch (19 vs 64 vs 128) is a "Blocker" (§1.1) | ✅ | Consistent with the whole transfer‑learning literature; P24/P22 require common matrix dimension; report's own §2 treats harmonization as mandatory. Reasonable. |
| 3 | "Double dissociation: MODMA = functional connectivity; Mumtaz/OpenNeuro = spectral 1/f + band power" (§1.1, §4) | 🔎 | **Author's own dataset observation** (cites own `nr`/`nrde`/`defu` results), explicitly noted as "no paper addresses it directly." Not literature‑traceable; correctly framed as interpretation. |
| 4 | "Distribution shift → covariance manifold means are far apart" (§1.1) | ✅ | Supported by P6, P21, P22, P23 (all built on the premise that cross‑domain Fréchet means differ). |
| 5 | **§1.2 "Key Insight from Survey (Paper 54)" — the quoted sentence** ("Transductive methods … outperform domain generalization across ALL categories. LOSO is the gold standard … other protocols often inflate performance.") | ❌/⚠️ | **The quotation is not in the survey.** Full‑text search of the 21‑page P54 PDF: "transductive" → 0 hits, "gold standard" → 0 hits, "inflate/inflation" → 1 hit but in the *subject‑dependent data‑leakage* context (accuracy "dropping from >95% to ~60% on unseen subjects"), **not** a LOSO‑vs‑other‑subject‑independent‑protocol claim. The survey lists LOSO as *one of several* subject‑independent protocols, not "the gold standard." **Action: remove the quotation marks and the attribution as a verbatim quote.** The *substance* ("use unlabeled target data") is independently supported (rows 6, 19, 25) — re‑ground it there. |

### §2 — Channel harmonization

| # | Report claim | Verdict | Evidence / Note |
|---|--------------|---------|-----------------|
| 6 | Three harmonization options; Option A (19‑ch 10‑20 intersection) recommended; "confirmed by SPARC‑Net to work" (§2 A) | 🟡 | The *engineering* logic is sound and standard. However **SPARC‑Net is not one of the 98 papers** — it is an external code reference; the claim "confirmed by SPARC‑Net" is not traceable to this corpus. Mark as implementation note, not literature‑supported. |
| 7 | Option B "spectral profiles → dataset‑agnostic 5‑dim feature … destroys spatial connectivity" (§2 B) | 🔎 | Author's design reasoning + own results. Plausible, not from papers. |
| 8 | Option C "Shared latent space (TSMNet/SPDDSMBN path)" (§2 C) | ✅ | P22 is exactly a learned shared‑space UDA method. Correct pointer. |

### §3.1 — Tier 1 papers

| # | Report claim | Verdict | Evidence / Note |
|---|--------------|---------|-----------------|
| 9 | **P44 WDANet** is "the only paper … that directly addresses cross‑domain depression on MODMA"; 3 discriminators (global/local/Wasserstein) + dynamic adversarial factor (§3.1) | ✅ (architecture) | PDF confirms two self‑collected datasets + MODMA, the three‑discriminator design, and the A‑distance dynamic factor. P8 also uses MODMA for depression but cross‑*subject* not cross‑*dataset*, so "only cross‑domain depression on MODMA" is defensible. |
| 10 | **P44 numeric comparison** — Report §6 table lists "WDANet … MODMA AUC ~0.73–0.83"; §9 "WDANet's best cross‑dataset result (~0.73–0.83)"; §6 compares it to `nr` AUC 0.7227 in one column | ❌ | **Primary‑source contradiction.** The WDANet PDF reports **classification accuracy** (83.33 / 75.52 / 73.93 / 76.04 / 70.94 %) and **F1** (77.83 / 87.32 %). Full‑text search of the 14‑page PDF for "AUC" → **0 hits**; WDANet never reports AUC. The 83.33 % is *within‑dataset cross‑subject* accuracy (Case 1), not a MODMA AUC and not strict LODO. **Placing accuracy in a column headed "AUC" and comparing it to `nr`'s 0.7227 AUC is an invalid apples‑to‑oranges comparison.** Action: relabel as accuracy, separate the datasets, and stop treating 0.73–0.83 as an AUC the `nr` model must "beat." |
| 11 | **P22 SPDDSMBN** tracks each domain's Fréchet mean via EMA during training → tangent transport → domain shift absorbed (§3.1) | ✅ | Verbatim‑accurate to P22 (NeurIPS 2022). |
| 12 | "Paper 69's review identifies SPDDSMBN as current SOTA for cross‑subject EEG UDA" (§3.1) | ✅ | P69 Key Findings: "SPDDSMBN (Paper 22)/TSMNet is current state‑of‑the‑art for cross‑subject EEG UDA." (Note: P69 is an arXiv **preprint** by the same group as P22 — mild self‑reference; see Part D‑3.) |
| 13 | "Transductive extension (feed unlabeled target at test time to update target Fréchet mean) … Paper 22 does not do this — your contribution" (§3.1) | ✅ | P22 summary: "No transductive component … target adapted only in the batch‑norm layer." P69 lists test‑time Fréchet adaptation as an open gap. Novelty claim is well‑grounded. |
| 14 | **P98 SPD‑DANN** "First adversarial training on the SPD manifold"; L_M log‑Euclidean mean distance; L_P triplet with entropy‑filtered pseudo‑labels (top 50 %) (§3.1) | ✅ (vs summary) / 🔎 (vs PDF) | Matches the P98 summary exactly. **No PDF in corpus → cannot verify against primary source.** |
| 15 | **P98 ablation** — "Euclidean DANN 33.4 % → SPD geometry alone 39.8 % → Full 44.9 %. **Geometry alone beats optimized Euclidean by 6.4 %.**" (§3.1) | ❌ | **Misattributed comparison.** Per the report's own P98 summary (Table 10): Euclidean DANN **33.4** → Euclidean DANN + same losses (= *optimized* Euclidean) **37.3** → SPD geometry‑only **39.8** → Full **44.9**. Geometry‑only beats **plain** Euclidean DANN by 6.4 (39.8−33.4); it beats the **optimized** Euclidean by only **2.5** (39.8−37.3). The sentence pairs "optimized" with the wrong delta. Action: state both deltas correctly. |
| 16 | **P23 PT formula** `P → M_t^{1/2} M_s^{-1/2} P M_s^{-1/2} M_t^{1/2}`; "PT alone fails when domains share a mean but differ in structure" (§3.1) | ✅ | Formula and claim match P23 exactly. |
| 17 | **P24 TSA** "+2.7 % over RPA across 18 databases" (§3.1) | ✅ | Matches P24 (Frontiers Hum. Neurosci. 2022). |
| 18 | **P24 TSA** "handles different channel counts naturally (tangent space is symmetric‑matrix space, not channel space)" (§3.1, §10) | 🟡 | **Internally tense.** Tangent‑space alignment needs equal matrix size for a *common* tangent space; the report itself makes 19‑ch harmonization the "mandatory first step" (§2), which contradicts "handles channel counts naturally." The report partly rescues this with "IF you first reduce to common channel space" (§3.1), but the standalone phrasing overstates TSA. Action: state that TSA aligns *distributions* once matrices are the same dimension; channel harmonization remains required. |
| 19 | **P4 CorSW** Sliced‑Wasserstein loss aligning each source's correlation distribution to a shared Gaussian reference; training‑only, zero inference cost (§3.1) | ✅ | Matches P4 (KDD 2026). Note P4 is *source‑only DG* with no depression/cross‑dataset eval; the report's LODO extension (align two source datasets) is a reasonable but **untested** adaptation — label as proposed extension. |

### §3.2 — Tier 2 papers

| # | Report claim | Verdict | Evidence / Note |
|---|--------------|---------|-----------------|
| 20 | **P6** adaptive reference update (re‑estimate Riemannian mean from unlabeled target) is the simplest transductive baseline (§3.2) | ✅ | P6 (Barachant et al. 2013) ARK‑SVM = exactly this. Note: P6's evidence is *cross‑session*, not cross‑subject/cross‑dataset; framing as "baseline transductive step" is fair. |
| 21 | **P21 S‑RCT** per‑dataset centering; "Independent evidence (brainprint task) validates this approach universally" (§3.2) | 🟡/🔎 | Technique = P6 rebiasing applied per‑session; supported in principle. "Validates universally" **overstates** a single Research Square **preprint** on a 22‑ch biometric task. No PDF → unverifiable. Soften to "convergent evidence from a biometric task." |
| 22 | "Warning from Paper 52: centering using all trials is biased if classes are imbalanced. Use the resting baseline windows (**first 30 s** of each recording)" (§3.2, §5.2) | 🟡 | The *class‑imbalance bias* warning is ✅ supported by P52. But "**first 30 s** of each recording" is **the report's own operationalization** — P52 uses *fixation/rest periods between trials* (online MI), not a 30‑s prefix. Label the 30‑s rule as a design choice, not a P52 finding. |
| 23 | **P3 RHOP** quotient‑Gaussian embedding normalizes covariance → correlation, removing amplitude/scale confounds; ~1K params, plug‑and‑play (§3.2) | ✅ | Matches P3 (ICLR 2026). Caveat: P3 has **no cross‑subject/LODO evaluation** (population‑level/within‑subject only); cross‑dataset benefit is plausible but untested — label as such. |
| 24 | **P46 HybridRDG** "geometric covariance structure, not merely Euclidean alignment, is central to zero‑shot EEG generalization"; validated on small clinical cohorts (<50) (§3.2) | ✅ | Direct quote‑equivalent in P46 (CVPR‑W 2026). Caveat: task is **ASD**, multi‑paradigm within one study (Cross‑Dataset: No), balanced‑accuracy (no AUC). |
| 25 | **P68** "OLM/LSM correlation‑manifold BiMap layers > SPD (covariance) manifold for cross‑subject EEG" (§3.2) | ✅ | P68 Main Result. Note P68 is an arXiv **preprint**. |
| 26 | **P31 + P35** "using all source subjects/datasets can hurt (negative transfer)"; select similar subjects via Riemannian MMD / rotation‑based SPD distance (§3.2, §8.5) | ✅ | P31: "negative transfer is a real problem … source selection is as important as the adaptation method; K≈5–7 optimal." Supported. (Application to *within‑source‑dataset subject* selection in a 3‑dataset LODO is a reasonable extension.) |
| 27 | **P50 PSDNorm** test‑time normalization via Monge OT, no labels (§3.2) | ✅ | Matches P50 (ICLR 2026). Caveat: sleep‑staging task; feature‑space (not SPD) normalization. |

### §3.3 — Tier 3 (P14, P32, P33, P25, P26, P49, P28)

| # | Report claim | Verdict | Evidence / Note |
|---|--------------|---------|-----------------|
| 28 | One‑line characterizations of P14 (PT‑VAE augmentation), P32 (contrastive cross‑dataset class alignment), P33 (Riemannian MMD), P25 (geodesic adversarial loss), P26 (Stiefel experts/routing), P49 (flow‑matching augmentation), P28 (SFDA multicentric pseudo‑labels) (§3.3 table) | ✅ | Each one‑liner matches the corresponding summary's core idea. These are low‑stakes pointers; no numeric claims to contest. |

### §4 — Double dissociation

| # | Report claim | Verdict | Evidence / Note |
|---|--------------|---------|-----------------|
| 29 | "no paper addresses [the double dissociation] directly" (§4) | 🔎 | Author's own framing; the numbers (nrde 0.951, defu 0.749) are own results. Honest as stated. |
| 30 | Multi‑stream connectivity+spectral fusion "Papers 5, 13, 68" (§4) | 🟡 | P5 = decision‑level fusion of FC streams ✅; P68 = correlation‑manifold net ✅; **P13 (RepSPD)** is dynamic‑graph SPD representation, not specifically a two‑stream connectivity+spectral fusion — weak citation. Replace P13 with a stronger fusion reference (e.g. P45 WRFE wavelet+Riemannian fusion, or P8 PSD+DE). |

### §5 — Recommended pipeline

| # | Report claim | Verdict | Evidence / Note |
|---|--------------|---------|-----------------|
| 31 | Preprocessing block: 125 Hz, 4 s/2 s windows, 1–40 Hz, Ledoit‑Wolf, correlation matrix, Riemannian mean, tangent space (§5.1) | 🔎 | Author's frozen pipeline (own work) + Ledoit‑Wolf justified by P46. Fine as own‑work + one supported component. |
| 32 | Training Stage 1–3 (per‑dataset Riemannian mean + S‑RCT + CorSW; pooled `nr` K=3; LR/SPDDSMBN head) (§5.2) | 🟡 | Each component is individually supported (P21/P6 centering, P4 CorSW, P22 head). The *composition* is a novel proposal, not a literature result — correctly framed as the contribution, but should not read as established. |
| 33 | Test‑time: target mean → PT → moments → identity removal → classify = "TSA + PT + moments (P24+P23) … none of the papers do all three together for multi‑source LODO depression" (§5.3) | ✅ (novelty) | Consistent with the corpus: no paper combines TSA+PT+moments for *dataset‑level* depression LODO. Novelty claim holds. |
| 34 | Evaluation: 3 LODO folds; subject‑level ROC‑AUC + balanced accuracy (§5.4) | ✅ | Sound protocol; AUC+balanced‑accuracy choice is supported by P8/P52 (imbalance) reasoning. |

### §6 — Baselines table

| # | Report claim | Verdict | Evidence / Note |
|---|--------------|---------|-----------------|
| 35 | `nr` K=3 = 0.7227 MODMA (within‑dataset) | 🔎 | Own result. |
| 36 | EEG‑RCformer sub‑indep = 0.7154, "Cross‑subject, single dataset" (P8) | ✅ | P8: MODMA sub‑independent AUC **0.7154 ± 0.0608** (5 held‑out subjects). Correctly labeled as single‑dataset cross‑subject (not LODO). Minor caveat: it is a *single 5‑subject hold‑out*, not full LOSO. |
| 37 | WDANet "~0.73–0.83" as MODMA AUC baseline to beat | ❌ | See row 10. Accuracy, not AUC; not MODMA‑specific; not LODO. |
| 38 | SPDDSMBN "Not on MODMA" | ✅ | Correct — P22 never used a psychiatric dataset. |

### §7 — Implementation roadmap

| # | Report claim | Verdict | Evidence / Note |
|---|--------------|---------|-----------------|
| 39 | Steps 1–5 (harmonize → nr baseline → S‑RCT → PT → CorSW) with expected "+2–5 % AUC on LODO‑1/3" (Step 3) (§7) | 🔎/🟡 | Engineering plan = own work (fine). The "+2–5 % AUC" expectations are **projections, not measured or literature‑derived** — should be labeled as hypotheses. P24's "+2.7 % over RPA" is *accuracy on BCI MI*, not AUC on depression LODO, so it cannot be transferred as a numeric expectation. |

### §8 — Key design decisions

| # | Report claim | Verdict | Evidence / Note |
|---|--------------|---------|-----------------|
| 40 | "Correlation > covariance for cross‑dataset — three independent papers (3, 4, 68) converge" (§8.1) | ✅ | Strongly supported: P3 (quotient Gaussian), P4 (CorSW finding), P68 (correlation‑manifold nets), and also P15. Best‑supported claim in the report. |
| 41 | "Transductive >> Domain Generalization — Paper 54: transductive outperforms zero‑shot DG across ALL categories" (§8.2) | 🟡 | The *direction* (using unlabeled target helps) is supported by P22, P23, P6, P50, P98 and by P54's DA‑vs‑DG framing. But "across ALL categories" and the survey attribution are **not** in P54's text (row 5). Re‑ground on the methods papers and soften "across ALL categories." |
| 42 | "Transductive use of unlabeled test EEG is not data leakage (no labels used)" (§8.2) | ✅ | Standard transductive/UDA position; consistent with P22/P23/P6/P50. Correct. |
| 43 | "Ledoit‑Wolf essential for **N < 100** subjects" (§8.3) | 🟡 | P46 states **< 50** subjects. The report loosens the threshold to <100. Minor overstatement — change to "<50 (P46); MODMA's N=53 is in the regime where shrinkage matters." |
| 44 | "Double dissociation is a feature, framing it as novel insight" (§8.4) | 🔎 | Author's framing of own results. Fine as labeled interpretation. |
| 45 | "Do NOT use all subjects as sources — Papers 31, 35, 39: negative transfer is real" (§8.5) | ✅ | P31 supports directly; P35/P39 are source‑selection papers consistent with the claim. |

### §9 — Expected AUC ranges

| # | Report claim | Verdict | Evidence / Note |
|---|--------------|---------|-----------------|
| 46 | Projected LODO AUCs (0.55–0.74 across folds), "Calibrated from Literature" (§9 table) | ⚠️ | **No paper provides these numbers.** They are the author's projections. The "Calibrated from Literature" label is misleading — relabel "Author projections (hypotheses, not measured)". The hardest‑fold reasoning (spectral→connectivity) follows from §4 (own observation), not from any paper. |
| 47 | "WDANet's best cross‑dataset result (~0.73–0.83) is on a larger proprietary dataset" (§9) | 🟡 | Partly right (Dataset 1 = 170 subjects, self‑collected) but conflates the five accuracy figures and mislabels them as cross‑dataset AUC. See row 10. |

### §10 — Reference summary table

| # | Report claim | Verdict | Evidence / Note |
|---|--------------|---------|-----------------|
| 48 | Per‑paper one‑line roles for P44, P22, P23, P24, P4, P98, P6, P21, P8, P46, P68, P3, P50, P31, P54, P69 | ✅ (mostly) | Roles match the summaries, with the same caveats as above for P44 (accuracy≠AUC), P21/P98 (no PDF), P54 (quote). "P6 ARK‑SVM" is labeled "ARK‑SVM" here vs "RK‑SVM" in §3.2 — harmonize the name. |

---

## Part C — Consolidated Corrections Required (priority‑ordered)

1. **[Critical] WDANet is accuracy, not AUC.** Remove WDANet's 0.73–0.83 from any "AUC" column. Present it as cross‑subject **accuracy** (max 83.3 % within‑dataset; cross‑dataset **F1** 77.8 / 87.3 %), note WDANet reports **no AUC**, and stop framing it as a MODMA‑AUC number `nr` must beat. (rows 10, 37, 47)
2. **[Critical] §1.2 survey "quote" is not in Paper 54.** Delete the quotation marks/attribution. Keep the *idea* (unlabeled‑target adaptation helps; subject‑dependent evaluation inflates accuracy) but cite the methods papers (P22, P23, P6, P50, P98) and P54's actual data‑leakage finding. (row 5)
3. **[High] SPD‑DANN "6.4 %" is misattributed.** Geometry‑only beats **plain** Euclidean DANN by 6.4 (33.4→39.8) and the **optimized** Euclidean by 2.5 (37.3→39.8). State both. (row 15)
4. **[High] Flag P21 and P98 as having no primary PDF** in the corpus; both partly preprint/non‑depression. Soften "validates universally" (P21) and present P98's numbers as summary‑sourced. (rows 14, 21; Part A)
5. **[High] §9 "Calibrated from Literature" → "Author projections."** No paper supplies these AUC ranges. (row 46)
6. **[Medium] Fix the LODO/LOSO terminology.** State explicitly that the report's LODO = Leave‑One‑*Dataset*‑Out, and that most cited "LODO" evidence is actually subject‑level LOSO; only P22/P24/P44/P50 approach true cross‑dataset evaluation. (Part A; Part D‑1)
7. **[Medium] Soften the TSA "handles different channel counts naturally" claim** — alignment needs equal matrix dimension; harmonization stays mandatory. (row 18)
8. **[Medium] Ledoit‑Wolf threshold:** change "N<100" to "<50 (P46)." (row 43)
9. **[Low] "First 30 s" centering window is a design choice**, not a P52 finding; label it. (row 22)
10. **[Low] Replace the weak P13 citation** in the multi‑stream‑fusion paragraph with P45 (WRFE) or P8 (PSD+DE). (row 30)
11. **[Low] Mark SPARC‑Net** as an external code reference (not one of the 98 papers). (row 6)
12. **[Low] Harmonize P6 naming** ("RK‑SVM" vs "ARK‑SVM"). (row 48)

---

## Part D — Missing Evidence, Missing Related Work, Missing Discussion

**D‑1. The report never defines its protocol relative to the literature's.** It should state up front that (a) LODO = Leave‑One‑Dataset‑Out, a *stricter* generalization test than the LOSO used by almost every cited paper, and (b) consequently most "supporting" results are *upper bounds* relative to what to expect at the dataset level.

**D‑2. Cross‑dataset–specific evidence is under‑used.** The papers that actually did cross‑*dataset* / cross‑*site* evaluation — **P22** (6 datasets), **P24** (inter‑dataset TL), **P44** (cross‑dataset depression), **P50** (leave‑dataset‑out on 10 sleep datasets), **P15** (multi‑site fMRI), **P48** (multi‑site ADHD harmonization), **P37** (cross‑dataset variability problem) — are the most relevant to LODO and deserve a dedicated subsection. **P37** and **P48** are not cited at all despite being directly on the "cross‑dataset variability" and "multi‑site harmonization" themes.

**D‑3. Peer‑review status is never disclosed.** A large share of the "Critical/High" backbone is 2026 pre‑prints / under‑review: P3 (ICLR'26), P4 (KDD'26), P44 (IEEE TAC'26), P45 (ICLR'26 under review), P46 (CVPR‑W'26), P54 (arXiv/IOP), P68 (arXiv), P69 (arXiv), P98 (2026), P21 (Research Square). P69 is by the **same group** as P22 (self‑reinforcing SOTA claim). A one‑line maturity caveat is warranted.

**D‑4. No statistical‑significance discussion.** Only P8 reports significance tests (p=0.0127, effect size 1.92) and P52 (p<0.05). The report sets AUC targets to two decimals (e.g. ">0.72") without any power/CI consideration for N≈53–121 subjects. With these N, AUC 95 % CIs are roughly ±0.07–0.10 — wider than several of the inter‑fold differences the report treats as meaningful. This belongs in the report (see new §"Statistical considerations").

**D‑5. No discussion of "when LODO fails" / common pitfalls.** The task brief explicitly requires it; the current report has none. (Added in the revised report.)

**D‑6. Negative results / disconfirming evidence are absent.** P23 ("PT alone fails"), P98 (transfer collapses to ~30 % for dissimilar subject pairs), P31 (negative transfer), and the P54 leakage finding all temper optimism and should be surfaced, not just the success stories.

---

## Part E — Claims That Could Not Be Verified (explicit list)

| Claim | Why unverifiable |
|-------|-------------------|
| P98 SPD‑DANN ablation numbers (33.4/37.3/39.8/44.9) | No PDF in corpus; checkable only against the summary, where an internal inconsistency was found. |
| P21 S‑RCT "98.33 % / 90.89 %" and "validates universally" | No PDF in corpus; Research Square preprint. |
| `nr` K=3 = 0.7227; nrde 0.951; defu 0.749; "AUC ≈ 0.72 on all 3" | Author's own results (`results/*.json`), not literature; not auditable from the papers. |
| §9 expected LODO AUC ranges; §7 "+2–5 % AUC" gains | Author projections; no paper provides them. |
| "SPARC‑Net `prepare_xdataset.py` does this" | External codebase, not in the 98‑paper corpus. |
| Whether any paper does *dataset‑level* LODO depression with TSA+PT+moments | Confirmed **absent** from the corpus (supports the novelty claim, but is an absence, not a positive citation). |

---

## Part F — Quality‑Control Checklist (Phase 5)

- [x] Every PDF in `Other Papers/` processed / accounted for → **96/96 covered** (Part A).
- [x] No paper omitted → confirmed; 2 summaries (P21, P98) lack a corpus PDF and are flagged.
- [x] Every load‑bearing claim traced to ≥1 paper or labeled as own‑work/projection → done (Part B).
- [x] Unverifiable claims explicitly listed → Part E.
- [x] Conflicting evidence highlighted → rows 5, 10, 15, 18; Part D‑6.
- [x] Speculation labeled → rows 1, 3, 29, 31, 39, 44, 46 flagged 🔎/⚠️.

**Bottom line:** The report's *strategy* is sound and its *qualitative* use of the literature is largely accurate (architecture descriptions, mechanisms, and the correlation‑manifold / transductive‑adaptation theses are well supported). The defects are concentrated in **quantitative comparison and citation hygiene**: one invalid accuracy‑as‑AUC baseline (WDANet), one fabricated survey quotation (P54), one misattributed ablation delta (SPD‑DANN), reliance on two papers with no primary PDF (P21, P98), un‑disclosed preprint status, and projected numbers labeled as literature‑derived. All are corrected in the revised `LODO_Strategy_Report.md`.
