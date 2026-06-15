# Cross‑Paper Synthesis & Evidence Matrix

**Scope:** Synthesis of the 96 papers in `Other Papers/` (summarized as P1–P98 in `Paper Summary.md`) with respect to the central question of `LODO_Strategy_Report.md`: *how should cross‑subject / cross‑dataset generalization be achieved and **evaluated** for EEG depression detection?*

**Reading the "Validation Strategy" column:** the literature conflates two protocols that this synthesis keeps distinct —
- **LOSO** = Leave‑One‑*Subject*‑Out (generalize to a new *subject*, same dataset).
- **LODO** = Leave‑One‑*Dataset*‑Out / cross‑dataset / cross‑site (generalize to a new *dataset*). **This is the report's target and is much rarer in the corpus.**
- **DG** = source‑only Domain Generalization (no target data). **DA/TTA** = uses unlabeled target (transductive).

Numbers are quoted only where the underlying summary/PDF gave them; "—" means not reported / not applicable. 🔎 marks a paper with **no PDF in the corpus** (P21, P98).

---

## 1. Evidence Matrix — Papers Bearing on Cross‑Subject/Cross‑Dataset Validation

| Paper | Dataset(s) | Validation Strategy | LODO (dataset‑level)? | Model | Metrics | Main Findings | Limitations |
|-------|-----------|---------------------|----------------------|-------|---------|---------------|-------------|
| **P44 WDANet** (IEEE TAC '26) | MODMA + 2 self‑collected (170 subj etc.) — **depression** | Cross‑subject **and cross‑dataset** | **Yes (cross‑dataset)** | Euclidean CNN + 3 discriminators (global/local/Wasserstein) + dynamic factor | **Accuracy** 83.3/75.5/73.9/76.0/70.9 %; F1 77.8/87.3 % — **no AUC** | Dynamic adversarial DA beats DANN/CORAL/DDA/R2G‑STNN on depression | Euclidean (no geometry); accuracy/F1 only; not strict LOSO; "MODMA AUC" does **not** exist |
| **P8 EEG‑RCformer** (Mathematics '26) | MODMA (depression) + SEED | Subject‑dep. 5‑fold; **subject‑indep. (5 held‑out)**; SEED LOSO | No (single‑dataset) | AIRM channel clustering + light Transformer | **AUC** 0.7154±0.061 sub‑indep (MODMA) | AIRM clustering +9 % over Euclidean cross‑subject; geometry matters most cross‑subject | 5‑subject hold‑out (not full LOSO); 2 datasets; no DA module |
| **P22 TSMNet/SPDDSMBN** (NeurIPS '22) | 6 MI/ERP/SSVEP datasets | Inter‑subject & inter‑session UDA (leave‑5 %‑out) | Near (multi‑dataset, subject‑level) | TSMNet + SPD domain‑specific momentum BN | Balanced acc vs domain‑specific TSM | First end‑to‑end SPD UDA to match supervised TSM; shift absorbed in SPD‑BN | Needs domain labels; train‑time only (no test‑time transduction); no psychiatric task |
| **P24 TSA** (Front. Hum. Neurosci. '22) | 18 BCI databases, 349 subj | LOSO‑style + **inter‑dataset TL** | **Partial (inter‑dataset)** | Tangent‑space affine alignment + SVM | Accuracy: **+2.7 % over RPA** | Full tangent alignment > mean recentering; broad cross‑paradigm robustness | Needs equal matrix dim; SVM/shallow; healthy BCI only |
| **P23 PT+Moments** (ICASSP '19) | High‑density EEG (unnamed) | Cross‑subject UDA | Implicit LOSO | Parallel transport + tangent moments matching | Acc > PT‑only > no‑adapt | **PT alone fails when means match but structure differs** | Small scale; noisy at 128‑ch; shallow |
| **P98 SPD‑DANN** 🔎 (Neural Networks '26) | 4 MI datasets (incl. imbalanced 4a) | Single‑source→single‑target UDA | Partial | SPDNet+MAtt + GRL discriminator on SPD; L_M, L_P, entropy pseudo‑labels | **Accuracy** (2a 44.9 %); ablation 33.4→37.3→39.8→44.9 | First SPD‑manifold adversarial DA; geometry > optimized Euclidean (+2.5); transfer tracks subject similarity | No PDF in corpus; covariance+LEM; train‑time only; acc only; MI |
| **P4 CorSW** (KDD '26) | MI / SSVEP / ERN | Cross‑subject **DG (source‑only)** | Partial | Sliced‑Wasserstein DG loss on correlation manifold | Acc: ERN +6.6 %, SSVEP +3.4 % | First SW on correlation manifold; correlation > covariance; zero inference cost | Source‑only (no target use); no depression; no cross‑dataset |
| **P6 RK‑SVM/ARK** (Neurocomputing '13) | BCI‑IV 2a | **Cross‑session** adaptation | No | Riemannian kernel SVM + adaptive reference (geometric mean) | Acc 86.0 % (ARK) | Adaptive reference (re‑estimate mean from unlabeled target) = simplest UDA | Cross‑session not cross‑subject; MI; SVM scaling |
| **P21 S‑RCT** 🔎 (Research Sq. '26) | BCI‑IV 2a (brainprint) | Cross‑session (biometric) | No | Session‑wise Riemannian centering to I | Identification acc 98.3 % | Per‑session centering ≡ rebiasing; transductive‑compatible | No PDF; preprint; biometric, not classification |
| **P50 PSDNorm** (ICLR '26) | 10 sleep datasets, 10K subj | **Leave‑dataset‑out**, cross‑site | **Yes (cross‑dataset)** | Monge‑OT temporal normalization layer | Balanced acc, F1 | Test‑time OT normalization → SoA on unseen datasets; robust to scarcity | Sleep task; feature‑space (not SPD); OT cost at 128‑ch |
| **P46 HybridRDG** (CVPR‑W '26) | Multi‑paradigm clinical EEG (ASD), <50 subj | **Zero‑shot DG**, subject‑held‑out | No (within‑study) | EEGNet branch + Riemannian (Ledoit‑Wolf, LogE) branch | Balanced acc (4/5 paradigms best) | "Geometric covariance structure central to zero‑shot EEG generalization"; Ledoit‑Wolf essential <50 subj | ASD not depression; no AUC; multi‑paradigm not cross‑dataset |
| **P45 ReTa‑Diffusion** (ICLR '26 ur) | 3 subclinical‑depression rs‑EEG sets | 5‑fold + **LOSO**, cross‑dataset | **Yes** | Bidirectional diffusion + Wavelet‑Riemannian fusion | Acc 54.5 % (multiclass) | rs→task diffusion + WRFE beats SOTA by 19 %; Riemannian features critical | Needs task‑EEG at train; multiclass; modest acc |
| **P15 fMRI correlation manifold** (KDD '26) | Multi‑site fMRI + EEG | Cross‑site/subject | Partial | Correlation‑manifold + Grassmann (OLM) | — | Correlation manifolds & eigen‑subspaces > covariance for multi‑site | fMRI‑centric; theory‑heavy |
| **P48 ADHD harmonization** (Front. Neuroinf. '22) | Multi‑site ADHD EEG | **Multi‑site harmonization** | **Yes (multi‑site)** | Riemannian FC‑matrix harmonization | — | Riemannian recentering harmonizes multi‑site connectivity | **Not cited by report**; ADHD not depression |
| **P37 Cross‑dataset variability** (Front. Hum. Neurosci. '20) | Multiple EEG DL datasets | **Cross‑dataset** | **Yes** | DL benchmark | — | Documents the cross‑dataset variability problem directly | **Cited only as P37 in summary, under‑used**; descriptive |
| **P31 Selective transfer** (Front. Neurosci. '21) | BCI‑IV 2a | LOSO | No | Tangent‑space source selection (MMD/KL) | Acc, kappa | **Negative transfer is real**; K≈5–7 best; geometry > Euclidean selection | MI; 9 subj |
| **P35 Rotation‑metric selection** (Front. Hum. Neurosci. '26) | BCI MI | Cross‑subject TL | — | Rotation‑based SPD distance for source selection | — | SPD distance picks useful source data | MI; selection‑only |
| **P36 DGFEL** (Cogn. Neurodyn. '26) | ERP BCI | **DG (calibration‑free)**, LOSO | Yes (subject) | Domain‑generalized feature embedding | Acc | Calibration‑free ERP via DG | ERP; subject‑level |
| **P25 RGT‑GAA** (Alexandria Eng. J. '26) | Cross‑subject emotion EEG | LOSO | Yes (subject) | Riemannian Graph Transformer + geodesic adversarial | Acc | Geodesic adversarial > Euclidean GAN | Emotion; subject‑level |
| **P26 Stiefel routing** (arXiv '26) | Multiple MI | Cross‑subject/domain | Yes (subject) | K Stiefel experts + routing | Acc | Adaptive subspace routing handles heterogeneity | MI; preprint |
| **P28 SFDA pseudo‑label** (MDPI '26) | EEG BCI | Source‑free DA, partial LOSO | Partial | Multicentric dynamic pseudo‑labeling | Acc | Privacy‑preserving adaptation without source data | Pseudo‑label noise; preprint |
| **P32 Self‑attn DA** (Eng. Sci. Tech. '26) | Driver drowsiness EEG | LOSO | Yes (subject) | Riemannian + contrastive DA | Acc | Pull same‑class features across domains in tangent space | Drowsiness; subject‑level |
| **P33 BARN‑DA** (MDPI '26) | MI EEG | Cross‑subject | Yes (subject) | Band‑aware Riemannian net + Riemannian MMD | Acc | Per‑band Riemannian alignment | MI; preprint |
| **P14 RGP‑VAE** (arXiv '26) | MI EEG | LOSO | Yes (subject) | Parallel‑transport VAE augmentation | Acc | PT‑based covariance augmentation for small N | Augmentation only |
| **P49 DiFFEoCFM** (NeurIPS '25) | Brain connectivity matrices | — | — | Riemannian flow matching (pullback geometry) | — | Manifold‑faithful covariance generation | Generative; not eval protocol |
| **P54 Survey: Cross‑subject DL** (arXiv/IOP '26) | Survey | — | — | Taxonomy (DA/DG/SSL/meta/Riemannian) | — | Subject‑dependent eval **inflates** acc (95→60 %); subject‑independent (LOSO/group‑kfold/holdout) required; Riemannian competitive | **No "transductive"/"gold standard" wording** (see audit row 5); preprint |
| **P69 Survey: SPD neuroimaging** (arXiv '26) | Survey | — | — | SPD‑learning taxonomy | — | SPDDSMBN = SoA cross‑subject EEG UDA; correlation manifold increasingly preferred; **test‑time Fréchet adaptation = open gap** | Preprint; same group as P22 |
| **P68 Correlation‑manifold nets** (arXiv '26) | MI/ERP + action | Cross‑subject | — | BiMap/LogEig over correlation manifold (5 geometries) | Acc | **Correlation‑manifold nets > SPD nets** cross‑subject | Preprint |
| **P3 RHOP** (ICLR '26) | TUAB/TUEV/BCIC2B/P300 | Population / within‑subject | No | Quotient‑Gaussian second‑order pooling head (~1K params) | BAcc, κ, AUROC | Scale‑invariant pooling head improves over CLS/GAP | **No LODO/cross‑subject eval**; preprint |
| **P52 Recenter (fixation)** (arXiv '26) | 8 subj online MI | Within‑subject online | No | Class‑agnostic fixation‑based recentering | **AUC** +56 %/+34 % | **Task‑based recentering biases under class imbalance**; use baseline windows | Within‑subject MI; online |
| **P17 Stiefel‑SPD GCN** (IEEE TNNLS '26) | 3 EEG datasets | Cross‑subject, 3‑dataset same‑HP | Partial | Stiefel‑SPD graph convolution end‑to‑end | Acc | End‑to‑end SPD graph learning across datasets | Subject‑level; preprint |
| **P95 Meta‑WGCNN** (MDPI '26) | Taste EEG | Cross‑subject meta‑learning, LOSO | Yes (subject) | Wavelet GCN + meta‑learning | Acc | Meta‑learning for cross‑subject EEG | Taste task |

> The remaining ~64 summaries are **foundational SPD/Riemannian theory, optimization, architecture, or non‑EEG application papers** (e.g. P65 SPDNet, P66 ARMAGNAC BN, P84–P94 geometry/optimization, P79 lane detection, P9–P11 MI methods, P59–P64 theses). They inform *method design* but do **not** add cross‑subject/cross‑dataset *validation evidence*, so they are intentionally excluded from this matrix. They remain fully summarized in `Paper Summary.md`.

---

## 2. Agreements Across Studies (consensus)

1. **Geometry beats Euclidean for cross‑subject EEG.** P8 (+9 % AIRM vs Euclidean), P98 (SPD geometry > optimized Euclidean), P46 ("geometric covariance structure is central"), P6 (Riemannian kernel > vectorized), P22, P69. *Strong, multi‑paper consensus.*
2. **Correlation (scale‑invariant) ≥ covariance for cross‑subject/cross‑site.** P3, P4, P15, P68 converge. *Strong consensus* — and the best‑supported design claim in the report.
3. **Aligning domain/subject Fréchet means is necessary.** P6 (adaptive reference), P21 (centering to I), P22 (SPD‑BN), P23 (parallel transport), P24 (tangent alignment) — five independent mechanisms for the same idea.
4. **Mean alignment alone is insufficient; structure must be aligned too.** P23 (PT+moments), P24 (full tangent alignment > recentering), P98 (L_P prototype loss beyond L_M mean loss).
5. **Using unlabeled target data (transductive/TTA) helps when available.** P6, P22, P23, P50, P98, and P54's DA‑vs‑DG framing.
6. **Small‑sample robustness needs shrinkage.** P46 (Ledoit‑Wolf essential <50 subj); echoed implicitly by the report's pipeline.
7. **Subject‑independent evaluation is mandatory; subject‑dependent inflates results.** P54 (95→60 % drop, data leakage), P8 (sub‑dep 0.98 vs sub‑indep 0.72 on MODMA), P3, P46.
8. **AUC/balanced accuracy over raw accuracy for imbalanced clinical data.** P8, P52 (both use AUC); the report's own §5.4 choice agrees.

## 3. Contradictions / Tensions Across Studies

1. **Source‑only DG vs transductive DA.** P4/P46 advocate source‑only (no target data) generalization; P6/P22/P23/P50/P98 show transductive use of target data wins. Not a true contradiction (different settings), but the report should not cite both as if uniformly endorsing its transductive thesis.
2. **Channel‑count handling.** P24 is described as handling "different channel counts naturally," yet every SPD/tangent method (P22, P23, P68, and P24 itself) requires equal matrix dimension — and the report mandates 19‑ch harmonization. The "naturally" claim is in tension with practice.
3. **Metric choice AIRM vs Log‑Euclidean.** P8/report prefer AIRM (geometrically exact); P98 and P46 deliberately use Log‑Euclidean for closed‑form means and stability. The "AIRM is always superior" stance (report §3.2 note on P21) is not unanimous.
4. **Does geometry beat *optimized* Euclidean?** P98's own ablation says geometry‑only beats *plain* Euclidean by 6.4 but *optimized* Euclidean by only 2.5 — a weaker claim than the report's headline.
5. **Best representation: covariance vs FC vs correlation.** P5 (FC via Laplace), P8 (PSD+DE), P3/P4/P68 (correlation) — no single winner; depends on signal type (cf. the report's own double‑dissociation observation).

## 4. Recurring Methodological Weaknesses (across the corpus)

- **Accuracy‑only reporting on imbalanced data** (P44, P98, P31, most MI papers) — undermines comparability for clinical AUC.
- **Subject‑level LOSO presented as the hard test**, while true **cross‑dataset** evaluation is rare (only ~P22, P24, P44, P50, P15, P48, P37).
- **Small N / few datasets** (P6, P23, P31 = 9 subj; P46 <50; P8 = 53). Wide confidence intervals rarely reported.
- **Preprint / under‑review status** for most 2026 "SOTA" claims (P3, P4, P44, P45, P46, P54, P68, P69, P98, P21).
- **Train‑time‑only adaptation** (P22, P98) — no test‑time transduction, the exact gap the report targets.
- **Single‑split evaluations** dressed as cross‑subject (P8's 5‑subject hold‑out).

## 5. Recurring Methodological Strengths

- **Ablations isolating the geometric component** (P98 Table 10, P8 AIRM‑vs‑Euclidean, P22 DSBN ablation) — clean attribution of gains.
- **Large multi‑dataset benchmarks** where present (P24 = 18 DBs, P50 = 10 datasets, P22 = 6) give credible generalization evidence.
- **Theoretical grounding** (P22 Fréchet‑mean convergence proof; P4 PEMSW unifying SW theory; P23 PT/moments decomposition).
- **Statistical testing** where done (P8 p‑values + effect size; P52 p<0.05).
- **Plug‑and‑play, low‑overhead modules** (P3 ~1K params, P4 zero inference cost, P50 drop‑in layer).

## 6. Evidence *Supporting* LODO (cross‑dataset evaluation) as the Right Test

- **P54**: subject‑dependent / leakage‑prone protocols inflate accuracy dramatically — only out‑of‑distribution evaluation is honest; cross‑dataset is the strongest such test.
- **P37** documents the cross‑dataset variability problem directly — motivating cross‑dataset evaluation.
- **P22, P24, P50** demonstrate that methods *can* be evaluated and made to work across many datasets/sites — LODO is feasible, not just punitive.
- **P44** is the proof‑of‑concept that cross‑dataset depression recognition is achievable (even if Euclidean/accuracy‑only).
- **P48, P15** show multi‑site harmonization is a tractable, well‑posed problem on the manifold.

## 7. Evidence *Against* / Cautioning LODO

- **No paper in the corpus performs strict Leave‑One‑*Dataset*‑Out depression detection with subject‑level AUC** — so empirical priors for the report's exact setting do not exist (the §9 ranges are projections).
- **P23, P31, P98** show alignment/transfer can **fail or reverse** (negative transfer; PT‑alone failure; ~30 % transfer for dissimilar pairs) — cross‑dataset shift can be too large to bridge.
- **The double‑dissociation** (report §4, own result): when source and target encode *different signal types* (connectivity vs spectral), a single model may be unable to bridge them — LODO can punish a model for a distribution gap no method closes.
- **Small N** (53–121 subj) ⇒ a single held‑out dataset gives a **high‑variance** point estimate; one LODO fold ≈ one noisy number, not a distribution.

## 8. When LODO Is Preferable vs When Alternatives Are

**Prefer LODO (cross‑dataset) when:**
- The deployment target is a *new site/dataset/montage*, not just a new subject (the report's actual goal).
- You must demonstrate that performance is **not** explained by dataset‑specific artifacts or subject‑identity leakage (P54).
- You have ≥3 datasets so that each serves as a held‑out fold and you can report a *spread*, not one number.

**Prefer LOSO / nested CV / within‑dataset when:**
- N per dataset is small and datasets are few → a single LODO fold is too high‑variance to interpret; LOSO gives more folds and tighter CIs.
- The double dissociation makes cross‑signal‑type transfer ill‑posed → first establish within‑dataset ceilings (LOSO) before attributing LODO drops to method failure.
- You need to tune hyper‑parameters honestly → use **nested** CV (inner LOSO for selection, outer LODO for reporting) to avoid optimistic bias.

**Prefer DG / TTA framing when:**
- *No* target data is available at deployment → source‑only DG (P4, P46).
- *Unlabeled* target data is available → transductive DA / test‑time adaptation (P6, P22, P23, P50, P98) — the report's chosen, well‑supported path.

---

### Provenance note
Quantitative cells were taken from the corresponding `Paper Summary.md` entries; the **WDANet (accuracy, no AUC)** and **survey (no "transductive"/"gold standard" text)** cells were additionally verified against the source PDFs. Cells marked "—" were not reported in the available summary/PDF and were **not** invented. P21 and P98 (🔎) have no PDF in the corpus and rest on their summaries only.
