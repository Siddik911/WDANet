# Paper Summaries

---

## Paper 1 — Alternatives to Sine Carrier in Auditory BCI

**Title:** Alternatives to Sine Carrier in Auditory BCI: Exploring Machine Learning Strategies for Assessing Modulation Detectability in EEG

**File:** Other Papers/EEG-BCI Classification/Alternatives_to_Sine_Carrier_in_Auditory_BCI_Exploring_Machine_Learning_Strategies_for_Assessing_Modulation_Detectability_in_EEG.pdf

**Authors:** Guého et al. (Orange Innovation / LORIA / Paris Brain Institute) | **Year:** 2026 | **Venue:** IEEE OJSP

**Dataset:** Custom experiment — 24 subjects per condition, 24-ch EEG, 4 auditory stimuli at 2 loudness levels (50 & 56 phons)

**Modality:** Physiological Signals (EEG)

---

**Research Problem — Main Objective:**
Assess whether non-sinusoidal amplitude-modulated sounds (Brownian noise, cicada song, cat's purr) can replace pure tones as BCI carrier signals for SSAEP-based auditory BCIs.

**Research Problem — Why It Matters:**
Prolonged listening to pure tones is unpleasant. If natural sounds produce equally detectable SSAEPs, they could improve user experience — but this had not been rigorously tested with proper loudness equalization and strong classifiers.

---

**Methodology — Core Idea:**
Compare 4 carrier sounds, loudness-equalized via HATS, across 10 classifiers (LDA, deep learning, Riemannian tangent-space methods) to detect modulation frequency in EEG.

**Methodology — Model Architecture:**

- Feature extraction: raw EEG epochs around SSAEP stimulation
- Backbone: 10 classifiers including LDA, CNNs, and Riemannian (TSM/tangent-space)
- Adaptation module: none (within-subject only)
- Classifier: LDA / deep networks / Riemannian TSM

**Methodology — Learning Strategy:** Supervised, within-subject

**Methodology — Key Innovation:** First study to loudness-equalize auditory BCI stimuli and benchmark with Riemannian classifiers alongside DL, at scale (24 subjects per condition).

---

**Evaluation Protocol — Dataset Split:** Within-subject binary classification per stimulus/loudness condition

**Subject Independence:** No | **Cross-Subject:** No | **LODO:** No | **Cross-Dataset:** No

**Metrics:** Classification accuracy (%)

---

**Findings — Main Results:**

- Riemannian (tangent space) classifiers consistently outperformed all others
- Pure tone: >83% accuracy | Cicada song: ~60% | Brownian noise & cat's purr: chance level
- Increasing loudness did NOT improve detection for any stimulus

**Findings — Important Observations:**

- Amplitude modulation detectability depends heavily on spectral and temporal structure of the carrier, not just loudness
- Riemannian classifiers dominate across all stimuli and conditions

**Findings — Failure Cases:**

- Brownian noise and cat's purr fully fail — unsuitable as SSAEP carriers despite being "natural"

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: Riemannian TSM as a reliable classifier even in small-N, noisy EEG settings
- Evaluation: importance of proper signal equalization before cross-condition comparison (analogous to EA normalization in cross-subject EEG)

**Knowledge Extraction — What We Should NOT Borrow:**

- Auditory BCI paradigm (SSAEP) — irrelevant to depression detection
- Stimulus design and loudness equalization methodology

---

**Relevance Score:** 2/10

**Reason:** Confirms Riemannian classifier superiority but in an auditory BCI context with no connection to depression, cross-subject learning, or transductive adaptation.

---

**Possible Integration:**

- Directly usable: nothing
- Requires modification: nothing
- Future experiment idea: none

---

## Paper 2 — DMFBTSM: Subject-Specific Discriminative Multi-Scale Filter Bank TSM

**Title:** A New Subject-Specific Discriminative and Multi-Scale Filter Bank Tangent Space Mapping Method for Recognition of Multiclass Motor Imagery

**File:** Other Papers/EEG-BCI Classification/A New Subject-Specific Discriminative and Multi-Scale Filter Bank Tangent Space Mapping Method for Recognition of Multiclass Motor Imagery.pdf

**Authors:** Wu et al. (Kunming University of Science and Technology) | **Year:** 2021 | **Venue:** Frontiers in Human Neuroscience

**Dataset:** BCI Competition IV-2a (4-class motor imagery, 9 subjects, 2 sessions)

**Modality:** Physiological Signals (EEG)

---

**Research Problem — Main Objective:**
Improve multiclass motor imagery (MI) classification by selecting subject-specific discriminative frequency bands before applying Riemannian tangent space feature extraction.

**Research Problem — Why It Matters:**
Prior TSM/Riemannian methods used fixed or generic filter banks (e.g., 8–30 Hz) identical across all subjects. Since different subjects have different informative frequency bands for MI, a fixed bank includes noise and redundant features, reducing accuracy and increasing computation.

---

**Methodology — Core Idea:**
DMFBTSM uses a non-parametric MANOVA test on each candidate frequency sub-band to identify which bands are statistically discriminative for a given subject. It builds a multi-scale filter bank from only those bands, applies Riemannian TSM per sub-band, concatenates tangent vectors, and classifies with a linear SVM.

**Methodology — Model Architecture:**

- Feature extraction: Multi-scale filter bank on raw EEG; covariance matrices per sub-band and time window
- Backbone: Riemannian TSM — projects SPD matrices to tangent space at Riemannian mean
- Adaptation module: Subject-specific band selection via MANOVA (sum-of-squared-distances) on training data
- Fusion: Concatenation of tangent vectors across all selected sub-bands and time windows
- Classifier: Linear SVM

**Methodology — Learning Strategy:** Supervised, within-subject band selection using training labels

**Methodology — Key Innovation:** Data-driven, subject-specific discriminative band selection before Riemannian mapping reduces dimensionality while improving class separability.

---

**Evaluation Protocol — Dataset Split:** Session-to-session (train session 1, test session 2); also 10-fold CV for DL comparison

**Subject Independence:** No | **Cross-Subject:** No | **LODO:** No | **Cross-Dataset:** No

**Metrics:** Classification accuracy (%), training time, test time

---

**Findings — Main Results:**

- Average 4-class accuracy: **77.33 ± 12.3%** (session-to-session)
- +2.56% over MFBTSM (best TSM baseline) | +3.36% over Supervised FGMDRM | +2.58% over deep ConvNet
- Training 3× faster, inference 2× faster than MFBTSM

**Findings — Important Observations:**

- Subject-specific band selection is the dominant driver; multi-scale aspect provides secondary gains
- Riemannian TSM outperforms deep learning even without large datasets — relevant to small-N depression settings
- Linear SVM sufficient when Riemannian feature space is well-structured

**Findings — Failure Cases:**

- Band selection uses the test subject's own training labels — cannot generalize to unseen subjects
- High inter-subject variance (±12.3%) suggests sensitivity to subject-specific SNR

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: Multi-scale filter bank + Riemannian TSM as a lightweight, interpretable baseline for depression EEG
- Training strategy: Data-driven frequency band selection before covariance extraction — reframe as population-level selection across LODO training subjects
- Adaptation: MANOVA-based band selection → choose bands discriminative across training subjects, not per individual
- Representation: Tangent space concatenation across multiple frequency bands as multi-view Riemannian representation
- Evaluation: Report per-subject accuracy + mean ± std to expose inter-subject variability

**Knowledge Extraction — What We Should NOT Borrow:**

- Subject-specific band selection using test subject's own labels — unavailable in LODO
- Motor imagery task design and 4-class label structure
- Session-to-session protocol — too weak for cross-subject generalization
- SVM may not scale to larger cross-subject feature spaces

---

**Relevance Score:** 4/10

**Reason:** Methodologically useful for band-selection idea and as Riemannian baseline, but entirely within-subject motor imagery. Needs significant redesign for cross-subject depression detection.

---

**Possible Integration:**

- Directly usable: Multi-scale filter bank + per-band Riemannian TSM as feature extraction backbone
- Requires modification: Band selection via MANOVA must be redesigned as population-level selection across training subjects in LODO folds
- Future experiment: Ablation — fixed vs. population-level discriminative filter bank in LODO depression setup

---

## Paper 3 — RHOP: Riemannian High-Order Pooling for Brain Foundation Models

**Title:** Riemannian High-Order Pooling for Brain Foundation Models

**File:** Other Papers/EEG-BCI Classification/2313_Riemannian_High_Order_Poo.pdf

**Authors:** Chen Hu, Ziheng Chen, Rui Wang, Yefeng Zheng, Nicu Sebe (Jiangnan University / Westlake University / University of Trento) | **Year:** 2026 | **Venue:** ICLR 2026

**Dataset:**

- TUAB (EEG abnormality detection, 409K segments, 23-ch, 256 Hz)
- TUEV (EEG event classification, 6-class, 112K segments, 23-ch)
- BCIC2B (2-class motor imagery, 10 subjects, 3-ch, 250 Hz)
- PhysioP300 (P300 speller ERP)

**Modality:** Physiological Signals (EEG)

---

**Research Problem — Main Objective:**
Replace the weak classification head (single CLS token or global average pooling) in EEG foundation models with a geometry-aware second-order pooling head that respects SPD manifold structure.

**Research Problem — Why It Matters:**
EEG foundation models (BIOT, LaBraM) invest heavily in transformer backbones but use trivial pooling at the classifier — discarding second-order statistics and spatiotemporal covariance structure critical for EEG decoding. Standard Global Covariance Pooling (GCP) flattens all tokens into a single covariance, losing temporal structure.

---

**Methodology — Core Idea:**
RHOP is a plug-and-play pooling head with three stages: (1) each transformer token is embedded as a *quotient Gaussian* — a scale-invariant SPD representation of its mean and normalized covariance (correlation matrix); (2) these per-token SPD matrices are aggregated via a *Riemannian Gaussian* (Fréchet mean + covariance on SPD manifold) into a single SPD descriptor; (3) sparse inverse covariance estimation (iSICE) extracts a compact precision vector fused with the CLS token for classification.

**Methodology — Model Architecture:**

- Feature extraction: Any EEG foundation backbone (BIOT / LaBraM) → token features X ∈ R^{D×T×N}
- Backbone: Frozen or fine-tuned transformer (BIOT / LaBraM-Base)
- RHOP head:
  - Quotient Gaussian Embedding: per-token (µ, Σ) → correlation matrix C → SPD matrix Y_n (Eq. 11)
  - Riemannian Gaussian Embedding: {Y_n} → Fréchet mean Y_m + Riemannian covariance Y_c → SPD descriptor G (Eq. 9)
  - iSICE + utvec: sparse precision vector g from G
- Fusion: Concatenate CLS token y_0 with g → linear + softmax
- Classifier: Linear layer (~1K additional parameters)

**Methodology — Learning Strategy:** Supervised — full fine-tuning, linear probing, and from-scratch training all tested

**Methodology — Key Innovation:** Quotient Gaussian embedding normalizes per-token covariances to remove scale variation, then aggregates via Riemannian Gaussian — jointly encoding first- and second-order spatiotemporal statistics in a scale-invariant, geometry-aware pooling head adding only ~1K parameters.

---

**Evaluation Protocol — Dataset Split:** Standard dataset-specific splits per backbone paper (TUAB/TUEV follow BIOT protocol; BCIC2B session-based; PhysioP300 standard)

**Subject Independence:** Partial (TUAB/TUEV population-level; BCIC2B within-subject) | **Cross-Subject:** Partial | **LODO:** No | **Cross-Dataset:** No

**Metrics:** Balanced Accuracy, Cohen's Kappa, Weighted F1 (TUEV); Balanced Accuracy, AUC-PR, AUROC (TUAB); Accuracy (BCIC2B, PhysioP300)

---

**Findings — Main Results:**

- BIOT+RHOP (non-pretrained) on TUEV: **53.55%** balanced acc vs 46.82% baseline (+6.73%)
- BIOT+RHOP (pretrained) on TUEV: **55.72%** vs 52.81% (+2.91%)
- LaBraM+RHOP on TUEV: **63.80%** vs 64.09% (marginal, within noise)
- TUAB: modest but consistent gains across BIOT variants
- RHOP: ~1K params, 0.53–2.25 min/epoch vs 4–14 min for iSQRT-COV / SVD-Padé
- iSQRT-COV and SVD-Padé consistently *hurt* performance despite high compute

**Findings — Important Observations:**

- Scale variation across EEG temporal segments is a real problem — quotient Gaussian normalization is the key fix
- One-step Fréchet mean approximation (Karcher flow iterations = 1) is sufficient and cheap
- RHOP is the only GCP head that consistently improves over baseline across all backbones and training regimes
- Joint encoding of mean + normalized covariance (first + second order) outperforms second-order alone

**Findings — Failure Cases:**

- LaBraM+RHOP on TUEV: marginal gain — strong pretrained backbone may saturate the pooling improvement
- No cross-subject / LODO evaluation; all benchmarks are population-level or within-subject
- No depression-specific evaluation

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: RHOP as a drop-in pooling head on top of any transformer encoder for depression EEG — replaces CLS-only with geometry-aware second-order statistics
- Architecture: Quotient Gaussian embedding — normalizing covariance to correlation form removes amplitude/scale confounds across subjects, directly relevant for cross-subject depression (subjects have different EEG amplitudes)
- Training: Linear probing — freeze pretrained backbone, train only RHOP head — fast cross-subject adaptation in low-data LODO scenarios
- Adaptation: ~1K params → can be re-trained per LODO fold with minimal overfitting risk; scale-invariant embedding reduces inter-subject covariance shift
- Representation: Riemannian Gaussian aggregation across tokens captures higher-order spatiotemporal interactions beyond mean pooling
- Fusion: Concatenate SPD statistical descriptor with semantic CLS token — combining global semantics with local second-order geometry
- Evaluation: Report Balanced Accuracy + Cohen's Kappa for imbalanced depression datasets

**Knowledge Extraction — What We Should NOT Borrow:**

- Full foundation model pretraining pipeline (BIOT/LaBraM) — too expensive for a focused depression model
- iSQRT-COV and SVD-Padé pooling — consistently hurt EEG classification
- Population-level (non-LODO) evaluation — not honest for subject-independent depression detection

---

**Relevance Score:** 7/10

**Reason:** RHOP is directly usable as a classification head in a cross-subject depression model. The quotient Gaussian embedding is especially valuable — it removes inter-subject amplitude variation via scale normalization, addressing a key source of domain shift in cross-subject EEG. Plug-and-play with ~1K parameters makes LODO integration trivial.

---

**Possible Integration:**

- Directly usable: RHOP head on top of our feature encoder; quotient Gaussian normalization as scale-invariant SPD preprocessing
- Requires modification: Riemannian Gaussian aggregation needs handling of variable token counts across subjects/sessions; iSICE lambda needs tuning per LODO fold
- Future experiment: Test RHOP as classifier head in LODO depression — compare vs. tangent-space SVM and standard linear head; does second-order pooling help when per-subject training data is small?

---

## Paper 4 — CorSW: Sliced-Wasserstein Domain Generalization on Correlation Matrices

**Title:** A Sliced-Wasserstein Framework on Correlation Matrices for EEG Decoding

**File:** Other Papers/EEG-BCI Classification/A Sliced-Wasserstein Framework on Correlation Matrices for EEG Decoding.pdf

**Authors:** Chen Hu, Rui Wang et al. (Westlake University / Jiangnan University / Sun Yat-sen University) | **Year:** 2026 | **Venue:** KDD 2026

**Dataset:**

- Motor Imagery (MI) — cross-subject
- SSVEP — cross-subject
- ERN (Error-Related Negativity) — cross-subject

**Modality:** Physiological Signals (EEG)

---

**Research Problem — Main Objective:**
Develop a geometrically principled, computationally efficient domain generalization framework for cross-subject EEG decoding by measuring and minimizing distribution shift between subjects using a Sliced-Wasserstein discrepancy defined natively on the manifold of full-rank correlation matrices.

**Research Problem — Why It Matters:**
Covariance-based EEG representations are sensitive to channel-wise amplitude scaling (inter-subject differences). Correlation matrices are scale-invariant alternatives, but no SW distance existed for the correlation manifold. SPD-SW methods cannot be applied directly since correlation matrices form a distinct quotient manifold. Full Wasserstein OT is O(n³ log n) — too expensive for training-time regularization.

---

**Methodology — Core Idea:**
Builds a unified theory — **PEMSW (Pullback Euclidean Metric Sliced Wasserstein)** — showing SW on any manifold with a global diffeomorphism reduces to Euclidean SW in the embedding space (no OT solver needed). Instantiates this on the correlation manifold under **OLM** and **LSM** metrics → **CorSW**. Used as a DG regularizer: align each source subject's distribution of correlation matrices toward a shared Gaussian reference on the manifold during training only; zero overhead at inference.

**Methodology — Model Architecture:**

- Feature extraction: Multi-channel EEG windows → full-rank correlation matrices
- Backbone: CorAtt (attention on correlation manifold) or standard EEG models (EEGNet, ShallowCNet, etc.)
- Adaptation module: CorSW DG loss — SW distance from each source subject's correlation distribution to a shared Gaussian reference; training regularizer only (Algorithm 1)
- No adaptation at test time (source-only DG)
- Classifier: Softmax linear head

**Methodology — Learning Strategy:** Supervised + domain generalization regularization (source-only, no target data)

**Methodology — Key Innovation:** First SW distance on the full-rank correlation manifold. Proves SPDSW is a special case of PEMSW — unifying framework. CorSW as a training regularizer with zero inference cost. Scale-invariant representation + geometry-aware distributional alignment.

---

**Evaluation Protocol — Dataset Split:** Cross-subject DG — each subject is a domain; source-only (no target subject data used at any point)

**Subject Independence:** Yes | **Cross-Subject:** Yes | **LODO:** Partial (source-only DG) | **Cross-Dataset:** No

**Metrics:** Classification accuracy (%) ± std over 10 runs

---

**Findings — Main Results:**

- ERN task: CorAtt+CorSW(LSM) → **81.33 ± 1.14%** vs 74.71% baseline — **+6.62%**
- SSVEP: 68.58 ± 1.01% vs 65.19% — +3.39%
- MI: 75.49 ± 2.34% vs 74.71% — modest gain
- CorSW-LSM slightly outperforms CorSW-OLM on most tasks
- CorSW reduces variance (std) as well as mean — more stable cross-subject generalization

**Findings — Important Observations:**

- Correlation matrices consistently outperform raw covariance in cross-subject settings
- CorSW is fast: matrix log + 1D Wasserstein via sorting — no OT solver
- SPDSW is a special case — CorSW is the correct generalization to the correlation quotient manifold
- Aligning each subject to a shared reference (not pairwise) scales to large N

**Findings — Failure Cases:**

- Source-only DG — ignores unlabeled target subject data; suboptimal vs. transductive/TTA
- No depression datasets evaluated
- No strict LODO across different datasets

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: Correlation matrices (not covariance) as primary EEG feature — scale-invariant, reduces inter-subject amplitude shift (Papers 3 and 5 now both converge on this — strong evidence)
- Adaptation: CorSW-LSM as plug-and-play DG loss in LODO training — align each training subject's correlation distribution to a shared manifold reference; zero inference cost
- Training: Reference distribution approach (shared Gaussian on the manifold) scales to large N; no pairwise subject matching needed
- Evaluation: Report accuracy ± std over multiple runs to assess cross-subject stability

**Knowledge Extraction — What We Should NOT Borrow:**

- Source-only DG assumption — our transductive setting has unlabeled target data; we should exploit it
- Full PEMSW theoretical apparatus — use CorSW as a practical training loss

---

**Relevance Score:** 8/10

**Reason:** Directly addresses cross-subject EEG DG using scale-invariant correlation matrices on Riemannian manifolds — the exact feature type and problem setting for depression LODO. CorSW-LSM is plug-and-play as a training regularizer with zero inference overhead. The correlation > covariance finding (confirmed by Papers 3 and 5) is a key design decision. Score is 8 not higher because it is source-only DG — our transductive setting can do better.

---

**Possible Integration:**

- Directly usable: CorSW-LSM as a subject alignment regularizer in the LODO training loop
- Directly usable: Full-rank correlation matrices as the primary EEG feature representation
- Requires modification: Extend CorSW from source-only to transductive — use unlabeled target subject's correlations as second reference and minimize CorSW between source and target
- Future experiment: Source-only CorSW-DG vs. Transductive CorSW in LODO depression — does adding unlabeled target correlation matrices into the alignment improve over source-only?

---

## Paper 5 — FC-Riemannian: Functional Connectivity Features + SPDNet for EEG Emotion Recognition

**Title:** A study on the combination of functional connection features and Riemannian manifold in EEG emotion recognition

**File:** Other Papers/EEG-BCI Classification/A study on the combination of functional connection features and Riemannian manifold in EEG emotion recognition.pdf

**Authors:** Wu, Ouyang et al. (Anhui University / Civil Aviation Flight University of China) | **Year:** 2024 | **Venue:** Frontiers in Neuroscience

**Dataset:** SEED (3-class emotion: positive/negative/neutral, 15 subjects, 62-ch EEG, 15 sessions)

**Modality:** Physiological Signals (EEG)

---

**Research Problem — Main Objective:**
Combine four functional connectivity (FC) features — PLV, PCC, COH, and MI — with Riemannian SPDNet deep learning to improve EEG emotion recognition, solving the problem that FC features are not naturally SPD and therefore cannot be fed into Riemannian networks.

**Research Problem — Why It Matters:**
Riemannian manifold methods are robust for BCI but require SPD inputs. Functional connectivity features (inter-channel phase/correlation measures) capture spatial brain network information relevant to emotion but produce semi-positive definite or indefinite matrices — incompatible with SPD-based methods. No prior work had systematically made FC features SPD and passed them through SPDNet.

---

**Methodology — Core Idea:**
Transform each FC matrix (PLV/PCC/COH/MI) into an undirected graph, compute its Laplace matrix, and apply a max operator to guarantee positive definiteness. Feed the resulting SPD matrices into SPDNet (BiMap + ReEig + LogEig layers) for deep spatial feature extraction. Apply decision-level fusion across all four feature streams for final emotion classification.

**Methodology — Model Architecture:**

- Feature extraction: 4 FC features computed from 62-ch EEG → Laplace matrix + max operator → SPD matrices
- Backbone: SPDNet — 4 BiMap layers, 4 ReEig layers, 1 LogEig layer (maps SPD to tangent/Euclidean space)
- Fusion: Decision-level fusion of predictions from 4 separate FC feature streams
- Classifier: Fully connected layer + softmax

**Methodology — Learning Strategy:** Supervised, subject-dependent within-subject cross-validation

**Methodology — Key Innovation:** Laplace matrix + max operator as a principled SPD transformation for graph-based FC features — enables any connectivity measure to be fed into Riemannian deep networks.

---

**Evaluation Protocol — Dataset Split:** Subject-dependent cross-validation on SEED (sessions split within each subject — not cross-subject)

**Subject Independence:** No | **Cross-Subject:** No | **LODO:** No | **Cross-Dataset:** No

**Metrics:** Classification accuracy (%)

---

**Findings — Main Results:**

- Best single feature: PCC → highest accuracy among the four
- Best fusion: PLV + PCC + COH → **91.05%** accuracy
- All four features fused: 90.16%
- PLV, PCC, COH perform similarly — suggests stable functional connectivity patterns across subjects during emotion elicitation

**Findings — Important Observations:**

- Optimal thresholds for FC feature binarization are stable within a fixed interval across subjects — suggests threshold is a robust hyperparameter, not subject-specific
- Decision-level fusion consistently improves over single-feature models
- MI underperforms relative to the other three — noise-sensitive and harder to estimate from short EEG segments

**Findings — Failure Cases:**

- Subject-dependent only — no cross-subject or LODO evaluation; performance may drop significantly in subject-independent settings
- SEED only — small dataset (15 subjects), limited generalizability assessment
- No comparison to covariance-based Riemannian methods (only FC features tested)

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: Laplace matrix + max operator for converting any FC matrix into SPD — allows using PLV, PCC, COH connectivity features in Riemannian networks for depression EEG (these features capture inter-channel brain network dynamics relevant to depression)
- Architecture: Decision-level fusion of multiple FC feature streams — a simple ensemble that improves robustness without complex fusion modules
- Representation: PCC (Pearson correlation) as the single best FC feature for emotion — relevant for depression since PCC is easy to compute and performs comparably to more complex measures
- Training: Fixed threshold for FC feature binarization is stable across subjects — can use a global threshold tuned on training subjects in LODO without per-subject calibration

**Knowledge Extraction — What We Should NOT Borrow:**

- Subject-dependent evaluation — our goal is cross-subject LODO
- SPDNet architecture alone (BiMap + ReEig + LogEig) without any domain adaptation module — insufficient for cross-subject generalization
- MI as a FC feature — noise-sensitive, computationally expensive, and underperforms

---

**Relevance Score:** 5/10

**Reason:** Emotion recognition on EEG using functional connectivity + Riemannian geometry is directly relevant to our problem space (affect → depression). The Laplace+max trick to make FC features SPD is a useful preprocessing tool. However, the evaluation is strictly within-subject with no cross-subject generalization — the method as-is is not deployable for LODO depression detection. The core value is the FC→SPD conversion technique and the finding that PCC/PLV/COH are the most informative FC features.

---

**Possible Integration:**

- Directly usable: Laplace matrix + max operator to convert PCC/PLV/COH connectivity matrices into SPD — use as alternative feature input to our Riemannian backbone
- Directly usable: Decision-level fusion of multiple FC feature streams as an ensemble classifier
- Requires modification: Add cross-subject domain adaptation (e.g., CorSW alignment from Paper 4) on top of the FC-SPD feature pipeline for LODO deployment
- Future experiment: Compare covariance-based SPD features vs. FC-derived SPD features (PCC-Laplace, PLV-Laplace) in LODO depression — which captures more subject-invariant depression biomarkers?

---

## Paper 6 — RK-SVM: Riemannian Kernel SVM for Covariance Matrix Classification in BCI

**Title:** Classification of covariance matrices using a Riemannian-based kernel for BCI applications

**File:** Other Papers/EEG-BCI Classification/Classification of covariance matrices using a Riemannian-based kernel for BCI applications.pdf

**Authors:** Barachant, Bonnet, Congedo, Jutten (CEA-LETI / GIPSA-lab, CNRS Grenoble) | **Year:** 2013 | **Venue:** Neurocomputing

**Dataset:** BCI Competition IV Dataset IIa (4-class motor imagery, 9 subjects, 22-ch EEG, 2 sessions)

**Modality:** Physiological Signals (EEG)

---

**Research Problem — Main Objective:**
Derive a principled Riemannian-based kernel for SVM classification of spatial covariance matrices extracted from EEG, replacing the traditional CSP+LDA pipeline by operating directly on the SPD manifold.

**Research Problem — Why It Matters:**
Standard approaches (CSP + log-variance + LDA) handle covariance matrices in Euclidean space, ignoring the curved geometry of the SPD manifold. Vectorizing covariance matrices distorts distances and violates the Gaussian assumption of LDA. A kernel that respects Riemannian geometry should yield better classification.

---

**Methodology — Core Idea:**
Define a Riemannian kernel k_R(C_i, C_j; C_ref) as the Frobenius inner product of the tangent space projections of two covariance matrices at a reference point C_ref (their geometric mean). This is equivalent to log-Euclidean inner product after whitening. Feed the half-vectorized tangent projections into a linear SVM. For cross-session adaptation, update C_ref using the geometric/arithmetic mean of the new session's data — equivalent to adaptive whitening, correcting session-to-session distribution shift.

**Methodology — Model Architecture:**

- Feature extraction: EEG trial X → sample covariance matrix C (8–35 Hz bandpass, no spatial filter)
- Backbone: Riemannian kernel k_R at geometric mean C_ref → tangent space projection → half-vectorization
- Adaptation module: Adaptive Reference Kernel (ARK) — re-estimate C_ref from test session data (unsupervised); equivalent to whitening the test session independently
- Classifier: Linear SVM on tangent vectors

**Methodology — Learning Strategy:** Supervised; unsupervised session adaptation (ARK-SVM)

**Methodology — Key Innovation:**
(1) First use of a Riemannian-based kernel for EEG covariance classification. (2) Adaptive reference point update for cross-session transfer — a simple, principled, and unsupervised domain shift correction that predates modern alignment methods.

---

**Evaluation Protocol — Dataset Split:** Session-to-session (train session 1, test session 2); evaluate all 6 binary pairs of 4 MI classes

**Subject Independence:** No | **Cross-Subject:** No | **LODO:** No | **Cross-Dataset:** No

**Metrics:** Binary classification accuracy (%) ± std per pair and averaged

---

**Findings — Main Results:**

- ARK-SVM (C_ref = G): **86.0 ± 4.0%** mean across all pairs — best overall
- RK-SVM (C_ref = G): 83.9% — better than CSP+LDA (no adaptive kernel needed if sessions are similar)
- CSP+LDA: ~79.9% mean (only 5/6 pairs reported)
- SVM vec (Euclidean vectorization): 75.9% — worst — confirms Euclidean treatment of covariance is suboptimal
- Geometric mean as reference consistently outperforms arithmetic mean and identity

**Findings — Important Observations:**

- Adaptive reference update (re-estimating C_ref from test session) corrects session-to-session mean shift — Figure 2 shows near-perfect distribution alignment after adaptation
- The kernel approach is equivalent to: (1) whitening all trials by C_ref, then (2) applying log-Euclidean inner product — simple to implement
- Geometric mean of training covariances is the optimal reference point (better local manifold approximation than arithmetic mean or identity)
- No spatial filter (CSP) needed — covariance matrix alone + Riemannian geometry suffices

**Findings — Failure Cases:**

- Session-to-session only — no cross-subject evaluation; adaptation updates C_ref using session data, not subject data
- Small dataset (9 subjects), motor imagery only
- SVM with Riemannian kernel may not scale well to large feature dimensions or many training samples

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: Riemannian kernel + tangent space projection at geometric mean as a foundational feature pipeline — this is the core of nearly all SPD-based EEG methods; confirmed as superior to Euclidean vectorization
- Adaptation: Adaptive reference point — re-estimate C_ref from the test subject's (unlabeled) data before classification. In LODO: compute geometric mean of test subject's covariances and re-center the tangent space projection. This is a zero-label, zero-parameter unsupervised adaptation that directly addresses subject-specific mean shift — exactly the kind of transductive step we need
- Representation: Geometric mean as the manifold reference point for tangent space projection — use population geometric mean (all training subjects) as C_ref during training; update to target subject's geometric mean at test time
- Training: Whitening interpretation — ARK-SVM is equivalent to whitening each session/subject independently before linear classification; this motivates using per-subject whitening as a preprocessing step in LODO

**Knowledge Extraction — What We Should NOT Borrow:**

- Motor imagery task and binary classification pairs
- Session-to-session evaluation — our LODO setting is cross-subject, not cross-session
- SVM classifier — limited scalability; modern alternatives (tangent space + logistic regression, or deep Riemannian networks) are preferred for cross-subject settings

---

**Relevance Score:** 7/10

**Reason:** This is a foundational paper — it establishes the Riemannian kernel / tangent space SVM pipeline that underlies most modern EEG covariance-based methods. The adaptive reference point trick (re-estimate C_ref from test data) is directly applicable to our LODO transductive setting: computing the geometric mean of the target subject's unlabeled EEG covariances and re-centering the tangent space is a simple, principled, zero-label adaptation step. Every paper in this collection builds on these ideas.

---

**Possible Integration:**

- Directly usable: Adaptive C_ref update from target subject's unlabeled covariances as a transductive alignment step in LODO — compute geometric mean of target covariances, re-project training tangent vectors; zero labels required
- Directly usable: Geometric mean as the reference for tangent space projection (population mean from all training subjects during training; target mean at test time)
- Requires modification: SVM classifier should be replaced with a more scalable cross-subject classifier (e.g., logistic regression on tangent space, or RHOP head from Paper 3)
- Future experiment: Compare adaptive C_ref (geometric mean of target subject) vs. no adaptation vs. CorSW alignment (Paper 4) in LODO depression — baseline comparison of simple whitening-based vs. distributional alignment approaches

---

## Paper 7 — DCCA-Riemannian-MDM: Detrended Cross-Correlation + Riemannian Geometry for BCI

**Title:** Combining detrended cross-correlation analysis with Riemannian geometry-based classification for improved brain-computer interface performance

**File:** Other Papers/EEG-BCI Classification/Combining detrended cross-correlation analysis with Riemannian geometry-based classification for improved brain-computer interface performance.pdf

**Authors:** Racz, Kumar et al. (UT Austin / Semmelweis University) | **Year:** 2024 | **Venue:** Frontiers in Neuroscience

**Dataset:**

- In-house offline MI dataset: 18 subjects, 22-ch EEG, 512 Hz, 2 sessions (left vs. right hand MI)
- Online MI demonstration: 8 subjects (same paradigm, single-day)
- PhysioNet EEG Motor Imagery dataset: 103 subjects, 64-ch, 160 Hz (2-class and 4-class MI)

**Modality:** Physiological Signals (EEG)

---

**Research Problem — Main Objective:**
Replace the sample covariance matrix (SCM) with a DCCA (Detrended Cross-Correlation Analysis) matrix as input to Riemannian geometry-based classification, to reduce the impact of local non-stationarities in EEG that corrupt standard covariance estimation.

**Research Problem — Why It Matters:**
Riemannian geometry-based classifiers (MDM, RK-SVM) operate on covariance matrices, but covariance estimation itself is biased by local non-stationarities (slow drifts, regional trends) ubiquitous in EEG. Domain adaptation (rebiasing) corrects for cross-session/subject mean shifts after covariance is estimated — but cannot fix corruption in the estimate itself. DCCA removes local trends before covariance computation, yielding a cleaner SPD feature.

---

**Methodology — Core Idea:**
DCCA computes cross-covariance between EEG channel pairs after local polynomial detrending within a sliding window of scale s. The resulting pairwise DCCA matrix is SPD (sum/average of SPD matrices), capturing the detrended functional connectivity structure. This DCCA matrix replaces the SCM as input to the standard Riemannian MDM classifier. A real-time implementation (rtDCCA) enables online BCI use. Rebiasing (re-centering all matrices by their global Riemannian mean) is applied for domain adaptation.

**Methodology — Model Architecture:**

- Feature extraction: 8–30 Hz bandpass → 1-s sliding windows → pairwise rtDCCA matrix (22×22 SPD) at scale s=128
- Backbone: MDM (Minimum Distance to Mean) classifier on Riemannian manifold — computes per-class Riemannian mean, assigns new sample to nearest class prototype
- Adaptation module: Rebiasing (Zanini et al., 2017) — re-center all matrices by global Riemannian mean R: C_i^rebias = R^{-1/2} C_i R^{-1/2}; adaptive variant updates R causally from incoming test data
- Classifier: MDM (nearest Riemannian mean per class)

**Methodology — Learning Strategy:** Supervised with unsupervised rebiasing adaptation; within-subject session-to-session

**Methodology — Key Innovation:** DCCA as a non-stationarity-robust covariance estimator for Riemannian BCI pipelines; real-time rtDCCA formula enables online deployment; DCCA matrices capture fractal functional connectivity structure absent from SCM.

---

**Evaluation Protocol — Dataset Split:** Session-to-session (offline: train session 1, test session 2); LORO-CV for single-session dataset; online real-time deployment for 8 subjects

**Subject Independence:** No | **Cross-Subject:** No | **LODO:** No | **Cross-Dataset:** No

**Metrics:** Sample-wise accuracy, Cohen's κ; normalized command delivery κ (online)

---

**Findings — Main Results:**

- Rebiased DCCA-MDM vs. Rebiased SCM-MDM (in-house, s=128): κ **0.2316 ± 0.1823** vs. 0.2190 ± 0.1825 (p=0.032, FDR-adjusted) — statistically significant improvement
- DCCA outperforms CSP+LDA with rebiasing (p=0.106 trend)
- Online BCI (8 subjects): command delivery κ_norm = **0.7409 ± 0.1515** (substantial); sample-wise κ = 0.5200 ± 0.1610 (moderate) — confirmed real-time feasibility on 22-ch EEG
- Detrending scale matters: s=32–64 hurts performance (acts as high-pass filter removing signal); s=128–512 improves consistently — plateau above s=128
- Post-hoc: DCCA reveals contralateral connectivity increase during MI — neurophysiologically valid

**Findings — Important Observations:**

- Rebiasing is critical: without it, DCCA advantage over SCM-MDM is non-significant; with it, consistently significant
- DCCA at s=128 captures ~250ms local trends — acts as a detrending window that removes slow drifts without removing mu/beta band MI content
- Adaptive rebiasing (online, causal) performs nearly as well as batch rebiasing — usable in real-time
- DCCA matrix is naturally SPD — no special transformation needed to use with Riemannian methods

**Findings — Failure Cases:**

- Gains are modest in absolute terms (Δκ ≈ 0.013) — DCCA is an incremental improvement over SCM, not a paradigm shift
- Within-subject session-to-session only — no cross-subject evaluation; rebiasing does not address subject-to-subject shift
- Scale s must be tuned; wrong scale degrades performance

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: DCCA matrix as a non-stationarity-robust alternative to SCM for Riemannian feature extraction in depression EEG — EEG depression recordings often have long slow drifts (fatigue, drowsiness, baseline wander) that DCCA's local detrending would remove
- Adaptation: Rebiasing (Zanini 2017) as a subject-level centering step: compute Riemannian mean of all training matrices → use as reference to re-center each subject's matrices → reduces cross-subject mean shift. This is complementary to CorSW (Paper 4) — rebiasing removes mean shift, CorSW aligns full distributions
- Representation: DCCA at scale s=128 captures detrended covariance at ~250ms — a principled way to remove low-frequency confounds before computing cross-channel coupling
- Training: Adaptive rebiasing — causally update the reference matrix from incoming unlabeled data → natural transductive step for LODO: update rebiasing reference from target subject's unlabeled EEG

**Knowledge Extraction — What We Should NOT Borrow:**

- MDM classifier — too simple for cross-subject depression; useful as a baseline
- Motor imagery task and session-to-session evaluation
- DCCA's fractal scaling analysis — neurophysiology post-hoc only, not needed for the classification pipeline
- Scale tuning overhead — use s=128 as default (250ms) based on this paper's finding

---

**Relevance Score:** 5/10

**Reason:** DCCA as a non-stationarity-robust covariance estimator is a useful addition to the feature extraction pipeline for depression EEG (which has prominent slow drifts). The rebiasing adaptation technique is directly useful for LODO (combined with adaptive updating from target subject data). However, the paper is strictly within-subject motor imagery, gains are modest, and the main contributions are engineering rather than methodological breakthroughs for cross-subject generalization.

---

**Possible Integration:**

- Directly usable: Rebiasing (Zanini 2017) as a preprocessing step — compute population Riemannian mean from all training subjects and re-center all matrices before training; update reference from target subject's unlabeled covariances at test time (transductive)
- Directly usable: DCCA matrix (s=128) as an alternative to SCM in the feature extraction pipeline — especially useful if depression datasets have pronounced slow EEG drifts
- Requires modification: rtDCCA scale s needs tuning for each depression dataset's sampling rate and epoch length
- Future experiment: SCM-based vs. DCCA-based covariance features in LODO depression — does detrending before covariance estimation improve cross-subject generalization, or is CorSW alignment sufficient to compensate?

---

## Paper 8 — EEG-RCformer: Riemannian Channel Clustering Transformer for Depression and Emotion Decoding (Relevance: 9/10)

**Title:** Decoding Cognitive States via Riemannian Geometry-Informed Channel Clustering for EEG Transformers
https://github.com/floydfeng-coding/EEG-RCFormer
**File:** Other Papers/EEG-BCI Classification/Decoding Cognitive States via Riemannian Geometry-Informed Channel Clustering for EEG Transformers.pdf

**Authors:** Feng, Yan (City University of Macau) | **Year:** 2026 | **Venue:** Mathematics (MDPI)

**Dataset:**

- MODMA (53 subjects, 128-ch resting-state EEG, MDD vs. normal control — depression dataset)
- SEED (15 subjects, 62-ch EEG, 3-class emotion recognition)

**Modality:** Physiological Signals (EEG)

---

**Research Problem — Main Objective:**
Design a Transformer for EEG decoding that reduces channel redundancy via Riemannian geometry-informed channel clustering, enabling computationally efficient and geometrically principled token construction for cognitive state (depression, emotion) classification.

**Research Problem — Why It Matters:**
Standard EEG Transformers treat each electrode as an independent token — quadratic attention cost with 64–128 channels, and nearby electrodes are partially redundant due to volume conduction. Purely anatomical grouping is too static; purely feature-driven clustering is unstable across subjects and trials. No prior method combined geometry-aware functional clustering with stable spatial priors for EEG token construction.

---

**Methodology — Core Idea:**
EEG-RCformer has three stages: (1) Extract per-channel PSD+DE features over 1s sliding windows; compute channel-wise SPD covariance matrices from these features. (2) Hybrid clustering: maintain K stable spatial centers (updated by EMA from batch-level AIRM-based functional centroid proposals using farthest-point sampling on the SPD manifold); assign all C channels to K balanced clusters. (3) Average-pool each cluster to get K tokens → standard Transformer encoder → global max pooling → MLP classifier.

**Methodology — Model Architecture:**

- Feature extraction: 0.5–50 Hz bandpass → 1s windows (80% overlap) → 7-band PSD + DE → z-score normalization → feature tensor F ∈ R^{C×Nw×14}
- Backbone: Lightweight Transformer (d_model=128, 4 heads, 3 layers, FFN=256, dropout=0.5)
- Adaptation module: Hybrid channel clustering — AIRM farthest-point sampling for functional centroid proposals; EMA update of stable spatial centers; balanced capacity assignment (K=5 clusters default)
- Token construction: Average pooling within each cluster → K tokens
- Classifier: Global max pooling + MLP

**Methodology — Learning Strategy:** Supervised; subject-dependent (5-fold CV within subject) and subject-independent (5 held-out subjects for MODMA; LOSO for SEED)

**Methodology — Key Innovation:** Hybrid AIRM + spatial EMA clustering reduces tokens from C to K≪C while preserving both stable anatomical structure and trial-adaptive functional geometry — bridges handcrafted Riemannian features with Transformer representation learning.

---

**Evaluation Protocol — Dataset Split:**

- Subject-dependent: 5-fold CV within each subject
- Subject-independent: 5 held-out subjects (MODMA); LOSO (SEED)

**Subject Independence:** Partial | **Cross-Subject:** Yes (subject-independent setting) | **LODO:** Yes (SEED LOSO) | **Cross-Dataset:** No

**Metrics:** AUC (primary)

---

**Findings — Main Results:**

MODMA (depression):

- EEG-RCformer (AIRM, K=5) sub-dependent: **AUC 0.9802 ± 0.0037** vs. best baseline TimesNet 0.9828 (comparable)
- EEG-RCformer (AIRM, K=5) sub-independent: **AUC 0.7154 ± 0.0608** vs. best baseline EmT 0.7069 — **+0.85%** (statistically significant p=0.0127, effect size 1.92)
- AIRM vs. Euclidean-only (sub-independent): 0.7154 vs. 0.6249 — **+9.05%** (p=0.0127, large effect)
- AIRM vs. No clustering (sub-independent): 0.7154 vs. 0.6337 — **+8.17%**

SEED (emotion):

- Sub-dependent: AUC 0.8541 — best among all models
- Sub-independent (LOSO): AUC 0.8011 vs. EmT 0.8020 (comparable; not significant p=0.38)
- AIRM vs. Euclidean (sub-dependent): significant (p=0.0426); sub-independent: not significant

**Findings — Important Observations:**

- AIRM metric is the critical driver: replacing AIRM with Euclidean distance in clustering drops MODMA sub-independent AUC by 9% — geometry matters most for cross-subject settings
- Clustering itself (any metric) is essential: no-clustering baseline drops AUC by 8% — channel compression reduces Transformer overfitting on small EEG datasets
- K=5 is optimal for both datasets; K=9 degrades (over-fragmentation); model is relatively insensitive to K=3 vs K=5 on MODMA
- MODMA gains are statistically significant; SEED sub-independent gains are not — likely because SEED is a smaller, harder cross-subject problem

**Findings — Failure Cases:**

- SEED sub-independent: no significant improvement over EmT (0.8011 vs. 0.8020) — Riemannian clustering helps more for depression (resting-state, high cross-subject variance) than stimulus-evoked emotion
- Only two datasets evaluated; broader validation needed
- No explicit domain adaptation module — relies on clustering to implicitly reduce inter-subject variance

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: AIRM-based channel clustering as a preprocessing step before any EEG encoder — reduces 128 → 5 tokens, dramatically cutting computation while improving cross-subject robustness. Directly applicable to depression LODO with high-density EEG (MODMA has 128 channels)
- Architecture: Hybrid spatial+functional clustering with EMA update — stable across trials and subjects while remaining trial-adaptive; balance constraint prevents degenerate clusters
- Representation: PSD + DE features over 7 frequency bands as multi-scale spectral descriptors — well-validated for both depression and emotion; more stable across subjects than raw EEG
- Training: The MODMA depression dataset with sub-independent evaluation (5 held-out subjects, 53 total) is the closest existing benchmark to our LODO depression goal — use these numbers as baselines
- Evaluation: Report AUC (not just accuracy) for depression classification — class imbalance makes AUC more informative

**Knowledge Extraction — What We Should NOT Borrow:**

- Subject-dependent evaluation as primary metric — our goal is subject-independent LODO
- Simple average pooling within clusters — attention-weighted pooling could better preserve discriminative channels
- No explicit domain adaptation — clustering alone is insufficient for deep cross-subject generalization; needs CorSW or rebiasing on top
- MLP classifier — replace with Riemannian head (RHOP, Paper 3) or tangent-space SVM for better geometry exploitation

---

**Relevance Score:** 9/10

**Reason:** This is the most directly relevant paper found so far. It uses the MODMA depression dataset under a subject-independent evaluation, demonstrates that AIRM-based Riemannian channel clustering significantly improves cross-subject depression AUC (+9% over Euclidean), and provides a concrete architecture combining handcrafted spectral features with Transformer modeling. The 0.7154 AUC on MODMA sub-independent is our primary baseline to beat. The paper validates that Riemannian geometry matters more for cross-subject depression than for within-subject settings — directly supporting our design hypothesis.

---

**Possible Integration:**

- Directly usable: AIRM-based channel clustering (K=5) as a preprocessing module to reduce 128-channel MODMA to 5 regional tokens before our encoder — reduces computation and cross-subject channel misalignment
- Directly usable: PSD+DE 7-band features as the primary spectral representation for depression EEG
- Directly usable: MODMA sub-independent AUC 0.7154 as the baseline to beat in our LODO framework
- Requires modification: Replace simple average pooling in clusters with attention-weighted pooling; add CorSW (Paper 4) alignment loss on cluster tokens; replace MLP head with RHOP (Paper 3)
- Future experiment: EEG-RCformer baseline + CorSW alignment + RHOP head + transductive C_ref update (Paper 6) in strict LODO on MODMA — stack all identified improvements and measure their cumulative effect vs. sub-independent baseline of 0.7154

---

## Paper 9 — MDRM + TSLDA: Multiclass BCI Classification by Riemannian Geometry (Relevance: 7/10)

**Title:** Multiclass Brain-Computer Interface Classification by Riemannian Geometry

**Year:** 2012

**Venue:** IEEE Transactions on Biomedical Engineering

**Dataset:** BCI Competition IV Dataset IIa — 9 subjects, 4-class motor imagery (right hand, left hand, foot, tongue), 22 electrodes, 576 trials/subject, 8–30 Hz bandpass, 0.5–2.5s post-cue window

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Propose a principled framework for multiclass EEG classification using the Riemannian geometry of covariance matrices, replacing ad-hoc feature engineering with geometrically consistent operations on the SPD manifold.

**Research Problem — Why It Matters:**

Prior BCI methods (CSP, LDA on log-band power) operate in Euclidean space, ignoring that covariance matrices lie on a curved manifold. Euclidean operations on SPD matrices introduce distortions (swelling effect); this paper shows that respecting the manifold structure yields better classifiers.

---

**Methodology — Core Idea:**

Two methods are proposed. MDRM (Minimum Distance to Riemannian Mean): compute the Riemannian (Fréchet/Karcher) mean of training covariance matrices per class, then assign a test trial to the nearest class mean under the affine-invariant Riemannian metric. TSLDA (Tangent Space LDA): map all covariances to the tangent space at the global geometric mean via Log map, half-vectorize the symmetric matrices, apply ANOVA-based variable selection (FDR-corrected), then run Fisher LDA. Both methods treat the covariance matrix itself as the feature, bypassing explicit band-power extraction.

**Methodology — Model Architecture:**

- Feature extraction: bandpass-filtered EEG epochs → sample covariance matrices (22×22 SPD matrices)
- Backbone: none (no deep network); purely geometric / statistical classifier
- MDRM path: per-class Riemannian mean (iterative gradient descent on manifold) → Riemannian distance to each class mean → nearest-mean assignment
- TSLDA path: global Riemannian mean P_G → Log_{P_G}(C) for each trial → half-vectorization (upper triangle, scaled off-diagonal) → SVD orthogonalization → ANOVA variable selection with FDR → Fisher LDA
- Classifier: nearest-mean (MDRM) or LDA (TSLDA)

**Methodology — Learning Strategy:**

- Supervised (within-subject, session-to-session split following BCI Competition IV protocol)
- No cross-subject or domain adaptation component

**Methodology — Key Innovation:**

First complete multiclass Riemannian framework for EEG BCI: (1) establishes MDRM as the geometric analogue of nearest-centroid classifiers on SPD manifolds; (2) derives the TSLDA pipeline as a principled linearization of the Riemannian geometry; (3) provides theoretical justification that CSP features are an approximation of the Riemannian manifold, unifying the CSP and Riemannian perspectives.

---

**Evaluation Protocol:**

**Dataset Split:** BCI Competition IV IIa official split — training session and test session per subject (within-subject evaluation). No cross-subject or LODO evaluation.

**Subject Independence:** No

**Cross-Subject Evaluation:** No

**Leave-One-Subject-Out (LODO):** No

**Cross-Dataset Evaluation:** No

**Metrics:** Classification accuracy (%), kappa coefficient; averaged over 9 subjects

---

**Findings — Main Results:**

- TSLDA: 70.2% mean accuracy vs. CSP+LDA reference: 65.1% — significant improvement (+5.1 pp)
- MDRM: 64.4% — comparable to CSP+LDA with far simpler classification
- Euclidean mean + Euclidean distance: 56.2% — confirms manifold structure is load-bearing; Euclidean operations lose discriminative information
- Both Riemannian methods outperform Euclidean counterparts consistently across all 9 subjects

**Findings — Important Observations:**

- The Euclidean distance experiment (same data, flat geometry) performs substantially worse, directly validating that the Riemannian metric captures structure the Euclidean metric does not
- CSP feature space is shown theoretically to be an approximated, filtered projection of the Riemannian manifold — explains why CSP works and suggests Riemannian geometry is the more complete version
- ANOVA variable selection in tangent space is critical for TSLDA performance; without it, the vectorized covariance has too many correlated dimensions
- Most discriminant tangent-space variable separates hand vs. foot/tongue classes, not hand vs. hand — physiologically expected from motor cortex topology

**Findings — Failure Cases:**

- Within-subject only — no evidence the method generalizes across subjects
- 9-subject dataset with single split — no statistical power for cross-subject claims
- MDRM is sensitive to small training sets because the Riemannian mean is slow to converge with few samples per class
- No regularization of covariance matrices (no shrinkage) — can break with few trials relative to channels

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: MDRM as a class prototype classifier on the SPD manifold — directly applicable as a Riemannian nearest-centroid baseline in our LODO depression framework; per-class Riemannian means as class prototypes from source subjects
- Architecture: TSLDA pipeline (global Riemannian mean → Log map → half-vectorize → LDA) is the canonical linearization of SPD data — use as a lightweight baseline before trying deep SPDNet or RHOP
- Representation: Sample covariance matrix as primary feature — foundational; all downstream papers (RHOP, CorSW, RK-SVM, DCCA) build on this
- Training: ANOVA/FDR variable selection in tangent space for high-dimensional SPD features — relevant when using full covariance matrices from 128-channel MODMA (128×128 → 8256-dim vectorization)
- Evaluation: Kappa alongside accuracy — useful for imbalanced datasets (depression detection often has unequal MDD/HC class sizes)
- Theory: The equivalence of CSP and Riemannian geometry provides justification for using covariance matrices as universal EEG features across tasks, including depression

**Knowledge Extraction — What We Should NOT Borrow:**

- Within-subject evaluation protocol — our goal is strict LODO
- No subject alignment — covariance matrices from different subjects are compared directly without re-centering; inter-subject drift will dominate in cross-subject settings
- No regularization of covariance estimator — with 128 MODMA channels, need shrinkage (LedoitWolf or OAS) or DCCA (Paper 7)
- Session-to-session split — not applicable to LODO depression scenario
- ANOVA variable selection without cross-subject stability check — selected variables may be subject-specific, not generalizable

---

**Relevance Score:** 7/10

**Reason:** This is the foundational paper for all Riemannian EEG methods in this review. MDRM and TSLDA are the baseline algorithms that Papers 2, 6, 7, and 8 all extend. The theoretical contribution linking CSP to Riemannian geometry justifies using covariance-based features across tasks. However, the paper itself is strictly within-subject and predates cross-subject generalization concerns — its value to us is primarily as the mathematical and algorithmic foundation, not a directly deployable cross-subject method.

---

**Possible Integration:**

- Directly usable: MDRM as a Riemannian nearest-centroid baseline in LODO — compute per-class Riemannian mean from all source subjects, classify test-subject trials by nearest mean; this is the simplest meaningful cross-subject Riemannian baseline
- Directly usable: TSLDA pipeline as a lightweight linear cross-subject baseline after applying rebiasing (Paper 6) or CorSW alignment (Paper 4)
- Requires modification: Extend MDRM to cross-subject by using rebiased (Paper 6) covariance matrices — compute Riemannian means after aligning all subjects to a common reference, then run MDRM in the aligned space; this becomes the "MDRM+rebiasing" cross-subject baseline
- Future experiment: MDRM baseline → TSLDA baseline → RK-SVM (Paper 6) → EEG-RCformer (Paper 8) → our transductive model; stepwise ablation showing each geometric component's marginal contribution in LODO depression


---

## Paper 10 — CTSSP: Temporal-Spectral-Spatial Joint Optimization for MI EEG (Relevance: 2/10)

**Title:** CTSSP: A Temporal–Spectral-Spatial Joint Optimization Algorithm for Motor Imagery EEG Decoding

**Year:** 2026

**Venue:** Journal of Neural Engineering

**Dataset:** BCI Competition IV 2a (9 subjects, 4-class MI); cross-session evaluation

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Jointly optimize temporal, spectral, and spatial filters for cross-session MI decoding using a unified end-to-end deep learning framework.

**Research Problem — Why It Matters:**

Prior MI BCI methods treat temporal, spectral, and spatial filtering as sequential stages, leading to suboptimal features; cross-session non-stationarity degrades performance.

---

**Methodology — Core Idea:**

CTSSP learns temporal convolutions (multi-scale), spectral filters (bank of bandpass filters), and spatial filters (depthwise convolution across channels) jointly in an end-to-end CNN architecture. A cross-session adaptation strategy aligns distributions between sessions.

**Methodology — Model Architecture:**

- Feature extraction: raw EEG → temporal convolution → spectral filter bank → spatial depthwise convolution
- Classifier: FC softmax
- Cross-session adaptation: batch normalization across sessions

**Methodology — Learning Strategy:** Supervised, with cross-session domain adaptation

**Methodology — Key Innovation:** Joint temporal-spectral-spatial optimization in a single learnable pipeline

---

**Evaluation Protocol:**

**Dataset Split:** Within-subject, cross-session (train on session 1, test on session 2)

**Subject Independence:** No

**Cross-Subject Evaluation:** No

**Leave-One-Subject-Out (LODO):** No

**Cross-Dataset Evaluation:** No

**Metrics:** Classification accuracy

---

**Findings — Main Results:** Outperforms EEGNet and CSP baselines on BCI-IV 2a cross-session evaluation; exact numbers not extracted.

**Findings — Important Observations:** Joint optimization beats sequential filtering; spectral component is most critical.

**Findings — Failure Cases:** Within-subject cross-session only; no cross-subject generalization demonstrated.

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: Multi-scale temporal convolution as a feature extraction front-end — applicable before Riemannian covariance computation
- Architecture: Spectral filter bank as a preprocessing stage — though simpler than DMFBTSM (Paper 2), still useful

**Knowledge Extraction — What We Should NOT Borrow:**

- Within-subject evaluation — not our target
- Cross-session adaptation without cross-subject validation
- No Riemannian geometry component — purely Euclidean deep learning pipeline

---

**Relevance Score:** 2/10

**Reason:** Pure Euclidean MI BCI paper with no Riemannian geometry and no cross-subject evaluation. Only the multi-scale temporal convolution idea has minor transferable value.

---

**Possible Integration:**

- Requires modification: Multi-scale temporal front-end could precede our covariance computation; but this is already covered better by DMFBTSM (Paper 2)

---

## Paper 11 — Pseudo Affine-Invariant Riemannian Metrics for Efficient BCI (Relevance: 4/10)

**Title:** Pseudo Affine-Invariant Riemannian Metrics for Efficient Brain-Computer Interfaces

**Year:** 2026

**Venue:** HAL preprint (GIPSA-lab, University Grenoble Alpes)

**Dataset:** 16 open-access BCI databases — 6 MI (69 sessions) + 10 P300 (155 sessions)

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Reduce the computational complexity of affine-invariant (AI) Riemannian operations from O(C³) to O(C²) via data preconditioning, preserving MDM classifier accuracy while dramatically reducing compute cost.

**Research Problem — Why It Matters:**

The AIRM has cubic complexity in matrix dimension — prohibitive for high-density EEG (128+ channels). Existing alternatives (log-Euclidean, log-Cholesky) sacrifice invariance properties.

---

**Methodology — Core Idea:**

Propose a preconditioning transformation that maps the AIRM into a "pseudo-affine-invariant" metric with quadratic computational complexity. The key insight: a data-dependent diagonal whitening transform converts the AIRM distance computation into an approximately Euclidean form. Applied to MDM: distances and barycenters computed with O(C²) cost instead of O(C³) while maintaining classification accuracy.

**Methodology — Model Architecture:**

- Feature extraction: sample covariance matrices
- Preconditioning: diagonal whitening transform based on data statistics
- Classifier: MDM with pseudo-AI metric (quadratic distance/barycenter computation)

**Methodology — Learning Strategy:** Supervised; within-session cross-validation

**Methodology — Key Innovation:** O(C²) pseudo-AI metric that preserves MDM classification performance; up to 15× speed-up for MI data, 4× for P300

---

**Evaluation Protocol:**

**Dataset Split:** Within-session cross-validation across 16 databases

**Subject Independence:** No (aggregated, not subject-independent)

**Cross-Subject Evaluation:** No

**Leave-One-Subject-Out (LODO):** No

**Cross-Dataset Evaluation:** Evaluated across 16 databases but not subject-independent

**Metrics:** Classification accuracy, speed-up factor

---

**Findings — Main Results:**

- Classification performance preserved vs. full AIRM across 16 databases
- Speed-up: up to 15× for MI (simpler preconditioner), 4× for P300
- Performance equivalent to full AIRM in >90% of tested cases

**Findings — Important Observations:**

- The pseudo-AI metric is particularly effective when input matrices are well-conditioned
- For P300, a more complex preconditioner is needed due to lower signal-to-noise ratio

**Findings — Failure Cases:**

- Within-session only — no cross-subject generalization tested
- Preconditioning may degrade with ill-conditioned matrices (few trials, many channels)

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: O(C²) pseudo-AI metric as a drop-in replacement for AIRM in MDM/TSLDA — critical for MODMA (128 channels, 128×128 matrices)
- Architecture: Diagonal whitening preconditioning as a lightweight normalization step before Riemannian operations
- Practical: For 128-channel MODMA, full AIRM at every LODO fold is very expensive; pseudo-AI metric would reduce compute substantially

**Knowledge Extraction — What We Should NOT Borrow:**

- Within-session evaluation
- No adaptation module — pure classification, no cross-subject alignment

---

**Relevance Score:** 4/10

**Reason:** Addresses a real computational bottleneck for our pipeline (128-channel EEG with full AIRM). The pseudo-AI metric could make our LODO framework feasible at scale. However, no cross-subject evaluation and purely engineering contribution.

---

**Possible Integration:**

- Directly usable: Pseudo-AI metric preconditioning to reduce LODO training cost on MODMA 128-channel data — apply before MDM/MDRM baseline
- Future experiment: Compare pseudo-AI metric vs. full AIRM in our LODO framework to assess if classification accuracy is preserved

---

## Paper 12 — RRSLVQ-LEML: Riemannian Quantification Learning for Real-time Emotion Recognition (Relevance: 4/10)

**Title:** Real-time EEG-based Emotion Recognition Using Riemannian Quantification Learning

**Year:** 2026

**Venue:** Biomedical Signal Processing and Control

**Dataset:** SEED, SEED-IV, SEED-V (Chinese Academy of Sciences emotion datasets); 7 online participants

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Real-time EEG emotion classification with few training samples and limited channels using Riemannian geometry — targeting practical deployment with minimal user calibration.

**Research Problem — Why It Matters:**

Most EEG emotion classifiers require large training sets and many channels; real-time scenarios require fast classification with minimal calibration data.

---

**Methodology — Core Idea:**

RRSLVQ-LEML (Robust Riemannian Soft Learning Vector Quantization with Log-Euclidean Metric Learning): a prototype-based classifier on the SPD manifold using the log-Euclidean metric. Each class is represented by prototype covariance matrices learned to maximize inter-class separation. Metric learning adapts the log-Euclidean distance to the specific dataset. Channel selection identifies universal 22-channel subset across datasets.

**Methodology — Model Architecture:**

- Feature extraction: SPD (covariance) matrices from selected 22 channels
- Backbone: none (prototype-based)
- Classifier: LVQ (learning vector quantization) on SPD manifold with log-Euclidean metric + metric learning

**Methodology — Learning Strategy:** Supervised; reduced training set (150 samples per class)

**Methodology — Key Innovation:** Prototype-based Riemannian classifier with metric learning; universal channel selection for cross-dataset deployment

---

**Evaluation Protocol:**

**Dataset Split:** Within-subject cross-validation (SEED: 9-fold); 150 samples per category for training reduction experiments

**Subject Independence:** No (within-subject primarily)

**Cross-Subject Evaluation:** No

**Leave-One-Subject-Out (LODO):** No

**Cross-Dataset Evaluation:** Yes (evaluated on SEED, SEED-IV, SEED-V with same channel set)

**Metrics:** Classification accuracy (mean ± std)

---

**Findings — Main Results:**

- SEED (22 channels, 150 samples): 88.33% ± 5.77% — significantly outperforms SPDNet (p < 0.05)
- SEED-IV: 76.05% ± 7.85%
- SEED-V: 70.62% ± 11.66%
- Online (real-time, 4-class, 7 subjects): 77.28% ± 6.68%; strong correlation with offline results (r = 0.82)

**Findings — Important Observations:**

- Log-Euclidean metric + LVQ outperforms SPDNet with far fewer parameters
- Universal 22-channel set transfers across SEED variants without reselection
- Strong offline-online correlation confirms real-time viability

**Findings — Failure Cases:**

- Within-subject only — no cross-subject generalization
- SEED datasets use video stimuli not resting-state; poor transfer to depression EEG paradigm
- Only 7 online subjects — small sample

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: Prototype-based Riemannian classifier (LVQ on SPD) as an alternative to MDM — learnable prototypes vs. empirical mean, better with imbalanced training sets
- Architecture: Metric learning on the SPD manifold to adapt the log-Euclidean distance to depression EEG specifics
- Training: Reduced-sample training (150 samples per class) with robust Riemannian method — relevant for LODO where target subject has zero labeled samples but source subjects may have limited data

**Knowledge Extraction — What We Should NOT Borrow:**

- Within-subject evaluation and SEED stimulus paradigm
- Channel selection specific to SEED datasets — MODMA uses different paradigm and regions
- Log-Euclidean metric as the primary metric — AIRM or pseudo-AI (Paper 11) is geometrically superior

---

**Relevance Score:** 4/10

**Reason:** Demonstrates prototype-based Riemannian classifiers work well with few training samples — relevant for our LODO setting where per-subject calibration is minimal. However, purely within-subject emotion recognition on stimulus-based datasets.

---

**Possible Integration:**

- Requires modification: Replace MDM's empirical Riemannian mean with a learnable LVQ prototype scheme, allowing source subjects' distribution to be more accurately captured
- Future experiment: LVQ-style per-class prototype updating as source subjects accumulate in LODO training set

---

## Paper 13 — RepSPD: Enhancing SPD Manifold Representation via Dynamic Graphs (Relevance: 5/10)

**Title:** RepSPD: Enhancing SPD Manifold Representation in EEGs via Dynamic Graphs

**Year:** 2026

**Venue:** arXiv preprint (cs.AI)

**Dataset:** BCI Competition IV 2a (9 subjects, 4-class MI, 22 channels); TUSZ v1.5.2 (seizure detection, 19 channels, 5612 recordings)

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Enrich SPD manifold representations by incorporating frequency-specific synchronization and local brain topology through dynamic graph neural networks, going beyond static covariance-based connectivity.

**Research Problem — Why It Matters:**

Existing SPD-based methods aggregate EEG via static linear statistics (covariance), ignoring temporal non-stationarity and task-specific frequency-band connectivity patterns. Dynamic functional connectivity is discarded.

---

**Methodology — Core Idea:**

RepSPD learns two complementary EEG representations jointly: (1) a Riemannian SPD manifold representation via SPDNet-style BiMap/ReEig layers; (2) a dynamic graph representation capturing time-varying functional connectivity via graph neural networks. A novel Dynamic Manifold Attention module fuses the graph-derived Euclidean features into the Riemannian manifold in a geometry-preserving way via cross-attention. A global bidirectional alignment loss reduces curvature-induced distortions in the tangent space.

**Methodology — Model Architecture:**

- Manifold branch: SPDNet (BiMap → ReEig → SPDNorm → LogEig) on covariance matrices
- Graph branch: dynamic GNN capturing time-varying connectivity (time-then-graph paradigm)
- Fusion: Dynamic Manifold Attention (cross-attention on Riemannian manifold) — maps graph Euclidean features to SPD manifold-compatible form
- Loss: geometry alignment loss (bidirectional SPD-Euclidean consistency)
- Classifier: linear softmax on fused tangent representation

**Methodology — Learning Strategy:** Supervised; standard within-subject evaluation

**Methodology — Key Innovation:** Cross-attention mechanism on the Riemannian manifold for fusing graph-derived functional connectivity with static SPD statistics; geometry alignment loss for consistent fusion

---

**Evaluation Protocol:**

**Dataset Split:** 5-fold cross-validation (BCI-IV 2a); train/test split (TUSZ)

**Subject Independence:** No (within-subject for MI)

**Cross-Subject Evaluation:** No

**Leave-One-Subject-Out (LODO):** No

**Cross-Dataset Evaluation:** No

**Metrics:** Classification accuracy (MI); F1-score (seizure detection)

---

**Findings — Main Results:**

- BCI-IV 2a: 81.05% accuracy — outperforms Graph-CSPNet (78.82%), CorAtt (75.56%), SPDNet, FBCNet
- TUSZ seizure detection: 3.3% improvement over strongest baseline
- Consistently lower variance than all baselines — improved robustness

**Findings — Important Observations:**

- Dynamic graph modeling captures information that static covariance misses — cross-attention integration is key
- Geometry alignment loss prevents the fusion from destroying manifold structure
- SPD-graph hybrid is more robust than either component alone

**Findings — Failure Cases:**

- Within-subject evaluation only
- Dynamic graph computation is expensive — temporal complexity increases with sequence length
- No cross-subject or domain adaptation component

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: Cross-attention between SPD manifold features and graph-derived functional connectivity — applicable to our depression model where static covariance misses temporal non-stationarity of MDD EEG
- Architecture: Dynamic Manifold Attention module as a plug-in enrichment layer for any SPD-based pipeline
- Loss: Geometry alignment loss for SPD-Euclidean fusion — prevents swelling effect when combining graph and Riemannian branches
- Representation: Time-then-graph paradigm for capturing non-stationary connectivity dynamics — relevant for resting-state depression EEG where connectivity evolves over recording time

**Knowledge Extraction — What We Should NOT Borrow:**

- Within-subject evaluation
- Full dynamic GNN computation at inference time — expensive for LODO with 128 channels
- CorAtt baseline (full-rank correlation attention) — already covered better by CorSW (Paper 4)

---

**Relevance Score:** 5/10

**Reason:** Introduces geometry-preserving fusion of dynamic graph and SPD representations — a novel architectural pattern worth borrowing. However, no cross-subject evaluation and purely MI/seizure tasks.

---

**Possible Integration:**

- Requires modification: Adapt the Dynamic Manifold Attention module to fuse PLV/PCC functional connectivity graph (subject-invariant) with per-subject covariance SPD features in our LODO depression pipeline
- Future experiment: Static SPDNet + dynamic graph fusion vs. static SPDNet only in LODO depression — measure if resting-state temporal connectivity dynamics improve cross-subject generalization

---

## Paper 14 — RGP-VAE: Riemannian Geometry-Preserving VAE for EEG Data Augmentation (Relevance: 5/10)

**Title:** Riemannian Geometry-Preserving Variational Autoencoder for MI-BCI Data Augmentation

**Year:** 2026

**Venue:** arXiv preprint (cs.LG)

**Dataset:** Faller et al. dataset — 12 subjects, 13-channel EEG, 2-class MI (right hand vs. feet), 5572 trials total (398–597 per subject)

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Generate synthetic EEG covariance matrices that are valid SPD matrices and improve cross-subject MI-BCI performance via data augmentation.

**Research Problem — Why It Matters:**

Deep learning EEG models are limited by data scarcity and inter-subject variability. Standard VAEs violate SPD manifold structure; geometric interpolation is restricted to the convex hull of observed data.

---

**Methodology — Core Idea:**

RGP-VAE: a VAE that operates in tangent space rather than directly on SPD matrices. Input SPD matrix → Log map to tangent space at class-specific reference Pref → vectorize → standard VAE encoder/decoder → unvectorize → Exp map back to SPD manifold. Loss function combines Riemannian distance, tangent-space reconstruction accuracy, and KL divergence. Parallel transport aligns subject-specific distributions to a global class reference before encoding — the key cross-subject invariance mechanism.

**Methodology — Model Architecture:**

- Preprocessing: parallel transport of each subject's covariance to global class reference mean
- Encoder: tangent-space vectorization → VAE encoder (FC layers) → latent distribution (μ, log σ²)
- Decoder: latent z → FC decoder → tangent-space unvectorize → Exp map → synthetic SPD matrix
- Loss: Riemannian distance + tangent-space MSE + KL divergence

**Methodology — Learning Strategy:** Generative (unsupervised latent space) + supervised classifier; cross-subject focus

**Methodology — Key Innovation:** VAE that preserves SPD manifold structure via Log/Exp map bridge; parallel transport for subject-invariant latent space; geometry-aware composite loss

---

**Evaluation Protocol:**

**Dataset Split:** Leave-one-subject-out cross-subject augmentation evaluation

**Subject Independence:** Yes (LOSO framework)

**Cross-Subject Evaluation:** Yes

**Leave-One-Subject-Out (LODO):** Yes

**Cross-Dataset Evaluation:** No

**Metrics:** Classification accuracy; comparison with/without augmentation

---

**Findings — Main Results:**

- RGP-VAE generates valid SPD matrices (all eigenvalues positive, unit diagonal maintained)
- Learns subject-invariant latent space (parallel transport alignment confirmed)
- Mixed augmentation results: improves cross-subject accuracy with some classifiers (MDM, TSLDA), neutral/negative with others
- Impact is classifier-dependent — geometry-aware classifiers benefit most

**Findings — Important Observations:**

- Parallel transport to global class reference is the key step enabling cross-subject invariance — aligns subjects on manifold before latent learning
- Synthetic samples extend beyond the convex hull of observed data — genuine generative capability on manifold
- SPD manifold structure fully preserved — no degenerate matrices generated

**Findings — Failure Cases:**

- Augmentation benefit is classifier-dependent; no improvement with Euclidean classifiers
- Small dataset (12 subjects, 13 channels) — limited generalizability
- No comparison with DCCA or CorSW as alignment baselines

---

**Knowledge Extraction — What We Can Borrow:**

- Technique: Parallel transport to global class reference mean as a subject alignment step — equivalent to rebiasing (Paper 6) but using Riemannian parallel transport instead of R^{-1/2}CR^{-1/2}; more geometrically faithful
- Architecture: Log/Exp map bridge for any neural network operating on SPD manifold — the encode-in-tangent-space/decode-back pattern is clean and reusable
- Training: Geometry-aware composite loss (Riemannian distance + tangent MSE + KL) for any generative model on SPD manifold
- Augmentation: In a low-data LODO setting (source subjects may have few trials), synthetic covariance generation via RGP-VAE could augment training set

**Knowledge Extraction — What We Should NOT Borrow:**

- Small 12-subject, 13-channel dataset — limited evidence
- MI paradigm — different from resting-state depression EEG
- Classifier-dependent augmentation benefits — unreliable as a core strategy

---

**Relevance Score:** 5/10

**Reason:** Uses LOSO protocol and addresses cross-subject generalization via parallel transport alignment + generative augmentation on SPD manifold. Parallel transport idea is complementary to rebiasing (Paper 6) and may provide a more principled cross-subject alignment mechanism.

---

**Possible Integration:**

- Directly usable: Parallel transport to global class reference as an alternative/complement to rebiasing for aligning source subjects in LODO training
- Requires modification: Adapt RGP-VAE to resting-state depression EEG (MODMA) — replace MI-specific 13-channel setup with MODMA 128-channel setup; requires pseudo-AI metric (Paper 11) for computational feasibility
- Future experiment: Data augmentation for minority subjects in LODO depression — source subjects with fewer trials augmented via RGP-VAE before computing class Riemannian means

---

## Paper 15 — Riemannian Geometry Meets fMRI: Correlation Manifolds and Grassmannian Subspaces (Relevance: 6/10)

**Title:** Riemannian Geometry Meets fMRI: The Advantages of Modeling Correlation Manifolds and Eigenvector Subspaces

**Year:** 2026

**Venue:** arXiv preprint (cs.LG) — King's College London / University of Padova

**Dataset:** 2 clinical cohorts (Parkinson's disease, psychosis/schizophrenia); 3 ageing fMRI datasets; resting-state fMRI

**Modality:** fMRI (functional MRI — not EEG, but directly applicable methods)

---

**Research Problem — Main Objective:**

Introduce scalable geometric methods for fMRI correlation matrix analysis: (1) Off-log metric for correlation manifolds with closed-form operations; (2) Grassmannian subspace discrimination for eigenvector-based representation.

**Research Problem — Why It Matters:**

Standard analyses treat correlation matrix entries independently. Existing correlation manifold methods (quotient-affine metric, QAM) lack closed-form operations, limiting scalability to large ROI sets. This is the first scalable closed-form framework for clinical neuroimaging with correlation matrices.

---

**Methodology — Core Idea:**

Two geometric tools: (1) **Off-log metric**: maps correlation matrices to symmetric zero-diagonal matrices via a modified log transform on off-diagonal entries, yielding closed-form distances, Fréchet means, and linear models — enabling standard ML without manifold optimization. (2) **Grassmannian subspace discrimination**: compares subjects via principal-angle distances between eigenvector subspaces of the graph Laplacian of their correlation matrix — removes sign and basis ambiguity, focuses on subspace geometry. Both tools are validated on clinical cohorts (Parkinson's, psychosis) with permutation tests and classification experiments.

**Methodology — Model Architecture:**

- Feature extraction: fMRI BOLD → correlation matrix → Off-log map → symmetric zero-diagonal matrix (linear representation) OR eigenvector subspace on Grassmannian
- Classifier: SVM, logistic regression, LDA in transformed space
- No deep network

**Methodology — Learning Strategy:** Supervised classification with cross-validation; clinical cohort-level analysis

**Methodology — Key Innovation:** Off-log metric provides closed-form Fréchet mean and distances for correlation matrices — eliminates the need for iterative Riemannian optimization

---

**Evaluation Protocol:**

**Dataset Split:** Cross-validation within clinical cohorts; some leave-one-out evaluation

**Subject Independence:** Yes (subject-level classification for clinical groups)

**Cross-Subject Evaluation:** Yes (classification separates patient groups)

**Leave-One-Subject-Out (LODO):** Partial (within-cohort leave-one-out for small cohorts)

**Cross-Dataset Evaluation:** Yes (3 ageing datasets + 2 clinical cohorts)

**Metrics:** Classification accuracy, AUC, permutation test p-values; brain-age prediction error

---

**Findings — Main Results:**

- Off-log metric increases sensitivity in permutation tests vs. Euclidean correlation analysis
- Classification: Off-log matched or exceeded Riemannian (QAM) and Euclidean baselines in 4/5 datasets
- Brain-age prediction: Riemannian metrics excelled in 2/3 cohorts
- Grassmannian method consistently outperformed Euclidean baselines; highlights disease-relevant networks
- Clinical cohort (psychosis): Grassmannian discrimination identified schizophrenia-relevant connectivity subspaces

**Findings — Important Observations:**

- Closed-form Off-log metric is as good as iterative AIRM for clinical classification — strong argument for using it in MODMA-sized problems
- Grassmannian subspace comparison is naturally invariant to eigenvector sign ambiguity — important for cross-subject functional connectivity analysis
- fMRI-derived conclusions likely transfer to EEG connectivity matrices (same statistical structure)

**Findings — Failure Cases:**

- fMRI not EEG — temporal resolution very different; functional connectivity computed over longer windows
- Psychosis/Parkinson's cohorts — different from depression; overlapping but not identical brain network pathology
- Grassmannian method requires reliable eigenvector estimation — unstable with few channels or short recordings

---

**Knowledge Extraction — What We Can Borrow:**

- Metric: Off-log metric as a closed-form, scalable alternative to AIRM for correlation matrices in our LODO depression framework — directly applicable to EEG correlation matrices (MODMA)
- Metric: Grassmannian subspace discrimination for comparing subject-specific principal connectivity modes — can identify which subjects share similar neural connectivity patterns, enabling better grouping for LODO
- Application: Validated on clinical cohorts (psychosis, Parkinson's) with subject-level classification — demonstrates the framework works for neurological/psychiatric classification under cross-subject conditions
- Architecture: Off-log → linear model pipeline as a very lightweight baseline for our LODO depression detection

**Knowledge Extraction — What We Should NOT Borrow:**

- fMRI-specific preprocessing and ROI selection
- Long TR temporal resolution assumptions — EEG is much higher frequency
- Brain-age regression objective — not our task

---

**Relevance Score:** 6/10

**Reason:** Applies geometric methods to clinical psychiatric cohorts (psychosis) in a cross-subject setting — structurally parallel to our depression detection goal. The Off-log metric for correlation matrices and Grassmannian subspace discrimination are both novel tools applicable to our EEG pipeline. Strong convergent evidence (with Papers 3, 4) that correlation matrices on the quotient manifold are the right representation for cross-subject neuroimaging.

---

**Possible Integration:**

- Directly usable: Off-log metric as a closed-form alternative to AIRM for correlation matrix Fréchet means in LODO — O(C²) with no iteration
- Directly usable: Grassmannian subspace comparison to measure subject similarity in our LODO framework — select most similar source subjects for each held-out test subject
- Future experiment: Off-log distance + MDM baseline vs. AIRM-MDM baseline in LODO depression on MODMA — measure classification parity and computational speedup

---

## Paper 16 — RST-GNN: Riemannian Spatio-Temporal Graph Neural Network for Cognitive Load (Relevance: 4/10)

**Title:** Riemannian Spatio-Temporal Graph Neural Network for Enhanced Cognitive Load Detection Using EEG

**Year:** 2026

**Venue:** Neurocomputing

**Dataset:** Dataset I (passive BCI cognitive load dataset, Hinss et al. 2023); Dataset II (multi-session cognitive load, Wang et al. 2023)

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Multi-class cognitive load detection from EEG by jointly modeling spatial covariance structure (Riemannian) and temporal dynamics (graph attention over time windows).

**Research Problem — Why It Matters:**

Prior GNN methods for EEG ignore Riemannian covariance structure; prior Riemannian methods ignore temporal dependencies within a trial.

---

**Methodology — Core Idea:**

RST-GNN: EEG → Riemannian spatial filtering (extracts covariance features) → nodes in a temporal graph → Graph-BiMap module (learnable SPD dimensionality reduction on graph) → temporal attention mechanism (fuses window-level features over trial duration). Graph-BiMap preserves SPD manifold structure while performing graph-based feature aggregation.

**Methodology — Model Architecture:**

- Feature extraction: Riemannian spatial filter → covariance matrices as graph nodes
- Graph layer: Graph-BiMap (BiMap transformation on SPD graph — manifold-preserving graph convolution)
- Temporal fusion: attention mechanism over time windows
- Classifier: softmax

**Methodology — Learning Strategy:** Supervised; cross-validation aggregated across subjects

**Methodology — Key Innovation:** Graph-BiMap — a graph convolution layer that operates directly on the SPD manifold; temporal attention over Riemannian features

---

**Evaluation Protocol:**

**Dataset Split:** Cross-validation within subjects; cross-validation within sessions for multi-session dataset

**Subject Independence:** No (within-subject per-subject validation)

**Cross-Subject Evaluation:** No

**Leave-One-Subject-Out (LODO):** No

**Cross-Dataset Evaluation:** No

**Metrics:** Accuracy, precision, recall, F1 (4 cognitive load levels)

---

**Findings — Main Results:**

- >96% accuracy on both datasets across 4 cognitive load levels
- Low inter-subject variability — consistent across subjects
- Outperforms all prior graph and Riemannian baselines

**Findings — Important Observations:**

- Graph-BiMap provides manifold-consistent graph convolution — keeps features on SPD manifold throughout the network
- Temporal attention over Riemannian features captures evolving cognitive states better than static covariance

**Findings — Failure Cases:**

- Within-subject only; 96% accuracy is suspiciously high — possible train/test leakage in within-subject CV
- Cognitive load different from depression; paradigm and brain dynamics distinct

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: Graph-BiMap module — manifold-preserving graph convolution that operates on SPD nodes; reusable in any architecture with spatial electrode graph
- Architecture: Temporal attention over covariance features for capturing dynamics within a resting-state EEG recording — relevant for depression (long resting-state sessions)
- Representation: Covariance matrices as graph nodes (one per time window) → temporal graph over a recording session

**Knowledge Extraction — What We Should NOT Borrow:**

- Within-subject evaluation; cognitive load paradigm
- Suspicious 96% accuracy — likely benefits from within-subject temporal correlation (not genuine generalization)
- No cross-subject validation; results non-transferable

---

**Relevance Score:** 4/10

**Reason:** Graph-BiMap is a useful architectural component (manifold-preserving graph convolution). However, no cross-subject validation and cognitive load task; limited direct relevance to depression LODO.

---

**Possible Integration:**

- Requires modification: Graph-BiMap as an electrode-graph aggregation layer replacing simple average pooling in our pipeline; combine with AIRM-based clustering (Paper 8) for hierarchical spatial aggregation

---

## Paper 17 — Stiefel-SPD Graph Convolution for Cross-Subject EEG Learning (Relevance: 6/10)

**Title:** Stiefel-SPD Manifold Graph Convolution for End-to-End EEG Learning

**Year:** 2026

**Venue:** IEEE Transactions on Neural Systems and Rehabilitation Engineering

**Dataset:** PhysioNet MI (109 subjects, BCI2000, 160 Hz); BNCI 2014-002; BCI-ERN (Berlin BCI competition)

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Design a fully geometry-consistent end-to-end EEG architecture that maintains SPD manifold structure throughout — from raw signal to classification — while remaining computationally tractable via Stiefel manifold dimensionality reduction.

**Research Problem — Why It Matters:**

Existing deep SPD architectures break manifold consistency by interleaving Euclidean layers; classical tangent-space pipelines use fixed global projections without learning task-specific geometry; both scale cubically with channel count.

---

**Methodology — Core Idea:**

Three-stage Stiefel→Graph-SPD→Log pipeline: (1) Depthwise-separable CNN extracts features; regularized covariances lie on SPD. (2) Learnable Stiefel projection (orthonormal, Riemannian SGD with QR retraction) reduces SPD dimension while preserving positive-definiteness and eigenvalue floor. (3) Scalp k-NN graph aggregation: neighbor covariances are parallel-transported to reference tangent space, attention-averaged, mapped back via Exp map — fully SPD throughout. Final Log-Euclidean map + softmax classifier. Cross-subject LOSO evaluation on three MI/ERN datasets.

**Methodology — Model Architecture:**

- Feature extraction: depthwise-separable CNN → regularized covariance matrices (SPD)
- Stiefel projection: learnable orthonormal B (optimized on Stiefel manifold via QR retraction) → reduced-rank SPD
- Graph layer: scalp k-NN → parallel transport → attention-weighted aggregation → Exp map back to SPD
- Classifier: LogEig map → linear softmax

**Methodology — Learning Strategy:** Supervised; cross-subject LOSO on 3 public MI/ERN datasets

**Methodology — Key Innovation:** End-to-end fully geometry-consistent architecture: all intermediates remain SPD; Stiefel projection reduces O(C³) to O(d³) with d << C; parallel-transport graph aggregation preserves geodesic relations

---

**Evaluation Protocol:**

**Dataset Split:** Leave-one-subject-out (LOSO) on all three datasets

**Subject Independence:** Yes

**Cross-Subject Evaluation:** Yes

**Leave-One-Subject-Out (LODO):** Yes

**Cross-Dataset Evaluation:** Evaluated on 3 datasets with same hyperparameters

**Metrics:** Classification accuracy, macro-F1, macro-AUROC, ECE (calibration); statistical significance tests

---

**Findings — Main Results:**

- Cross-subject LOSO: 83.2% (PhysioNet MI) / 81.5% (BNCI 2014-002) / 79.7% (BCI-ERN) accuracy
- Macro-AUROC ≈ 0.90; ECE ≤ 0.04 (well-calibrated probabilities)
- Outperforms strong Euclidean CNNs (EEGNet, DeepConvNet, ShallowConvNet) and Riemannian baselines (TSA, RPA) by ~1.2% on all datasets
- Single hyperparameter set across all three datasets

**Findings — Important Observations:**

- Full geometric consistency chain (Stiefel→Graph-SPD→Log) is the key: removing any step degrades performance
- Parallel-transport graph aggregation outperforms simple tangent-space averaging
- Well-calibrated probabilities (ECE ≤ 0.04) — rare in deep BCI models, useful for clinical deployment
- Single hyperparameter set across 3 different datasets — strong generalization

**Findings — Failure Cases:**

- MI task only — no affect or psychiatric task
- Stiefel projection adds Riemannian SGD complexity — harder to implement and train
- Parallel transport requires a reference SPD matrix per subject/batch — adds a preprocessing step

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: Stiefel manifold projection for dimensionality reduction of large SPD matrices (128×128 → d×d) — directly applicable to MODMA 128-channel data; reduces O(128³) to O(d³)
- Architecture: Parallel-transport graph aggregation on scalp k-NN graph — a geometry-consistent alternative to simple channel clustering (Paper 8); preserves geodesic relations when aggregating electrode neighborhoods
- Architecture: Well-calibrated Riemannian classifier via LogEig + linear softmax — calibration matters for clinical depression detection (probability estimates, not just class labels)
- Evaluation: LOSO cross-subject on multiple datasets with single hyperparameter set — this is the evaluation protocol we should adopt for MODMA

**Knowledge Extraction — What We Should NOT Borrow:**

- MI task; no adaptation module for cross-subject alignment
- Stiefel optimization (QR retraction) adds implementation complexity — may not be worth it if pseudo-AI metric (Paper 11) suffices for dimensionality reduction
- No transductive component — pure domain generalization, not transductive learning

---

**Relevance Score:** 6/10

**Reason:** This is a complete cross-subject LOSO pipeline that is fully geometry-consistent. The Stiefel projection, parallel-transport graph aggregation, and single-hyperparameter-set across datasets are all directly applicable to our MODMA LODO depression framework. No psychiatric task, but the architectural and evaluation patterns transfer directly.

---

**Possible Integration:**

- Directly usable: Stiefel manifold projection as a dimensionality reduction step before our Riemannian operations on 128-channel MODMA data
- Directly usable: LOSO with single hyperparameter set as our evaluation protocol — validates generalization, not overfitting to one dataset
- Requires modification: Replace MI task-specific CNN front-end with PSD+DE feature extraction (Paper 8 approach) for resting-state depression EEG; add transductive C_ref update (Paper 6) and CorSW alignment (Paper 4) on top

---

## Paper 18 — Cybathlon Longitudinal: Riemannian Geometry of MI-BCI User Learning (Relevance: 2/10)

**Title:** The Riemannian Geometry of User Learning in MI-BCI: A Cybathlon Longitudinal Study

**Year:** 2026

**Venue:** Research Square preprint (University of Padua)

**Dataset:** Single-subject longitudinal Cybathlon pilot data (multiple training sessions over weeks)

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Track how a user's neural representations change over longitudinal MI-BCI training using Riemannian geometry metrics on the SPD manifold.

---

**Methodology — Core Idea:**

Introduce Riemannian metrics (geodesic distance + cosine similarity in tangent space) to quantify neural trajectory changes across training sessions. Identify "emergent" vs. "stable" learning states based on these trajectories.

**Methodology — Learning Strategy:** Unsupervised analysis of longitudinal manifold trajectories

**Methodology — Key Innovation:** Riemannian trajectory metrics for characterizing user learning states in BCI

---

**Evaluation Protocol:**

**Subject Independence:** No (single-subject longitudinal)

**Cross-Subject Evaluation:** No

**Leave-One-Subject-Out (LODO):** No

---

**Findings — Main Results:** Riemannian features capture meaningful changes in neural representations across training sessions; emergent vs. stable learning states identifiable from geodesic trajectory metrics.

---

**Knowledge Extraction — What We Can Borrow:**

- Analysis: Riemannian manifold trajectory analysis for tracking non-stationarity over time — could be used to monitor depression-related covariance drift over a long resting-state recording

**Knowledge Extraction — What We Should NOT Borrow:**

- Single-subject, longitudinal MI learning — no relevance to depression LODO
- No cross-subject evaluation; no psychiatric application

---

**Relevance Score:** 2/10

**Reason:** Single-subject longitudinal MI study — methodologically interesting but not relevant to cross-subject depression detection.

---

## Paper 19 — Filter Bank CSP with Riemannian Weighting (Relevance: 2/10)

**Title:** Filter Bank CSP with Riemannian Weighting for Disability-Centric Motor Imagery BCI

**Year:** 2026

**Venue:** Brain Informatics

**Dataset:** BCI Competition IV Dataset 2a (9 subjects, 4-class MI, 22 electrodes)

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Extend standard CSP with Riemannian geometry-based weighting of covariance matrices to reduce noise sensitivity in multi-band, multiclass MI BCI.

---

**Methodology — Core Idea:**

Filter bank CSP computes spatial filters in multiple frequency bands. Riemannian weighting down-weights trials whose covariance matrices are far from the class Riemannian mean (outlier detection via Riemannian distance). Multi-classifier ensemble (LDA, RFC, MLP) with majority vote.

**Methodology — Learning Strategy:** Supervised, within-subject

**Methodology — Key Innovation:** Riemannian outlier weighting in CSP to reduce noise-contaminated trial influence

---

**Evaluation Protocol:**

**Subject Independence:** No (within-subject)

**Cross-Subject Evaluation:** No

**Leave-One-Subject-Out (LODO):** No

**Metrics:** Accuracy 81.83%, Precision 82.74%, F1 81.87%

---

**Knowledge Extraction — What We Can Borrow:**

- Technique: Riemannian outlier detection (distance from class mean) as a quality filter for trial selection — applicable in preprocessing to remove artifact-contaminated trials before computing class prototypes in LODO

**Knowledge Extraction — What We Should NOT Borrow:**

- CSP-centric pipeline, within-subject evaluation; no cross-subject generalization

---

**Relevance Score:** 2/10

**Reason:** Standard MI BCI paper with minor Riemannian novelty. Outlier filtering idea has minor transferable value.

---

## Paper 20 — Musical Familiarity EEG with Tangent Space Mapping (Relevance: 2/10)

**Title:** Objective Assessment of Familiarity in Music Using Imagery and EEG-Based Machine Learning

**Year:** 2026

**Venue:** Scientific Reports (Nature)

**Dataset:** 20 participants, EEG during silent gaps in familiar vs. unfamiliar songs; stimulus-evoked paradigm

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Distinguish neural responses to familiar vs. unfamiliar music using EEG, comparing spectral features vs. Riemannian TSM.

---

**Methodology — Core Idea:**

Within-subject classifiers trained independently per subject on EEG during musical imagery silences. Features: spectral power OR Riemannian TSM. Classifiers: logistic regression, SVM.

**Methodology — Key Innovation:** Application of TSM to musical familiarity EEG

---

**Evaluation Protocol:**

**Subject Independence:** No (within-subject)

**Cross-Subject Evaluation:** No

**Metrics:** Accuracy 76.5% ± 8.0% (TSM + logistic regression)

---

**Knowledge Extraction — What We Can Borrow:**

- Result: TSM outperforms spectral features for within-subject EEG classification in a passive paradigm — supporting use of Riemannian features for affective EEG in general

**Knowledge Extraction — What We Should NOT Borrow:**

- Within-subject; music paradigm; auditory cortex focus (irrelevant to depression)

---

**Relevance Score:** 2/10

**Reason:** Within-subject music paradigm; confirms TSM > spectral features generally but adds no cross-subject insights.

---

## Paper 21 — S-RCT: Session-wise Riemannian Alignment for Brainprint Recognition (Relevance: 5/10)

**Title:** Robust Brainprint Recognition via Session-wise Riemannian Alignment: Overcoming Non-stationary Brain Dynamics

**Year:** 2026

**Venue:** Research Square preprint (Xi'an University of Science and Technology)

**Dataset:** BCI Competition IV 2a (9 subjects, 2-session, 22 channels)

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Overcome cross-session EEG distributional drift for brainprint (biometric identity) recognition by aligning each session's covariance distribution to a common geometric reference on the Riemannian manifold.

**Research Problem — Why It Matters:**

Cross-session EEG exhibits significant distributional drift on the SPD manifold due to electrode impedance changes, cognitive state fluctuations, and temporal non-stationarity — same problem as cross-subject drift in our depression LODO setting.

---

**Methodology — Core Idea:**

S-RCT (Session-wise Riemannian Centerization): for each session, compute its Riemannian (Fréchet) mean → apply affine transform to translate the distribution so its geometric center aligns with the identity matrix I. This removes session-specific bias while preserving within-session geometry. Formally: for session k with mean M_k, transform each matrix P → M_k^{-1/2} P M_k^{-1/2} (equivalent to rebiasing from Paper 6 but applied session-wise). Topological channel analysis identifies optimal 5-channel parieto-occipital subset for identity-discriminative features.

**Methodology — Model Architecture:**

- Preprocessing: per-session Riemannian mean computation → S-RCT alignment (M_k^{-1/2} P M_k^{-1/2})
- Feature extraction: aligned covariance matrices → MDRM or TSLDA
- Channel optimization: exhaustive search for minimal stable channel array

**Methodology — Learning Strategy:** Supervised; cross-session (biometric) evaluation

**Methodology — Key Innovation:** Session-wise Riemannian centering to I as a principled alignment strategy; demonstrates parieto-occipital region has highest cross-session identity discriminability within the Default Mode Network

---

**Evaluation Protocol:**

**Dataset Split:** Cross-session (train session 1, test session 2) — biometric identification

**Subject Independence:** N/A (task is to identify each subject, not generalize across them)

**Cross-Subject Evaluation:** N/A

**Leave-One-Subject-Out (LODO):** No

**Cross-Dataset Evaluation:** No

**Metrics:** Identification accuracy (top-1)

---

**Findings — Main Results:**

- S-RCT cross-session identification: 98.33% accuracy with 22 channels
- 5-channel parieto-occipital array: 90.89% — stable minimal configuration
- Topological analysis confirms parieto-occipital region is the most stable across sessions (linked to Default Mode Network / visual processing)
- Without S-RCT alignment: substantially lower cross-session accuracy (exact baseline not extracted)

**Findings — Important Observations:**

- S-RCT is mathematically identical to the adaptive rebiasing from Paper 6 (Zanini 2017) but applied session-wise rather than trial-wise — converging evidence for the same technique from a different task domain
- Parieto-occipital stability aligns with depression EEG literature (posterior alpha asymmetry, DMN hyperactivation in MDD)
- Cross-task feature invariance confirmed — features reflect anatomy not task, supporting cross-subject transfer

**Findings — Failure Cases:**

- Biometric task — different objective from depression classification
- 22-channel BCI dataset — may not generalize to 128-channel resting-state MODMA

---

**Knowledge Extraction — What We Can Borrow:**

- Technique: S-RCT (subject-wise Riemannian centering) as a direct preprocessing step for LODO depression: center each subject's covariance distribution by mapping their session mean to I before pooling across subjects — reduces inter-subject drift without requiring labeled target data
- Technique: Per-subject Riemannian mean → identity alignment is transductive-compatible — works with unlabeled target subject data (compute target subject mean from test trials, apply S-RCT)
- Insight: Parieto-occipital stability supports using parietal/occipital channels for cross-subject depression features (posterior alpha and theta rhythms)
- Architecture: Apply S-RCT alignment before MDRM/TSLDA baselines in our LODO framework — this should be our standard cross-subject alignment preprocessing step

**Knowledge Extraction — What We Should NOT Borrow:**

- Biometric identity task
- 5-channel minimal array — designed for identity, not depression biomarkers
- Cross-task invariance claim specific to motor imagery

---

**Relevance Score:** 5/10

**Reason:** S-RCT is functionally identical to the rebiasing technique (Paper 6) independently validated on a completely different task (brainprint vs. BCI). This convergent evidence from two independent papers strongly supports per-subject Riemannian centering as a robust, universally applicable cross-session/cross-subject alignment technique. The transductive compatibility is particularly valuable — S-RCT can be applied to unlabeled target subject test data in our LODO framework.

---

**Possible Integration:**

- Directly usable: S-RCT as the first preprocessing step in our LODO pipeline — align all source subjects AND the held-out test subject to identity before any classification; no labels required for target subject alignment
- Directly usable: Cross-subject baseline = S-RCT + MDRM (Papers 6+9+21 combined)
- Future experiment: Compare adaptive per-trial S-RCT (Paper 6 adaptive variant) vs. session-level S-RCT on MODMA — measure which granularity of alignment is more effective for depression LODO


---

## Paper 22 — TSMNet/SPDDSMBN: SPD Domain-Specific BN for Unsupervised EEG Domain Adaptation (Relevance: 9/10)

**Title:** SPD Domain-Specific Batch Normalization to Crack Interpretable Unsupervised Domain Adaptation in EEG

**Year:** 2022

**Venue:** NeurIPS 2022

**Dataset:** 6 diverse EEG BCI datasets — BNCI2014001 (9 subjects, 4-class MI), BNCI2014004, BNCI2015001, PhysioNet, Lee2019 EEG dataset, competition dataset (inter-subject TL)

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Enable end-to-end learning of domain-invariant TSM (tangent space mapping) models for multi-source, multi-target unsupervised domain adaptation in EEG BCI, closing the gap to supervised domain-specific methods.

**Research Problem — Why It Matters:**

Neither deep learning UDA methods nor shallow Riemannian alignment methods had consistently closed the performance gap to supervised TSM domain-specific methods. This paper achieves it for the first time via a principled SPD batch normalization building block.

---

**Methodology — Core Idea:**

SPDDSMBN (SPD Domain-Specific Momentum Batch Normalization): a geometric deep learning building block that tracks domain-specific Fréchet means on the SPD manifold during training via momentum updates. Each domain gets its own running Fréchet mean estimate; features are transported to the tangent space at each domain's mean → the domain shift is absorbed into the normalization layer. Combined with DSBN (domain-specific batch norm), SPDDSMBN transforms domain-specific SPD inputs into domain-invariant SPD outputs. TSMNet = feature extractor (subject-specific temporal/spatial filters) → SPDDSMBN layer → LogEig → linear softmax classifier.

**Methodology — Model Architecture:**

- Feature extraction: bandpass EEG → sample covariance matrices (SPD)
- Domain alignment: SPDDSMBN — per-domain Fréchet mean tracking via exponential moving average → transport to tangent space at identity
- Backbone: TSMNet (simple interpretable architecture: temporal filter + spatial filter + SPDDSMBN + LogEig + linear)
- Classifier: linear softmax

**Methodology — Learning Strategy:** Multi-source, multi-target UDA; unsupervised (no target labels); end-to-end trainable

**Methodology — Key Innovation:** First end-to-end learning framework for TSM UDA: SPDMBN tracks Fréchet mean during training; SPDDSMBN absorbs inter-domain shifts geometrically; proven convergence to true Fréchet mean under reasonable assumptions

---

**Evaluation Protocol:**

**Dataset Split:** Leave-5%-of-subjects-out cross-validation (inter-subject TL) OR leave-5%-of-sessions-out (inter-session TL)

**Subject Independence:** Yes (inter-subject TL evaluation)

**Cross-Subject Evaluation:** Yes

**Leave-One-Subject-Out (LODO):** Near-LOSO (5% subjects held out)

**Cross-Dataset Evaluation:** Evaluated on 6 diverse datasets

**Metrics:** Balanced accuracy relative to domain-specific TSM reference; statistical significance

---

**Findings — Main Results:**

- TSMNet achieves SoA across all 6 datasets in inter-subject and inter-session UDA
- First method to consistently match or exceed domain-specific TSM reference (previous methods consistently fell below)
- Ablation confirms: performance gain is primarily driven by DSBN on the SPD manifold (SPDDSMBN), not other architecture choices
- Interpretable: patterns extracted from TSMNet eigenvectors are neurophysiologically meaningful

**Findings — Important Observations:**

- Domain shift in EEG is primarily absorbed in the SPD manifold structure — normalizing at the manifold level outperforms normalizing in tangent space or feature space
- Momentum Fréchet mean tracking converges theoretically and empirically
- SPDDSMBN is a universal building block — pluggable into any architecture using SPD features

**Findings — Failure Cases:**

- Inter-subject UDA harder than inter-session — some datasets show only marginal improvement
- Requires knowing domain labels (subject identity) during adaptation — not applicable when subject identity is unknown
- 6 MI/ERP/SSVEP datasets only — no psychiatric/affective task

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: SPDDSMBN layer as the domain alignment module in our LODO depression framework — per-subject Fréchet mean tracking → transport to tangent at I → domain-invariant features; this is the most principled end-to-end UDA mechanism on the SPD manifold found so far
- Architecture: TSMNet's minimalist interpretable design (temporal filter + spatial filter + SPDDSMBN + LogEig + linear) — should be the baseline architecture for our model before adding complexity
- Training: Multi-source UDA setup (all source subjects → train SPDDSMBN together → test on unseen target subject) maps directly to our LODO protocol
- Theory: Convergence proof for Fréchet mean tracking during end-to-end training — mathematical foundation for including our transductive test-subject alignment

**Knowledge Extraction — What We Should NOT Borrow:**

- No transductive component — target domain is adapted only in the batch norm layer, not via test-time update of target-specific parameters
- Domain label required at test time — in pure domain generalization (no domain ID for new subject) this breaks; need adaptation
- MI/ERP/SSVEP tasks — different spectral signatures from depression resting-state

---

**Relevance Score:** 9/10

**Reason:** TSMNet + SPDDSMBN is the most principled and empirically validated end-to-end cross-subject Riemannian UDA framework in the literature. SPDDSMBN directly addresses the inter-subject Fréchet mean shift that causes LODO performance degradation. This architecture should be the baseline (or core component) for our depression LODO model — adding transductive test-subject Fréchet mean adaptation on top of SPDDSMBN would be our key novel contribution.

---

**Possible Integration:**

- Directly usable: TSMNet + SPDDSMBN as the primary cross-subject UDA baseline in LODO depression on MODMA — train all source subjects together with SPDDSMBN absorbing inter-subject shifts
- Directly usable: SPDDSMBN as a plug-in alignment layer in our architecture after Stiefel projection (Paper 17) and before RHOP head (Paper 3)
- Requires modification: Extend SPDDSMBN to transductive mode — at test time, feed unlabeled target subject trials to update the target-domain Fréchet mean estimate; this is the transductive adaptation novelty
- Future experiment: TSMNet baseline → TSMNet + transductive SPDDSMBN update → full model; ablation on MODMA LODO

---

## Paper 23 — Riemannian DA via Parallel Transport + Moments Alignment (Relevance: 8/10)

**Title:** Domain Adaptation Using Riemannian Geometry of SPD Matrices

**Year:** 2019

**Venue:** ICASSP 2019

**Dataset:** High-density EEG, multiple subjects; electrophysiological signals (specific dataset not named explicitly)

**Modality:** EEG / Electrophysiology (physiological signals)

---

**Research Problem — Main Objective:**

Unsupervised domain adaptation for multi-subject EEG by combining parallel transport (PT) on the SPD manifold with higher-order moments alignment — handling both domain mean differences and structural differences.

**Research Problem — Why It Matters:**

Parallel transport (PT) alone adjusts the Riemannian mean of each domain but ignores within-domain structure. Two subjects can have the same Riemannian mean but very different covariance distributions; PT alone will fail to align them. Moments alignment complements PT by matching higher-order statistics.

---

**Methodology — Core Idea:**

Two-step Riemannian UDA: (1) **Parallel Transport**: for each source domain, compute its Riemannian mean M_s; transport all SPD matrices along the geodesic from M_s to the common reference (target mean M_t) using a congruence transform: P → M_t^{1/2} M_s^{-1/2} P M_s^{-1/2} M_t^{1/2}. This aligns domain means. (2) **Moments Alignment**: project to tangent space at the aligned mean; match higher-order moments (variance, skewness) of the tangent-space vectors across domains via iterative normalization. Combined: PT brings domain centers together; moments alignment equates the shape of the distributions.

**Methodology — Model Architecture:**

- Feature extraction: EEG → sample covariance matrices (SPD)
- Alignment: parallel transport to common reference + higher-order moments matching in tangent space
- Classifier: any standard classifier (SVM, MDM) on aligned tangent-space features

**Methodology — Learning Strategy:** Unsupervised DA (no target labels required)

**Methodology — Key Innovation:** Explicitly separating mean alignment (parallel transport) and structural alignment (moments matching) — showing that PT alone is insufficient when domain structures differ

---

**Evaluation Protocol:**

**Dataset Split:** Cross-subject (train on source subjects, test on unseen target subject)

**Subject Independence:** Yes

**Cross-Subject Evaluation:** Yes

**Leave-One-Subject-Out (LODO):** Implicit LOSO-style evaluation

**Cross-Dataset Evaluation:** No

**Metrics:** Classification accuracy; compared to no-adaptation baseline and PT-only

---

**Findings — Main Results:**

- PT + Moments Alignment outperforms PT-alone baseline
- PT alone fails when source and target have similar means but different distributions
- Moments alignment provides additional accuracy gain beyond mean alignment alone
- Both methods outperform no-adaptation Euclidean baseline

**Findings — Important Observations:**

- This is the foundational paper establishing that multi-order Riemannian statistics are needed for cross-subject EEG DA — not just mean alignment
- Parallel transport is the principled generalization of rebiasing (Paper 6) — rebiasing transports to identity, PT transports to the target mean
- Higher-order moments alignment in tangent space is a natural extension of first-order (mean) alignment

**Findings — Failure Cases:**

- Moments alignment requires target domain data (unlabeled) — transductive, not domain-generalization
- High-dimensional covariance matrices (128+ channels) make moment estimation noisy
- Limited scale — small dataset, no large-scale validation

---

**Knowledge Extraction — What We Can Borrow:**

- Technique: PT + moments alignment as a transductive preprocessing step in LODO: compute source-specific Riemannian means, transport each source subject to target subject's estimated mean, then match variance in tangent space — directly applicable when target (test) subject's unlabeled EEG is available
- Technique: The explicit decomposition of DA into mean alignment + structural alignment provides a diagnostic framework: test if mean shift alone explains MODMA cross-subject variance or if higher-order alignment is needed
- Theory: Parallel transport as the rigorous alternative to simple rebiasing — richer geometric tool for cross-subject alignment

**Knowledge Extraction — What We Should NOT Borrow:**

- Iterated moments matching is expensive at scale (128 channels)
- Small dataset, limited evidence
- No deep learning integration — purely shallow pipeline

---

**Relevance Score:** 8/10

**Reason:** Directly relevant to our transductive LODO setting — the method uses unlabeled target data (test subject's EEG) to compute parallel transport alignment. This is exactly the transductive alignment we want to implement. The theoretical insight that PT+moments > PT-only has direct implications for our design.

---

**Possible Integration:**

- Directly usable: PT + tangent-space variance matching as our primary transductive alignment step — transport all source subjects to test subject's Riemannian mean, then match variance; no labels needed from test subject
- Requires modification: Combine with SPDDSMBN (Paper 22) in an end-to-end framework — SPDDSMBN handles source-source alignment during training; PT handles source-to-target alignment at test time
- Future experiment: Ablation: no alignment → rebiasing (Paper 6) → PT-only → PT+moments → SPDDSMBN+PT in LODO depression

---

## Paper 24 — TSA: Tangent Space Alignment for BCI Transfer Learning (Relevance: 8/10)

**Title:** Tangent Space Alignment: Transfer Learning for Brain-Computer Interface

**Year:** 2022

**Venue:** Frontiers in Human Neuroscience

**Dataset:** 18 BCI databases, 349 total subjects — ERP, MI, SSVEP paradigms; extensive cross-paradigm evaluation

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Reduce BCI calibration time and improve accuracy by aligning EEG data from one subject/session to another in the tangent space of the SPD manifold — enabling cross-subject and cross-session transfer learning.

**Research Problem — Why It Matters:**

Per-session calibration is expensive; existing approaches (Riemannian Procrustes Analysis, rebiasing) either lack the full tangent-space perspective or are limited in expressiveness. TSA provides a principled, unified framework.

---

**Methodology — Core Idea:**

TSA: for each source-target pair, compute the Riemannian mean of source (M_source) and target (M_target) respectively. Map all source covariances to tangent space at M_source → apply an affine alignment transform in tangent space → map to tangent space at M_target (concatenating log maps). This produces source features that appear to come from the target distribution in tangent space. The alignment transform is estimated from (unlabeled) target data + labeled source data. Handles different number of channels via manifold structure.

**Methodology — Model Architecture:**

- Feature extraction: EEG epochs → sample covariance matrices
- Alignment: TSA (tangent space affine alignment between source and target distributions)
- Classifier: SVM on aligned tangent-space features

**Methodology — Learning Strategy:** Transfer learning (labeled source, unlabeled target for alignment)

**Methodology — Key Innovation:** Full tangent-space alignment (not just mean recentering); intrinsic ability to handle inter-dataset transfer with different channel counts; significant improvement over RPA (Riemannian Procrustes Analysis) baseline

---

**Evaluation Protocol:**

**Dataset Split:** Cross-subject and cross-session TL on 18 databases; intra- and inter-dataset

**Subject Independence:** Yes

**Cross-Subject Evaluation:** Yes

**Leave-One-Subject-Out (LODO):** Evaluated in LOSO-style across 18 databases

**Cross-Dataset Evaluation:** Yes (inter-dataset TL with different channel counts)

**Metrics:** Classification accuracy; comparison to no-TL, rebiasing, RPA, other TL methods

---

**Findings — Main Results:**

- TSA: +2.7% accuracy over RPA (previously best Riemannian TL method) across 18 databases
- Significant improvement on ERP paradigm; no degradation on MI and SSVEP
- Handles cross-dataset TL (different number of channels) naturally via manifold structure
- Consistent improvement across paradigms — robust generalization

**Findings — Important Observations:**

- ERP paradigm benefits most from TSA — event-related components are more consistent across subjects than oscillatory activity
- The ability to handle different channel counts is unique — other TL methods require same channels; TSA works because tangent space is symmetric matrix space, not channel space
- TSA subsumes rebiasing as a special case (rebiasing = TSA with identity target mean)

**Findings — Failure Cases:**

- MI paradigm: marginal improvement — oscillatory patterns more variable across subjects
- Target subject still needs some unlabeled data for mean estimation — not truly zero-calibration
- 18 databases all from healthy BCI subjects — no psychiatric validation

---

**Knowledge Extraction — What We Can Borrow:**

- Technique: TSA as the primary cross-subject alignment method in LODO depression — map source subjects to target subject's estimated tangent space; more general than rebiasing
- Architecture: Full tangent-space alignment (not just mean shift) is the right operation — mean recentering alone leaves variance/covariance structure unaligned
- Practical: Cross-dataset TL capability (different channel counts) is critical if we want to transfer from standard BCI datasets to MODMA (128 channels vs 22-channel BCI-IV datasets)
- Evaluation: 18-database evaluation framework as inspiration for our cross-dataset depression analysis

**Knowledge Extraction — What We Should NOT Borrow:**

- SVM classifier — should be replaced with RHOP or SPDDSMBN-based end-to-end model
- No deep learning integration — purely shallow
- ERP-focused improvements — depression EEG is resting-state, similar to MI in variability

---

**Relevance Score:** 8/10

**Reason:** TSA is a principled, extensively validated cross-subject tangent-space alignment method. It is the most general form of the rebiasing approach (Paper 6) and directly applicable to our LODO depression setting. The 18-database evaluation provides strong evidence that TSA generalizes across paradigms.

---

**Possible Integration:**

- Directly usable: TSA as preprocessing before MDRM/TSLDA/SPDDSMBN in LODO depression — align source subjects to test subject's tangent space before classification
- Directly usable: TSA + SPDDSMBN (Paper 22) as the combined shallow-deep alignment pipeline
- Future experiment: Compare TSA (shallow) vs. SPDDSMBN (end-to-end deep) vs. TSA+SPDDSMBN on MODMA LODO

---

## Paper 25 — RGT-GAA: Riemannian Graph Transformer with Geodesic Adversarial Adaptation (Relevance: 7/10)

**Title:** Cross-Subject EEG Emotion Recognition Using Riemannian Graph Transformers with Geodesic Adversarial Adaptation

**Year:** 2026

**Venue:** Alexandria Engineering Journal

**Dataset:** SEED (45 subjects, 3-class emotion), SEED-IV (45 subjects, 4-class emotion)

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Cross-subject EEG emotion recognition combining Riemannian geometry (SPD covariance) with graph-based attention and geodesic adversarial domain adaptation.

**Research Problem — Why It Matters:**

Conventional deep learning flattens SPD covariance matrices losing geometry; existing adversarial DA operates in Euclidean space losing manifold structure; no prior work combines graph attention, Riemannian geometry, and geodesic adversarial alignment for cross-subject emotion.

---

**Methodology — Core Idea:**

RGT-GAA: (1) **Log-Euclidean Graph Transformer**: project covariance matrices to tangent space → construct a distance-based graph (edges between subjects/channels with small geodesic distance) → multi-head attention over graph nodes preserving geodesic relationships. (2) **Geodesic Adversarial Adaptation Network (GAA)**: domain adversarial training where the discriminator loss is based on Log-Euclidean distance between source and target distributions rather than Euclidean domain classifier loss — geometrically faithful cross-subject alignment.

**Methodology — Model Architecture:**

- Feature extraction: EEG → covariance matrices → Log map to tangent space
- Graph construction: k-NN graph based on Log-Euclidean distance between covariance matrices
- Backbone: Graph Transformer with manifold-aware attention (distance-weighted)
- Adaptation: GAA — geodesic distance-based adversarial loss to minimize source-target distributional gap
- Classifier: softmax on adapted features

**Methodology — Learning Strategy:** Unsupervised domain adaptation (adversarial); cross-subject

**Methodology — Key Innovation:** Geodesic adversarial loss (instead of standard GAN-style domain adversarial loss) — aligns source and target distributions in the Riemannian metric rather than Euclidean space

---

**Evaluation Protocol:**

**Dataset Split:** Cross-subject evaluation (leave-one-subject-out style)

**Subject Independence:** Yes

**Cross-Subject Evaluation:** Yes

**Leave-One-Subject-Out (LODO):** Yes

**Cross-Dataset Evaluation:** No

**Metrics:** Classification accuracy

---

**Findings — Main Results:**

- SEED: outperforms strong cross-subject baselines (DANN, MIDA, EEGNet+DA)
- SEED-IV: state-of-the-art among Riemannian + graph methods for cross-subject emotion
- Geodesic adversarial loss consistently better than Euclidean adversarial loss

**Findings — Important Observations:**

- Geodesic adversarial loss significantly contributes to improvement — geometry-aware alignment matters for cross-subject emotion
- Graph construction using geodesic distance identifies functionally related channels across subjects more reliably than Euclidean distance
- Log-Euclidean Graph Transformer preserves inter-subject geometric relationships better than standard attention

**Findings — Failure Cases:**

- SEED emotion datasets (stimulus-evoked, not resting-state) — different from depression EEG
- No explicit test on non-emotional psychiatric tasks
- Geodesic adversarial loss requires careful balancing of classification vs. alignment objectives

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: Geodesic adversarial adaptation loss — replace Euclidean domain discriminator with Log-Euclidean distance-based discriminator for cross-subject depression alignment; more geometrically faithful
- Architecture: Distance-based graph construction using geodesic distances between subject covariances — identifies truly similar subjects for graph aggregation, not just spatially proximate electrodes
- Training: Adversarial alignment using Riemannian metric is a generalizable technique directly applicable to our LODO depression framework as an optional adversarial alignment stage

**Knowledge Extraction — What We Should NOT Borrow:**

- SEED stimulus-evoked emotion paradigm
- Full graph transformer complexity — may be overkill for small MODMA dataset
- Log-Euclidean (not AIRM) as primary metric — AIRM is geometrically superior; use geodesic distances from AIRM

---

**Relevance Score:** 7/10

**Reason:** Directly addresses cross-subject EEG emotion recognition (adjacent to depression detection) with a novel geodesic adversarial alignment mechanism. The paradigm shift from Euclidean to geodesic adversarial loss has direct applicability to our LODO depression framework.

---

**Possible Integration:**

- Requires modification: Geodesic adversarial adaptation layer as an optional alignment component after SPDDSMBN — minimize geodesic (AIRM) distance between source and target distributions during training, while target is unlabeled
- Future experiment: SPDDSMBN baseline vs. SPDDSMBN + geodesic adversarial adaptation on MODMA LODO

---

## Paper 26 — Riemannian DA via Stiefel Routing: Adaptive Subspace Selection (Relevance: 7/10)

**Title:** Routing on the Stiefel Manifold: When Does Adaptive Subspace Selection Help for Cross-Domain EEG Decoding?

**Year:** 2026

**Venue:** arXiv preprint (stat.ML) — GIPSA-lab, Grenoble

**Dataset:** Multiple EEG BCI datasets (MI paradigm); cross-subject evaluation

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Determine when and how adaptive subspace selection (routing among multiple Stiefel manifold experts) improves cross-domain EEG decoding, and identify the conditions necessary to prevent degeneracy collapse.

**Research Problem — Why It Matters:**

Single fixed Stiefel projections fail for cross-subject EEG where different subjects occupy different regions of the SPD manifold. A pool of expert projections with subject-adaptive routing is more expressive, but naive routing collapses to ensemble averaging.

---

**Methodology — Core Idea:**

K expert projection filters on the Stiefel manifold, each specialized for a different SPD manifold region. Input covariance → cross-attention routing assigns it to the most appropriate filter. Key finding: **naive routing provably collapses** to ensemble averaging when routing weights are uniform. Three structural properties prevent collapse: (1) symmetric anchor W_base ∈ St(n,k) to remove proximity bias; (2) frozen domain-discriminative query encoder — learns subject-cluster assignments; (3) diversity regularization on expert filters to prevent filter collapse. With these properties, routing provides genuine per-sample subspace adaptation.

**Methodology — Model Architecture:**

- K Stiefel expert filters (learnable orthonormal projections)
- Cross-attention routing: query = frozen domain encoder(subject); keys = expert filter representations; routing weights
- Feature extraction: covariance → selected Stiefel projection → reduced SPD → tangent space
- Classifier: MDM or TSM-based

**Methodology — Learning Strategy:** Multi-source cross-domain supervised training

**Methodology — Key Innovation:** Theoretical analysis of routing collapse; three sufficient conditions to enable genuine adaptive routing on Stiefel manifold

---

**Evaluation Protocol:**

**Dataset Split:** Cross-subject (leave-one-subject-out)

**Subject Independence:** Yes

**Cross-Subject Evaluation:** Yes

**Leave-One-Subject-Out (LODO):** Yes

**Cross-Dataset Evaluation:** Evaluated on multiple MI datasets

**Metrics:** Classification accuracy; ablation of routing conditions

---

**Findings — Main Results:**

- Routing without collapse conditions: degenerates to equal-weight ensemble (no benefit)
- Routing with all three conditions: consistent improvement over single fixed Stiefel projection
- Frozen domain encoder is the most critical component — without it, routing collapses

**Findings — Important Observations:**

- The theoretical result (routing collapses without diversity) is an important design insight: any mixture-of-experts SPD architecture needs diversity regularization
- Adaptive routing per sample (not per domain) is necessary — averaging routing weights per subject still helps less than per-sample routing

**Findings — Failure Cases:**

- Requires knowing subject domain at routing time — routing encoder needs to identify subject cluster
- Computational overhead of K experts vs. single filter
- K-expert routing provides diminishing returns when source domain size is large

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: K Stiefel expert projection pool with cross-attention routing as a subject-adaptive dimensionality reduction front-end for our LODO pipeline — different test subjects are routed to different subspaces
- Theory: Routing collapse conditions are a general design principle for any mixture-of-experts architecture on SPD manifolds — apply when designing multi-expert components
- Technique: Frozen domain-discriminative query encoder for routing — compute a subject embedding from unlabeled target EEG (transductively) to route to appropriate experts; zero labels needed

**Knowledge Extraction — What We Should NOT Borrow:**

- MI task; K experts may be overkill for small MODMA (53 subjects)
- Full routing complexity for a first model iteration

---

**Relevance Score:** 7/10

**Reason:** The theoretical insight about routing collapse + diversity conditions applies directly to any multi-expert SPD architecture we design. The adaptive Stiefel routing idea is relevant for handling the diverse subject subpopulations in MODMA depression data.

---

**Possible Integration:**

- Requires modification: Adapt Stiefel routing as a transductive front-end — use test subject's unlabeled covariances to route to the most appropriate expert subspace; train experts on source subjects
- Future experiment: Single Stiefel projection (Paper 17) vs. K-expert routing (this paper) in LODO depression

---

## Paper 27 — Geometric Moment Alignment via Siegel Embeddings (Relevance: 5/10)

**Title:** Geometric Moment Alignment for Domain Adaptation via Siegel Embeddings

**Year:** 2026

**Venue:** ICLR 2026 (under review at time of preprint)

**Dataset:** Image denoising and image classification benchmarks (not EEG)

**Modality:** Images (not physiological signals — methodology is transferable)

---

**Research Problem — Main Objective:**

Principled moment matching for UDA using Riemannian distances on the SPD manifold — encoding both first- and second-order moments as a single SPD matrix (Siegel embedding) for joint geometric alignment.

---

**Methodology — Core Idea:**

Express both mean μ and covariance Σ as a single SPD matrix via Siegel embedding: augmented matrix M = [[Σ + μμᵀ, μ], [μᵀ, 1]] — first and second moments combined in one SPD object. The Riemannian distance on this augmented manifold measures both mean and covariance shift simultaneously. Domain adaptation = minimize Riemannian distance between source and target Siegel matrices.

**Methodology — Key Innovation:** Siegel embedding unifies mean and covariance into one SPD object; geometric distance is theoretically connected to target-domain error bound

---

**Evaluation Protocol:** Image benchmarks; no EEG/BCI evaluation

**Subject Independence:** N/A (image domain shift, not EEG)

---

**Findings — Main Results:**

- Outperforms standard moment-matching (Euclidean) on image DA benchmarks
- Siegel embedding with Riemannian distance provides more faithful cross-domain comparison

---

**Knowledge Extraction — What We Can Borrow:**

- Technique: Siegel embedding to unify first- and second-order EEG covariance statistics into a single SPD matrix for cross-subject alignment — applicable to MODMA LODO where we want to align both mean and variance of source/target covariance distributions

**Knowledge Extraction — What We Should NOT Borrow:**

- Image benchmarks — no EEG validation
- Full Siegel manifold geometry is complex; overhead may not be justified

---

**Relevance Score:** 5/10

**Reason:** Elegant mathematical framework applicable to our cross-subject alignment problem. The Siegel embedding idea provides a unified moment-matching objective without separate mean and variance alignment steps.

---

## Paper 28 — SFDA Pseudo-Labeling: Source-Free Domain Adaptation for Privacy-Preserving EEG BCI (Relevance: 5/10)

**Title:** Transformer-Based SFDA by Class-Balanced Multicentric Dynamic Pseudo-Labeling for Privacy-Preserving EEG-Based BCI Systems

**Year:** 2026

**Venue:** MDPI journal (Hangzhou Dianzi University)

**Dataset:** BCI Competition IV 2a; SEED

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Source-free domain adaptation (SFDA) for EEG BCI — adapt to target subject without accessing source subject data; only the pre-trained model is available (privacy-preserving).

**Research Problem — Why It Matters:**

Privacy regulations may prevent sharing subject EEG data; SFDA allows adaptation using only the model. Existing EEG SFDA methods suffer from pseudo-label noise and class imbalance during adaptation.

---

**Methodology — Core Idea:**

Transformer-based SFDA: pre-trained source model → deploy at target without source data → use class-balanced multicentric dynamic pseudo-labeling on target data to adapt. Multiple class centroids (not one per class) capture within-class diversity; dynamic updating prevents centroid drift; class-balancing prevents majority-class pseudo-label dominance.

**Methodology — Key Innovation:** Multicentric (multiple prototypes per class) + dynamic + class-balanced pseudo-labeling for stable SFDA

---

**Evaluation Protocol:**

**Subject Independence:** Yes

**Cross-Subject Evaluation:** Yes

**Leave-One-Subject-Out (LODO):** Partial (target-only adaptation without source data)

**Metrics:** Accuracy

---

**Findings — Main Results:**

- Outperforms standard SFDA and some source-available DA methods
- Class-balanced multicentric pseudo-labeling reduces adaptation noise
- Privacy-preserving — no source data needed at adaptation time

---

**Knowledge Extraction — What We Can Borrow:**

- Technique: Multicentric class prototypes (multiple Riemannian means per class, not one) for capturing within-class variance — relevant for depression EEG where MDD subtypes may form distinct clusters
- Technique: SFDA paradigm — privacy-preserving adaptation; if MODMA data cannot be shared across sites, SFDA is the appropriate protocol
- Training: Class-balanced pseudo-labeling for test-time adaptation — directly applicable to transductive LODO where target labels are unknown

**Knowledge Extraction — What We Should NOT Borrow:**

- Non-Riemannian transformer — no manifold geometry used in adaptation
- Pseudo-label reliability is a challenge — unstable with few target samples

---

**Relevance Score:** 5/10

**Reason:** SFDA paradigm is directly relevant to privacy-constrained LODO scenarios. Multicentric class prototypes and class-balanced pseudo-labeling are useful techniques for test-time transductive adaptation.

---

**Possible Integration:**

- Requires modification: Replace Euclidean centroids with Riemannian means (MDRM-style) in the multicentric pseudo-labeling scheme for test-time transductive adaptation on MODMA
- Future experiment: Riemannian multicentric pseudo-labeling as a test-time label propagation step in our transductive LODO framework

---

## Paper 29 — Cross-Subject EEG Zero-Calibration via Spatiotemporal Transformers (Relevance: 5/10)

**Title:** Cross-Subject EEG Decoding with Spatiotemporal Transformers for Zero-Calibration Brain-Computer Interfaces

**Year:** 2026

**Venue:** Alexandria Engineering Journal

**Dataset:** Multiple MI datasets; zero-calibration evaluation

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Zero-calibration cross-subject EEG decoding using spatiotemporal transformer with domain adaptation — no target subject data required at training time.

---

**Methodology — Core Idea:**

Spatiotemporal Transformer trained on multiple source subjects with domain adversarial alignment. Cross-subject attention mechanism aligns spatial-temporal patterns across subjects. Zero-calibration inference: deploy directly on unseen test subject.

**Methodology — Key Innovation:** Zero-calibration spatiotemporal transformer with cross-subject attention

---

**Evaluation Protocol:**

**Subject Independence:** Yes

**Cross-Subject Evaluation:** Yes

**Leave-One-Subject-Out (LODO):** Yes

---

**Findings — Main Results:**

- Competitive zero-calibration performance vs. calibration-requiring baselines
- Cross-subject attention improves generalization vs. standard self-attention

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: Cross-subject attention mechanism for spatial-temporal EEG encoding — explicitly learns which temporal patterns are shared across subjects
- Evaluation: Zero-calibration protocol (no target data at all) as the strictest evaluation baseline; distinguish from transductive (has unlabeled target data)

**Knowledge Extraction — What We Should NOT Borrow:**

- No Riemannian geometry — Euclidean transformer only
- MI task; no psychiatric application

---

**Relevance Score:** 5/10

**Reason:** Zero-calibration setting is relevant for our LODO framework (no target labels). Cross-subject attention idea is useful.

---

## Paper 30 — Transfer Learning for Riemannian Tangent Space in BCI (Bleuzé 2021) (Relevance: 6/10)

**Title:** Transfer Learning for the Riemannian Tangent Space: Applications to Brain-Computer Interfaces

**Year:** 2021

**Venue:** ICEET 2021

**Dataset:** Multiple BCI databases (ERP, MI paradigms); conference version of TSA

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Initial presentation of the tangent space alignment (TSA) concept for BCI transfer learning — conference version preceding the full Frontiers paper (Paper 24).

---

**Methodology — Core Idea:**

Same as Paper 24 (TSA) but evaluated on fewer databases; establishes the foundational TSA concept. Cross-subject alignment by mapping source covariances to the tangent space at the target subject's Riemannian mean.

**Methodology — Key Innovation:** Initial proposal of TSA framework

---

**Evaluation Protocol:**

**Subject Independence:** Yes

**Cross-Subject Evaluation:** Yes

**Leave-One-Subject-Out (LODO):** Partial

---

**Knowledge Extraction — What We Can Borrow:** Same as Paper 24 (TSA) — this is the earlier, shorter version. Refer to Paper 24 for the complete framework.

**Knowledge Extraction — What We Should NOT Borrow:** Same limitations as Paper 24.

---

**Relevance Score:** 6/10

**Reason:** Conference precursor to Paper 24. Establishes the TSA concept — refer to Paper 24 for implementation details.

---

## Paper 31 — Selective Cross-Subject Transfer via Riemannian Tangent Space (Relevance: 5/10)

**Title:** Selective Cross-Subject Transfer Learning Based on Riemannian Tangent Space for Motor Imagery Brain-Computer Interface

**Year:** 2021

**Venue:** Frontiers in Neuroscience

**Dataset:** BCI Competition IV 2a (9 subjects, 4-class MI)

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Cross-subject MI classification using Riemannian tangent space features with selective transfer — choose source subjects most similar to the target subject for knowledge transfer.

**Research Problem — Why It Matters:**

Indiscriminate use of all source subjects can hurt performance when some subjects are highly different from the target (negative transfer).

---

**Methodology — Core Idea:**

Project all subjects' covariance matrices to a common tangent space (common Riemannian mean). Measure source-target similarity by comparing tangent-space distributions (e.g., Maximum Mean Discrepancy or KL divergence). Select K most similar source subjects. Train classifier on selected subjects + adapt to target using tangent space alignment.

**Methodology — Key Innovation:** Selective transfer based on Riemannian tangent space similarity to avoid negative transfer

---

**Evaluation Protocol:**

**Subject Independence:** Yes

**Cross-Subject Evaluation:** Yes

**Leave-One-Subject-Out (LODO):** Yes (9-subject LOSO on BCI-IV 2a)

**Metrics:** Accuracy, kappa

---

**Findings — Main Results:**

- Selective transfer outperforms using all subjects when some sources are dissimilar
- Riemannian tangent space similarity is a better selection criterion than Euclidean distance

**Findings — Important Observations:**

- Source selection is as important as the adaptation method — negative transfer is a real problem in cross-subject EEG
- K ≈ 5-7 most similar subjects is optimal (not all N-1)

---

**Knowledge Extraction — What We Can Borrow:**

- Technique: Source subject selection by Riemannian tangent space similarity in LODO depression — select most similar source subjects to reduce negative transfer; Riemannian distance between subject means as the selection criterion
- Design insight: Not all source subjects help — a subject-selection layer should precede training in LODO

**Knowledge Extraction — What We Should NOT Borrow:**

- Within-MI paradigm; small 9-subject dataset
- K selection requires ground-truth validation — needs held-out set

---

**Relevance Score:** 5/10

**Reason:** Source subject selection is directly relevant to our LODO framework. Negative transfer from dissimilar subjects is a real concern in MODMA where MDD subtypes may vary across subjects.

---

**Possible Integration:**

- Directly usable: Riemannian tangent-space similarity as source subject selection criterion in LODO depression — select K most similar source subjects to the held-out test subject
- Future experiment: All source subjects vs. K-most-similar vs. domain-adversarial selection in MODMA LODO

---

## Paper 32 — Self-Attention DA with Riemannian + Contrastive Learning for Drowsiness (Relevance: 5/10)

**Title:** A Self-Attention Domain Adaptation Network Based on Riemannian Representation and Contrastive Learning for EEG Driver Drowsiness Detection

**Year:** 2026

**Venue:** Engineering Science and Technology

**Dataset:** Drowsiness EEG dataset (multi-subject)

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Cross-subject driver drowsiness detection using Riemannian covariance features + contrastive learning domain adaptation.

---

**Methodology — Core Idea:**

Self-attention encoder on Riemannian tangent-space features → contrastive domain adaptation loss to align source and target subject distributions while preserving class-discriminative structure. Subject-level contrastive pairs: same-class/different-subject vs. different-class pairs.

**Methodology — Key Innovation:** Contrastive learning for cross-subject Riemannian domain adaptation in drowsiness detection

---

**Evaluation Protocol:**

**Subject Independence:** Yes

**Cross-Subject Evaluation:** Yes

**Leave-One-Subject-Out (LODO):** Yes

**Metrics:** Accuracy; F1 for drowsiness vs. alert classification

---

**Findings — Main Results:**

- Contrastive + Riemannian outperforms Riemannian-only and contrastive-only baselines
- Drowsiness detection is neurologically adjacent to depression (fatigue, alertness) — similar EEG patterns

---

**Knowledge Extraction — What We Can Borrow:**

- Training: Contrastive loss for cross-subject alignment — subject-invariant representation learning: pull same-class features from different subjects together, push different-class features apart; directly applicable to LODO depression
- Architecture: Self-attention on Riemannian tangent-space features — attention over the vectorized SPD features preserves geometry while learning temporal context

**Knowledge Extraction — What We Should NOT Borrow:**

- Drowsiness-specific features; driving paradigm vs. resting-state

---

**Relevance Score:** 5/10

**Reason:** Contrastive learning for cross-subject Riemannian alignment in an affect-adjacent task (drowsiness). The contrastive objective is directly applicable to depression LODO.

---

**Possible Integration:**

- Requires modification: Add contrastive loss on Riemannian tangent-space features to our LODO depression training — pull MDD features from different source subjects together; push MDD-vs-HC apart
- Future experiment: SPDDSMBN baseline vs. SPDDSMBN + contrastive cross-subject loss on MODMA LODO

---

## Paper 33 — BARN-DA: Band-Aware Riemannian Network with Domain Adaptation (Relevance: 5/10)

**Title:** A Band-Aware Riemannian Network with Domain Adaptation for Motor Imagery EEG Signal Decoding

**Year:** 2026

**Venue:** MDPI (Hangzhou Dianzi University)

**Dataset:** BCI Competition IV 2a, 2b; two additional MI datasets; cross-session and cross-subject

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

End-to-end MI decoding with frequency-band-aware features + Riemannian manifold mapping + R-MMD domain adaptation loss for cross-session/subject generalization.

---

**Methodology — Core Idea:**

BARN-DA: (1) BACA (Band-Aware Channel Attention) — models channel-band feature interactions, enhancing discriminative channels per frequency band. (2) MSKP (Multi-Scale Kernel Perception) — multi-scale CNN for spatio-temporal feature extraction. (3) Riemannian manifold mapping: extracted features → SPD matrices → tangent space. (4) R-MMD loss: Riemannian Maximum Mean Discrepancy using log-Euclidean kernel to minimize source-target SPD distribution distance.

**Methodology — Key Innovation:** R-MMD loss (MMD defined on SPD manifold using log-Euclidean kernel) for domain alignment — geometrically faithful MMD

---

**Evaluation Protocol:**

**Subject Independence:** Yes

**Cross-Subject Evaluation:** Yes

**Leave-One-Subject-Out (LODO):** Cross-session + cross-subject

**Metrics:** Accuracy, kappa

---

**Findings — Main Results:**

- BCI-IV 2a cross-session: 84.65% ± 8.97%
- BCI-IV 2b: 89.19% ± 7.69%
- R-MMD consistently better than Euclidean MMD

---

**Knowledge Extraction — What We Can Borrow:**

- Loss: R-MMD (Riemannian MMD using log-Euclidean kernel) as an unsupervised domain alignment objective — minimize SPD distribution distance between source and target subjects in LODO
- Architecture: Band-Aware Channel Attention for depression EEG — frequency-band-specific channel weighting aligns with the band-specific depression biomarkers (alpha asymmetry, theta elevation)

**Knowledge Extraction — What We Should NOT Borrow:**

- MI task; R-MMD alone without manifold-preserving architecture
- No LODO evaluation (session-to-session not LOSO)

---

**Relevance Score:** 5/10

**Reason:** R-MMD is a useful distribution alignment objective for our LODO framework. Band-aware attention is relevant for depression's frequency-band-specific biomarkers.

---

## Paper 34 — FB-RCSP-RA: Filter-Bank CSP with Per-Band Riemannian Alignment (Relevance: 4/10)

**Title:** FB-RCSP-RA: A Filter-Bank Regularized CSP Framework with Per-Band Riemannian Alignment for Cross-Subject Motor Imagery EEG Decoding

**Year:** 2026

**Venue:** HAL preprint

**Dataset:** BCI Competition IV (MI); cross-subject evaluation

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Improve cross-subject MI decoding by applying Riemannian alignment independently in each frequency band before CSP feature extraction.

---

**Methodology — Core Idea:**

Apply TSA/rebiasing alignment separately in each frequency band (not across the full-spectrum covariance). Per-band Riemannian alignment compensates for subject-specific spectral differences before combining across bands.

**Methodology — Key Innovation:** Per-band (not full-spectrum) Riemannian alignment

---

**Knowledge Extraction — What We Can Borrow:**

- Technique: Band-specific Riemannian alignment — align source-to-target separately in delta, theta, alpha, beta bands before pooling; more precise than broad-spectrum alignment for depression EEG with band-specific biomarkers

---

**Relevance Score:** 4/10

**Reason:** Per-band alignment idea is useful but the paper is a minor extension of TSA applied band-specifically. The core idea (per-band alignment) can be combined with our framework.

---

## Paper 35 — Rotation-Based Metric on SPD for Source Data Selection (Relevance: 4/10)

**Title:** Rotation-Based Metric on the Riemannian Manifold of SPD Matrices with Applications to Source Data Selection for BCI Transfer Learning

**Year:** 2026

**Venue:** Frontiers in Human Neuroscience

**Dataset:** BCI MI datasets; source data selection evaluation

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Propose a rotation-based metric on the SPD manifold for measuring subject similarity, applied to source data selection for BCI transfer learning.

---

**Methodology — Core Idea:**

Rotation-based metric: distance between two SPD matrices defined by the rotation needed to transform one eigenvector basis into the other. Rotation-based distance captures principal component similarity (orientation of connectivity patterns) rather than scaling difference (eigenvalue magnitude). Applied to select source subjects with similar eigenvector structure to the target subject.

**Methodology — Key Innovation:** Rotation-based SPD metric emphasizing eigenvector orientation over eigenvalue magnitude — different from AIRM which weights both

---

**Knowledge Extraction — What We Can Borrow:**

- Metric: Rotation-based distance as a source selection criterion complementary to AIRM distance — selects subjects with similar brain connectivity orientation, not just similar signal power
- Application: Source subject selection in LODO using rotation-based similarity to test subject's covariance

**Knowledge Knowledge — What We Should NOT Borrow:** MI task; no psychiatric validation

---

**Relevance Score:** 4/10

**Reason:** Novel source selection metric with direct application to LODO subject selection. Complementary to Paper 31 (tangent-space MMD selection).

---

## Paper 36 — DGFEL: Domain Generalized Feature Embedded Learning for Calibration-Free ERP BCI (Relevance: 5/10)

**Title:** Domain Generalized Feature Embedded Learning for Calibration-Free Event-Related Potentials Recognition

**Year:** 2026

**Venue:** Cognitive Neurodynamics

**Dataset:** Two public ERP datasets; multi-subject calibration-free evaluation

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Calibration-free ERP recognition across subjects using domain generalization — train on source subjects, deploy on unseen target without any calibration data.

---

**Methodology — Core Idea:**

DGFEL: (1) align each subject's ERPs by covariance centroid alignment (equivalent to TSA/rebiasing); (2) xDAWN spatial filtering → spatio-temporal feature extraction; (3) adversarial domain generalization loss on features to maximize cross-subject invariance; (4) neural network backbone for classification.

**Methodology — Key Innovation:** Covariance centroid alignment + adversarial domain generalization in a single calibration-free pipeline; no target data needed (true domain generalization, not DA)

---

**Evaluation Protocol:**

**Subject Independence:** Yes

**Cross-Subject Evaluation:** Yes

**Leave-One-Subject-Out (LODO):** Yes

**Cross-Dataset Evaluation:** No

**Metrics:** Accuracy

---

**Findings — Main Results:**

- Outperforms several state-of-the-art methods without any target calibration data
- Covariance centroid alignment is critical — without it, adversarial DG alone is insufficient
- Works for both ERP datasets (P300, N200)

---

**Knowledge Extraction — What We Can Borrow:**

- Insight: Domain generalization (no target data) combined with covariance alignment is feasible — relevant as a zero-calibration baseline for LODO depression before adding transductive components
- Architecture: Adversarial DG on top of covariance-aligned features as a subject-invariant representation learning objective

**Knowledge Extraction — What We Should NOT Borrow:**

- ERP paradigm (stimulus-evoked), not resting-state depression
- xDAWN filter (ERP-specific)

---

**Relevance Score:** 5/10

**Reason:** Demonstrates covariance alignment + adversarial DG achieves calibration-free cross-subject ERP recognition. The pipeline (align → adversarial DG → classify) is directly applicable to our LODO framework as a zero-calibration DG baseline.

---

## Paper 37 — Cross-Dataset Variability Problem in EEG Deep Learning (Relevance: 3/10)

**Title:** Cross-Dataset Variability Problem in EEG Decoding with Deep Learning

**Year:** 2020

**Venue:** Frontiers in Human Neuroscience

**Dataset:** Multiple EEG BCI datasets; cross-dataset evaluation

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Characterize and analyze the cross-dataset variability problem in deep learning EEG decoders — why models trained on one dataset/subject fail on others.

---

**Methodology — Core Idea:**

Systematic study of cross-dataset performance degradation in EEGNet, DeepConvNet, ShallowConvNet. Analyzes sources of variability: different electrode configs, sampling rates, mental task differences, subject population differences.

---

**Knowledge Extraction — What We Can Borrow:**

- Analysis: Domain variability taxonomy for EEG (electrode, sampling, population) — use as a checklist when designing MODMA cross-subject evaluation
- Baseline: Standard deep learning models perform poorly cross-dataset without alignment — confirms need for Riemannian geometry

---

**Relevance Score:** 3/10

**Reason:** Background analysis paper; confirms the problem we are solving but provides no new methods.

---

## Paper 38 — Latent Alignment in Deep Learning for EEG Decoding (Relevance: 4/10)

**Title:** Latent Alignment in Deep Learning Models for EEG Decoding

**Year:** 2025

**Venue:** Journal of Neural Engineering

**Dataset:** Multiple EEG datasets; inter-subject alignment evaluation

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Systematic evaluation of latent-space alignment methods for cross-subject EEG deep learning — comparing different alignment objectives in the network's latent space.

---

**Methodology — Core Idea:**

Compare latent alignment objectives (MMD, adversarial, contrastive, mean centering) applied to the intermediate layers of deep EEG models. Analysis of which alignment layer and objective works best across datasets.

---

**Knowledge Extraction — What We Can Borrow:**

- Empirical finding: Contrastive and adversarial objectives outperform MMD for cross-subject alignment in deep learning — confirms contrastive approach (Paper 32) is strong
- Architecture: Latent alignment should be applied at multiple network depths, not just the output layer

---

**Relevance Score:** 4/10

**Reason:** Useful empirical comparison for choosing alignment objectives in our deep model.

---

## Paper 39 — Source Data Selection Based on Simple Features (Relevance: 4/10)

**Title:** Source Data Selection for Brain-Computer Interfaces Based on Simple Features

**Year:** 2026

**Venue:** IEEE Access

**Dataset:** Multiple BCI MI datasets

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Lightweight source subject selection method (Transfer Performance Predictor, TPP) for BCI — select the most useful source subjects for a given target subject using simple, fast features.

---

**Methodology — Core Idea:**

TPP: predict which source subjects will transfer well to a given target subject using simple features (channel PSD ratios, covariance matrix statistics) computed quickly without full classifier training. Outperforms exhaustive search and Riemannian distance-based selection in computational efficiency.

---

**Knowledge Extraction — What We Can Borrow:**

- Practical: Source selection via simple covariance statistics (fast TPP) before expensive LODO training — relevant for MODMA where 52 source subjects may include poor transfers

---

**Relevance Score:** 4/10

**Reason:** Source selection is relevant but the simple-features approach may not outperform Riemannian distance selection (Papers 31, 35) for our geometric pipeline.

---

## Paper 40 — Dual Selections Knowledge Transfer for Cross-Subject MI (Relevance: 3/10)

**Title:** Dual Selections Based Knowledge Transfer Learning for Cross-Subject Motor Imagery EEG Classification

**Year:** 2023

**Venue:** Frontiers in Neuroscience

**Dataset:** BCI IV 2a; cross-subject evaluation

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Knowledge distillation-based cross-subject transfer with dual selection: select both useful source subjects and useful features for knowledge transfer.

---

**Knowledge Extraction — What We Can Borrow:**

- Concept: Dual selection (subject selection + feature selection) as a principled approach to cross-subject knowledge transfer; combine with Riemannian feature selection (ANOVA in tangent space, Paper 9)

---

**Relevance Score:** 3/10

---

## Paper 41 — Maximizing Single-Feature Separability for MI Transfer Learning (Relevance: 2/10)

**Title:** Maximizing Single-Feature Separability for Improving Transfer Learning in Motor Imagery EEG Decoding

**Year:** 2026

**Venue:** MDPI

**Dataset:** BCI IV 2a, 2b

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Lightweight regularization (MSFS) during target-subject fine-tuning that maximizes feature separability — useful when adapting pre-trained models to a target subject with limited data.

---

**Knowledge Extraction — What We Can Borrow:**

- Training: Silhouette-based separability maximization during fine-tuning — could be applied as a transductive objective during test-time adaptation on MODMA target subject

---

**Relevance Score:** 2/10

**Reason:** Minor regularization for within-dataset TL. Limited applicability to our strict LODO protocol.

---

## Paper 42 — HIVE-CODAs: Hierarchical Vote Collective of DA for Seizure Prediction (Relevance: 2/10)

**Title:** Seizure Prediction With HIVE-CODAs: The Hierarchical Vote Collective of Domain Adaptation Methods

**Year:** 2022

**Venue:** Frontiers in Physics

**Dataset:** Epileptic EEG; seizure prediction

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:** Ensemble of domain adaptation methods for cross-subject seizure prediction.

**Knowledge Extraction — What We Can Borrow:** Ensemble of DA methods idea — combine rebiasing, TSA, SPDDSMBN in a voted ensemble for more robust LODO predictions.

**Relevance Score:** 2/10 — Seizure task, limited transfer to depression.

---

## Paper 43 — TFRMDANet: Time-Frequency-Riemannian Domain Adaptation in Brain Source Space (Relevance: 3/10)

**Title:** A Time-Frequency Transform and Riemannian Manifold-Based Domain Adaptation Method for Motor Imagery in Brain Source Space

**Year:** 2026

**Venue:** Chinese biomedical engineering journal (BiJie)

**Dataset:** BCI IV 2a, High-Gamma; cross-subject MI

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Cross-subject MI DA using brain source imaging (cortical dipoles) + time-frequency features + Riemannian manifold embedding + adversarial DA.

**Methodology — Key Innovation:** Source-space (dipole) feature extraction before Riemannian manifold mapping — reduces inter-subject electrode placement variability by mapping to source space first.

**Knowledge Extraction — What We Can Borrow:**

- Technique: Source-space covariance matrices (rather than electrode-space) may reduce inter-subject spatial variability — relevant if MODMA provides sufficient spatial resolution for source imaging

**Relevance Score:** 3/10 — MI paradigm; source imaging from 128-ch MODMA is feasible but adds complexity.


---

## Paper 44 — WDANet: Wasserstein Adversarial DA for Cross-Domain EEG Depression Recognition (Relevance: 10/10)

**Title:** WDANet: Wasserstein Distribution Inspired Dynamic Adversarial Network for EEG-Based Cross-Domain Depression Recognition

**Year:** 2026

**Venue:** IEEE Transactions on Affective Computing

**Dataset:** MODMA (public, 53 subjects, 128-ch resting-state EEG); Dataset 1 (170 subjects, 81 MDD + 89 HC, self-collected); Dataset 2 (additional self-collected); cross-subject and cross-dataset evaluation

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Cross-domain EEG-based depression recognition — simultaneously addressing cross-subject and cross-dataset distribution variability using dynamic adversarial training with Wasserstein distance-based distribution alignment.

**Research Problem — Why It Matters:**

Depression EEG classifiers trained on one subject/dataset fail on others due to inter-subject distribution drift. Prior DA methods use static global or local discriminators and fail to capture dynamic distribution transformations. Standard adversarial DA ignores intermediate distribution states during domain transformation.

---

**Methodology — Core Idea:**

WDANet: three discriminators for comprehensive distribution alignment: (1) **Global discriminator**: aligns marginal distributions across source and target domains (cross-subject/dataset). (2) **Local discriminator**: aligns conditional distributions (class-specific patterns). (3) **Wasserstein distribution discriminator**: leverages Wasserstein distance to model intermediate states of distribution transformation — captures the geometry of distribution change, not just the end-state. A **dynamic adversarial factor** (derived from A-distance between discriminator losses) automatically balances global and local alignment losses during each training iteration. This ensures the model adapts to distribution shifts that evolve differently across batches.

**Methodology — Model Architecture:**

- Feature extraction: raw EEG → CNN-based spatiotemporal encoder
- Adaptation: three discriminators (global + local + Wasserstein) with dynamic adversarial factor
- Classifier: softmax on adapted features

**Methodology — Learning Strategy:** Multi-source, multi-target UDA; unsupervised domain adaptation; cross-subject and cross-dataset

**Methodology — Key Innovation:** Wasserstein distribution discriminator capturing distribution transformation intermediates; dynamic adversarial factor balancing global/local alignment; validated on depression-specific cross-domain EEG

---

**Evaluation Protocol:**

**Dataset Split:** Cross-subject (train on source subjects, test on unseen subjects) AND cross-dataset (train on one dataset, test on another); MODMA used as one benchmark

**Subject Independence:** Yes

**Cross-Subject Evaluation:** Yes

**Leave-One-Subject-Out (LODO):** Partial (cross-subject experiments)

**Cross-Dataset Evaluation:** Yes (3-dataset cross-domain evaluation)

**Metrics:** Classification accuracy (binary MDD vs. HC)

---

**Findings — Main Results:**

- Cross-subject accuracy across 3 datasets: 83.33%, 75.52%, 73.93%, 76.04%, 70.94%
- Outperforms state-of-the-art cross-domain depression methods including DANN, CORAL, DDA, R2G-STNN
- Cross-dataset generalization (train on one dataset, test on another) also demonstrated
- MODMA explicitly used as one evaluation dataset

**Findings — Important Observations:**

- Wasserstein discriminator captures distribution transition dynamics that static MMD/CORAL miss — key for depression EEG where distribution shift is gradual across subjects
- Dynamic adversarial factor prevents dominance of one discriminator over others — critical for stable training with heterogeneous subject distributions
- Cross-dataset transfer is more challenging than cross-subject but still feasible with dynamic adversarial DA

**Findings — Failure Cases:**

- No Riemannian geometry — purely Euclidean CNN features; may miss the geometric structure of SPD manifold that Papers 3, 4, 8, 22 show is critical
- No LODO evaluation with strict leave-one-subject-out protocol — cross-subject evaluation may not be strictly LODO
- No explicit AUC reporting — accuracy-only evaluation is insufficient for imbalanced depression data

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: Three-discriminator design (global + local + Wasserstein) for cross-subject depression DA — the local (class-conditional) discriminator is particularly important for depression where MDD subtype differences add within-class variance
- Loss: Dynamic adversarial factor to balance global-local alignment — prevents one discriminator from overwhelming the other; applicable in our LODO framework
- Loss: Wasserstein distance for distribution alignment (instead of standard GAN loss or MMD) — geometrically more faithful for high-dimensional EEG covariance distributions
- Validation: MODMA cross-subject depression results as a direct comparison point; WDANet's best cross-subject accuracy (~83% on one dataset) is our secondary target baseline
- Architecture idea: Combine WDANet's adversarial alignment with Riemannian geometry (SPD covariances) for a hybrid approach stronger than either alone

**Knowledge Extraction — What We Should NOT Borrow:**

- Euclidean CNN features — replace with Riemannian covariance features (SPD manifold) + SPDDSMBN alignment (Paper 22)
- No LODO evaluation — use strict LOSO for MODMA
- Accuracy-only metric — report AUC for imbalanced depression data

---

**Relevance Score:** 10/10

**Reason:** This is the most directly relevant paper found — it addresses EEG-based cross-domain depression recognition on MODMA specifically, proposes dynamic adversarial domain adaptation, and provides cross-subject accuracy baselines we need to beat. The combination of Wasserstein adversarial alignment with our Riemannian geometry pipeline would be a key design integration. Our novel contribution is: WDANet's adversarial alignment + Riemannian SPD features + transductive SPDDSMBN adaptation = expected improvement over WDANet's Euclidean baseline.

---

**Possible Integration:**

- Directly usable: WDANet cross-subject depression accuracy on MODMA as the strong non-Riemannian baseline to beat
- Requires modification: Replace WDANet's Euclidean CNN feature extractor with Riemannian SPD features + SPDDSMBN (Paper 22) normalization; keep the three-discriminator adversarial structure; add transductive test-subject Fréchet mean update
- Future experiment: WDANet (Euclidean) vs. Riemannian-WDANet (our model) vs. Riemannian-WDANet + transductive adaptation in strict LODO on MODMA — measures the contribution of each geometric component

---

## Paper 45 — ReTa-Diffusion: Resting-to-Task EEG Diffusion for Subclinical Depression Detection (Relevance: 8/10)

**Title:** ReTa-Diffusion: Exploring Task-State from Resting-State in EEG Signals via Bidirectional Decoupling and Latent Guiding for Early Detection of Subclinical Depression

**Year:** 2026

**Venue:** ICLR 2026 (under review)

**Dataset:** Three resting-state EEG datasets for subclinical depression (SD) / subthreshold depression; cross-subject evaluation

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Early detection of subclinical depression from resting-state EEG by generating and exploiting task-state-informed features — bridging the gap between task-free rs-EEG and task-specific neural signatures for depression.

**Research Problem — Why It Matters:**

Task-state EEG contains richer depression-specific neural markers but is impractical for screening; resting-state EEG is easier to acquire but has weaker signals. The theoretical claim: rs-EEG contains the neural substrates of all potential task-states — a diffusion model can extract task-relevant features from rs-EEG.

---

**Methodology — Core Idea:**

Two modules: (1) **BDCD (Bidirectional Decoupled Conditional Diffusion)**: jointly models resting-state and task-state EEG via bidirectional diffusion — learns to generate task-state-informed EEG from resting-state input, and vice versa, using conditional diffusion. The bidirectional conditioning decouples task-specific from state-general features. (2) **WRFE (Wavelet-Riemannian Feature Extraction)**: from BDCD-generated signals, extract multi-scale wavelet features capturing temporal dynamics; compute Riemannian covariance matrices of generated signals; cross-attention fusion of wavelet temporal features and Riemannian spatial features → final depression classifier.

**Methodology — Model Architecture:**

- BDCD: conditional diffusion model for resting↔task EEG synthesis
- WRFE: wavelet decomposition → multi-scale temporal features + covariance → Riemannian manifold → cross-attention fusion
- Classifier: MLP on fused features

**Methodology — Learning Strategy:** Supervised with generative pretraining (BDCD); cross-subject evaluation via 5-fold CV and LOSO

**Methodology — Key Innovation:** Bidirectional resting-task diffusion for extracting hidden task-state neural markers from rs-EEG; Wavelet-Riemannian hybrid for multi-scale temporal + spatial functional connectivity features

---

**Evaluation Protocol:**

**Dataset Split:** 5-fold CV + leave-one-subject-out validation

**Subject Independence:** Yes (LOSO confirmed)

**Cross-Subject Evaluation:** Yes

**Leave-One-Subject-Out (LODO):** Yes

**Cross-Dataset Evaluation:** Yes (3 depression datasets)

**Metrics:** Classification accuracy (multiclass: 54.52%)

---

**Findings — Main Results:**

- Multiclass classification: 54.52% — outperforms SOTA by 18.95% on the same multiclass task
- LOSO validation confirms cross-subject generalization
- Visual analysis validates that generated task-state EEG corresponds to neurophysiologically valid brain regions for depression

**Findings — Important Observations:**

- Riemannian covariance features in WRFE significantly contribute — confirms our core design hypothesis
- Bidirectional diffusion extracts richer features than rs-EEG alone — but requires both resting and task EEG at training time
- Subclinical depression (subthreshold) is a harder task than clinical MDD vs. HC — multiclass accuracy is lower

**Findings — Failure Cases:**

- Requires both task-state and resting-state EEG for training — at deployment, only rs-EEG is used but training needs task-state
- MODMA is HC vs. MDD binary; this paper is multiclass (depression severity levels)
- 54.52% multiclass accuracy is modest — binary task would be significantly higher

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: WRFE module (Wavelet-Riemannian Feature Extraction) — wavelet decomposition for multi-scale temporal features + Riemannian covariance for spatial connectivity features + cross-attention fusion; directly applicable to MODMA resting-state depression EEG
- Technique: Diffusion-based data augmentation for depression EEG — generate additional resting-state samples conditioned on class labels; complementary to RGP-VAE (Paper 14)
- Evaluation: LOSO is validated here for depression detection — confirms our evaluation protocol is appropriate

**Knowledge Extraction — What We Should NOT Borrow:**

- Requires task-state EEG at training time — MODMA is resting-state only
- Multiclass depression severity — our target is binary MDD vs. HC
- Diffusion model training complexity — expensive and data-hungry

---

**Relevance Score:** 8/10

**Reason:** Directly addresses depression detection from resting-state EEG with LOSO cross-subject evaluation. The WRFE module (Wavelet-Riemannian fusion) is a novel, directly applicable architecture component. The paper confirms Riemannian features are critical for cross-subject depression generalization.

---

**Possible Integration:**

- Directly usable: WRFE module — wavelet multi-scale temporal features + Riemannian covariance + cross-attention as the feature extraction stage of our LODO depression model; replace PSD+DE (Paper 8) with wavelet+Riemannian hybrid
- Future experiment: WRFE features vs. PSD+DE features (Paper 8) in LODO on MODMA — which multi-scale representation gives better cross-subject generalization

---

## Paper 46 — HybridRDG: Zero-Shot ASD Biomarker Detection via Hybrid Riemannian DG (Relevance: 7/10)

**Title:** HybridRDG: Zero-Shot ASD Biomarker Detection from Multi-Paradigm EEG via Hybrid Deep-Riemannian Domain Generalization

**Year:** 2026

**Venue:** CVPR Workshop 2026

**Dataset:** Multi-paradigm EEG (5 paradigms) for ASD (Autism Spectrum Disorder) genetic biomarker detection; strict subject-held-out evaluation

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Zero-shot cross-subject domain generalization for clinical biomarker detection from EEG — detect an autism-linked genetic variant without any target-subject data during training.

**Research Problem — Why It Matters:**

Clinical EEG datasets have <50 labeled subjects; domain shift is geometric (covariance structure), not stylistic; existing image-domain DG methods fail on EEG.

---

**Methodology — Core Idea:**

Two-branch hybrid: (1) EEGNet branch with adversarial regularization — captures temporal waveform morphology. (2) Riemannian geometry branch — covariance matrices from multiple time windows per paradigm, Ledoit-Wolf shrinkage for robustness, Log-Euclidean alignment on source subjects only. The branches are fused and trained with domain adversarial regularization. Everything fitted exclusively on source subjects (zero-shot).

**Methodology — Model Architecture:**

- Branch 1: EEGNet (adversarially regularized) → temporal features
- Branch 2: multi-window covariance → Ledoit-Wolf shrinkage → LogE alignment → SPD features
- Fusion: concatenation → linear classifier

**Methodology — Learning Strategy:** Zero-shot domain generalization (no target data; strict held-out subject evaluation)

**Methodology — Key Innovation:** Hybrid deep-Riemannian DG for clinical EEG with zero target data; Ledoit-Wolf shrinkage for small-sample robustness; multi-window SPD decomposition per paradigm

---

**Evaluation Protocol:**

**Dataset Split:** Strict subject-held-out (zero-shot DG)

**Subject Independence:** Yes

**Cross-Subject Evaluation:** Yes

**Leave-One-Subject-Out (LODO):** Yes

**Cross-Dataset Evaluation:** No (multi-paradigm within one clinical EEG study)

**Metrics:** Balanced accuracy (subject-level classification)

---

**Findings — Main Results:**

- HybridRDG achieves highest balanced accuracy in 4/5 paradigm evaluations vs. 10 baselines (Riemannian, DL, DG)
- Key finding: "geometric covariance structure, not merely Euclidean feature alignment, is central to zero-shot EEG generalization"
- Riemannian branch alone outperforms EEGNet branch on most paradigms — geometry dominates

**Findings — Important Observations:**

- Ledoit-Wolf shrinkage is essential for robustness with few subjects (<50) — critical for MODMA (53 subjects)
- Multi-window SPD decomposition captures temporal non-stationarity within each paradigm
- Zero-shot (no target data) Riemannian DG outperforms target-adapted Euclidean methods

**Findings — Failure Cases:**

- ASD genetic variant, not depression — different clinical task
- Multi-paradigm EEG (different than pure resting-state MODMA)
- Balanced accuracy metric — no AUC reported

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: Hybrid EEGNet + Riemannian SPD branch with adversarial regularization — dual-branch architecture covering both temporal waveform and covariance structure; directly applicable to MODMA depression
- Technique: Ledoit-Wolf shrinkage covariance estimator — essential for robust SPD computation with few subjects/trials (MODMA has 53 subjects, each with limited epochs)
- Design principle: Geometric covariance structure is central to zero-shot EEG generalization — strongest endorsement found for prioritizing Riemannian geometry over Euclidean deep learning in clinical LODO
- Training: Multi-window SPD decomposition (multiple overlapping windows) for temporal non-stationarity handling in resting-state EEG

**Knowledge Extraction — What We Should NOT Borrow:**

- ASD genetic variant task
- Log-Euclidean only (not AIRM) for alignment — AIRM is geometrically superior
- Zero-shot only — our transductive setting is better when unlabeled target data is available

---

**Relevance Score:** 7/10

**Reason:** Provides the strongest direct empirical evidence that "geometric covariance structure is central to zero-shot EEG generalization" across clinical populations (<50 subjects). Ledoit-Wolf shrinkage and multi-window SPD are directly applicable to MODMA. The hybrid architecture is immediately borrowable.

---

**Possible Integration:**

- Directly usable: Ledoit-Wolf shrinkage in covariance computation for all MODMA subjects — replace naive sample covariance with shrinkage estimator
- Directly usable: Hybrid EEGNet + Riemannian branch as the base architecture for our LODO depression model; add SPDDSMBN (Paper 22) and transductive alignment on top
- Future experiment: Zero-shot DG (HybridRDG baseline) vs. Transductive (our model) in LODO on MODMA — measures the value of unlabeled target data

---

## Paper 47 — REDDI: Riemannian Ensemble for Differential Diagnosis of Neurodegenerative Diseases (Relevance: 6/10)

**Title:** REDDI: A Riemannian Ensemble Learning Framework for Interpretable Differential Diagnosis of Neurodegenerative Diseases

**Year:** 2026

**Venue:** medRxiv preprint (Sorbonne / Paris Brain Institute)

**Dataset:** Multi-site EEG/MEG from neurodegenerative disease patients (ALS, Parkinson's, related conditions); multi-institutional clinical cohort

**Modality:** EEG/MEG (physiological signals)

---

**Research Problem — Main Objective:**

Interpretable differential diagnosis of neurodegenerative diseases from EEG/MEG using an ensemble of Riemannian classifiers — combining MDM, TSLDA, and other Riemannian methods for robust cross-site clinical classification.

---

**Methodology — Core Idea:**

REDDI: ensemble of Riemannian classifiers (MDM, TSLDA, Riemannian SVM with different kernels) trained on multi-site data. Site-specific Riemannian recentering (rebiasing) applied before pooling. Ensemble voting combines diverse Riemannian perspectives. Interpretability via tangent-space variable importance (ANOVA/FDR on tangent-space features per diagnostic class).

**Methodology — Key Innovation:** Ensemble Riemannian learning for clinical multi-site data; interpretable differential diagnosis via tangent-space variable importance

---

**Evaluation Protocol:**

**Subject Independence:** Yes (multi-site clinical cohort)

**Cross-Subject Evaluation:** Yes

**Leave-One-Subject-Out (LODO):** Cross-site evaluation

**Metrics:** Accuracy, balanced accuracy, interpretability analysis

---

**Findings — Main Results:**

- Ensemble of Riemannian classifiers outperforms single Riemannian classifier and Euclidean baselines
- Rebiasing per site improves cross-site generalization
- Interpretable tangent-space features identify disease-relevant brain connectivity patterns

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: Ensemble of Riemannian classifiers (MDM + TSLDA + RK-SVM) as a robust baseline — more stable than single classifier across MODMA subjects
- Technique: Site-specific rebiasing for multi-site data harmonization — directly applicable if MODMA or future depression datasets come from multiple acquisition sites

**Knowledge Extraction — What We Should NOT Borrow:**

- Neurodegenerative diseases — different brain dynamics from MDD
- Ensemble alone is not novel enough for our contribution

---

**Relevance Score:** 6/10

**Reason:** Clinical multi-site Riemannian ensemble with rebiasing harmonization — the multi-site extension of our single-dataset LODO framework.

---

## Paper 48 — Riemannian Geometry for Multi-Site ADHD Data Harmonization (Relevance: 6/10)

**Title:** Riemannian Geometry of Functional Connectivity Matrices for Multi-Site ADHD Data Harmonization

**Year:** 2022

**Venue:** Frontiers in Neuroinformatics

**Dataset:** Multi-site ADHD-200 dataset (resting-state fMRI functional connectivity); multi-site harmonization

**Modality:** fMRI (functional MRI — methodology transferable to EEG)

---

**Research Problem — Main Objective:**

Harmonize multi-site resting-state fMRI functional connectivity matrices using Riemannian geometry to enable pooled cross-site ADHD classification.

**Research Problem — Why It Matters:**

Multi-site neuroimaging suffers from site-specific systematic biases in functional connectivity matrices; standard Euclidean normalization fails to preserve matrix structure; Riemannian geometry provides site-invariant harmonization.

---

**Methodology — Core Idea:**

Per-site Riemannian recentering (parallel transport to common reference) on functional connectivity matrices, followed by pooling of all harmonized matrices for ADHD classification. Riemannian geometry identifies and removes site-specific distributional biases while preserving within-site class-discriminative structure.

**Methodology — Key Innovation:** Multi-site Riemannian harmonization for psychiatric resting-state functional connectivity

---

**Evaluation Protocol:**

**Subject Independence:** Yes (cross-site)

**Cross-Subject Evaluation:** Yes

**Leave-One-Subject-Out (LODO):** Cross-site evaluation (multi-site leave-one-site-out)

**Metrics:** Classification accuracy (ADHD vs. HC)

---

**Findings — Main Results:**

- Riemannian harmonization significantly improves pooled cross-site ADHD classification vs. no harmonization and Euclidean normalization
- Riemannian recentering preserves within-site discriminative structure better than linear methods

---

**Knowledge Extraction — What We Can Borrow:**

- Technique: Riemannian harmonization for multi-site EEG depression data — directly applicable if future work involves pooling MODMA with other depression EEG datasets
- Insight: Per-site Riemannian mean → transport to identity is the correct harmonization procedure for psychiatric resting-state neuroimaging

**Knowledge Extraction — What We Should NOT Borrow:**

- fMRI modality (not EEG); ADHD task (different neural dynamics from MDD)
- Multi-site harmonization is secondary to single-site LODO for our primary goal

---

**Relevance Score:** 6/10

**Reason:** Validates Riemannian geometry as the correct multi-site harmonization approach for psychiatric resting-state neuroimaging — directly applicable to future depression dataset pooling.

---

## Paper 49 — DIFFEOCFM: Riemannian Flow Matching for Brain Connectivity Matrices (Relevance: 6/10)

**Title:** Riemannian Flow Matching for Brain Connectivity Matrices via Pullback Geometry

**Year:** 2025

**Venue:** NeurIPS 2025

**Dataset:** ADNI (Alzheimer's), ABIDE (autism), OASIS-3 (ageing) fMRI — 4600 scans, 2800 subjects; BNCI2014-002 and BNCI2015-001 EEG (30000 trials, 26 subjects)

**Modality:** fMRI and EEG (physiological signals)

---

**Research Problem — Main Objective:**

Efficient generative modeling of brain connectivity matrices (SPD/correlation) using conditional flow matching on Riemannian manifolds via pullback geometry — enabling data augmentation and heterogeneity analysis in clinical neuroimaging.

---

**Methodology — Core Idea:**

DiFFEoCFM: Conditional Flow Matching (CFM) on SPD/correlation manifolds by expressing the Riemannian metric as a pullback from a simpler Euclidean space. The key insight: Riemannian CFM with pullback metrics is equivalent to standard CFM after a data transformation (log for covariance, normalized Cholesky for correlation) — enabling fast ODE-solver sampling without explicit manifold operations. Evaluated on both fMRI connectivity and EEG covariance matrices.

**Methodology — Key Innovation:** Pullback equivalence reduces Riemannian CFM to standard Euclidean CFM — computationally efficient while preserving manifold structure; generates valid SPD matrices without rejection sampling

---

**Evaluation Protocol:**

**Subject Independence:** Yes (disease vs. HC classification from generated + real data)

**Cross-Subject Evaluation:** Yes

**Metrics:** FID-like score for generated matrices; classification performance with augmented data

---

**Findings — Main Results:**

- State-of-the-art on 3 large-scale fMRI datasets and 2 EEG datasets for generating valid connectivity matrices
- Augmentation with DiFFEoCFM improves classification on small clinical cohorts
- Normalized Cholesky transformation for correlation matrices is better than matrix log for covariance

---

**Knowledge Extraction — What We Can Borrow:**

- Technique: DiFFEoCFM for generating synthetic MODMA depression EEG covariance matrices — data augmentation for the small depression dataset (53 subjects)
- Architecture: Normalized Cholesky as a simple, fast alternative to log for correlation matrix transformation — applicable in our pipeline
- Application: Disease-class-conditional generation of connectivity matrices — generate additional MDD covariance matrices for augmenting minority class in MODMA

**Knowledge Extraction — What We Should NOT Borrow:**

- fMRI as primary modality; large-scale dataset assumption
- Full flow matching training overhead may not be justified for 53-subject MODMA

---

**Relevance Score:** 6/10

**Reason:** Provides efficient Riemannian data augmentation for clinical connectivity matrices. The EEG evaluation and disease-classification augmentation make this directly applicable to MODMA data scarcity problem.

---

## Paper 50 — PSDNorm: Test-Time Temporal Normalization for Deep Learning in Sleep Staging (Relevance: 6/10)

**Title:** PSDNorm: Temporal Normalization for Deep Learning in Sleep Staging

**Year:** 2026

**Venue:** ICLR 2026

**Dataset:** 10 sleep staging datasets; 10K subjects; cross-site and cross-subject evaluation

**Modality:** EEG/polysomnography (physiological signals)

---

**Research Problem — Main Objective:**

Mitigate distribution shift across subjects, recording sites, and devices in deep learning sleep staging — using Monge mapping-based temporal normalization that leverages temporal context of feature maps.

**Research Problem — Why It Matters:**

BatchNorm, LayerNorm, and InstanceNorm ignore temporal autocorrelation when normalizing over time; for physiological signals with complex temporal dependencies, this causes distribution shift to persist.

---

**Methodology — Core Idea:**

PSDNorm: normalize feature maps using the Monge optimal transport map (OT-based equalization) computed over temporal context windows, rather than batch statistics. The Monge map aligns each feature's distribution to a reference distribution while preserving temporal autocorrelation. Applied as a drop-in layer in U-Net and Transformer architectures for sleep staging.

**Methodology — Key Innovation:** Monge mapping for temporal normalization — optimal transport-based feature map equalization that preserves temporal structure; tested at test time (unseen subjects/sites)

---

**Evaluation Protocol:**

**Subject Independence:** Yes

**Cross-Subject Evaluation:** Yes

**Leave-One-Subject-Out (LODO):** Yes (leave-out-dataset evaluation on 10 datasets)

**Cross-Dataset Evaluation:** Yes

**Metrics:** Balanced accuracy, F1

---

**Findings — Main Results:**

- PSDNorm achieves SoA on unseen datasets at test time
- More robust to data scarcity than standard normalization layers
- Works with U-Net and Transformer backbones

**Findings — Important Observations:**

- Test-time normalization is critical for EEG cross-subject/site generalization — aligns individual subject's feature distribution without training
- Monge mapping preserves temporal autocorrelation — important for EEG where temporal structure is informative

---

**Knowledge Extraction — What We Can Borrow:**

- Technique: PSDNorm as a test-time normalization layer in our LODO depression model — apply to target subject features at inference time without any target labels; complementary to SPDDSMBN (which normalizes in SPD space, not feature space)
- Technique: Monge optimal transport for feature map alignment — could be applied to align source and target EEG features at the representation level (above SPD manifold)
- Architecture: PSDNorm as a drop-in layer in any backbone — simple to integrate

**Knowledge Extraction — What We Should NOT Borrow:**

- Sleep staging task (not depression)
- Full OT computation at every test batch may be expensive with 128-channel EEG

---

**Relevance Score:** 6/10

**Reason:** PSDNorm addresses test-time cross-subject normalization via optimal transport — a clean, computationally efficient transductive technique applicable to our LODO framework. Validated on a large cross-site physiological signal dataset.

---

**Possible Integration:**

- Directly usable: PSDNorm as a test-time feature normalization layer in our LODO depression model — align test subject features to source distribution without any labels; stack with SPDDSMBN

---

## Paper 51 — NeuroBRIDGE: Koopman Dynamics + Riemannian Alignment for Substance Use Prediction (Relevance: 5/10)

**Title:** NeuroBRIDGE: Behavior-Conditioned Koopman Dynamics with Riemannian Alignment for Early Substance Use Initiation Prediction

**Year:** 2026

**Venue:** arXiv (ICASSP/ICIP style)

**Dataset:** ABCD (Adolescent Brain Cognitive Development) longitudinal fMRI; 10K+ subjects

**Modality:** fMRI (functional MRI — longitudinal)

---

**Research Problem — Main Objective:**

Predict future substance use initiation from longitudinal functional connectome data by aligning trajectories in Riemannian tangent space and modeling temporal dynamics via Koopman operators.

---

**Methodology — Core Idea:**

NeuroBRIDGE: (1) Riemannian tangent space alignment of longitudinal connectivity matrices — aligns each subject's time-series of functional connectivity matrices to a common reference. (2) GNN with dual-time attention over aligned matrices. (3) Behavioral-conditioned Koopman dynamics — Koopman operator models the linear evolution of nonlinear brain dynamics in a lifted space; behavior (clinical assessment) conditions the Koopman evolution.

**Methodology — Key Innovation:** Koopman operator + Riemannian alignment for longitudinal functional connectome; behavior-conditioned temporal prediction

---

**Knowledge Extraction — What We Can Borrow:**

- Technique: Koopman operator for modeling longitudinal EEG dynamics — applicable if depression detection uses longitudinal EEG across multiple sessions
- Technique: Behavior-conditioned feature alignment — conditioning on clinical assessments (PHQ-9, HDRS) to guide Riemannian feature learning for depression

**Knowledge Extraction — What We Should NOT Borrow:**

- fMRI + longitudinal paradigm; MODMA is cross-sectional (single session)
- Koopman dynamics complexity for a single-session LODO task

---

**Relevance Score:** 5/10

**Reason:** Behavior-conditioned Riemannian alignment is interesting for depression (clinical score as conditioning signal) but requires longitudinal data which MODMA lacks.

---

## Paper 52 — Real-Time EEG Recentering for Motor-Task BCI: Class-Agnostic Fixation-Based Method (Relevance: 6/10)

**Title:** Real-Time Decoding of Movement Onset and Offset for Brain-Controlled Rehabilitation Exoskeleton

**Year:** 2026

**Venue:** arXiv (robotics/BCI)

**Dataset:** 8 participants, 2 online sessions, upper-limb MI BCI for exoskeleton control

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Real-time online EEG BCI control with a class-agnostic fixation-based recentering method that prevents drift without requiring knowledge of trial labels.

**Research Problem — Why It Matters:**

Standard task-based recentering (using movement trial covariances) introduces systematic class-driven bias that worsens over sessions. A class-agnostic method uses fixation periods (no motor activity) to track drift without contaminating class geometry.

---

**Methodology — Core Idea:**

Fixation-based Riemannian recentering: instead of re-estimating the Riemannian mean from task trials (which biases toward the dominant class), compute a running Riemannian mean from fixation-period (baseline, rest) trials only — class-agnostic drift tracking. Remap all trial covariances using this fixation-based reference. Eliminates the asymmetric margin artifact introduced by class-imbalanced recentering.

**Methodology — Key Innovation:** Class-agnostic fixation-period recentering — theoretically unbiased drift correction for online BCI; AUC gains: onset +56%, offset +34%

---

**Evaluation Protocol:**

**Subject Independence:** No (within-subject online sessions)

**Cross-Subject Evaluation:** No (single subject per session)

**Metrics:** AUC, hit rates; statistical significance

---

**Findings — Main Results:**

- Class-agnostic recentering: AUC +56% (onset), +34% (offset) vs. standard task-based recentering
- Eliminates asymmetric margin bias confirmed statistically (p < 0.05)
- Generalizes across days within the same subject

**Findings — Important Observations:**

- Task-based recentering introduces class bias when trial classes are imbalanced — critical warning for depression LODO where MDD:HC ratios are often unbalanced
- Rest/baseline periods are sufficient for tracking drift — no need for labeled task trials

---

**Knowledge Extraction — What We Can Borrow:**

- Technique: Class-agnostic baseline-period Riemannian recentering — for depression LODO, use resting-state baseline segments (eyes-open or eyes-closed rest before EEG task) to compute a subject-specific drift reference, avoiding the class-imbalance bias
- Warning: Standard S-RCT (Paper 21) using all trials for mean estimation is biased if class proportions are unequal — use fixation/baseline-only mean estimation
- Practical: AUC as primary metric — confirms AUC, not accuracy, is the right metric for class-imbalanced settings

**Knowledge Extraction — What We Should NOT Borrow:**

- Within-subject online setting; motor imagery exoskeleton task

---

**Relevance Score:** 6/10

**Reason:** Critical practical insight: standard recentering using all trials is biased when classes are imbalanced. For MODMA depression (MDD vs. HC) with potentially unequal class sizes, class-agnostic baseline-period recentering is the correct approach for our transductive alignment step.

---

**Possible Integration:**

- Directly usable: Fixation-based (baseline-period) Riemannian mean estimation for S-RCT alignment of test subjects — use the initial eyes-closed rest segment of each test subject's MODMA recording to estimate their Riemannian mean for alignment

---

## Paper 53 — EEG-Based Alzheimer's/FTD Classification Using Functional Connectivity (Relevance: 5/10)

**Title:** EEG-Based Classification of Alzheimer's Disease and Frontotemporal Dementia Using Functional Connectivity

**Year:** 2026

**Venue:** Scientific Reports

**Dataset:** EEG from Alzheimer's disease and frontotemporal dementia patients; multi-class clinical classification

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Classify Alzheimer's disease vs. FTD vs. HC from EEG using functional connectivity features — cross-subject clinical classification.

---

**Methodology — Core Idea:**

Functional connectivity (PLV, coherence, etc.) computed from EEG → feature matrix classification. Comparison of different connectivity measures and classifiers for neurodegenerative disease EEG.

---

**Knowledge Extraction — What We Can Borrow:**

- Insight: Functional connectivity features (PLV, coherence) are effective for neurological/psychiatric clinical EEG classification — validates FC + Riemannian approach (Paper 5) for depression
- Practical: Cross-subject clinical EEG with small patient cohorts — similar sample size challenges to MODMA (53 subjects)

---

**Relevance Score:** 5/10

**Reason:** Clinical EEG classification for neurodegenerative diseases; functional connectivity approach validated. Limited direct applicability to depression-specific methods.


---

## Paper 54 — Cross-Subject Generalization for EEG Decoding: A Survey of Deep Learning Methods (Relevance: 7/10)

**Title:** Cross-Subject Generalization for EEG Decoding: A Survey of Deep Learning Methods

**Year:** 2026

**Venue:** arXiv preprint / IOP Journal

**Dataset:** Survey across multiple EEG BCI datasets

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Comprehensive survey of deep learning approaches for cross-subject generalization in EEG decoding — categorizing methods, identifying trends, and highlighting future directions.

---

**Methodology — Core Idea:**

Systematic review of cross-subject deep learning EEG methods across categories: data augmentation, domain adaptation (adversarial, statistical), domain generalization, self-supervised learning, meta-learning, and Riemannian geometry-based approaches. Includes taxonomy, comparison tables, and analysis of evaluation protocols.

---

**Key Findings (Survey):**

- Riemannian geometry-based methods consistently competitive with or superior to deep learning for cross-subject generalization
- Transductive methods (using unlabeled target) outperform domain generalization across all categories
- LOSO evaluation is the gold standard — other protocols often inflate performance
- Data augmentation on SPD manifold is emerging as a key technique

---

**Knowledge Extraction — What We Can Borrow:**

- Taxonomy: Complete classification of cross-subject EEG methods — use to position our contribution relative to the literature
- Trend: Hybrid Riemannian + deep learning consistently outperforms either alone
- Benchmark: Standard baselines to compare against in our LODO evaluation

**Relevance Score:** 7/10

**Reason:** The most comprehensive recent survey on our exact problem domain. Must-read for situating our contribution and identifying research gaps.

---

## Paper 55 — Survey: Deep Learning for Short/Zero-Calibration EEG BCI (Relevance: 5/10)

**Title:** A Survey on Deep Learning-Based Short/Zero-Calibration Approaches for EEG-Based Brain-Computer Interfaces

**Year:** 2021

**Venue:** Frontiers in Human Neuroscience

**Relevance Score:** 5/10

**Summary:** Surveys deep learning methods for zero-calibration BCI. Covers transfer learning, data augmentation, and domain adaptation. Foundational background for our cross-subject framework design.

**Knowledge Extraction — What We Can Borrow:** Calibration-time reduction framework; zero-calibration vs. short-calibration vs. full-calibration performance tradeoffs.

---

## Paper 56 — Review: Signal Processing for BCI Calibration Time Reduction (Relevance: 4/10)

**Title:** A Review on Signal Processing Approaches to Reduce Calibration Time in EEG-Based Brain-Computer Interface

**Year:** 2021

**Venue:** Frontiers in Neuroscience

**Relevance Score:** 4/10

**Summary:** Reviews signal processing methods for reducing calibration time, including Riemannian approaches. Lower-level than our focus but provides background.

---

## Paper 57 — Survey: Few-Shot Learning for EEG Classification (Relevance: 5/10)

**Title:** Harnessing Few-Shot Learning for EEG Signal Classification: A Survey of State-of-the-Art Techniques

**Year:** 2024

**Venue:** Frontiers in Human Neuroscience

**Relevance Score:** 5/10

**Summary:** Surveys few-shot EEG classification methods — directly relevant to our LODO setting where the target subject has zero labeled samples.

**Knowledge Extraction — What We Can Borrow:** Prototypical networks, meta-learning (MAML, ProtoNet) for zero-shot cross-subject EEG — could complement our Riemannian approach; meta-learning on source subjects to adapt to target with zero labels.

---

## Paper 58 — Generalizability of ML Models for EEG (MSc Thesis) (Relevance: 4/10)

**Title:** Generalizability of Machine Learning Models Applied to Electroencephalographic Recordings

**Year:** 2026

**Venue:** MSc Thesis, Université Laval (Quebec)

**Relevance Score:** 4/10

**Summary:** Systematic evaluation of generalizability across subjects and datasets for EEG ML models. Confirms the challenge we address.

---

## Paper 59 — PhD Thesis: Geometry-Aware Learning for EEG Spectral Dynamics (Relevance: 6/10)

**Title:** [Geometric-aware Interpretability and Understanding of EEG Spectral Dynamics] (Obfuscated title — University of Verona, PhD thesis)

**Year:** 2026

**Venue:** PhD Thesis (Università degli Studi di Verona, Computer Science)

**Relevance Score:** 6/10

**Summary:** PhD thesis on geometry-aware approaches to EEG spectral dynamics — directly relevant background. Covers Riemannian geometry on SPD manifolds, spectral EEG features, and interpretability.

**Knowledge Extraction — What We Can Borrow:** Survey of geometric EEG methods from a thesis perspective — comprehensive treatment of the mathematical foundations we rely on.

---

## Paper 60 — PhD Thesis: Geometric Representation Learning on Riemannian Manifolds (Tibermacine) (Relevance: 7/10)

**Title:** Geometric Representation Learning on Riemannian Manifolds for Multidimensional Signal Decoding and Modeling

**Year:** 2026

**Venue:** PhD Thesis (Sapienza University of Rome — same author as Paper 17 Stiefel-SPD)

**Relevance Score:** 7/10

**Summary:** Comprehensive PhD thesis covering Riemannian manifold learning for EEG — covers SPDNet, Stiefel projections, graph-SPD fusion, and cross-subject evaluation (the work culminating in Paper 17). Provides full mathematical derivations and ablation studies not included in the conference paper.

**Knowledge Extraction — What We Can Borrow:** Complete derivations of Stiefel projection, parallel-transport graph aggregation, and all geometric operations used in Paper 17 — reference for implementing these in our model.

---

## Paper 61 — Non-Invasive BCI: Neural Signal Decoding and Flexible Bioelectronics Review (Relevance: 2/10)

**Title:** Non-Invasive Brain-Computer Interfaces: Converging Frontiers in Neural Signal Decoding and Flexible Bioelectronics Integration

**Year:** 2026

**Venue:** Nano-Micro Letters

**Relevance Score:** 2/10

**Summary:** Hardware-focused review of non-invasive BCI technologies. Not relevant to our signal processing / learning approach.

---

## Paper 62 — PhD Thesis: EEG for Emotional Vocalisations (Tang, Auckland) (Relevance: 2/10)

**Title:** Feature and Classification Analyses of the EEG Recorded During the Perception and Production of Emotional Vocalisations

**Year:** 2026

**Venue:** PhD Thesis (University of Auckland)

**Relevance Score:** 2/10

**Summary:** Emotion vocalisation EEG analysis — not relevant to depression resting-state or cross-subject generalization.

---

## Paper 63 — Technical Report: MI EEG for Rehabilitation Exoskeleton (Titkanlou) (Relevance: 2/10)

**Title:** Classification of MI EEG Signal Using an Advanced Deep Learning Architecture for a Lower-Limb Rehabilitation Exoskeleton

**Year:** 2025

**Venue:** Technical Report, University of West Bohemia

**Relevance Score:** 2/10 — MI rehabilitation BCI, not relevant.

---

## Paper 64 — PhD Thesis: Accessible and Explainable AI for EEG BCI (Colafiglio) (Relevance: 3/10)

**Title:** Accessible and Explainable AI for EEG Decoding in Brain-Computer Interfaces

**Year:** 2025

**Venue:** PhD Thesis (Politecnico di Bari)

**Relevance Score:** 3/10

**Summary:** Explainable AI for EEG BCI. Minor relevance for interpretability components in our final model.

**Knowledge Extraction — What We Can Borrow:** XAI techniques for explaining Riemannian EEG classification decisions — applicable to our clinical depression model where interpretability matters.

---

## Paper 65 — SPDNet: Riemannian Network for SPD Matrix Learning (AAAI 2017) (Relevance: 7/10)

**Title:** A Riemannian Network for SPD Matrix Learning

**Year:** 2017

**Venue:** AAAI 2017

**Dataset:** Action recognition (Charades, HDM05); face recognition datasets

**Modality:** Video/image (methodology transferable to EEG covariance matrices)

---

**Research Problem — Main Objective:**

Propose SPDNet — the first deep neural network that operates entirely on the SPD manifold — replacing Euclidean layers with manifold-preserving layers for covariance matrix classification.

---

**Methodology — Core Idea:**

SPDNet has three types of layers: (1) **BiMap layers**: P → B^T P B (bilinear mapping, reduces dimension while preserving SPD structure; B is trainable full-rank matrix). (2) **ReEig layers**: rectify eigenvalues to prevent near-zero/negative values (P → U diag(max(ε, λ_i)) U^T). (3) **LogEig layers**: P → U diag(log λ_i) U^T — maps SPD to tangent space for final vectorization. All layers are differentiable and respect the SPD manifold structure via Riemannian gradient descent.

**Methodology — Key Innovation:** First deep network on SPD manifold: BiMap (geometry-preserving dimensionality reduction) + ReEig (SPD rectification) + LogEig (tangent-space mapping) — all differentiable, enabling end-to-end Riemannian backpropagation

---

**Evaluation Protocol:** Action recognition and face recognition; within-dataset

---

**Findings — Main Results:** SPDNet outperforms Euclidean deep networks and classical Riemannian methods (KPCA+SVM, TSLDA) on action and face recognition tasks.

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: SPDNet (BiMap + ReEig + LogEig) as the standard deep Riemannian backbone — use as backbone for our LODO depression model; all extensions (SPDDSMBN, RHOP, RepSPD) build on this
- Training: Riemannian gradient descent for BiMap layers — essential for end-to-end SPD training
- Layers: The three-layer decomposition (dimensionality reduction + eigenvalue rectification + tangent mapping) is the canonical deep Riemannian architecture

**Relevance Score:** 7/10

**Reason:** SPDNet is the foundational deep Riemannian architecture. All Papers 3, 13, 22, 33, 65 use or extend SPDNet. Our LODO model's Riemannian backbone should be SPDNet-based (or a modern variant).

---

## Paper 66 — ARMAGNAC: Parametric Batch Normalization for SPDNet (Relevance: 5/10)

**Title:** ARMAGNAC: A New Parametric Batch Normalization Layer for SPDNet Architecture

**Year:** 2026

**Venue:** HAL preprint

**Dataset:** EEG BCI datasets (MI)

**Modality:** EEG

---

**Research Problem — Main Objective:**

Improve SPDNet training stability and cross-domain generalization by introducing a learnable parametric batch normalization layer on the SPD manifold — extending SPDNet with normalization analogous to SPDDSMBN (Paper 22) but with additional learnable affine parameters.

---

**Methodology — Core Idea:**

ARMAGNAC: SPD batch normalization with learnable scale and shift parameters on the SPD manifold (analogous to BatchNorm's γ and β in Euclidean space, but on the Riemannian manifold). Computes per-batch Fréchet mean → normalizes → applies learned affine transform. Enables the network to learn optimal normalization for each SPD layer — not just transport to identity.

**Methodology — Key Innovation:** Parametric (learnable γ, β) SPD batch normalization — generalizes SPDMBN (Paper 22) with additional learnable parameters

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: ARMAGNAC as an upgraded SPD batch normalization with learnable affine parameters — more expressive than SPDDSMBN; integrate into our SPDNet backbone for domain-adaptive normalization

**Relevance Score:** 5/10

---

## Paper 67 — FedSPDnet: Federated Geometry-Aware Deep Learning (Relevance: 4/10)

**Title:** FedSPDnet: Geometry-Aware Federated Deep Learning with SPDnet

**Year:** 2026

**Venue:** arXiv

**Relevance Score:** 4/10

**Summary:** Federated learning with SPDNet — Riemannian mean aggregation of SPD-manifold-valued model parameters across clients. Relevant if MODMA data cannot be centralized (privacy-preserving LODO).

**Knowledge Extraction — What We Can Borrow:** Riemannian federated aggregation (Fréchet mean of model parameters) as a privacy-preserving alternative to centralized LODO training.

---

## Paper 68 — Riemannian Networks over Full-Rank Correlation Matrices (Relevance: 8/10)

**Title:** Riemannian Networks over Full-Rank Correlation Matrices

**Year:** 2026

**Venue:** arXiv (Chen et al. — Jiangnan University / Tübingen)

**Dataset:** Multiple EEG datasets (BCI, ERP); also action recognition

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Extend Euclidean deep learning to the correlation manifold (full-rank correlation matrices) using five recently developed Riemannian geometries for correlation matrices.

**Research Problem — Why It Matters:**

SPDNet operates on covariance matrices (scale-dependent); correlation matrices are scale-invariant and more appropriate for cross-subject EEG. But deep learning on the correlation manifold lacks a systematic framework.

---

**Methodology — Core Idea:**

Systematically extend MLR, FC, and convolutional layers to the correlation manifold using five geometries: ECM, LECM, PHCM, OLM, LSM (Off-log metric). Provides correlation manifold analogues of BiMap, ReEig, LogEig for SPDNet.

**Methodology — Key Innovation:** First systematic deep learning framework over the correlation manifold; comparative evaluation of 5 correlation geometries

---

**Findings — Main Results:** Correlation manifold networks outperform SPD-manifold networks for cross-subject EEG — scale invariance helps cross-subject generalization.

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: Correlation manifold deep network (OLM or LSM geometry) as the primary backbone — scale-invariant, better for cross-subject depression EEG; convergent with Paper 4 (CorSW) finding that correlation > covariance
- Geometry: OLM (Off-Log Metric, Paper 15) as the preferred geometry — closed-form, permutation-invariant
- Design principle: Correlation matrices are the right representation for cross-subject EEG — this paper provides the deep learning implementation of this principle

**Relevance Score:** 8/10

**Reason:** Provides deep learning framework on correlation manifold — directly implements the design principle established by Papers 3, 4, and 15 that correlation matrices are superior for cross-subject EEG. Must borrow.

---

**Possible Integration:**

- Directly usable: Correlation manifold network (OLM/LSM geometry) as an alternative to SPDNet backbone in our LODO framework — scale-invariant, more appropriate for cross-subject depression EEG

---

## Paper 69 — SPD Matrix Learning for Neuroimaging: Perspectives and Challenges (Relevance: 8/10)

**Title:** SPD Matrix Learning for Neuroimaging Analysis: Perspectives, Methods, and Challenges

**Year:** 2026

**Venue:** arXiv (survey — Inria/RIKEN/NTU, including Kobler, Collas, Thirion)

**Dataset:** fMRI, EEG, MEG neuroimaging (survey)

**Modality:** Multiple (survey)

---

**Research Problem — Main Objective:**

Comprehensive review of SPD matrix learning for neuroimaging — from Riemannian geometry foundations to deep learning applications, unifying EEG, fMRI, MEG methods under one framework.

---

**Key Findings (Survey):**

- SPD matrices are the universal representation for second-order neuroimaging statistics — valid across modalities
- Riemannian geometry consistently improves cross-site/subject generalization over Euclidean methods
- Correlation manifold (quotient SPD) is increasingly preferred for cross-subject tasks
- SPDDSMBN (Paper 22) / TSMNet is current state-of-the-art for cross-subject EEG UDA

---

**Knowledge Extraction — What We Can Borrow:**

- Comprehensive taxonomy: Full taxonomy of SPD learning methods for neuroimaging — use to position our contribution
- Identified gap: Transductive learning with test-subject Fréchet mean adaptation is underexplored — this is our key contribution

**Relevance Score:** 8/10

**Reason:** Authoritative review by the same group as SPDDSMBN/TSMNet (Paper 22), explicitly identifying transductive adaptation as an open research direction.

---

## Paper 70 — Res2SPDNet: Multi-Granularity SPD Residual Learning (Relevance: 5/10)

**Title:** Res2SPDNet: Multi-Granularity SPD Matrix Residual Learning for Signal Classification

**Year:** 2026

**Venue:** CVPR Findings 2026

**Dataset:** EEG BCI, video, action recognition

**Modality:** EEG + others

**Relevance Score:** 5/10

**Summary:** Residual learning on SPD manifold with multi-granularity covariance (different temporal window sizes). Residual connection preserves long-range manifold geometry while learning local refinements.

**Knowledge Extraction — What We Can Borrow:** Multi-granularity covariance (multiple window sizes) as residual SPD features — captures both slow (long-window) and fast (short-window) functional connectivity dynamics in resting-state depression EEG.

---

## Paper 71 — GeoMamba: Dual-Path Riemannian SSM for Functional Dynamics (Relevance: 6/10)

**Title:** Geo-Mamba: Dual-Path Riemannian State Space Models for Functional Dynamics

**Year:** 2026

**Venue:** ICLR 2026 (under review)

**Dataset:** fMRI functional connectivity; multi-subject

**Modality:** fMRI (methodology transferable to EEG)

---

**Research Problem — Main Objective:**

Model functional connectivity dynamics with a Riemannian State Space Model (SSM/Mamba) that operates on the SPD manifold — capturing temporal evolution of connectivity matrices while respecting geometry.

---

**Methodology — Core Idea:**

Dual-path Riemannian SSM: (1) Manifold path — Mamba SSM operating on SPD matrices via geodesic state updates; (2) Euclidean path — standard SSM on tangent-space vectors. Dual-path merges the two via attention fusion. Models temporal sequence of connectivity matrices without the quadratic cost of transformers.

**Methodology — Key Innovation:** Riemannian State Space Model (Mamba) for sequential connectivity matrices — linear complexity, geometry-preserving temporal modeling

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: Riemannian SSM (Mamba) for modeling temporal evolution of covariance matrices within a long resting-state EEG recording — linear complexity vs. O(N²) attention; applicable to MODMA long recordings
- Technique: Dual manifold/Euclidean path fusion — captures both geometric structure and temporal dynamics

**Relevance Score:** 6/10

---

## Paper 72 — Unified SPD Token Transformer for EEG Classification (Relevance: 5/10)

**Title:** A Unified SPD Token Transformer Framework for EEG Classification: Systematic Comparison of Geometric Embeddings

**Year:** 2026

**Venue:** arXiv

**Dataset:** BCI2a, BCIcha, MAMEM (SSVEP, ERP); within-subject focus

**Modality:** EEG

**Relevance Score:** 5/10

**Summary:** Systematic comparison of geometric embedding choices (BWSPD, Log-Euclidean) in a SPD token Transformer. Key finding: BWSPD provides better gradient conditioning for high-dimensional SPD matrices (d ≥ 22 channels) — directly relevant for MODMA (128 channels).

**Knowledge Extraction — What We Can Borrow:** BWSPD (Bures-Wasserstein on SPD) as the embedding choice for high-dimensional (128-channel) covariance matrices — better gradient conditioning than LogE for our 128×128 MODMA matrices.

---

## Paper 73 — GyroAtt: Gyrovector Attention on Matrix Manifolds (NeurIPS 2025) (Relevance: 5/10)

**Title:** Towards a General Attention Framework on Gyrovector Spaces for Matrix Manifolds

**Year:** 2025

**Venue:** NeurIPS 2025

**Dataset:** 4 EEG datasets

**Modality:** EEG

**Relevance Score:** 5/10

**Summary:** Proposes a general attention mechanism (Gyro Attention / GyroAtt) on SPD, SPSD, and Grassmannian manifolds using gyrovector algebra — enables geometry-preserving attention for any matrix manifold. State-of-the-art on EEG.

**Knowledge Extraction — What We Can Borrow:** GyroAtt as a drop-in attention module for our SPD token Transformer — preserves SPD manifold structure in the attention mechanism; applicable to the RHOP head (Paper 3) or any attention over covariance tokens.

---

## Paper 74 — Multi-Branch CNN-LSTM for Cross-Subject MI (Relevance: 3/10)

**Title:** A Robust Multi-Branch CNN-LSTM Architecture for Cross-Subject Motor Imagery Classification

**Year:** 2026

**Venue:** MDPI journal

**Relevance Score:** 3/10 — Multi-scale temporal CNN-LSTM, no Riemannian geometry, MI task. Limited transferability.

---

## Paper 75 — SPDLearn: Geometric Deep Learning Python Library for Neural Decoding (Relevance: 4/10)

**Title:** SPDLearn: A Geometric Deep Learning Python Library for Neural Decoding Through Trivialization

**Year:** 2026

**Venue:** Unknown (software paper)

**Dataset:** EEG BCI

**Relevance Score:** 4/10

**Summary:** Python library implementing Riemannian deep learning for EEG — BiMap, ReEig, LogEig, SPDDSMBN, and trivialization-based optimization. Directly usable for implementing our LODO depression model.

**Knowledge Extraction — What We Can Borrow:** Ready-to-use implementation of all SPDNet-based operations including SPDDSMBN, BiMap, correlation manifold layers — use as the coding foundation for our model.

---

## Paper 76 — Channel Adaptation for EEG Foundation Models (Relevance: 4/10)

**Title:** Channel Adaptation for EEG Foundation Models: A Systematic Benchmark

**Year:** 2026

**Venue:** Unknown (benchmark paper)

**Relevance Score:** 4/10

**Summary:** Benchmark of channel adaptation methods when EEG channel configurations differ across subjects/datasets. Relevant if we need to adapt from standard BCI channels to MODMA's 128-channel setup.

---

## Paper 77 — EEG-Deformer: Dense Convolutional Transformer for BCI (Relevance: 3/10)

**Title:** EEG-Deformer: A Dense Convolutional Transformer for Brain-Computer Interfaces

**Year:** 2026

**Venue:** Unknown

**Relevance Score:** 3/10 — Dense conv transformer for EEG, no Riemannian geometry, no cross-subject focus.

---

## Paper 78 — NeuroPhysNet: FitzHugh-Nagumo Physics-Informed EEG Neural Network (Relevance: 2/10)

**Title:** NeuroPhysNet: A FitzHugh-Nagumo-Based Physics-Informed Neural Network for EEG Analysis and Motor Imagery Classification

**Year:** 2026

**Relevance Score:** 2/10 — Physics-informed MI EEG; interesting approach but not relevant to cross-subject depression detection.

---

## Paper 79 — ReManNet: Riemannian Manifold Network for 3D Lane Detection (Relevance: 1/10)

**Title:** ReManNet: A Riemannian Manifold Network for Monocular 3D Lane Detection

**Year:** 2026

**Relevance Score:** 1/10 — Computer vision (not EEG); Riemannian geometry for 3D detection, not physiological signals.

---

## Paper 80 — LAtte: Hyperbolic Lorentz Attention for Cross-Subject EEG (Relevance: 6/10)

**Title:** Latte: Hyperbolic Lorentz Attention for Joint-Subject EEG Classification

**Year:** 2026

**Venue:** arXiv

**Dataset:** Multiple EEG BCI datasets; cross-subject evaluation

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Cross-subject EEG classification using hyperbolic geometry — modeling the hierarchical brain organization structure that Euclidean and SPD methods cannot capture efficiently.

---

**Methodology — Core Idea:**

Lorentz attention (cross-attention on Lorentz model of hyperbolic space) over EEG representations. Subjects are embedded in hyperbolic space where inter-subject hierarchical relationships are modeled more efficiently than Euclidean embedding. Hyperbolic geodesics between subjects reflect neurological proximity.

**Methodology — Key Innovation:** Lorentz model attention for cross-subject EEG — hyperbolic geometry efficiently represents the tree-like hierarchy of brain connectivity across subjects

---

**Evaluation Protocol:** Cross-subject, LOSO

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: Hyperbolic embedding for cross-subject brain space — captures hierarchical subject similarity structure that Euclidean and even SPD geometry misses
- Technique: Subject embeddings in hyperbolic space for subject selection / weighting in LODO — subjects closer in hyperbolic space to the test subject are given more weight

**Knowledge Extraction — What We Should NOT Borrow:**

- Lorentz model adds complexity; Riemannian SPD already models most relevant structure for covariance matrices

**Relevance Score:** 6/10

**Reason:** Cross-subject LOSO with hyperbolic geometry is a novel perspective complementary to SPD Riemannian methods. Hyperbolic subject embeddings for weighted source selection is worth exploring.

---

## Paper 81 — HEEGNet: Hyperbolic Embeddings for EEG Cross-Subject Generalization (Relevance: 6/10)

**Title:** HEEGNet: Hyperbolic Embeddings for EEG

**Year:** 2026

**Venue:** ICLR 2026 (arXiv preprint)

**Dataset:** Multiple MI/ERP EEG datasets; cross-subject evaluation (RIKEN AIP / ATR)

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Demonstrate that EEG data exhibits hyperbolicity (tree-like hierarchical structure) and that hyperbolic embeddings improve cross-subject generalization compared to Euclidean.

---

**Methodology — Core Idea:**

Empirically measure hyperbolicity of EEG data via Gromov δ-hyperbolicity — confirms EEG exhibits significant hyperbolicity. HEEGNet: encode EEG into hyperbolic space using hyperbolic neural network layers; cross-subject attention in hyperbolic space; classification in the Poincaré ball.

**Methodology — Key Innovation:** First empirical demonstration that EEG is hyperbolic; HEEGNet as the first hyperbolic EEG network with cross-subject validation

---

**Findings — Main Results:**

- EEG data is strongly hyperbolic (δ-hyperbolicity significantly < Euclidean upper bound)
- HEEGNet outperforms Euclidean EEGNet for cross-subject generalization
- Hyperbolic embeddings improve subject-invariant representation learning

---

**Knowledge Extraction — What We Can Borrow:**

- Insight: Depression EEG likely also exhibits hyperbolicity (hierarchical brain network structure) — test Gromov δ-hyperbolicity on MODMA to motivate hyperbolic components
- Architecture: Hybrid SPD+hyperbolic: use Riemannian SPD for covariance features + hyperbolic space for subject embedding hierarchy
- Training: Hyperbolic cross-subject attention — subjects close in hyperbolic space contribute more to test-subject classification

**Relevance Score:** 6/10

**Reason:** Empirical hyperbolicity demonstration in EEG is novel and relevant — if confirmed for depression EEG, adding a hyperbolic subject embedding module to our LODO framework would be theoretically grounded.

---

## Paper 82 — EEG-MoCE: Hyperbolic Mixture-of-Curvature Experts for EEG Multimodal Learning (Relevance: 5/10)

**Title:** EEG-Based Multimodal Learning via Hyperbolic Mixture-of-Curvature Experts

**Year:** 2026

**Venue:** arXiv

**Dataset:** EEG + facial expression multimodal dataset

**Modality:** EEG + face video (multimodal)

**Relevance Score:** 5/10

**Summary:** Hyperbolic Mixture-of-Experts (MoE) for EEG+facial expression emotion recognition. Different curvature experts capture different hierarchical structures in the multimodal brain-face data. Hyperbolic MoE outperforms Euclidean MoE and single-curvature models.

**Knowledge Extraction — What We Can Borrow:** Mixture-of-Curvature Experts idea — different curvature values for different subject subpopulations; applicable to MODMA where MDD subtypes may have different geometry.

---

## Paper 83 — Robust Hyperbolic Learning with Curvature-Aware Optimization (NeurIPS 2025) (Relevance: 4/10)

**Title:** Robust Hyperbolic Learning with Curvature-Aware Optimization

**Year:** 2025

**Venue:** NeurIPS 2025

**Dataset:** Various EEG and graph datasets

**Relevance Score:** 4/10

**Summary:** Curvature-aware optimization for hyperbolic neural networks — solves the numerical instability issues in hyperbolic learning. Relevant technical background for implementing hyperbolic components in our LODO model.

---

## Paper 84 — Beyond Rigid Geometries: Spline-Pullback Metric for SPD (Relevance: 4/10)

**Title:** Beyond Rigid Geometries: The Spline-Pullback Metric for Universal Diffeomorphic SPD Representation Learning

**Year:** 2026

**Venue:** arXiv (Riemannian Geometry & SPD folder)

**Relevance Score:** 4/10

**Summary:** Proposes a flexible learnable Riemannian metric for SPD manifolds via spline-parameterized pullback — less constrained than AIRM or LogE. May improve SPDNet on heterogeneous datasets.

---

## Paper 85 — Riemannian Adversarial Attacks on SPD Matrices (Relevance: 3/10)

**Title:** Riemannian Adversarial Attacks on Symmetric Positive Definite Matrices

**Year:** 2026

**Venue:** arXiv

**Relevance Score:** 3/10

**Summary:** Adversarial attacks on Riemannian SPD classifiers — how to fool them. Relevant for understanding robustness of our LODO model.

---

## Paper 86 — Riemannian Block SPD Coupling Manifold and Optimal Transport (Relevance: 4/10)

**Title:** Riemannian Block SPD Coupling Manifold and Its Application to Optimal Transport

**Year:** 2026

**Venue:** arXiv

**Relevance Score:** 4/10

**Summary:** Novel block SPD manifold structure for optimal transport between covariance distributions — potentially useful for our Wasserstein-based alignment objectives.

---

## Paper 87 — Sheaf Neural Networks on SPD Manifolds (Relevance: 3/10)

**Title:** Sheaf Neural Networks on SPD Manifolds: Second-Order Geometric Representation Learning

**Year:** 2026

**Relevance Score:** 3/10 — Novel theoretical framework, limited practical applicability to EEG LODO.

---

## Paper 88 — Riemannian Optimization over SPD with Alpha-Procrustes Geometry (Relevance: 4/10)

**Title:** Riemannian Optimization over Symmetric Positive Definite Matrices with the Alpha-Procrustes Geometry

**Year:** 2026

**Venue:** arXiv

**Relevance Score:** 4/10

**Summary:** Alpha-Procrustes metric interpolates between different SPD geometries. Provides a unified optimization framework for SPD matrices.

---

## Paper 89 — Fast and Stable Riemannian Methods (Relevance: 4/10)

**Title:** Fast and Stable Riemannian SPD Methods

**Year:** 2026 (inferred from folder)

**Venue:** Unknown (filename 161_Fast_and_Stable_Riemannian.pdf)

**Relevance Score:** 4/10

**Summary:** Computational methods for fast stable SPD operations — relevant for implementing our LODO pipeline with 128×128 MODMA matrices efficiently.

---

## Paper 90 — Building Transformation Layers for SPDNet (Relevance: 4/10)

**Title:** Building Transformation Layers (filename 164_Building_Transformation_La.pdf)

**Year:** 2026

**Venue:** Unknown

**Relevance Score:** 4/10 — SPDNet layer design; background material for implementing our Riemannian backbone.

---

## Paper 91 — Riemannian Diffusion Models on General Manifolds (Relevance: 5/10)

**Title:** Riemannian Diffusion Models on General Manifolds via Physics-Informed Neural Networks

**Year:** 2026

**Venue:** arXiv

**Relevance Score:** 5/10

**Summary:** Generative diffusion models on Riemannian manifolds including SPD — applicable to EEG covariance matrix generation for MODMA augmentation.

**Knowledge Extraction — What We Can Borrow:** Riemannian diffusion for SPD data augmentation — more powerful than RGP-VAE (Paper 14) but potentially harder to train.

---

## Paper 92 — Riemannian Quasi-Newton Optimization with Euclidean Bounds (Relevance: 3/10)

**Title:** A Riemannian Quasi-Newton Algorithm for Optimization with Euclidean Bounds

**Year:** 2026

**Venue:** arXiv

**Relevance Score:** 3/10 — Numerical optimization for Riemannian problems; background for SPDNet training.

---

## Paper 93 — Multivariate Intrinsic Local Polynomial Regression on Isometric Riemannian Manifolds (Relevance: 3/10)

**Title:** Multivariate Intrinsic Local Polynomial Regression on Isometric Riemannian Manifolds

**Year:** 2026

**Venue:** arXiv

**Relevance Score:** 3/10 — Statistical regression on manifolds; background material.

---

## Paper 94 — Batch Normalization for Neural Networks on Complex Domains (Relevance: 3/10)

**Title:** Batch Normalization for Neural Networks on Complex Domains

**Year:** 2026

**Venue:** arXiv

**Relevance Score:** 3/10 — General batch normalization theory on complex manifolds; background for SPDDSMBN.


---

## Paper 95 — EEG Cross-Subject Taste Classification via Meta-Learning WGCNN (Relevance: 4/10)

**Title:** EEG Cross-Subject Taste Classification Method: A Meta-Learning Wavelet Graph Convolutional Neural Network Under Sweet and Bitter Stimuli

**Year:** 2026

**Venue:** MDPI journal

**Dataset:** Taste EEG dataset (sweet/bitter stimuli, 6 concentrations); cross-subject evaluation

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Cross-subject EEG classification of taste perception (sweet vs. bitter) using meta-learning + wavelet graph convolutional network.

---

**Methodology — Core Idea:**

ML-WGCNet: Wavelet transform on EEG temporal signals → graph convolutional network on electrode graph → meta-learning (MAML-style) for cross-subject adaptation. Meta-learning trains on source subjects to quickly adapt to new subjects with few labeled samples.

**Methodology — Key Innovation:** Meta-learning (MAML) for cross-subject EEG taste classification; wavelet-GCN for multi-scale spatiotemporal features

---

**Evaluation Protocol:**

**Subject Independence:** Yes

**Cross-Subject Evaluation:** Yes

**Leave-One-Subject-Out (LODO):** Yes

**Metrics:** Accuracy

---

**Knowledge Extraction — What We Can Borrow:**

- Technique: MAML-style meta-learning for cross-subject EEG — train source subjects to enable fast adaptation to target subject with K labeled samples; applicable to our LODO when a few labeled target samples are available
- Architecture: Wavelet-GCN for multi-scale spatiotemporal features — complementary to our Riemannian covariance approach

**Knowledge Extraction — What We Should NOT Borrow:**

- Taste perception (gustatory EEG) is very different from depression resting-state EEG
- Meta-learning requires labeled few-shot samples from target — our transductive setting has zero labels

**Relevance Score:** 4/10

**Reason:** Meta-learning cross-subject framework is transferable, but taste task and few-shot assumption don't align with our zero-label LODO protocol.

---

## Paper 96 — NEED: Cross-Subject and Cross-Task EEG Generalization for Video/Image Reconstruction (Relevance: 3/10)

**Title:** NEED: Cross-Subject and Cross-Task Generalization for Video and Image Reconstruction from EEG Signals

**Year:** 2025

**Venue:** NeurIPS 2025

**Dataset:** EEG visual decoding datasets; cross-subject and cross-task evaluation

**Modality:** EEG (physiological signals)

---

**Research Problem — Main Objective:**

Cross-subject and cross-task generalization for reconstructing visual stimuli (video, images) from EEG — enabling neural decoding without per-subject calibration.

---

**Methodology — Core Idea:**

NEED framework: multi-subject training with subject-adaptive normalization + task-agnostic representation learning. Cross-subject attention aligns representations across subjects; cross-task training improves general EEG visual features.

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: Subject-adaptive normalization in EEG representation learning — analogous to SPDDSMBN (Paper 22) but for Euclidean features
- Evaluation: Cross-subject AND cross-task evaluation demonstrates generalizability beyond single-paradigm

**Relevance Score:** 3/10

**Reason:** Visual decoding from EEG has different EEG dynamics than resting-state depression. Limited direct transferability.

---

## Paper 97 — Riemannian Outlier Detection for Manifold-Valued Functional Data (Relevance: 3/10)

**Title:** Outlier Detection for Riemannian Manifold-Valued Functional Data

**Year:** 2026

**Venue:** Applied Intelligence

**Dataset:** Long-haul flight trajectories (not EEG); spherical manifold

**Modality:** Manifold-valued functional data (methodology transferable)

---

**Research Problem — Main Objective:**

Robust Fréchet mean estimation and outlier detection for manifold-valued functional data via max-min clean subset selection + hypothesis testing on tangent space.

---

**Knowledge Extraction — What We Can Borrow:**

- Technique: Robust Fréchet mean estimation (outlier-resistant) — apply to MODMA covariance matrices to avoid contamination from EEG artifacts; cleaner per-subject and per-class Riemannian means

**Relevance Score:** 3/10

**Reason:** Statistical outlier detection on manifolds — minor relevance for robust covariance estimation in MODMA preprocessing.

---

## Paper 98 — SPD-DANN: Adversarial Domain Adaptation Directly on the SPD Manifold (Relevance: 8/10)

**Title:** SPD-DANN: An SPD manifold unsupervised domain adaptation method for cross subject motor imagery EEG decoding

**File:** Other Papers/EEG-BCI Classification/SPD-DANN... .pdf

**Authors:** Junshi Cheng, Ruisheng Ran, Bin Fang (Chongqing Normal University / Chongqing University) | **Year:** 2026 | **Venue:** Neural Networks (Elsevier)

**Dataset:** 4 MI BCI datasets — BCICIV-2a (9 subj, 22-ch, 4-class), BCICIV-2b (9 subj, 3-ch, binary), BCICIII-3a (3 subj, 60-ch, 4-class), BCICIII-4a (5 subj, 118-ch, binary, **class-imbalanced**)

**Modality:** Physiological Signals (EEG)

---

**Research Problem — Main Objective:**
Single-source unsupervised domain adaptation for cross-subject MI decoding by performing adversarial domain-invariant feature learning **directly on the SPD manifold** rather than in Euclidean space.

**Research Problem — Why It Matters:**
Existing cross-subject EEG UDA (DANN, MCD, CORAL, MMD-based) operates in Euclidean space and cannot capture the non-linear SPD structure of EEG covariance. No prior work had applied adversarial (GAN/DANN-style) training on the SPD manifold.

---

**Methodology — Core Idea:**
SPD-DANN = SPDNet + Manifold Attention (MAtt, Pan 2022) feature extractor, a classifier, and a domain discriminator connected via a **Gradient Reversal Layer (GRL)** — all on the SPD manifold. Two custom losses on top of the adversarial objective: (1) **SPD domain feature alignment loss** L_M = log-Euclidean distance between the log-E means of source and target SPD features (marginal alignment); (2) **SPD class prototype pair loss** L_P = margin-based triplet pulling same-class source/target log-E prototypes together and pushing different-class apart. Target prototypes use **pseudo-labels filtered by self-entropy** (keep top 50% most-confident), because a wrong pseudo-label distorts the log-Euclidean mean non-linearly (error amplification unique to the manifold).

**Methodology — Model Architecture:**

- Feature extraction: Conv+BN front-end → split temporal axis into n=3 non-overlapping segments → trace-normalized covariance (+εI) per segment → SPD sequence
- Backbone: Manifold Attention Module (log-Euclidean attention over the SPD sequence) → ReEig
- Metric: **Log-Euclidean (LEM)** chosen over AIRM for closed-form means
- Classifier / Discriminator: LogEig → FC; discriminator is deeper and GRL-coupled
- Adaptation: adversarial (GRL, dynamic λ_D schedule, Ganin) + L_M + L_P

**Methodology — Learning Strategy:** Single-source unsupervised domain adaptation (uses unlabeled target at **training** time); no target labels.

**Methodology — Key Innovation:** First adversarial training on the SPD manifold; self-entropy-filtered manifold class-prototype loss to safely use pseudo-labels on the manifold.

---

**Evaluation Protocol — Dataset Split:** Single-source → single-target cross-subject (each source subject → each target subject); session split for 2a/2b.

**Subject Independence:** Yes | **Cross-Subject:** Yes | **LODO:** Partial (pairwise single-source, not multi-source LOSO) | **Cross-Dataset:** No

**Metrics:** Classification accuracy (incl. on imbalanced 4a — no AUC reported)

---

**Findings — Main Results:**

- Beats 9 SoA UDA baselines (DANN, MCD, JDD, DALN, DDAN, SAAE, JDA, S3T, Reverse KL) on all 4 datasets
- 2a: 44.9% (4-class, +6.6% over MCD, +11.5% over DANN); 2b: 76.5% (+14.8% over MCD); 3a: 49.4%; 4a (imbalanced): 74.7% (+8.0% over JDD)
- **Disentangling ablation (Table 10):** Euclidean DANN 33.4% → Euclidean DANN + same losses 37.3% → **SPD geometry-only (no custom losses) 39.8%** → Full SPD-DANN 44.9%. Geometry alone beats optimized Euclidean; losses add +5.1%.
- L_P contributes more than L_M (−3.2% vs −2.5% when removed); dynamic λ_D ≫ fixed; pseudo-label entropy filter essential.

**Findings — Important Observations:**

- Transfer accuracy strongly tracks **source↔target subject similarity** (A1↔A3 ~68% both directions vs A2↔A3 ~30%) — quantitative evidence for source selection.
- Pseudo-labels are geometrically risky on the SPD manifold (non-linear mean distortion) → confidence filtering required.
- Works on imbalanced data (4a) without explicit balancing.

**Findings — Failure Cases:**

- **Single-source → single-target** protocol (one subject as source) → low absolute numbers; not multi-source LOSO.
- Covariance (not correlation); LEM (not AIRM); **train-time UDA only** — no test-time / inductive-then-adapt mode.
- Accuracy-only (even on imbalanced 4a); MI task, not psychiatric.

---

**Knowledge Extraction — What We Can Borrow:**

- Architecture: GRL adversarial discriminator **on the SPD manifold** with dynamic λ schedule — validates SPD-adversarial alignment as an alternative/complement to SPDDSMBN normalization (Paper 22).
- Loss: **SPD class prototype pair loss** + **self-entropy pseudo-label filtering** — directly usable for our transductive class-prototype alignment on MODMA (replaces/augments Paper 28 multicentric pseudo-labeling with a manifold-safe version).
- Loss: SPD domain feature alignment (log-E mean distance) — simple marginal alignment regularizer.
- Architecture: MAtt log-Euclidean manifold attention over temporal SPD segments — pooling/attention alternative to RHOP (Paper 3) / GyroAtt (Paper 73).
- Evidence: Table 10 is the cleanest single-paper proof that SPD geometry > optimized Euclidean (supports CP1); subject-similarity result supports source selection (Papers 31, 35).

**Knowledge Extraction — What We Should NOT Borrow:**

- Single-source→single-target protocol — must be re-run as multi-source LODO for fair comparison.
- Covariance + LEM — we use correlation manifold (CP2) + AIRM/OLM where affordable.
- Train-time-only UDA — our contribution is the test-time transductive step it lacks.
- Accuracy-only reporting — use AUC for imbalanced depression.

---

**Relevance Score:** 8/10

**Reason:** Closest source-side architectural competitor to our proposed model and the first adversarial DA on the SPD manifold — directly validates Riemannian-adversarial alignment for cross-subject EEG and supplies a manifold-safe pseudo-label class-prototype loss we can borrow. Capped below 9–10 because it is MI (not depression), covariance+LEM (not correlation), single-source train-time UDA (not multi-source LODO with test-time transduction), and accuracy-only.

---

**Possible Integration:**

- Directly usable: entropy-filtered SPD class-prototype pair loss as our transductive label-propagation/alignment objective; GRL adversarial branch as an alignment-philosophy alternative to SPDDSMBN.
- Requires modification: re-run SPD-DANN in multi-source LODO on MODMA as a baseline; swap covariance→correlation, LEM→AIRM/OLM; add test-time target Fréchet update.
- Future experiment: alignment-philosophy ablation — normalization-based (SPDDSMBN) vs discriminator-based (SPD-DANN GRL) vs both, on MODMA LODO.

