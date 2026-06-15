# TNSRE Writing Style Guide

## Source papers analyzed
- **WDANet** — Shen et al., IEEE Transactions on Affective Computing, Vol. 17, No. 1, Jan–Mar 2026
- **EEG-RCformer** — Feng & Yan, Mathematics 2026, 14, 1327 (MDPI open access)
- **SPD-DANN** — Cheng et al., Neural Networks 202 (2026) 109076 (Elsevier)
- **TSA** — Bleuzé et al., ICEET 2021 (IEEE conference, Riemannian BCI)

---

## 1. Paper Structure

### WDANet (IEEE TAC, 14 pages, two-column)
```
Abstract + Index Terms
I. Introduction
II. Related Work
    A. EEG-Based Depression Recognition
    B. EEG-Based Domain Adaptation
III. Method
    A. Overview
    B. Feature Extractor
    C. Label Classifier
    D. Global and Local Domain Discriminators
    E. Dynamic Adversarial in WDANet
    F. Wasserstein Distribution Discriminator
    G. Training Process
IV. Experiments
    A. Datasets
    B. Implementation Details
    C. Results
    D. Ablation Study
    E. Statistical Analysis
V. Discussion
VI. Conclusion
References
Author Biographies
```

### EEG-RCformer (Mathematics MDPI, 15 pages, single-column)
```
Abstract + Keywords + MSC
1. Introduction
2. Materials and Methods
    2.1. Spatiotemporal Feature Extraction
    2.2. Hybrid Clustering with AIRM-Informed Centroid Proposals
    2.3. Cluster-Based Token Generation
    2.4. Cognitive State Classification with a Transformer Encoder
    2.5. Datasets, Preprocessing, and Experimental Setup
3. Results
    3.1. Main Results
    3.2. Computational Complexity Analysis
    3.3. Model Visualization and Interpretation
    3.4. Ablation Studies
        3.4.1. Impact of the Clustering Module
        3.4.2. Impact of the Clustering Metric: Riemannian vs. Euclidean
        3.4.3. Sensitivity to the Number of Clusters
4. Discussion
    4.1. Why Geometry-Aware Clustering Benefits EEG Decoding
    4.2. Interpretation of Frequency-Band and Regional Effects
    4.3. Limitations and Future Work
5. Conclusions
References
```

### SPD-DANN (Neural Networks, Elsevier, ~12 pages, two-column)
```
Abstract + Keywords
1. Introduction
2. Related Work
    2.1. Unsupervised domain adaptation
    2.2. SPD manifold [background subsection]
    2.3. SPDNet and manifold attention module
3. Proposed Method
    3.1. SPD-DANN network structure
    3.2. Domain adaptation strategies
        3.2.1. SPD domain feature alignment loss function
        3.2.2. SPD class prototype pair loss function
    3.3. Optimization
4. Experiment
    4.1. Dataset and preprocessing
    4.2. Experimental setup
    4.3–4.6. Experimental results on [dataset name] (one subsection per dataset)
    4.7. Ablation study, sensitivity analysis of hyperparameters, and feature visualization
5. Conclusion (+ future prospects)
References
```

### Prescriptive rules for your paper
- **Use Roman numerals for top-level sections** (IEEE TNSRE style): I. Introduction, II. Related Work, III. Methods, IV. Experiments, V. Discussion, VI. Conclusion.
- **Use capital letters for first-level subsections** (A., B., C.) and Arabic numerals for second-level (A.1, A.2 or numbered subsections without letter prefix in TNSRE).
- **Aim for 6 top-level sections** in the order shown above.
- **Methods section**: 5–7 subsections covering (A) problem formulation/notation, (B) feature extraction, (C) Riemannian geometry operations, (D) architecture overview, (E) classification/adaptation module, (F) optimization/training.
- **Experiments section**: always include (A) Datasets, (B) Implementation Details / Baselines, (C) Main Results, (D) Ablation Study, (E) Statistical Analysis. The statistical analysis subsection is present in WDANet and should be included.
- **Discussion section**: separate from Conclusion; 2–4 subsections. Covers (a) interpretation of why the method works, (b) limitations, (c) future work.

---

## 2. Introduction Pattern

### Paragraph-by-paragraph structure (prescriptive)

**P1 — Clinical/epidemiological motivation (2–4 sentences).**
Open with a single declarative sentence establishing the clinical problem at population scale. Follow immediately with 1–2 sentences on consequences (functional impairment, suicide risk). Close with a sentence on the clinical need for objective detection.

> WDANet: "Depression is a prevalent mental disorder affecting millions of individuals globally, leading to emotional distress, cognitive and motor functional disruptions, and impaired performance in professional and daily life activities."

**P2 — EEG as a tool + ML/DL on EEG (3–5 sentences).**
Introduce EEG as a non-invasive modality, list 2–3 application domains where it has been used, then pivot to ML/DL methods applied to EEG for depression/BCI.

**P3 — Core technical limitation (3–5 sentences).**
State the distribution shift / cross-subject variability problem concisely. Provide a concrete example of failure ("a model trained on one group may fail to generalize to another"). Use this paragraph to frame the gap, not yet the solution.

**P4 — Survey of existing approaches + their residual limitations (4–8 sentences).**
Review 2–3 families of methods (statistical divergence, adversarial training, Riemannian methods). Each family gets 1–2 sentences. End with a "Nevertheless, these methods have limitations..." sentence that names the specific gap your paper fills.

**P5 — Proposed method + mechanism (3–5 sentences).**
Introduce the proposed model by name ("we propose X, a Y for Z"). State the 2–3 key technical mechanisms. Do not yet list contributions — save that for P6.

**P6 — Contribution list (numbered or bulleted, 3–4 items).**
WDANet uses a numbered list with italic subheadings ("*Dynamic domain adversarial training:* ..."). EEG-RCformer uses prose bullets starting "First, ... Second, ... Third, ...". SPD-DANN uses a numbered list starting "To the best of our knowledge, this is the first work to...".

**P7 — Paper organization sentence (1 sentence).**
"The subsequent sections are structured as follows: In Section II, we delve into the related work. Subsequently, we detail..."

### Key opening conventions
- Do NOT open with a rhetorical question or a quote.
- Open with a statement of scale or clinical consequence, not with "In recent years."
- First sentence should name the disease/problem, not the modality.
- Average introduction length: **6–7 paragraphs**, approximately 700–900 words.

### Contribution list format
- Use **3–4 contributions**, not 2 and not 5+.
- Each bullet begins with a past or present-tense active verb: "We introduce," "We propose," "We demonstrate," "We validate."
- Each bullet is 2–4 sentences: one sentence naming what you do, one sentence explaining the mechanism, one sentence on the benefit.
- WDANet style: numbered list `1) ... 2) ... 3)` with italic subheading followed by colon.
- TNSRE standard: numbered list preferred over bullets.

---

## 3. Related Work Pattern

### Grouping convention
All papers group by **method family**, not by year. WDANet uses two subsections: "A. EEG-Based Depression Recognition" and "B. EEG-Based Domain Adaptation." SPD-DANN uses "Unsupervised domain adaptation" and "SPD manifold" (a background subsection for mathematical prerequisites).

**For a TNSRE paper on Riemannian EEG depression detection, use:**
```
A. EEG-Based Depression Detection
B. Domain Adaptation for EEG
C. Riemannian Geometry in EEG Analysis
```

### Within each subsection
- Open the subsection with a 1–2 sentence orientation sentence naming the topic and its relevance.
- Each paragraph covers one sub-family or technique type (e.g., "machine learning approaches," "deep learning approaches").
- Citation density: **3–6 citations per paragraph** on average. WDANet regularly cites 4–5 papers per paragraph (e.g., "[37] and [38] both developed CNN-based models... Wu et al. [39] employed... Ksibi et al. [31] compared...").
- Author-year citation format (IEEE numeric) — always "[N]" inline, never "(Author, year)."
- Each subsection ends with a 2–3 sentence "gap" paragraph that begins: "Although these methods have demonstrated effectiveness, challenges remain in..." or "Nevertheless, these methods have limitations in..."

### Transition to your work
Do NOT use "In contrast to all prior work, we..." in Related Work. Instead, close Related Work with a paragraph in the final subsection that identifies the specific unaddressed challenge and states that the proposed method addresses it — but save the explicit comparison for the Discussion section. WDANet closes its Related Work with "However, a critical aspect that remains underexplored... This scheme facilitates a nuanced understanding of domain shifts, making it particularly relevant for domain adaptation tasks."

---

## 4. Methods Pattern

### Notation introduction
- Both WDANet and SPD-DANN introduce notation **inline within the first subsection** (Overview or network structure), not in a dedicated "Notation" section.
- SPD-DANN uses a bolded "**Notation.**" paragraph (unindented, run-in heading) early in the relevant subsection. Follow this pattern: bold the word "Notation." and then define symbols in a single paragraph.
- Use $\mathcal{D}_s$ / $\mathcal{D}_t$ for source/target domains consistently throughout.
- Define all symbols at first use with "where X denotes..." or "where X is the..."

### Mathematical definitions
- No "Definition" environment (theorem-style box). Definitions are stated inline: "Let $\Sigma_c \in \mathcal{S}^d_{++}$ denote the symmetric positive definite (SPD) covariance matrix for channel $c$..."
- Equations are numbered sequentially: (1), (2), (3)... No equation naming.
- Each equation is followed immediately by a sentence beginning "where..." that defines every symbol.
- After the where-clause, add 1–2 sentences interpreting what the equation means operationally.

### Describing SPD/Riemannian operations
Follow this exact level of detail:
1. State the space: "SPD matrices naturally lie on a Riemannian manifold rather than a flat vector space."
2. Name the metric: "Channel similarity is quantified using the affine-invariant Riemannian metric (AIRM):" followed by the equation $\delta_R(\Sigma_i, \Sigma_j) = \|\log(\Sigma_i^{-1/2} \Sigma_j \Sigma_i^{-1/2})\|_F$.
3. Explain why Euclidean fails first, then explain why Riemannian works.
4. Keep the mathematical exposition at the level of: what the operation is, its formula, and what it computes — do not prove properties.

### Architecture description order
All papers use **top-down (overview first, then detail)**:
1. One paragraph giving the full pipeline in prose, referencing the architecture figure ("As illustrated in Fig. 1, the proposed X comprises three modules: ...").
2. Then one subsection per module, each opening with "The [module name] is designed to..." and closing with its loss function.
3. Training/optimization procedure is always the last subsection of Methods, often with a formal Algorithm box (WDANet has Algorithm 1 and 2; SPD-DANN has Algorithm 1 for optimization).

### Algorithm boxes
WDANet and SPD-DANN both include formal algorithm pseudocode boxes with **Input:**, **Output:**, **Initialization:**, and a **repeat...until convergence** or **while not converged do** loop. Include one algorithm box for the training procedure.

---

## 5. Results Pattern

### Table formatting conventions
- **Bold the best result in each column/row** — all four papers do this consistently.
- Second-best is underlined in SPD-DANN; not marked in WDANet (WDANet only bolds the best).
- Use $\pm$ notation for standard deviation: "0.9802 $\pm$ 0.0037" (EEG-RCformer style). WDANet reports accuracy without $\pm$ in the main comparison table but includes $\pm$ in ablation.
- Decimal places: accuracy reported to **2 decimal places** (e.g., 83.33, 75.52). AUC reported to **4 decimal places** (e.g., 0.9802).
- Metrics reported per-row in WDANet: Accuracy, Sensitivity, Specificity, F1-score — all in a single large table. For TNSRE, report at minimum: Accuracy, Sensitivity (Recall), Specificity, F1-score. Include AUC if available.
- Table caption format: All-caps title ("TABLE II"), followed by a descriptive title in small caps or italic: "EXPERIMENTAL RESULTS OF WDANET FOR FIVE CASES ON THREE DATASETS COMPARED WITH SOTA METHODS."
- Table numbers use Roman numerals in IEEE style: TABLE I, TABLE II, TABLE III.

### Framing comparisons
Use these sentence patterns (observed across all papers):

- "X achieved [metric] of Y%, demonstrating its [property]." (WDANet)
- "X outperforms Y by Z%, underscoring [mechanism]." (WDANet)
- "As demonstrated in Table N, X achieves a [metric] of Y, outperforming the most competitive baseline by Z." (SPD-DANN)
- "X consistently outperforms all comparative methods in terms of per-subject accuracy." (SPD-DANN)
- "The observed improvements may be related to several aspects of the proposed design." (EEG-RCformer — hedged)
- "These results suggest that [mechanism] can provide [benefit]." (EEG-RCformer)

Do NOT write "X is superior to Y" or "X is better than Y." Prefer "X outperforms Y" or "X achieves higher [metric] than Y."

### Statistical tests
- WDANet uses **Friedman test** (non-parametric, multi-method comparison) with **Nemenyi post-hoc test** for pairwise significance. Reports F-statistic and critical difference (CD). Reports p < 0.05 as threshold.
- WDANet ablation uses **paired samples T-test**, reports p-values (p < 0.05 marked with *, p < 0.01 marked with **).
- EEG-RCformer uses **paired t-test** between two variants, reports exact p-values and Cohen's d effect size.
- For TNSRE: use Friedman + Nemenyi for multi-method comparison; paired t-test or Wilcoxon signed-rank for ablation. Always report exact p-values, not just "p < 0.05."

### Ablation study structure
- Always a separate subsection (D. Ablation Study or 3.4. Ablation Studies).
- Test **one component removed at a time** (ablative, not combinatorial). WDANet tests: full model, minus WDD, minus LDD, minus all discriminators. EEG-RCformer tests: full AIRM, Euclidean-only, no clustering.
- Report all ablation results in a dedicated table with the variant name in the first column.
- Interpret each ablation result in 1–2 sentences, connecting it back to the mechanism.
- Close ablation with a synthesis sentence: "These results confirm the essential contributions of [component] to [model]'s [property]."

---

## 6. Sentence-Level Style

### Sentence length and structure
- Average sentence length: **25–35 words**. Mix short declarative sentences (10–15 words) with longer complex sentences (40–50 words), but avoid sentences exceeding 55 words.
- Favor **active voice** for methodological claims: "We propose," "We introduce," "The model captures," "The discriminator learns."
- Use **passive voice** selectively for results and procedural descriptions: "EEG signals were band-pass filtered," "The data was randomly partitioned," "All experiments were implemented on."
- Active/passive ratio: approximately **60% active, 40% passive**.

### Hedging conventions
These papers hedge consistently when interpreting results (especially EEG-RCformer, which is exemplary in calibrated hedging). Use these hedges:

| Claim type | Preferred hedges |
|---|---|
| Explaining observed improvement | "may be related to," "may be attributed to," "could be partly explained by" |
| Mechanism claims | "suggests that," "indicates that," "is consistent with" |
| Generalization claims | "demonstrates," "shows," "confirms" (stronger; use only with stat. sig. results) |
| Negative/null results | "does not reach statistical significance," "shows only a non-significant mean improvement" |
| Limitations | "remains limited to," "should be interpreted cautiously," "leave these questions to future work" |

Do NOT use "proves," "obviously," "clearly demonstrates," or "it is evident that."

### Transition phrases (observed repeatedly across all papers)
Use these transitions; they are natural to the genre:

1. "To address these challenges/issues/limitations, we propose..."
2. "Specifically, [model/component] [verb phrase]..."
3. "Furthermore, [claim]."
4. "In addition, [supporting point]."
5. "However, [limitation or contrast]."
6. "As illustrated in Fig. N / As shown in Table N, ..."
7. "In this study/work, we [conduct/employ/propose]..."
8. "To this end, [mechanism]..."
9. "Nevertheless, these methods have limitations in [specific aspect]."
10. "Subsequently, [next processing step]..."

### Equation references
- IEEE two-column papers (WDANet): use "(N)" inline: "as defined in (6)," "the loss in (14) enables..."
- MDPI single-column (EEG-RCformer): same pattern: "as shown in Eq. (4)" or "(Equation 5)."
- Do NOT write "Equation (N)" spelled out unless starting a sentence. Start sentences with "Eq. (N)" or restructure to avoid opening with the equation reference.

### Forbidden phrases and patterns
- Do not write "in recent years" as an opening — too generic.
- Do not write "state-of-the-art" as a noun phrase standing alone; always specify what it refers to.
- Do not write "It is worth noting that" (overused filler).
- Do not write "The experimental results show that our method is superior to..."
- Do not write "Due to the above reasons" (unclear antecedent).
- Do not start consecutive sentences with "The."
- Avoid "very" as an intensifier; instead use domain-appropriate qualifiers ("particularly pronounced," "substantially higher").

---

## 7. Abstract Template

### WDANet abstract structure (IEEE two-column, ~220 words)
- **S1** — The clinical/technical problem: "Researchers have long sought objective and quantifiable methods for recognizing depression."
- **S2** — The modality and its promise: EEG introduced, why it is relevant.
- **S3** — The core technical challenge: "However, the practical application of EEG signals faces significant challenges arising from distribution variability across different datasets and subjects."
- **S4** — Secondary challenge: conventional methods fail to capture dynamic transformations.
- **S5** — Proposed solution: "To address these issues, we propose [name], a [type] for [task]."
- **S6** — Technical description: the 2–3 key components explained in 1–2 sentences each.
- **S7** — Quantitative results: exact accuracy numbers across datasets/conditions: "WDANet achieved classification accuracies of 83.33%, 75.52%, 73.93%, 76.04%, and 70.94% in cross-subject, cross-dataset experiments."
- **S8** — Significance claim: "demonstrating its effectiveness and superiority compared to state-of-the-art methods."
- **S9** — Broader claim: impact on the field, clinical relevance, future directions.

### EEG-RCformer abstract structure (MDPI single-column, ~200 words)
- **S1** — Modality + task + why it is hard: EEG provides high temporal resolution but high-density recordings challenge Transformer-based models.
- **S2** — Second limitation: conventional Euclidean representations miss intrinsic geometry.
- **S3** — "To address these issues, we propose [name], a [description]."
- **S4** — Pipeline description (2–3 sentences): what the model does step by step.
- **S5** — Evaluation: datasets and paradigms tested.
- **S6** — Quantitative results: exact AUC numbers with $\pm$ notation.
- **S7** — Statistical significance: "Paired statistical tests further showed significant gains for MODMA..."
- **S8** — Null/hedged result: "while SEED still showed a positive but non-significant mean improvement in the subject-independent setting."

### Fill-in-the-blank template for your paper
```
[Disease/problem] affects [scale] individuals globally, resulting in [consequences].
[Modality] provides [properties], making it a promising tool for [task].
However, [core technical challenge: distribution shift / cross-subject variability] limits
the practical deployment of existing approaches. [Secondary limitation: e.g.,
Euclidean assumptions, single-dataset evaluation]. To address these issues, we propose
[model name], a [description] for [task]. [Model name] [mechanism 1] by [how],
[mechanism 2] by [how], and [mechanism 3] by [how]. We evaluate [model name] on
[datasets] under [paradigms], achieving [metric 1] of [value] on [dataset 1] and
[metric 2] of [value] on [dataset 2]. [Statistical significance sentence if applicable].
These results [demonstrate/indicate] [model name]'s [property], suggesting its potential
for [clinical/practical application].
```

---

## 8. Key Phrases (APPROVED)

These phrases appear verbatim or near-verbatim in multiple source papers and are safe to model:

### Proposing methods
- "we propose [name], a [adjective] [architecture] for [task]"
- "In this work/study, we introduce a novel framework for..."
- "we present [name], which [mechanism] by [approach]"
- "To effectively [goal], we propose..."

### Describing improvements
- "outperforms several baseline and SOTA methods, including..."
- "achieving classification accuracies of X%, Y%, and Z%..."
- "demonstrating its effectiveness and superiority compared to state-of-the-art methods"
- "consistently outperforms all comparative methods in terms of [metric]"
- "achieves a [metric] of X, outperforming the most competitive baseline by Y"
- "improves the average [metric] by X% compared to [baseline]"

### Describing mechanisms
- "captures both [coarse/global]-grained and [fine/local]-grained [discrepancies/features]"
- "facilitates the extraction of [domain-invariant / subject-invariant] features"
- "effectively reduces [distribution discrepancies / domain shift] between source and target [domains/subjects]"
- "leveraging the [Riemannian/Wasserstein] [metric/distance] to [measure/align]..."
- "naturally lie on a Riemannian manifold rather than a flat vector space"
- "the affine-invariant Riemannian metric (AIRM)"
- "symmetric positive definite (SPD) covariance matrices"
- "domain-invariant representations"

### Describing limitations and future work
- "the current evaluation is restricted to [N] datasets; broader validation is still needed"
- "these visualizations should not be interpreted as [X]; rather, they reflect [Y]"
- "Future work will [action 1] and [action 2]"
- "We leave these questions to future work"
- "These results should be interpreted cautiously, because..."

### Describing results with hedging
- "The observed improvements may be related to several aspects of the proposed design"
- "These results suggest that [mechanism] can provide [benefit]"
- "This finding suggests that [component] plays an important role in [function]"
- "This observation is broadly consistent with prior [EEG/neuroscience] studies showing that..."
- "the difference does not reach statistical significance" (for null results)

### Riemannian-specific phrases
- "respecting the intrinsic non-Euclidean geometry of covariance matrices"
- "embedded directly on the SPD manifold"
- "the log-Euclidean metric (LEM)"
- "projected onto the tangent space at [the geometric mean / a reference point]"
- "geodesic distance on the manifold"
- "manifold-aware [distance / representation / clustering]"
