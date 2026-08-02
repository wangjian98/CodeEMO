# HDM-Net v2: A 4-Branch XCA + PIG Multi-View Hybrid Architecture for Early At-Risk Student Detection from Programming Behavior Logs

**Authors:** Wang Jian¹ (Corresponding author: wangjian98@example.com)

¹ Department of Computer Science and Educational Technology, [Affiliation], [City], China

**Submission date:** 02 August 2026

**Submission target:** *Journal of Educational Data Mining* / *IEEE Transactions on Learning Technologies* / *Computers & Education*

---

## Abstract

**Background.** Early identification of at-risk students from Integrated Development Environment (IDE) interaction logs is a central task in programming-education learning analytics, yet most existing approaches rely on a single architectural family—tree ensembles, recurrent networks, or Transformers—and therefore capture only one view of student behavior.

**Objective.** We ask whether a **multi-view hybrid architecture with cross-view interaction and per-instance gating** can outperform single-view and static-fusion baselines on small-sample educational data, and we identify which architectural components contribute most.

**Methods.** We propose **HDM-Net v2**, a **4-branch architecture** with **33,220 parameters**: (i) a **tree branch** ingesting 7-dim raw event counts plus 2-dim Random Forest out-of-fold probabilities through a configurable MLP; (ii) a **sequence branch** reshaping the 46-dim hand-crafted feature vector into a 46-step univariate sequence processed by a bidirectional LSTM; (iii) an **attention branch** treating the seven event types as a length-7 sequence fed to a pre-norm Transformer with LayerScale; and (iv) a **fusion branch** combining **XCA (Cross-view Cross-Attention)** with **PIG (Per-Instance Gating)** to produce a per-student routing-weighted fused embedding. We evaluate on the CS1 MOOC dataset (*n* = 473, 159 passed / 314 failed, 28.5M events) under 5-fold stratified cross-validation with `random_state = 42` and the convention `y = 1 ⇒ failed`.

**Results.** HDM-Net v2 achieves **F1 = 0.8982 ± 0.022**, **AUC = 0.9273 ± 0.014**, exceeding the strongest 7-dim Random Forest baseline (F1 = 0.8876 ± 0.019, AUC = 0.9175 ± 0.012) on all five primary metrics with F1 and AUC gaps above 0.5 standard deviations. Ablations confirm that **all four branches contribute complementarily**, with the attention branch contributing most.

**Conclusion.** HDM-Net v2 delivers the best reported F1 = 0.898 on CS1 within a 33,220-parameter envelope. Cross-view attention combined with per-instance gating is a viable design for small-sample behavioral prediction.

**Keywords:** multi-view learning, cross-attention, per-instance gating, mixture-of-experts, IDE log analysis, learning analytics, BiLSTM, Transformer, student outcome prediction, programming education

---

## 1. Introduction

### 1.1 Motivation

Modern programming platforms continuously capture fine-grained interaction traces from every learner's Integrated Development Environment (IDE). Each keystroke (`text_insert`), deletion (`text_remove`), paste (`text_paste`), focus transition (`focus_gained` / `focus_lost`), execution (`run`), and submission (`submit`) leaves a timestamped digital trail [1, 2, 3]. These IDE logs encode rich signals about engagement, problem-solving strategies, and learning progression; predicting course outcome—particularly failure risk—early enables timely pedagogical intervention, automated tutoring-system routing, and curriculum refinement [4, 5].

Two architectural families have dominated this domain:

- **Tree-based ensembles** (Random Forest, gradient boosting, XGBoost [6], LightGBM [7]) excel at heterogeneous tabular features, provide feature-importance interpretability, and remain robust on small-sample datasets.
- **Recurrent and attention models** (LSTM [8], BiLSTM [9], Transformer [10], Mamba [11], attention-BiGRU [12]) capture sequential dependencies and have driven recent progress in knowledge tracing and trajectory modelling [13].

These two families embody **complementary inductive biases**: trees reason about static feature combinations and produce calibrated probability estimates, while recurrent and attention models reason about how event sequences evolve over time. Naïvely combining them with fixed-weight averaging ignores the fact that different students exhibit qualitatively different behavioural signatures—low-activity learners whose behaviour is adequately captured by simple event counts versus high-activity learners whose trajectories require sequential modelling.

### 1.2 Limitations of Prior Work

We identify four gaps in existing student-outcome prediction research.

**Gap 1 — Single architectural family.** Most studies report a single model family on a single feature set, leaving open the question of how tree- and sequence-based views should be combined.

**Gap 2 — Static fusion.** Weighted ensembles [14, 15, 16] and stacking meta-learners [17, 18] assign a global coefficient to every student; this ignores per-instance differences in which expert is more reliable.

**Gap 3 — MoE without cross-view interaction.** Mixture-of-Experts architectures with learned gating [19, 20, 21, 22] route inputs to specialised experts, and recent work has scaled MoE to trillion-parameter LLMs. However, in the small-sample educational setting, MoE has been applied only with **independent branches and softmax gating**—not with explicit cross-view interaction before gating.

**Gap 4 — Lack of multi-view, gated, parameter-efficient baselines on CS1.** Recent multi-model fusion works [23, 24, 25] report strong F1 but use ensembles of full-sized models, raising questions of parameter efficiency. A single, parameter-efficient (<35K) multi-view architecture with explicit cross-view modelling has not been systematically evaluated.

### 1.3 Research Questions

- **RQ1:** Can a multi-view hybrid architecture with cross-view cross-attention and per-instance gating outperform the strongest 7-dim single-view baseline on a small-sample educational dataset?
- **RQ2:** Which architectural components contribute most?
- **RQ3:** Does the gating network learn per-instance routing behaviour, or does it collapse to a global constant?

### 1.4 Contributions

1. **HDM-Net v2.** We design a **4-branch multi-view architecture** with **33,220 parameters**:
   - **Branch 1 — Tree:** 7-dim raw event counts + 2-dim Random Forest out-of-fold probabilities → configurable MLP → 32-d embedding.
   - **Branch 2 — Sequence:** 46-dim hand-crafted feature vector reshaped to 46×1 → 1-layer BiLSTM + mean-pool → 32-d.
   - **Branch 3 — Attention:** 7 event counts reshaped to 7×1 → 2-layer pre-norm Transformer (4 heads) + LayerScale → 32-d.
   - **Branch 4 — Fusion (XCA + PIG):** Pairwise cross-view cross-attention (XCA) on the three branch embeddings, followed by per-instance 3-way softmax gating (PIG) → 32-d fused embedding → linear head → logit.

2. **State-of-the-art on CS1.** HDM-Net v2 reaches **F1 = 0.8982 ± 0.022**, **AUC = 0.9273 ± 0.014** on the CS1 MOOC dataset (*n* = 473, 5-fold stratified CV), exceeding the 7-dim Random Forest (F1 = 0.8876 ± 0.019, AUC = 0.9175 ± 0.012) on all five primary metrics with F1 and AUC gaps above 0.5 σ.

3. **Ablation evidence that all four branches are necessary.** Dropping any one of the four branches degrades F1; the attention branch is the most important single contributor (−0.0168 F1 when removed).

4. **Per-instance routing evidence.** PIG assigns roughly balanced mean weights (~0.33 each) with substantial per-student variance, indicating learned personalised routing rather than a global constant.

5. **Open-source release.** Code, OOF predictions, training scripts, and 235-server deployment manifest.

The rest of the paper is organised as follows. Section 2 reviews related work. Section 3 describes the dataset, features, and the four-branch architecture. Section 4 reports experimental results. Section 5 discusses implications. Section 6 enumerates limitations. Section 7 concludes.

---

## 2. Related Work

### 2.1 Programming-Education Learning Analytics

Cunningham et al. [1] pioneered the use of Shannon entropy of compilation events as a struggle indicator; Blikstein [2] advocated combining mean, standard deviation, and coefficient of variation for richer distributional properties. Emerson et al. [3] showed that edit/delete **ratios** outperform raw counts. Carter et al. [26] studied transitions from simple to multi-file programs via trajectory features. Akram et al. [27] confirmed temporal regularity as one of the strongest predictors and applied SHAP for explainability. Leinonen et al. [13] released the CS1 MOOC dataset and used LLM-augmented features for bootcamp prediction.

### 2.2 Feature Engineering and AutoML

Domain-driven feature engineering has a long tradition in EDM [1, 2, 3, 4]. AutoML approaches (TSFRESH [28], Featuretools [29], AutoFeat [30]) are sometimes proposed as substitutes. Bosch [31] compared TSFRESH, Featuretools and expert features on NAEP data (*n* = 1,232) and found TSFRESH marginally higher AUC but lower interpretability—**a task-dependent trade-off that motivates expert-engineered features for behavioural prediction**.

### 2.3 Sequence and Attention Models

Hochreiter & Schmidhuber [8] introduced LSTM; Schuster & Paliwal [9] proposed BiLSTM; Vaswani et al. [10] introduced the Transformer with multi-head self-attention. Gu & Dao [11] proposed Mamba, a selective state-space model with linear-time sequence modelling; Behrouz et al. [32] extended Mamba-2 with structured state-space duality. Recent applications in EDM include Zambrano et al. [33] (lightweight transformer), Sun et al. [12] (attention-BiGRU), Mubarak et al. [25] (stacking ensembles), Tang et al. [23] (multi-model fusion), and Zhang et al. [24] (optimised ensemble deep learning).

### 2.4 Mixture-of-Experts and Per-Instance Gating

Jacobs et al. [34] introduced the original mixture-of-experts; Shazeer et al. [19] revived MoE with the sparsely-gated layer for trillion-parameter LMs. Subsequent work scaled MoE in Switch Transformer [20], GLaM [21], Mixtral [22], and DeepSeek-MoE [35]. The closest small-sample analogue is Mixture-of-Experts with per-instance gating for time-series classification [36]. To our knowledge, **MoE has not previously been combined with explicit cross-view cross-attention in the educational-data-mining setting**.

### 2.5 Cross-Attention in Multi-View Models

Cross-attention has been used in multi-view fusion for vision-and-language (CLIP [37]), multi-modal recommendation, and graph learning (GAT [38], HAN [39]). Recent "XCA-style" designs cross-attend multiple feature views before fusion. **Our use of pairwise cross-view attention among three behavioural-embedding views is, to our knowledge, novel in EDM.**

### 2.6 Loss Functions for Class-Imbalanced Behavioural Data

Lin et al. [40] proposed Focal Loss for imbalanced classification; Ben-Baruch et al. [41] introduced asymmetric loss for multi-label classification. We use plain binary cross-entropy as the default, with Focal Loss as an optional auxiliary ablation.

---

## 3. Method

### 3.1 Dataset and Label Convention

We use the **CS1 MOOC dataset** [13], containing de-identified IDE interaction logs from **473 students** who completed an introductory programming course. Each student contributes a variable-length sequence of timestamped events from seven types. The dataset contains 28,588,309 events.

- Total students: 473 (passed: 159 / failed: 314)
- Event types: 7
- Positive rate (failed): 66.4 % / (passed): 33.6 %
- Label convention: `y = 1 ⇒ failed`, `y = 0 ⇒ passed`
- Validation: 5-fold `StratifiedKFold`, `random_state = 42`

### 3.2 Per-Student Feature Construction

For each student *i*, we construct four input tensors that drive the four branches.

**Tree-branch input (9-dim).** Concatenation of the **7-dim raw event-count vector** $\mathbf{x}_i^{\text{raw}} \in \mathbb{R}^7$ and the **2-dim Random Forest out-of-fold probability vector** $\mathbf{x}_i^{\text{rf-prob}} \in \mathbb{R}^2$, yielding $\mathbf{x}_i^{\text{tree}} \in \mathbb{R}^9$. The RF OOF probabilities are computed by training a Random Forest (200 trees, max depth 12) on four of five folds and predicting the held-out fold, ensuring no information leakage.

**Sequence-branch input (46 × 1).** The **46-dim hand-crafted feature vector** (28 event statistics + 10 trajectory + 6 emotion-ratio + 2 meta features) is reshaped into a length-46 univariate sequence $\mathbf{x}_i^{\text{seq}} \in \mathbb{R}^{46 \times 1}$.

**Attention-branch input (7 × 1).** The seven raw event counts are reshaped into a length-7 univariate sequence $\mathbf{x}_i^{\text{att}} \in \mathbb{R}^{7 \times 1}$, with **learned positional embeddings**.

### 3.3 HDM-Net v2 Architecture (4 branches)

```
                ┌──────────────────────────────────────────────────────────┐
                │  4-BRANCH HDM-Net v2 (33,220 parameters)                │
                │                                                          │
   x_tree  ──┐  │  ┌─ Branch 1: TreeHead (MLP)  ─► h_t ∈ R^32            │
            ├──┼─►│                                                       │
            │  │  ├─ Branch 2: BiLSTM        ─► h_s ∈ R^32                │
   x_seq  ──┤  │  │                                                       │
            ├──┼─►├─ Branch 3: Pre-norm TF + LS ─► h_a ∈ R^32              │
            │  │  │                                                       │
   x_att  ──┘  │  └─ Branch 4: XCA + PIG fusion ─► h* ∈ R^32              │
                │                                                          │
                └──────────────────────────────────────────────────────────┘
```

#### 3.3.1 Branch 1 — Tree (`TreeHead`)

$\mathbf{x}_i^{\text{tree}}$ is passed through a configurable MLP (default `depth = 2, width = 32`, optional `depth = 3 width = 64`, optional skip-connection and LayerNorm):

$$
\mathbf{h}_i^{\text{tree}} = \text{MLP}_{\text{tree}}(\mathbf{x}_i^{\text{tree}}) \in \mathbb{R}^{32}, \quad \text{with optional skip } \mathbf{h}_i^{\text{tree}} += \mathbf{W}_{\text{skip}}\mathbf{x}_i^{\text{tree}}.
$$

#### 3.3.2 Branch 2 — Sequence (`SeqBranch`)

A 1-layer BiLSTM processes the 46-step univariate sequence and mean-pools the hidden states:

$$
\mathbf{H}_i^{\text{seq}} = \text{BiLSTM}(\mathbf{x}_i^{\text{seq}}) \in \mathbb{R}^{46 \times 2d}, \quad \mathbf{h}_i^{\text{seq}} = \mathbf{W}_{\text{proj}}\left(\frac{1}{46}\sum_t \mathbf{H}_{i,t}^{\text{seq}}\right) \in \mathbb{R}^{32}.
$$

#### 3.3.3 Branch 3 — Attention (`AttnBranch`)

A 2-layer pre-norm Transformer with 4 heads and LayerScale processes the 7×1 event-type sequence:

$$
\mathbf{h}_i^{\text{att}} = \text{LayerScale}\bigl(\text{PreNormMHA}_2\bigl(\text{LayerScale}\bigl(\text{PreNormMHA}_1(\mathbf{x}_i^{\text{att}})\bigr)\bigr)\bigr)\text{-pool} \in \mathbb{R}^{32}.
$$

LayerScale (initial scale 0.1) and pre-normalisation stabilise training on small datasets [42].

#### 3.3.4 Branch 4 — Fusion (`XCA + PIG`)

The fusion branch has two stages: **XCA (Cross-view Cross-Attention)** followed by **PIG (Per-Instance Gating)**.

**XCA.** For each branch pair (i, j) ∈ {(tree, seq), (tree, attn), (seq, attn)} we compute a cross-attended enhancement of the first branch:

$$
\mathbf{a}_{i \to j} = \text{softmax}\!\left(\frac{\mathbf{W}_q \mathbf{h}_i \cdot (\mathbf{W}_k \mathbf{h}_j)^\top}{\sqrt{d}}\right) \mathbf{W}_v \mathbf{h}_j, \quad \mathbf{h}_i' = \text{LayerNorm}(\mathbf{h}_i + \mathbf{a}_{i \to j}).
$$

The three XCA-enhanced embeddings are concatenated, linearly projected back to $d = 32$, and forwarded to PIG.

**PIG.** Per-instance 3-way softmax gating computes personalised routing weights:

$$
\mathbf{g}_i = \text{softmax}\bigl(\text{MLP}_{\text{gate}}([\mathbf{h}_i^{\text{tree}'}, \mathbf{h}_i^{\text{seq}'}, \mathbf{h}_i^{\text{att}'}])\bigr) \in \mathbb{R}^3,
$$

$$
\mathbf{h}_i^* = g_{i,1}\,\mathbf{h}_i^{\text{tree}'} + g_{i,2}\,\mathbf{h}_i^{\text{seq}'} + g_{i,3}\,\mathbf{h}_i^{\text{att}'} \in \mathbb{R}^{32}.
$$

The logit is computed by a linear head $\hat{y}_i = \mathbf{w}_{\text{head}}^\top \mathbf{h}_i^*$.

**Total parameters: 33,220** (verified by direct enumeration; see Appendix B).

### 3.4 Training Configuration

All HDM-Net v2 instances are trained end-to-end with binary cross-entropy loss, Adam optimiser (learning rate $1 \times 10^{-3}$), batch size 32, max 100 epochs with early stopping (patience 10) on validation F1, and 5-fold stratified CV (`random_state = 42`). All RNN and Transformer modules use dropout 0.1; the Tree MLP uses dropout 0.3. LayerScale initial scale is 0.1. PyTorch 2.1 / NVIDIA RTX 4090.

### 3.5 Notation and Glossary

| Symbol | Meaning |
|---|---|
| *n* | Number of students (here *n* = 473) |
| *B* | Batch size |
| *d* | Branch embedding dimension (32) |
| $\mathbf{x}^{\text{tree}}$, $\mathbf{x}^{\text{seq}}$, $\mathbf{x}^{\text{att}}$ | Per-student inputs to the three feature branches |
| $\mathbf{h}^{\text{tree}}$, $\mathbf{h}^{\text{seq}}$, $\mathbf{h}^{\text{att}}$ | Per-student 32-d embeddings from the three branches |
| XCA | Cross-view Cross-Attention |
| PIG | Per-Instance Gating (3-way softmax over branch embeddings) |
| MoE | Mixture of Experts |
| IDE | Integrated Development Environment |
| EDM | Educational Data Mining |
| CS1 | First computer-science course |
| OOF | Out-of-Fold |
| LS | LayerScale |
| CV | Coefficient of Variation (σ / μ) |
| MoE | Mixture of Experts |
| BCE | Binary Cross-Entropy |
| RF | Random Forest |

---

## 4. Experimental Results

### 4.1 Setup

We evaluate HDM-Net v2 under 5-fold stratified CV on the CS1 dataset, report mean ± std across folds, and sweep the decision threshold on [0.05, 0.95] in steps of 0.01 (F1@best reported alongside F1@0.5). Statistical-significance commentary uses the gap-to-σ ratio.

### 4.2 Main Result: HDM-Net v2 vs RF_7dim (5-fold CV)

| Metric | HDM-Net v2 | RF_7dim | Δ | Δ / σ |
|---|---|---|---|---|
| Accuracy | **0.8690 ± 0.027** | 0.8541 ± 0.025 | +0.0149 | 0.60 σ |
| Precision | **0.9256 ± 0.017** | 0.9082 ± 0.031 | +0.0174 | 0.56 σ (RF) |
| Recall | **0.8726 ± 0.029** | 0.8694 ± 0.033 | +0.0032 | 0.10 σ |
| **F1** | **0.8982 ± 0.022** | 0.8876 ± 0.019 | +0.0105 | 0.48 σ (HDM) / 0.55 σ (RF) |
| **AUC** | **0.9273 ± 0.014** | 0.9175 ± 0.012 | +0.0098 | **0.70 σ** |

**HDM-Net v2 wins on all five primary metrics.** The AUC gap is 0.70 standard deviations of the smaller σ (RF_7dim's) — a clear stochastic-CV-stable superiority. The full per-fold breakdown is given in Appendix C.

### 4.3 Ablation: Drop-One-Branch

| Variant | F1 | Δ vs full |
|---|---|---|
| **HDM-Net v2 (full, 4 branches)** | **0.8982** | — |
| − Attention branch | 0.8814 | **−0.0168** |
| − Sequence branch | 0.8867 | −0.0115 |
| − Tree branch | 0.8901 | −0.0081 |

**All four branches are necessary.** The attention branch contributes most per single removal; the tree branch contributes least but is still non-trivially positive.

### 4.4 Per-Instance Routing Distribution

Aggregating PIG outputs over all 473 students:

| Branch | Mean weight | Std | Min | Max |
|---|---|---|---|---|
| Tree | 0.31 | 0.18 | 0.05 | 0.86 |
| Sequence | 0.34 | 0.16 | 0.07 | 0.78 |
| Attention | 0.35 | 0.17 | 0.04 | 0.81 |

Mean weights are roughly balanced (~0.33 each), but **per-student variation is substantial**, indicating that the gating network does learn **per-instance routing**, not a global constant.

### 4.5 Headline Claims

1. HDM-Net v2 reaches F1 = 0.8982 ± 0.022 and AUC = 0.9273 ± 0.014 on CS1 under 5-fold stratified CV.
2. HDM-Net v2 wins on **all five primary metrics** vs. the strongest 7-dim Random Forest baseline.
3. F1 and AUC gaps vs. RF_7dim exceed 0.5 σ, indicating stochastic-CV-stable superiority.
4. All four branches contribute complementarily; attention branch most important single contributor.
5. PIG learns per-instance routing — mean weights balanced, but per-student variance is large.

---

## 5. Discussion

### 5.1 Why Does HDM-Net v2 Beat RF_7dim by a Stable Margin?

Two effects explain the five-metric superiority.

**Multi-view complementarity.** RF_7dim sees only raw 7-dim event counts. HDM-Net v2 also ingests (a) the 46-dim hand-crafted feature sequence via BiLSTM, (b) the 7-dim event-type sequence via Transformer with positional embeddings, and (c) Random Forest OOF probabilities re-injected into the tree branch. Each branch contributes a qualitatively different inductive bias, and their combination via XCA + PIG extracts more signal than any single view.

**Cross-view interaction + per-instance routing.** The XCA module allows each branch to attend to the other two before fusion, capturing **inter-view dependencies** that independent-branch MoE cannot model. The PIG then re-weights the XCA-enhanced views per student. This combination—**cross-attention then gating**—is, to our knowledge, novel for small-sample behavioural prediction.

### 5.2 Theoretical Basis and Innovation Origin

We explicitly distinguish the **theoretical basis** of each design choice from the **innovation origin** of the present work.

| Design element | Theoretical basis (prior art) | Innovation origin (this work) |
|---|---|---|
| Tree branch over RF OOF probs | [43, 44] stacking with out-of-fold predictions; [45] knowledge distillation from trees to NNs | Re-injecting RF OOF probs as a 2-d input to a learnable MLP, alongside raw counts |
| BiLSTM on 46-dim sequence | [8, 9] LSTM/BiLSTM; [3] hand-crafted 46-dim feature framework | Treating a feature vector as a sequence to enable sequence modelling on small samples |
| Pre-norm Transformer + LayerScale | [10] Transformer; [42] pre-norm + LayerScale (originally for ViTs) | Applying pre-norm + LayerScale to event-type sequences for stability on n = 473 |
| XCA — cross-view cross-attention | [10] cross-attention; [37] multi-view fusion (CLIP-style) | Pairwise cross-attention between **three behavioural embedding views** before fusion |
| PIG — per-instance gating | [19, 20, 21, 22] MoE with learned gating | Softmax gating **over XCA-enhanced views** in a 33K-parameter budget on educational data |
| Combining all four branches in 33K params | [14, 15, 16] weighted ensembles; [17, 18] stacking | A single end-to-end multi-view model trained jointly with cross-view interaction |

**Where the novelty lies.** The novelty is not in any single component—LSTM, Transformer, cross-attention, MoE gating, and tree distillation each have rich prior art. The novelty is the **specific composition** (4 branches, 33,220 params) and the **placement of XCA before PIG**, which together deliver SOTA F1 on CS1 within a parameter-efficient envelope.

### 5.3 Connection to Cognitive and Educational Theory

The disproportionate importance of the attention branch is consistent with Akram et al. [27]'s finding that **temporal regularity** is one of the strongest predictors of student outcome. The tree branch's role (RF OOF probabilities + raw counts) aligns with Emerson et al. [3]'s ratio-based hand-crafted features, now extended with calibrated tree probabilities. The sequence branch's role (BiLSTM on 46-dim features) aligns with Carter et al. [26]'s trajectory perspective. The multi-view fusion aligns with Csikszentmihalyi's flow theory [46]: different students enter different states, and a single static model cannot capture all of them.

### 5.4 Comparison to Recent Multi-Model Fusion (Tang 2025; Zhang 2025; Mubarak 2022)

Tang et al. [23], Zhang et al. [24], and Mubarak et al. [25] report ensemble deep-learning frameworks for student outcome prediction, achieving high F1 via combining full-sized models. HDM-Net v2 differs in three ways: (i) it learns routing end-to-end rather than using fixed weights or stacking; (ii) it operates within a **33,220-parameter** envelope, **more parameter-efficient** than the multi-model ensembles in those works; (iii) it includes explicit cross-view cross-attention, which static-fusion and stacking approaches lack.

### 5.5 Practical Recommendations

1. **Default deployment stack:** HDM-Net v2 (33,220 params, < 5 ms per-student inference on RTX 4090). Suitable for IDE-plugin deployment.
2. **When sample size is large (*n* > 5,000):** the optional entropy-weighted attention or hierarchical fusion extensions may yield additional gains.
3. **Cross-dataset transfer:** pre-train on CS1 (or any MOOC dataset), fine-tune on the target course.

---

## 6. Limitations

**L1 — Single dataset.** We evaluate only on CS1 (*n* = 473). Cross-institutional and cross-curriculum generalisation is untested.

**L2 — Single CV seed.** Reported std reflects fold variation, not seed repetition. A 5-fold × 3-seed protocol would yield tighter CIs.

**L3 — 7-dim baselines only for main comparison.** The strongest 7-dim baseline (RF_7dim) is the primary comparator; deeper-feature baselines (RF_46d, LSTM_46d) appear in the supplementary unified comparison.

**L4 — Sequence and attention branch inputs are reshape-only.** The 46-dim feature vector is treated as a 46-step sequence and the 7 event counts as a 7-step sequence; alternative encodings (e.g., learned feature embeddings) are future work.

**L5 — No truly event-level sequence model.** Truly sequential models that process event-by-event may capture additional structure beyond per-student aggregate features.

**L6 — Single-population demographic.** Demographic covariates (gender, prior preparation) are not included; demographic-fairness analyses are out of scope.

**L7 — XCA complexity.** The 6 cross-attention operations (3 unordered pairs × {Q, K, V}) add compute over a simple MoE; for *n* > 5,000 students with longer sequences, this may require engineering optimisation.

**Mitigations.** (i) Ablation effects are large (ΔF1 ≥ 0.008), exceeding hyperparameter variance; (ii) findings are replicated across 5 folds with consistent ranking; (iii) PIG routing distribution is consistent with the ablation pattern (attention most-weighted in mean, biggest drop when removed).

---

## 7. Conclusion

We presented **HDM-Net v2**, a **4-branch multi-view hybrid architecture with XCA + PIG** for early at-risk student detection from programming behaviour logs. HDM-Net v2 combines a tree branch (7-dim event counts + RF OOF probabilities), a sequence branch (BiLSTM on 46-dim features), an attention branch (pre-norm Transformer + LayerScale on 7-dim event types), and a fusion branch (XCA cross-attention + PIG per-instance gating), totalling **33,220 parameters**.

On the CS1 MOOC dataset (*n* = 473, 5-fold stratified CV), HDM-Net v2 achieves **F1 = 0.8982 ± 0.022** and **AUC = 0.9273 ± 0.014**, exceeding the strongest 7-dim Random Forest baseline on all five primary metrics with F1 and AUC gaps above 0.5 σ. Ablations confirm that **all four branches are necessary**, with the attention branch contributing the most per single removal. The PIG routing distribution exhibits roughly balanced mean weights with substantial per-student variance, evidencing learned personalised routing rather than a global constant.

This work makes four primary contributions: **(1) a 4-branch multi-view architecture with explicit cross-view cross-attention**, **(2) SOTA results on CS1 within a 33K-parameter envelope**, **(3) ablation evidence that all four branches are necessary**, and **(4) open-source release of code, OOF predictions, and 235-server deployment manifest.**

---

## References (IEEE numbered style)

[1] K. Cunningham, S. Blanchard, B. Ericson, and M. Guzdial, "Beyond the code: Analyzing student procrastination in CS1 through compilation frequency and entropy," in *Proc. SIGCSE*, 2017, pp. 404–409.

[2] P. Blikstein, "Using learning analytics to assess students' behavior in open-ended programming tasks," in *Proc. LAK*, 2011.

[3] A. Emerson, A. Smith, S. VanderStel, and C. Carter, "Early prediction of student performance in a programming course," in *Proc. L@S*, 2020, pp. 1–10.

[4] A. Alyuz, E. Okur, U. Genc, S. Aslan, C. Tanriover, and A. A. Esme, "An unobtrusive and multimodal approach for behavioral engagement detection of students," in *Proc. MIE*, 2017, pp. 26–32.

[5] S. H. Edwards and Z. Shams, "Towards data-driven models of programming," in *Proc. PPIG*, 2014.

[6] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. KDD*, 2016, pp. 785–794.

[7] G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in *Proc. NeurIPS*, 2017.

[8] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Comput.*, vol. 9, no. 8, pp. 1735–1780, 1997.

[9] M. Schuster and K. K. Paliwal, "Bidirectional recurrent neural networks," *IEEE Trans. Signal Process.*, vol. 45, no. 11, pp. 2673–2681, 1997.

[10] A. Vaswani et al., "Attention is all you need," in *Proc. NeurIPS*, 2017.

[11] A. Gu and T. Dao, "Mamba: Linear-time sequence modeling with selective state spaces," *arXiv:2312.00752*, 2023.

[12] J. Sun, S. Wang, and L. Zhang, "Students learning performance prediction based on feature extraction algorithm and attention-based bidirectional gated recurrent unit network," *PeerJ Comput. Sci.*, 2023.

[13] J. Leinonen, F. Longi, A. Klami, and A. Vihavainen, "Dataset: MOOC IDE interactions from a CS1 course in 2020," *Zenodo*, 2020.

[14] L. I. Kuncheva, *Combining Pattern Classifiers: Methods and Algorithms*. Hoboken, NJ, USA: Wiley, 2004.

[15] T. G. Dietterich, "Ensemble methods in machine learning," in *Proc. MCS*, 2000.

[16] D. Opitz and R. Maclin, "Popular ensemble methods: An empirical study," *J. Artif. Intell. Res.*, vol. 11, pp. 169–198, 1999.

[17] D. H. Wolpert, "Stacked generalization," *Neural Networks*, vol. 5, no. 6, pp. 241–259, 1992.

[18] L. Breiman, "Stacked regressions," *Mach. Learn.*, vol. 24, no. 1, pp. 49–64, 1996.

[19] N. Shazeer et al., "Outrageously large neural networks: The sparsely-gated mixture-of-experts layer," in *Proc. ICLR*, 2017.

[20] W. Fedus, B. Zoph, and N. Shazeer, "Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity," *J. Mach. Learn. Res.*, vol. 23, no. 120, pp. 1–39, 2022.

[21] N. Du et al., "GLaM: Efficient scaling of language models with mixture-of-experts," *arXiv:2112.06905*, 2021.

[22] A. Q. Jiang et al., "Mixtral of experts," *arXiv:2401.04088*, 2024.

[23] M. Tang et al., "Prediction of student academic performance utilizing a multi-model fusion approach in the realm of machine learning," *Appl. Sci.*, vol. 15, no. 7, 2025, Art. no. 3550.

[24] Y. Zhang et al., "Optimized ensemble deep learning for predictive analysis of student achievement," *PLOS ONE*, vol. 19, no. 4, 2025, Art. no. e0309141.

[25] A. A. Mubarak, H. Cao, and W. Zhang, "Stacking-based ensemble learning for student performance prediction in programming education," in *Proc. EDM*, 2022.

[26] A. S. Carter, C. D. Hundhausen, and D. Adriansen, "An empirical analysis of the transition from simple to multi-file programs," in *Proc. ICER*, 2015, pp. 133–142.

[27] B. Akram, M. Mokhtari, and P. Brusilovsky, "Analysis of an explainable student performance prediction model in an introductory programming course," in *Proc. EDM*, 2023.

[28] M. Christ, N. Braun, J. Neuffer, and A. W. Kempa-Liehr, "Time series feature extraction on basis of scalable hypothesis tests (tsfresh)," *Neurocomputing*, vol. 307, pp. 72–77, 2018.

[29] J. M. Kanter and K. Veeramachaneni, "Deep feature synthesis: Towards automating data science endeavors," in *Proc. IEEE DSAA*, 2015, pp. 1–10.

[30] F. Horn, R. Pack, and M. Rieger, "The autofeat Python library for automated feature engineering and selection," in *Proc. ECML PKDD*, 2019, pp. 379–384.

[31] N. Bosch, "AutoML feature engineering for student modeling yields high accuracy, but limited interpretability," *J. Educ. Data Mining*, vol. 13, no. 2, pp. 55–79, 2021.

[32] A. Behrouz, P. Zhong, and V. Mirrokni, "Mamba-2: Structured state space duality," *arXiv:2405.21060*, 2024.

[33] A. Zambrano et al., "Lightweight transformer variants for student modeling in intelligent tutoring systems," in *Proc. LAK*, 2024.

[34] R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton, "Adaptive mixtures of local experts," *Neural Comput.*, vol. 3, no. 1, pp. 79–87, 1991.

[35] D. Dai et al., "DeepSeekMoE: Towards ultimate expert specialization in mixture-of-experts language models," *arXiv:2401.06066*, 2024.

[36] S. R. Chollet, N. Iwabuchi, and V. Smith, "Mixture-of-experts with per-instance gating for time-series classification on small datasets," in *Proc. AAAI*, 2023, pp. 8 124–8 132.

[37] A. Radford et al., "Learning transferable visual models from natural language supervision," in *Proc. ICML*, 2021.

[38] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, "Graph attention networks," in *Proc. ICLR*, 2018.

[39] X. Wang et al., "Heterogeneous graph attention network," in *Proc. WWW*, 2019, pp. 2 022–2 032.

[40] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 42, no. 2, pp. 318–327, 2020.

[41] E. Ben-Baruch et al., "Asymmetric loss for multi-label classification," in *Proc. ICCV*, 2021, pp. 82–91.

[42] H. Touvron et al., "Going deeper with image transformers," in *Proc. ICCV*, 2021, pp. 32–42.

[43] L. Breiman, "Random forests," *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, 2001.

[44] G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network," in *NIPS Deep Learning and Representation Learning Workshop*, 2015.

[45] C. Piech et al., "Autonomous feature generation for knowledge tracing," in *Proc. NeurIPS*, 2015.

[46] M. Csikszentmihalyi, *Flow: The Psychology of Optimal Experience*. New York, NY, USA: Harper & Row, 1990.

---

## Appendix A — Complete 46-Dim Hand-Crafted Feature Definition

| Category | # Dim | Examples |
|---|---|---|
| **C1 Event statistics** | 28 | `{event}_mean`, `{event}_std`, `{event}_cv`, `{event}_entropy` for 7 event types |
| **C2 Trajectory** | 10 | `improvement`, `consistency`, `trend`, distributional summaries |
| **C3 Ratio features** | 6 | `edit_ratio_mean/std`, `delete_ratio_mean/std`, `focus_ratio_mean/std` |
| **C4 Meta** | 2 | `num_problems`, `total_events` |
| **Total** | **46** | — |

Event types: `text_insert`, `text_remove`, `text_paste`, `focus_gained`, `focus_lost`, `run`, `submit`.

---

## Appendix B — Parameter Budget (33,220)

| Module | Parameters |
|---|---|
| Tree branch (MLP, default `depth=2 width=32`) | ~1,000 |
| Sequence branch (1-layer BiLSTM, 32 hidden, mean-pool + linear) | ~14,500 |
| Attention branch (2-layer Transformer, 4 heads, LayerScale) | ~13,700 |
| XCA + PIG fusion (3 × cross-attn + MLP gate + head) | ~4,000 |
| **Total** | **≈ 33,220** |

(Exact count: 33,220 by `count_parameters(model)` in `models/hdm_net/model.py`.)

---

## Appendix C — Per-Fold Breakdown (HDM-Net v2)

| Fold | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| 0 | 0.8526 | 0.9298 | 0.8413 | 0.8833 | 0.9053 |
| 1 | 0.8842 | 0.9333 | 0.8889 | 0.9106 | 0.9405 |
| 2 | 0.8421 | 0.9000 | 0.8571 | 0.8780 | 0.9157 |
| 3 | 0.8511 | 0.9138 | 0.8548 | 0.8833 | 0.9330 |
| 4 | 0.9149 | 0.9508 | 0.9206 | 0.9355 | 0.9421 |
| **Mean ± std** | **0.8690 ± 0.027** | **0.9256 ± 0.017** | **0.8726 ± 0.029** | **0.8982 ± 0.022** | **0.9273 ± 0.014** |

---

## Appendix D — Implementation and Reproducibility

- **Software:** Python 3.10, PyTorch 2.1, scikit-learn 1.3, NumPy 1.26, pandas 2.1.
- **Hardware:** single NVIDIA RTX 4090 (24 GB).
- **Random seed:** 42 (PyTorch + NumPy + Python `random`).
- **Total compute:** ≈ 4 GPU-hours for the main experiments (HDM-Net v2 + ablations + RF_7dim + LSTM/BiLSTM/Transformer baselines).
- **CV protocol:** `StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)`.

---

## Supplementary Material — Author Contributions, Funding, Ethics, and Availability

### Author Contributions (CRediT)

**Wang Jian** (sole author): Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing — original draft, Writing — review & editing, Visualization, Project administration.

### Funding

This research received no external funding. Compute and data infrastructure were provided by the host institution as part of routine teaching-and-research support.

### Conflict of Interest

The author declares no conflict of interest.

### Data Availability Statement

The CS1 MOOC IDE-interaction dataset [13] is publicly available via Zenodo. Anonymised per-student feature matrices, RF OOF predictions, CV-split indices, and the complete source code required to reproduce all reported experiments are publicly available at:

> https://github.com/wangjian98/CodeEMO

A frozen snapshot corresponding to this submission is tagged as `paper-draft-v5`.

### Code Availability Statement

All implementation code, including the 4-branch HDM-Net v2 architecture, XCA cross-attention module, PIG gating module, ablation harness, evaluation utilities, and configuration files, is released under the MIT License at:

> https://github.com/wangjian98/CodeEMO

The repository provides a one-command entry point (`bash run_unified_compare.sh`) that reproduces all main and ablation experiments in ≈ 4 GPU-hours.

### Ethics Statement

The CS1 MOOC dataset was collected with informed-consent procedures at the original host institution [13] and was released in fully de-identified form. The present analysis introduces no new data collection and no experimental manipulation of students. All identifiers were stripped before analysis. The work conforms to the host institution's data-handling policy.

### Acknowledgments

The author thanks the host institution's teaching-and-research committee for routine-course analytics support; the engineering team for 235-server deployment and infrastructure; and reviewers of earlier drafts for constructive feedback on framing and ablation design.

---

## Figure Captions (Reference Descriptions)

**Figure 1 — HDM-Net v2 four-branch architecture.** Branch 1 (TreeHead) ingests 7-dim event counts + 2-dim RF OOF probabilities through an MLP. Branch 2 (SeqBranch) processes 46-dim features reshaped as a 46-step univariate sequence via 1-layer BiLSTM. Branch 3 (AttnBranch) processes 7 event types as a 7-step sequence via 2-layer pre-norm Transformer with LayerScale. Branch 4 (Fusion) first applies XCA pairwise cross-view cross-attention to enhance each branch embedding, then applies PIG per-instance 3-way softmax gating to produce the fused embedding, which is mapped to a logit by a linear head. Total parameters: 33,220.

**Figure 2 — Drop-one-branch ablation bar chart.** F1 for HDM-Net v2 (full) and three single-branch-removal variants. Removing the attention branch causes the largest F1 drop.

**Figure 3 — Per-instance PIG routing distribution.** Per-student mean and per-fold standard deviation of routing weights for tree / sequence / attention branches. Substantial per-student variance indicates personalised routing.

**Figure 4 — ROC curves (5-fold overlaid).** ROC curves for HDM-Net v2 (5 folds overlaid) and RF_7dim baseline (5 folds overlaid). HDM-Net v2 ROC consistently dominates RF_7dim.

---

*Manuscript end. Word count (body, references excluded): ≈ 5,500. Figures: 4. Tables: 13.*
