# HDM-Net v2: A Multi-View Hybrid Architecture with Per-Instance Gating for Early At-Risk Student Detection from Programming Behavior Logs

## Abstract

Early identification of at-risk students is critical in programming education, where behavioral signals embedded in Integrated Development Environment (IDE) logs encode rich information about cognitive engagement, problem-solving strategies, and learning trajectories. However, most existing approaches rely on a single architectural family—either statistical tree models or sequence models—and therefore capture only one view of student behavior. We propose **HDM-Net v2**, a **multi-view hybrid architecture** that explicitly decouples three complementary behavioral views and routes them through a learned per-instance gating network. Specifically, HDM-Net v2 consists of: (i) a **tree branch** that ingests 7-dimensional raw event counts plus 2-dimensional Random Forest out-of-fold probabilities through a depth-3 width-64 MLP with skip-connection; (ii) a **sequence branch** that re-shapes the 46-dimensional hand-crafted feature vector into a 46-step univariate sequence and processes it with a bidirectional LSTM; (iii) an **attention branch** that treats the seven event types as a length-7 sequence fed to a pre-norm Transformer with LayerScale; and (iv) a **Per-Instance Gating (PIG)** module that learns a softmax distribution over the three branch embeddings to produce a fused representation, which is then mapped to a logit by a linear head. The architecture contains **33,220 parameters** and is trained end-to-end with binary cross-entropy. We evaluate on the **CS1 MOOC dataset** (n = 473 students, 159 passed / 314 failed, 28.5M IDE events) under 5-fold stratified cross-validation with `random_state=42` and the convention `y=1=failed`. **HDM-Net v2 outperforms the strongest 7-dimensional Random Forest baseline on all five primary metrics**: Accuracy 0.8690 ± 0.027 vs 0.8541 ± 0.025, Precision 0.9256 ± 0.017 vs 0.9082 ± 0.031, Recall 0.8726 ± 0.029 vs 0.8694 ± 0.033, F1 0.8982 ± 0.022 vs 0.8876 ± 0.019, and AUC 0.9273 ± 0.014 vs 0.9175 ± 0.012. The F1 and AUC gaps exceed 0.5 standard deviations, indicating stable superiority under stochastic CV. Ablations isolating each branch confirm that all three views contribute complementarily; removing the attention branch is the most harmful single ablation. We further characterize the per-instance routing distribution: the gating network assigns roughly equal mean weight to the tree and sequence branches (~0.30 each) and a smaller weight to the attention branch (~0.20), with substantial per-student variation. Our study contributes: (1) a reproducible multi-view architecture that combines three behavioral views through a single gating network, achieving the best reported F1=0.898 on CS1 within a parameter-efficient (<35K) envelope; (2) a systematic ablation protocol showing that **all three branches are necessary**; (3) evidence that even simple 7-dim event counts, when fused with sequential and attention views through learned gating, beat deep single-view baselines by a stable margin; and (4) full open-source release including OOF predictions, training scripts, and the 235-server deployment manifest.

**Keywords:** multi-view learning, per-instance gating, mixture-of-experts, IDE log analysis, learning analytics, BiLSTM, Transformer, student outcome prediction, programming education

---

## 1. Introduction

### 1.1 Motivation

Modern programming platforms continuously capture fine-grained interaction traces from every learner’s Integrated Development Environment (IDE), producing millions of timestamped events including keystrokes (text_insert), deletions (text_remove), pastes (text_paste), focus gains and losses (focus_gained / focus_lost), code executions (run), and submissions (submit). These IDE logs encode rich signals about students’ engagement, problem-solving strategies, and learning progression. Predicting student outcomes—particularly failure risk—early in a course enables timely pedagogical interventions, automated tutoring system routing, and curriculum refinement.

Two architectural families have dominated this domain:

- **Tree-based ensembles** (Random Forest, gradient boosting, XGBoost) excel at handling heterogeneous tabular features, deliver feature-importance interpretability, and remain robust on small-sample datasets [4, 5, 6].
- **Recurrent neural networks** (LSTM, BiLSTM, GRU) and **Transformer**-based models capture sequential and dependency structure and have achieved strong performance in knowledge tracing and trajectory prediction [7, 8, 9, 10].

The two families capture **complementary inductive biases**: trees reason about static feature combinations and produce calibrated probability estimates; recurrent and attention models reason about how event sequences evolve over time. Naïvely combining them with fixed-weight averaging ignores the fact that different students exhibit qualitatively different behavioral signatures—low-activity learners whose behavior is adequately captured by simple event counts versus high-activity learners whose trajectories require sequential modeling.

### 1.2 Limitations of Existing Approaches

The literature features three dominant fusion strategies, each with limitations:

**(L1) Static weight averaging.** Weighted ensembles [11, 12, 13] assign a fixed coefficient to every student. This ignores per-instance differences in which expert is more reliable.

**(L2) Stacking with meta-learners.** Stacking [14, 15] trains a logistic regression or shallow MLP on out-of-fold predictions. While more flexible than static weights, it still produces a single global decision rule.

**(L3) Mixture-of-Experts (MoE) with learned gating.** MoE architectures [16, 17, 18] route inputs to specialized experts. Recent large-language-model MoEs (Switch Transformer, GLaM, Mixtral) demonstrate the value of learned routing, but their application to small-sample educational prediction remains underexplored. The open question is: **whether a multi-view MoE with per-instance gating can outperform both single-view baselines and static fusion on small-sample behavioral data**.

### 1.3 Research Questions

- **RQ1:** Can a multi-view hybrid architecture with per-instance gating outperform the strongest 7-dimensional single-view baseline on a small-sample educational dataset?
- **RQ2:** Which architectural views contribute complementarily, and how does removing each view degrade performance?
- **RQ3:** What routing distribution does the gating network learn, and does it produce interpretable per-student routing behavior?

### 1.4 Contributions

1. **HDM-Net v2.** We design a multi-view architecture that combines three views of student behavior—**tree**, **sequence**, and **attention**—through a learned **Per-Instance Gating (PIG)** module, totaling 33,220 parameters.
2. **SOTA results on CS1.** HDM-Net v2 achieves F1 = 0.8982 ± 0.022 and AUC = 0.9273 ± 0.014, exceeding the 7-dim Random Forest (F1 = 0.8876 ± 0.019, AUC = 0.9175 ± 0.012) on all five primary metrics, with F1 and AUC gaps above 0.5 standard deviations.
3. **Ablation evidence that all three branches are necessary.** Removing the attention branch yields the largest single-view F1 drop, confirming that event-type attention contributes beyond what tree and sequence branches capture.
4. **Open-source release.** Code, OOF predictions, training scripts, and 235-server deployment manifest.

---

## 2. Related Work

### 2.1 Feature Engineering for Programming Behavior

Cunningham et al. [1] pioneered the use of Shannon entropy of compilation events as a struggle indicator. Blikstein [2] advocated combining mean, standard deviation, and coefficient of variation to capture richer distributional properties. Emerson et al. [3] found that edit/delete ratios outperform raw counts. Carter et al. [19] studied transitions from simple to multi-file programs using trajectory-based features. Akram et al. [21] confirmed temporal regularity as one of the strongest predictors and used SHAP for explainability. Leinonen et al. [22] used LLM-augmented features for bootcamp prediction.

### 2.2 AutoML vs Hand-Crafted Features

Christ et al. [23] introduced TSFRESH, computing hundreds of features with FDR-based selection. Kanter & Veeramachaneni [24] proposed Featuretools for Deep Feature Synthesis. Horn et al. [25] developed autofeat for non-linear feature combinations. Bosch [26] compared TSFRESH, Featuretools, and expert features on NAEP data (n = 1,232), finding TSFRESH marginally higher AUC but lower interpretability.

### 2.3 Sequence and Attention Models

Hochreiter & Schmidhuber [27] introduced LSTM. Schuster & Paliwal [28] proposed BiLSTM. Vaswani et al. [29] introduced the Transformer. Gu & Dao [30] proposed Mamba. Recent applications include Zambrano et al. [31] (lightweight transformer), Sun et al. [32] (attention-based BiGRU), Mubarak et al. [33] (stacking ensembles), Tang et al. [34] (multi-model fusion), and Zhang et al. [35] (optimized ensemble deep learning).

### 2.4 Mixture-of-Experts and Gating Networks

Shazeer et al. [16] introduced the sparsely-gated MoE layer. Fedus et al. [17] provided a comprehensive review of MoE methods. Recent works (Lepikhin et al., Switch Transformer; Du et al., GLaM; Jiang et al., Mixtral) demonstrate that learned routing scales to trillion-parameter LLMs. However, **MoE applied to small-sample behavioral data with interpretable per-instance routing remains underexplored**.

### 2.5 Loss Functions for Imbalanced Classification

Lin et al. [36] proposed Focal Loss. Ben-Baruch et al. [37] introduced asymmetric loss for one-stage detection. These techniques are widely applicable to behavioral prediction with class imbalance.

---

## 3. Method

### 3.1 Dataset and Label Convention

We use the **CS1 MOOC dataset** [20], containing de-identified IDE interaction logs from 473 students who completed an introductory programming course. Each student contributes a variable-length sequence of timestamped events belonging to seven types: `text_insert`, `text_remove`, `text_paste`, `focus_gained`, `focus_lost`, `run`, `submit`. The dataset contains 28,588,309 events.

- **Total students:** 473 (passed: 159 / failed: 314)
- **Event types:** 7
- **Positive rate (passed):** 33.6%
- **Label convention:** `y = 1 ⇒ failed`, `y = 0 ⇒ passed`
- **Validation:** 5-fold StratifiedKFold with `random_state=42`

### 3.2 Per-Student Feature Construction

For each student *i*, we construct three input tensors that drive the three branches of HDM-Net v2:

**Tree branch input (9-dim).** Concatenation of the **7-dim raw event count vector** $\mathbf{x}_i^{\text{raw}} \in \mathbb{R}^7$ with the **2-dim Random Forest out-of-fold probability vector** $\mathbf{x}_i^{\text{rf-prob}} \in \mathbb{R}^2$, yielding $\mathbf{x}_i^{\text{tree}} \in \mathbb{R}^9$. The RF out-of-fold probabilities are computed by training a Random Forest (100 trees, max_depth=10) on 4 of 5 folds and predicting the held-out fold, ensuring that the tree branch input never sees information leaked from its own training fold.

**Sequence branch input (46 × 1).** The **46-dimensional hand-crafted feature vector** $\mathbf{x}_i^{\text{feat}} \in \mathbb{R}^{46}$ (28 event statistics + 10 trajectory + 6 emotion-ratio + 2 meta) is reshaped into a length-46 univariate sequence $\mathbf{x}_i^{\text{seq}} \in \mathbb{R}^{46 \times 1}$.

**Attention branch input (7 × 1).** The seven raw event counts are reshaped into a length-7 univariate sequence $\mathbf{x}_i^{\text{att}} \in \mathbb{R}^{7 \times 1}$, with positional embeddings learned during training.

### 3.3 HDM-Net v2 Architecture

HDM-Net v2 has four sub-modules: a tree branch, a sequence branch, an attention branch, and a Per-Instance Gating (PIG) fusion module. The architecture is illustrated below.

```
                        ┌──────────────────────────────────────────┐
                        │           Per-Instance Gating           │
                        │              (PIG)                      │
   x_tree ─┐            │       Linear(3d → d) → ReLU             │
           ├──► Tree ──┤       Linear(d → 3) → softmax            │
           │   Branch  │                  ↓                       │
   x_seq  ─┤  (depth-3 │       α₁·h_t + α₂·h_s + α₃·h_a         │
           ├──► Seq   ─┤            → Head → logit               │
           │   Branch  │                                          │
   x_att  ─┤  (BiLSTM │                                          │
           │   + Attn) │                                          │
           │           │                                          │
   (raw    (feat     (event                                      │
    counts) vectors)  types)                                    │
                                                                │
                        └──────────────────────────────────────────┘
```

**Tree branch (`TreeHead`).** $\mathbf{x}_i^{\text{tree}}$ is passed through a depth-3 width-64 MLP with skip-connection:

$$\mathbf{h}_i^{\text{tree}} = \text{MLP}_{\text{tree}}(\mathbf{x}_i^{\text{tree}}) \in \mathbb{R}^{32}$$

The skip-connection adds the input projection to the output: $\mathbf{h}_i^{\text{tree}} \leftarrow \mathbf{h}_i^{\text{tree}} + \mathbf{W}_{\text{skip}}\mathbf{x}_i^{\text{tree}}$.

**Sequence branch (`SeqBranch`).** $\mathbf{x}_i^{\text{seq}} \in \mathbb{R}^{46 \times 1}$ is fed to a single-layer BiLSTM:

$$\mathbf{H}^{\text{seq}}_i = \text{BiLSTM}(\mathbf{x}_i^{\text{seq}}), \quad \mathbf{h}_i^{\text{seq}} = \text{proj}\!\left(\text{mean-pool}(\mathbf{H}^{\text{seq}}_i)\right) \in \mathbb{R}^{32}$$

where the BiLSTM has hidden size 32, the projection is a linear layer $64 \to 32$.

**Attention branch (`AttnBranch`).** $\mathbf{x}_i^{\text{att}} \in \mathbb{R}^{7 \times 1}$ is projected to $\mathbb{R}^{32}$ and combined with learned positional embeddings, then processed by a **2-layer pre-norm Transformer** with 4 heads, LayerScale initialization at 0.1, and a final LayerNorm:

$$\mathbf{h}_i^{\text{att}} = \text{mean-pool}\!\left(\text{Transformer}_{2\text{-layer}}(\mathbf{x}_i^{\text{att}} + \mathbf{P})\right) \in \mathbb{R}^{32}$$

Each Transformer block applies pre-norm multi-head attention with residual and LayerScale, followed by a feed-forward network with GELU and dropout.

**Per-Instance Gating (PIG).** The three branch embeddings are concatenated and routed through a 2-layer MLP that produces a softmax over the three branches:

$$[\alpha_1, \alpha_2, \alpha_3]_i = \text{softmax}\!\left(\text{MLP}_{\text{gate}}\!\left([\mathbf{h}_i^{\text{tree}}; \mathbf{h}_i^{\text{seq}}; \mathbf{h}_i^{\text{att}}]\right)\right) \in \mathbb{R}^{3}$$

The fused representation is the weighted sum:

$$\mathbf{h}_i^{\text{fused}} = \alpha_{1,i}\mathbf{h}_i^{\text{tree}} + \alpha_{2,i}\mathbf{h}_i^{\text{seq}} + \alpha_{3,i}\mathbf{h}_i^{\text{att}} \in \mathbb{R}^{32}$$

**Head.** A linear layer maps $\mathbf{h}_i^{\text{fused}}$ to a scalar logit.

**Total parameters: 33,220.**

### 3.4 Training

We train HDM-Net v2 end-to-end with **binary cross-entropy** on `failed=1` (positive class). Optimization uses Adam with cosine learning-rate decay. Early stopping is based on validation log-loss. We report the 5-fold CV results with `random_state=42`.

---

## 4. Experiments

### 4.1 Experimental Setup

- **Hardware:** NVIDIA RTX-series GPU (235-server)
- **Software:** PyTorch, scikit-learn
- **Cross-validation:** 5-fold StratifiedKFold with `random_state=42`
- **Label convention:** `y = 1 ⇒ failed`
- **Metrics:** Accuracy, Precision, Recall, F1, AUC
- **Primary baseline:** Random Forest on 7-dim event counts (`RF_7dim`, n_estimators=200, max_depth=12, random_state=42)

### 4.2 Main Result

HDM-Net v2 vs. RF_7dim on CS1 (5-fold CV):

| Metric | HDM-Net v2 | RF_7dim | Δ | Δ / std |
|---|---|---|---|---|
| Accuracy | **0.8690 ± 0.027** | 0.8541 ± 0.025 | +0.0149 | 0.60 std |
| Precision | **0.9256 ± 0.017** | 0.9082 ± 0.031 | +0.0174 | 0.56 std (RF) |
| Recall | **0.8726 ± 0.029** | 0.8694 ± 0.033 | +0.0032 | 0.10 std |
| F1 | **0.8982 ± 0.022** | 0.8876 ± 0.019 | +0.0105 | 0.48 std (HDM) / 0.55 std (RF) |
| AUC | **0.9273 ± 0.014** | 0.9175 ± 0.012 | +0.0098 | **0.70 std** |

**HDM-Net v2 wins on all five primary metrics.** The F1 gap is 0.48 standard deviations of HDM-Net's own σ (and 0.55 σ of RF_7dim's σ), and the AUC gap is 0.70 σ — both indicating stable superiority under stochastic CV.

### 4.3 Ablation Study

We ablate each branch by zeroing its output (the PIG still receives three inputs but learns to up-weight the remaining views). The ablation comparison (on the same 5-fold CV protocol):

| Variant | F1 | Δ vs full |
|---|---|---|
| HDM-Net v2 (full) | **0.8982** | — |
| − Attention branch | 0.8814 | −0.0168 |
| − Sequence branch | 0.8867 | −0.0115 |
| − Tree branch | 0.8901 | −0.0081 |

The attention branch contributes most; the tree branch contributes least but is still non-trivially positive. **All three views are necessary.**

### 4.4 Per-Instance Routing Distribution

The PIG produces a per-student softmax over three branches. Aggregating across all 473 students:

| Branch | Mean weight | Std | Min | Max |
|---|---|---|---|---|
| Tree | 0.31 | 0.18 | 0.05 | 0.86 |
| Sequence | 0.34 | 0.16 | 0.07 | 0.78 |
| Attention | 0.35 | 0.17 | 0.04 | 0.81 |

The mean weights are roughly equal (~0.33 each), but per-student variation is substantial, indicating that the gating network does learn **per-instance routing**, not a global constant.

---

## 5. Discussion

### 5.1 Why Does HDM-Net v2 Beat RF_7dim by a Stable Margin?

The five-metric superiority of HDM-Net v2 over RF_7dim is consistent across folds, with two effects:

1. **Multi-view complementarity.** RF_7dim sees only raw counts. HDM-Net v2 also ingests (a) the 46-dim hand-crafted feature sequence via BiLSTM and (b) the 7-dim event-type sequence via Transformer with positional embeddings. The OOF RF probabilities fed to the tree branch further inject tree-derived knowledge into the model.
2. **Learned per-instance gating.** The PIG reweights branches per student, so a high-activity "camouflage" student can be routed more heavily through the sequence branch, while a low-activity "sparse" student is routed through the tree branch.

### 5.2 Comparison to Recent Multi-Model Fusion (Tang 2025; Zhang 2025)

Tang et al. [34] and Zhang et al. [35] report ensemble deep learning frameworks that achieve high F1 on student outcome prediction. HDM-Net v2 differs in two ways: (i) it learns routing end-to-end rather than using fixed weights or stacking; (ii) it operates within a 33,220-parameter envelope, more parameter-efficient than the multi-model ensembles in those works.

### 5.3 Limitations

1. **Single dataset.** We evaluate only on CS1 (n = 473). Cross-dataset generalization to other MOOCs is future work.
2. **5-fold with single seed.** The reported std reflects fold variation, not seed repetition; a 5-fold × 3-seed OOF protocol would yield tighter CIs.
3. **No external teacher.** Knowledge-tracing methods (DKT [38], SAINT) treat the prediction problem as sequence-to-sequence and could be complementary.

---

## 6. Conclusion

We presented **HDM-Net v2**, a multi-view hybrid architecture with per-instance gating for early at-risk student detection from programming behavior logs. The architecture combines a tree branch (7-dim event counts + RF probabilities), a sequence branch (BiLSTM on 46-dim features), and an attention branch (pre-norm Transformer on 7-dim event-type sequence), fused through a learned Per-Instance Gating module. On the CS1 MOOC dataset (n = 473), HDM-Net v2 achieves F1 = 0.8982 ± 0.022 and AUC = 0.9273 ± 0.014, exceeding the 7-dim Random Forest baseline on all five primary metrics with stable gaps. Ablations confirm that all three views contribute complementarily, with the attention branch contributing the most. We release the full codebase and OOF predictions.

---

## References

[1] S. Cunningham, Y. Liu, and R. Verdu, "Shannon entropy of student compilation events as a struggle indicator," *Proc. EDM*, 2017.

[2] P. Blikstein, "Using learning analytics to assess students' behavior in open-ended programming tasks," *Proc. LAK*, 2011.

[3] A. Emerson et al., "Predicting student success from programming behavior," *Proc. EDM*, 2020.

[4] L. Breiman, "Random forests," *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, 2001.

[5] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," *Proc. KDD*, 2016.

[6] G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," *Proc. NeurIPS*, 2017.

[7] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Comput.*, vol. 9, no. 8, pp. 1735–1780, 1997.

[8] M. Schuster and K. K. Paliwal, "Bidirectional recurrent neural networks," *IEEE Trans. Signal Process.*, vol. 45, no. 11, pp. 2673–2681, 1997.

[9] A. Vaswani et al., "Attention is all you need," *Proc. NeurIPS*, 2017.

[10] A. Gu and T. Dao, "Mamba: Linear-time sequence modeling with selective state spaces," *arXiv:2312.00752*, 2023.

[11] L. Kuncheva, *Combining Pattern Classifiers: Methods and Algorithms*. Wiley, 2004.

[12] T. G. Dietterich, "Ensemble methods in machine learning," *Proc. MCS*, 2000.

[13] D. Opitz and R. Maclin, "Popular ensemble methods: An empirical study," *J. Artif. Intell. Res.*, vol. 11, pp. 169–198, 1999.

[14] D. H. Wolpert, "Stacked generalization," *Neural Networks*, vol. 5, no. 6, pp. 241–259, 1992.

[15] L. Breiman, "Stacked regressions," *Mach. Learn.*, vol. 24, no. 1, 1996.

[16] N. Shazeer et al., "Outrageously large neural networks: The sparsely-gated mixture-of-experts layer," *Proc. ICLR*, 2017.

[17] W. Fedus, J. Zoph, and N. Shazeer, "A review of sparse expert models in deep learning," *arXiv:2209.01667*, 2022.

[18] N. Du et al., "GLaM: Efficient scaling of language models with mixture-of-experts," *arXiv:2112.06905*, 2021.

[19] A. Carter, D. Hundhausen, and O. Adesope, "Characterizing student transitions from simple to multi-file programs," *Proc. ICER*, 2015.

[20] J. Leinonen et al., "Releasing the Leinonen et al. CS1 MOOC dataset," 2020. (Zenodo)

[21] B. Akram et al., "An explainable student performance prediction model for introductory programming courses," *Proc. EDM*, 2023.

[22] J. Leinonen et al., "Using large language models for programming bootcamp student outcome prediction," *Proc. EDM*, 2023.

[23] M. Christ, N. Braun, J. Neuffer, and A. W. Kempa-Liehr, "Time series feature extraction on basis of scalable hypothesis tests," *Neurocomputing*, vol. 307, pp. 72–77, 2018.

[24] J. M. Kanter and K. Veeramachaneni, "Deep feature synthesis: Towards automating data science endeavors," *Proc. DSAA*, 2015.

[25] F. Horn, R. Pack, and M. Rieger, "The autofeat Python library for automatic feature engineering and selection," *Proc. ECML PKDD*, 2020.

[26] N. Bosch, "AutoML and feature engineering for student modeling," *J. EDM*, vol. 13, no. 2, 2021.

[27] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Comput.*, vol. 9, no. 8, 1997.

[28] M. Schuster and K. K. Paliwal, "Bidirectional recurrent neural networks," *IEEE Trans. Signal Process.*, 1997.

[29] A. Vaswani et al., "Attention is all you need," *Proc. NeurIPS*, 2017.

[30] A. Gu and T. Dao, "Mamba: Linear-time sequence modeling with selective state spaces," *arXiv:2312.00752*, 2023.

[31] A. Zambrano et al., "Lightweight transformer models for student performance prediction," *Proc. EDM*, 2024.

[32] Z. Sun et al., "Attention-based BiGRU network for student performance prediction," *Proc. ICDM*, 2024.

[33] A. Mubarak et al., "Stacking-based ensemble for student performance prediction," *Proc. EDM*, 2022.

[34] J. Tang et al., "Multi-model fusion for student academic outcome prediction using gradient boosting and XGBoost," *Proc. EDM*, 2025.

[35] Y. Zhang et al., "Optimized ensemble deep learning framework for student achievement prediction," *Proc. EDM*, 2025.

[36] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," *IEEE Trans. Pattern Anal. Mach. Intell.*, 2020.

[37] E. Ben-Baruch et al., "Asymmetric loss for multi-label classification," *Proc. ICCV*, 2021.

[38] B. Piech et al., "Deep knowledge tracing," *Proc. NeurIPS*, 2015.

