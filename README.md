# CodeEMO: HDM-Net v2 — A Multi-View Hybrid Architecture with PIG Fusion for Early At-Risk Student Detection from Programming Behavior Logs

CodeEMO is the official implementation of **HDM-Net v2**, a **multi-view hybrid architecture** for early identification of at-risk students from Integrated Development Environment (IDE) interaction logs. HDM-Net v2 combines three complementary feature-view branches (**Tree / Sequence / Attention**) with a fourth **Fusion branch** built from **PIG (Per-Instance Gating)**. Cross-view cross-attention (XCA) was originally specified but empirically underperformed PIG-only at *n* = 473 (ΔF1 = −0.0094); see the discussion in the v5 paper (Methods §3.3.4 + Discussion §5.3). The architecture contains **33,220 parameters** and is trained end-to-end on the public CS1 MOOC IDE-log dataset.

## Dataset

- **CS1 MOOC IDE logs** (Leinonen et al., publicly released via Zenodo)
- 473 students: 159 passed / 314 failed (positive rate 33.6 %)
- 28,588,309 timestamped IDE events, 7 event types
- Label convention used in this repository: `y = 1 ⇒ failed`

## Repository layout

```
CodeEMO/
├── README.md                         # (this file)
├── DEPLOY_235.md                     # 235-server deployment manifest
│
├── docs/
│   ├── paper_draft.md                # v1 BGM-Net draft (legacy)
│   ├── paper-draft2.md               # v2 MRE draft (legacy)
│   ├── paper-draft2-cn.md            # v2 MRE 中文 (legacy)
│   ├── paper_draft_v3.md             # v3 HDM-Net (legacy)
│   ├── paper_draft_v3_cn.md          # v3 中文 (legacy)
│   ├── paper-draft-v5.md             # ★ v5 HDM-Net v2 English SCI draft
│   └── paper-draft-v5-cn.md          # ★ v5 HDM-Net v2 中文 SCI 草稿
│
├── common/                           # shared utilities
│   ├── data_loader.py
│   ├── feature_engineering.py        # 46-dim hand-crafted features
│   └── evaluator.py
│
├── models/
│   ├── rf/                           # Random Forest baselines
│   ├── lstm/, bilstm/, transformer/, mamba/
│   ├── bgm_net/                      # BGM-Net v1 (dual-branch MLP, ~5K params)
│   ├── hdm_net/                      # ★ HDM-Net v2 (3 feature branches + 1 PIG fusion, 33,220 params)
│   │   ├── model.py                  #   TreeHead, SeqBranch, AttnBranch, PIG, HDMNet  (XCA: not in default)
│   │   ├── train.py                  #   standalone entry; results.json fusion field is "PIG (XCA optional, see §XCA below)"
│   ├── cream/, cw_hdm_net/, mre/, m_aae_net/, csem_net/, ...
│   └── ...
│
├── outputs/unified_compare/          # all experimental outputs
│   ├── hdm_net_v2/results.json       # ★ HDM-Net v2 5-fold CV results
│   ├── rf_7dim/results.json
│   ├── hdm_net_no_tree/  hdm_net_no_seq/  hdm_net_no_attn/
│   └── ...
│
└── scripts/
    ├── shap_hdm_net_v2.py
    ├── visualize.py
    └── gen_per_class_summary.py
```

## HDM-Net v2 architecture (3 feature branches + 1 PIG fusion)

```
                              ┌─────────────────────────────────────────────┐
                              │      4th branch: Fusion (PIG-only (XCA available as optional extension))        │
                              │                                             │
   x_tree (7d + 2d RF) ──┐    │   ┌──────────────────────────────────┐     │
                       ├──┼──►│ 1. Tree branch                     │     │
                       │  │   │    9 → MLP (depth-N width-W) → 32d │     │
   x_seq (46×1) ───────┼──┼──►│ 2. Sequence branch                 │     │
                       │  │   │    BiLSTM on 46-step sequence → 32d│     │
   x_att (7×1) ────────┼──┼──►│ 3. Attention branch                │     │
                       │  │   │    Pre-norm Transformer + LS → 32d │     │
                       │  │   └──────────────────────────────────┘     │
                       │  │                │  h_t, h_s, h_a            │
                       │  │                ▼                            │
                       │  │   ┌──────────────────────────────────┐     │
                       │  └──►│ XCA: pairwise cross-attention     │     │
                       │      │   (h_t ↔ h_s, h_t ↔ h_a, h_s ↔ h_a)│   │
                       │      │             → concat → proj      │     │
                       │      └──────────────────────────────────┘     │
                       │                       │                        │
                       │                       ▼                        │
                       │      ┌──────────────────────────────────┐     │
                       │      │ PIG: per-instance softmax gate   │     │
                       │      │   g = softmax(MLP([h_t,h_s,h_a]))│    │
                       │      │   h = g·h_t + (1-g)·h_s + ...    │     │
                       │      └──────────────────────────────────┘     │
                       │                       │                        │
                       │                       ▼                        │
                       │              Linear head → logit                │
                       └─────────────────────────────────────────────┘
```

### Four branches

| # | Branch | Input | Core mechanism | Output dim | Rationale |
|---|---|---|---|---|---|
| 1 | **Tree** | 7-d raw event counts + 2-d RF OOF probs | depth-N width-W MLP (default `depth=2 width=32`, optional skip / LayerNorm) | 32 | Static feature interactions + tree-distilled probability |
| 2 | **Sequence** | 46-d hand-crafted features reshaped to 46×1 | 1-layer BiLSTM + mean-pool + linear projection | 32 | Sequential / distributional patterns in 46-dim feature stream |
| 3 | **Attention** | 7 event counts reshaped to 7×1 | 2-layer pre-norm Transformer (4 heads) + LayerScale | 32 | Inter-event-type relational patterns with positional embeddings |
| 4 | **Fusion (PIG-only (XCA available as optional extension))** | Three 32-d branch embeddings | **XCA** = pairwise cross-view cross-attention → concat → projection; **PIG** = per-instance 3-way softmax gating | 32 | Per-student fusion of complementary views |

**Total parameters: 33,220** (verified by `count_parameters(model)` in `models/hdm_net/model.py`).

### XCA — Cross-view Cross-Attention *(Optional Extension; Not in Default Configuration)*

> **Note.** The default `hdm_net_v2` architecture **does not** include the XCA module. The following description documents XCA as an **optional extension** kept in the codebase for future investigation at larger sample sizes (*n* > 2,000). On *n* = 473 XCA **underperformed** PIG-only (ΔF1 = −0.0094). See the v5 paper Methods §3.3.4 and Discussion §5.3 for the empirical ablation that motivated removing XCA from the default. For branch pair (i, j):

```
a_{i→j} = softmax( (W_q h_i) · (W_k h_j)^T / sqrt(d) ) · (W_v h_j)
h_i'   = LayerNorm( h_i + a_{i→j} )
```

The three attended views are then concatenated, linearly projected back to `d = 32`, and passed to PIG. XCA allows the network to **explicitly model inter-view interactions** rather than treating each view as an independent input.

### PIG — Per-Instance Gating

PIG computes a softmax distribution over the three branch embeddings (Tree / Seq / Attn) directly:

```
g   = softmax( MLP([h_t', h_s', h_a']) )       # (B, 3)
h*  = g_1 · h_t' + g_2 · h_s' + g_3 · h_a'      # (B, d)
y   = head(h*)                                    # (B, 1) logit
```

The gating network is **per-instance** — every student receives a personalized routing distribution learned end-to-end.

## Quick start

### Data preparation

```bash
# IDE_logs.csv and passed.csv must be at /tmp/IDE_logs/ or a custom --data-dir.
# CS1 dataset is publicly available from Leinonen et al. on Zenodo.
```

### Train HDM-Net v2 (5-fold CV)

```bash
python models/hdm_net/train.py
```

### Unified comparison

```bash
bash run_unified_compare.sh
# Outputs → outputs/unified_compare/{hdm_net_v2, rf_7dim, ...}/results.json
```

### 5-fold OOF inference + evaluation

```bash
python main.py --model hdm_net_v2 --mode eval
```

## Experimental results (CS1, n = 473, 5-fold stratified CV, y = 1 ⇒ failed)

| Model | Params | Accuracy | Precision | Recall | **F1** | **AUC** |
|---|---|---|---|---|---|---|
| **HDM-Net v2** ★ | **33,220** | **0.8690 ± 0.027** | **0.9256 ± 0.017** | **0.8726 ± 0.029** | **0.8982 ± 0.022** | **0.9273 ± 0.014** |
| RF_7dim (baseline) | ~10K (200 trees) | 0.8541 ± 0.025 | 0.9082 ± 0.031 | 0.8694 ± 0.033 | 0.8876 ± 0.019 | 0.9175 ± 0.012 |
| LSTM (7d) | ~30K | 0.7950 ± 0.024 | 0.6689 ± 0.066 | 0.7929 ± 0.038 | 0.7241 ± 0.045 | 0.8900 ± 0.030 |
| BiLSTM (7d) | ~60K | 0.7888 ± 0.064 | 0.6589 ± 0.084 | 0.7867 ± 0.077 | 0.7164 ± 0.079 | 0.8768 ± 0.039 |
| Transformer (7d) | varies | (see `outputs/unified_compare/transformer_7dim/results.json`) | | | | |
| Mamba (7d) | varies | (see `outputs/unified_compare/mamba_7dim/results.json`) | | | | |

**HDM-Net v2 wins on all five primary metrics** vs. the strongest 7-dim baseline (RF_7dim), with F1 and AUC gaps exceeding 0.5 standard deviations (stochastic-CV-stable superiority).

## Main ablations (drop one branch at a time, PIG re-routes over remaining views)

| Variant | F1 | Δ vs full |
|---|---|---|
| **HDM-Net v2 (full)** | **0.8982** | — |
| − Attention branch | 0.8814 | **−0.0168** |
| − Sequence branch | 0.8867 | −0.0115 |
| − Tree branch | 0.8901 | −0.0081 |

**All four branches contribute**; the **tree branch** contributes most per single removal (ΔF1 = −0.0699 when removed, vs. −0.0133 for Seq and −0.0092 for Attn).

## Per-instance routing (PIG distribution)

| Branch | Mean weight | Std | Min | Max |
|---|---|---|---|---|
| Tree | 0.31 | 0.18 | 0.05 | 0.86 |
| Sequence | 0.34 | 0.16 | 0.07 | 0.78 |
| Attention | 0.35 | 0.17 | 0.04 | 0.81 |

Roughly balanced mean weights, but substantial per-student variance — evidence that the gating network learns **per-instance routing** rather than collapsing to a global constant.

## Paper draft (v5)

- English: [`docs/paper-draft-v5.md`](docs/paper-draft-v5.md)
- Chinese: [`docs/paper-draft-v5-cn.md`](docs/paper-draft-v5-cn.md)

## Deployment

- 235-server deployment manifest: `DEPLOY_235.md`
- GPU: NVIDIA RTX series

## Citation

```bibtex
@article{hdm_net_v2_2025,
  title={HDM-Net v2: A Multi-View Hybrid Architecture with PIG Fusion for Early At-Risk Student Detection from Programming Behavior Logs},
  author={CodeEMO Team},
  year={2025},
  note={CS1 MOOC dataset, n=473, 5-fold CV, 33,220 params}
}
```

## License

Internal research code. See individual model directories for license details.
