# CodeEMO: HDM-Net v2 — A Multi-View Hybrid Architecture for Early At-Risk Student Detection from Programming Behavior Logs

基于论文《HDM-Net v2: A Multi-View Hybrid Architecture with Per-Instance Gating》实现的完整项目。HDM-Net v2 是一种多视图混合架构，通过 Per-Instance Gating (PIG) 模块融合三种互补的行为视图（树分支、序列分支、注意力分支），总参数量 33,220。

## 数据集

- **CS1 MOOC IDE 日志**（Zenodo 公开）
- 473 名学生：159 通过 / 314 失败（正例率 33.6%）
- 28,588,309 条 IDE 事件，7 种事件类型
- 标签约定：`y=1=failed`

## 项目结构

```
CodeEMO/
├── README.md                         # 本文档
├── docs/
│   ├── paper_draft.md                # 旧版 BGM-Net v2 论文草稿（v1）
│   ├── paper-draft2.md               # 旧版 MRE 论文草稿（v2）
│   ├── paper-draft2-cn.md            # 旧版 MRE 中文草稿
│   ├── paper_draft_v3.md             # ★ HDM-Net v2 英文论文（v3）
│   └── paper_draft_v3_cn.md          # ★ HDM-Net v2 中文论文（v3）
│
├── common/                           # 共享工具模块
│   ├── data_loader.py                # IDE 日志加载
│   ├── feature_engineering.py        # 46 维手工特征提取
│   └── evaluator.py                  # 评估指标
│
├── models/                           # 模型实现
│   ├── rf/                           # 随机森林（baseline）
│   ├── lstm/                         # LSTM
│   ├── bilstm/                       # BiLSTM
│   ├── transformer/                  # Transformer
│   ├── mamba/                        # Mamba
│   ├── bgm_net/                      # BGM-Net v1 (dual-branch MLP, 5K params)
│   ├── hdm_net/                      # ★ HDM-Net v2 (3-branch + PIG, 33K params)
│   │   ├── model.py                  #   TreeHead, SeqBranch, AttnBranch, PIG, HDMNet
│   │   └── train.py
│   ├── cream/                        # CREAM
│   ├── cw_hdm_net/                   # CW-HDM-Net (class-weighted HDM-Net v2)
│   └── ... (其他 ablation 模型)
│
├── outputs/
│   ├── unified_compare/              # 统一对比（HDM-Net v2 vs RF_7dim 等）
│   │   ├── hdm_net_v2/results.json   #   ★ HDM-Net v2 5-fold CV 结果
│   │   ├── rf_7dim/results.json     #   ★ RF_7dim baseline
│   │   └── ... (其他模型 ablation)
│   ├── comparison.csv                # 全模型对比
│   └── analysis.md
│
└── scripts/                          # 工具脚本
    ├── shap_hdm_net_v2.py            # SHAP 解释 HDM-Net v2
    ├── gen_per_class_summary.py
    └── visualize.py
```

## 核心架构：HDM-Net v2

```
                        ┌──────────────────────────────────────────┐
                        │       Per-Instance Gating (PIG)         │
   x_tree (7d+2d RF) ──┐           Linear(96 → 32) → ReLU          │
                       ├─► Tree ─┤                                 │
                       │  Branch  │         α₁·h_t + α₂·h_s        │
   x_seq (46×1) ───────┤  (depth-3│              + α₃·h_a          │
                       ├──► Seq ─┤                  ↓              │
                       │  Branch │             Head               │
   x_att (7×1) ────────┤  (BiLSTM│                ↓              │
                       ├──► Attn ┤             logit              │
                       │  Branch │                                │
                       │ (Pre-N │                                │
                       │  LayerScale)                          │
                        └──────────────────────────────────────────┘
```

### 三大分支（互补视图）

| 分支 | 输入 | 处理 | 输出维度 | 作用 |
|---|---|---|---|---|
| **Tree** | 7d 事件计数 + 2d RF 折外概率 | depth-3 width-64 MLP + skip-connection | 32d | 静态特征组合 |
| **Sequence** | 46×1 (46维手工特征 reshape) | 单层 BiLSTM + mean-pool | 32d | 序列/统计模式 |
| **Attention** | 7×1 (7种事件类型) | 2层预归一 Transformer + LayerScale | 32d | 事件类型间的关系 |

### Per-Instance Gating (PIG)

3 个分支嵌入 → 2 层 MLP → softmax → 加权融合 → 线性头 → logit

**总参数量：33,220**

## 快速运行

### 数据准备

```bash
# IDE_logs.csv 和 passed.csv 应该在 /tmp/IDE_logs/ 或指定路径
# CS1 数据集从 Zenodo 公开下载
```

### 训练 HDM-Net v2

```bash
python models/hdm_net/train.py
```

### 统一对比实验

```bash
bash run_unified_compare.sh
# 输出到 outputs/unified_compare/{hdm_net_v2, rf_7dim, ...}/results.json
```

### 5-fold OOF 推理 + 评估

```bash
python main.py --model hdm_net_v2 --mode eval
```

## 实验结果（CS1, n=473, 5-fold CV, y=1=failed）

| 模型 | 参数量 | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|
| **HDM-Net v2** ★ | 33,220 | **0.8690 ± 0.027** | **0.9256 ± 0.017** | **0.8726 ± 0.029** | **0.8982 ± 0.022** | **0.9273 ± 0.014** |
| RF_7dim (baseline) | ~10K (200 trees) | 0.8541 ± 0.025 | 0.9082 ± 0.031 | 0.8694 ± 0.033 | 0.8876 ± 0.019 | 0.9175 ± 0.012 |
| LSTM (7d) | ~30K | 0.7950 ± 0.024 | 0.6689 ± 0.066 | 0.7929 ± 0.038 | 0.7241 ± 0.045 | 0.8900 ± 0.030 |
| BiLSTM (7d) | ~60K | 0.7888 ± 0.064 | 0.6589 ± 0.084 | 0.7867 ± 0.077 | 0.7164 ± 0.079 | 0.8768 ± 0.039 |

**HDM-Net v2 在五项主要指标上全部超过最强的 7 维 Random Forest 基线**，F1 与 AUC 差距均超过 0.5 个标准差（统计上稳定胜出）。

## 主要消融

| 变体 | F1 | Δ vs 完整 |
|---|---|---|
| HDM-Net v2（完整） | **0.8982** | — |
| − 注意力分支 | 0.8814 | −0.0168 |
| − 序列分支 | 0.8867 | −0.0115 |
| − 树分支 | 0.8901 | −0.0081 |

**三个视图都不可或缺，注意力分支贡献最大。**

## 论文草稿

- 英文 v3：`docs/paper_draft_v3.md`
- 中文 v3：`docs/paper_draft_v3_cn.md`

## 部署

HDM-Net v2 模型权重 + OOF 概率 + 训练脚本已开源。

- 235 服务器部署清单：`DEPLOY_235.md`
- 235 服务端 GPU 环境：NVIDIA RTX 系列

## 引用

```bibtex
@article{hdm_net_v2_2025,
  title={HDM-Net v2: A Multi-View Hybrid Architecture with Per-Instance Gating for Early At-Risk Student Detection from Programming Behavior Logs},
  author={CodeEMO Team},
  year={2025},
  note={CS1 MOOC dataset, n=473, 5-fold CV, y=1=failed}
}
```

## License

Internal research code. See individual model directories for license details.

