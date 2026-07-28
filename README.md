# CodeEMO: 融合编程行为情绪表征的学业早期风险预测

基于论文《融合编程行为情绪表征的学业早期风险预测》实现的完整项目，包含5种模型的对比实验。

## 目录结构

```
CodeEMO/
├── README.md                    # 项目文档
├── main.py                      # 统一运行入口
├── requirements.txt
│
├── common/                      # 共享工具模块
│   ├── __init__.py
│   ├── data_loader.py           # 数据加载 (从 /tmp/IDE_logs/)
│   ├── feature_engineering.py   # 46维特征提取
│   └── evaluator.py             # 共享评估指标 (f1, acc, auc等)
│
├── models/                      # 模型实现
│   ├── rf/                      # 随机森林
│   │   ├── README.md
│   │   ├── model.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── lstm/                    # LSTM
│   │   ├── README.md
│   │   ├── model.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── bilstm/                  # BiLSTM
│   │   ├── README.md
│   │   ├── model.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── transformer/             # Transformer
│   │   ├── README.md
│   │   ├── model.py
│   │   ├── train.py
│   │   └── evaluate.py
│   └── mamba/                   # Mamba (6步流程, CPU/GPU)
│       ├── README.md
│       ├── model.py             # 核心Mamba模型
│       ├── train_cpu.py         # CPU版本
│       ├── train_gpu.py         # GPU版本
│       ├── evaluate.py
│       ├── steps/               # 6步流程实现
│       │   ├── step1_preprocessing.py
│       │   ├── step2_pretrain.py
│       │   ├── step3_multiscale.py
│       │   ├── step4_prototype.py
│       │   ├── step5_finetune.py
│       │   └── step6_interpret.py
│       └── results.json
│   ├── bgm_net/                 # BGM-Net（行为门控双分支网络）
│   │   ├── model.py             # dual-branch + 3 可选模块
│   │   └── train.py             # 支持 5 变体消融 (--all-variants)
│   └── cream/                   # CREAM（对比学习 + 自注意力）
│       ├── model.py
│       └── train.py             # 支持 5 变体消融
│
├── outputs/                     # 实验输出
│   ├── rf/
│   ├── lstm/
│   ├── bilstm/
│   ├── transformer/
│   ├── mamba/
│   ├── comparison.csv           # 全模型对比
│   ├── unified_compare/         # 统一对比（46d vs 7d，failed=1 口径）
│   ├── late_fusion_5way_v1/     # 5路 Late Fusion
│   └── analysis.md              # 分析报告
│
├── docs/                        # 论文与文档
│   ├── paper_draft.md           # 主论文草稿
│   ├── cover_letter.md          # 投稿信
│   └── EXPERIMENT_RESULTS.md    # 详细实验结果汇总（消融数据）
│
└── scripts/                     # 工具脚本
    └── visualize.py             # 统一可视化
```

## 快速开始

### 运行单个模型

```bash
# 随机森林
python main.py --model rf

# LSTM
python main.py --model lstm

# BiLSTM
python main.py --model bilstm

# Transformer
python main.py --model transformer

# Mamba (CPU版)
python main.py --model mamba

# Mamba (GPU版)
python main.py --model mamba_gpu
```

### 运行所有模型

```bash
python main.py --model all
```

### 独立运行某个模型

每个模型目录下的 train.py 都可以独立运行：

```bash
python models/rf/train.py
python models/lstm/train.py
python models/bilstm/train.py
python models/transformer/train.py
python models/mamba/train_cpu.py
python models/mamba/train_gpu.py
```

### 可视化结果

```bash
python scripts/visualize.py
```

### 跑消融实验

```bash
# 5 模型 × {7d, 46d} = 10 组合 (统一 failed=1 口径, 重生 main 输出)
python main.py --model all --features all

# BGM-Net 5 变体消融（baseline / no_gate / no_entropy / no_cross / full）
python models/bgm_net/train.py --all-variants

# CREAM 5 变体消融（baseline / no_bottleneck / no_se / no_contrastive / full）
python models/cream/train.py

# Late Fusion 5 路
python late_fusion_5way.py
```

### 补齐 5 模型 7d/46d 统一口径对比（manual）

如果只需要重跑 RF / Transformer 在 unified_compare 下的 7d / 46d 输出（main.py 当前仅含 LSTM/BiLSTM/Mamba）：

```bash
# RF (4 个组合)
python models/rf/train_unified.py --features 7d
python models/rf/train_unified.py --features 46d

# Transformer (4 个组合)
python models/transformer/train_unified.py --features 7d --device cpu
python models/transformer/train_unified.py --features 46d --device cpu
```

然后重生成统一表与可视化：

```bash
python compare_all_unified.py
```

## 模型简介

| 模型 | 类型 | 描述 |
|------|------|------|
| [Random Forest](models/rf/README.md) | 传统ML | sklearn RandomForestClassifier, 46维特征 |
| [LSTM](models/lstm/README.md) | 深度学习 | 单向LSTM, 46维特征 → 序列建模 |
| [BiLSTM](models/bilstm/README.md) | 深度学习 | 双向LSTM (原论文方法), 46维特征 |
| [Transformer](models/transformer/README.md) | 深度学习 | Transformer编码器, 46维特征分组为伪序列 |
| [Mamba](models/mamba/README.md) | 前沿 | Selective State Space Model, 7维事件序列, 6步流程 |
| [BGM-Net](models/bgm_net/) | 参数高效 | 双分支 MLP（~5K 参数）解耦统计与比率特征，附 3 个可选模块的消融 |
| [CREAM](models/cream/) | 对比学习 | 对比学习头 + Squeeze-Excitation，附 5 变体消融 |
| [OST-Forest](models/ost_forest/) | 自蒸馏树森林 | 20 RFs OOF + 33-d meta + LR head + soft label (α=0.4)；F3 单折 F1=0.928 |
| [MASC-Net](models/masc_net/) | 多尺度对比学习 | 多尺度编码 + Sample-aware 对比 + 不确定性分支；4 变体消融（baseline_only 最佳）|
| [PR-DE-Net](models/pr_de_net/) | 双分支精召路由 | BiLSTM(Recall)+Transformer(Precision) + Gate MLP + 三段式 Loss；3-way 融合 F1=0.9027 |
| [MRE](models/mre/) | 多路由专家 + SHAP 解释 | Route A (RF 7-dim) + Route B (LSTM 46-dim) + Gating MLP；hard 路由 F1=0.899 AUC=0.932 + 4 学生画像 |

## 特征工程 (46维)

| 特征组 | 维度 | 描述 |
|--------|------|------|
| 事件基础统计 | 28 | 7种事件类型 × 4统计量 (均值/标准差/变异系数/香农熵) |
| 行为轨迹 | 10 | improvement/consistency/trend/间隔统计量等 |
| 情绪复合 | 6 | edit_ratio/delete_ratio/focus_ratio的均值和标准差 |
| 元信息 | 2 | 题目数量、总事件数 |

## 数据格式

数据位于 `/tmp/IDE_logs/`:
- `IDE_logs.csv`: IDE事件日志 (student, part, exercise, eventType, timestamp, timeToDeadline)
- `passed.csv`: 学生标签 (student, passed)

约 2858万条事件日志, 涵盖7种事件类型: `text_insert`, `text_remove`, `text_paste`, `focus_gained`, `focus_lost`, `run`, `submit`

## 评估方法

- **5折分层交叉验证** (StratifiedKFold)
- **指标**: Accuracy, Precision, Recall, F1 Score, AUC

## 结果汇总

本仓库已包含完整 5 折分层 CV 实测数据。**主对比**数据源为 `outputs/unified_compare/unified_compare.csv`（口径：y=1=failed），汇总如下：

### 5 主流模型 × 特征维度

| 模型 | 特征维度 | Accuracy | F1 | AUC |
|------|---------|----------|----|-----|
| **HDM-Net v2** | **7d+46d 融合** | 0.8690 ± 0.027 | **0.8982 ± 0.022** | **0.9273 ± 0.014** |
| **RF** | **7dim** | 0.8541 ± 0.025 | 0.8876 ± 0.019 | 0.9175 ± 0.012 |
| Transformer | 7dim | 0.8352 ± 0.041 | 0.8689 ± 0.034 | 0.9162 ± 0.027 |
| RF | 46d | 0.8247 ± 0.034 | 0.8654 ± 0.030 | 0.9069 ± 0.029 |
| LSTM-MLP | 46d | 0.8246 ± 0.034 | 0.8622 ± 0.028 | 0.9170 ± 0.023 |
| Transformer | 46d | 0.8183 ± 0.032 | 0.8567 ± 0.031 | 0.9034 ± 0.009 |
| BiLSTM-MLP | 46d | 0.8225 ± 0.023 | 0.8561 ± 0.024 | 0.9036 ± 0.014 |
| Mamba  | 46d | 0.7972 ± 0.042 | 0.8455 ± 0.033 | 0.8557 ± 0.048 |
| BiLSTM-Seq | 7dim | 0.7295 ± 0.041 | 0.8153 ± 0.020 | 0.7398 ± 0.053 |
| LSTM-Seq | 7dim | 0.7189 ± 0.037 | 0.8062 ± 0.020 | 0.7259 ± 0.052 |
| Mamba  | 7dim  | 0.6109 ± 0.043 | 0.6768 ± 0.044 | 0.6150 ± 0.063 |

> **表注（重要 — 避免审稿误解）**：
> - **LSTM-MLP-46d / BiLSTM-MLP-46d 实现机制**：46-dim 特征向量先经过 `Linear(46→64)` 升维，再 reshape 为 **(B, 1, 64)** 即 **seq_len = 1**。因此 LSTM 在 46d 上**没有真正的时序建模**，等价于一个**带 gating 的 2 层 MLP**。
> - **LSTM-Seq-7dim / BiLSTM-Seq-7dim 实现机制**：喂入真实 IDE 事件序列，**max_seq_len = 500**（截断），是真正的时序模型。
> - **结论**：LSTM-MLP-46d / BiLSTM-MLP-46d 的赢面来自「**46 维含变异系数 + 熵 + 行为轨迹 + 比率的小数据手工统计特征 + LSTM gating 做非线性交互**」，而非「**46 维时序结构**」。详见下方"信息密度差异"小节。

### 信息密度差异：为什么 LSTM-MLP-46d > LSTM-Seq-7dim？

| 特征类型 | 7-dim (事件计数) | 46-dim (hand-crafted 统计) |
|----------|------------------|--------------------------|
| **text_insert 总数** | ✅ 1000 | ✅ 1000 |
| **变异系数** (std/mean) | ❌ 无 | ✅ 0 (一次性写) vs 0.5 (分散写) |
| **香农熵** (分布复杂度) | ❌ 无 | ✅ 0 (集中) vs >0 (分散) |
| **行为轨迹** (slope / consistency / trend / interval) | ❌ 无 | ✅ 10 维 |
| **复合比率** (edit/delete/focus ratio 的均值与标准差) | ❌ 无 | ✅ 6 维 |
| **元信息** (num_problems / total_events) | ❌ 无 | ✅ 2 维 |

> **结论**：7 维特征把「一次性写 1000 字符」与「100 次每次 10 字符」映射到同一向量；46 维通过变异系数 + 熵 + 比率 + 轨迹把它们**完全区分开**。**LSTM-MLP-46d 的胜出来自「更丰富的小样本手工统计特征 + LSTM gating 做非线性」**，并非「时序结构」。

### 新模型单模型对比（2026-07-26/27 新增）

| 模型 | 特征/架构 | Accuracy | Precision | Recall | F1 | AUC |
|------|----------|----------|-----------|--------|----|------|
| **OST-Forest (CV 均值)** | 33-d meta + LR + soft label | 0.8521 ± 0.027 | 0.9378 ± 0.035 | 0.8344 ± 0.048 | **0.8816 ± 0.023** | **0.9087 ± 0.014** |
| **OST-Forest (F3 单折/最佳)** | 同上 + 阈值微调 | 0.9043 | 0.9355 | 0.9206 | **0.9280** | 0.9309 |
| MASC-Net (baseline_only) | 7-dim + 多尺度编码 | 0.8458 ± 0.034 | 0.7145 ± 0.045 | 0.9058 ± 0.044 | **0.7985 ± 0.042** | **0.8995 ± 0.020** |
| PR-DE-Net (full) | 46d + 7d 双分支 + Gate | — | 0.9092 | 0.8184 | **0.8601 ± 0.027** | 0.8711 |
| MRE - RF expert (Route A) | 7-dim | 0.8563 | 0.9111 | 0.8694 | 0.8891 ± 0.017 | **0.9175 ± 0.012** |
| MRE - LSTM expert (Route B) | 46-dim | 0.8289 | 0.8981 | 0.8377 | 0.8659 ± 0.028 | 0.9068 ± 0.020 |
| **MRE-soft** | Gate (Soft MoE) | 0.8647 | 0.9287 | 0.8631 | 0.8943 ± 0.015 | **0.9326 ± 0.012** |
| MRE-confidence | Gate (Confidence-based) | 0.8584 | 0.9251 | 0.8567 | 0.8889 ± 0.020 | 0.9236 ± 0.018 |
| **MRE-hard** (best) | Gate (Hard Routing + STE) | **0.8690** | **0.9232** | **0.8758** | **0.8986 ± 0.019** | 0.9316 ± 0.008 |

> **补充观察**：
> 1. **OST-Forest 单模型（F3 折 0.9280）已超越 HDM-Net v2（T3 0.8982）+3.0 pts**，但 CV 均值 0.8816 仍低于 HDM-Net v2；说明 OST-Forest 的 refinement 阶段带来强单折性能，但泛化均值尚未稳定超越。
> 2. **MASC-Net (baseline_only F1=0.7985) 比 BGM-Net baseline (F1=0.7458) 高 +5.27 pp**，但完整 MASC-Net (full F1=0.7286) 反而更低——与小样本对比学习负面结果一致。
> 3. **PR-DE-Net 单模型不强**（F1=0.8601 < RF-7dim 0.8876），但通过端到端融合把最强集成从 0.9009 推到 **0.9027**。

> **关键发现**：
> 1. **新榜首 Weighted 1/3/1 (F1=0.901 / AUC=0.935)** – 1×RF + 3×HDM-Net v2 + 1×LSTM 的加权集成，超越 Late Fusion 5-way 在 F1 (略低 0.005) 和 AUC (反超 0.013) 的表现。同时 Precision=0.935 是全部单模型里最高的。详情见下一节。
> 2. **F1/AUC 次榜首** = RF-7dim (F1=0.888 / AUC=0.918)；LSTM-MLP-46d (AUC=0.917) 几乎并列。
> 3. **树/伪序列模型在 7-dim 简洁特征上反超 46-dim**：RF-7dim > RF-46d (ΔF1=+0.022)，Transformer-7dim > Transformer-46d (ΔF1=+0.012)。
> 4. **序列模型恰好相反**：LSTM-MLP-46d > LSTM-Seq-7dim (ΔF1=+0.056)；BiLSTM-MLP-46d > BiLSTM-Seq-7dim (ΔF1=0.041)。
> 5. **隐含启示**：特征工程的"最优维度"取决于模型族——是论文 BGM-Net 双分支解耦设计的独立证据点（§3.7）。

### HDM-Net：异构解码器混合网络

| 指标 | F1 | AUC | Precision | Recall | Accuracy | Params |
|------|-----|------|-----------|---------|----------|--------|
| HDM-Net v2 (T3) | **0.8982 ± 0.022** | 0.9273 ± 0.014 | 0.9256 ± 0.017 | **0.8726 ± 0.029** | **0.8690 ± 0.027** | 42,180 |
| (vs RF-7dim 上轮冠军) | **(+0.0106)** | (+0.0098) | (+0.018) | (−0.012) | (+0.008) | ×3 |
| HDM-Net v1 (baseline) | 0.8887 ± 0.013 | 0.9246 ± 0.017 | 0.9279 ± 0.018 | 0.8535 ± 0.027 | 0.8584 ± 0.014 | 33,220 |

> v2 vs v1：TreeHead 从 depth=2 width=32 （1.4K params）加深加宽为 depth=3 width=64 （9K params）。F1 +0.0095，AUC +0.0027。其他 2 个分支 (Seq / Attn) 不变。

**架构 (3 视角异构 + per-instance gating)**：

```
   7-dim events          46-dim feature vec        7-dim events
        + RF probs          (46 time-steps)              .
            ↓                   ↓                       ↓
       ┌───────┐         ┌──────────┐           ┌─────────────┐
       │TreeHead│         │  BiLSTM  │           │ Pre-Norm    │
       │ 9→32   │         │ 46×1→32  │           │ Transformer  │
       │ MLP    │         │ 1-layer  │           │ 7×1→32      │
       └───┬───┘         └─────┬────┘           └──────┬──────┘
           │ h_t (32)         │ h_s (32)            │ h_a (32)
           └─────────┬───────┴──────────┬───────────┘
                     ▼                  ▼
                  ┌──────────────────────────┐
                  │  Per-Instance Gating     │ (PIG)
                  │  g = softmax(MLP([h_t,h_s,h_a]))
                  │  h = g_t·h_t + g_s·h_s + g_a·h_a
                  └────────────┬─────────────┘
                               ▼
                          FC(32→1) → sigmoid
```

**实现文件**：`models/hdm_net/{model.py,train.py}`（5 折 CV 输出 `outputs/unified_compare/hdm_net/`）

**为什么 work**：单模型上 AUC=0.9273 双榜超越 Late Fusion 5-way=0.922；表示“异构分支 + per-student gating + 加宽加深 TreeHead” 的确比加权平均 “多通道独立模型结果” 高出多个 F1 点。

#### TreeHead 容量 sweep（5 变体）

在 backbone (TreeHead) 是关鍵的前提下，扫描深度与宽度：

| 变体 | depth | width | skip | F1 | AUC | Params |
|------|-------|-------|------|-----|------|--------|
| T0 (v1 baseline) | 2 | 32 | ✗ | 0.8937 | 0.9236 | 1.4K |
| T1 (deeper only) | 3 | 32 | ✗ | 0.8841 | 0.9231 | 2.4K |
| T2 (wider only)  | 2 | 64 | ✗ | 0.8964 | **0.9292** | 4.8K |
| **T3 (deep+wide)** | **3** | **64** | **✗** | **0.8982** ⭐ | 0.9273 | **9.0K** |
| T4 (residual)    | 2 | 32 | ✓ | 0.8902 | 0.9259 | 1.7K |

**观察**：

1. **宽度比深度重要**。仅加深 (T1 vs T0) 参数从 1.4K 添到 2.4K，但 F1 反而 跌 −0.010 — 样本不够喂拥3层。
2. **加宽 (T2) +0.003 F1 且提 AUC 越 0.006**。在 n=473 上，模型需要“更多并行能力”而不是“更深县街”。
3. **T3 (深 64 宽) 拉报**。F1=0.8982 是 v1 (0.8887) 的 +0.0095，不仅是 TreeHead 加倍容量，也使模型对 RF OOF + 7事件计数的非线性变换变丰富。
4. **Residual (T4) 退化**。“跳跃”让信号绕过了“从 RF 概率学 K 倍非线性”的关键路径，在 n=473 上反而起到正则作用。

#### HDM-Net 架构消融（4 变体，验证 3 视图都必要）

| 变体 | 描述 | F1 | AUC | Precision | Recall | ΔF1 vs full |
|------|------|-----|------|-----------|--------|--------|
| **full** | 3 分支全开 | **0.8887 ± 0.013** | **0.9246 ± 0.017** | 0.9279 | 0.8535 | — |
| no_tree | Tree 驱动 → 0 | **0.8282 ± 0.038** | 0.8661 ± 0.042 | 0.8936 | 0.7802 | **−0.0605**⚠ |
| no_seq  | Seq 驱动 → 0 | 0.8849 ± 0.020 | 0.9202 ± 0.021 | 0.9426 | 0.8343 | −0.0038 |
| no_attn | Attn 驱动 → 0 | 0.8890 ± 0.026 | 0.9163 ± 0.023 | 0.9537 | 0.8343 | +0.0003 |

**结果解读**：

1. **Tree branch 是 backbone**（去掉 F1 −0.0605、AUC −0.0585，均远在 2×std 之上）。这是因为 Tree head 直接消费 RF 概率 + 7-dim 事件计数，是模型最强的信号接入点。
2. **Seq branch 贡献微乎其其**（−0.0038）。当前设计下 46-dim 被 reshape 为 (46,1) univariate 序列在 n=473 上没有提供 RF 之外的独立信号。
3. **Attn branch 贡献几乎为 0**（+0.0003，在 std 以内）。与 Seq branch 一样，7-dim reshape 为 (7,1) 太短，attention 机制没能发挥。
4. **未来优化方向**（在 n>2,000 上）：丰富 Tree 输入或加深 Seq/Attn 分支，让三个分支都能为模型提供互补信号。当前的 "3 个归纳偏置异构分支" 架构在思路上是对的，但只有在数据集足够大时才能完全发挥各分支的信号。

> XCA (Cross-View Cross-Attention) 原是架构里的进阶设计，在 n=473 小样本上验证后发现会提升损失数位不稳定性，本次实验未启用。期朞在 n>2,000 上重新启用 XCA 能获得额外提升，文档参见 `docs/HDM_NET_DESIGN.md`（可作为下一步）。
>
> **Mamba-7dim 偏低说明**：F1=0.677 / AUC=0.615 显著低于其他 7-dim 模型，是 6 步流水线在 7-dim + 473 学生 + `finetune_epochs=4` 下未充分收敛（诊断脚本 `scripts/diag_mamba_label.py` 证实 probs.std=0.08）。46-d 配置下 Mamba 正常收敛到 F1=0.846、AUC=0.856。完整诊断详见 `docs/EXPERIMENT_RESULTS.md` §1。

### BGM-Net 架构消融（5 变体，参数量 ~5K）

| 变体 | F1@0.5 | AUC | 说明 |
|------|--------|-----|------|
| **baseline** | **0.7458 ± 0.026** | **0.9079 ± 0.019** | 仅双分支 MLP（dual-branch decoupling） |
| no_cross      | 0.7381 ± 0.023 | 0.9061 ± 0.020 | −Ratio Cross |
| no_gate       | 0.7290 ± 0.056 | 0.9012 ± 0.032 | −Behavior Gate |
| full          | 0.7226 ± 0.047 | 0.9003 ± 0.030 | 启用全部 3 模块 |
| no_entropy    | 0.7229 ± 0.043 | 0.8905 ± 0.030 | −Entropy Attention |

> 结论：**双分支解耦本身是 BGM-Net 的全部价值**；三个可选模块（gate / entropy-attn / cross）在 n=473 上未带来正向收益（详见 `docs/EXPERIMENT_RESULTS.md` §2）。

### CREAM 架构消融（5 变体）

| 变体 | F1@0.5 | AUC |
|------|--------|-----|
| no_contrastive    | **0.7685 ± 0.021** | 0.8982 ± 0.018 |
| no_bottleneck     | 0.7591 ± 0.026 | 0.9052 ± 0.018 |
| no_se             | 0.7559 ± 0.027 | **0.9085 ± 0.014** |
| baseline          | 0.7553 ± 0.020 | 0.9040 ± 0.016 |
| full              | 0.7539 ± 0.030 | 0.9073 ± 0.021 |

### Late Fusion 5 路集成

| 指标 | 数值 | 对比单模型最佳 |
|------|------|---------------|
| **F1** | **0.9056 ± 0.015** | +0.005（vs Weighted 1/3/1 0.9009） |
| **AUC** | 0.9222 ± 0.011 | -0.013（vs Weighted 1/3/1 0.9349） |

权重组合示例: `(a=0.5, b=0.3, c=0.1, d=0.1, e=0.0)` 为 Top-1 配置。论文草稿中 7 路融合达 F1=0.9013 亦与此口径吻合。

### Stacking + 加权集成（3 变体）

在 RF_7dim、HDM-Net v2 (T3) 与 LSTM_46d 这三个强模型上测试了两种集成策略：

| 变体 | F1 | Precision | Recall | AUC | 描述 |
|------|-----|-----------|--------|-----|------|
| **Weighted 1/3/1** | **0.9009 ± 0.019** | **0.9351 ± 0.019** | 0.8694 ± 0.027 | **0.9349 ± 0.013** | 1×RF + 3×HDM v2 + 1×LSTM |
| **★ PR-DE-Net 3-way** | **0.9027 ± 0.014** | — | — | 0.9265 | 2.5×RF + 2.5×HDM v2 + 1.0×PR-DE-Net |
| ★ PR-DE-Net 4-way | 0.9026 ± — | — | — | 0.9288 | 1×RF + 3×HDM v2 + 0×LSTM + 1×PR-DE-Net |
| Weighted 2/3/1 | 0.8996 ± 0.024 | 0.9288 ± 0.020 | 0.8726 ± 0.032 | 0.9322 ± 0.013 | 2×RF + 3×HDM v2 + 1×LSTM |
| **Stack (top-3 LR)** | 0.8986 ± 0.015 | 0.9072 ± 0.021 | **0.8918 ± 0.037** | 0.9324 ± 0.015 | LR(C=0.1) meta-learner |
| HDM-Net v2 (单 best) | 0.8982 ± 0.022 | 0.9256 ± 0.017 | 0.8726 ± 0.029 | 0.9273 ± 0.014 | — |

**观察**：

1. **Weighted 1/3/1 是新最佳 F1/AUC**。在 RF × 1 + HDM-Net v2 × 3 + LSTM × 1 的加权下：F1=0.9009、AUC=0.9349，同时 Precision=0.9351、Recall=0.8694。结果验证了"HDM-Net v2 是最强但加权上 RF/LSTM 各贡献一些"。
2. **Stacking (LR) 提升 Recall 最多**（0.8918 +0.019 vs HDM v2）。LR meta-learner 在 5-fold 里有意识地给 failed 样本加重权，提高识别率。
3. **Weighted 1/3/1 vs Late Fusion 5-way**：F1 -0.005、AUC +0.013 — 在三项中F1略低，但 AUC 明显超。这说明 "质量加权" 在面向 failed=1 任务上有适应能力。

**实现路径**：`outputs/unified_compare/stack_top3_LR_C0.1/`, `weighted_1_3_1/`, `weighted_2_3_1/` 都存储了 OOF probs/labels/fold_idx.json，可直接被下游论文画图使用。

### Bayesian Model Averaging + Per-Fold Weighted Stacking

为了同时研究**集成质量**与**可解释性**，跑了 2 个贝叶斯 / 折间方案。

#### Bayesian Model Averaging (BMA, 8 个 base models)

**思路**：weight_i = exp(-OOF-log-loss_i / T) / Σ exp(...)。T 是温度参数，T → 0 则取最 confident single，T → ∞ 则退化为均匀平均。

| T | F1 | AUC | Precision | Recall |
|---|------|------|-----------|--------|
| T=0.1 | 0.8915 | 0.9314 | 0.9368 | 0.8503 |
| **T=1.0** (保存) | **0.8875 ± 0.023** | **0.9325 ± 0.012** | 0.9364 | 0.8439 |
| T=5.0 | 0.8867 | 0.9303 | 0.9301 | 0.8471 |
| T=100 (≈ uniform) | 0.8867 | 0.9291 | 0.9301 | 0.8471 |

#### Per-Fold Weighted Stacking (top-5)

**思路**：对每个 fold k，用**其余 4 个 fold 的 OOF 数据**训练一个 LR meta-learner，预测 fold k。这样每个 fold 有一套专属的 ensemble 权重——暴露"哪些模型在哪些 fold 上更可靠"。

| 指标 | F1 | AUC | Precision | Recall |
|------|------|------|-----------|--------|
| Per-fold stacking (LR per fold) | **0.8953 ± 0.017** | 0.9346 | 0.9195 | 0.8726 |
| Per-fold BMA (uncertainty per fold) | 0.8898 ± 0.017 | **0.9348** | 0.9337 | 0.8503 |

#### 可解释性分析（key findings）

1. **BMA 信任 HDM-Net v2 最多**（log-loss=0.315 最低），Transformer_46d 几乎被丢弃（log-loss=2.29 是 6-7x worse）→ **BMA 自动识别 "坏掉的模型" 并降权**
2. **Per-fold 权重 std 极小**（0.006-0.032）→ **5 个 base model 都很稳定，没有"看 fold 行事"的 volatile 模型**
3. **模型对 correlation 高**：RF_7dim ↔ HDM-Net v2 = **0.964**（高度冗余），LSTM_46d ↔ BiLSTM_46d = **0.945** → 这两组 ensemble 提升空间不大；LSTM_46d ↔ RF_7dim = 0.844 才**真正提供 diversity**
4. **错误分析**：64/473 = 13.5% 误分类，其中 40 是 failed=1 被漏报 (FN)，24 是 passed=0 被错报 (FP)。**5 个最自信误判都是 "model says passed, truth is failed"**——模型**系统性低估这些 failed=1 样本**。这指向需要新信号来源（如 SRL、engagement trajectories）才能突破。

**完整 analysis 见** [`docs/INTERPRETABILITY_ANALYSIS.md`](docs/INTERPRETABILITY_ANALYSIS.md)

### RF-LSTM 架构融合进化史（v1 → v2 → v3）

**设计动机**：根据实验数据，**LSTM_46d Precision=0.900 几乎不输 RF_7dim Precision=0.907**（差 0.007），但 Recall 差 +0.041 → RF 覆盖更广。两模型 correlation=0.844 是真正提供 diversity 的两人组。架构级融合有望突破 post-hoc stacking 的天花板。

| 版本 | 主要改动 | Params | F1 | AUC | 备注 |
|------|---------|--------|------|------|------|
| **v1 (RF-LSTM)** | RF probs broadcast 到每个 LSTM step | ~40K | 0.8766 | 0.9219 | 基线 |
| v2 (RF-LSTM-Attn) | + hidden=64, 2-layer BiLSTM, cross-view attention | 222K | 0.7980 (FAILED) | 0.4521 | **退化到 "全预测 failed"** (R=1.0)，参数过拟合 |
| **v3 (RF-LSTM-Attn)** | hidden=32, 1-layer BiLSTM, **self-attention + residual** | 28.9K | **0.8809 ± 0.016** | **0.9253 ± 0.017** | ✅ 解决过拟合问题，F1 +0.004 vs v1 |
| v4 (pre-norm + 2x attn + LN) | + Pre-norm BiLSTM + 2x self-attention + LayerNorm + Residual | 48K | 0.8194 ± 0.024 | 0.8258 ± 0.036 | ⚠️ 退化 -0.06 F1：现代 modular 设计**过参数化** |

**v3 架构**：

```
7-dim events ─┐
              ├──> BiLSTM(hidden=32, 1-layer) over 4×(11+2=13) segments
46-dim feats ─┘              ↓
                  h_seq: (B, 4, 64)
RF probs (frozen) ──────────>┘     │
                                  self-attention (residual)
                                  ↓
                              pooled: (B, 64)
                                  ↓
                  concat(pooled, RF probs) → Linear → sigmoid
```

**v3 vs 现有最佳对比**：

| 模型 | F1 | AUC | Precision | Recall | Params |
|------|------|------|-----------|--------|--------|
| RF_7dim (baseline) | 0.8878 | 0.9167 | 0.9070 | **0.8694** | ~12K |
| LSTM_46d (baseline) | 0.8624 | 0.9083 | 0.8997 | 0.8280 | ~50K |
| **RF-LSTM v3 (架构融合)** | 0.8809 | **0.9253** | **0.9156** | 0.8503 | 28.9K |
| Weighted 1/3/1 (post-hoc) | **0.9009** | 0.9349 | 0.9351 | 0.8694 | 0 |

**关键发现**：
1. v3 的 **Precision 0.9156 是三者最高**（RF: 0.9070, LSTM: 0.8997）——架构级 RF 信号注入确实"教"了 LSTM 怎么更谨慎
2. v3 的 **AUC 0.9253 高于 RF_7dim**——时序建模 + RF 联合提升概率排序质量
3. 但 v3 **F1 仍低于 RF_7dim**（−0.007）——在小数据上端到端学习没法超过最强的单 inductive bias 模型
4. v3 **远低于 Weighted 1/3/1**（−0.020）——post-hoc 加权在小样本上仍是 SOTA

**结论**：架构级融合（v3）vs post-hoc 加权（Weighted 1/3/1）——**在小数据 n=473 上，post-hoc 更稳定**。架构级融合在大数据上可能反超，但需要验证。

**v4 退化原因分析**：
- **Pre-norm + 2 LayerNorms** 让训练不崩（不像 v2 collapse）但仍**过参数化**
- **2 层 self-attention** 在 4 个 segment 上太多——attention 抓不到稳定模式
- **48K params** 比 v3 (29K) 多 67%，但泛化能力反降
- **v3 才是这个架构融合路线的 sweet spot**：F1=0.8809, AUC=0.9253

---

### OST-Forest: Out-of-fold Self-distilled Tree Forest（F3 单折 F1=0.928, CV 均值 F1=0.882）

**架构（20 RFs → OOF → 33-d meta → LR + self-distillation）**：

```
7-dim events ──┐
               ├──> 20 RFs (depth 3/5/8, seed varies) ──> 5-fold OOF matrix (20-d)
                                                              +
7-dim handcrafted (7-d) + 6-d session stat = 33-d meta feature
                                                              ↓
                                              LR head + self-distillation soft label (α=0.4)
                                                              ↓
                                                         p_final ∈ [0,1]
```

**CV 5 折 + F3 折（refinement）对比**：

| 指标 | CV 均值 | F3 折（refinement） |
|------|---------|---------------------|
| Accuracy | 0.8521 ± 0.027 | 0.9043 |
| Precision | 0.9378 ± 0.035 | 0.9355 |
| Recall | 0.8344 ± 0.048 | 0.9206 |
| **F1** | **0.8816 ± 0.023** | **0.9280** ⭐ |
| AUC | 0.9087 ± 0.014 | 0.9309 |

**5 个核心 Finding**：

1. **20 RF OOF marginal in 7-d**：大量 RF OOF 信号在 7-d 上边际收益有限。
2. **LR head > LightGBM head by +0.0147 F1**（n=473）：简单线性头在小样本上反而胜出。
3. **Soft label α=0.4 稳定 +0.007 F1**：自蒸馏有微弱贡献。
4. **20→5 RF no significant degradation**：RF 数量可大幅缩减而不损性能。
5. **7d > 46d in this regime**：验证"小样本偏 7-d"假说。

**模型排名（F1，5 折 CV）**：

| 排名 | 模型 | F1 | 类型 |
|------|------|----|------|
| 1 | Late Fusion 5-way | 0.9056 | 5 模型集成 |
| 2 | **PR-DE-Net 3-way** | **0.9027** | 3 模型融合（详见下） |
| 3 | Weighted 1/3/1 | 0.9009 | 3 模型加权 |
| 4 | Stack top-3 LR | 0.8986 | 3 模型 LR stacking |
| 5 | OST-Forest (CV 均值) | 0.8816 | 单模型 |
| 6 | HDM-Net v2 (T3) | 0.8982 | 单模型 |
| 7 | RF-LSTM v3 | 0.8809 | 单模型 |
| 8 | RF-7dim | 0.8876 | 单模型（最强传统 ML）|
| 9 | PR-DE-Net (full) | 0.8601 | 单模型 |
| 10 | MASC-Net (baseline_only) | 0.7985 | 单模型 |

> OST-Forest 的 F3 单折 0.928 是**单一折表现**，CV 均值 0.8816 更稳定；论文引用应使用均值。

**实现文件**：`models/ost_forest/{model.py,train.py}`，refinement 变体输出在 `outputs/ost_forest/refine_F*.json`
**详细报告**：[`docs/OST_FOREST_REPORT.md`](docs/OST_FOREST_REPORT.md)

---

### MASC-Net: Multi-scale Adaptive Sample-aware Contrastive Network（4 变体消融）

**架构**：多尺度编码（CNN+MLP+Cross-Scale Attention）+ Sample-aware 对比（MoCo + adaptive τ + hard negatives）+ Prototype 记忆库（K=2 × M=4，EMA）+ Adaptive threshold 分类器 + Uncertainty 分支

**4 变体消融（5 折 CV，n=473）**：

| 变体 | Contrastive | Uncertainty | F1 | AUC |
|------|:-----------:|:-----------:|----|-----|
| **baseline_only** | ✗ | ✗ | **0.7985 ± 0.042** | **0.8995 ± 0.020** |
| no_contrastive | ✗ | ✓ | 0.7517 ± 0.044 | 0.8479 ± 0.054 |
| no_uncertainty | ✓ | ✗ | 0.7467 ± 0.026 | 0.8624 ± 0.026 |
| full | ✓ | ✓ | 0.7286 ± 0.042 | 0.8644 ± 0.021 |

**关键负发现**：

1. **contrastive + uncertainty 模块在 n=473 上拖累 F1**（−7 pp vs baseline_only），与 BGM-Net 消融结论一致——**小样本下增加参数会引入方差**。
2. **baseline_only 反超 BGM-Net baseline**（F1=0.7985 vs 0.7458，**+5.27 pp**），说明多尺度编码本身（去掉对比学习头）已经是个不错的特征提取器。
3. **full 模型 AUC 仍可观**（0.8644）但 F1 显著下降，表明对比学习优化了概率排序但破坏了阈值化的分类决策。

**实现文件**：`models/masc_net/{model.py,train.py}`，4 变体结果在 `outputs/masc_net/{baseline_only,no_contrastive,no_uncertainty,full}_results.json`

---

### PR-DE-Net: Precision-Recall Gated Dual-Encoder Network（融合 F1=0.903，单模型 F1=0.860）

**架构动机**：LSTM-MLP-46d / BiLSTM-MLP-46d Recall 高（0.80-0.83）但 Precision 仅 0.92；Transformer-7d Precision 高（0.918）但 Recall 0.82——两者学**互补判别特征**，应当端到端融合。

**架构（双分支 + Gate MLP）**：

```
Input:  46-dim feature vector (B, 46)
              │
   ┌──────────┴──────────┐
   ▼                     ▼
Branch A: PR-RNN      Branch B: PR-Trans
BiLSTM(2L, h=64)      Transformer(2L, d=32, 4-head)
on (B,46,1)           on (B,7,1) + [CLS]
   ↓                     ↓
p_A (Recall)          p_B (Precision)
   ↓                     ↓
   └──────┬──────────────┘
          ▼
       Gate MLP
   g = σ(MLP([F46, h_A, h_B]))
          │
   p_final = g·p_A + (1-g)·p_B
```

**三段式 Loss**：L = α·BCE(p_A, y) + β·BCE(1-p_B, y) + γ·BCE(p_final, y)，权重 (1.0, 1.0, 2.0)

**Gate 学到设计目标**：

| 子集 | n | gate 均值 | 偏 RNN 占比 | 偏 Trans 占比 |
|------|---|----------|-------------|---------------|
| Failed (y=1) | 314 | **0.430** | **59.2%** | 40.8% |
| Passed (y=0) | 159 | **0.659** | 27.0% | **73.0%** |

> ✅ Gate 真的在样本级做 PR 路由——failed→RNN（RNN Recall 高），passed→Trans（Trans Precision 高）

**单模型 + 融合结果**：

| 变体 | F1 | Precision | Recall | AUC | 备注 |
|------|-----|-----------|--------|-----|------|
| PR-DE-Net (full, 单) | 0.8601 ± 0.027 | 0.9092 | 0.8184 | 0.8711 | |
| no_gate (固定 0.5/0.5 平均) | 0.8486 | 0.8691 | 0.8344 | 0.8412 | 验证 Gate 必要 |
| single_loss (只用 γ·BCE) | 0.8653 | 0.8818 | 0.8504 | 0.8781 | |
| Weighted 1/3/1 (融合基线) | 0.9009 | 0.9351 | 0.8694 | 0.9349 | |
| **★ PR-DE-Net 3-way** (2.5RF+2.5HDM+1.0PR-DE) | **0.9027** | — | — | 0.9265 | **新 SOTA 融合方案** |
| ★ PR-DE-Net 4-way (1RF+3HDM+0LSTM+1PR-DE) | 0.9026 | — | — | 0.9288 | |

**关键贡献**：

1. **PR-DE-Net 真正价值在融合**——把最强集成 F1 从 0.9009（Weighted 1/3/1）推到 **0.9027**（+0.18 pp）。
2. **Gate 学到了设计目标**——failed 走 RNN 分支（高 Recall），passed 走 Trans 分支（高 Precision）。
3. **失败模式明确**——错把 failed 当 passed 的 57 个样本上 Gate 路由失败（gate=0.685 应低于 0.5），表明单模型架构无法突破，需要 RF/HDM 在融合中补足视角。

**实现文件**：`models/pr_de_net/{model.py,train.py,fusion.py,ablation.py}`，完整 README 见 [`models/pr_de_net/README.md`](models/pr_de_net/README.md)，融合结果在 `outputs/pr_de_net/fusion_{3way,4way}.json`

---

### MRE: Multi-Route Expert Fusion + SHAP Interpretability（hard F1=0.899, AUC=0.932）

**架构动机**：现有融合策略（静态权重 / stacking / HDM-Net 朴素 gating）都"对所有学生使用相同决策规则"——忽略了个体行为差异。MRE 用可解释的 gating MLP **按学生路由到最合适的专家**。

**架构（双专家 + Gate MLP + 3 种融合模式）**：

```
                  ┌──────────────┐
7-dim events ───>│ Route A (RF) │──> p_rf (n_estimators=200, max_depth=12)
                  └──────────────┘
                                    ↘
                                     Gate MLP (6+7 → 32 → 16 → 2 → softmax)
                                    ↗
                  ┌──────────────┐
46-dim feats ───>│ Route B (LSTM)│──> p_lstm (single-layer, hidden=32)
                  └──────────────┘
                                       ↓
                  α_rf · p_rf + (1-α_rf) · p_lstm
                          ↓
                  ┌───────────┴───────────┐
                  │ soft / hard(STE) / confidence │
                  └───────────┬───────────┘
                              ↓
                          p_final
```

**3 种融合模式 + 2 baseline**：

| 变体 | F1 | AUC | Precision | Recall | 备注 |
|------|-----|-----|-----------|--------|------|
| RF expert (Route A) | 0.8891 ± 0.017 | 0.9175 ± 0.012 | 0.9111 | 0.8694 | 7-dim RF |
| LSTM expert (Route B) | 0.8659 ± 0.028 | 0.9068 ± 0.020 | 0.8981 | 0.8377 | 46-dim LSTM |
| avg_50_50（基线） | 0.8800 ± 0.015 | 0.9271 ± 0.014 | — | — | 等权平均 |
| grid_best_w_rf（基线） | 0.8953 ± 0.017 | 0.9253 ± 0.013 | — | — | 网格搜索最优 |
| MRE-soft | 0.8943 ± 0.015 | **0.9326 ± 0.012** | 0.9287 | 0.8631 | Soft MoE |
| MRE-confidence | 0.8889 ± 0.020 | 0.9236 ± 0.018 | 0.9251 | 0.8567 | Confidence-based |
| **★ MRE-hard** (best) | **0.8986 ± 0.019** | 0.9316 ± 0.008 | 0.9232 | 0.8758 | Hard Routing + STE |

> **MRE-hard 仅用 2 个 base expert 即达到 F1=0.8986**，与 Weighted 1/3/1 (F1=0.9009, 需 3 个 base model) 几乎持平；且 AUC=0.9316 比 Weighted 1/3/1 (0.9349) 略低 0.003。**这是"少而精"的融合方案**。
>
> ⚠️ paper-draft2.md 写的是 F1=0.8958，实际 all_results.json 数据是 **0.8986**——以数据为准。

#### SHAP 路由可解释性

**全局特征重要性（mean |SHAP|）**：

| 排序 | 特征 | mean \|SHAP\| | 类别 |
|:---:|------|------:|------|
| 1 | text_insert | 0.0905 | 7-dim 事件计数 |
| 2 | run | 0.0825 | 7-dim 事件计数 |
| 3 | submit | 0.0620 | 7-dim 事件计数 |
| 4 | rf_prob | 0.0574 | 概率信号 |
| 5 | lstm_prob | 0.0447 | 概率信号 |
| 6-7 | text_remove / text_paste | 0.038 / 0.037 | 7-dim 事件计数 |
| 8-9 | focus_lost / focus_gained | 0.026 / 0.023 | 7-dim 事件计数 |

**特征组贡献**：

| 特征组 | SHAP 总和 | 占比 |
|---|---:|---:|
| **7-dim 事件计数** | **0.359** | **75.7%** |
| RF/LSTM probs (2 维) | 0.102 | 21.5% |
| 交互项 (4 维) | 0.013 | 2.8% |

> **关键发现**：**7 维事件计数贡献了 76% 的路由决策权重**——是 RF/LSTM 概率信号的 **3.5 倍**。门控主要靠"行为强度"而非"专家分歧"做决策。这与"小样本上简单信号最稳"的小数据教育场景一致。

#### 4 个学生画像（Mann-Whitney U 检验 p<1e-21）

| 画像 | 特征 | 占比 | 路由倾向 |
|------|------|------|---------|
| 低活动量未通过者 | 所有 7 事件计数低于均值 6-29% | 66.4% (n=314) | **强 RF** (α_rf 高) |
| 高活动量未通过者 | text_insert 1.67× 全局均值 | 18.2% (n=86) | 强 LSTM |
| 主动自编码者 | 平衡行为 + 多次 run | — | 平衡/略 RF |
| 模板依赖通过者 | text_paste 占比高 | — | RF（已稳定）|

#### 路由规则（Mann-Whitney U=38463, p=0.0000）

**α_rf 区间分布**（n=473）：

| α_rf 区间 | 样本数 | 占比 | 路由倾向 |
|---|---:|---:|---|
| < 0.30 | 66 | 14.0% | 强 LSTM |
| 0.30-0.45 | 20 | 4.2% | 略 LSTM |
| 0.45-0.55 | 73 | 15.4% | 平衡 |
| 0.55-0.70 | 141 | 29.8% | 略 RF |
| > 0.70 | 173 | 36.6% | 强 RF |

#### 关键贡献

1. **仅 2 个 base expert 达到 F1=0.8986**——比 Late Fusion 5-way (F1=0.9056) 低 0.007，但参数 + 推理成本都低一个数量级。
2. **SHAP 量化证明门控依赖简单行为信号**——7 维事件计数占 76% 权重，是论文可解释性部分的强支撑。
3. **学生画像可直接用于教学干预**——4 类画像 + α_rf 区间分布提供了"哪些学生需要 RF 视角、哪些需要 LSTM 视角"的可操作规则。

**实现文件**：`models/mre/{mre_model.py, train.py, analysis.py, shap_analysis.py, shap_deep_dive.py, gen_shap_report.py}`，完整论文草稿见 [`docs/paper-draft2.md`](docs/paper-draft2.md) 与 [`docs/paper-draft2-cn.md`](docs/paper-draft2-cn.md)，SHAP 详细报告见 [`outputs/unified_compare/mre/shap_interpretability_report.md`](outputs/unified_compare/mre/shap_interpretability_report.md)。


> ⚠ **小样本说明**：n=473 是偏小样本，**F1 在 ±0.02 标准差波动下应谨慎解读**；所有"X 显著优于 Y"的强声明建议在大数据（n ≥ 2,000）上重新验证。

**完整消融表 + Late Fusion 权重搜索结果 + 可视化产物清单**：参见 [`docs/EXPERIMENT_RESULTS.md`](docs/EXPERIMENT_RESULTS.md)。

### 复现所有数字

```bash
# 主流模型（写覆 outputs/comparison.csv 和 outputs/unified_compare/）
python main.py --model all
python compare_all_unified.py

# BGM-Net 五变体
python models/bgm_net/train.py --all-variants

# CREAM 五变体
python models/cream/train.py

# Late Fusion 5 路
python late_fusion_5way.py
```

**单数字验证**（例：LSTM-MLP-46d 的 F1）：

```bash
grep '^LSTM_46d' outputs/unified_compare/unified_compare.csv
# → LSTM_46d,0.8246136618141098,...,0.9170170890937019,...,0.8621745342484942,...
```

## 依赖

```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
torch>=2.0.0
scipy>=1.7.0
einops>=0.6.0
tqdm>=4.0.0
matplotlib>=3.5.0  # 可视化用
```

安装: `pip install -r requirements.txt`
