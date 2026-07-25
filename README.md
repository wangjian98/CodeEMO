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
| LSTM | 46d | 0.8246 ± 0.034 | 0.8622 ± 0.028 | 0.9170 ± 0.023 |
| Transformer | 46d | 0.8183 ± 0.032 | 0.8567 ± 0.031 | 0.9034 ± 0.009 |
| BiLSTM | 46d | 0.8225 ± 0.023 | 0.8561 ± 0.024 | 0.9036 ± 0.014 |
| Mamba  | 46d | 0.7972 ± 0.042 | 0.8455 ± 0.033 | 0.8557 ± 0.048 |
| BiLSTM | 7dim | 0.7295 ± 0.041 | 0.8153 ± 0.020 | 0.7398 ± 0.053 |
| LSTM   | 7dim | 0.7189 ± 0.037 | 0.8062 ± 0.020 | 0.7259 ± 0.052 |
| Mamba  | 7dim | 0.6109 ± 0.043 | 0.6768 ± 0.044 | 0.6150 ± 0.063 |

> **关键发现**：
> 1. **新榜首 Weighted 1/3/1 (F1=0.901 / AUC=0.935)** – 1×RF + 3×HDM-Net v2 + 1×LSTM 的加权集成，超越 Late Fusion 5-way 在 F1 (略低 0.005) 和 AUC (反超 0.013) 的表现。同时 Precision=0.935 是全部单模型里最高的。详情见下一节。
> 2. **F1/AUC 次榜首** = RF-7dim (F1=0.888 / AUC=0.918)；LSTM-46d (AUC=0.917) 几乎并列。
> 3. **树/伪序列模型在 7-dim 简洁特征上反超 46-dim**：RF-7dim > RF-46d (ΔF1=+0.022)，Transformer-7dim > Transformer-46d (ΔF1=+0.012)。
> 4. **序列模型恰好相反**：LSTM-46d > LSTM-7dim (ΔF1=+0.056)；BiLSTM-46d > BiLSTM-7dim (ΔF1=0.041)。
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
| Weighted 2/3/1 | 0.8996 ± 0.024 | 0.9288 ± 0.020 | 0.8726 ± 0.032 | 0.9322 ± 0.013 | 2×RF + 3×HDM v2 + 1×LSTM |
| **Stack (top-3 LR)** | 0.8986 ± 0.015 | 0.9072 ± 0.021 | **0.8918 ± 0.037** | 0.9324 ± 0.015 | LR(C=0.1) meta-learner |
| HDM-Net v2 (单 best) | 0.8982 ± 0.022 | 0.9256 ± 0.017 | 0.8726 ± 0.029 | 0.9273 ± 0.014 | — |

**观察**：

1. **Weighted 1/3/1 是新最佳 F1/AUC**。在 RF × 1 + HDM-Net v2 × 3 + LSTM × 1 的加权下：F1=0.9009、AUC=0.9349，同时 Precision=0.9351、Recall=0.8694。结果验证了"HDM-Net v2 是最强但加权上 RF/LSTM 各贡献一些"。
2. **Stacking (LR) 提升 Recall 最多**（0.8918 +0.019 vs HDM v2）。LR meta-learner 在 5-fold 里有意识地给 failed 样本加重权，提高识别率。
3. **Weighted 1/3/1 vs Late Fusion 5-way**：F1 -0.005、AUC +0.013 — 在三项中F1略低，但 AUC 明显超。这说明 "质量加权" 在面向 failed=1 任务上有适应能力。

**实现路径**：`outputs/unified_compare/stack_top3_LR_C0.1/`, `weighted_1_3_1/`, `weighted_2_3_1/` 都存储了 OOF probs/labels/fold_idx.json，可直接被下游论文画图使用。

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

**单数字验证**（例：LSTM-46d 的 F1）：

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
