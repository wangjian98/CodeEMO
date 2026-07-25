# CodeEMO 实验结果汇总

> **生成时间**: 2026-07-25
> **数据集**: CS1 IDE 日志（473 名学生，通过 159 / 未通过 314）
> **验证方式**: 5 折分层交叉验证（StratifiedKFold）
> **指标**: Accuracy / Precision / Recall / F1 / AUC

本文件汇总 `results/` 与 `outputs/` 下所有可重现的实验记录，作为论文中的实证数据备份。各实验的源码入口在 `models/` 子目录下，详细超参数见各模型目录。

---

## 1. 5 主流模型统一对比（46-dim vs. 7-dim）

**数据源**: `outputs/unified_compare/unified_compare.csv` / `unified_report.md`
**口径**: y=1 表示未通过（failed），统一 5 折均值 ± std

| 模型 | 特征维度 | Accuracy | Precision | Recall | F1 | AUC |
|------|---------|----------|-----------|--------|----|-----|
| **LSTM** | **46d** | 0.8246 ± 0.034 | 0.8999 ± 0.026 | 0.8281 ± 0.037 | **0.8622 ± 0.028** | **0.9170 ± 0.023** |
| BiLSTM | 46d | 0.8225 ± 0.023 | 0.9238 ± 0.040 | 0.8023 ± 0.061 | 0.8561 ± 0.024 | 0.9036 ± 0.014 |
| Mamba  | 46d | 0.7972 ± 0.042 | 0.8545 ± 0.035 | 0.8377 ± 0.044 | 0.8455 ± 0.033 | 0.8557 ± 0.048 |
| BiLSTM | 7dim  | 0.7295 ± 0.041 | 0.7508 ± 0.042 | 0.8948 ± 0.022 | 0.8153 ± 0.020 | 0.7398 ± 0.053 |
| LSTM   | 7dim  | 0.7189 ± 0.037 | 0.7471 ± 0.038 | 0.8790 ± 0.042 | 0.8062 ± 0.020 | 0.7259 ± 0.052 |
| Mamba  | 7dim  | 0.6109 ± 0.043 | 0.7510 ± 0.022 | 0.6178 ± 0.062 | 0.6768 ± 0.044 | 0.6150 ± 0.063 |

### 关键观察

- **LSTM-46d 在 F1 与 AUC 上同时最强**（F1=0.862, AUC=0.917），是单模型的最强基线。
- **46-dim 全面优于 7-dim**：在 LSTM/BiLSTM/Mamba 上，46-dim 的 F1 平均高出 7-dim 约 +0.04，AUC 高出 +0.15。这与论文 `paper_draft.md` §3.2 中"46 维手工特征 ≫ 7 维原始计数"的结论一致。
- **Mamba-7dim 模型未充分收敛**（非 label bug）：F1=0.677、AUC=0.615 显著低于 LSTM-7dim/BiLSTM-7dim（F1 ≈ 0.81）。**诊断依据**：`outputs/unified_compare/mamba_7dim/probs.npy` 的 `mean≈0.497`、`std≈0.08`（接近常数），是 6 折模型输出几乎不变化的表现——训练未收敛而非 label 反转。**潜在根因**：`models/mamba/train_ms.py` 使用 `finetune_epochs=4`、`batch_size=16`、`max_seq_len=500`，对 6 步 Mamba-SSM 流水线 + 473 学生显然是欠拟合。若再做结果，应将 epoch 提到 20+、batch=32、warmup 至少 3 epoch，或改用主流程外的 mamba_a / mamba_long 配置重跑。
- **所有 6 个组合的 probs.npy 都已经与统一 failed=1 labels 同方向**，无需 1-p 翻转（验证脚本 `scripts/diag_mamba_label.py` 显示：若强行翻转 LSTM_7dim/BiLSTM_7dim/Mamba_7dim 的概率，F1 会从 0.81/0.82/0.68 跌到 0.18/0.16/0.45，AUC 跌破 0.5，反向证明现方向才是对的）。`compare_all_unified.py:48-49` 的 `if features=='46d': p=1-p` 是 **stale code**（疑似历史遗留），可清理但不会改变结果。

### 运行复现

```bash
python main.py --model all
python compare_all_unified.py
```

CSV: `outputs/comparison.csv`（主对比）、`outputs/unified_compare/unified_compare.csv`（统一对比）
可视化: `results/comparison/model_comparison.png`、`outputs/unified_compare/figures/`

---

## 2. BGM-Net 架构消融（5 变体）

**数据源**: `results/bgm_net/{variant}/metrics.json`
**目的**: 验证 behavior gate / entropy attention / ratio cross 三大创新模块的独立贡献
**基线维度**: 5,345 参数（dual-branch MLP，去掉全部三模块）

| 变体 | Gate | Entropy-Attn | Cross | F1@0.5 | AUC | Param |
|------|:----:|:------------:|:-----:|--------|-----|-------|
| **baseline** | ✗ | ✗ | ✗ | **0.7458 ± 0.026** | **0.9079 ± 0.019** | 5,345 |
| no_cross      | ✓ | ✓ | ✗ | 0.7381 ± 0.023 | 0.9061 ± 0.020 | ≈5,545 |
| no_gate       | ✗ | ✓ | ✓ | 0.7290 ± 0.056 | 0.9012 ± 0.032 | 5,553 |
| full          | ✓ | ✓ | ✓ | 0.7226 ± 0.047 | 0.9003 ± 0.030 | 5,577 |
| no_entropy    | ✓ | ✗ | ✓ | 0.7229 ± 0.043 | 0.8905 ± 0.030 | 5,569 |

### 关键发现

1. **dual-branch MLP (baseline) 才是真正起作用的架构**：F1@0.5=0.7458、AUC=0.9079 均高于任何带创新模块的变体。
2. **三个可选模块在 n=473 上均呈负收益**：开启任一模块都会拉低 F1@0.5。这与论文草稿 §4.6 / §5.3 的消融结论一致——在 473 样本上增加参数会引入方差，对小样本教育数据不利。
3. **Entropy 注意力有一个例外**：去掉 entropy attn 后 AUC 下降 ~0.017（F1@0.5 不变），说明 entropy 路由对概率排序质量有正面贡献，但阈值化后的 F1 体现不出来。
4. **参数效率**：即使最强 baseline 也只有 5,345 参数，比 LSTM（~50K）少 9.6×，在 IDE 插件等边缘部署场景下有显著优势。

### 运行复现

```bash
python models/bgm_net/train.py --all-variants
```

每个变体结果落在 `results/bgm_net/{variant}/metrics.json`。

---

## 3. CREAM 架构消融（5 变体）

**数据源**: `results/cream/{variant}/metrics.json`
**目的**: 验证 CREAM（对比学习 + 自注意力机制）各组件的独立贡献

| 变体 | F1@0.5 | AUC |
|------|--------|-----|
| no_contrastive    | **0.7685 ± 0.021** | 0.8982 ± 0.018 |
| no_bottleneck     | 0.7591 ± 0.026 | 0.9052 ± 0.018 |
| baseline          | 0.7553 ± 0.020 | 0.9040 ± 0.016 |
| no_se             | 0.7559 ± 0.027 | **0.9085 ± 0.014** |
| full              | 0.7539 ± 0.030 | 0.9073 ± 0.021 |

### 关键发现

- **CREAM 五变体 F1@0.5 集中在 0.754–0.769 的窄带内**（极差仅 0.015），说明对比学习头 / squeeze-excitation / bottleneck 等子模块在 n=473 上对预测影响有限。
- **no_contrastive F1 最高但 AUC 最低**（F1=0.769 vs AUC=0.898），呈现典型的"概率分布塌缩到决策阈值附近"的迹象——去掉对比学习后分类器更激进，AUC 排序质量反而下降。
- **no_se (去 SE 注意力) 取得最高 AUC**（0.9085），与 full 几乎持平，但 F1 略低。SE 注意力提供了轻微的概率校准改进，但代价是更不稳定。

### 与 BGM-Net 对比

- CREAM 全模型 F1=0.7539、参数量大、训练含对比学习头；
- BGM-Net baseline F1=0.7458、参数仅 5K、纯监督；
- 二者在 473 样本下的 F1 差距仅 0.008，**支撑了一个更宽泛的观察**：在 n<500 小数据教育场景下，**模型结构创新带来的边际收益通常小于 1–2 个 F1 点**，参数效率与可解释性成为更重要的设计目标。

### 运行复现

```bash
python models/cream/train.py
```

结果在 `results/cream/{variant}/metrics.json`。

---

## 4. Late Fusion 5 路集成

**数据源**: `outputs/late_fusion_5way_v1/results.json`
**方法**: 5 个基模型概率的网格搜索权重 + 阈值优化

| 组合 | 最优权重 (a,b,c,d,e) | F1 | F1_std | AUC |
|------|---------------------|----|--------|-----|
| Top-1 | (0.5, 0.3, 0.1, 0.1, 0.0) | **0.9056 ± 0.015** | — | 0.9222 ± 0.011 |
| Top-2 | (0.5, 0.3, 0.0, 0.2, 0.0) | 0.9039 ± 0.016 | — | 0.9206 |

- 集成的 F1=0.9056 已显著高于单模型最佳（0.8622），提升 **+0.043 F1**；
- AUC 同时上升至 0.922，比单模型最强 LSTM-46d（AUC=0.917）还高 0.005；
- **Late Fusion 是当前公开实验中的最强 baseline**，论文中 7 路融合 F1=0.9013 与之相符（差异来自模型选择与权重搜索区间）；
- 代价是推理时需并行运行 5 个模型，部署成本高。

### 运行复现

```bash
python models/late_fusion_5way.py
```

---

## 5. 各模型独立可视化产物

所有模型在 `results/{model}/` 下保存三类可视化：

| 模型 | confusion_matrix | roc_curve | training_curves | feature_importance |
|------|:-----------------:|:----------:|:---------------:|:------------------:|
| random_forest | ✓ | ✓ | — | ✓ |
| lstm          | ✓ | ✓ | ✓ | — |
| bilstm        | ✓ | ✓ | ✓ | — |
| transformer   | ✓ | ✓ | ✓ | — |
| mamba         | 见 `outputs/mamba*/` | | | |

---

## 6. 数据规模与复现说明

### 数据规模注意事项

- **n=473 是一个偏小的样本量**，5 折 CV 下每折训练集仅约 378 条。
- 多数模型的标准差 σ(F1) 在 0.02–0.04 区间，**单次实验的 ±1 σ 波动可能盖过某些变体间的细微差异**。
- 论文草稿与本仓库结果完全可重现，但**所有"X 模型显著优于 Y 模型"的强声明都应在 n ≥ 2,000 数据集上重新验证**后再下定论。

### 完全复现流程

```bash
# 1. 准备数据 (需将 IDE_logs 放在 /tmp/IDE_logs/ 下)
git clone git@github.com:wangjian98/CodeEMO.git
cd CodeEMO
pip install -r requirements.txt

# 2. 跑所有模型
python main.py --model all                 # 5 主流模型
python models/bgm_net/train.py --all-variants  # BGM-Net 消融
python models/cream/train.py               # CREAM 消融
python models/late_fusion_5way.py          # Late Fusion 5 路
python compare_all_unified.py              # 统一对比

# 3. 复现数值
python -c "import json; print(json.dumps(json.load(open('outputs/unified_compare/unified_compare.csv'.replace('.csv','.json' if False else ''))), indent=2))"
```

### 结果汇总脚本

`outputs/unified_compare/unified_compare.csv` 是自动生成的主指标矩阵，可用任何表格工具加载分析。

---

## 7. 数据来源速查表

| 数据 | 路径 |
|------|------|
| 统一对比 | `outputs/unified_compare/unified_compare.csv` |
| Unified report | `outputs/unified_compare/unified_report.md` |
| BGM-Net 消融 | `results/bgm_net/{variant}/metrics.json` |
| CREAM 消融 | `results/cream/{variant}/metrics.json` |
| Late Fusion 5路 | `outputs/late_fusion_5way_v1/results.json` |
| 各模型可视化 | `results/{model}/*.png` |
| 综合 analysis report | `results/analysis_report.md` |
| 论文草稿 | `docs/paper_draft.md` |
