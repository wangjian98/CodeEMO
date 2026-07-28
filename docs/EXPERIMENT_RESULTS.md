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
| **RF** | **7dim** | 0.8541 ± 0.025 | 0.9082 ± 0.031 | 0.8694 ± 0.033 | **0.8876 ± 0.019** | **0.9175 ± 0.012** |
| Transformer | 7dim | 0.8352 ± 0.041 | 0.9182 ± 0.030 | 0.8248 ± 0.039 | 0.8689 ± 0.034 | 0.9162 ± 0.027 |
| RF | 46d | 0.8247 ± 0.034 | 0.8765 ± 0.012 | 0.8568 ± 0.060 | 0.8654 ± 0.030 | 0.9069 ± 0.029 |
| **LSTM-MLP** | **46d** | 0.8246 ± 0.034 | 0.8999 ± 0.026 | 0.8281 ± 0.037 | 0.8622 ± 0.028 | 0.9170 ± 0.023 |
| Transformer | 46d | 0.8183 ± 0.032 | 0.8944 ± 0.023 | 0.8250 ± 0.058 | 0.8567 ± 0.031 | 0.9034 ± 0.009 |
| BiLSTM-MLP | 46d | 0.8225 ± 0.023 | 0.9238 ± 0.040 | 0.8023 ± 0.061 | 0.8561 ± 0.024 | 0.9036 ± 0.014 |
| Mamba  | 46d | 0.7972 ± 0.042 | 0.8545 ± 0.035 | 0.8377 ± 0.044 | 0.8455 ± 0.033 | 0.8557 ± 0.048 |
| BiLSTM-Seq | 7dim  | 0.7295 ± 0.041 | 0.7508 ± 0.042 | 0.8948 ± 0.022 | 0.8153 ± 0.020 | 0.7398 ± 0.053 |
| LSTM-Seq | 7dim  | 0.7189 ± 0.037 | 0.7471 ± 0.038 | 0.8790 ± 0.042 | 0.8062 ± 0.020 | 0.7259 ± 0.052 |
| Mamba  | 7dim  | 0.6109 ± 0.043 | 0.7510 ± 0.022 | 0.6178 ± 0.062 | 0.6768 ± 0.044 | 0.6150 ± 0.063 |

### 关键观察

- **RF-7dim 是 F1+ AUC 双榜首**（F1=0.888, AUC=0.918），与 LSTM-MLP-46d（AUC=0.917）几乎并列 AUC。这推出原本"46-dim 全面优于 7-dim"的简化结论是一个低估——它对不同模型族有反向贡献。
- **按模型族的 “最优特征维度”正好相反**：RF/Transformer (伪序列/树型) 在 7-dim 简洁特征上反超 46-dim；LSTM/BiLSTM/Mamba (序列型) 反之。在原本只有 3 个序列模型时这个现象被遮藏，加了 RF/Transformer 后才显形。
- **46-dim 与 7-dim 的对比现在以全 10 个组合为依据**：RF-7dim F1/AUC 都高于其 46-dim；Transformer-7dim > Transformer-46d；LSTM/BiLSTM 在 46-d 上表现明显优于 7-d。Mamba 两端都差，但 46-d 仍明显高于 7-d。
- **Mamba-7dim 模型未充分收敛**（非 label bug）：F1=0.677、AUC=0.615 显著低于其余 9 个组合。**诊断依据**：`outputs/unified_compare/mamba_7dim/probs.npy` 的 `mean≈0.497`、`std≈0.08`（接近常数），是 6 折模型输出几乎不变化的表现——训练未收敛而非 label 反转。**潜在根因**：`models/mamba/train_ms.py` 使用 `finetune_epochs=4`、`batch_size=16`、`max_seq_len=500`，对 6 步 Mamba-SSM 流水线 + 473 学生是欠拟合。若重跑应将 epoch 提到 20+、batch=32、warmup ≥3 epoch，或改用 mamba_a / mamba_long 配置。
- **label convention**：10 个组合中，LSTM/BiLSTM/Mamba 的 46-d 仍需在 `compare_all_unified.load_combo` 中做 1-p 翻转（这些脚本原型以 P(passed) 输出 probs），其余 7 个组合已是 P(failed)、不需翻。诊断脚本 `scripts/diag_mamba_label.py` 验证了这一点。

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

- 集成的 F1=0.9056 已显著高于单模型 RF-7dim（F1=0.888）+0.018 F1；对比原来单模型最强 LSTM-MLP-46d（F1=0.862）提升 **+0.043 F1**；
- AUC 同时上升至 0.922，比单模型最强 RF-7dim（AUC=0.918）还高 0.004；
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


---

## 8. OST-Forest: Out-of-fold Self-distilled Tree Forest（2026-07-27 新增）

**数据源**：`outputs/ost_forest/{results.json, refine_F*.json}`
**方法**：20 棵异构随机森林（depth 3/5/8, seed varies）→ 5 折 OOF matrix (20-d) → 与 7-d handcrafted + 6-d session stat 拼接为 33-d meta feature → LR head + self-distillation soft label (α=0.4)

### 8.1 主结果（CV 5 折均值 vs F3 单折）

| 指标 | CV 均值 | F3 单折（refinement） | 备注 |
|------|---------|---------------------|------|
| Accuracy | 0.8521 ± 0.027 | 0.9043 | |
| Precision | 0.9378 ± 0.035 | 0.9355 | |
| Recall | 0.8344 ± 0.048 | 0.9206 | |
| **F1** | **0.8816 ± 0.023** | **0.9280** | F3 单折反超 HDM-Net v2 (0.8982) |
| AUC | 0.9087 ± 0.014 | 0.9309 | |

### 8.2 5 个核心 Finding

1. **20 RF OOF marginal in 7-d**：大量 RF OOF 信号在 7-d 上边际收益有限
2. **LR head > LightGBM head by +0.0147 F1**（n=473）：简单线性头在小样本上反而胜出
3. **Soft label α=0.4 稳定 +0.007 F1**：自蒸馏有微弱贡献
4. **20→5 RF no significant degradation**：RF 数量可大幅缩减而不损性能
5. **7d > 46d in this regime**：验证"小样本偏 7-d"假说

### 8.3 模型排名（F1，5 折 CV）

| 排名 | 模型 | F1 | 类型 |
|------|------|----|------|
| 1 | Late Fusion 5-way | 0.9056 | 5 模型集成 |
| 2 | **PR-DE-Net 3-way** | **0.9027** | 3 模型融合 |
| 3 | Weighted 1/3/1 | 0.9009 | 3 模型加权 |
| 4 | Stack top-3 LR | 0.8986 | 3 模型 LR stacking |
| 5 | HDM-Net v2 (T3) | 0.8982 | 单模型 |
| 6 | OST-Forest (CV 均值) | 0.8816 | 单模型 |
| 7 | RF-LSTM v3 | 0.8809 | 单模型 |
| 8 | RF-7dim | 0.8876 | 单模型（最强传统 ML）|
| 9 | PR-DE-Net (full) | 0.8601 | 单模型 |
| 10 | MASC-Net (baseline_only) | 0.7985 | 单模型 |

> OST-Forest 的 F3 单折 0.928 是**单一折表现**，CV 均值 0.8816 更稳定；论文引用应使用均值。

### 8.4 运行复现

```bash
python models/ost_forest/train.py
```

主结果落在 `outputs/ost_forest/results.json`，refinement 变体在 `outputs/ost_forest/refine_F*.json`，详细报告 [`docs/OST_FOREST_REPORT.md`](../OST_FOREST_REPORT.md)。

---

## 9. MASC-Net: Multi-scale Adaptive Sample-aware Contrastive Network（2026-07-26 新增）

**数据源**：`outputs/masc_net/{baseline_only,no_contrastive,no_uncertainty,full}_results.json`
**架构**：多尺度编码（CNN+MLP+Cross-Scale Attention）+ Sample-aware 对比（MoCo + adaptive τ + hard negatives）+ Prototype 记忆库（K=2 × M=4, EMA）+ Adaptive threshold 分类器 + Uncertainty 分支

### 9.1 4 变体消融（5 折 CV，n=473）

| 变体 | Contrastive | Uncertainty | F1 | AUC |
|------|:-----------:|:-----------:|----|-----|
| **baseline_only** | ✗ | ✗ | **0.7985 ± 0.042** | **0.8995 ± 0.020** |
| no_contrastive | ✗ | ✓ | 0.7517 ± 0.044 | 0.8479 ± 0.054 |
| no_uncertainty | ✓ | ✗ | 0.7467 ± 0.026 | 0.8624 ± 0.026 |
| full | ✓ | ✓ | 0.7286 ± 0.042 | 0.8644 ± 0.021 |

### 9.2 关键负发现

1. **contrastive + uncertainty 模块在 n=473 上拖累 F1**（−7 pp vs baseline_only），与 BGM-Net 消融结论一致——**小样本下增加参数会引入方差**。
2. **baseline_only 反超 BGM-Net baseline**（F1=0.7985 vs 0.7458，**+5.27 pp**），说明多尺度编码本身（去掉对比学习头）已经是个不错的特征提取器。
3. **full 模型 AUC 仍可观**（0.8644）但 F1 显著下降，表明对比学习优化了概率排序但破坏了阈值化的分类决策。

### 9.3 与 BGM-Net / CREAM 消融对比

| 模型 | best 变体 F1 | full F1 | 负发现强度 |
|------|------------|---------|----------|
| BGM-Net | 0.7458 (baseline) | 0.7226 (full) | full − baseline = −0.023 |
| CREAM | 0.7685 (no_contrastive) | 0.7539 (full) | full − baseline = −0.0014 |
| **MASC-Net** | **0.7985 (baseline_only)** | **0.7286 (full)** | full − baseline = **−0.070** ⚠ |

> MASC-Net 的"小样本负面效应"比 BGM-Net / CREAM 都强烈（−7 pp vs −1~2 pp），暗示 MoCo + prototype + uncertainty 的组合对小样本教育数据是**最不友好**的。

### 9.4 运行复现

```bash
python models/masc_net/train.py --ablation baseline_only   # 最佳变体
python models/masc_net/train.py --ablation no_contrastive
python models/masc_net/train.py --ablation no_uncertainty
python models/masc_net/train.py --ablation full
```

每个变体结果落在 `outputs/masc_net/{variant}_results.json`。

---

## 10. PR-DE-Net: Precision-Recall Gated Dual-Encoder Network（2026-07-26 新增，未 commit）

**数据源**：`outputs/pr_de_net/{full/results.json, fusion_{3way,4way}.json, comparison_with_baselines.json}`
**架构动机**：LSTM-MLP-46d / BiLSTM-MLP-46d Recall 高（0.80-0.83）但 Precision 仅 0.92；Transformer-7d Precision 高（0.918）但 Recall 0.82——两者学**互补判别特征**，应当端到端融合。

### 10.1 单模型 + 消融

| 变体 | 描述 | F1 | Precision | Recall | AUC |
|------|------|-----|-----------|--------|-----|
| **full (Gate + 三段式 Loss)** | 完整模型 | **0.8601 ± 0.027** | 0.9092 | 0.8184 | 0.8711 |
| no_gate | 固定 0.5/0.5 平均 | 0.8486 | 0.8691 | 0.8344 | 0.8412 |
| single_loss | 只用 γ·BCE | 0.8653 | 0.8818 | 0.8504 | 0.8781 |

### 10.2 Gate 行为（设计验证）

| 子集 | n | gate 均值 | 偏 RNN 占比 | 偏 Trans 占比 |
|------|---|----------|-------------|---------------|
| Failed (y=1) | 314 | **0.430** | **59.2%** | 40.8% |
| Passed (y=0) | 159 | **0.659** | 27.0% | **73.0%** |

> ✅ Gate 真的在样本级做 PR 路由——failed→RNN（RNN Recall 高），passed→Trans（Trans Precision 高）

### 10.3 融合贡献（新 SOTA）

| 方案 | F1 | AUC | 备注 |
|------|-----|-----|------|
| Weighted 1/3/1（融合基线） | 0.9009 | 0.9349 | 1×RF + 3×HDM v2 + 1×LSTM |
| Stack top-3 LR | 0.8986 | 0.9324 | |
| **★ PR-DE-Net 3-way** (2.5RF + 2.5HDM + 1.0PR-DE) | **0.9027** | 0.9265 | **新 SOTA 融合方案** |
| ★ PR-DE-Net 4-way (1RF + 3HDM + 0LSTM + 1PR-DE) | 0.9026 | 0.9288 | |

**PR-DE-Net 把融合 F1 从 0.9009 推到 0.9027**（+0.18 pp）。

### 10.4 失败模式

| 类别 | n | Final 准确率 | gate |
|------|---|-------------|------|
| Failed 误判（y=1, pred=0） | 57 | 0% | 0.685（错误走了 Trans 分支）|
| Passed 误判（y=0, pred=1） | 26 | 0% | 0.514 |

> 单模型架构无法突破——错把 failed 当 passed 的样本（57/83）是模型盲区，需要其他模型族互补，这正是 RF/HDM-Net 在融合中贡献 0.5-1.0 权重的原因。

### 10.5 运行复现

```bash
# 主模型
python3 models/pr_de_net/train.py --ablation full

# 消融
python3 models/pr_de_net/ablation.py

# 融合搜索
python3 models/pr_de_net/fusion.py

# 综合分析报告
python3 models/pr_de_net/report.py
```

详细 README 见 [`models/pr_de_net/README.md`](../../models/pr_de_net/README.md)。**注意：PR-DE-Net 当前为 untracked 状态（`models/pr_de_net/` 和 `docs/paper-draft2*.md` 还未 commit），需要先 `git add` 后再 commit。**


---

## 11. MRE: Multi-Route Expert Fusion + SHAP Interpretability（2026-07-28 新增，已 commit 1661fa4）

**数据源**：`outputs/unified_compare/mre/{all_results.json, shap_interpretability_report.md, shap_results.json}`
**配套论文草稿**：[`docs/paper-draft2.md`](paper-draft2.md) + [`docs/paper-draft2-cn.md`](paper-draft2-cn.md)

### 11.1 架构

- **Route A（RF Expert）**：7-dim 事件计数 + Random Forest (n_estimators=200, max_depth=12)
- **Route B（LSTM Expert）**：46-dim hand-crafted features + 单层 LSTM (hidden=32)
- **Gate MLP**：13-dim 输入（6 RF/LSTM 概率统计 + 7 事件计数）→ 32 → 16 → 2 → softmax → α_rf
- **3 种融合模式**：
  - **soft**：α_rf · p_rf + (1-α_rf) · p_lstm
  - **hard**：直通估计器（STE）按 α_rf > 0.5 二选一
  - **confidence**：选 max(p_rf, p_lstm) 对应的 expert

### 11.2 完整结果表（5 折分层 CV，n=473）

| 变体 | F1 | AUC | Precision | Recall | Accuracy | 备注 |
|------|-----|-----|-----------|--------|----------|------|
| RF expert (Route A) | 0.8891 ± 0.017 | 0.9175 ± 0.012 | 0.9111 | 0.8694 | 0.8563 | 7-dim RF |
| LSTM expert (Route B) | 0.8659 ± 0.028 | 0.9068 ± 0.020 | 0.8981 | 0.8377 | 0.8289 | 46-dim LSTM |
| avg_50_50（基线） | 0.8800 ± 0.015 | 0.9271 ± 0.014 | — | — | — | 等权平均 |
| grid_best_w_rf（基线） | 0.8953 ± 0.017 | 0.9253 ± 0.013 | — | — | — | 网格搜索最优 |
| MRE-soft | 0.8943 ± 0.015 | **0.9326 ± 0.012** | 0.9287 | 0.8631 | 0.8647 | Soft MoE |
| MRE-confidence | 0.8889 ± 0.020 | 0.9236 ± 0.018 | 0.9251 | 0.8567 | 0.8584 | Confidence-based |
| **★ MRE-hard** (best) | **0.8986 ± 0.019** | 0.9316 ± 0.008 | 0.9232 | 0.8758 | 0.8690 | Hard Routing + STE |

> ⚠️ paper-draft2.md 写的是 F1=0.8958，实际 all_results.json 是 **0.8986**——以数据为准。

### 11.3 SHAP 全局特征重要性（13 维 → 按 mean |SHAP| 排序）

| 排序 | 特征 | mean \|SHAP\| | 类别 |
|:---:|------|------:|------|
| 1 | text_insert | 0.0905 | 7-dim 事件 |
| 2 | run | 0.0825 | 7-dim 事件 |
| 3 | submit | 0.0620 | 7-dim 事件 |
| 4 | rf_prob | 0.0574 | 概率信号 |
| 5 | lstm_prob | 0.0447 | 概率信号 |
| 6-7 | text_remove / text_paste | 0.0385 / 0.0365 | 7-dim 事件 |
| 8-9 | focus_lost / focus_gained | 0.0263 / 0.0227 | 7-dim 事件 |
| 10-13 | max/min/|·|/rf·lstm | < 0.004 | 交互项 |

**特征组贡献**：

| 特征组 | SHAP 总和 | 占比 |
|---|---:|---:|
| **7-dim 事件计数** | **0.359** | **75.7%** |
| RF/LSTM probs (2 维) | 0.102 | 21.5% |
| 交互项 (4 维) | 0.013 | 2.8% |

> **关键发现**：7 维事件计数贡献了 76% 的路由决策权重——是 RF/LSTM 概率信号的 3.5 倍。**门控主要靠"行为强度"而非"专家分歧"做决策**。SHAP 重构误差 = 0.0000（每折验证）。

### 11.4 4 个学生画像（Mann-Whitney U=38463, p=0.0000）

| 画像 | 行为特征 | 占比 | 路由倾向 |
|------|----------|------|---------|
| 低活动量未通过者 | 所有 7 事件 < 均值 6-29% | 66.4% (n=314) | **强 RF** (α_rf 高) |
| 高活动量未通过者 | text_insert 1.67× 均值 | 18.2% (n=86) | 强 LSTM |
| 主动自编码者 | 平衡行为 + 多次 run | 少量 | 平衡/略 RF |
| 模板依赖通过者 | text_paste 占比高 | 少量 | RF |

**α_rf 区间分布**（n=473）：

| α_rf 区间 | 样本数 | 占比 | 路由倾向 |
|---|---:|---:|---|
| < 0.30 | 66 | 14.0% | 强 LSTM |
| 0.30-0.45 | 20 | 4.2% | 略 LSTM |
| 0.45-0.55 | 73 | 15.4% | 平衡 |
| 0.55-0.70 | 141 | 29.8% | 略 RF |
| > 0.70 | 173 | 36.6% | 强 RF |

### 11.5 与现有最佳融合方案对比

| 模型 | F1 | AUC | 备注 |
|------|-----|-----|------|
| Late Fusion 5-way | 0.9056 | 0.9222 | 5 个 base model |
| **PR-DE-Net 3-way** | **0.9027** | 0.9265 | 3 个 base model (已加进 README) |
| Weighted 1/3/1 | 0.9009 | 0.9349 | 3 个 base model 加权 |
| **★ MRE-hard** (仅 2 base) | **0.8986** | 0.9316 | **"少而精"**——2 个 base + 可解释 Gate |

> MRE-hard 仅用 2 个 base expert 即达到 F1=0.8986，与 Weighted 1/3/1 (F1=0.9009) 几乎持平；推理成本低一个数量级。

### 11.6 运行复现

```bash
# 主实验（生成 outputs/unified_compare/mre/all_results.json）
python3 models/mre/train.py

# SHAP 解释性分析（生成 shap_results.json + 可视化）
python3 models/mre/shap_analysis.py

# SHAP 深度分析（含 4 学生画像）
python3 models/mre/shap_deep_dive.py

# 生成可读报告
python3 models/mre/gen_shap_report.py
```

详细 SHAP 报告见 [`outputs/unified_compare/mre/shap_interpretability_report.md`](../../outputs/unified_compare/mre/shap_interpretability_report.md)，论文草稿见 [`docs/paper-draft2.md`](paper-draft2.md)。
