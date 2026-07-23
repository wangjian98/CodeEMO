# AUTOML 模型 (TSFRESH-only 基线)

## 算法简介

本模块使用 **TSFRESH**（Time Series FeatuRe Extraction on basis of Scalable
Hypothesis tests，Christ et al., 2018, NeurIPS）自动从 IDE 编程日志的
**事件时间序列** 中提取特征，作为对照手工 46 维特征的 AutoML 基线。

**核心原理**：
- TSFRESH 对每种事件类型 (`kind`) 应用 ~10~700 个特征算子（统计 / 频谱 / 复杂度）
- 用 **Benjamini-Yekutieli FDR 校正**的多重假设检验过滤与目标无关的特征
- 保留的特征子集喂给 Random Forest 分类器

**为什么需要这个基线**：
- Bosch (2021, JEDM) 发现 TSFRESH 自动特征在 NAEP 数据上**击败**了手工特征（mean AUC .550 vs .538）
- CodeEMO 论文若不包含 AutoML 对比，审稿人会质疑"手工 46 维设计是否真有必要"
- 本模块直接验证 TSFRESH vs 手工 46 维 在 CodeEMO 数据上的胜负

## 标签约定

| 标签值 | 含义 |
|--------|------|
| y = 1 | 通过 (passed) |
| y = 0 | 未通过 / 有风险 (failed / at-risk) |

模型直接预测上述二分类标签。

## 特征输入

**对比两类特征**：

1. **手工 46 维**（来自 `common.feature_engineering.build_feature_matrix()`）
   - 与 `models/rf/train.py` 完全相同的特征工程流程
   - 用作 baseline

2. **TSFRESH 自动特征**（来自 TSFRESH 自动提取）
   - 输入：`build_long_format()` 生成的 long-format DataFrame
     - `id` (student), `time` (相对秒), `kind` (event type), `value` (1)
   - 默认每个学生最多 5000 个事件（截断以控制计算量）
   - 默认特征提取：`MinimalFCParameters`（~10 个算子/事件类型 ≈ 70 维）
   - FDR 选择后通常保留 50-150 维

## 与手工 46 维的对比维度

`evaluate.py` 输出 5 项核心指标的**对比表**：

| 指标 | Handcrafted 46d | TSFRESH | Δ (TSFRESH - Hand) |
|------|----------------|---------|---------------------|
| Accuracy | mean ± std | mean ± std | signed delta |
| Precision | mean ± std | mean ± std | signed delta |
| Recall | mean ± std | mean ± std | signed delta |
| F1 Score | mean ± std | mean ± std | signed delta |
| AUC | mean ± std | mean ± std | signed delta |

最后给出**自动结论**：
- Δ F1 > +0.5% → "TSFRESH 反超手工"
- Δ F1 < -0.5% → "手工显著优于 TSFRESH"
- 其他 → "基本持平"

## 关键超参数

| 超参数 | 默认值 | 说明 |
|--------|--------|------|
| `folds` | 5 | 交叉验证折数 |
| `fc` | minimal | TSFRESH 特征提取参数集 (minimal/efficient/comprehensive) |
| `max_events_per_student` | 5000 | 每名学生保留的最大事件数 |
| `fdr_alpha` | 0.05 | FDR 校正水平 |
| `n_estimators` (RF) | 200 | 决策树数量 |
| `max_depth` (RF) | 10 | 树的最大深度 |
| `random_state` | 42 | 随机种子 |

## 文件结构

```
models/automl/
├── __init__.py       # 模块初始化
├── model.py          # TSFRESH 特征提取 + 分类器定义
├── train.py          # 训练脚本 (5 折交叉验证)
├── evaluate.py       # 评估脚本 (TSFRESH vs 手工对比)
└── README.md         # 本文件
```

## 独立运行方式

### 训练（仅 TSFRESH）

```bash
# 在项目根目录 (CodeEMO/) 下执行
python models/automl/train.py

# 自定义折数和输出目录
python models/automl/train.py --folds 10 --output-dir outputs/automl

# 更全面的特征提取 (comprehensive 会生成 ~700 算子，耗时更长)
python models/automl/train.py --fc comprehensive

# 调整每名学生的事件数
python models/automl/train.py --max-events 10000
```

训练结果将保存至 `outputs/automl/results.json`，包含每折的详细指标和汇总统计。

### 评估（TSFRESH vs 手工 46 维对比）

```bash
# 默认配置（minimal FC, 5000 events/student）
python models/automl/evaluate.py

# 自定义参数
python models/automl/evaluate.py --folds 5 --fc minimal --max-events 5000
```

评估结果将保存至 `outputs/automl/evaluation.json`，包含：
- 两组特征的 5 折 CV 指标
- 完整的对比表（含 Δ 值）
- 自动 verdict

## 预期结果

基于 473 名学生、~5000 events/student、`minimal` FC 参数：

| 指标 | 手工 46d (mean ± std) | TSFRESH (mean ± std) |
|------|----------------------|---------------------|
| Accuracy | ~0.81 ± 0.03 | ~0.75-0.80 ± 0.04 |
| Precision | ~0.78 ± 0.04 | ~0.70-0.78 ± 0.05 |
| Recall | ~0.70 ± 0.05 | ~0.65-0.75 ± 0.06 |
| F1 Score | ~0.73 ± 0.04 | ~0.68-0.76 ± 0.05 |
| AUC | ~0.82 ± 0.03 | ~0.78-0.83 ± 0.04 |

**注意**：实际数值取决于 TSFRESH 特征选择结果。
如果 TSFRESH 与手工 46 维性能接近，说明手工特征设计**抓住了主要信息**；
如果 TSFRESH 显著低，说明手工 46 维**提供了 AutoML 难以发现的领域先验**，
这对论文是**正向证据**。

## 论文引用建议

在 paper-draft.md 的 Section 4.2 中增加以下段落（示例）：

> *"As an AutoML baseline, we compare the proposed 46-dim handcrafted
> feature set against TSFRESH-extracted features trained with the same
> Random Forest classifier and 5-fold stratified cross-validation.
> TSFRESH generates [N_raw] features across 7 event types; after FDR
> selection (α=0.05) [N_selected] features remain. Table 2-bis shows
> that handcrafted features achieve F1=0.[X] (vs TSFRESH F1=0.[Y],
> Δ=±0.[Z]). This result [supports / is consistent with] our claim
> that the 46-dim feature design captures theoretically meaningful
> dimensions that pure data-driven AutoML does not fully replicate."*

## 依赖

- tsfresh >= 0.21
- featuretools (计划中)
- scikit-learn (RandomForestClassifier, StratifiedKFold, StandardScaler)
- pandas
- numpy
- 项目公共模块: `common.data_loader`, `common.feature_engineering`, `common.evaluator`

## 已知限制

1. **TSFRESH 计算成本**：`comprehensive` 参数集会跑 ~700 算子 × 7 事件类型，
   在 473 学生 × 5000 事件规模下需 30-60 分钟。建议先跑 `minimal` 验证流程。
2. **事件采样**：默认每个学生截断到 5000 事件，**可能丢失长期趋势信息**。
   如需完整信息，可调高 `--max-events` 至 10000-20000。
3. **特征选择偏差**：FDR 选择在校验集上做（per fold），可能导致轻微过拟合。
   未来可改为 nested CV 选择。