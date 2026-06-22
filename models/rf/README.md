# 随机森林模型 (Random Forest)

## 算法简介

本项目使用 **随机森林 (Random Forest)** 作为传统机器学习基线模型，用于基于 IDE 编程日志的学生早期风险预测。

随机森林是一种基于决策树集成的监督学习算法，通过 Bootstrap 采样构建多棵决策树并对预测结果进行多数投票。该算法能够有效处理高维特征、捕捉特征间的非线性关系，且对过拟合有较好的抵抗力，适合作为与传统机器学习方法对比的基线模型。

本模块基于 `scikit-learn` 的 `RandomForestClassifier` 实现。

## 标签约定

| 标签值 | 含义 |
|--------|------|
| y = 1 | 通过 (passed) |
| y = 0 | 未通过 / 有风险 (failed / at-risk) |

模型直接预测上述二分类标签。

## 特征输入

使用 `common/feature_engineering.py` 中的 `build_feature_matrix()` 构建 **46 维特征矩阵**，包括:

1. **事件基础统计 (28维)**: 7种事件类型 x 4统计量 (均值/标准差/变异系数/香什农熵)
2. **行为轨迹 (10维)**: improvement、consistency、trend、时间间隔统计等
3. **情绪复合特征 (6维)**: edit_ratio、delete_ratio、focus_ratio 的均值和标准差
4. **元信息 (2维)**: 题目数量、总事件数

## 关键超参数

| 超参数 | 默认值 | 说明 |
|--------|--------|------|
| `n_estimators` | 100 | 决策树数量 |
| `max_depth` | 10 | 树的最大深度 |
| `random_state` | 42 | 随机种子 |
| `n_jobs` | -1 | 并行计算 (使用所有CPU核心) |

## 文件结构

```
models/rf/
├── __init__.py      # 模块初始化
├── model.py         # 模型定义 (create_model)
├── train.py         # 训练脚本 (5折交叉验证)
├── evaluate.py      # 评估脚本 (全量训练 + 交叉验证)
└── README.md        # 本文件
```

## 独立运行方式

### 训练

```bash
# 在项目根目录 (CodeEMO/) 下执行
python models/rf/train.py

# 自定义折数和输出目录
python models/rf/train.py --folds 10 --output-dir outputs/rf
```

训练结果将保存至 `outputs/rf/results.json`，包含每折的详细指标和汇总统计。

### 评估

```bash
# 在项目根目录 (CodeEMO/) 下执行
python models/rf/evaluate.py

# 自定义折数和输出目录
python models/rf/evaluate.py --folds 10 --output-dir outputs/rf
```

评估结果将保存至 `outputs/rf/evaluation.json`，包含训练集拟合表现、交叉验证指标以及特征重要性排名。

## 预期结果

基于 46 维特征和 5 折分层交叉验证，随机森林模型的典型性能如下 (实际数值可能因数据版本略有差异):

| 指标 | 均值 | 标准差 |
|------|------|--------|
| Accuracy | ~0.81 | ~0.03 |
| Precision | ~0.78 | ~0.04 |
| Recall | ~0.70 | ~0.05 |
| F1 Score | ~0.73 | ~0.04 |
| AUC | ~0.82 | ~0.03 |

## 依赖

- scikit-learn (RandomForestClassifier, StratifiedKFold, StandardScaler)
- numpy
- pandas
- 项目公共模块: `common.data_loader`, `common.feature_engineering`, `common.evaluator`
