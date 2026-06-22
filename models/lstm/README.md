# LSTM 单向模型

## 算法描述

本模块实现了单向LSTM（长短期记忆网络）分类器，用于**学生早期风险预测**任务。模型基于IDE编程日志提取的46维聚合特征，预测学生是否能通过课程。

### 模型架构

```
输入 (batch, 46)
  ↓ Linear(46 → 64)
  ↓ unsqueeze → (batch, 1, 64)
  ↓ LSTM(input=64, hidden=64, layers=2, batch_first=True)
  ↓ 取最后一层隐藏状态 → (batch, 128)
  ↓ Dropout(0.3)
  ↓ Linear(128 → 64)
  ↓ ReLU
  ↓ Dropout(0.3)
  ↓ Linear(64 → 1)
  ↓ Sigmoid
输出 (batch, 1) — 概率值
```

### 特征说明

输入为46维聚合特征，由 `common/feature_engineering.py` 生成:

| 特征组 | 维度 | 描述 |
|--------|------|------|
| 事件基础统计 | 28 | 7种事件类型 x 4统计量(均值/标准差/变异系数/香农熵) |
| 行为轨迹 | 10 | 改善趋势/一致性/间隔时间统计等 |
| 情绪复合特征 | 6 | 编辑比率/删除比率/专注比率 (均值+标准差) |
| 元信息 | 2 | 题目数量、总事件数 |

## 关键超参数

| 超参数 | 默认值 | 说明 |
|--------|--------|------|
| `hidden_dim` | 64 | LSTM隐藏层维度 |
| `num_layers` | 2 | LSTM层数 |
| `dropout` | 0.3 | Dropout比率 |
| `lr` | 0.001 | Adam优化器学习率 |
| `epochs` | 100 | 最大训练轮数 |
| `patience` | 10 | 早停耐心值 (基于验证F1) |
| `batch_size` | 32 | 批次大小 |
| `folds` | 5 | 交叉验证折数 |

## 文件结构

```
models/lstm/
├── __init__.py      # 模块初始化
├── model.py         # LSTM分类器模型定义
├── train.py         # 训练脚本 (5折交叉验证)
├── evaluate.py      # 评估脚本 (交叉验证 + 全量重训练)
└── README.md        # 本文件
```

## 独立运行方式

### 训练模型

```bash
# 从项目根目录运行
python models/lstm/train.py

# 自定义参数
python models/lstm/train.py --folds 5 --output-dir outputs/lstm
```

### 评估模型

```bash
# 从项目根目录运行
python models/lstm/evaluate.py

# 自定义参数
python models/lstm/evaluate.py --folds 5 --output-dir outputs/lstm
```

### 仅测试模型结构

```bash
python models/lstm/model.py
```

## 数据依赖

脚本运行时需要以下数据文件存在于 `/tmp/IDE_logs/` 目录:

- `IDE_logs.csv` — IDE编程日志数据
- `passed.csv` — 学生通过/未通过标签

数据加载由 `common/data_loader.py` 的 `load_ide_logs()` 完成。

## 输出结果

### 训练输出

- `outputs/lstm/results.json` — 包含交叉验证各折结果及汇总指标

### 评估输出

- `outputs/lstm/evaluation.json` — 包含交叉验证结果、全量数据自评估和详细报告

## 预期结果

基于46维聚合特征，LSTM单向模型的预期性能:

| 指标 | 预期值 |
|------|--------|
| Accuracy | ~0.80 |
| Precision | ~0.80 |
| Recall | ~0.80 |
| F1 Score | ~0.80 |
| AUC | ~0.85 |

> 注: 实际结果取决于数据集规模和分布，上述为参考值。

## 技术细节

- **特征标准化**: 使用 `StandardScaler` 对特征进行标准化，仅用训练集拟合
- **早停机制**: 基于验证集F1指标，连续10轮无提升则停止训练
- **设备自适应**: 自动检测并使用GPU (如有)，否则使用CPU
- **随机种子**: 固定随机种子为42，保证结果可复现
