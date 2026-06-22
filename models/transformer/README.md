# Transformer 学生早期风险预测模型

## 算法简介

本模块实现了一个基于 **Transformer编码器** 的二分类模型，用于学生课程通过/不通过的早期风险预测。模型的输入是由IDE编程日志提取的46维统计特征。

### 核心思路

传统的Transformer模型用于处理序列数据（如NLP中的文本序列）。本模型通过将46维特征向量**分割为多个段(segment)**，构造伪序列，从而使Transformer的自注意力机制能够建模不同特征组之间的交互关系。

### 特征分组策略

将46维特征切分为 **4个段**，每个段包含 **11维**（共使用44维，末尾2维截断）：

| 段编号 | 特征来源 | 维度范围 | 说明 |
|--------|----------|----------|------|
| 段1 | 事件统计 (前部分) | [0:11] | text_insert / text_remove 的事件统计 |
| 段2 | 事件统计 (后部分) | [11:22] | text_paste / focus_gained 等事件统计 |
| 段3 | 事件统计 + 轨迹 | [22:33] | 其余事件统计及行为轨迹特征 |
| 段4 | 轨迹 + 情绪 + 元信息 | [33:44] | 行为轨迹、情绪复合特征、元信息 |

每个段通过线性投影映射到 `d_model=64` 维，加入可学习的位置编码后送入Transformer编码器。

### 模型架构

```
输入: (batch, 46)
  │
  ├─ 截断为44维, 重塑为 (batch, 4, 11)  -- 4个段, 每段11维
  │
  ├─ Linear(11 → 64)                    -- 段投影, 每段映射到d_model
  │    → (batch, 4, 64)
  │
  ├─ + 位置编码 (可学习参数)
  │
  ├─ TransformerEncoder
  │    ├─ TransformerEncoderLayer x 3
  │    │    ├─ MultiHeadAttention (4 heads)
  │    │    ├─ FeedForward (64 → 128 → 64)
  │    │    └─ Dropout (0.2) + 残差连接 + LayerNorm
  │    └─ → (batch, 4, 64)
  │
  ├─ Mean Pooling (对序列维度取平均)
  │    → (batch, 64)
  │
  └─ 分类头
       ├─ Linear(64 → 32) + ReLU + Dropout(0.2)
       ├─ Linear(32 → 1)
       └─ Sigmoid → (batch, 1)  -- 输出通过概率 [0, 1]
```

## 关键超参数

| 超参数 | 默认值 | 说明 |
|--------|--------|------|
| `d_model` | 64 | Transformer内部隐藏维度 |
| `nhead` | 4 | 多头注意力头数 |
| `num_layers` | 3 | Transformer编码器层数 |
| `dim_feedforward` | 128 | 前馈网络中间维度 |
| `dropout` | 0.2 | Dropout比率 |
| `n_segments` | 4 | 特征分段数 |
| `segment_size` | 11 | 每段特征维度 (46 // 4) |
| `lr` | 1e-3 | Adam优化器学习率 |
| `weight_decay` | 1e-5 | L2正则化系数 |
| `epochs` | 100 | 最大训练轮数 |
| `batch_size` | 32 | 批大小 |
| `patience` | 10 | 早停耐心值 |

## 文件结构

```
models/transformer/
├── __init__.py       # 包初始化
├── model.py          # TransformerClassifier 模型定义
├── train.py          # 训练脚本 (5折交叉验证)
├── evaluate.py       # 评估脚本 (交叉验证 + 全量重训练)
└── README.md         # 本文件
```

## 独立运行方式

### 1. 训练

在项目根目录下执行：

```bash
python models/transformer/train.py
```

可选参数：

```bash
# 指定交叉验证折数
python models/transformer/train.py --folds 10

# 指定输出目录
python models/transformer/train.py --output-dir outputs/transformer

# 自定义训练参数
python models/transformer/train.py --epochs 200 --batch-size 64 --patience 15
```

训练结果保存至 `outputs/transformer/results.json`。

### 2. 评估

```bash
python models/transformer/evaluate.py
```

评估脚本会执行以下两个步骤：
1. **5折分层交叉验证**：获得泛化性能估计
2. **全量数据重训练**：在全部数据上训练最终模型，并报告训练集表现

评估结果保存至 `outputs/transformer/evaluation.json`。

### 3. 模型结构测试

单独测试模型结构和前向传播：

```bash
python models/transformer/model.py
```

## 依赖

- Python 3.8+
- PyTorch >= 1.10
- scikit-learn
- numpy / pandas
- 项目公共模块 (`common/` 目录)

## 预期结果

在5折分层交叉验证下，Transformer模型的典型表现：

| 指标 | 预期范围 | 说明 |
|------|----------|------|
| Accuracy | 0.70 ~ 0.85 | 整体分类准确率 |
| Precision | 0.70 ~ 0.85 | 预测为"通过"的准确率 |
| Recall | 0.70 ~ 0.90 | 实际通过的学生被正确识别的比例 |
| F1 Score | 0.72 ~ 0.85 | Precision与Recall的调和平均 |
| AUC | 0.75 ~ 0.90 | ROC曲线下面积 |

> **注意**：实际结果取决于数据集规模、正负样本比例及特征质量。由于Transformer在
> 小数据集上容易过拟合，早停机制和Dropout对于控制过拟合至关重要。

## 设计说明

1. **为什么用伪序列？** 46维特征本身不是时间序列，但通过分段构造伪序列可以让
   Transformer的自注意力机制自动学习不同特征组之间的关联（如事件统计与行为轨迹
   之间的关系），这比简单的全连接网络具有更强的特征交互能力。

2. **为什么选择4个段？** 46维特征来源于4个语义组（事件统计28维、轨迹10维、情绪6维、
   元信息2维）。将46维均匀切分为4段（每段11维），虽然不完全与语义组对齐，但保证
   了每个段具有相近的维度，有利于Transformer的稳定训练。

3. **均值池化 vs CLS Token：** 本模型使用均值池化聚合序列信息，而非CLS Token。
   均值池化在短序列（4个token）上通常更稳定，避免了CLS Token学习不充分的问题。
