# 双向LSTM模型 (Bidirectional LSTM)

## 算法简介

双向LSTM（Bidirectional Long Short-Term Memory）是一种序列建模方法，通过前向和后向两个方向的LSTM网络同时处理输入序列，从而捕捉双向的时序依赖关系。

在本项目中，我们将每个学生的46维特征向量视为单步序列输入，通过双向LSTM层提取前向和后向的特征表示，用于预测学生是否能通过课程（二分类任务：通过=1，不通过=0）。

### 模型架构

```
输入: (batch, 46)
  │
  ├── Linear(46 → 64)            特征嵌入
  │
  ├── unsqueeze → (batch, 1, 64) 添加序列维度
  │
  ├── BiLSTM(64, hidden=64, layers=2, bidirectional=True)
  │       前向LSTM (64维) + 后向LSTM (64维) = 128维
  │
  ├── 取最后一步输出 (128维)
  │
  ├── Dropout(0.3)
  │
  ├── Linear(128 → 64)
  │
  ├── ReLU
  │
  ├── Dropout(0.3)
  │
  ├── Linear(64 → 1)
  │
  └── Sigmoid → 概率输出 (0~1)
```

### 论文参考

双向LSTM是原始论文中使用的序列建模方法之一。该方法在学生编程行为分析领域被广泛应用，能够有效捕捉学生编程行为序列中的时序模式和双向依赖关系。

## 关键超参数

| 超参数 | 默认值 | 说明 |
|--------|--------|------|
| `input_dim` | 46 | 输入特征维度（46维特征向量） |
| `hidden_dim` | 64 | LSTM隐藏层维度 |
| `num_layers` | 2 | LSTM层数 |
| `dropout` | 0.3 | Dropout比率 |
| `epochs` | 100 | 最大训练轮数 |
| `batch_size` | 32 | 批大小 |
| `patience` | 10 | 早停耐心值 |
| `lr` | 1e-3 | Adam优化器学习率 |

## 文件结构

```
models/bilstm/
├── __init__.py      # 模块初始化
├── model.py         # BiLSTM模型定义
├── train.py         # 训练脚本 (5折交叉验证)
├── evaluate.py      # 评估脚本 (全量训练+5折交叉验证)
└── README.md        # 本文档
```

## 独立运行方式

### 训练模型

在项目根目录 `CodeEMO/` 下运行：

```bash
# 默认参数运行（5折交叉验证）
python models/bilstm/train.py

# 自定义参数
python models/bilstm/train.py --folds 10 --epochs 200 --batch-size 64

# 指定输出目录
python models/bilstm/train.py --output-dir outputs/bilstm_v2
```

训练结果将保存到 `outputs/bilstm/results.json`。

### 评估模型

```bash
# 默认参数运行（全量训练+5折交叉验证）
python models/bilstm/evaluate.py

# 自定义参数
python models/bilstm/evaluate.py --folds 10 --epochs 200
```

评估结果将保存到 `outputs/bilstm/evaluation.json`。

### 可选命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--folds` | int | 5 | 交叉验证折数 |
| `--output-dir` | str | `outputs/bilstm` | 结果输出目录 |
| `--epochs` | int | 100 | 最大训练轮数 |
| `--batch-size` | int | 32 | 批大小 |
| `--patience` | int | 10 | 早停耐心值 |
| `--lr` | float | 1e-3 | 学习率 |
| `--seed` | int | 42 | 随机种子 |

## 数据依赖

- 数据路径：`/tmp/IDE_logs/IDE_logs.csv` 和 `/tmp/IDE_logs/passed.csv`
- 特征维度：46维（由 `common/feature_engineering.py` 生成）
- 标签：二分类（1=通过，0=不通过）

## 预期结果

运行5折交叉验证后，BiLSTM模型预期输出以下指标（均值±标准差）：

| 指标 | 说明 |
|------|------|
| Accuracy | 准确率 |
| Precision | 精确率 |
| Recall | 召回率 |
| F1 Score | F1分数 |
| AUC | ROC曲线下面积 |

具体数值取决于数据分布和训练条件，结果将自动保存为JSON格式。

## 技术细节

- **特征标准化**：每折内独立使用 `StandardScaler` 进行标准化，避免数据泄露
- **早停策略**：当验证集损失连续 `patience` 轮不下降时停止训练
- **最优模型**：训练过程中保留验证集上表现最好的模型权重
- **损失函数**：BCELoss（二元交叉熵损失）
- **优化器**：Adam
