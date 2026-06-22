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
│
├── outputs/                     # 实验输出
│   ├── rf/
│   ├── lstm/
│   ├── bilstm/
│   ├── transformer/
│   ├── mamba/
│   ├── comparison.csv           # 全模型对比
│   └── analysis.md              # 分析报告
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

## 模型简介

| 模型 | 类型 | 描述 |
|------|------|------|
| [Random Forest](models/rf/README.md) | 传统ML | sklearn RandomForestClassifier, 46维特征 |
| [LSTM](models/lstm/README.md) | 深度学习 | 单向LSTM, 46维特征 → 序列建模 |
| [BiLSTM](models/bilstm/README.md) | 深度学习 | 双向LSTM (原论文方法), 46维特征 |
| [Transformer](models/transformer/README.md) | 深度学习 | Transformer编码器, 46维特征分组为伪序列 |
| [Mamba](models/mamba/README.md) | 前沿 | Selective State Space Model, 7维事件序列, 6步流程 |

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

运行 `python main.py --model all` 后，结果保存至 `outputs/comparison.csv` 和 `outputs/analysis.md`。

| 模型 | Accuracy | F1 | AUC |
|------|----------|----|-----|
| RF | ~0.81 | ~0.73 | ~0.91 |
| LSTM | ~0.80 | ~0.72 | ~0.89 |
| BiLSTM | ~0.82 | ~0.75 | ~0.91 |
| Transformer | ~0.81 | ~0.74 | ~0.90 |
| Mamba (CPU) | ~0.83 | ~0.77 | ~0.92 |

> 以上为预期结果范围，实际结果取决于数据和环境。

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
