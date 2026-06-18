# CodeEMO: 融合编程行为情绪表征的学业早期风险预测

基于论文《融合编程行为情绪表征的学业早期风险预测》实现的完整项目

## 项目结构

```
CodeEMO/
├── README.md
├── requirements.txt
├── scripts/
│   ├── data_processing.py    # 数据预处理
│   └── feature_engineering.py # 特征工程
├── models/
│   ├── simple_nn.py          # SimpleNN轻量神经网络
│   ├── bi_lstm.py            # Bi-LSTM模型
│   ├── bayesian_lstm.py      # Bayesian LSTM模型
│   └── random_forest.py      # 随机森林模型
├── train.py                  # 训练脚本
├── experiment.py             # 实验脚本（消融实验、SOTA对比）
└── outputs/                  # 实验结果输出目录
```

## 特征工程（46维）

1. **事件基础统计（28维）**：对7种事件类型各计算均值、标准差、变异系数、香农熵
2. **行为轨迹（10维）**：improvement、consistency、trend等
3. **情绪复合特征（6维）**：edit_ratio_mean/std、delete_ratio_mean/std、focus_ratio_mean/std
4. **元信息（2维）**：题目数量、总事件数

## 模型

- **SimpleNN**：轻量神经网络（1.5K参数）
- **Bi-LSTM**：序列方法
- **Bayesian LSTM**：原论文方法
- **Random Forest**：传统机器学习基线

## 实验

- 5折交叉验证
- 消融实验：7维 vs 38维 vs 46维特征
- SOTA对比：25%和100%数据量
