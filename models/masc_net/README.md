# MASC-Net: Multi-scale Adaptive Sample-aware Contrastive Network

## 概述

针对 CodeEMO 小样本(n=473) + 不平衡(1:2) + 折间零正样本问题设计的分类模型。

## 架构(5 模块)

1. **Multi-Scale Encoder (MSE)** — 三尺度编码 + 跨尺度注意力
   - Local: 1D-CNN (kernel=3,5) over 46-dim
   - Mid: 2-layer MLP on Cat1+Cat3
   - Global: 2-layer MLP on full 46-dim
   - Cross-Scale Multihead Attention 融合

2. **Sample-Aware Contrastive (SACM)** — MoCo 风格对比 + 自适应温度
   - Query/Key encoder (动量更新 m=0.999)
   - Adaptive τ 由样本自身决定 (τ ∈ [0.05, 0.5])
   - InfoNCE 损失

3. **Prototype Memory Bank (PMB)** — K=2 × M=4 个可学习原型
   - EMA 更新 (momentum=0.9)
   - 用于距离分类 + 难负样本挖掘

4. **Adaptive Threshold Classifier (ATC)**
   - 距离 = 1 - max_similarity(到该类最近原型)
   - 阈值从训练集 5%/95% 分位动态初始化
   - 输出概率 + 不确定性

5. **Uncertainty-Aware Loss Fusion**
   - Focal(α=0.7, γ=2.0) + 0.3·InfoNCE + 0.3·Proto + 0.05·Unc

## 实验结果 (CS1 5-fold CV)

| 变体 | Params | F1 | AUC |
|------|--------|----|-----|
| full (含对比+不确定性) | 25,475 | 0.7286±0.0416 | 0.8644±0.0207 |
| no_contrastive | 18,176 | 0.7517±0.0439 | 0.8479±0.0536 |
| no_uncertainty | 25,475 | 0.7467±0.0264 | 0.8624±0.0264 |
| **baseline_only (推荐)** | **18,176** | **0.7985±0.0416** | **0.8995±0.0203** |

vs. CodeEMO 已有最强 baseline:
- **超过 BGM-Net baseline (F1=0.7458) +5.27 pp**
- 仍低于 LSTM-46d (F1=0.8622) -6.37 pp, RF-7dim (F1=0.8876) -8.91 pp

## 关键发现

**对比学习 + 不性确定性模块在 n=473 上负收益(-7 pp F1)**, 与 BGM-Net 消融结论一致.
CodeEMO 数据集规模下, 简单模型(LSTM/RF) 归纳偏置仍最优; MASC-Net 设计可能在更大样本下展现优势.

## 文件结构

```
models/masc_net/
├── __init__.py
├── model.py       # MASCNet 架构 + FocalLoss
├── train.py       # 5-fold CV 训练入口
└── README.md      # 本文件

outputs/masc_net/
├── full_results.json           # 主实验
├── no_contrastive_results.json
├── no_uncertainty_results.json
├── baseline_only_results.json
├── *_probs.npy                 # 每折预测概率(可拼接用于集成)
├── *_uncs.npy                  # 不确定性输出
├── labels.npy
└── COMPARISON.md               # 完整对比表
```

## 复现

```bash
cd /home/ubuntu/CodeEMO
.venv/bin/python models/masc_net/train.py --folds 5 --epochs 120 --patience 20 \
    --output-dir outputs/masc_net --ablation full
```

消融见 `outputs/masc_net/COMPARISON.md` 第 5 节.
