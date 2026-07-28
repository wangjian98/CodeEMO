# Pure MLP-46D:消融对照实验

> **核心结论(2026-07-28 消融实验)**:
> Pure MLP-46D(F1=0.8687)**甚至略超过** LSTM-MLP-46D(F1=0.8622),
> **远超过** LSTM-Seq-7D(F1=0.8062)。
> → 证明"赢的关键是 46d 信息密度",**不是 LSTM 的 gating**。

## 动机

之前两轮分析指出 **LSTM-46D 实际是 gated MLP**(seq_len=1),
并非真时序模型。这引发一个核心问题:

> **LSTM-46D 的胜出到底是因为「LSTM 的 gating」还是「46d 的信息密度」?**

要回答这个问题,需要构造一个**纯 MLP**(无 LSTM,无 attention,无任何时序/序列结构)
作为对照。如果纯 MLP ≈ LSTM-MLP-46D >> LSTM-Seq-7D,则**信息密度是主导因素**。

## 模型架构

```
输入 (B, 46) — 46d hand-crafted 统计特征
  ↓
Linear(46 → 64) + BatchNorm1d + ReLU + Dropout(0.3)
  ↓
Linear(64 → 64) + BatchNorm1d + ReLU + Dropout(0.3)
  ↓
Linear(64 → 1)
  ↓ Sigmoid
输出 (B,) — P(failed)
```

**参数量**: ≈ 7,841(46×64 + 64 + 64×64 + 64 + 64 + 1)

**关键**:没有任何 LSTM/attention/时序结构,纯 MLP。

## 与其他模型对比

| 模型 | F1 | AUC | 类型 | 时序? |
|------|-----|-----|------|-------|
| **LSTM-Seq-7D** | 0.8062 ± 0.020 | 0.7259 ± 0.052 | event sequence | ✅ 真时序(max_seq_len=500) |
| **LSTM-MLP-46D** | 0.8622 ± 0.028 | 0.9170 ± 0.023 | gated MLP(seq_len=1) | ❌ seq_len=1 |
| **Pure MLP-46D** | **0.8687 ± 0.026** | **0.9160 ± 0.017** | 纯 MLP | ❌ 无 LSTM |

**核心发现**:
- Pure MLP-46D ≈ LSTM-MLP-46D(AUC 几乎一致,F1 略高 +0.6 pp)
- Pure MLP-46D >> LSTM-Seq-7D(+6.3 pp F1, +19 pp AUC)
- **结论**:**LSTM 的 gating 对这个任务贡献微弱,46d 的信息密度才是关键**。

## 训练细节

| 项 | 值 |
|----|----|
| Optimizer | Adam(lr=1e-3, weight_decay=1e-4) |
| Epochs | 120 (max),patience=15 early stop |
| Batch size | 32 |
| Gradient clip | 1.0 |
| CV | 5-fold StratifiedKFold(random_state=42) |
| 设备 | GPU (CUDA) |
| 总耗时 | **8.3 秒**(n=473,纯 MLP 计算量极小) |

## 运行

```bash
cd ~/CodeEMO
python3 models/pure_mlp_46d/model.py
```

输出落到 `outputs/unified_compare/pure_mlp_46d/{probs.npy, labels.npy, fold_idx.npy, results.json}`。

## 对论文的影响

1. **实验表新增一行**:LSTM-MLP-46D 的赢面可完全归因于 46d 特征,非 LSTM gating。
2. **§5 Results / §7 Discussion**:应增加一段说明 "纯 MLP 即足够"——这是**对 BGM-Net 双分支设计的更强支持**(双分支用 MLP 已经够,不需要 LSTM 这种重型序列模型)。
3. **§7.4 Limitations** L5 可加强:**LSTM-46D 是 gated MLP 不是真时序;LSTM-Seq-7D 在 n=473 上输给纯 MLP**,建议未来工作**直接报告 MLP 而非"LSTM-on-46d"**。

## 文件

- `models/pure_mlp_46d/model.py` — 训练 + 评估(单一脚本,跑出 results.json)
- `outputs/unified_compare/pure_mlp_46d/results.json` — 实验结果(被 .gitignore 排除)
- 本 README — 说明与论文影响
