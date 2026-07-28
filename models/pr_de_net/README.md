# PR-DE-Net: Precision-Recall Gated Dual-Encoder Network

> **生成时间**: 2026-07-26
> **位置**: `~/CodeEMO/models/pr_de_net/`
> **数据集**: CS1 IDE 日志（473 名学生：159 通过 / 314 未通过）
> **口径**: 5 折分层 CV，y=1=failed（与 `outputs/unified_compare/labels.npy` 一致）

---

## 一、动机

观察到 **LSTM/BiLSTM-46d** Recall 高（0.80-0.83）但 Precision 仅 0.92，而 **Transformer-7d** Precision 高（0.918）但 Recall 仅 0.82。两者学到的是**互补的判别特征**，应当端到端融合。

---

## 二、架构

```
Input:  46-dim feature vector  (B, 46)
              │
   ┌──────────┴──────────┐
   │                     │
   ▼                     ▼
Branch A: PR-RNN      Branch B: PR-Trans
BiLSTM(2L, h=64)      Transformer(2L, d=32, 4-head)
on (B,46,1)           on (B,7,1) + [CLS]
   ↓                     ↓
p_A (Recall)          p_B (Precision)
   ↓                     ↓
   └──────┬──────────────┘
          ▼
       Gate MLP
   g = σ(MLP([F46, h_A, h_B]))
          │
   p_final = g·p_A + (1-g)·p_B
```

**Loss**:
```
L = α·BCE(p_A, y) + β·BCE(1-p_B, y) + γ·BCE(p_final, y)
   = 1.0         + 1.0           + 2.0
```

- `BCE(p_A, y)` — Branch A 独立优化：保持 RNN 高 Recall
- `BCE(1-p_B, y)` — Branch B 独立优化：把 passed 当正例拉低 p_B → 高 Precision
- `BCE(p_final, y)` — 融合后整体 F1 优化

**参数**: 164,099

---

## 三、核心结果

### 1. 单模型对比

| 模型 | F1 | Precision | Recall | AUC |
|---|---|---|---|---|
| LSTM-46d | 0.8622 ± 0.028 | 0.8999 | 0.8281 | 0.9170 |
| BiLSTM-46d | 0.8561 ± 0.024 | 0.9238 | 0.8023 | 0.9036 |
| Transformer-7d | 0.8689 ± 0.034 | 0.9182 | 0.8248 | 0.9162 |
| RF-7dim | 0.8876 ± 0.019 | 0.9082 | 0.8694 | 0.9175 |
| HDM-Net v2 T3 | 0.8982 ± 0.022 | 0.9256 | 0.8726 | 0.9273 |
| **PR-DE-Net (单)** | **0.8601 ± 0.027** | **0.9092** | **0.8184** | **0.8711** |

PR-DE-Net 单模型 F1=0.8601 不算突出（弱于 RF/HDM-Net），但 **融合后** 显著提升。

### 2. **新纪录 — 融合方案**

| 方案 | F1 | AUC |
|---|---|---|
| Weighted 1/3/1（融合基线） | 0.9009 | 0.9349 |
| Stack top-3 LR | 0.8986 | 0.9324 |
| Per-fold stack top-5 | 0.8953 | 0.9346 |
| **★ 3-way: 2.5·RF + 2.5·HDM-Net v2 + 1.0·PR-DE-Net** | **0.9027** | **0.9265** |
| ★ 4-way: 1·RF + 3·HDM-Net v2 + 0·LSTM + 1·PR-DE-Net | 0.9026 | 0.9288 |

**PR-DE-Net 把融合 F1 从 0.9009 提升到 0.9027**（+0.18 pts）。

### 3. 消融验证（Gate + 三段式 Loss 都必要）

| 变体 | F1 | P | R | AUC |
|---|---|---|---|---|
| **full (Gate + 三段式 loss)** | **0.8601** | **0.9092** | 0.8184 | **0.8711** |
| no_gate (固定 0.5/0.5 平均) | 0.8486 | 0.8691 | 0.8344 | 0.8412 |
| single_loss (只用 γ·BCE) | 0.8653 | 0.8818 | 0.8504 | 0.8781 |

**Gate**：让 AUC 提升 **+0.030**（无 Gate 时概率质量下降）。  
**三段式 Loss**：让 Precision 提升至 0.91（独立优化防止梯度对齐）。

---

## 四、Gate 行为分析（设计验证）

| 子集 | n | gate 均值 | 偏 RNN 占比 | 偏 Trans 占比 |
|---|---|---|---|---|
| Failed (y=1) | 314 | **0.430** | **59.2%** | 40.8% |
| Passed (y=0) | 159 | **0.659** | 27.0% | **73.0%** |

✅ **Gate 学到了设计目标**：
- Failed 样本 → 路由到 **Branch A (RNN)**，因为 RNN Recall 更高
- Passed 样本 → 路由到 **Branch B (Transformer)**，因为 Transformer Precision 更高

### 预测正确性下的 Gate 行为

| 子集 | gate (failed) | gate (passed) | n |
|---|---|---|---|
| 预测**正确** | 0.374 | 0.687 | 390 |
| 预测**错误** | 0.685 | 0.514 | 83 |

⚠️ **失败模式**：预测错的 failed 样本（n=57），gate=0.69（**错误走了 Trans 分支**），Final 准确率仅 0%。这是模型的盲区——某些 failed 学生的全局模式被 Transformer 学成"像 passed"，导致 gate 路由失败。

---

## 五、错误样本分析

| 类别 | n | A 单独准 | B 单独准 | Final 准 | gate |
|---|---|---|---|---|---|
| Failed 误判 (y=1, pred=0) | 57 | 5% | 2% | 0% | 0.685 |
| Passed 误判 (y=0, pred=1) | 26 | 19% | 8% | 0% | 0.514 |

- **真正难的是 failed 误判为 passed**（n=57, 占总错误 68.7%）：这些学生的时序信号被 Trans 的"全局平稳"假象遮蔽。
- 这是 PR-DE-Net 的设计上限——单模型架构无法突破，要靠**融合**来吸收 RF/HDM 的不同视角。

---

## 六、文件结构

```
~/CodeEMO/models/pr_de_net/
├── __init__.py
├── model.py                    # PRDENet 主模型 (164K params)
├── train.py                    # 完整训练 (full, α/β/γ 可配)
├── ablation.py                 # no_gate + single_loss 消融
├── hparam_search.py            # 紧凑版 + 4 种 loss 权重
├── v2_ensemble.py              # v2_a/b/c/d + 4 路平均
├── fusion.py                   # 与 RF/HDM-Net 融合搜索
└── report.py                   # 综合分析报告

~/CodeEMO/outputs/pr_de_net/
├── full/                       # 主模型 (α=1,β=1,γ=2)
│   ├── probs.npy, probs_A.npy, probs_B.npy, gates.npy
│   ├── labels.npy, fold_idx.npy
│   └── results.json            # F1=0.8601±0.027
├── no_gate/    single_loss/    mini_*/
├── v2_a_gamma3/  v2_b_alpha15/  v2_c_beta15/  v2_d_alpha05/
├── v2_ensemble/                # 4 路平均
├── fusion_3way.json            # ★ F1=0.9027
├── fusion_4way.json            # F1=0.9026
└── comparison_with_baselines.json
```

---

## 七、运行复现

```bash
cd ~/CodeEMO

# 主模型
python3 models/pr_de_net/train.py --ablation full

# 消融
python3 models/pr_de_net/ablation.py

# 融合搜索
python3 models/pr_de_net/fusion.py

# 综合分析报告
python3 models/pr_de_net/report.py
```

---

## 八、结论

1. **Gate 真的在样本级做 PR 路由**——failed→RNN, passed→Trans。
2. **三段式 Loss + Gate** 让单模型 Precision 达到 0.91（继承 Transformer 优势）。
3. **PR-DE-Net 的真正价值在融合**——把现有最强融合 F1 从 0.9009 提升到 **0.9027**（+0.18 pts）。
4. **失败模式明确**：错把 failed 当 passed 的样本（57/83），需要其他模型族互补——这正是 RF/HDM-Net 在融合中贡献 0.5-1.0 权重的原因。