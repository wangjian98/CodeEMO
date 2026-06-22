# Mamba-SSM: 选择性状态空间模型用于学生早期风险预测

## 算法简介

**Mamba** (Selective State Space Model, S6) 是一种基于选择性状态空间模型的序列建模方法,
由 Albert Gu 和 Tri Dao 于 2023 年提出。与 Transformer 的自注意力机制不同,
Mamba 通过**选择性状态空间机制**实现了线性时间的序列建模,
同时保持了与 Transformer 相当甚至更优的性能。

### 核心创新

Mamba 的核心创新在于 **S6 (Selective SSM) 机制**:

1. **选择性参数**: 状态空间模型的参数 (B, C) 从输入动态计算, 而非固定不变。
   这使得模型能够根据输入内容"选择性地"记忆或遗忘信息。

2. **硬件感知实现**: 通过并行选择性扫描算法高效实现, 适合现代 GPU 硬件。

3. **状态方程**:
   - 状态更新: `h_k = A * h_{k-1} + B * x_k`
   - 输出计算: `y_k = C * h_k`
   - 离散化: 参数 A, B, C 通过输入相关的门控机制动态调整

### 论文引用

```
Gu, A., & Dao, T. (2023).
"Mamba: Linear-Time Sequence Modeling with Selective State Spaces."
arXiv preprint arXiv:2312.00752.
```

---

## 6步流程说明

本项目将 Mamba-SSM 应用于学生早期风险预测, 采用完整的6步流程:

### Step 1: 数据预处理 - 7维事件编码

将每个学生的 IDE 日志事件序列编码为模型可用的张量格式。
每个事件编码为以下维度:

| 维度 | 说明 | 取值范围 |
|------|------|----------|
| `event_type` | 事件类型索引 | 0-6 (7类事件) |
| `time_interval` | 距上一事件的时间间隔 (log-normalized) | [0, 1] |
| `deadline_dist` | 距截止时间距离 (归一化) | [0, 1] |
| `part_id` | 题目部分编号 | 1-7 |

**事件类型 (7类)**:
| 索引 | 事件类型 | 说明 |
|------|----------|------|
| 0 | `focus_gained` | 获得焦点 |
| 1 | `focus_lost` | 失去焦点 |
| 2 | `text_insert` | 插入文本 |
| 3 | `text_remove` | 删除文本 |
| 4 | `text_paste` | 粘贴文本 |
| 5 | `run` | 运行代码 |
| 6 | `submit` | 提交答案 |

**标签约定**: `risk=0` 表示通过 (passed), `risk=1` 表示挂科 (at-risk/failed)。

### Step 2: Mamba 预训练 - 下一事件预测 (自监督)

使用**下一事件预测** (Next-Event Prediction) 任务进行自监督预训练:

- 输入: 前 n-1 个事件
- 目标: 预测第 n 个事件类型
- 损失函数: CrossEntropyLoss
- 优化器: AdamW (lr=1e-3, weight_decay=0.01)
- 梯度裁剪: max_norm=1.0

### Step 3: 多尺度特征提取

从 Mamba 编码器输出中提取多尺度特征表示:

**CPU版本 (SimplifiedMambaStudent)**:
- 全局均值: 整个序列的均值池化
- 最后状态: 序列最后一个时间步的隐藏状态
- 分部均值: 按 part 分组的均值池化
- 融合方式: 三种尺度直接平均

**GPU版本 (FullMambaStudent)**:
- 细粒度 (Fine): 每100事件窗口的均值, 捕捉短期行为模式
- 中粒度 (Medium): 全局均值, 捕捉整体行为特征
- 粗粒度 (Coarse): 按 part 分组均值, 捕捉跨部分学习模式
- 融合方式: 交叉注意力 (Cross-Attention) 融合 + 线性投影

### Step 4: 原型发现 - K-Means 聚类

对多尺度特征表示进行 K-Means 聚类 (n_clusters=4),
发现学生行为模式的原型:

- 原型间通过聚类中心距离区分不同行为模式
- 分析每个原型的风险比例 (at-risk ratio)
- 原型特征参与最终风险预测

### Step 5: 预测微调 - 5折交叉验证

在预训练权重基础上进行有监督微调:

- **冻结骨干网络**: 仅训练 `risk_head` 分类器
- **训练目标**: 风险分类 (2类: passed=0, at-risk=1)
- **交叉验证**: StratifiedKFold, 5折
- **每折训练**: 5 epochs
- **评估指标**: Accuracy, Precision, Recall, F1, AUC

### Step 6: 可解释性分析

从训练好的模型中提取可解释性信息:

1. **事件类型重要性**: 基于事件嵌入权重范数, 分析哪些事件类型对预测贡献最大
2. **时间模式分析**: 近期行为 vs 早期行为的比率, 识别"临时抱佛脚"模式
3. **原型分布**: 学生被分配到各行为原型的概率, 分析原型与风险的关联

---

## CPU vs GPU 版本对比

| 特性 | CPU版本 | GPU版本 |
|------|---------|---------|
| 模型类 | `SimplifiedMambaStudent` | `FullMambaStudent` |
| d_model | 32 | 64 |
| n_layers | 2 | 6 |
| d_state | 8 | 16 |
| max_seq_len | 2,000 | 2,000 |
| 预训练 epochs | 2 | 3 |
| 预训练 batch_size | 16 | 32 |
| 微调 batch_size | 16 | 16 |
| 多尺度融合 | 直接平均 | 交叉注意力 |
| 预计训练时间 | ~5-10分钟 | ~2-5分钟 |

---

## 文件结构

```
models/mamba/
  __init__.py            # 模块初始化
  model.py               # 核心模型 (S6Block, MambaEncoder, Simplified/FullMambaStudent)
  train_cpu.py           # CPU 完整训练流程
  train_gpu.py           # GPU 完整训练流程
  evaluate.py            # 独立评估脚本
  README.md              # 本文件
  steps/
    __init__.py
    step1_preprocessing.py   # Step 1: 数据预处理
    step2_pretrain.py        # Step 2: 自监督预训练
    step3_multiscale.py      # Step 3: 多尺度特征提取
    step4_prototype.py       # Step 4: K-Means 原型发现
    step5_finetune.py        # Step 5: 5折交叉验证微调
    step6_interpret.py       # Step 6: 可解释性分析
```

---

## 运行方法

### 环境要求

```
Python >= 3.8
PyTorch >= 2.0
scikit-learn
pandas
numpy
```

### 数据位置

```
/tmp/IDE_logs/IDE_logs.csv   # IDE日志 (student, part, exercise, eventType, timestamp, timeToDeadline)
/tmp/IDE_logs/passed.csv     # 标签 (student, passed)
```

### CPU 训练

```bash
# 默认5折交叉验证
python models/mamba/train_cpu.py

# 10折交叉验证, 自定义输出目录
python models/mamba/train_cpu.py --folds 10 --output-dir outputs/mamba_cpu

# 自定义预训练和微调轮数
python models/mamba/train_cpu.py --pretrain-epochs 3 --finetune-epochs 10
```

### GPU 训练

```bash
# 默认配置 (自动检测GPU)
python models/mamba/train_gpu.py

# 自定义配置
python models/mamba/train_gpu.py --folds 10 --pretrain-epochs 5 --pretrain-batch-size 64
```

### 独立评估

```bash
# 默认评估
python models/mamba/evaluate.py

# 指定设备
python models/mamba/evaluate.py --device gpu
python models/mamba/evaluate.py --device cpu
```

### 单独运行各步骤

```bash
# Step 1: 数据预处理
python models/mamba/steps/step1_preprocessing.py

# Step 2: 预训练
python models/mamba/steps/step2_pretrain.py

# Step 3: 多尺度特征提取
python models/mamba/steps/step3_multiscale.py

# Step 4: K-Means 原型发现
python models/mamba/steps/step4_prototype.py

# Step 5: 微调交叉验证
python models/mamba/steps/step5_finetune.py

# Step 6: 可解释性分析
python models/mamba/steps/step6_interpret.py
```

---

## 关键超参数

| 超参数 | CPU默认值 | GPU默认值 | 说明 |
|--------|-----------|-----------|------|
| `d_model` | 32 | 64 | 隐藏层维度 |
| `n_layers` | 2 | 6 | Mamba层数 |
| `d_state` | 8 | 16 | SSM状态维度 |
| `n_prototypes` | 4 | 4 | K-Means聚类数 |
| `max_seq_len` | 2,000 | 2,000 | 最大序列长度 |
| `pretrain_epochs` | 2 | 3 | 预训练轮数 |
| `finetune_epochs` | 5 | 5 | 微调轮数/折 |
| `batch_size (pretrain)` | 16 | 32 | 预训练批大小 |
| `batch_size (finetune)` | 16 | 16 | 微调批大小 |
| `learning_rate` | 1e-3 | 1e-3 | 学习率 |
| `weight_decay` | 0.01 | 0.01 | 权重衰减 |
| `grad_clip` | 1.0 | 1.0 | 梯度裁剪阈值 |
| `dropout` | 0.2 | 0.2 | Dropout率 |
| `n_folds` | 5 | 5 | 交叉验证折数 |
| `seed` | 42 | 42 | 随机种子 |

---

## 预期结果

基于典型的学生IDE日志数据集, 预期性能指标:

| 指标 | CPU版本 (预期) | GPU版本 (预期) |
|------|----------------|----------------|
| Accuracy | 0.70 - 0.85 | 0.75 - 0.90 |
| Precision | 0.65 - 0.85 | 0.70 - 0.88 |
| Recall | 0.60 - 0.80 | 0.65 - 0.85 |
| F1 Score | 0.65 - 0.80 | 0.70 - 0.85 |
| AUC | 0.70 - 0.85 | 0.75 - 0.90 |

> 注意: 实际结果取决于数据集规模、类别分布和数据质量。

---

## 标签说明

本项目使用以下标签约定:

| 标签值 | 含义 | 说明 |
|--------|------|------|
| `risk=0` | 通过 (passed) | 学生通过了课程/考试 |
| `risk=1` | 挂科 (at-risk/failed) | 学生未通过, 需要预警 |

`passed.csv` 中的 `passed` 列为布尔值: `True` 对应 `risk=0`, `False` 对应 `risk=1`。

`risk_head` 输出2个类别: `class 0 = passed`, `class 1 = at-risk`。

---

## 技术细节

### 选择性状态空间模型 (S6)

S6 的核心思想是从输入序列动态计算状态空间参数:

1. **输入投影**: 将输入投影到内部维度 `d_inner = expand * d_model`
2. **局部卷积**: 使用 depthwise 1D 卷积捕捉局部依赖
3. **选择性参数计算**: 从卷积输出动态计算 `dt`, `B`, `C` 参数
4. **离散化**: 通过 `dt` 对连续状态空间方程进行离散化
5. **选择性扫描**: 沿序列维度执行状态更新和输出计算
6. **输出投影**: 将 SSM 输出投影回原始维度

### 多尺度特征提取

**SimplifiedMambaStudent** 采用简化的3尺度融合:
```python
multi_scale_repr = (seq_mean + seq_last + part_repr) / 3
```

**FullMambaStudent** 采用增强的3尺度交叉注意力融合:
```python
# 细粒度: 每100事件窗口均值
# 中粒度: 全局均值
# 粗粒度: 按 part 分组均值
fused = scale_fusion(cat([fine_enhanced, medium, coarse]))
```

### 原型发现与风险预测

原型发现通过两个层级实现:

1. **无监督层 (Step 4)**: K-Means 聚类发现行为模式原型
2. **有监督层 (Step 5)**: 可学习原型中心 + softmax 权重分配

最终风险预测结合多尺度表示和原型表示:
```python
combined = cat([multi_scale_repr, proto_repr])
risk_pred = risk_head(combined)  # (batch, 2)
```
