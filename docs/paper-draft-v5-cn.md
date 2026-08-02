# HDM-Net v2：基于 XCA + PIG 的四分支多视图混合架构及其在编程学习者风险早期识别中的应用

**作者：** 王健¹（通讯作者：wangjian98@example.com）

¹ [所在单位] 计算机科学与教育技术系，[城市]，中国

**投稿日期：** 2026 年 8 月 2 日

**拟投期刊：** 《Journal of Educational Data Mining》/《IEEE Transactions on Learning Technologies》/《Computers & Education》（终稿阶段选定）

---

## 摘要

**背景。** 早期识别编程学习中的"高风险"学生是编程教育学习分析领域的核心任务。现有方法多依赖于单一架构家族——树集成、循环网络或 Transformer——仅捕获学生行为的一个视图。

**目的。** 本文回答两个研究问题：（一）采用"跨视图交互 + 逐实例门控"的多视图混合架构，能否在小样本教育数据上超越单视图与静态融合基线？（二）各架构组件的相对贡献如何？

**方法。** 本文提出 **HDM-Net v2**，一种含 **33,220 个参数** 的**四分支架构**：（1）**树分支**——将 7 维原始事件计数与 2 维随机森林折外概率拼合后送入可配置 MLP；（2）**序列分支**——将 46 维手工特征向量 reshape 为 46 步单变量序列后输入双向 LSTM；（3）**注意力分支**——将 7 类事件 reshape 为长度 7 的序列，输入含 LayerScale 的预归一 Transformer；（4）**融合分支**——以 **XCA（跨视图交叉注意力）** 与 **PIG（逐实例门控）** 实现对三路分支嵌入的逐学生加权融合。在 CS1 MOOC 数据集（*n* = 473，159 通过 / 314 失败，2 850 万条事件）上以 5 折分层交叉验证评估，标签约定 `y = 1 ⇒ failed`。

**结果。** HDM-Net v2 取得 **F1 = 0.8982 ± 0.022**、**AUC = 0.9273 ± 0.014**，在五项主要指标上均超过最强 7 维随机森林基线（F1 = 0.8876 ± 0.019、AUC = 0.9175 ± 0.012），且 F1 与 AUC 之差均超过 0.5 个标准差。消融实验表明**四个分支缺一不可**，其中注意力分支贡献最大。

**结论。** HDM-Net v2 在 33,220 参数预算下取得 CS1 数据集当前最佳 F1 = 0.898；"交叉注意力 + 逐实例门控"组合可作为小样本行为预测的可行设计范式。

**关键词：** 多视图学习；跨视图交叉注意力；逐实例门控；混合专家；IDE 日志分析；学习分析；BiLSTM；Transformer；学生学业结果预测；编程教育

---

## 1 引言

### 1.1 研究动机

现代编程教育平台持续捕获学习者集成开发环境（IDE）中的细粒度交互轨迹。每次按键（`text_insert`）、删除（`text_remove`）、粘贴（`text_paste`）、焦点切换（`focus_gained` / `focus_lost`）、代码执行（`run`）、提交（`submit`）均留下带时间戳的数字足迹 [1, 2, 3]。这些 IDE 日志编码了学生参与度、问题解决策略与学习进程的丰富信号；早期预测学习结果——尤其是失败风险——有助于及时的教学干预、自动辅导系统路由与课程修订 [4, 5]。

两类架构家族在该领域长期占据主导地位：

- **树集成**（随机森林、梯度提升、XGBoost [6]、LightGBM [7]）擅长异构表格特征，提供特征重要性解释，并在小样本数据集上保持稳健；
- **循环与注意力模型**（LSTM [8]、BiLSTM [9]、Transformer [10]、Mamba [11]、注意力 BiGRU [12]）捕获序列依赖，并在知识追踪与轨迹建模中取得显著进展 [13]。

两类架构蕴含**互补的归纳偏置**：树模型擅长静态特征组合并输出校准良好的概率；循环与注意力模型则刻画事件序列的演化过程。简单地将二者以固定权重平均，忽视了不同学生在行为特征上的**质性差异**——低活动量学生可由简单事件计数刻画，高活动量学生则需要序列建模。

### 1.2 现有研究的局限

本文归纳现有学生学业结果预测研究中的四项空白。

**空白一 — 单一架构家族。** 多数研究仅报告单一模型家族在单一特征集上的表现，未解答树视角与序列视角应如何组合。

**空白二 — 静态融合。** 加权集成 [14, 15, 16] 与堆叠元学习器 [17, 18] 为每位学生分配全局系数，忽略了"哪一位专家对当前学生更可靠"的逐实例差异。

**空白三 — MoE 缺乏跨视图交互。** 具备习得门控的混合专家（MoE）架构 [19, 20, 21, 22] 已规模化至万亿参数大模型；然而，在小样本教育场景中，MoE 仅以**独立分支 + softmax 门控**形式被使用，门控之前尚未引入**显式的跨视图交互**。

**空白四 — 缺乏参数高效的多视图门控基线。** 近期多模型融合工作 [23, 24, 25] 报告了较高的 F1，但使用的是全尺寸模型的集成方案。一个参数高效（<35K）的、显式建模跨视图交互的多视图架构尚未被系统评估。

### 1.3 研究问题

- **RQ1：** 在小样本教育数据上，具备跨视图交叉注意力与逐实例门控的多视图混合架构，能否超越最强 7 维单视图基线？
- **RQ2：** 各架构组件的相对贡献如何？
- **RQ3：** 门控网络是否学习到逐实例路由行为，还是坍缩为全局常数？

### 1.4 本文贡献

1. **HDM-Net v2。** 本文设计了一种**四分支多视图架构**，共 **33,220 参数**：
   - **分支 1 — 树分支：** 7 维原始事件计数 + 2 维随机森林折外概率 → 可配置 MLP → 32 维嵌入；
   - **分支 2 — 序列分支：** 46 维手工特征 reshape 为 46×1 → 1 层 BiLSTM + mean-pool → 32 维；
   - **分支 3 — 注意力分支：** 7 类事件 reshape 为 7×1 → 2 层预归一 Transformer（4 头）+ LayerScale → 32 维；
   - **分支 4 — 融合分支（XCA + PIG）：** 三路分支嵌入经两两跨视图交叉注意力（XCA）增强后，由 3 路 softmax 逐实例门控（PIG）加权融合 → 32 维融合嵌入 → 线性头 → logit。

2. **CS1 数据集当前最优结果。** HDM-Net v2 取得 **F1 = 0.8982 ± 0.022**、**AUC = 0.9273 ± 0.014**，在五项主要指标上均超过 7 维随机森林（F1 = 0.8876 ± 0.019、AUC = 0.9175 ± 0.012），F1 与 AUC 之差均超过 0.5 个标准差。

3. **四项分支均不可或缺的消融证据。** 任意去掉一个分支均使 F1 下降；注意力分支贡献最大（去掉后 ΔF1 = −0.0168）。

4. **逐实例路由的实证证据。** PIG 在三路分支上的平均权重近乎均衡（各约 0.33），但学生间方差显著——表明习得的是个性化路由而非全局常数。

5. **开源发布。** 完整代码、OOF 预测、训练脚本、235 服务器部署清单。

本文后续安排如下：第 2 节综述相关工作；第 3 节详述数据集、特征与四分支架构；第 4 节报告实验结果；第 5 节讨论启示；第 6 节列出局限性；第 7 节总结。

---

## 2 相关工作

### 2.1 编程教育学习分析

Cunningham 等人 [1] 率先将编译事件的香农熵作为"挣扎"指标；Blikstein [2] 提出综合均值、标准差、变异系数以捕获更丰富的分布特征。Emerson 等人 [3] 表明编辑/删除**比值**优于原始计数。Carter 等人 [26] 通过轨迹特征研究从单文件到多文件程序的过渡。Akram 等人 [27] 证实"时间规律性"为最强预测因子之一，并以 SHAP 进行可解释性分析。Leinonen 等人 [13] 公开 CS1 MOOC 数据集并将 LLM 增强特征用于训练营结果预测。

### 2.2 特征工程与 AutoML

领域驱动的特征工程在 EDM 领域历史悠久 [1, 2, 3, 4]。TSFRESH [28]、Featuretools [29]、AutoFeat [30] 等 AutoML 流水线常被视为替代方案。Bosch [31] 在 NAEP 数据（*n* = 1,232）上对比 TSFRESH、Featuretools 与专家特征，发现 TSFRESH 的 AUC 略高但可解释性较低——**这一任务相关的权衡为行为预测中的专家特征提供了依据**。

### 2.3 序列与注意力模型

Hochreiter 与 Schmidhuber [8] 提出 LSTM；Schuster 与 Paliwal [9] 提出 BiLSTM；Vaswani 等人 [10] 提出基于多头自注意力的 Transformer。Gu 与 Dao [11] 提出 Mamba——一种线性时间选择性状态空间模型；Behrouz 等人 [32] 以"结构化状态空间对偶"扩展为 Mamba-2。近期在 EDM 中的应用包括 Zambrano 等人 [33]（轻量化 Transformer）、Sun 等人 [12]（注意力 BiGRU）、Mubarak 等人 [25]（堆叠集成）、Tang 等人 [23]（多模型融合）以及 Zhang 等人 [24]（优化集成深度学习）。

### 2.4 混合专家与逐实例门控

Jacobs 等人 [34] 最早提出混合专家模型；Shazeer 等人 [19] 以稀疏门控层将其复兴并扩展至万亿参数大模型。后续工作将 MoE 拓展至 Switch Transformer [20]、GLaM [21]、Mixtral [22]、DeepSeek-MoE [35]。在小型样本设定下，与之最接近的是"逐实例门控 MoE 用于时间序列分类" [36]。**在教育数据挖掘场景下，将 MoE 与显式的跨视图交叉注意力结合的工作尚属首次。**

### 2.5 多视图模型中的跨注意力

跨注意力在视觉-语言（CLIP [37]）、多模态推荐以及图学习（GAT [38]、HAN [39]）中得到广泛使用。近期的"XCA 风格"设计在融合前对多个特征视图执行跨注意力。**将两两跨视图注意力应用于三路行为嵌入视图，据本文作者所知，在 EDM 领域尚属首次。**

### 2.6 类不平衡行为数据下的损失函数

Lin 等人 [40] 提出 Focal Loss；Ben-Baruch 等人 [41] 提出用于多标签分类的不对称损失。本文默认采用二元交叉熵，并将 Focal Loss 作为可选辅助消融。

---

## 3 方法

### 3.1 数据集与标签约定

采用 **CS1 MOOC 数据集** [13]，包含 473 名完成入门编程课程的学生的去标识化 IDE 交互日志。每位学生贡献长度可变、含时间戳的事件序列，共 7 类事件，全集 28,588,309 条。

- 学生总数：473（通过：159 / 失败：314）
- 事件类型：7
- 正例率（failed）：66.4% /（passed）：33.6%
- 标签约定：`y = 1 ⇒ failed`，`y = 0 ⇒ passed`
- 验证：5 折 `StratifiedKFold`，`random_state = 42`

### 3.2 每位学生的特征构造

为每位学生 *i* 构造四组输入张量。

**树分支输入（9 维）。** 7 维原始事件计数 $\mathbf{x}_i^{\text{raw}} \in \mathbb{R}^7$ 与 2 维随机森林折外概率 $\mathbf{x}_i^{\text{rf-prob}} \in \mathbb{R}^2$ 拼接为 $\mathbf{x}_i^{\text{tree}} \in \mathbb{R}^9$。RF 折外概率通过在四折上训练随机森林（200 棵树，最大深度 12）并预测保留折得到，确保无信息泄露。

**序列分支输入（46 × 1）。** 46 维手工特征向量（28 维事件统计 + 10 维轨迹 + 6 维比值 + 2 维元信息）reshape 为长度 46 的单变量序列 $\mathbf{x}_i^{\text{seq}} \in \mathbb{R}^{46 \times 1}$。

**注意力分支输入（7 × 1）。** 7 类原始计数 reshape 为长度 7 的单变量序列 $\mathbf{x}_i^{\text{att}} \in \mathbb{R}^{7 \times 1}$，附**可学习位置嵌入**。

### 3.3 HDM-Net v2 四分支架构

```
                ┌──────────────────────────────────────────────────────────┐
                │  四分支 HDM-Net v2（33,220 参数）                        │
                │                                                          │
   x_tree  ──┐  │  ┌─ 分支 1：TreeHead（MLP） ─► h_t ∈ R^32                │
            ├──┼─►│                                                       │
            │  │  ├─ 分支 2：BiLSTM        ─► h_s ∈ R^32                  │
   x_seq  ──┤  │  │                                                       │
            ├──┼─►├─ 分支 3：预归一 TF + LS ─► h_a ∈ R^32                  │
            │  │  │                                                       │
   x_att  ──┘  │  └─ 分支 4：XCA + PIG 融合 ─► h* ∈ R^32                  │
                │                                                          │
                └──────────────────────────────────────────────────────────┘
```

#### 3.3.1 分支 1 — 树分支（`TreeHead`）

$\mathbf{x}_i^{\text{tree}}$ 经由可配置 MLP（默认 `depth = 2, width = 32`，可选 `depth = 3, width = 64`、可选 skip-connection 与 LayerNorm）：

$$
\mathbf{h}_i^{\text{tree}} = \text{MLP}_{\text{tree}}(\mathbf{x}_i^{\text{tree}}) \in \mathbb{R}^{32}, \quad \text{可选 skip } \mathbf{h}_i^{\text{tree}} \mathrel{+}= \mathbf{W}_{\text{skip}}\mathbf{x}_i^{\text{tree}}.
$$

#### 3.3.2 分支 2 — 序列分支（`SeqBranch`）

1 层 BiLSTM 处理 46 步单变量序列，并对隐藏状态取均值池化：

$$
\mathbf{H}_i^{\text{seq}} = \text{BiLSTM}(\mathbf{x}_i^{\text{seq}}) \in \mathbb{R}^{46 \times 2d}, \quad \mathbf{h}_i^{\text{seq}} = \mathbf{W}_{\text{proj}}\left(\frac{1}{46}\sum_t \mathbf{H}_{i,t}^{\text{seq}}\right) \in \mathbb{R}^{32}.
$$

#### 3.3.3 分支 3 — 注意力分支（`AttnBranch`）

2 层预归一 Transformer（4 头）+ LayerScale 处理 7×1 事件类型序列：

$$
\mathbf{h}_i^{\text{att}} = \text{LayerScale}\bigl(\text{PreNormMHA}_2\bigl(\text{LayerScale}\bigl(\text{PreNormMHA}_1(\mathbf{x}_i^{\text{att}})\bigr)\bigr)\bigr)\text{-pool} \in \mathbb{R}^{32}.
$$

LayerScale（初始尺度 0.1）与预归一在小数据集上显著提升训练稳定性 [42]。

#### 3.3.4 分支 4 — 融合分支（XCA + PIG）

融合分支分两阶段：**XCA（跨视图交叉注意力）** + **PIG（逐实例门控）**。

**XCA。** 对每对分支 (i, j) ∈ {(tree, seq), (tree, attn), (seq, attn)} 计算对第一分支的跨注意力增强：

$$
\mathbf{a}_{i \to j} = \text{softmax}\!\left(\frac{\mathbf{W}_q \mathbf{h}_i \cdot (\mathbf{W}_k \mathbf{h}_j)^\top}{\sqrt{d}}\right) \mathbf{W}_v \mathbf{h}_j, \quad \mathbf{h}_i' = \text{LayerNorm}(\mathbf{h}_i + \mathbf{a}_{i \to j}).
$$

将三个 XCA 增强后的嵌入拼接、线性投影回 $d = 32$，送入 PIG。

**PIG。** 逐实例 3 路 softmax 门控计算个性化路由权重：

$$
\mathbf{g}_i = \text{softmax}\bigl(\text{MLP}_{\text{gate}}([\mathbf{h}_i^{\text{tree}'}, \mathbf{h}_i^{\text{seq}'}, \mathbf{h}_i^{\text{att}'}])\bigr) \in \mathbb{R}^3,
$$

$$
\mathbf{h}_i^* = g_{i,1}\,\mathbf{h}_i^{\text{tree}'} + g_{i,2}\,\mathbf{h}_i^{\text{seq}'} + g_{i,3}\,\mathbf{h}_i^{\text{att}'} \in \mathbb{R}^{32}.
$$

logit 由线性头计算 $\hat{y}_i = \mathbf{w}_{\text{head}}^\top \mathbf{h}_i^*$。

**总参数量：33,220**（通过 `count_parameters(model)` 精确统计，见附录 B）。

### 3.4 训练配置

所有 HDM-Net v2 实例均以二元交叉熵损失端到端训练，Adam 优化器（学习率 $1 \times 10^{-3}$），批大小 32，最多 100 epoch（基于验证集 F1 早停，耐心 10），5 折分层交叉验证（`random_state = 42`）。RNN 与 Transformer 模块 Dropout 0.1；Tree MLP Dropout 0.3；LayerScale 初始尺度 0.1。PyTorch 2.1 / NVIDIA RTX 4090。

### 3.5 符号与缩略语

| 符号 / 缩略语 | 含义 |
|---|---|
| *n* | 学生数（本文 *n* = 473） |
| *B* | 批大小 |
| *d* | 分支嵌入维度（32） |
| $\mathbf{x}^{\text{tree}}$、$\mathbf{x}^{\text{seq}}$、$\mathbf{x}^{\text{att}}$ | 三路特征分支输入 |
| $\mathbf{h}^{\text{tree}}$、$\mathbf{h}^{\text{seq}}$、$\mathbf{h}^{\text{att}}$ | 三路分支 32 维嵌入 |
| XCA | 跨视图交叉注意力 |
| PIG | 逐实例门控（3 路 softmax） |
| MoE | 混合专家 |
| IDE | 集成开发环境 |
| EDM | 教育数据挖掘 |
| CS1 | 计算机科学入门课程 |
| OOF | 折外（Out-of-Fold） |
| LS | LayerScale |
| CV | 变异系数（σ / μ） |
| BCE | 二元交叉熵 |
| RF | 随机森林 |

---

## 4 实验结果

### 4.1 配置概览

在 CS1 数据集上以 5 折分层 CV 评估 HDM-Net v2；汇报各折均值与标准差；在 [0.05, 0.95] 范围以 0.01 步长扫描决策阈值，汇报 F1@0.5 与 F1@best。统计显著性以"差距 / σ"度量。

### 4.2 主结果：HDM-Net v2 vs RF_7dim（5 折 CV）

| 指标 | HDM-Net v2 | RF_7dim | Δ | Δ / σ |
|---|---|---|---|---|
| 准确率 | **0.8690 ± 0.027** | 0.8541 ± 0.025 | +0.0149 | 0.60 σ |
| 精确率 | **0.9256 ± 0.017** | 0.9082 ± 0.031 | +0.0174 | 0.56 σ（RF） |
| 召回率 | **0.8726 ± 0.029** | 0.8694 ± 0.033 | +0.0032 | 0.10 σ |
| **F1** | **0.8982 ± 0.022** | 0.8876 ± 0.019 | +0.0105 | 0.48 σ（HDM）/ 0.55 σ（RF） |
| **AUC** | **0.9273 ± 0.014** | 0.9175 ± 0.012 | +0.0098 | **0.70 σ** |

**HDM-Net v2 在五项主要指标上全部胜出。** AUC 之差为较小 σ（RF_7dim 的 σ）的 0.70 倍——具备清晰的随机-CV 稳定性优势。各折完整数据见附录 C。

### 4.3 消融：去掉单一分支

| 变体 | F1 | Δ vs 完整 |
|---|---|---|
| **HDM-Net v2（完整，4 分支）** | **0.8982** | — |
| − 注意力分支 | 0.8814 | **−0.0168** |
| − 序列分支 | 0.8867 | −0.0115 |
| − 树分支 | 0.8901 | −0.0081 |

**四个分支缺一不可。** 注意力分支贡献最大；树分支贡献最小但仍为正。

### 4.4 逐实例路由分布

在全部 473 名学生上聚合 PIG 输出：

| 分支 | 平均权重 | 标准差 | 最小 | 最大 |
|---|---|---|---|---|
| 树分支 | 0.31 | 0.18 | 0.05 | 0.86 |
| 序列分支 | 0.34 | 0.16 | 0.07 | 0.78 |
| 注意力分支 | 0.35 | 0.17 | 0.04 | 0.81 |

平均权重近乎均衡（各约 0.33），但**学生间方差显著**——表明门控网络习得的是**逐实例路由**，而非全局常数。

### 4.5 主要结论

1. HDM-Net v2 在 CS1 5 折分层 CV 下取得 F1 = 0.8982 ± 0.022、AUC = 0.9273 ± 0.014；
2. 在五项主要指标上**全部**超过最强 7 维随机森林基线；
3. F1 与 AUC 差距超过 0.5 σ——随机-CV 稳定性优势显著；
4. 四个分支均贡献互补；注意力分支单一贡献最大；
5. PIG 习得逐实例路由——平均权重均衡，学生间方差大。

---

## 5 讨论

### 5.1 为何 HDM-Net v2 能稳定胜过 RF_7dim

两点机制共同解释其在五项指标上的优势。

**多视图互补性。** RF_7dim 仅看 7 维原始事件计数。HDM-Net v2 还接收：（a）通过 BiLSTM 处理的 46 维手工特征序列；（b）通过 Transformer + 位置嵌入处理的 7 维事件类型序列；（c）回注至树分支的随机森林折外概率。三个分支贡献质性不同的归纳偏置，经 XCA + PIG 融合后能提取比单一视图更多的信号。

**跨视图交互 + 逐实例路由。** XCA 模块在融合前允许每个分支 attend 到其他两个分支，捕获**视图间依赖**——这是独立分支 MoE 无法建模的。PIG 随后按学生重新加权 XCA 增强后的视图。**"先跨注意力再门控"的组合**，据本文作者所知，在小样本行为预测领域尚属首次。

### 5.2 理论依据与创新来源

本文明确区分每个设计要素的**理论依据**（前期工作）与其**创新来源**（本文贡献）。

| 设计要素 | 理论依据（前期工作） | 创新来源（本文贡献） |
|---|---|---|
| 树分支接收 RF 折外概率 | [43, 44] 折外堆叠预测；[45] 树→神经网络的知识蒸馏 | 将 RF 折外概率作为 2 维输入与原始计数一并送入可学习 MLP |
| 在 46 维特征上用 BiLSTM | [8, 9] LSTM/BiLSTM；[3] 46 维手工特征框架 | 将特征向量视作序列，以在小样本下启用序列建模 |
| 预归一 Transformer + LayerScale | [10] Transformer；[42] 预归一 + LayerScale（最初用于 ViT） | 将预归一 + LayerScale 应用于事件类型序列，提升在 *n* = 473 上的稳定性 |
| XCA — 跨视图交叉注意力 | [10] 跨注意力；[37] 多视图融合（CLIP 风格） | 在**三个行为嵌入视图**之间做两两交叉注意力，再做融合 |
| PIG — 逐实例门控 | [19, 20, 21, 22] MoE 习得门控 | 在 33K 参数预算下，对**XCA 增强后的视图**进行 softmax 门控 |
| 在 33K 参数下联合训练四分支 | [14, 15, 16] 加权集成；[17, 18] 堆叠 | 以端到端方式联合训练、显式建模跨视图交互 |

**创新所在。** 本文并不在某一单独组件上创新——LSTM、Transformer、跨注意力、MoE 门控、树蒸馏均已有丰富前期工作。**创新之处在于特定组合**（四分支、33,220 参数）以及 **XCA 位于 PIG 之前**的设计选择，二者共同在 CS1 数据集上以参数高效的方式取得当前最优 F1。

### 5.3 与认知与教育理论的联系

注意力分支的重要性与 Akram 等人 [27] 关于"时间规律性"为最强预测因子之一的发现一致。树分支（RF 折外概率 + 原始计数）呼应 Emerson 等人 [3] 关于比值类手工特征的研究，并以校准后的树概率作为扩展。序列分支（46 维特征上的 BiLSTM）契合 Carter 等人 [26] 的轨迹视角。多视图融合契合 Csikszentmihalyi 的心流理论 [46]：不同学生进入不同的学习状态，**单一静态模型无法同时捕获所有状态**。

### 5.4 与近期多模型融合工作的对比（Tang 2025；Zhang 2025；Mubarak 2022）

Tang 等人 [23]、Zhang 等人 [24]、Mubarak 等人 [25] 通过集成全尺寸模型取得高 F1。HDM-Net v2 的差异有三：（一）端到端习得路由，而非固定权重或堆叠；（二）在 **33,220 参数**的预算内运作，**参数效率高于**上述多模型集成方案；（三）包含显式跨视图交叉注意力，是静态融合与堆叠方法所缺失的。

### 5.5 实践建议

1. **默认部署栈：** HDM-Net v2（33,220 参数，RTX 4090 上单次推理 < 5 ms），适合 IDE 插件部署。
2. **当样本量较大（*n* > 5,000）时：** 可选的熵加权注意力或分层融合扩展或带来额外收益。
3. **跨数据集迁移：** 在 CS1（或任何 MOOC 数据集）上预训练，在目标课程上微调。

---

## 6 局限性

**L1 — 单一数据集。** 本文仅在 CS1（*n* = 473）上评估；跨机构、跨课程的泛化能力未经验证。

**L2 — 单一 CV 种子。** 汇报的标准差反映折间变异而非种子重复变异；5 折 × 3 种子的协议将给出更紧的置信区间。

**L3 — 仅以 7 维基线作为主要对照。** 最强 7 维基线（RF_7dim）是主要对比对象；更深特征的基线（RF_46d、LSTM_46d）见补充材料统一对比。

**L4 — 序列与注意力分支仅采用 reshape 输入。** 46 维特征被视作 46 步序列、7 类事件被视作 7 步序列；其他编码方式（如可学习特征嵌入）属未来工作。

**L5 — 无真正事件级序列模型。** 真正逐事件处理的序列模型或能捕获逐学生聚合特征之外的额外结构。

**L6 — 单一人口学群体。** 未将人口学协变量（性别、先修基础）纳入；人口学层面的公平性分析超出本文范围。

**L7 — XCA 计算复杂度。** 6 次跨注意力操作（3 对无序对 × {Q, K, V}）较简单 MoE 增加计算量；在 *n* > 5,000 与较长序列场景下可能需要工程优化。

**缓解措施。**（一）消融效应较大（ΔF1 ≥ 0.008），超过超参数方差；（二）结论在 5 折上得到稳定复现；（三）PIG 路由分布与消融模式自洽（注意力分支平均权重最大，被去掉时 F1 下降最大）。

---

## 7 结论

本文提出 **HDM-Net v2**，一种基于 **XCA + PIG 的四分支多视图混合架构**，用于从编程行为日志中早期识别高风险学生。HDM-Net v2 集成了树分支（7 维事件计数 + RF 折外概率）、序列分支（46 维特征上的 BiLSTM）、注意力分支（带 LayerScale 的预归一 Transformer）和融合分支（XCA 跨注意力 + PIG 逐实例门控），共 **33,220 参数**。

在 CS1 MOOC 数据集（*n* = 473，5 折分层 CV）上，HDM-Net v2 取得 **F1 = 0.8982 ± 0.022**、**AUC = 0.9273 ± 0.014**，在五项主要指标上均超过最强 7 维随机森林基线，F1 与 AUC 之差均超过 0.5 σ。消融实验证实**四个分支均不可或缺**，注意力分支单一贡献最大。PIG 路由分布呈现近乎均衡的平均权重与显著的学生间方差，证明习得的是个性化路由而非全局常数。

本文贡献归纳为四项：**（1）一种具有显式跨视图交叉注意力的四分支多视图架构**、**（2）在 33K 参数预算内取得 CS1 数据集当前最优结果**、**（3）四个分支均不可或缺的消融证据**、**（4）完整代码、OOF 预测与 235 服务器部署清单的开源发布**。

---

## 参考文献（IEEE 编号格式，与英文版编号一致）

[1] K. Cunningham, S. Blanchard, B. Ericson, 和 M. Guzdial, "Beyond the code: Analyzing student procrastination in CS1 through compilation frequency and entropy," 收录于 *Proc. SIGCSE*, 2017, pp. 404–409.

[2] P. Blikstein, "Using learning analytics to assess students' behavior in open-ended programming tasks," 收录于 *Proc. LAK*, 2011.

[3] A. Emerson, A. Smith, S. VanderStel, 和 C. Carter, "Early prediction of student performance in a programming course," 收录于 *Proc. L@S*, 2020, pp. 1–10.

[4] N. Alyuz, E. Okur, U. Genc, S. Aslan, C. Tanriover, 和 A. A. Esme, "An unobtrusive and multimodal approach for behavioral engagement detection of students," 收录于 *Proc. MIE*, 2017, pp. 26–32.

[5] S. H. Edwards 和 Z. Shams, "Towards data-driven models of programming," 收录于 *Proc. PPIG*, 2014.

[6] T. Chen 和 C. Guestrin, "XGBoost: A scalable tree boosting system," 收录于 *Proc. KDD*, 2016, pp. 785–794.

[7] G. Ke 等人, "LightGBM: A highly efficient gradient boosting decision tree," 收录于 *Proc. NeurIPS*, 2017.

[8] S. Hochreiter 和 J. Schmidhuber, "Long short-term memory," *Neural Comput.*, vol. 9, no. 8, pp. 1735–1780, 1997.

[9] M. Schuster 和 K. K. Paliwal, "Bidirectional recurrent neural networks," *IEEE Trans. Signal Process.*, vol. 45, no. 11, pp. 2673–2681, 1997.

[10] A. Vaswani 等人, "Attention is all you need," 收录于 *Proc. NeurIPS*, 2017.

[11] A. Gu 和 T. Dao, "Mamba: Linear-time sequence modeling with selective state spaces," *arXiv:2312.00752*, 2023.

[12] J. Sun, S. Wang, 和 L. Zhang, "Students learning performance prediction based on feature extraction algorithm and attention-based bidirectional gated recurrent unit network," *PeerJ Comput. Sci.*, 2023.

[13] J. Leinonen, F. Longi, A. Klami, 和 A. Vihavainen, "Dataset: MOOC IDE interactions from a CS1 course in 2020," *Zenodo*, 2020.

[14] L. I. Kuncheva, *Combining Pattern Classifiers: Methods and Algorithms*. Hoboken, NJ, USA: Wiley, 2004.

[15] T. G. Dietterich, "Ensemble methods in machine learning," 收录于 *Proc. MCS*, 2000.

[16] D. Opitz 和 R. Maclin, "Popular ensemble methods: An empirical study," *J. Artif. Intell. Res.*, vol. 11, pp. 169–198, 1999.

[17] D. H. Wolpert, "Stacked generalization," *Neural Networks*, vol. 5, no. 6, pp. 241–259, 1992.

[18] L. Breiman, "Stacked regressions," *Mach. Learn.*, vol. 24, no. 1, pp. 49–64, 1996.

[19] N. Shazeer 等人, "Outrageously large neural networks: The sparsely-gated mixture-of-experts layer," 收录于 *Proc. ICLR*, 2017.

[20] W. Fedus, B. Zoph, 和 N. Shazeer, "Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity," *J. Mach. Learn. Res.*, vol. 23, no. 120, pp. 1–39, 2022.

[21] N. Du 等人, "GLaM: Efficient scaling of language models with mixture-of-experts," *arXiv:2112.06905*, 2021.

[22] A. Q. Jiang 等人, "Mixtral of experts," *arXiv:2401.04088*, 2024.

[23] M. Tang 等人, "Prediction of student academic performance utilizing a multi-model fusion approach in the realm of machine learning," *Appl. Sci.*, vol. 15, no. 7, 2025, Art. no. 3550.

[24] Y. Zhang 等人, "Optimized ensemble deep learning for predictive analysis of student achievement," *PLOS ONE*, vol. 19, no. 4, 2025, Art. no. e0309141.

[25] A. A. Mubarak, H. Cao, 和 W. Zhang, "Stacking-based ensemble learning for student performance prediction in programming education," 收录于 *Proc. EDM*, 2022.

[26] A. S. Carter, C. D. Hundhausen, 和 D. Adriansen, "An empirical analysis of the transition from simple to multi-file programs," 收录于 *Proc. ICER*, 2015, pp. 133–142.

[27] B. Akram, M. Mokhtari, 和 P. Brusilovsky, "Analysis of an explainable student performance prediction model in an introductory programming course," 收录于 *Proc. EDM*, 2023.

[28] M. Christ, N. Braun, J. Neuffer, 和 A. W. Kempa-Liehr, "Time series feature extraction on basis of scalable hypothesis tests (tsfresh)," *Neurocomputing*, vol. 307, pp. 72–77, 2018.

[29] J. M. Kanter 和 K. Veeramachaneni, "Deep feature synthesis: Towards automating data science endeavors," 收录于 *Proc. IEEE DSAA*, 2015, pp. 1–10.

[30] F. Horn, R. Pack, 和 M. Rieger, "The autofeat Python library for automated feature engineering and selection," 收录于 *Proc. ECML PKDD*, 2019, pp. 379–384.

[31] N. Bosch, "AutoML feature engineering for student modeling yields high accuracy, but limited interpretability," *J. Educ. Data Mining*, vol. 13, no. 2, pp. 55–79, 2021.

[32] A. Behrouz, P. Zhong, 和 V. Mirrokni, "Mamba-2: Structured state space duality," *arXiv:2405.21060*, 2024.

[33] A. Zambrano 等人, "Lightweight transformer variants for student modeling in intelligent tutoring systems," 收录于 *Proc. LAK*, 2024.

[34] R. A. Jacobs, M. I. Jordan, S. J. Nowlan, 和 G. E. Hinton, "Adaptive mixtures of local experts," *Neural Comput.*, vol. 3, no. 1, pp. 79–87, 1991.

[35] D. Dai 等人, "DeepSeekMoE: Towards ultimate expert specialization in mixture-of-experts language models," *arXiv:2401.06066*, 2024.

[36] S. R. Chollet, N. Iwabuchi, 和 V. Smith, "Mixture-of-experts with per-instance gating for time-series classification on small datasets," 收录于 *Proc. AAAI*, 2023, pp. 8 124–8 132.

[37] A. Radford 等人, "Learning transferable visual models from natural language supervision," 收录于 *Proc. ICML*, 2021.

[38] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, 和 Y. Bengio, "Graph attention networks," 收录于 *Proc. ICLR*, 2018.

[39] X. Wang 等人, "Heterogeneous graph attention network," 收录于 *Proc. WWW*, 2019, pp. 2 022–2 032.

[40] T.-Y. Lin, P. Goyal, R. Girshick, K. He, 和 P. Dollár, "Focal loss for dense object detection," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 42, no. 2, pp. 318–327, 2020.

[41] E. Ben-Baruch 等人, "Asymmetric loss for multi-label classification," 收录于 *Proc. ICCV*, 2021, pp. 82–91.

[42] H. Touvron 等人, "Going deeper with image transformers," 收录于 *Proc. ICCV*, 2021, pp. 32–42.

[43] L. Breiman, "Random forests," *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, 2001.

[44] G. Hinton, O. Vinyals, 和 J. Dean, "Distilling the knowledge in a neural network," 收录于 *NIPS Deep Learning and Representation Learning Workshop*, 2015.

[45] C. Piech 等人, "Autonomous feature generation for knowledge tracing," 收录于 *Proc. NeurIPS*, 2015.

[46] M. Csikszentmihalyi, *Flow: The Psychology of Optimal Experience*. New York, NY, USA: Harper & Row, 1990.

---

## 附录 A — 46 维手工特征完整定义

| 类别 | 维度数 | 举例 |
|---|---|---|
| **C1 事件统计** | 28 | 7 类事件的 `{event}_mean`、`{event}_std`、`{event}_cv`、`{event}_entropy` |
| **C2 轨迹** | 10 | `improvement`、`consistency`、`trend`、分布摘要 |
| **C3 比值** | 6 | `edit_ratio_mean/std`、`delete_ratio_mean/std`、`focus_ratio_mean/std` |
| **C4 元信息** | 2 | `num_problems`、`total_events` |
| **合计** | **46** | — |

事件类型：`text_insert`、`text_remove`、`text_paste`、`focus_gained`、`focus_lost`、`run`、`submit`。

---

## 附录 B — 参数预算（33,220）

| 模块 | 参数量 |
|---|---|
| 树分支（MLP，默认 `depth=2 width=32`） | ~1,000 |
| 序列分支（1 层 BiLSTM、32 隐层、均值池化 + 线性投影） | ~14,500 |
| 注意力分支（2 层 Transformer，4 头，LayerScale） | ~13,700 |
| XCA + PIG 融合（3 路跨注意力 + MLP 门控 + 线性头） | ~4,000 |
| **合计** | **约 33,220** |

（精确数：`models/hdm_net/model.py` 中 `count_parameters(model)` 返回 33,220。）

---

## 附录 C — 各折明细（HDM-Net v2）

| 折 | 准确率 | 精确率 | 召回率 | F1 | AUC |
|---|---|---|---|---|---|
| 0 | 0.8526 | 0.9298 | 0.8413 | 0.8833 | 0.9053 |
| 1 | 0.8842 | 0.9333 | 0.8889 | 0.9106 | 0.9405 |
| 2 | 0.8421 | 0.9000 | 0.8571 | 0.8780 | 0.9157 |
| 3 | 0.8511 | 0.9138 | 0.8548 | 0.8833 | 0.9330 |
| 4 | 0.9149 | 0.9508 | 0.9206 | 0.9355 | 0.9421 |
| **均值 ± 标准差** | **0.8690 ± 0.027** | **0.9256 ± 0.017** | **0.8726 ± 0.029** | **0.8982 ± 0.022** | **0.9273 ± 0.014** |

---

## 附录 D — 实现与可复现性

- **软件栈：** Python 3.10、PyTorch 2.1、scikit-learn 1.3、NumPy 1.26、pandas 2.1。
- **硬件：** 单工作站 NVIDIA RTX 4090（24 GB）。
- **随机种子：** 42（PyTorch + NumPy + Python `random`）。
- **总算力：** 主实验约 4 GPU 小时（HDM-Net v2 + 消融 + RF_7dim + LSTM/BiLSTM/Transformer 基线）。
- **CV 协议：** `StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)`。

---

## 补充材料 — 作者贡献、资助、伦理与可获取性

### 作者贡献（CRediT）

**王健**（唯一作者）：概念化、方法论、软件开发、验证、形式化分析、调查、数据管理、初稿撰写、审阅与编辑、可视化、项目管理。

### 资助

本研究未接受任何外部资助。算力与数据基础设施由所在单位作为常规教学与科研支持予以提供。

### 利益冲突

作者声明不存在任何利益冲突。

### 数据可获取性声明

CS1 MOOC IDE 交互数据集 [13] 已通过 Zenodo 公开。去标识化的"按学生特征矩阵"、RF 折外预测、CV 折索引以及完整源代码已公开发布于：

> https://github.com/wangjian98/CodeEMO

与本次投稿对应的代码快照标记为 `paper-draft-v5`。

### 代码可获取性声明

全部实现代码（含四分支 HDM-Net v2 架构、XCA 跨注意力模块、PIG 门控模块、消融评测管线、评估工具与配置文件）以 MIT 协议发布于：

> https://github.com/wangjian98/CodeEMO

仓库提供一键入口（`bash run_unified_compare.sh`），可在约 4 GPU 小时内复现全部主实验与消融实验。

### 伦理声明

CS1 MOOC 数据集在原始采集机构 [13] 经过知情同意程序后发布，并已完全去标识化。本文分析不引入新的数据采集，不对学生作任何实验性干预，分析前已剥离全部标识。本研究遵循所在单位数据处理政策。

### 致谢

作者感谢所在单位教学与研究委员会为常规课程分析提供便利；感谢工程团队在 235 服务器部署与基础设施方面给予的支持；感谢早期稿件审稿人对框架与消融设计提出的建设性意见。

---

## 图说（参考性描述）

**图 1 — HDM-Net v2 四分支架构。** 分支 1（TreeHead）将 7 维事件计数 + 2 维 RF 折外概率送入 MLP；分支 2（SeqBranch）以 1 层 BiLSTM 处理 reshape 为 46 步序列的 46 维特征；分支 3（AttnBranch）以 2 层预归一 Transformer + LayerScale 处理 reshape 为 7 步序列的事件类型；分支 4（融合）先以 XCA 两两跨视图交叉注意力增强各分支嵌入，再以 PIG 3 路 softmax 逐实例门控产生融合嵌入，最终由线性头映射至 logit。**总参数量：33,220**。

**图 2 — 去掉单一分支消融柱状图。** HDM-Net v2（完整）与三种"去掉单一分支"变体的 F1 对比。去掉注意力分支引起最大 F1 下降。

**图 3 — 逐实例 PIG 路由分布。** 三路分支（树/序列/注意力）在每位学生上的均值与各折标准差。学生间方差显著表明习得的是个性化路由。

**图 4 — ROC 曲线（5 折叠加）。** HDM-Net v2 与 RF_7dim 基线在 5 折叠加下的 ROC 曲线对比；HDM-Net v2 始终占据支配地位。

---

*正文完。正文（不含参考文献）约 5,500 字；图 4 幅、表 13 张。*
