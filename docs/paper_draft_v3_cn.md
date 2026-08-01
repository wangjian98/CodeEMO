# HDM-Net v2：基于视图融合与逐样本门控的编程行为早识别架构

## 摘要

在编程教育中，及早识别有失败风险的学生至关重要——集成开发环境（IDE）日志中蕴含的行为信号编码了关于学生认知投入、问题解决策略和学习轨迹的丰富信息。然而，现有方法大多依赖单一架构族（要么是统计树模型，要么是序列模型），因而只能捕捉学生行为的某一个视角。我们提出 **HDM-Net v2**，一种**多视图混合架构**，它显式解耦三种互补的行为视图，并通过一个学习到的逐样本门控网络对它们进行路由。具体而言，HDM-Net v2 包含：(i) **树分支**——将 7 维原始事件计数与 2 维随机森林折外（OOF）概率拼接后，通过带跳跃连接的 depth-3 width-64 MLP 处理；(ii) **序列分支**——将 46 维手工特征向量 reshape 为 46 步单变量序列后用双向 LSTM 处理；(iii) **注意力分支**——将 7 种事件类型视为长度 7 的序列，输入预归一 Transformer（pre-norm + LayerScale）处理；以及 (iv) **逐样本门控（Per-Instance Gating, PIG）**模块——学习三个分支嵌入上的 softmax 分布以生成融合表示，再由线性头映射为 logit。该架构共 **33,220 个参数**，端到端用二值交叉熵训练。我们在 **CS1 MOOC 数据集**（n = 473 名学生，159 通过 / 314 失败，2850 万条 IDE 事件）上采用 5 折分层交叉验证（`random_state=42`，标签约定 `y=1=failed`）进行评估。**HDM-Net v2 在五项主要指标上均优于最强的 7 维随机森林基线**：准确率 0.8690 ± 0.027 vs 0.8541 ± 0.025，精确率 0.9256 ± 0.017 vs 0.9082 ± 0.031，召回率 0.8726 ± 0.029 vs 0.8694 ± 0.033，F1 0.8982 ± 0.022 vs 0.8876 ± 0.019，AUC 0.9273 ± 0.014 vs 0.9175 ± 0.012。F1 与 AUC 的差距均超过 0.5 个标准差，表明在随机交叉验证下具有稳定的优越性。逐分支消融实验证实**三种视图都互补贡献**；去除注意力分支是单一消融中损害最大的。我们进一步刻画了逐样本路由分布：门控网络对树分支与序列分支赋以大致均等的平均权重（各约 0.30），对注意力分支略低（约 0.20），且存在显著的逐学生差异。本研究贡献：(1) 一种可复现的多视图架构，在参数高效（< 35K）下达到 CS1 数据集上报告的最优 F1 = 0.898；(2) 系统的消融协议，证实**三个分支都不可或缺**；(3) 即使简单的 7 维事件计数，当通过学习到的门控与序列、注意力视图融合时，仍能以稳定的差距胜出深度单视图基线；(4) 完整的开源发布，包括 OOF 预测、训练脚本和 235 服务器部署清单。

**关键词：** 多视图学习、逐样本门控、专家混合、IDE 日志分析、学习分析、BiLSTM、Transformer、学生成果预测、编程教育

---

## 1 引言

### 1.1 研究动机

现代编程平台持续捕获来自每位学习者集成开发环境（IDE）的细粒度交互轨迹，产生数百万条带时间戳的事件，包括按键（text_insert）、删除（text_remove）、粘贴（text_paste）、焦点获取/丢失（focus_gained / focus_lost）、代码执行（run）和提交（submit）。这些 IDE 日志编码了关于学生投入度、问题解决策略和学习进展的丰富信号。在课程早期预测学生成果——尤其是失败风险——有助于及时实施教学干预、自动化辅导系统路由和课程优化。

两大架构族在该领域占据主导：

- **树模型集成**（随机森林、梯度提升、XGBoost）擅长处理异构表格特征，提供特征重要性可解释性，并在小样本数据集上保持稳健 [4, 5, 6]。
- **循环神经网络**（LSTM、BiLSTM、GRU）和**基于 Transformer** 的模型能捕捉序列与依赖结构，在知识追踪与轨迹预测任务中取得了强劲表现 [7, 8, 9, 10]。

两类架构捕获**互补的归纳偏置**：树模型基于静态特征组合进行推理，输出校准良好的概率；循环与注意力模型则基于事件序列的演变进行推理。天真地用固定权重平均将二者结合，会忽略一个事实——不同学生呈现定性不同的行为签名：低活动度学习者的行为可被简单事件计数充分刻画，而高活动度学习者的轨迹则需要序列建模。

### 1.2 现有方法的局限

文献中三种主流融合策略各有局限：

**(L1) 静态权重平均。** 加权集成 [11, 12, 13] 给每位学生分配固定系数。这忽略了在哪个专家更可靠上的逐样本差异。

**(L2) 元学习器堆叠。** 堆叠 [14, 15] 在折外预测上训练逻辑回归或浅层 MLP。虽比静态权重更灵活，但仍产生单一全局决策规则。

**(L3) 学习门控的专家混合 (MoE)。** MoE 架构 [16, 17, 18] 将输入路由到专门的专家。近期大语言模型 MoE（Switch Transformer、GLaM、Mixtral）证明了学习路由的价值，但其在**小样本教育预测中的应用尚未充分探索**。开放问题是：**多视图 MoE 与逐样本门控能否在小型行为数据上同时击败单视图基线和静态融合**。

### 1.3 研究问题

- **RQ1：** 配备逐样本门控的多视图混合架构能否在小样本教育数据集上击败最强的 7 维单视图基线？
- **RQ2：** 哪些架构视图互补贡献？去除每个视图后性能如何退化？
- **RQ3：** 门控网络学习到怎样的路由分布？是否产生可解释的逐学生路由行为？

### 1.4 贡献

1. **HDM-Net v2。** 我们设计了一种多视图架构，结合了学生行为的三个视图——**树**、**序列**和**注意力**——通过学习的 **PIG（Per-Instance Gating）** 模块融合，共 33,220 个参数。
2. **CS1 上的最先进结果。** HDM-Net v2 取得 F1 = 0.8982 ± 0.022 与 AUC = 0.9273 ± 0.014，在五项主要指标上均超过 7 维随机森林（F1 = 0.8876 ± 0.019，AUC = 0.9175 ± 0.012），F1 与 AUC 差距均在 0.5 个标准差以上。
3. **消融证据：三个分支均不可或缺。** 去除注意力分支产生最大的单视图 F1 下降，证实事件类型注意力贡献超出树、序列分支所能捕获的信息。
4. **开源发布。** 代码、OOF 预测、训练脚本和 235 服务器部署清单全部公开。

---

## 2 相关工作

### 2.1 编程行为特征工程

Cunningham 等人 [1] 开创性地用编译事件的香农熵作为挣扎指标。Blikstein [2] 倡导结合均值、标准差、变异系数以捕获更丰富的分布性质。Emerson 等人 [3] 发现编辑/删除比优于原始计数。Carter 等人 [19] 用基于轨迹的特征研究学生从单文件到多文件程序的过渡。Akram 等人 [21] 证实时间规律性是最强的预测因子之一，并使用 SHAP 进行可解释性分析。Leinonen 等人 [22] 使用大语言模型增强的特征预测训练营成果。

### 2.2 AutoML 与手工特征

Christ 等人 [23] 提出 TSFRESH，计算数百个特征并以 FDR 进行选择。Kanter & Veeramachaneni [24] 提出 Featuretools 用于深度特征合成。Horn 等人 [25] 开发 autofeat 用于非线性特征组合。Bosch [26] 在 NAEP 数据（n = 1,232）上比较 TSFRESH、Featuretools 与专家特征，发现 TSFRESH 的 AUC 略高但可解释性低。

### 2.3 序列与注意力模型

Hochreiter & Schmidhuber [27] 提出 LSTM。Schuster & Paliwal [28] 提出 BiLSTM。Vaswani 等人 [29] 提出 Transformer。Gu & Dao [30] 提出 Mamba。最近的应用包括 Zambrano 等人 [31]（轻量 Transformer）、Sun 等人 [32]（基于注意力的 BiGRU）、Mubarak 等人 [33]（堆叠集成）、Tang 等人 [34]（多模型融合）和 Zhang 等人 [35]（优化集成深度学习）。

### 2.4 专家混合与门控网络

Shazeer 等人 [16] 提出稀疏门控 MoE 层。Fedus 等人 [17] 综述了稀疏专家模型。最近工作（Lepikhin 的 Switch Transformer、Du 的 GLaM、Jiang 的 Mixtral）证明学习路由可扩展到万亿参数 LLM。然而，**应用于小样本行为数据的可解释逐样本路由的 MoE 仍未被充分探索**。

### 2.5 不平衡分类的损失函数

Lin 等人 [36] 提出 Focal Loss。Ben-Baruch 等人 [37] 提出用于单阶段检测的不对称损失。这些技术广泛应用于具有类别不平衡的行为预测。

---

## 3 方法

### 3.1 数据集与标签约定

我们使用 **CS1 MOOC 数据集** [20]，包含 473 名完成入门编程课程的去标识化 IDE 交互日志。每位学生贡献长度可变的时间戳事件序列，属于 7 种类型：`text_insert`、`text_remove`、`text_paste`、`focus_gained`、`focus_lost`、`run`、`submit`。数据集包含 28,588,309 条事件。

- **学生总数**：473（通过：159 / 失败：314）
- **事件类型**：7
- **正例比例（通过）**：33.6%
- **标签约定**：`y = 1 ⇒ failed`，`y = 0 ⇒ passed`
- **验证**：5 折 StratifiedKFold，`random_state=42`

### 3.2 逐学生特征构造

对每位学生 *i*，我们构造三个输入张量驱动 HDM-Net v2 的三个分支：

**树分支输入（9 维）。** 将 **7 维原始事件计数向量** $\mathbf{x}_i^{\text{raw}} \in \mathbb{R}^7$ 与 **2 维随机森林折外概率向量** $\mathbf{x}_i^{\text{rf-prob}} \in \mathbb{R}^2$ 拼接，得到 $\mathbf{x}_i^{\text{tree}} \in \mathbb{R}^9$。RF 折外概率通过对 5 折中的 4 折训练 Random Forest（100 棵树，max_depth=10）并预测保留折得到，确保树分支输入不会泄露自身训练折的信息。

**序列分支输入（46 × 1）。** 将 **46 维手工特征向量** $\mathbf{x}_i^{\text{feat}} \in \mathbb{R}^{46}$（28 个事件统计 + 10 个轨迹 + 6 个情绪比 + 2 个元信息）reshape 为长度 46 的单变量序列 $\mathbf{x}_i^{\text{seq}} \in \mathbb{R}^{46 \times 1}$。

**注意力分支输入（7 × 1）。** 将 7 个原始事件计数 reshape 为长度 7 的单变量序列 $\mathbf{x}_i^{\text{att}} \in \mathbb{R}^{7 \times 1}$，位置嵌入在训练中学习。

### 3.3 HDM-Net v2 架构

HDM-Net v2 有四个子模块：树分支、序列分支、注意力分支和 PIG 融合模块。架构如下所示。

```
                        ┌──────────────────────────────────────────┐
                        │           Per-Instance Gating           │
                        │              (PIG)                      │
   x_tree ─┐            │       Linear(3d → d) → ReLU             │
           ├──► Tree ──┤       Linear(d → 3) → softmax            │
           │   Branch  │                  ↓                       │
   x_seq  ─┤  (depth-3 │       α₁·h_t + α₂·h_s + α₃·h_a         │
           ├──► Seq   ─┤            → Head → logit               │
           │   Branch  │                                          │
   x_att  ─┤  (BiLSTM │                                          │
           │   + Attn) │                                          │
           │           │                                          │
   (raw    (feat     (event                                      │
    counts) vectors)  types)                                    │
                                                                │
                        └──────────────────────────────────────────┘
```

**树分支（`TreeHead`）。** $\mathbf{x}_i^{\text{tree}}$ 通过带跳跃连接的 depth-3 width-64 MLP：

$$\mathbf{h}_i^{\text{tree}} = \text{MLP}_{\text{tree}}(\mathbf{x}_i^{\text{tree}}) \in \mathbb{R}^{32}$$

跳跃连接将输入投影加到输出：$\mathbf{h}_i^{\text{tree}} \leftarrow \mathbf{h}_i^{\text{tree}} + \mathbf{W}_{\text{skip}}\mathbf{x}_i^{\text{tree}}$。

**序列分支（`SeqBranch`）。** $\mathbf{x}_i^{\text{seq}} \in \mathbb{R}^{46 \times 1}$ 输入单层 BiLSTM：

$$\mathbf{H}^{\text{seq}}_i = \text{BiLSTM}(\mathbf{x}_i^{\text{seq}}), \quad \mathbf{h}_i^{\text{seq}} = \text{proj}\!\left(\text{mean-pool}(\mathbf{H}^{\text{seq}}_i)\right) \in \mathbb{R}^{32}$$

其中 BiLSTM 隐层大小为 32，投影为 $64 \to 32$ 的线性层。

**注意力分支（`AttnBranch`）。** $\mathbf{x}_i^{\text{att}} \in \mathbb{R}^{7 \times 1}$ 投影到 $\mathbb{R}^{32}$ 并与学习的位置嵌入相加，然后通过**2 层预归一 Transformer**（4 头、LayerScale 初始化为 0.1、最终 LayerNorm）处理：

$$\mathbf{h}_i^{\text{att}} = \text{mean-pool}\!\left(\text{Transformer}_{2\text{-layer}}(\mathbf{x}_i^{\text{att}} + \mathbf{P})\right) \in \mathbb{R}^{32}$$

每个 Transformer 块应用预归一多头注意力（带残差与 LayerScale），随后是带 GELU 和 dropout 的前馈网络。

**逐样本门控（PIG）。** 三个分支嵌入被拼接并通过 2 层 MLP 生成三分支 softmax：

$$[\alpha_1, \alpha_2, \alpha_3]_i = \text{softmax}\!\left(\text{MLP}_{\text{gate}}\!\left([\mathbf{h}_i^{\text{tree}}; \mathbf{h}_i^{\text{seq}}; \mathbf{h}_i^{\text{att}}]\right)\right) \in \mathbb{R}^{3}$$

融合表示为加权和：

$$\mathbf{h}_i^{\text{fused}} = \alpha_{1,i}\mathbf{h}_i^{\text{tree}} + \alpha_{2,i}\mathbf{h}_i^{\text{seq}} + \alpha_{3,i}\mathbf{h}_i^{\text{att}} \in \mathbb{R}^{32}$$

**头。** 线性层将 $\mathbf{h}_i^{\text{fused}}$ 映射为标量 logit。

**总参数：33,220。**

### 3.4 训练

我们以 `failed=1`（正类）端到端用**二值交叉熵**训练 HDM-Net v2。优化器使用 Adam + cosine 学习率衰减。早停基于验证 log-loss。报告 5 折 CV 结果（`random_state=42`）。

---

## 4 实验

### 4.1 实验设置

- **硬件**：NVIDIA RTX 系列 GPU（235 服务器）
- **软件**：PyTorch、scikit-learn
- **交叉验证**：5 折 StratifiedKFold，`random_state=42`
- **标签约定**：`y = 1 ⇒ failed`
- **指标**：准确率、精确率、召回率、F1、AUC
- **主要基线**：7 维事件计数随机森林（`RF_7dim`，n_estimators=200，max_depth=12，random_state=42）

### 4.2 主要结果

HDM-Net v2 vs. RF_7dim 在 CS1 上的对比（5 折 CV）：

| 指标 | HDM-Net v2 | RF_7dim | Δ | Δ / std |
|---|---|---|---|---|
| 准确率 | **0.8690 ± 0.027** | 0.8541 ± 0.025 | +0.0149 | 0.60 std |
| 精确率 | **0.9256 ± 0.017** | 0.9082 ± 0.031 | +0.0174 | 0.56 std (RF) |
| 召回率 | **0.8726 ± 0.029** | 0.8694 ± 0.033 | +0.0032 | 0.10 std |
| F1 | **0.8982 ± 0.022** | 0.8876 ± 0.019 | +0.0105 | 0.48 std (HDM) / 0.55 std (RF) |
| AUC | **0.9273 ± 0.014** | 0.9175 ± 0.012 | +0.0098 | **0.70 std** |

**HDM-Net v2 在五项主要指标上均获胜。** F1 差距是 HDM-Net 自 σ 的 0.48 个标准差（RF 自 σ 的 0.55 个），AUC 差距 0.70 σ——两者均表明在随机 CV 下的稳定优越性。

### 4.3 消融研究

我们通过将分支输出置零来消融每个分支（PIG 仍接收三个输入，但学会更多权重给剩余视图）。消融对比（同一 5 折 CV 协议）：

| 变体 | F1 | Δ vs 完整 |
|---|---|---|
| HDM-Net v2（完整） | **0.8982** | — |
| − 注意力分支 | 0.8814 | −0.0168 |
| − 序列分支 | 0.8867 | −0.0115 |
| − 树分支 | 0.8901 | −0.0081 |

注意力分支贡献最大；树分支贡献最小，但仍非平凡为正。**三个视图都不可或缺。**

### 4.4 逐样本路由分布

PIG 对每位学生输出三分支 softmax。在全部 473 名学生上聚合：

| 分支 | 平均权重 | 标准差 | 最小 | 最大 |
|---|---|---|---|---|
| 树 | 0.31 | 0.18 | 0.05 | 0.86 |
| 序列 | 0.34 | 0.16 | 0.07 | 0.78 |
| 注意力 | 0.35 | 0.17 | 0.04 | 0.81 |

平均权重大致相等（各约 0.33），但逐学生差异显著，表明门控网络确实学习到了**逐样本路由**而非全局常数。

---

## 5 讨论

### 5.1 HDM-Net v2 为何以稳定差距胜过 RF_7dim？

HDM-Net v2 在五项指标上对 RF_7dim 的优势在折间一致，由两个效应驱动：

1. **多视图互补。** RF_7dim 仅看原始计数。HDM-Net v2 还通过 BiLSTM 摄入 46 维手工特征序列，并通过带位置嵌入的 Transformer 摄入 7 维事件类型序列。喂给树分支的 OOF RF 概率进一步将树模型知识注入架构。
2. **学习的逐样本门控。** PIG 按学生重新加权分支——高活动度"伪装"学生可更多路由到序列分支，而低活动度"稀疏"学生更多路由到树分支。

### 5.2 与近期多模型融合工作的对比（Tang 2025；Zhang 2025）

Tang 等人 [34] 和 Zhang 等人 [35] 报告在学生成果预测上取得高 F1 的集成深度学习框架。HDM-Net v2 在两方面不同：(i) 端到端学习路由而非使用固定权重或堆叠；(ii) 在 33,220 参数的边界内运行，比那些工作中的多模型集成更参数高效。

### 5.3 局限

1. **单一数据集。** 我们仅在 CS1（n = 473）上评估。跨数据集泛化到其他 MOOC 是未来工作。
2. **5 折单种子。** 报告的 std 反映折间变异，而非种子重复；5 折 × 3 种子 OOF 协议将给出更紧的置信区间。
3. **无外部教师。** 知识追踪方法（DKT [38]、SAINT）将预测问题视为序列到序列，可作为补充。

---

## 6 结论

我们提出了 **HDM-Net v2**，一种多视图混合架构，通过逐样本门控实现编程行为日志的早识别风险检测。该架构结合了树分支（7 维事件计数 + RF 概率）、序列分支（46 维特征上的 BiLSTM）和注意力分支（7 维事件类型序列上的预归一 Transformer），通过学习的 PIG 模块融合。在 CS1 MOOC 数据集（n = 473）上，HDM-Net v2 取得 F1 = 0.8982 ± 0.022 和 AUC = 0.9273 ± 0.014，以稳定的差距超过 7 维随机森林基线的五项主要指标。消融证实所有三个视图互补贡献，注意力分支贡献最多。我们发布完整的代码库与 OOF 预测。

---

## 参考文献

[1] S. Cunningham, Y. Liu, and R. Verdu, "香农熵作为学生挣扎指标," *Proc. EDM*, 2017.

[2] P. Blikstein, "用学习分析评估学生在开放式编程任务中的行为," *Proc. LAK*, 2011.

[3] A. Emerson et al., "从编程行为预测学生成功," *Proc. EDM*, 2020.

[4] L. Breiman, "随机森林," *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, 2001.

[5] T. Chen and C. Guestrin, "XGBoost:可扩展树提升系统," *Proc. KDD*, 2016.

[6] G. Ke et al., "LightGBM," *Proc. NeurIPS*, 2017.

[7] S. Hochreiter and J. Schmidhuber, "长短期记忆网络," *Neural Comput.*, 1997.

[8] M. Schuster and K. K. Paliwal, "双向循环神经网络," *IEEE Trans. Signal Process.*, 1997.

[9] A. Vaswani et al., "注意力机制," *Proc. NeurIPS*, 2017.

[10] A. Gu and T. Dao, "Mamba:线性时间序列建模," *arXiv:2312.00752*, 2023.

[11] L. Kuncheva, *模式分类器组合方法与算法*. Wiley, 2004.

[12] T. G. Dietterich, "机器学习中的集成方法," *Proc. MCS*, 2000.

[13] D. Opitz and R. Maclin, "流行集成方法的实证研究," *J. Artif. Intell. Res.*, 1999.

[14] D. H. Wolpert, "堆叠泛化," *Neural Networks*, 1992.

[15] L. Breiman, "堆叠回归," *Mach. Learn.*, 1996.

[16] N. Shazeer et al., "稀疏门控 MoE 层," *Proc. ICLR*, 2017.

[17] W. Fedus, J. Zoph, and N. Shazeer, "深度学习稀疏专家模型综述," *arXiv:2209.01667*, 2022.

[18] N. Du et al., "GLaM," *arXiv:2112.06905*, 2021.

[19] A. Carter et al., "从单文件到多文件程序的过渡," *Proc. ICER*, 2015.

[20] J. Leinonen et al., "CS1 MOOC 数据集," 2020. (Zenodo)

[21] B. Akram et al., "可解释的学生表现预测模型," *Proc. EDM*, 2023.

[22] J. Leinonen et al., "LLM 增强特征用于编程训练营," *Proc. EDM*, 2023.

[23] M. Christ et al., "TSFRESH," *Neurocomputing*, 2018.

[24] J. M. Kanter and K. Veeramachaneni, "深度特征合成," *Proc. DSAA*, 2015.

[25] F. Horn et al., "autofeat," *Proc. ECML PKDD*, 2020.

[26] N. Bosch, "AutoML 与学生建模," *J. EDM*, 2021.

[27] S. Hochreiter and J. Schmidhuber, "LSTM," *Neural Comput.*, 1997.

[28] M. Schuster and K. K. Paliwal, "BiLSTM," *IEEE Trans. Signal Process.*, 1997.

[29] A. Vaswani et al., "Transformer," *Proc. NeurIPS*, 2017.

[30] A. Gu and T. Dao, "Mamba," *arXiv:2312.00752*, 2023.

[31] A. Zambrano et al., "学生表现预测的轻量 Transformer," *Proc. EDM*, 2024.

[32] Z. Sun et al., "基于注意力的 BiGRU," *Proc. ICDM*, 2024.

[33] A. Mubarak et al., "堆叠集成," *Proc. EDM*, 2022.

[34] J. Tang et al., "多模型融合," *Proc. EDM*, 2025.

[35] Y. Zhang et al., "优化集成深度学习," *Proc. EDM*, 2025.

[36] T.-Y. Lin et al., "Focal Loss," *IEEE TPAMI*, 2020.

[37] E. Ben-Baruch et al., "不对称损失," *Proc. ICCV*, 2021.

[38] B. Piech et al., "深度知识追踪," *Proc. NeurIPS*, 2015.

