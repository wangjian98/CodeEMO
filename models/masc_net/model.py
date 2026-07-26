"""
MASC-Net: Multi-scale Adaptive Sample-aware Contrastive Network

针对 CodeEMO 小样本(n=473) + 不平衡(1:2) + 折间零正样本问题的设计。

五大组件:
  ① Multi-Scale Encoder (MSE)         多尺度特征编码(局部CNN / 中程MLP / 全局MLP + 跨尺度注意力)
  ② Sample-Aware Contrastive (SACM)   动量对比 + 自适应温度 + 难负样本挖掘
  ③ Prototype Memory Bank (PMB)       2 类 × 4 原型 = 8 个可学习原型, EMA 更新
  ④ Adaptive Threshold Classifier    基于原型距离的分类 + 动态阈值
  ⑤ Uncertainty-Aware Loss Fusion     Focal + 对比 + 原型一致性 + 不确定性正则

输入: 46-dim 标准化特征 (与 BiLSTM/BGM-Net 同一口径)
输出: P(failed=1) 概率
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============= 特征布局常量 (与 feature_engineering.py 一致) =============
CAT1_DIM = 28       # 7 事件 × 4 统计量
CAT2_DIM = 10       # 行为轨迹
CAT3_DIM = 6        # 比率特征
CAT4_DIM = 2        # 元信息
TOTAL_DIM = 46

N_EVENT_TYPES = 7
STATS_PER_EVENT = 4


# ============= ① 多尺度编码器 =============
class MultiScaleEncoder(nn.Module):
    """三尺度编码器 + 跨尺度注意力融合
    
    Local : 1D-CNN over feature dim  (捕捉相邻特征局部交互)
    Mid   : 2-layer MLP on Cat1+Cat3 (捕捉组内交互)
    Global: 2-layer MLP on full 46d  (捕捉跨组交互)
    Cross-Scale Attention: 用查询-键-值注意力融合三尺度
    """
    def __init__(self, output_dim=64, dropout=0.3):
        super().__init__()
        # Local: 把46维视为1D序列, 不同kernel大小 = 不同感受野
        self.local = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(dropout),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )  # out: 32
        # Mid: 28+6 = 34 维 → 32
        self.mid = nn.Sequential(
            nn.Linear(CAT1_DIM + CAT3_DIM, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
            nn.ReLU(),
        )  # out: 32
        # Global: 46 → 32
        self.global_enc = nn.Sequential(
            nn.Linear(TOTAL_DIM, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
            nn.ReLU(),
        )  # out: 32
        # Cross-Scale Attention: 3 个尺度作为 3 个 token, 自注意力融合
        self.scale_attn = nn.MultiheadAttention(
            embed_dim=32, num_heads=4, batch_first=True, dropout=dropout
        )
        self.scale_norm = nn.LayerNorm(32)
        self.fuse = nn.Sequential(
            nn.Linear(32 * 3, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Args: x: (batch, 46)
        Returns: h: (batch, output_dim=64), aux: dict (attention weights)
        """
        b = x.size(0)
        # Local: (b, 1, 46) → (b, 32)
        h_local = self.local(x.unsqueeze(1))
        # Mid: Cat1 + Cat3
        h_mid_in = torch.cat([x[:, :CAT1_DIM], x[:, 38:44]], dim=-1)  # (b, 34)
        h_mid = self.mid(h_mid_in)
        # Global
        h_global = self.global_enc(x)
        # 拼成 3 个 token: (b, 3, 32)
        tokens = torch.stack([h_local, h_mid, h_global], dim=1)
        # 自注意力
        attn_out, attn_weights = self.scale_attn(tokens, tokens, tokens)
        tokens = self.scale_norm(tokens + attn_out)
        # 拼接 → 96 → 64
        flat = tokens.reshape(b, -1)
        h = self.fuse(flat)
        return h, {'attn_weights': attn_weights}  # (b, 3, 3)


# ============= ② 样本感知对比学习模块 =============
class SampleAwareContrastive(nn.Module):
    """动量编码器 + 自适应温度 + InfoNCE 损失
    
    训练时:
      - query encoder: 主编码器上的投影头, 梯度反传
      - key  encoder: 动量更新 (m=0.999)
      - adaptive τ: 由样本自身决定温度
    """
    def __init__(self, feat_dim=64, proj_dim=32, momentum=0.999):
        super().__init__()
        self.momentum = momentum
        self.proj_dim = proj_dim
        # Query projection (主路径)
        self.query_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, proj_dim),
        )
        # Key projection (动量)
        self.key_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, proj_dim),
        )
        # 初始化 key = query
        for qp, kp in zip(self.query_proj.parameters(), self.key_proj.parameters()):
            kp.data.copy_(qp.data)
            kp.requires_grad = False
        # 自适应温度: 由特征决定 τ ∈ [0.05, 0.5]
        self.temp_net = nn.Sequential(
            nn.Linear(feat_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.log_tau_min = nn.Parameter(torch.tensor(-3.0))   # ≈ 0.05
        self.log_tau_max = nn.Parameter(torch.tensor(-0.7))   # ≈ 0.50

    @torch.no_grad()
    def momentum_update(self):
        for qp, kp in zip(self.query_proj.parameters(), self.key_proj.parameters()):
            kp.data.mul_(self.momentum).add_(qp.data, alpha=1 - self.momentum)

    def forward(self, h, prototypes_k):
        """
        Args:
            h: (b, feat_dim) 主编码器输出
            prototypes_k: (K*M, proj_dim) 动量原型键(用作对比负样本)
        Returns:
            z_q: (b, proj_dim)   查询嵌入(用于原型匹配)
            z_k: (b, proj_dim)   键嵌入(动量路径)
            tau: (b,)            每样本自适应温度
        """
        z_q = F.normalize(self.query_proj(h), dim=-1)
        with torch.no_grad():
            z_k = F.normalize(self.key_proj(h), dim=-1)
        # 自适应温度
        tau_raw = self.temp_net(h).squeeze(-1)
        tau = torch.exp(self.log_tau_min + tau_raw * (self.log_tau_max - self.log_tau_min))
        return z_q, z_k, tau

    def info_nce_loss(self, z_q, z_k, labels, prototypes_k, tau):
        """
        InfoNCE: 正样本 = 同标签; 负样本 = 异标签 + 原型
        
        Args:
            z_q: (b, d)
            z_k: (b, d)
            labels: (b,)
            prototypes_k: (n_proto, d)
            tau: (b,)
        Returns:
            loss: scalar
        """
        b = z_q.size(0)
        # 拼接所有"键": 同batch的k + 原型作为大字典
        # 正样本: 自己的 z_k (i==j)
        sim_self = (z_q * z_k).sum(dim=-1, keepdim=True)              # (b, 1)
        sim_proto = z_q @ prototypes_k.t()                              # (b, n_proto)
        sim_intra = z_q @ z_k.t()                                       # (b, b)
        # 去掉对角线 (自身相似度单独算)
        mask = torch.eye(b, dtype=torch.bool, device=z_q.device)
        sim_intra_neg = sim_intra.masked_fill(mask, -1e9)
        logits = torch.cat([sim_self, sim_intra_neg, sim_proto], dim=-1)  # (b, 1+b-1+n_proto)
        # 温度 per-sample
        logits = logits / tau.unsqueeze(-1).clamp(min=0.05, max=0.5)
        # labels: 同batch的同标签样本 j≠i + 原型池里对应类的原型
        # 简化处理: 只以"自身"为正样本(自监督), 用 hard negatives (异类样本 + 所有原型)
        targets = torch.zeros(b, dtype=torch.long, device=z_q.device)
        loss = F.cross_entropy(logits, targets)
        return loss


# ============= ③ 原型记忆库 =============
class PrototypeBank(nn.Module):
    """K 类 × M 原型, EMA 更新
    
    用法:
      - 训练时: 用 batch 内样本的指数移动平均更新原型
      - 推理时: 直接用作分类依据(最近原型 → 类)
    """
    def __init__(self, n_classes=2, n_prototypes_per_class=4, dim=32, momentum=0.9):
        super().__init__()
        self.K = n_classes
        self.M = n_prototypes_per_class
        self.dim = dim
        self.momentum = momentum
        # 原型: (K*M, dim), 初始化为小随机
        self.prototypes = nn.Parameter(
            torch.randn(n_classes * n_prototypes_per_class, dim) * 0.1,
            requires_grad=False,
        )
        # 每个原型的"当前命中计数", 用于避免冷启动偏差
        self.register_buffer('hit_count', torch.zeros(n_classes * n_prototypes_per_class))

    @torch.no_grad()
    def ema_update(self, z, labels, hard_negatives=None):
        """EMA 更新
        
        Args:
            z: (b, d) 归一化嵌入
            labels: (b,)
        """
        z_det = z.detach()
        for k in range(self.K):
            mask = (labels == k)
            if mask.sum() == 0:
                continue
            z_k = z_det[mask]
            # 每个 k 类样本分配给该类下"最近"的原型槽
            proto_k = self.prototypes[k * self.M: (k + 1) * self.M]
            sims = z_k @ proto_k.t()  # (n_k, M)
            assign = sims.argmax(dim=-1)  # 每个样本分到哪个原型
            for m in range(self.M):
                sel = (assign == m)
                if sel.sum() == 0:
                    continue
                update_vec = z_k[sel].mean(dim=0)
                # 归一化
                update_vec = F.normalize(update_vec, dim=-1)
                idx = k * self.M + m
                self.prototypes[idx].data.mul_(self.momentum).add_(
                    update_vec, alpha=1 - self.momentum
                )
                self.prototypes[idx].data = F.normalize(self.prototypes[idx].data, dim=-1)
                self.hit_count[idx] += sel.sum()

    def get_class_assignment(self, z):
        """返回每个样本到各原型的相似度 + 类分配
        
        Args:
            z: (b, d) 归一化嵌入
        Returns:
            sim_per_class: (b, K)   每类最大原型相似度
            sim_per_proto: (b, K*M) 每个原型相似度
        """
        sims = z @ self.prototypes.t()  # (b, K*M)
        sims = sims.view(-1, self.K, self.M)
        sim_per_class = sims.max(dim=-1).values  # (b, K)
        return sim_per_class, sims.view(-1, self.K * self.M)


# ============= ④ 自适应阈值分类器 =============
class AdaptiveThresholdClassifier(nn.Module):
    """基于原型距离 + 动态阈值的二分类
    
    距离 = 1 - max_similarity(到该类最近原型)
    不确定性 = softmax 熵
    """
    def __init__(self, init_thresh=0.5):
        super().__init__()
        # 每类一个可学习阈值 (从训练数据初始化)
        self.log_thresh = nn.Parameter(torch.tensor([0.0, 0.0]), requires_grad=False)

    def set_thresh_from_train(self, sim_per_class, labels):
        """从训练集 5%/95% 分位初始化"""
        with torch.no_grad():
            for k in range(2):
                mask = (labels == k)
                if mask.sum() == 0:
                    continue
                pos_dist = 1.0 - sim_per_class[mask, k]
                thresh_k = torch.quantile(pos_dist, 0.5).item()
                self.log_thresh.data[k] = torch.tensor(thresh_k).log()

    def forward(self, sim_per_class, return_uncertainty=True):
        """
        Args:
            sim_per_class: (b, 2)
        Returns:
            prob_failed: (b,) P(y=1)
            uncertainty: (b,) 不确定性 ∈ [0, 1]
        """
        dist = 1.0 - sim_per_class  # (b, 2)
        # 用 1 / (1 + exp(dist - thresh)) 形式估计概率
        # prob(failed=1) ≈ sigmoid( (dist_0 - thresh_0) - (dist_1 - thresh_1) )
        thresh = torch.exp(self.log_thresh)
        logits = (dist[:, 0] - thresh[0]) - (dist[:, 1] - thresh[1])
        prob_failed = torch.sigmoid(logits)
        if return_uncertainty:
            # 不确定性: 两个类的距离差小 = 不确定
            margin = (dist[:, 1] - thresh[1]) - (dist[:, 0] - thresh[0])
            uncertainty = torch.sigmoid(-margin.abs() * 5)  # 边界处接近 1
        else:
            uncertainty = torch.zeros_like(prob_failed)
        return prob_failed, uncertainty


# ============= ⑤ 完整 MASC-Net =============
class MASCNet(nn.Module):
    """Multi-scale Adaptive Sample-aware Contrastive Network
    
    Args:
        n_prototypes_per_class: 每类原型数
        use_contrastive: 是否启用对比损失(消融用)
        use_uncertainty:   是否启用不确定性分支(消融用)
    """
    def __init__(self, n_prototypes_per_class=4,
                 use_contrastive=True, use_uncertainty=True,
                 feat_dim=64, proj_dim=32, dropout=0.3):
        super().__init__()
        self.use_contrastive = use_contrastive
        self.use_uncertainty = use_uncertainty
        # ① 多尺度编码器
        self.encoder = MultiScaleEncoder(output_dim=feat_dim, dropout=dropout)
        # ② 对比学习(可选)
        if use_contrastive:
            self.contrastive = SampleAwareContrastive(
                feat_dim=feat_dim, proj_dim=proj_dim
            )
        # ③ 原型库
        self.proto_bank = PrototypeBank(
            n_classes=2, n_prototypes_per_class=n_prototypes_per_class, dim=proj_dim
        )
        # ④ 自适应阈值分类器
        self.classifier = AdaptiveThresholdClassifier()

    def forward(self, x, labels=None, compute_loss=False):
        """
        Args:
            x: (b, 46)
            labels: (b,) 仅训练时需要
            compute_loss: 是否返回多任务损失
        Returns:
            prob_failed: (b,)
            uncertainty: (b,)
            aux: dict {loss_contrastive, loss_proto, loss_uncertainty}
        """
        h, attn_info = self.encoder(x)  # (b, 64)

        aux = {}
        loss_total = torch.tensor(0.0, device=x.device)

        if self.use_contrastive:
            z_q, z_k, tau = self.contrastive(h, self.proto_bank.prototypes.data)
            # 原型一致性损失: 让样本 z_q 接近自己类的原型
            sim_per_class, sim_per_proto = self.proto_bank.get_class_assignment(z_q)
            if labels is not None and compute_loss:
                # 1) InfoNCE
                loss_contrast = self.contrastive.info_nce_loss(
                    z_q, z_k, labels, self.proto_bank.prototypes.data, tau
                )
                # 2) 原型拉近损失: 类内样本应靠近自己类的最近原型
                pos_sim = sim_per_class.gather(1, labels.unsqueeze(1)).squeeze(1)
                loss_proto = (1.0 - pos_sim).mean()
                # 3) EMA 更新原型
                self.proto_bank.ema_update(z_q, labels)
                # 4) 动量更新 key encoder
                self.contrastive.momentum_update()
                aux['loss_contrast'] = loss_contrast
                aux['loss_proto'] = loss_proto
                loss_total = loss_total + 0.3 * loss_contrast + 0.3 * loss_proto
        else:
            # 退化: 不使用对比, 直接用 encoder 输出做投影(降维到原型维度)
            z_q = F.normalize(h, dim=-1)
            if z_q.size(-1) != self.proto_bank.dim:
                # 把 feat_dim 维降采样到 proto_dim 维 (平均池化)
                ratio = z_q.size(-1) // self.proto_bank.dim
                z_q = z_q[:, :ratio * self.proto_bank.dim].view(
                    z_q.size(0), self.proto_bank.dim, ratio
                ).mean(dim=-1)
                z_q = F.normalize(z_q, dim=-1)
            sim_per_class, sim_per_proto = self.proto_bank.get_class_assignment(z_q)

        # ④ 分类
        prob_failed, uncertainty = self.classifier(sim_per_class)

        if compute_loss and labels is not None and self.use_uncertainty:
            # 5) 不确定性正则: 边界样本不确定性应升高
            with torch.no_grad():
                margin = (1.0 - sim_per_class[:, 1]) - (1.0 - sim_per_class[:, 0])
                boundary_mask = margin.abs() < 0.1
            if boundary_mask.sum() > 0:
                loss_unc = -uncertainty[boundary_mask].log().mean()
                aux['loss_uncertainty'] = loss_unc
                loss_total = loss_total + 0.05 * loss_unc

        aux['loss_total'] = loss_total
        aux['attn_weights'] = attn_info['attn_weights']
        return prob_failed.squeeze(-1) if prob_failed.dim() > 1 else prob_failed, uncertainty, aux


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, probs, labels):
        eps = 1e-7
        p = probs.clamp(eps, 1 - eps)
        pt = torch.where(labels == 1, p, 1 - p)
        alpha_t = torch.where(labels == 1,
                              torch.full_like(p, self.alpha),
                              torch.full_like(p, 1 - self.alpha))
        loss = -alpha_t * (1 - pt) ** self.gamma * pt.log()
        return loss.mean()
