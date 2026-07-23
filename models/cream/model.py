"""
CREAM: Contrastive REgularized Attention Mixer

核心设计思路（基于BGM-Net实验问题的根因分析）:

问题1: BGM-Net注意力退化为均匀分布(CV<0.4)
  → 不用熵做注意力,改用可学习的channel attention(类似SE-Net的squeeze-excitation)

问题2: Precision=0.74偏低(假阳性多)
  → 加入对比学习损失,拉开passed/failed的表示空间距离

问题3: 单模型在precision-recall tradeoff上只能顾一端
  → 用multi-objective head: 分类头+排序头,同时优化BCE和margin ranking loss

问题4: 46维特征对473样本偏多
  → 加入特征瓶颈层(bottleneck),让模型自动压缩到低维表示

架构:
  Input(46d) → Feature Bottleneck(46→40→24) → SE-Attention(24) →
  Classifier Head(24→1→sigmoid) + Contrastive Embedding(24→16 for loss)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """SE-Net风格的channel attention, 不依赖熵值"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, channels)
        s = self.squeeze(x.unsqueeze(-1)).squeeze(-1)  # (batch, channels)
        e = self.excitation(s)  # (batch, channels)
        return x * e  # channel-wise reweighting


class CREAM(nn.Module):
    def __init__(self, input_dim=46, bottleneck_dim=24, embed_dim=16,
                 dropout=0.2, use_se=True, use_contrastive=True,
                 margin=0.5, contrastive_weight=0.1):
        super().__init__()
        self.use_se = use_se
        self.use_contrastive = use_contrastive
        self.margin = margin
        self.contrastive_weight = contrastive_weight

        # Feature bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 40),
            nn.BatchNorm1d(40),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(40, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
        )

        # SE attention
        if use_se:
            self.se = SqueezeExcitation(bottleneck_dim, reduction=4)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )

        # Contrastive embedding head (for auxiliary loss)
        if use_contrastive:
            self.projector = nn.Sequential(
                nn.Linear(bottleneck_dim, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
            )

        self.last_embedding = None

    def forward(self, x):
        h = self.encoder(x)
        if self.use_se:
            h = self.se(h)
        logit = self.classifier(h).squeeze(-1)
        prob = torch.sigmoid(logit)

        if self.use_contrastive:
            self.last_embedding = self.projector(h)
        else:
            self.last_embedding = h

        return prob, logit, self.last_embedding

    def compute_loss(self, prob, logit, embedding, label, bce_loss):
        """总损失 = BCE + 对比损失"""
        loss = bce_loss(logit, label.float())

        if self.use_contrastive:
            # Margin-based contrastive loss
            # 拉近同类(passed-passed), 推远异类(passed-failed)
            pos_mask = label == 1
            neg_mask = label == 0

            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                pos_emb = embedding[pos_mask]  # (n_pos, dim)
                neg_emb = embedding[neg_mask]  # (n_neg, dim)

                # 计算每对pos-neg的距离
                # pos centroid
                pos_center = pos_emb.mean(dim=0, keepdim=True)  # (1, dim)
                neg_center = neg_emb.mean(dim=0, keepdim=True)

                # 类内紧凑性: pos样本到pos中心的平均距离
                pos_intra = (F.pairwise_distance(pos_emb, pos_center.expand_as(pos_emb))).mean()
                neg_intra = (F.pairwise_distance(neg_emb, neg_center.expand_as(neg_emb))).mean()
                intra_loss = (pos_intra + neg_intra) / 2

                # 类间分离度: 中心间距
                inter_dist = F.pairwise_distance(pos_center, neg_center)

                # margin loss: max(0, margin + intra - inter)
                contrastive_loss = F.relu(self.margin + intra_loss - inter_dist)

                loss = loss + self.contrastive_weight * contrastive_loss

        return loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
