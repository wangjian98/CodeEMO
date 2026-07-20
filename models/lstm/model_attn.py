"""
LSTM + Attention Pooling 模型 (TabTransformer 风格)

架构:
  Input (batch, n_features)
    ↓
  Linear (n_features → n_features × embed_dim_per_feature)  # 每个特征独立嵌入
    ↓
  Reshape to (batch, n_features, embed_dim)
    ↓
  Multi-Head Self-Attention (特征间交互)
    ↓
  Add & Norm
    ↓
  Feed-Forward + Add & Norm
    ↓
  Attention Pooling (learnable query)
    ↓
  Classifier
"""
import torch
import torch.nn as nn
import math


class AttentionPooling(nn.Module):
    """可学习的注意力池化:用一个可学习 query 对输入序列做 attention"""
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (batch, seq_len, dim)
        b, s, d = x.shape
        q = self.query.expand(b, -1, -1)  # (batch, 1, dim)
        k = self.key_proj(x)              # (batch, seq_len, dim)
        v = self.value_proj(x)            # (batch, seq_len, dim)
        scores = torch.einsum('bqd,bsd->bqs', q, k) / math.sqrt(d)
        weights = torch.softmax(scores, dim=-1)  # (batch, 1, seq_len)
        out = torch.einsum('bqs,bsd->bqd', weights, v).squeeze(1)  # (batch, dim)
        return out


class LSTMClassifierWithAttention(nn.Module):
    """46/120维特征 + 特征级 self-attention + attention pooling"""
    def __init__(self, input_dim=46, embed_dim=16, n_heads=4,
                 ff_dim=64, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # 特征嵌入: 每个特征 → embed_dim 维向量
        self.feature_embed = nn.Linear(1, embed_dim)

        # 位置编码 (可学习)
        self.pos_embedding = nn.Parameter(torch.randn(1, input_dim, embed_dim) * 0.02)

        # 多头自注意力 (让特征间交互)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # 注意力池化
        self.pool = AttentionPooling(embed_dim)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, input_dim)
        b = x.shape[0]

        # 每个特征 → embed_dim
        x_tok = x.unsqueeze(-1)              # (batch, input_dim, 1)
        x_emb = self.feature_embed(x_tok)    # (batch, input_dim, embed_dim)
        x_emb = x_emb + self.pos_embedding  # 加位置编码

        # 自注意力
        attn_out, _ = self.attn(x_emb, x_emb, x_emb)
        x_emb = self.norm1(x_emb + self.dropout(attn_out))

        # FFN
        ff_out = self.ff(x_emb)
        x_emb = self.norm2(x_emb + self.dropout(ff_out))

        # 注意力池化
        pooled = self.pool(x_emb)            # (batch, embed_dim)

        # 分类
        return self.classifier(pooled)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LSTMClassifierMeanPool(nn.Module):
    """46/120维特征 + self-attention + mean pooling (用于对比)"""
    def __init__(self, input_dim=46, embed_dim=16, n_heads=4,
                 ff_dim=64, dropout=0.3):
        super().__init__()
        self.feature_embed = nn.Linear(1, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, input_dim, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b = x.shape[0]
        x_tok = x.unsqueeze(-1)
        x_emb = self.feature_embed(x_tok)
        x_emb = x_emb + self.pos_embedding
        attn_out, _ = self.attn(x_emb, x_emb, x_emb)
        x_emb = self.norm1(x_emb + self.dropout(attn_out))
        ff_out = self.ff(x_emb)
        x_emb = self.norm2(x_emb + self.dropout(ff_out))
        pooled = x_emb.mean(dim=1)  # mean pooling
        return self.classifier(pooled)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires.grad) if False else \
               sum(p.numel() for p in self.parameters() if p.requires_grad)