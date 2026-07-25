"""
Mamba-Attention 融合模型 (CREAM variant)

架构:
  7维事件序列 (event_type, time_interval, deadline_dist)
        │
        ├──[A] Mamba 路径: 线性扫描长序列 → 64 维局部特征
        │     • 复用 FullMambaStudent 的 S6Block 骨架 (n_layers=3)
        │     • 多尺度池化 (fine 100步 / medium 全局 / coarse 按part分组)
        │
        ├──[B] Attention 路径: 自注意力捕捉远距离依赖 → 64 维全局特征
        │     • 2层 MultiHeadAttention (4 heads)
        │     • 降采样到 256 步后做 attention, O(n²) 可控
        │     • CLS token + mean pooling
        │
        ↓
  [G] 门控融合 (Adaptive Gate)
        • g = σ(W_g · [mamba; attn])
        • fused = g ⊙ mamba + (1-g) ⊙ attn     ← 64 维
        ↓
  [Prototype] 4 个可学习原型 (soft assignment)
        ↓
  [Risk Head] FC → 2 (passed/failed)
        ↓ 输出 risk logits

输入:
  batch['event_types']    (B, L) Long
  batch['time_intervals'] (B, L) Float
  batch['deadline_dists'] (B, L) Float
  batch['part_ids']       (B, L) Long

说明:
  - 与 FullMambaStudent 的区别: 多了 Attention 分支 + 门控融合
  - 完全使用 7 维原始特征 (event_type, time, deadline, part)
  - 兼容现有 pretrain (next-event prediction) 和 finetune 接口
"""

import os, sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 复用 model.py 中的基本构件 (SiLU / RMSNorm / S6Block / MambaBlock / MambaEncoder)
from models.mamba.model import (
    SiLU, RMSNorm, S6Block, MambaBlock, MambaEncoder,
    EVENT_TYPES,
)


# =====================================================================
# 共享事件编码器
# =====================================================================
class SharedEventEncoder(nn.Module):
    """event_type (idx) + time_interval + deadline_dist + part_id → d_model"""
    def __init__(self, n_event_types=7, d_model=64, max_parts=8):
        super().__init__()
        self.ev = nn.Embedding(n_event_types, 24)
        self.te = nn.Linear(1, 12)
        self.de = nn.Linear(1, 12)
        self.pe = nn.Embedding(max_parts, 16)              # part id 0..7
        self.proj = nn.Linear(24 + 12 + 12 + 16, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, et, ti, dd, pi):
        x = torch.cat([
            self.ev(et),
            self.te(ti.unsqueeze(-1)),
            self.de(dd.unsqueeze(-1)),
            self.pe(pi),
        ], dim=-1)
        return self.norm(self.proj(x))


# =====================================================================
# A. Mamba 路径
# =====================================================================
class MambaPath(nn.Module):
    """长序列线性扫描 + 多尺度池化 → 64 维表征"""
    def __init__(self, d_model=64, n_layers=3, d_state=16):
        super().__init__()
        self.mamba = MambaEncoder(d_model=d_model, n_layers=n_layers, d_state=d_state)
        self.fine_pool = nn.AdaptiveAvgPool1d(1)
        self.medium_pool = nn.AdaptiveAvgPool1d(1)
        self.coarse_proj = nn.Linear(d_model, d_model)
        self.se_gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.fuse = nn.Linear(d_model * 3, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, part_ids):
        """
        x: (B, L, d_model)
        part_ids: (B, L)
        Returns: (B, d_model)
        """
        h = self.mamba(x)                                # (B, L, d)
        # Fine: 每 100 步窗口均值
        if h.shape[1] >= 100:
            h_t = h.transpose(1, 2)                     # (B, d, L)
            h_t = F.adaptive_avg_pool1d(h_t, min(64, h.shape[1]))
            fine = h_t.mean(dim=-1)                     # (B, d)
        else:
            fine = h.mean(dim=1)                        # (B, d)

        # Medium: 全局均值
        medium = h.mean(dim=1)                           # (B, d)

        # Coarse: 按 part 分组
        coarse_parts = []
        for p in range(1, 8):
            mask = (part_ids == p).unsqueeze(-1).float()    # (B, L, 1)
            if mask.sum(dim=1).min().item() == 0:
                coarse_parts.append(torch.zeros(h.shape[0], h.shape[-1], device=h.device))
            else:
                p_mean = (h * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
                coarse_parts.append(p_mean)
        coarse_stack = torch.stack(coarse_parts, dim=1) # (B, 7, d)
        coarse = self.coarse_proj(coarse_stack.mean(dim=1))  # (B, d)

        cat = torch.cat([fine, medium, coarse], dim=-1)  # (B, d*3)
        se = self.se_gate(cat)
        fused = self.fuse(cat) * se
        return self.norm(fused)                          # (B, d)


# =====================================================================
# B. Attention 路径
# =====================================================================
class AttentionPath(nn.Module):
    """自注意力捕捉远距离依赖 → 64 维表征"""
    def __init__(self, d_model=64, n_layers=2, n_heads=4, downsample_steps=256):
        super().__init__()
        self.downsample_steps = downsample_steps
        # CLS token + positional embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, downsample_steps + 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, L, d_model)
        Returns: (B, d_model)
        """
        B, L, D = x.shape
        # 降采样到固定步数 (avg_pool over time)
        if L != self.downsample_steps:
            # (B, D, L) → avg_pool1d → (B, D, steps)
            x_ds = F.adaptive_avg_pool1d(x.transpose(1, 2), self.downsample_steps)
            x_ds = x_ds.transpose(1, 2)                  # (B, steps, D)
        else:
            x_ds = x
        # CLS token
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, D)
        seq = torch.cat([cls, x_ds], dim=1)             # (B, steps+1, D)
        seq = seq + self.pos_embed[:, :seq.shape[1], :]
        out = self.encoder(seq)                         # (B, steps+1, D)
        # 取 CLS 输出 + 全局均值混合
        cls_out = out[:, 0, :]                          # (B, D)
        mean_out = out[:, 1:, :].mean(dim=1)            # (B, D)
        return self.norm(cls_out + mean_out)            # (B, D)


# =====================================================================
# G. 门控融合
# =====================================================================
class GatedFusion(nn.Module):
    """g = σ(W_g · [mamba; attn]), fused = g ⊙ mamba + (1-g) ⊙ attn"""
    def __init__(self, d_model=64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, mamba_repr, attn_repr):
        cat = torch.cat([mamba_repr, attn_repr], dim=-1)
        g = self.gate(cat)                               # (B, d)
        fused = g * mamba_repr + (1.0 - g) * attn_repr
        fused = self.proj(fused)
        return self.norm(fused), g                       # 返回 gating 权重可解释


# =====================================================================
# 主模型
# =====================================================================
class MambaAttentionFusion(nn.Module):
    """
    Mamba + Attention 门控融合模型 (CREAM variant)
    使用 7 维原始事件特征
    """
    def __init__(self,
                 n_event_types=7,
                 d_model=64,
                 n_mamba_layers=3,
                 n_attn_layers=2,
                 n_heads=4,
                 d_state=16,
                 n_prototypes=4,
                 max_seq_len=2000,
                 downsample_steps=256):
        super().__init__()
        self.encoder = SharedEventEncoder(n_event_types, d_model)
        self.input_proj = nn.Linear(d_model, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        self.mamba_path = MambaPath(d_model, n_mamba_layers, d_state)
        self.attn_path = AttentionPath(d_model, n_attn_layers, n_heads, downsample_steps)
        self.fusion = GatedFusion(d_model)

        # 原型
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, d_model) * 0.1)

        # Heads
        #   event_head: 下一事件预测 (预训练用)
        #   risk_head: 风险二分类 (微调用)
        self.event_head = nn.Linear(d_model, n_event_types)
        self.risk_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
        )
        self.dropout = nn.Dropout(0.3)

    def forward(self, batch, return_repr=False):
        et = batch['event_types']
        ti = batch['time_intervals']
        dd = batch['deadline_dists']
        pi = batch.get('part_ids', torch.zeros_like(et))

        # 编码
        h = self.encoder(et, ti, dd, pi)                 # (B, L, d)
        h = self.input_proj(h)
        h = self.input_norm(h)

        # 双路径
        mamba_repr = self.mamba_path(h, pi)              # (B, d)
        attn_repr = self.attn_path(h)                    # (B, d)
        fused, gate = self.fusion(mamba_repr, attn_repr) # (B, d)

        # 原型
        dists = torch.cdist(fused, self.prototypes)      # (B, P)
        proto_w = torch.softmax(-dists, dim=-1)          # (B, P)
        proto_repr = torch.matmul(proto_w, self.prototypes)  # (B, d)
        fused_proto = fused + proto_repr                 # (B, d)

        # Heads
        fused_drop = self.dropout(fused_proto)
        risk_logits = self.risk_head(fused_drop)         # (B, 2)
        event_logits = self.event_head(fused)            # (B, n_event_types)

        out = {'risk': risk_logits, 'event': event_logits}
        if return_repr:
            out.update({
                'mamba_repr': mamba_repr,
                'attn_repr': attn_repr,
                'fused': fused,
                'gate': gate,
                'proto_w': proto_w,
            })
        return out


def create_model(d_model=64, n_mamba_layers=3, n_attn_layers=2,
                 n_heads=4, d_state=16, n_prototypes=4,
                 max_seq_len=2000, downsample_steps=256):
    return MambaAttentionFusion(
        d_model=d_model,
        n_mamba_layers=n_mamba_layers,
        n_attn_layers=n_attn_layers,
        n_heads=n_heads,
        d_state=d_state,
        n_prototypes=n_prototypes,
        max_seq_len=max_seq_len,
        downsample_steps=downsample_steps,
    )


if __name__ == '__main__':
    # 简单冒烟测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model().to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    batch = {
        'event_types': torch.randint(0, 7, (4, 500), device=device),
        'time_intervals': torch.rand(4, 500, device=device),
        'deadline_dists': torch.rand(4, 500, device=device),
        'part_ids': torch.randint(0, 7, (4, 500), device=device),
    }
    out = model(batch, return_repr=True)
    print(f"risk shape: {out['risk'].shape}")
    print(f"event shape: {out['event'].shape}")
    print(f"gate mean: {out['gate'].mean().item():.4f}")
    print("✓ 烟雾测试通过")
