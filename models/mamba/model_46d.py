"""
Mamba-46d 简化模型 (v2 - 修复 batch 维度不匹配)

输入: X shape=(n_samples, 46), 输出: P(passed=1)
架构: Linear(1→d_model) → [Conv1d + GLU-gate] x N → mean+max pool → FC → sigmoid
"""
import torch
import torch.nn as nn
import math


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class GatedConvBlock(nn.Module):
    """
    Mamba 风格的简化 block:
      - 输入投影 (d_model -> d_inner=2*d_model)
      - 深度卷积 (捕获局部时序)
      - 门控 (类似 Mamba 的 selective gating)
      - 输出投影
      - 残差连接
    """
    def __init__(self, d_model, expand=2, d_conv=3):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand

        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)   # 同时产出 value 和 gate
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner
        )
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.act = SiLU()

    def forward(self, x):
        # x: (B, L, d_model)
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)                # (B, L, 2*d_inner)
        x_val, x_gate = xz.chunk(2, dim=-1) # each (B, L, d_inner)
        # 深度卷积
        x_val = self.conv1d(x_val.transpose(1, 2))[:, :, :x.size(1)].transpose(1, 2)
        x_val = self.act(x_val)
        # Mamba 风格 gating: SiLU(x_val) * sigmoid(x_gate)
        y = x_val * torch.sigmoid(x_gate)
        y = self.out_proj(y)
        return y + residual


class Mamba46dClassifier(nn.Module):
    """
    Mamba 处理 46d 特征的简化分类器

    输入: (batch, 46) - 46d 手工特征
    处理: reshape (batch, 46, 1) → Linear(1→d_model) → 堆叠 GatedConvBlock → pool → FC → sigmoid
    """
    def __init__(self, input_dim=46, d_model=48, n_layers=4, expand=2, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.embed = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([
            GatedConvBlock(d_model, expand=expand) for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        # x: (batch, 46)
        x = x.unsqueeze(-1)              # (batch, 46, 1)
        x = self.embed(x)                 # (batch, 46, d_model)
        for blk in self.blocks:
            x = blk(x)
        x = self.final_norm(x)            # (batch, 46, d_model)
        # 双池化 (mean + max) - 比只用 mean 强
        mean_pool = x.mean(dim=1)         # (batch, d_model)
        max_pool = x.max(dim=1).values    # (batch, d_model)
        combined = torch.cat([mean_pool, max_pool], dim=-1)  # (batch, 2*d_model)
        return torch.sigmoid(self.classifier(combined)).squeeze(-1)


def create_model(device='cuda'):
    model = Mamba46dClassifier()
    model.to(device)
    return model