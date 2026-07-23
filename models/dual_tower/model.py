"""
BiLSTM + Mamba 双塔融合模型

两种输入模态:
  1. 46维统计特征 → BiLSTM Tower (序列视角: (46,) → (1,46) 当作1步序列)
  2. 7维事件序列  → Mamba Tower  (选择性状态空间建模)
  3. 46维统计特征 → 特征塔 MLP  (直接非线性变换)

融合: concat → 风险分类

架构:
  46-dim ──→ BiLSTMTower ──→ 128-dim ──┐
  7-dim序列 ─→ MambaTower  ──→  64-dim ─┼→ concat (238-dim) → Fusion → 2
  46-dim ──→ FeatTower    ──→  46-dim ──┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─── Mamba 轻量组件 ────────────────────────────────────────────

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * output


class S6Block(nn.Module):
    """S6 选择性状态空间块"""
    def __init__(self, d_model, d_state=8, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        self.input_proj = nn.Linear(d_model, self.d_inner)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                padding=d_conv - 1, groups=self.d_inner)
        self.dt_rank = math.ceil(self.d_inner / 16) if self.d_inner > 16 else self.d_inner
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)

        self.dt_init = nn.Parameter(torch.empty(self.dt_rank))
        nn.init.uniform_(self.dt_init, -1.0, 0.0)

        # A 矩阵: 负对角，保证 dA = exp(dt*A) ∈ (0,1) 数值稳定
        A = -torch.abs(torch.arange(1, d_state + 1, dtype=torch.float32))
        A = A.unsqueeze(0).expand(self.d_inner, -1).clone()
        self.A = nn.Parameter(A)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.output_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.act = SiLU()

    def selective_scan(self, x, dt, A, B, C, D):
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        dt = F.softplus(dt)
        dt_exp = dt.unsqueeze(-1)
        A_exp = A.unsqueeze(0).unsqueeze(0)
        dA = torch.exp(dt_exp * A_exp).clamp(max=1.0 - 1e-6)
        B_exp = B.unsqueeze(2)
        dB = (dt_exp * B_exp).clamp(max=1e6)

        h = torch.zeros(batch, d_inner, d_state, dtype=x.dtype, device=x.device)
        ys = []
        for t in range(seq_len):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            y = torch.bmm(h, C[:, t].unsqueeze(-1)).squeeze(-1)
            ys.append(y)
        y = torch.stack(ys, dim=1)
        return y + x * D

    def forward(self, x):
        batch, seq_len, _ = x.shape
        x_inner = self.input_proj(x)
        x_conv = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len].transpose(1, 2)
        x_conv = self.act(x_conv)

        x_flat = x_conv.reshape(-1, self.d_inner)
        x_params = self.x_proj(x_flat)
        dt, B_seq, C_seq = torch.split(
            x_params, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = dt.reshape(batch, seq_len, self.dt_rank)
        B_seq = B_seq.reshape(batch, seq_len, self.d_state)
        C_seq = C_seq.reshape(batch, seq_len, self.d_state)

        if self.dt_rank < self.d_inner:
            dt_padded = torch.zeros(batch, seq_len, self.d_inner,
                                    device=dt.device, dtype=dt.dtype)
            dt_padded[:, :, :self.dt_rank] = dt
            dt = dt_padded

        y = self.selective_scan(x_conv, dt, self.A, B_seq, C_seq, self.D)
        return self.output_proj(y)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=8, d_conv=4, expand=2):
        super().__init__()
        self.mixer = S6Block(d_model, d_state, d_conv, expand)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        return self.mixer(self.norm(x)) + x


# ─── 共享事件编码器 ────────────────────────────────────────────

class EventEncoder(nn.Module):
    """7维事件编码器 (event_type embedding + time + deadline → d_model)"""
    def __init__(self, n_event_types=7, d_model=64):
        super().__init__()
        self.event_embed = nn.Embedding(n_event_types, 16)
        self.time_embed = nn.Linear(1, 8)
        self.deadline_embed = nn.Linear(1, 8)
        self.input_proj = nn.Linear(16 + 8 + 8, d_model)

    def forward(self, event_types, time_intervals, deadline_dists):
        ev = self.event_embed(event_types)
        te = self.time_embed(time_intervals.unsqueeze(-1))
        de = self.deadline_embed(deadline_dists.unsqueeze(-1))
        return self.input_proj(torch.cat([ev, te, de], dim=-1))


# ─── Tower A: BiLSTM (处理 46维特征，reshape为伪序列) ─────────

class BiLSTMTower(nn.Module):
    """
    Tower A: 46维统计特征 → 双向 LSTM → 128维表征

    将 46维向量 reshape 为 (1, 46) 的单步序列，
    BiLSTM  bidirectional 能看到整个向量的前向+后向信息，
    比直接 MLP 更能捕获特征间的依赖关系。
    """
    def __init__(self, input_dim=46, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 将 46维 映射到 d_model=hidden_dim 作为 LSTM 输入
        self.feat_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        # seq_mean(128) + fwd_last(64) + bwd_last(64) = 256 → 128
        self.output_proj = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * 2  # 128

    def forward(self, feat_x):
        """
        feat_x: (batch, 46) 统计特征
        Returns: (batch, 128)
        """
        # (batch, 46) → (batch, 1, hidden_dim)
        x = self.feat_proj(feat_x).unsqueeze(1)
        lstm_out, _ = self.lstm(x)  # (batch, 1, 128)
        seq_mean = lstm_out.mean(dim=1)             # (batch, 128)
        fwd_last = lstm_out[:, -1, :self.hidden_dim]       # (batch, 64)
        bwd_last = lstm_out[:, -1, self.hidden_dim:]        # (batch, 64)
        repr = torch.cat([seq_mean, fwd_last, bwd_last], dim=-1)  # (batch, 256)
        repr = self.output_proj(repr)                   # (batch, 128)
        return self.dropout(repr)


# ─── Tower B: Mamba (处理 7维事件序列) ─────────────────────────

class MambaTower(nn.Module):
    """
    Tower B: 7维事件序列 → Mamba 选择性扫描 → 64维表征
    """
    def __init__(self, d_model=48, n_layers=2, d_state=8, dropout=0.2):
        super().__init__()
        self.encoder = EventEncoder(n_event_types=7, d_model=d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.output_dim = d_model  # 48

    def forward(self, batch, max_len=None):
        et = batch['event_types']
        ti = batch['time_intervals']
        dd = batch['deadline_dists']
        pi = batch.get('part_ids', torch.zeros_like(et))

        if max_len:
            et, ti, dd, pi = et[:, :max_len], ti[:, :max_len], dd[:, :max_len], pi[:, :max_len]

        x = self.encoder(et, ti, dd)  # (batch, seq, d_model)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)  # (batch, seq, d_model)

        seq_mean = x.mean(dim=1)
        seq_last = x[:, -1, :]

        # 按 part 分组均值
        part_means = []
        for p in range(1, 8):
            mask = (pi == p)
            if mask.any():
                pm = (x * mask.unsqueeze(-1)).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
            else:
                pm = torch.zeros_like(seq_mean)
            part_means.append(pm)
        part_repr = torch.stack(part_means, dim=1).mean(dim=1)

        return (seq_mean + seq_last + part_repr) / 3  # (batch, d_model)


# ─── Tower C: 特征 MLP (处理 46维统计特征) ─────────────────────

class FeatTower(nn.Module):
    """
    Tower C: 46维统计特征 → 3层 MLP → 46维表征

    与 BiLSTM Tower 互补:
      - BiLSTM: 捕获特征间序列依赖 (双向)
      - MLP:    直接非线性变换，无偏置
    """
    def __init__(self, input_dim=46, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        self.output_dim = input_dim  # 46

    def forward(self, feat_x):
        return self.net(feat_x)  # (batch, 46)


# ─── 双塔融合模型 ─────────────────────────────────────────────

class DualTowerModel(nn.Module):
    """
    三塔融合: BiLSTM(46) + Mamba(7-dim序列) + MLP(46)

    三个 tower 各司其职:
      - BiLSTM: 对 46维特征做双向序列建模
      - Mamba:  对 7维事件序列做选择性状态空间建模
      - MLP:    对 46维特征做直接非线性变换

    融合: 128 + 48 + 46 = 222 维 → 风险分类
    """
    def __init__(self,
                 bilstm_hidden=64, bilstm_layers=2,
                 mamba_d_model=48, mamba_layers=2, mamba_d_state=8,
                 feat_hidden=64,
                 max_seq_len=500,
                 dropout=0.3):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.bilstm_tower = BiLSTMTower(
            input_dim=46, hidden_dim=bilstm_hidden,
            num_layers=bilstm_layers, dropout=dropout
        )
        self.mamba_tower = MambaTower(
            d_model=mamba_d_model, n_layers=mamba_layers,
            d_state=mamba_d_state, dropout=dropout
        )
        self.feat_tower = FeatTower(
            input_dim=46, hidden_dim=feat_hidden, dropout=dropout
        )

        combined_dim = (self.bilstm_tower.output_dim +
                        self.mamba_tower.output_dim +
                        self.feat_tower.output_dim)
        # 128 + 48 + 46 = 222

        self.fusion = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )
        # Xavier 初始化
        for m in self.fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feat_x, mamba_batch):
        """
        Args:
            feat_x: (batch, 46) 统计特征
            mamba_batch: dict (event_types, time_intervals, deadline_dists, part_ids)
        Returns:
            logits: (batch, 2)
        """
        bilstm_repr = self.bilstm_tower(feat_x)     # (B, 128)
        mamba_repr = self.mamba_tower(mamba_batch, max_len=self.max_seq_len)  # (B, 48)
        feat_repr = self.feat_tower(feat_x)         # (B, 46)

        combined = torch.cat([bilstm_repr, mamba_repr, feat_repr], dim=-1)  # (B, 222)
        return self.fusion(combined)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
