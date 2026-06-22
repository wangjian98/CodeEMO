"""
Mamba-SSM (Selective State Space Model) 核心模型

基于论文: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Albert Gu & Tri Dao, 2023)

实现 S6 (Selective SSM) 机制和 Mamba block。

包含:
  - S6Block: 选择性状态空间核心机制
  - MambaBlock: 带残差连接的 Mamba block
  - MambaEncoder: 多层 Mamba 编码器
  - SimplifiedMambaStudent: 完整的6步流程模型 (CPU版本)
  - FullMambaStudent: 完整的6步流程模型 (GPU版本)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class SiLU(nn.Module):
    """SiLU (Sigmoid Linear Unit) / Swish 激活函数"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * output


class S6Block(nn.Module):
    """
    S6: 选择性状态空间模型核心机制

    Mamba 的关键创新: 参数 A, B, C 从输入计算得出 (选择性)，
    不同于经典 SSM 中的固定参数。这使得模型可以动态决定记忆什么和遗忘什么。

    状态方程: h_k = A * h_{k-1} + B * x_k
    输出方程: y_k = C * h_k
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)

        # 输入投影
        self.input_proj = nn.Linear(d_model, self.d_inner)

        # 局部卷积
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # SSM 参数 (选择性 - 从输入计算)
        self.dt_rank = math.ceil(self.d_inner / 16) if self.d_inner > 16 else self.d_inner

        # 投影 x -> (dt, B, C)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)

        # dt 初始化
        self.dt_init = nn.Parameter(torch.empty(self.dt_rank))
        nn.init.uniform_(self.dt_init, -1.0, 0.0)

        # A 矩阵 (状态转移)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_inner, -1).clone().log()
        self.A = nn.Parameter(A)

        # D 项 (跳跃连接)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 输出投影
        self.output_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.act = SiLU()

    def selective_scan(self, x, dt, A, B, C, D):
        """
        并行选择性扫描算法

        x: (batch, seq_len, d_inner)
        dt: (batch, seq_len, d_inner)
        A: (d_inner, d_state)
        B: (batch, seq_len, d_state)
        C: (batch, seq_len, d_state)
        D: (d_inner,)

        Returns: (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        # 离散化
        dt = F.softplus(dt)
        dt_expanded = dt.unsqueeze(-1)  # (batch, seq, d_inner, 1)
        A_expanded = A.unsqueeze(0).unsqueeze(0)  # (1, 1, d_inner, d_state)
        dA = torch.exp(dt_expanded * A_expanded)  # (batch, seq, d_inner, d_state)

        B_expanded = B.unsqueeze(2)  # (batch, seq, 1, d_state)
        dB = dt_expanded * B_expanded  # (batch, seq, d_inner, d_state)

        # 扫描计算
        h = torch.zeros(batch, d_inner, d_state, dtype=x.dtype, device=x.device)
        ys = []

        for t in range(seq_len):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            y = torch.bmm(h, C[:, t].unsqueeze(-1)).squeeze(-1)
            ys.append(y)

        y = torch.stack(ys, dim=1)
        y = y + x * D

        return y

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # 输入投影
        x_inner = self.input_proj(x)

        # 局部卷积
        x_conv = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.act(x_conv)

        # 选择性 SSM 参数
        x_flat = x_conv.reshape(-1, self.d_inner)
        x_params = self.x_proj(x_flat)

        dt, B_seq, C_seq = torch.split(
            x_params,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )

        dt = dt.reshape(batch, seq_len, self.dt_rank)
        B_seq = B_seq.reshape(batch, seq_len, self.d_state)
        C_seq = C_seq.reshape(batch, seq_len, self.d_state)

        # 投影 dt 到 d_inner
        if self.dt_rank < self.d_inner:
            dt_padded = torch.zeros(batch, seq_len, self.d_inner, device=dt.device, dtype=dt.dtype)
            dt_padded[:, :, :self.dt_rank] = dt
            dt = dt_padded

        # 选择性扫描
        y = self.selective_scan(x_conv, dt, self.A, B_seq, C_seq, self.D)

        # 输出投影
        y = self.output_proj(y)

        return y


class MambaBlock(nn.Module):
    """带残差连接的 Mamba block"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mixer = S6Block(d_model, d_state, d_conv, expand)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        return self.mixer(self.norm(x)) + x


class MambaEncoder(nn.Module):
    """
    Mamba 编码器 - 多层 Mamba block 堆叠
    """
    def __init__(self, d_model=64, n_layers=6, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


# ============================================================
# 事件编码常量
# ============================================================

EVENT_TYPES = ['focus_gained', 'focus_lost', 'text_insert',
               'text_remove', 'text_paste', 'run', 'submit']
EVENT_TO_IDX = {et: i for i, et in enumerate(EVENT_TYPES)}


def encode_events(student_df, max_events=None):
    """
    Step 1 辅助: 将单个学生的事件序列编码为7维表示

    每个事件编码为:
      - event_type: 事件类型索引 (0-6)
      - time_interval: 距上一事件的时间间隔 (log-normalized)
      - deadline_dist: 距截止时间距离 (归一化)

    Returns:
        dict with tensors
    """
    events = student_df.sort_values('timestamp')

    timestamps = events['timestamp'].values
    event_types_raw = events['eventType'].values

    n_events = len(events)

    if max_events and n_events > max_events:
        timestamps = timestamps[-max_events:]
        event_types_raw = event_types_raw[-max_events:]
        n_events = max_events

    # 事件类型索引
    event_type_idx = np.zeros(n_events, dtype=np.int64)
    for i, et in enumerate(event_types_raw):
        event_type_idx[i] = EVENT_TO_IDX.get(et, 0)

    # 时间间隔 (log-normalized)
    time_intervals = np.zeros(n_events, dtype=np.float32)
    for i in range(1, n_events):
        dt = (timestamps[i] - timestamps[i-1]) / np.timedelta64(1, 's')
        dt = max(dt, 1)
        time_intervals[i] = np.log1p(dt)
    if time_intervals.max() > 0:
        time_intervals = time_intervals / (time_intervals.max() + 1e-8)

    # 截止时间距离
    if 'timeToDeadline' in events.columns:
        deadline_dists = events['timeToDeadline'].values.astype(np.float32) / 3600.0
        if deadline_dists.max() > 0:
            deadline_dists = deadline_dists / (deadline_dists.max() + 1e-8)
    else:
        deadline_dists = np.zeros(n_events, dtype=np.float32)

    # Part IDs
    if 'part' in events.columns:
        part_ids = np.clip(events['part'].values.astype(np.int64), 1, 7)
    else:
        part_ids = np.ones(n_events, dtype=np.int64)

    return {
        'event_types': torch.LongTensor(event_type_idx),
        'time_intervals': torch.FloatTensor(time_intervals),
        'deadline_dists': torch.FloatTensor(deadline_dists),
        'part_ids': torch.LongTensor(part_ids),
        'n_events': n_events,
    }


class SimplifiedMambaStudent(nn.Module):
    """
    简化版 Mamba 模型 (CPU优化) - 完整6步流程

    Step 1: 7维事件编码
    Step 2: Mamba 骨干网络 (预训练 + 微调)
    Step 3: 多尺度特征提取 (简化版: 全局均值 + 最后状态 + 分部均值)
    Step 4: 原型发现 (可学习原型中心)
    Step 5: 风险预测
    Step 6: 可解释性

    配置: d_model=32, n_layers=2, seq_len capped at 2000
    """
    def __init__(self, n_event_types=7, d_model=32, n_layers=2, d_state=8,
                 n_prototypes=4, max_seq_len=2000):
        super().__init__()

        self.max_seq_len = max_seq_len

        # Step 1: 事件编码
        self.event_embed = nn.Embedding(n_event_types, 16)
        self.time_embed = nn.Linear(1, 8)
        self.deadline_embed = nn.Linear(1, 8)

        d_input = 16 + 8 + 8  # 32
        self.input_proj = nn.Linear(d_input, d_model)

        # Step 2: Mamba 骨干
        self.mamba = MambaEncoder(d_model=d_model, n_layers=n_layers, d_state=d_state)
        self.final_norm = RMSNorm(d_model)

        # Step 3: 多尺度 (简化: 全局 + 最后 + 分部)
        self.part_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)

        # Step 4: 原型发现
        self.prototype_centers = nn.Parameter(torch.randn(n_prototypes, d_model) * 0.1)

        # Step 5a: 事件预测头 (预训练用)
        self.event_head = nn.Linear(d_model, n_event_types)

        # Step 5b: 风险预测头 (微调用)
        self.risk_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, 2)
        )

        self.dropout = nn.Dropout(0.2)

    def forward(self, batch, return_repr=False):
        """前向传播"""
        event_types = batch['event_types']
        time_intervals = batch['time_intervals']
        deadline_dists = batch['deadline_dists']
        part_ids = batch.get('part_ids', torch.ones_like(event_types))

        # Step 1: 编码
        event_emb = self.event_embed(event_types)
        time_emb = self.time_embed(time_intervals.unsqueeze(-1))
        dl_emb = self.deadline_embed(deadline_dists.unsqueeze(-1))

        x = torch.cat([event_emb, time_emb, dl_emb], dim=-1)
        x = self.input_proj(x)

        # Step 2: Mamba 编码
        mamba_out = self.mamba(x)
        mamba_out = self.final_norm(mamba_out)

        # Step 3: 多尺度特征 (简化)
        seq_mean = mamba_out.mean(dim=1)
        seq_last = mamba_out[:, -1, :]

        part_means = []
        for p in range(1, 8):
            mask = (part_ids == p)
            if mask.any():
                part_mean = (mamba_out * mask.unsqueeze(-1)).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
            else:
                part_mean = torch.zeros_like(seq_mean)
            part_means.append(part_mean)
        part_repr = torch.stack(part_means, dim=1).mean(dim=1)

        multi_scale_repr = (seq_mean + seq_last + part_repr) / 3

        # Step 4: 原型发现
        dists = torch.cdist(multi_scale_repr, self.prototype_centers)
        proto_weights = torch.softmax(-dists, dim=-1)
        proto_repr = torch.matmul(proto_weights, self.prototype_centers)

        # Step 5: 预测
        combined = torch.cat([multi_scale_repr, proto_repr], dim=-1)
        combined = self.dropout(combined)
        risk_pred = self.risk_head(combined)

        event_pred = self.event_head(mamba_out[:, -1, :])

        if return_repr:
            return {
                'risk': risk_pred,
                'event': event_pred,
                'repr': multi_scale_repr,
                'proto_weights': proto_weights,
                'mamba_out': mamba_out
            }
        return {'risk': risk_pred, 'event': event_pred}

    def get_interpretability(self, batch):
        """Step 6: 可解释性"""
        outputs = self.forward(batch, return_repr=True)

        with torch.no_grad():
            event_importance = self.event_embed.weight.norm(dim=-1)
            event_importance = torch.softmax(event_importance, dim=0)

            mamba_out = outputs['mamba_out']
            seq_len = mamba_out.shape[1]

            if seq_len >= 100:
                last_100 = mamba_out[:, -100:].mean(dim=1)
                first_part = mamba_out[:, :max(1, seq_len - 100)].mean(dim=1)
                temporal_ratio = (last_100.norm(dim=-1) / (first_part.norm(dim=-1) + 1e-8)).unsqueeze(-1)
            else:
                temporal_ratio = torch.ones(mamba_out.shape[0], 1, device=mamba_out.device)

            proto_id = outputs['proto_weights'].argmax(dim=-1)

            return {
                'event_importance': event_importance.cpu().numpy(),
                'event_type_names': EVENT_TYPES,
                'temporal_ratio': temporal_ratio.cpu().numpy(),
                'proto_weights': outputs['proto_weights'].cpu().numpy(),
                'proto_id': proto_id.cpu().numpy(),
                'n_prototypes': 4,
            }


class FullMambaStudent(nn.Module):
    """
    完整版 Mamba 模型 (GPU版本) - 完整6步流程

    配置: d_model=64, n_layers=6, 完整序列

    增强功能:
      - 更大的模型容量 (d_model=64, 6层)
      - 多尺度特征提取器 (细/中/粗三级)
      - 交叉注意力融合
    """
    def __init__(self, n_event_types=7, d_model=64, n_layers=6, d_state=16,
                 n_prototypes=4, max_seq_len=80000):
        super().__init__()

        self.max_seq_len = max_seq_len

        # Step 1: 事件编码
        self.event_embed = nn.Embedding(n_event_types, 16)
        self.time_embed = nn.Linear(1, 8)
        self.deadline_embed = nn.Linear(1, 8)

        d_input = 16 + 8 + 8
        self.input_proj = nn.Linear(d_input, d_model)

        # Step 2: Mamba 骨干 (完整版)
        self.mamba = MambaEncoder(d_model=d_model, n_layers=n_layers, d_state=d_state)
        self.final_norm = RMSNorm(d_model)

        # Step 3: 多尺度特征提取 (完整版)
        self.fine_window = 100
        self.fine_proj = nn.Linear(d_model, d_model)
        self.medium_proj = nn.Linear(d_model, d_model)
        self.coarse_proj = nn.Linear(d_model, d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=4, batch_first=True, dropout=0.1
        )
        self.scale_fusion = nn.Linear(d_model * 3, d_model)

        # Step 4: 原型发现
        self.prototype_centers = nn.Parameter(torch.randn(n_prototypes, d_model) * 0.1)

        # Step 5a: 事件预测头
        self.event_head = nn.Linear(d_model, n_event_types)

        # Step 5b: 风险预测头
        self.risk_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, 2)
        )

        self.dropout = nn.Dropout(0.2)

    def _multi_scale_extract(self, mamba_out, part_ids):
        """Step 3: 多尺度特征提取"""
        batch, seq_len, d_model = mamba_out.shape

        # Fine-grained: 每100事件窗口
        n_windows = max(seq_len // self.fine_window, 1)
        if seq_len >= self.fine_window:
            x_trunc = mamba_out[:, :n_windows * self.fine_window]
            x_windows = x_trunc.view(batch, n_windows, self.fine_window, d_model)
            fine_features = self.fine_proj(x_windows.mean(dim=2))
        else:
            fine_features = self.fine_proj(mamba_out.mean(dim=1, keepdim=True))

        # Medium-grained: 全局均值
        medium_features = self.medium_proj(mamba_out.mean(dim=1, keepdim=True))

        # Coarse-grained: 按part分组
        coarse_means = []
        for p in range(1, 8):
            mask = (part_ids == p)
            if mask.any():
                p_feat = (mamba_out * mask.unsqueeze(-1)).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
            else:
                p_feat = torch.zeros(batch, d_model, device=mamba_out.device)
            coarse_means.append(p_feat)
        coarse_repr = torch.stack(coarse_means, dim=1).mean(dim=1, keepdim=True)
        coarse_features = self.coarse_proj(coarse_repr)

        # 交叉注意力融合
        fine_enhanced, _ = self.cross_attn(fine_features, medium_features, medium_features)

        # 拼接三种尺度
        fused = torch.cat([
            fine_enhanced.mean(dim=1),
            medium_features.squeeze(1),
            coarse_features.squeeze(1)
        ], dim=-1)
        fused = self.scale_fusion(fused)

        return fused

    def forward(self, batch, return_repr=False):
        event_types = batch['event_types']
        time_intervals = batch['time_intervals']
        deadline_dists = batch['deadline_dists']
        part_ids = batch.get('part_ids', torch.ones_like(event_types))

        # Step 1: 编码
        event_emb = self.event_embed(event_types)
        time_emb = self.time_embed(time_intervals.unsqueeze(-1))
        dl_emb = self.deadline_embed(deadline_dists.unsqueeze(-1))

        x = torch.cat([event_emb, time_emb, dl_emb], dim=-1)
        x = self.input_proj(x)

        # Step 2: Mamba 编码
        mamba_out = self.mamba(x)
        mamba_out = self.final_norm(mamba_out)

        # Step 3: 多尺度特征
        multi_scale_repr = self._multi_scale_extract(mamba_out, part_ids)

        # Step 4: 原型发现
        dists = torch.cdist(multi_scale_repr, self.prototype_centers)
        proto_weights = torch.softmax(-dists, dim=-1)
        proto_repr = torch.matmul(proto_weights, self.prototype_centers)

        # Step 5: 预测
        combined = torch.cat([multi_scale_repr, proto_repr], dim=-1)
        combined = self.dropout(combined)
        risk_pred = self.risk_head(combined)

        event_pred = self.event_head(mamba_out[:, -1, :])

        if return_repr:
            return {
                'risk': risk_pred,
                'event': event_pred,
                'repr': multi_scale_repr,
                'proto_weights': proto_weights,
                'mamba_out': mamba_out
            }
        return {'risk': risk_pred, 'event': event_pred}

    def get_interpretability(self, batch):
        """Step 6: 可解释性"""
        outputs = self.forward(batch, return_repr=True)

        with torch.no_grad():
            event_importance = self.event_embed.weight.norm(dim=-1)
            event_importance = torch.softmax(event_importance, dim=0)

            mamba_out = outputs['mamba_out']
            seq_len = mamba_out.shape[1]

            if seq_len >= 100:
                window_size = 100
                n_windows = seq_len // window_size
                mamba_reshaped = mamba_out[:, :n_windows * window_size]
                mamba_reshaped = mamba_reshaped.view(-1, n_windows, window_size, mamba_out.shape[-1])
                window_scores = mamba_reshaped.mean(dim=2).norm(dim=-1)
                temporal_importance = torch.softmax(window_scores, dim=-1)
            else:
                temporal_importance = torch.ones(mamba_out.shape[0], 1, device=mamba_out.device)

            proto_id = outputs['proto_weights'].argmax(dim=-1)

            return {
                'event_importance': event_importance.cpu().numpy(),
                'event_type_names': EVENT_TYPES,
                'temporal_importance': temporal_importance.cpu().numpy(),
                'proto_weights': outputs['proto_weights'].cpu().numpy(),
                'proto_id': proto_id.cpu().numpy(),
                'n_prototypes': 4,
            }


def create_model(device='cpu'):
    """创建模型 (自动选择CPU/GPU版本)"""
    import torch
    if device == 'cpu' or not torch.cuda.is_available():
        model = SimplifiedMambaStudent()
    else:
        model = FullMambaStudent()
    model.to(device)
    return model
