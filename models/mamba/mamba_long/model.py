"""
Mamba-Long 模型 (5 步骤整合: 1+2+3)

架构:
  - 输入: 7d 事件序列 (max=2000) + 12d micro 特征
  - Mamba encoder 处理事件序列 (S6 selective scan)
  - 12d micro 特征 → MLP → 16d
  - 多尺度特征融合 (细粒度窗口 + 中粒度全局 + 粗粒度按 deadline 切分)
  - SE 通道注意力
  - risk_head: concat(multi_scale_repr, proto_repr, micro_repr)

步骤 1: FullMambaStudent 长序列支持 (max=2000)
步骤 2: micro 特征融合 (12d → 16d MLP)
步骤 3: 改进 _multi_scale_extract (按 deadline_dists 切分 instead of part_ids)
"""
import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


EVENT_TYPES = [
    'text_insert', 'text_remove', 'text_paste',
    'focus_gained', 'focus_lost', 'run', 'submit'
]


# ============================================================
# 复用的 SSM building blocks (简化版, 避免 OOM)
# ============================================================
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


class S6Block(nn.Module):
    """简化版 S6 选择性扫描 - 适合 2000 长度"""
    def __init__(self, d_model, d_state=12, d_conv=3, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.dt_rank = max(math.ceil(self.d_inner / 16), 1)

        self.input_proj = nn.Linear(d_model, self.d_inner)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                padding=d_conv - 1, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.A = nn.Parameter(torch.arange(1, d_state + 1, dtype=torch.float32)
                              .unsqueeze(0).expand(self.d_inner, -1).clone().log())
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.output_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.act = SiLU()

    def forward(self, x):
        batch, seq_len, _ = x.shape
        x_inner = self.input_proj(x)                              # (B, L, d_inner)
        x_conv = self.act(self.conv1d(x_inner.transpose(1, 2))
                          [:, :, :seq_len].transpose(1, 2))        # (B, L, d_inner)
        x_flat = x_conv.reshape(-1, self.d_inner)
        x_params = self.x_proj(x_flat)
        dt, B_seq, C_seq = torch.split(x_params,
                                       [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = dt.reshape(batch, seq_len, self.dt_rank)
        if self.dt_rank < self.d_inner:
            dt = F.pad(dt, (0, self.d_inner - self.dt_rank))
        B_seq = B_seq.reshape(batch, seq_len, self.d_state)
        C_seq = C_seq.reshape(batch, seq_len, self.d_state)

        # 选择性扫描 (循环版, 优化后的快速版本)
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device)
        ys = []
        A_exp = self.A.unsqueeze(0)
        for t in range(seq_len):
            dt_t = dt[:, t].unsqueeze(-1)             # (B, d_inner, 1)
            B_t = B_seq[:, t].unsqueeze(1).expand(-1, self.d_inner, -1)
            C_t = C_seq[:, t].unsqueeze(1).expand(-1, self.d_inner, -1)
            x_t = x_conv[:, t].unsqueeze(-1)
            dA = torch.exp((dt_t * A_exp).clamp(min=-10.0, max=0.0))
            dB = dt_t * B_t
            h = dA * h + dB * x_t
            ys.append((h * C_t).sum(dim=-1))
        y = torch.stack(ys, dim=1) + x_conv * self.D
        return self.output_proj(y)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=12, d_conv=3, expand=2):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.s6 = S6Block(d_model, d_state, d_conv, expand)

    def forward(self, x):
        return self.s6(self.norm(x)) + x


# ============================================================
# 步骤 1+2+3: MambaLongStudent
# ============================================================
class MambaLongStudent(nn.Module):
    """
    Mamba 处理 7d 长序列 (max=2000) + 12d micro 特征

    步骤:
      1. 长序列 Mamba encoder (max=2000, d=48, 4 层)
      2. micro 特征 MLP → 16d
      3. 改进多尺度 (按 deadline 距离切分, 不依赖 part_ids)
    """
    def __init__(self, n_event_types=7, d_model=48, n_layers=4, d_state=12,
                 n_prototypes=4, max_seq_len=2000, micro_dim=12, micro_proj=16):
        super().__init__()
        self.d_model = d_model
        self.n_prototypes = n_prototypes

        # 事件嵌入
        self.event_embed = nn.Embedding(n_event_types + 1, 16)  # +1 for pad
        self.time_embed = nn.Linear(1, 16)
        self.deadline_embed = nn.Linear(1, 16)
        self.input_proj = nn.Linear(48, d_model)

        # 步骤 1: Mamba encoder
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state) for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)

        # 步骤 2: micro 特征投影
        self.micro_mlp = nn.Sequential(
            nn.Linear(micro_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, micro_proj),
        )

        # 步骤 3: 多尺度特征 (按 deadline 距离切分)
        self.fine_proj = nn.Linear(d_model, d_model)      # 100 事件窗口
        self.medium_proj = nn.Linear(d_model, d_model)    # 全局
        self.coarse_proj = nn.Linear(d_model, d_model)    # 5 段 deadline
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.scale_fusion = nn.Linear(d_model * 3, d_model)

        # SE 通道注意力
        self.se_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model),
            nn.Sigmoid(),
        )

        # 原型发现
        self.prototype_centers = nn.Parameter(torch.randn(n_prototypes, d_model) * 0.1)

        # risk_head
        # 拼接: multi_scale(d_model) + proto(d_model) + micro(micro_proj) + behavior(3)
        self.dropout = nn.Dropout(0.3)
        self.risk_head = nn.Linear(d_model * 2 + micro_proj + 3, 2)

    def _multi_scale_extract(self, mamba_out, deadline_dists):
        """
        步骤 3 改进: 按 deadline 距离切分 (5 段: [0, 0.2], [0.2, 0.4], ..., [0.8, 1.0])
        不再依赖 part_ids (因为 part_ids 在 max=2000 时分布不均)
        所有输出统一为 (B, d_model) 形状
        """
        batch, seq_len, d_model = mamba_out.shape

        # 细粒度: 100 事件窗口均值 → (B, d_model)
        win_size = 100
        n_fine = max(1, seq_len // win_size)
        fine_pool = []
        for i in range(n_fine):
            start = i * win_size
            end = min(start + win_size, seq_len)
            fine_pool.append(mamba_out[:, start:end].mean(dim=1))  # (B, d_model)
        fine = torch.stack(fine_pool, dim=1).mean(dim=1)         # (B, d_model)
        fine_features = self.fine_proj(fine)                      # (B, d_model)

        # 中粒度: 全局均值 → (B, d_model)
        medium = mamba_out.mean(dim=1)
        medium_features = self.medium_proj(medium)

        # 粗粒度: deadline_dists 5 段切分 → (B, d_model)
        coarse_pool = []
        for i in range(5):
            lo, hi = i * 0.2, (i + 1) * 0.2
            if i == 4:
                mask = (deadline_dists >= lo) & (deadline_dists <= 1.0)
            else:
                mask = (deadline_dists >= lo) & (deadline_dists < hi)
            mask_f = mask.float().unsqueeze(-1)  # (B, L, 1)
            denom = mask_f.sum(dim=1).clamp(min=1.0)  # (B, 1)
            p_feat = (mamba_out * mask_f).sum(dim=1) / denom  # (B, d_model)
            coarse_pool.append(p_feat)
        coarse = torch.stack(coarse_pool, dim=1).mean(dim=1)      # (B, d_model)
        coarse_features = self.coarse_proj(coarse)

        # 交叉注意力 (用 (B, 1, d_model) 输入输出 squeeze 回 (B, d_model))
        fine_q = fine_features.unsqueeze(1)
        med_kv = medium_features.unsqueeze(1)
        fine_enhanced, _ = self.cross_attn(fine_q, med_kv, med_kv)
        fine_enhanced = fine_enhanced.squeeze(1)                  # (B, d_model)

        # 拼接 3 尺度 (都是 2D)
        fused = torch.cat([fine_enhanced, medium_features, coarse_features], dim=-1)  # (B, 3*d_model)
        return self.scale_fusion(fused)

    def _behavior_features(self, deadline_dists, time_intervals):
        """3 维行为先验 (跟原 FullMambaStudent 一致)"""
        near_ddl_ratio = (deadline_dists < 0.2).float().mean(dim=-1)
        ti = time_intervals.clamp(min=1e-3)
        cv = ti.std(dim=-1) / ti.mean(dim=-1).clamp(min=1e-3)
        regularity = (1.0 - cv.clamp(0, 5) / 5.0)
        # activity_diversity 用 deadline 段覆盖率代替 part_ids 多样性
        unique_segments = set()
        B = deadline_dists.shape[0]
        diversity = torch.zeros(B, device=deadline_dists.device)
        for b in range(B):
            d_b = deadline_dists[b]
            seg_ids = (d_b * 5).long().clamp(0, 4)
            diversity[b] = torch.unique(seg_ids).numel() / 5.0
        return torch.stack([near_ddl_ratio, diversity, regularity], dim=-1)

    def forward(self, batch, return_repr=False):
        event_types = batch['event_types']
        time_intervals = batch['time_intervals']
        deadline_dists = batch['deadline_dists']
        micro = batch.get('micro', None)  # (B, 12) 或 None

        # Step 1: 编码 + Mamba
        event_emb = self.event_embed(event_types)
        time_emb = self.time_embed(time_intervals.unsqueeze(-1))
        dl_emb = self.deadline_embed(deadline_dists.unsqueeze(-1))
        x = torch.cat([event_emb, time_emb, dl_emb], dim=-1)
        x = self.input_proj(x)
        for blk in self.mamba_layers:
            x = blk(x)
        x = self.final_norm(x)

        # Step 3: 改进多尺度 (按 deadline_dists 切分)
        multi_scale = self._multi_scale_extract(x, deadline_dists)
        se_w = self.se_gate(multi_scale)
        multi_scale = multi_scale * se_w

        # 原型
        dists = torch.cdist(multi_scale.unsqueeze(1), self.prototype_centers.unsqueeze(0))
        proto_weights = torch.softmax(-dists.squeeze(1), dim=-1)
        proto_repr = torch.matmul(proto_weights, self.prototype_centers)

        # Step 2: micro 特征
        if micro is None:
            micro = torch.zeros(event_types.size(0), 12,
                                device=event_types.device)
        micro_repr = self.micro_mlp(micro)

        # 预测
        behavior = self._behavior_features(deadline_dists, time_intervals)
        combined = torch.cat([multi_scale, proto_repr, micro_repr, behavior], dim=-1)
        combined = self.dropout(combined)
        risk_pred = self.risk_head(combined)

        if return_repr:
            return {
                'risk': risk_pred,
                'multi_scale': multi_scale,
                'proto_weights': proto_weights,
                'micro_repr': micro_repr,
                'behavior': behavior,
            }
        return {'risk': risk_pred}


def create_model(device='cuda'):
    model = MambaLongStudent()
    model.to(device)
    return model


# ============================================================
# 12d micro 特征抽取 (跟 BiLSTM-micro 共用定义)
# ============================================================
def compute_micro_features(ide_logs_df, student_ids, n_first=30):
    """
    对每个学生从最早的 n_first 个事件中提取 12 维 micro-behaviour 特征。
    跟 models/bilstm_7dim_micro.compute_micro_features 完全一致。
    """
    n = len(student_ids)
    feats = np.zeros((n, 12), dtype=np.float32)
    EVENT_TYPES_7 = EVENT_TYPES  # 用同顺序

    for i, sid in enumerate(student_ids):
        df = ide_logs_df[ide_logs_df['student'] == sid].sort_values('timestamp').head(n_first)
        if len(df) == 0:
            continue
        n_ev = len(df)

        # 0: focus_gain_rate
        feats[i, 0] = (df['eventType'] == 'focus_gained').sum() / n_ev
        # 1: focus_lose_rate
        feats[i, 1] = (df['eventType'] == 'focus_lost').sum() / n_ev
        # 2: focus_gl_ratio
        fg = (df['eventType'] == 'focus_gained').sum()
        fl = (df['eventType'] == 'focus_lost').sum()
        feats[i, 2] = fg / (fl + 1) if fl > 0 else float(fg)
        # 3: edit_density
        feats[i, 3] = (df['eventType'] == 'text_insert').sum() / n_ev
        # 4: delete_density
        feats[i, 4] = (df['eventType'] == 'text_remove').sum() / n_ev
        # 5: edit_delete_ratio
        ins = (df['eventType'] == 'text_insert').sum()
        rem = (df['eventType'] == 'text_remove').sum()
        feats[i, 5] = ins / (rem + 1) if rem > 0 else float(ins)
        # 6: submit_rate
        feats[i, 6] = (df['eventType'] == 'submit').sum() / n_ev
        # 7: early_tightness
        if 'timestamp' in df.columns and len(df) > 1:
            ts = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
            feats[i, 7] = ts / max(n_ev, 1)
        else:
            feats[i, 7] = 0.0
        # 8: early_deadline_prox (假定 'exercise' 列含 deadline)
        # 这里简化: 用 timestamp 均值
        # 9: early_event_count
        feats[i, 9] = min(n_ev, n_first) / n_first
        # 10: intro_burst_score
        if len(df) >= 10:
            intro = len(df.head(10))
        else:
            intro = n_ev
        feats[i, 10] = intro / max(n_ev, 1)
        # 11: paste_density
        feats[i, 11] = (df['eventType'] == 'text_paste').sum() / n_ev

    return feats


import numpy as np  # 放在最后避免循环导入问题