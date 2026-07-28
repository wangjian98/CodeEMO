"""
Multi-Route Expert (MRE) 模型定义

设计动机:
  - CS1 数据集 (n=473) 上 RF_7dim 在 Accuracy / Precision 上最强
    (Acc=0.8541, Precision=0.9082, F1=0.8876)
  - LSTM_46d 在 Recall 上有互补价值 (Recall=0.828 vs RF=0.869)
  - 多路由专家:用一个小门控网络决定每个样本该走哪条路由
    * Route A (RF Expert): 高准确率/精度
    * Route B (LSTM Expert): 高召回率
    * Gate: 决定每个样本该信任哪个 expert 或如何融合

三种融合策略:
  1. Soft MoE: P_final = alpha * P_rf + (1-alpha) * P_lstm
  2. Confidence Routing: 高置信样本走单一 expert，否则加权
  3. Hard Routing: gate 离散选择唯一 expert
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingMLP(nn.Module):
    """门控网络: 输入 (rf_prob, lstm_prob, |rf_prob-lstm_prob|, rf*lstm, max, min, raw_features)
       输出 (alpha_rf, alpha_lstm) softmax 权重
    """
    def __init__(self, in_dim, hidden=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x):
        logits = self.net(x)
        return F.softmax(logits, dim=-1)  # (B, 2)


class MREFusion(nn.Module):
    """Multi-Route Expert Fusion 模型

    Args:
        rf_prob: (B,) RF OOF prob for failed=1
        lstm_prob: (B,) LSTM OOF prob for failed=1
        raw_7d: (B, 7) 7-dim event count features (for gate context)
    Returns:
        fused_prob: (B,) P(failed)
        gate_weights: (B, 2) (alpha_rf, alpha_lstm)
    """
    def __init__(self, raw_dim=7, hidden=32, dropout=0.2, fusion_mode='soft'):
        super().__init__()
        # Gate input: (rf_prob, lstm_prob, |rf-lstm|, rf*lstm, max, min, raw_7d)
        gate_in = 6 + raw_dim
        self.gate = GatingMLP(gate_in, hidden=hidden, dropout=dropout)
        self.fusion_mode = fusion_mode  # 'soft' | 'confidence' | 'hard'

    def _gate_input(self, rf_p, lstm_p, raw):
        diff = (rf_p - lstm_p).abs()
        prod = rf_p * lstm_p
        mx = torch.stack([rf_p, lstm_p], dim=-1).max(dim=-1).values
        mn = torch.stack([rf_p, lstm_p], dim=-1).min(dim=-1).values
        return torch.cat([rf_p.unsqueeze(-1), lstm_p.unsqueeze(-1),
                          diff.unsqueeze(-1), prod.unsqueeze(-1),
                          mx.unsqueeze(-1), mn.unsqueeze(-1), raw], dim=-1)

    def forward(self, rf_p, lstm_p, raw):
        g_in = self._gate_input(rf_p, lstm_p, raw)
        w = self.gate(g_in)  # (B, 2) softmax weights

        if self.fusion_mode == 'soft':
            fused = w[:, 0] * rf_p + w[:, 1] * lstm_p
        elif self.fusion_mode == 'hard':
            # Straight-Through Estimator: forward 离散, backward 用 soft
            hard_mask = (w[:, 0] > w[:, 1]).float()  # (B,)
            soft_fused = w[:, 0] * rf_p + w[:, 1] * lstm_p
            hard_fused = hard_mask * rf_p + (1 - hard_mask) * lstm_p
            fused = hard_fused + soft_fused - soft_fused.detach()
        elif self.fusion_mode == 'confidence':
            rf_conf = (rf_p - 0.5).abs()
            lstm_conf = (lstm_p - 0.5).abs()
            rf_trust = (rf_conf > 0.30).float()
            lstm_trust = (lstm_conf > 0.30).float()
            both_trust = ((rf_trust + lstm_trust) > 1.5).float()
            one_trust = (((rf_trust + lstm_trust) > 0.5).float() - both_trust)
            neither = 1.0 - both_trust - one_trust

            fused = (both_trust * 0.5 * (rf_p + lstm_p)
                     + one_trust * (rf_trust * rf_p + lstm_trust * lstm_p)
                     + neither * (w[:, 0] * rf_p + w[:, 1] * lstm_p))
        else:
            raise ValueError(f'Unknown fusion_mode: {self.fusion_mode}')

        return fused, w


if __name__ == '__main__':
    nB = 8
    rf = torch.sigmoid(torch.randn(nB))
    lstm = torch.sigmoid(torch.randn(nB))
    raw = torch.randn(nB, 7)
    for mode in ['soft', 'confidence', 'hard']:
        m = MREFusion(fusion_mode=mode)
        out, w = m(rf, lstm, raw)
        print(f'{mode}: out={out.shape} {out.tolist()[:3]}..., w={w.shape} {w.tolist()[:2]}')