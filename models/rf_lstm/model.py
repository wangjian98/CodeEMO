"""
RF-Enhanced BiLSTM
===================
设计动机：
  * ablation 显示：RF_7dim 单模型 F1=0.888 已经极强；HDM-Net 3-branch 设计在小样本下 PIG
    routing 难以稳定转化为 F1 增益
  * 因此换思路：把 RF 的"已训练好的强归纳偏置"作为辅助信号直接注入 LSTM 的每一步
  * 不再做 per-instance gating，而是单一模型端到端学 "RF 概率 + 46-dim 时序"

架构:
  46-dim 特征 reshape 成 (B, 4, 11) segments
  每个 segment 拼接 2-dim RF OOF probs (frozen feature) -> (B, 4, 13)
  BiLSTM 处理 segment sequence
  mean-pool over segments + 拼接 RF probs -> FC -> sigmoid

相比 HDM-Net 的关键差异：
  * 单一 backbone，没有多分支 per-instance 路由
  * RF 信号进入 LSTM 内部，而不是在最后做 gate
  * 端到端：LSTM 自己学"什么时候信任 RF vs 自己"
"""
import torch
import torch.nn as nn


class RFEnhancedBiLSTM(nn.Module):
    """Single-backbone model: BiLSTM on 46-dim with RF OOF probs as auxiliary input.

    Args:
        seq_dim: total feature dim per student (46)
        rf_dim: RF probability dim (2: [P(passed), P(failed)])
        seg_size: features per segment (default 11)
        n_segments: number of segments (default 4)
        hidden: LSTM hidden dim per direction (default 64)
        layers: number of LSTM layers (default 1)
        dropout: dropout after LSTM (default 0.2)
    """
    def __init__(self, seq_dim=46, rf_dim=2, seg_size=11, n_segments=4,
                  hidden=64, layers=1, dropout=0.2):
        super().__init__()
        assert seg_size * n_segments <= seq_dim, \
            f"seg_size*n_segments ({seg_size*n_segments}) must be <= seq_dim ({seq_dim})"
        self.seg_size = seg_size
        self.n_segments = n_segments
        self.hidden = hidden
        # per-step input = seg_size + rf_dim (RF probs)
        self.input_per_step = seg_size + rf_dim
        # BiLSTM
        self.lstm = nn.LSTM(self.input_per_step, hidden, batch_first=True,
                             bidirectional=True, num_layers=layers,
                             dropout=dropout if layers > 1 else 0)
        # FC head: pool + RF probs
        self.fc = nn.Linear(2 * hidden + rf_dim, 1)
        # init
        for m in [self.fc]:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.05)
                nn.init.zeros_(m.bias)

    def forward(self, x_seq, rf_probs):
        """
        x_seq: (B, seq_dim) or (B, n_segments, seg_size)
        rf_probs: (B, rf_dim)
        returns logits (B,)
        """
        if x_seq.dim() == 2:
            # (B, seq_dim) -> (B, n_segments, seg_size) by slicing the first usable
            B = x_seq.size(0)
            usable = self.n_segments * self.seg_size
            x_seq = x_seq[:, :usable].view(B, self.n_segments, self.seg_size)
        B = x_seq.size(0)
        # broadcast RF probs to every segment
        rf_expanded = rf_probs.unsqueeze(1).expand(-1, self.n_segments, -1)  # (B, n_segments, rf_dim)
        x_combined = torch.cat([x_seq, rf_expanded], dim=-1)  # (B, n_segments, seg_size+rf_dim)
        # BiLSTM
        h, _ = self.lstm(x_combined)  # (B, n_segments, 2*hidden)
        z = h.mean(dim=1)              # (B, 2*hidden)
        # concat RF probs for final FC
        z_final = torch.cat([z, rf_probs], dim=-1)  # (B, 2*hidden + rf_dim)
        return self.fc(z_final).squeeze(-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    m = RFEnhancedBiLSTM()
    print(f"Total params: {count_parameters(m):,}")
    nB = 8
    x_seq = torch.randn(nB, 46)
    rf = torch.softmax(torch.randn(nB, 2), dim=-1)
    out = m(x_seq, rf)
    print(f"Output shape: {out.shape}")
    assert out.shape == (nB,)
    print("[OK] forward pass works")