"""
RF-LSTM v3: SIMPLIFIED design after v2 over-parameterization failure.

Lessons:
  - v1 (RF-LSTM, hidden=32, 1-layer, ~40K params): F1=0.8766 ✓
  - v2 (hidden=64, 2-layer + cross-view attention, ~222K params): FAILED (degenerate)

v3: keep v1's capacity (~40K params) but add lightweight self-attention
    on LSTM output + residual RF skip to head.
"""
import torch
import torch.nn as nn


class RFLSTMv3(nn.Module):
    """RF-LSTM v3: simplified + self-attention + RF skip."""
    def __init__(self, hidden=32, rf_dim=2, seg_size=11, n_segments=4,
                  dropout=0.2):
        super().__init__()
        self.n_segments = n_segments
        self.seg_size = seg_size
        self.hidden = hidden
        # BiLSTM: 1-layer (vs v2's 2-layer) to keep params small
        self.lstm = nn.LSTM(seg_size + rf_dim, hidden, batch_first=True, bidirectional=True)
        # Lightweight self-attention on LSTM output (4 segments x 2*hidden)
        self.attn = nn.MultiheadAttention(2 * hidden, num_heads=4, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(2 * hidden)
        # Mean-pool + RF skip to FC head
        # Head input: attn-pooled (2*hidden) + RF probs (rf_dim)
        self.head = nn.Linear(2 * hidden + rf_dim, 1)
        for m in [self.head]:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.05)
                nn.init.zeros_(m.bias)

    def forward(self, x_7d, x_46d, rf_probs):
        # Reshape 46-dim to (B, 4, 11) segments, append RF to each
        B = x_46d.size(0)
        usable = self.n_segments * self.seg_size
        x_seg = x_46d[:, :usable].view(B, self.n_segments, self.seg_size)
        rf_exp = rf_probs.unsqueeze(1).expand(-1, self.n_segments, -1)
        x_combined = torch.cat([x_seg, rf_exp], dim=-1)  # (B, 4, 13)

        h, _ = self.lstm(x_combined)  # (B, 4, 2*hidden)

        # Self-attention on LSTM output (residual)
        h_attn, _ = self.attn(h, h, h)
        h = self.attn_norm(h + h_attn)  # residual + norm

        # Mean pool
        pooled = h.mean(dim=1)  # (B, 2*hidden)

        # Concat with RF probs (direct residual)
        feat = torch.cat([pooled, rf_probs], dim=-1)
        return self.head(feat).squeeze(-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    m = RFLSTMv3()
    print(f"Total params: {count_parameters(m):,}")
    nB = 8
    x_7d = torch.randn(nB, 7)
    x_46d = torch.randn(nB, 46)
    rf = torch.softmax(torch.randn(nB, 2), dim=-1)
    out = m(x_7d, x_46d, rf)
    print(f"Output: {out.shape}")