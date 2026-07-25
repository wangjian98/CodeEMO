"""
RF-LSTM v4: LayerNorm + Residual + Self-Attention stack.

Evolution:
  v1 (RF-LSTM):            ~40K params, F1=0.8766  (RF skip baseline)
  v2 (RF-LSTM-Attn):      222K params, F1=0.7980  (FAILED, over-param)
  v3 (RF-LSTM-Attn simp):  29K params, F1=0.8809  (1 attn layer)
  v4 (this):                Pre-norm + 2x attn + LayerNorm + Residual

v4 design rationale (借鉴 GPT-2 / LLaMA):
  * Pre-norm (LayerNorm before attention) is more stable than post-norm
    on small data -- prevents the "logit explosion" we saw in v2.
  * Stacked self-attention (2 layers) gives the model richer
    "what-segment-relates-to-what" reasoning on top of LSTM.
  * LayerNorm after every sub-layer (input -> LSTM -> attn -> out)
    normalizes activations and stabilizes training.
  * Residual connections: every sub-layer has identity skip,
    so gradient flows freely; allows deeper architecture without
    divergence.
  * Head: pre-norm (LayerNorm) before FC, with raw 7d + RF probs
    bypass to keep RF signal as residual anchor.
"""
import torch
import torch.nn as nn


class RFLSTMv4(nn.Module):
    """RF-LSTM v4: Pre-norm + stacked self-attention + LayerNorm + Residual."""
    def __init__(self, hidden=32, rf_dim=2, seg_size=11, n_segments=4,
                  dropout=0.2, n_attn_layers=2):
        super().__init__()
        self.n_segments = n_segments
        self.seg_size = seg_size
        self.hidden = hidden
        d = 2 * hidden  # BiLSTM output dim

        # ---- Pre-norm BiLSTM ----
        self.input_norm = nn.LayerNorm(seg_size + rf_dim)
        self.lstm = nn.LSTM(seg_size + rf_dim, hidden, batch_first=True, bidirectional=True)
        # post-LSTM LayerNorm for residual
        self.lstm_out_norm = nn.LayerNorm(d)

        # ---- Stacked Pre-norm Self-Attention ----
        self.attn_layers = nn.ModuleList()
        for _ in range(n_attn_layers):
            self.attn_layers.append(nn.ModuleDict({
                'norm': nn.LayerNorm(d),
                'attn': nn.MultiheadAttention(d, num_heads=4, dropout=dropout, batch_first=True),
            }))
        self.final_norm = nn.LayerNorm(d)

        # ---- Head with residual RF skip ----
        # Concat: pooled attention output (d) + raw 7-dim (7) + RF probs (2)
        head_in = d + 7 + rf_dim
        self.head_norm = nn.LayerNorm(head_in)
        self.head = nn.Sequential(
            nn.Linear(head_in, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        # init
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.05)
                nn.init.zeros_(m.bias)

    def forward(self, x_7d, x_46d, rf_probs):
        B = x_46d.size(0)
        # Reshape 46-dim to segments, append RF probs to each step
        usable = self.n_segments * self.seg_size
        x_seg = x_46d[:, :usable].view(B, self.n_segments, self.seg_size)
        rf_exp = rf_probs.unsqueeze(1).expand(-1, self.n_segments, -1)
        x_combined = torch.cat([x_seg, rf_exp], dim=-1)  # (B, 4, 13)

        # Pre-norm + BiLSTM
        x_norm = self.input_norm(x_combined)
        h, _ = self.lstm(x_norm)  # (B, 4, 2*hidden=64)
        h = self.lstm_out_norm(h)  # post-LSTM LN

        # Stacked pre-norm self-attention with residual
        for layer in self.attn_layers:
            x_norm = layer['norm'](h)
            attn_out, _ = layer['attn'](x_norm, x_norm, x_norm)
            h = h + attn_out  # residual
        h = self.final_norm(h)  # final LN

        # Mean pool over 4 segments
        pooled = h.mean(dim=1)  # (B, 2*hidden)

        # Concat with raw 7-dim events + RF probs (residual bypass)
        feat = torch.cat([pooled, x_7d, rf_probs], dim=-1)
        feat = self.head_norm(feat)  # pre-head LN
        return self.head(feat).squeeze(-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    m = RFLSTMv4()
    print(f"Total params: {count_parameters(m):,}")
    nB = 8
    x_7d = torch.randn(nB, 7)
    x_46d = torch.randn(nB, 46)
    rf = torch.softmax(torch.randn(nB, 2), dim=-1)
    out = m(x_7d, x_46d, rf)
    print(f"Output: {out.shape}")
    print("[OK] RF-LSTM v4 forward pass works")