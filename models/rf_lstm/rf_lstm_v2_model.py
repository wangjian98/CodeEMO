"""
RF-LSTM-Attn v2: Deep RF-LSTM Fusion with Cross-View Attention

Architecture motivation:
  - RF_7dim: high Recall (0.869), good Precision (0.907), simple 7-dim input
  - LSTM_46d: comparable Precision (0.900), low Recall (0.828), temporal modeling
  - Pairwise corr = 0.844 (genuinely diverse)
  - Goal: deep architecture-level fusion, not just post-hoc averaging

Architecture:
  Step 1: Concatenate raw features [7-dim events + 2-dim RF probs] -> 9-dim "RF-augmented event"
  Step 2: BiLSTM over 4 segments of [11-dim features + RF probs] = (B, 4, 13)
  Step 3: Cross-view attention: query from RF-augmented event (B, 1, 9),
          key/value from LSTM segments (B, 4, 2*hidden)
  Step 4: Concat attended feature + raw RF + raw 7-dim + LSTM last-step hidden
  Step 5: 2-layer MLP head

Compared to v1 (RF-LSTM F1=0.8766):
  - Deeper BiLSTM (2 layers vs 1)
  - Cross-view attention: RF event attends to LSTM hidden states
  - Direct residual: raw 7-dim + RF probs bypass through to head
"""
import torch
import torch.nn as nn


class RFEventEncoder(nn.Module):
    """Encode [7-dim events + 2-dim RF probs] into a hidden vector."""
    def __init__(self, in_dim=9, hidden=32, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim), nn.ReLU(),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.05)
                nn.init.zeros_(m.bias)

    def forward(self, x_7d, rf_probs):
        cat = torch.cat([x_7d, rf_probs], dim=-1)  # (B, 9)
        return self.net(cat)  # (B, out_dim)


class LSTMStream(nn.Module):
    """BiLSTM over 46-dim reshaped to (4, 13) segments, each with RF probs appended."""
    def __init__(self, hidden=64, layers=2, dropout=0.2, rf_dim=2, seg_size=11, n_segments=4):
        super().__init__()
        self.n_segments = n_segments
        self.seg_size = seg_size
        self.input_per_step = seg_size + rf_dim
        self.lstm = nn.LSTM(self.input_per_step, hidden, batch_first=True,
                             bidirectional=True, num_layers=layers,
                             dropout=dropout if layers > 1 else 0)
        self.hidden = hidden

    def forward(self, x_46d, rf_probs):
        B = x_46d.size(0)
        usable = self.n_segments * self.seg_size
        x_seg = x_46d[:, :usable].view(B, self.n_segments, self.seg_size)
        rf_exp = rf_probs.unsqueeze(1).expand(-1, self.n_segments, -1)
        x_combined = torch.cat([x_seg, rf_exp], dim=-1)  # (B, 4, 13)
        h, _ = self.lstm(x_combined)  # (B, 4, 2*hidden)
        return h  # keep sequence form for attention


class CrossViewAttention(nn.Module):
    """RF-augmented event (query) attends to LSTM hidden states (key/value).

    This lets the model learn "for this student's event pattern, which
    LSTM segment is most informative" — a learned per-sample routing
    between the RF branch and the temporal branch.
    """
    def __init__(self, query_dim=32, kv_dim=128, nhead=4, dropout=0.1):
        super().__init__()
        # Project query to match kv dim
        self.q_proj = nn.Linear(query_dim, kv_dim)
        self.attn = nn.MultiheadAttention(kv_dim, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(kv_dim)
        # Project back to query dim for residual
        self.out_proj = nn.Linear(kv_dim, query_dim)
        self.out_norm = nn.LayerNorm(query_dim)

    def forward(self, query, kv):
        # query: (B, d_q) -> (B, 1, d_q)
        # kv: (B, T, d_kv)
        q = self.q_proj(query).unsqueeze(1)  # (B, 1, d_kv)
        attn_out, _ = self.attn(q, kv, kv)    # (B, 1, d_kv)
        out = self.norm(attn_out + q)         # residual + norm
        # Project back to query dim for residual with original query
        out = self.out_proj(out.squeeze(1))    # (B, d_q)
        return self.out_norm(out + query)      # residual + norm


class RFLSTMAttnV2(nn.Module):
    """RF-LSTM-Attn v2: deep architecture-level RF + LSTM fusion."""
    def __init__(self, hidden=64, lstm_layers=2, dropout=0.2, rf_dim=2):
        super().__init__()
        self.rf_event = RFEventEncoder(in_dim=7+rf_dim, hidden=32, out_dim=32)
        self.lstm = LSTMStream(hidden=hidden, layers=lstm_layers,
                                dropout=dropout, rf_dim=rf_dim)
        # Cross-view attention: query from RF event (32-d) -> kv from LSTM (2*hidden=128-d)
        self.cva = CrossViewAttention(query_dim=32, kv_dim=2*hidden, nhead=4, dropout=0.1)
        # Head input dims:
        #   - cross-attended RF event (32)
        #   - LSTM last-step hidden (2*hidden = 128)
        #   - raw 7-dim (7)
        #   - raw RF probs (2)
        head_in = 32 + 2*hidden + 7 + rf_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.05)
                nn.init.zeros_(m.bias)

    def forward(self, x_7d, x_46d, rf_probs):
        # Step 1: encode 7-dim + RF probs as "RF event"
        rf_event = self.rf_event(x_7d, rf_probs)         # (B, 32)
        # Step 2: BiLSTM over 46-dim segments + RF per-step
        lstm_seq = self.lstm(x_46d, rf_probs)            # (B, 4, 128)
        lstm_last = lstm_seq[:, -1, :]                   # (B, 128) last-step hidden
        # Step 3: RF event attends to LSTM segments
        attended = self.cva(rf_event, lstm_seq)          # (B, 32)
        # Step 4: concat all features + raw signals
        feat = torch.cat([attended, lstm_last, x_7d, rf_probs], dim=-1)  # (B, 32+128+7+2=169)
        return self.head(feat).squeeze(-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    m = RFLSTMAttnV2()
    print(f"Total params: {count_parameters(m):,}")
    import torch
    nB = 8
    x_7d = torch.randn(nB, 7)
    x_46d = torch.randn(nB, 46)
    rf = torch.softmax(torch.randn(nB, 2), dim=-1)
    out = m(x_7d, x_46d, rf)
    print(f"Output shape: {out.shape}")
    print("[OK] RF-LSTM-Attn v2 forward pass works")