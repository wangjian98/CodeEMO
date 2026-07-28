"""
PR-DE-Net: Precision-Recall Gated Dual-Encoder Network
========================================================

核心思想:
  RNN 类(LSTM/BiLSTM-46d) 在 Recall 上有优势 (0.80-0.83) → 善于"捕捉 failed"
  Transformer-7d 在 Precision 上有优势 (0.918)        → 善于"识别 passed"

  设计双分支端到端架构:
    Branch A (PR-RNN):    BiLSTM on 46d → p_A    (高 Recall)
    Branch B (PR-Trans):  Transformer on 7d → p_B (高 Precision)
    Gate: σ(MLP([F46, h_A, h_B]))                  (样本自适应路由)

  Loss 三段式:
    L = α·BCE(p_A, y) + β·BCE(1-p_B, y) + γ·BCE(p_final, y)
    让两分支独立优化各自的 P/R 目标，避免梯度对齐导致互补性丧失。

参数量: ~35K
"""
import torch
import torch.nn as nn


# ============================================================
# Branch A: PR-RNN (Failed-Aware, Recall-Oriented)
# ============================================================
class PRRNNBranch(nn.Module):
    """BiLSTM on 46-dim sequence (46-step univariate).

    目的: 学习"什么样的事件统计模式 → failed"，最大化 Recall。
    """
    def __init__(self, d=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=1,
            hidden_size=d,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        # bi-lstm output dim = 2*d
        self.proj = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # output: probability of failed
        self.head = nn.Linear(d, 1)

    def forward(self, x_seq):
        """
        Args:
            x_seq: (B, 46, 1) — 46-dim features reshaped as a 46-step sequence
        Returns:
            p_A: (B, 1) — probability of failed
            h_A: (B, d) — hidden representation for gate
        """
        # BiLSTM
        h, _ = self.bilstm(x_seq)        # (B, 46, 2*d)
        # mean pool over sequence
        h_pool = h.mean(dim=1)           # (B, 2*d)
        h_A = self.proj(h_pool)          # (B, d)
        p_A = torch.sigmoid(self.head(h_A))   # (B, 1)
        return p_A, h_A


# ============================================================
# Branch B: PR-Trans (Passed-Aware, Precision-Oriented)
# ============================================================
class PRTransformerBranch(nn.Module):
    """Transformer on 7-dim reshape (B, 7, 1) with [CLS] token.

    目的: 学习"passed 学生的全局平稳模式"，最大化 Precision。
    """
    def __init__(self, n_segments=7, d=32, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.n_segments = n_segments
        self.d = d
        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.cls_token, std=0.02)
        # projection: 1 -> d
        self.proj = nn.Linear(1, d)
        # positional embedding
        self.pos = nn.Parameter(torch.zeros(1, n_segments + 1, d))
        nn.init.normal_(self.pos, std=0.02)
        # Transformer encoder (Pre-LN, more stable)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=nhead,
            dim_feedforward=d * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm_out = nn.LayerNorm(d)
        # output head
        self.head = nn.Linear(d, 1)

    def forward(self, x_att):
        """
        Args:
            x_att: (B, 7, 1) — 7 event types as a 7-step sequence
        Returns:
            p_B: (B, 1) — probability of failed
            h_B: (B, d) — hidden representation (CLS) for gate
        """
        B = x_att.shape[0]
        # project
        h = self.proj(x_att)             # (B, 7, d)
        # prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, d)
        h = torch.cat([cls, h], dim=1)           # (B, 8, d)
        # add positional embedding
        h = h + self.pos
        # transformer
        h = self.encoder(h)                       # (B, 8, d)
        h = self.norm_out(h)
        # take CLS
        h_B = h[:, 0, :]                          # (B, d)
        p_B = torch.sigmoid(self.head(h_B))       # (B, 1)
        return p_B, h_B


# ============================================================
# Gate Module (Sample-Adaptive Routing)
# ============================================================
class PRGate(nn.Module):
    """Sample-adaptive gate: decides how much to trust A vs B per sample.

    g = σ(MLP([F46, h_A, h_B]))
    p_final = g * p_A + (1-g) * p_B
    """
    def __init__(self, feat_dim=46, h_A_dim=64, h_B_dim=32, hidden=32, dropout=0.2):
        super().__init__()
        in_dim = feat_dim + h_A_dim + h_B_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, F46, h_A, h_B, p_A, p_B):
        """
        Returns:
            p_final: (B, 1)
            gate_value: (B, 1) — g value, for interpretability
        """
        gate_in = torch.cat([F46, h_A, h_B], dim=-1)
        g = torch.sigmoid(self.net(gate_in))      # (B, 1)
        p_final = g * p_A + (1 - g) * p_B
        return p_final, g


# ============================================================
# Full Model
# ============================================================
class PRDENet(nn.Module):
    """PR-Gated Dual-Encoder Network."""

    def __init__(self,
                 rnn_hidden=64, rnn_layers=2,
                 trans_d=32, trans_heads=4, trans_layers=2,
                 gate_hidden=32,
                 dropout=0.3):
        super().__init__()
        self.branch_A = PRRNNBranch(d=rnn_hidden, num_layers=rnn_layers, dropout=dropout)
        self.branch_B = PRTransformerBranch(
            n_segments=7, d=trans_d, nhead=trans_heads,
            num_layers=trans_layers, dropout=dropout * 0.7,
        )
        self.gate = PRGate(
            feat_dim=46,
            h_A_dim=rnn_hidden,
            h_B_dim=trans_d,
            hidden=gate_hidden,
            dropout=dropout * 0.7,
        )

    def forward(self, x):
        """
        Args:
            x: (B, 46) — 46-dim feature vector
        Returns:
            dict with p_A, p_B, p_final, gate
        """
        # reshape for branches
        x_seq = x.unsqueeze(-1)                  # (B, 46, 1) for BiLSTM
        x_att = x[:, :7].unsqueeze(-1)           # (B, 7, 1)  for Transformer (first 7 dims are event counts)

        p_A, h_A = self.branch_A(x_seq)
        p_B, h_B = self.branch_B(x_att)
        p_final, gate = self.gate(x, h_A, h_B, p_A, p_B)

        return {
            'p_A': p_A,
            'p_B': p_B,
            'p_final': p_final,
            'gate': gate,
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# quick sanity check
if __name__ == '__main__':
    m = PRDENet()
    x = torch.randn(8, 46)
    out = m(x)
    print(f"Parameters: {m.count_parameters():,}")
    print(f"p_A:    {out['p_A'].shape}")
    print(f"p_B:    {out['p_B'].shape}")
    print(f"p_final:{out['p_final'].shape}")
    print(f"gate:   {out['gate'].shape}")