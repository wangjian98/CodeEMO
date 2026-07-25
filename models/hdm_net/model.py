"""
HDM-Net v2: 简化版 + Pre-norm Transformer + LayerScale (训练稳定性增强)
"""
import torch
import torch.nn as nn


class TreeHead(nn.Module):
    """Tree-view embedding: 7-dim event counts + 2-dim RF probs -> 32-d."""
    def __init__(self, in_dim=9, d=32, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, d), nn.ReLU(),
        )
        # init small
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.05)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x_tree, tree_probs):
        return self.net(torch.cat([x_tree, tree_probs], dim=-1))


class SeqBranch(nn.Module):
    """BiLSTM on 46-dim as 46-step univariate sequence -> 32-d."""
    def __init__(self, in_dim_per_step=1, d=32, num_layers=1, dropout=0.1):
        super().__init__()
        self.bilstm = nn.LSTM(in_dim_per_step, d, batch_first=True,
                               bidirectional=True, num_layers=num_layers,
                               dropout=dropout if num_layers > 1 else 0)
        self.proj = nn.Linear(2 * d, d)

    def forward(self, x_seq):
        # x_seq: (B, T=46, F=1)
        h, _ = self.bilstm(x_seq)
        return self.proj(h.mean(dim=1))


class AttnBranch(nn.Module):
    """Custom Pre-norm Transformer on 7-dim reshape (B, 7, 1) -> (B, 32)."""
    def __init__(self, n_segments=7, in_dim=1, d=32, nhead=4,
                  num_layers=2, dropout=0.1):
        super().__init__()
        self.n_segments = n_segments
        self.proj = nn.Linear(in_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, n_segments, d))
        nn.init.normal_(self.pos, std=0.01)
        nn.init.normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None: nn.init.zeros_(self.proj.bias)

        self.norms1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(num_layers)])
        self.attns = nn.ModuleList([
            nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norms2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(num_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, d * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d * 2, d),
            ) for _ in range(num_layers)
        ])
        # LayerScale as separate Parameter lists
        self.ls1 = nn.ParameterList([nn.Parameter(torch.ones(d) * 0.1) for _ in range(num_layers)])
        self.ls2 = nn.ParameterList([nn.Parameter(torch.ones(d) * 0.1) for _ in range(num_layers)])
        self.norm_out = nn.LayerNorm(d)

    def forward(self, x_att):
        h = self.proj(x_att) + self.pos
        for i in range(len(self.attns)):
            x_norm = self.norms1[i](h)
            attn_out, _ = self.attns[i](x_norm, x_norm, x_norm)
            h = h + self.ls1[i] * attn_out
            x_norm2 = self.norms2[i](h)
            h = h + self.ls2[i] * self.ffns[i](x_norm2)
        h = self.norm_out(h)
        return h.mean(dim=1)


class PIG(nn.Module):
    """Per-Instance Gating (softmax over 3 branch outputs)."""
    def __init__(self, d=32):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(3 * d, d), nn.ReLU(),
            nn.Linear(d, 3)
        )

    def forward(self, h_t, h_s, h_a):
        cat = torch.cat([h_t, h_s, h_a], dim=-1)
        g = torch.softmax(self.gate_mlp(cat), dim=-1)
        return g[:, 0:1] * h_t + g[:, 1:2] * h_s + g[:, 2:3] * h_a


class HDMNet(nn.Module):
    def __init__(self, n_event_types=7, d=32, dropout=0.1):
        super().__init__()
        self.tree = TreeHead(in_dim=n_event_types + 2, d=d, dropout=0.3)
        self.seq = SeqBranch(in_dim_per_step=1, d=d, dropout=dropout)
        self.attn = AttnBranch(n_segments=n_event_types, in_dim=1, d=d,
                                nhead=4, num_layers=2, dropout=dropout)
        self.pig = PIG(d=d)
        self.head = nn.Linear(d, 1)
        # init head 小
        nn.init.normal_(self.head.weight, std=0.05)
        nn.init.zeros_(self.head.bias)

    def forward(self, x_tree, tree_probs, x_seq, x_att):
        h_t = self.tree(x_tree, tree_probs)
        h_s = self.seq(x_seq)
        h_a = self.attn(x_att)
        h_final = self.pig(h_t, h_s, h_a)
        return self.head(h_final).squeeze(-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    m = HDMNet()
    print(f"Total params: {count_parameters(m):,}")
    nB = 8
    x_tree = torch.randn(nB, 7)
    tree_probs = torch.softmax(torch.randn(nB, 2), dim=-1)
    x_seq = torch.randn(nB, 46, 1)
    x_att = torch.randn(nB, 7, 1)
    out = m(x_tree, tree_probs, x_seq, x_att)
    print(f"Output shape: {out.shape}")
    print("[OK] forward pass works")
