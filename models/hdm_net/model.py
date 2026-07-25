"""
HDM-Net v2: 简化版 + Pre-norm Transformer + LayerScale (训练稳定性增强)
"""
import torch
import torch.nn as nn


class TreeHead(nn.Module):
    """Tree-view embedding: 7-dim event counts + 2-dim RF probs -> d.

    Variants (selectable via args):
      depth=1  width=32 : linear only            (0 hidden, ~ 0.3K params)
      depth=2  width=32 : 9 -> 32 -> 32        (default, ~1.0K params)
      depth=3  width=32 : 9 -> 32 -> 32 -> 32  (deeper, ~1.4K params)
      depth=2  width=64 : 9 -> 64 -> 64        (wider,  ~2.0K params)
      depth=3  width=64 : 9 -> 64 -> 64 -> 64  (deeper+wider)
      depth=3  width=64 + LayerNorm             (with normalization)

    use_skip=True adds a residual connection from the input concat to
    the final hidden state, so depth>=2 becomes a proper ResNet block.
    """
    def __init__(self, in_dim=9, d=32, depth=2, dropout=0.3,
                  use_skip=False, use_bn=False):
        super().__init__()
        assert depth >= 1, "depth must be >= 1"
        self.in_dim = in_dim
        self.d = d
        self.depth = depth
        self.use_skip = use_skip and (in_dim <= d)  # only meaningful if d >= in_dim
        self.use_bn = use_bn
        layers = []
        prev = in_dim
        for i in range(depth):
            layers.append(nn.Linear(prev, d))
            if use_bn:
                layers.append(nn.LayerNorm(d))
            layers.append(nn.ReLU())
            if dropout > 0 and i < depth - 1:
                layers.append(nn.Dropout(dropout))
            prev = d
        # remove the LAST ReLU if it's the last layer (we want raw features)
        if isinstance(layers[-1], nn.ReLU):
            layers = layers[:-1]
        self.net = nn.Sequential(*layers)
        # skip projection if dimensions mismatched
        if self.use_skip and in_dim != d:
            self.skip_proj = nn.Linear(in_dim, d)
        else:
            self.skip_proj = None
        # init small
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.05)
                if m.bias is not None: nn.init.zeros_(m.bias)
        if self.skip_proj is not None:
            nn.init.normal_(self.skip_proj.weight, std=0.05)
            nn.init.zeros_(self.skip_proj.bias)

    def forward(self, x_tree, tree_probs):
        x = torch.cat([x_tree, tree_probs], dim=-1)
        h = self.net(x)
        if self.use_skip:
            skip = x if self.skip_proj is None else self.skip_proj(x)
            h = h + skip
        return h


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
    """HDM-Net with optional ablation: disable any of (tree, seq, attn) branch.
    Disabled branch outputs zeros of shape (B, d) so PIG still receives 3 inputs
    but learns to weight the remaining views more. Total params shrinks accordingly
    when measured without head/PIG (since the branch layers still exist in memory
    but are zeroed at forward time).
    """
    def __init__(self, n_event_types=7, d=32, dropout=0.1,
                  disable_tree=False, disable_seq=False, disable_attn=False,
                  tree_depth=2, tree_width=None, tree_use_skip=False,
                  tree_use_bn=False):
        super().__init__()
        self.d = d
        self.disable_tree = disable_tree
        self.disable_seq = disable_seq
        self.disable_attn = disable_attn
        # TreeHead now configurable; default behaviour preserved.
        if tree_width is None: tree_width = d
        self.tree = TreeHead(in_dim=n_event_types + 2, d=tree_width,
                              depth=tree_depth, dropout=0.3,
                              use_skip=tree_use_skip, use_bn=tree_use_bn)
        # If tree returns width != d, project to d at HDMNet level via identity
        # using a small linear to keep PIG dimension consistent:
        if tree_width != d:
            self.tree_proj = nn.Linear(tree_width, d)
            nn.init.normal_(self.tree_proj.weight, std=0.05)
            nn.init.zeros_(self.tree_proj.bias)
        else:
            self.tree_proj = None
        self.seq = SeqBranch(in_dim_per_step=1, d=d, dropout=dropout)
        self.attn = AttnBranch(n_segments=n_event_types, in_dim=1, d=d,
                                nhead=4, num_layers=2, dropout=dropout)
        self.pig = PIG(d=d)
        self.head = nn.Linear(d, 1)
        # init head 小
        nn.init.normal_(self.head.weight, std=0.05)
        nn.init.zeros_(self.head.bias)

    def forward(self, x_tree, tree_probs, x_seq, x_att):
        B = x_tree.size(0)
        if self.disable_tree:
            h_t = torch.zeros(B, self.d, device=x_tree.device, dtype=x_tree.dtype)
        else:
            h_t = self.tree(x_tree, tree_probs)
            if self.tree_proj is not None:
                h_t = self.tree_proj(h_t)
        if self.disable_seq:
            h_s = torch.zeros(B, self.d, device=x_tree.device, dtype=x_tree.dtype)
        else:
            h_s = self.seq(x_seq)
        if self.disable_attn:
            h_a = torch.zeros(B, self.d, device=x_tree.device, dtype=x_tree.dtype)
        else:
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
