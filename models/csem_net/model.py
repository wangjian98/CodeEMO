"""
CSEM-Net v3: STABLE
- 3 streams + concat (incl. raw RF probs as feature) + 2-layer MLP head
- Standard BCE without pos_weight
- Higher dropout (0.5) to prevent fold-specific overfitting
"""
import torch
import torch.nn as nn


class CNNAttentionStream(nn.Module):
    def __init__(self, n_kernels=8, nhead=2, dropout=0.3):
        super().__init__()
        self.conv = nn.Conv1d(1, n_kernels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(n_kernels)
        self.attn = nn.MultiheadAttention(n_kernels, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(n_kernels)
        self.feature_head = nn.Linear(7 * n_kernels, 32)
        for m in [self.feature_head]:
            nn.init.normal_(m.weight, std=0.05)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.size(0)
        h = x.unsqueeze(1)
        h = torch.relu(self.bn(self.conv(h)))
        h = h.permute(0, 2, 1)
        attn_out, _ = self.attn(h, h, h)
        h = self.norm(attn_out + h)
        h = h.flatten(1)
        return self.feature_head(h)


class BiLSTMStream(nn.Module):
    def __init__(self, hidden=32, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(11, hidden, batch_first=True, bidirectional=True)
        self.feature_proj = nn.Linear(2 * hidden, 32)
        for m in [self.feature_proj]:
            nn.init.normal_(m.weight, std=0.05)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.size(0)
        x = x[:, :44].view(B, 4, 11)
        h, _ = self.lstm(x)
        z = h.mean(dim=1)
        return self.feature_proj(z)


class CrossFeatureMLPStream(nn.Module):
    def __init__(self, in_dim=7+46+2, hidden=32, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_7d, x_46d, rf_probs):
        cat = torch.cat([x_7d, x_46d, rf_probs], dim=-1)
        return self.net(cat)


class CSEMNet(nn.Module):
    """v3: RF probs added as direct features to head (skip connection)."""
    def __init__(self, n_event_types=7, dropout=0.3):
        super().__init__()
        self.cnn_attn = CNNAttentionStream(n_kernels=8, nhead=2, dropout=dropout)
        self.lstm = BiLSTMStream(hidden=32, dropout=dropout)
        self.cross = CrossFeatureMLPStream(in_dim=n_event_types + 46 + 2,
                                            hidden=32, dropout=0.5)
        # Concat head: 3 stream feats (32 each) + RF probs (2) = 98-dim
        self.head = nn.Sequential(
            nn.Linear(32 + 32 + 32 + 2, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1)
        )
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.05)
                nn.init.zeros_(m.bias)

    def forward(self, x_7d, x_46d, rf_probs):
        feat_local = self.cnn_attn(x_7d)
        feat_seq = self.lstm(x_46d)
        feat_fuse = self.cross(x_7d, x_46d, rf_probs)
        cat = torch.cat([feat_local, feat_seq, feat_fuse, rf_probs], dim=-1)
        return self.head(cat).squeeze(-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    m = CSEMNet()
    print(f"Total params: {count_parameters(m):,}")
    nB = 8
    x_7d = torch.randn(nB, 7); x_46d = torch.randn(nB, 46); rf = torch.softmax(torch.randn(nB, 2), dim=-1)
    out = m(x_7d, x_46d, rf)
    print(f"Output shape: {out.shape}")
    print("[OK] v3 forward pass works")