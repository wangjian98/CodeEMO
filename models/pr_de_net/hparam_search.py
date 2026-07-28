"""
PR-DE-Net 超参搜索：缩模型 + 调 loss 权重

目标: 把参数量从 164K 降到 ~35K（避免 n=473 过拟合），
      同时调整 alpha/beta/gamma 让 F1 超过 0.87。

运行 4 个候选配置:
  - mini_full:        默认 + 小模型（hidden=32, d=24, layers=1）
  - mini_alpha_high:  alpha=2.0, beta=0.5, gamma=1.5 (强化 RNN 分支)
  - mini_balanced:    alpha=1.5, beta=1.5, gamma=2.0 (平衡)
  - mini_gamma_low:   alpha=1.0, beta=1.0, gamma=1.0 (弱化融合 loss)
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, get_device, set_seed
from common.feature_engineering import build_feature_matrix
from common.evaluator import evaluate, summarize_fold_results, print_results_table


# ============================================================
# 紧凑版 PR-DE-Net
# ============================================================
class MiniPRRNN(nn.Module):
    def __init__(self, d=32, num_layers=1, dropout=0.3):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=1, hidden_size=d, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.proj = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(d, 1)

    def forward(self, x_seq):
        h, _ = self.bilstm(x_seq)
        h_pool = h.mean(dim=1)
        h_A = self.proj(h_pool)
        p_A = torch.sigmoid(self.head(h_A))
        return p_A, h_A


class MiniPRTrans(nn.Module):
    def __init__(self, n_segments=7, d=24, nhead=4, num_layers=1, dropout=0.2):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.cls_token, std=0.02)
        self.proj = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.zeros(1, n_segments + 1, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=d * 2,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x_att):
        B = x_att.shape[0]
        h = self.proj(x_att)
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1) + self.pos
        h = self.norm_out(self.encoder(h))
        h_B = h[:, 0, :]
        p_B = torch.sigmoid(self.head(h_B))
        return p_B, h_B


class MiniGate(nn.Module):
    def __init__(self, feat_dim=46, h_A_dim=32, h_B_dim=24, hidden=24, dropout=0.2):
        super().__init__()
        in_dim = feat_dim + h_A_dim + h_B_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, F46, h_A, h_B, p_A, p_B):
        g = torch.sigmoid(self.net(torch.cat([F46, h_A, h_B], dim=-1)))
        p_final = g * p_A + (1 - g) * p_B
        return p_final, g


class MiniPRDENet(nn.Module):
    def __init__(self, rnn_d=32, trans_d=24):
        super().__init__()
        self.A = MiniPRRNN(d=rnn_d)
        self.B = MiniPRTrans(d=trans_d)
        self.gate = MiniGate(h_A_dim=rnn_d, h_B_dim=trans_d)

    def forward(self, x):
        x_seq = x.unsqueeze(-1)
        x_att = x[:, :7].unsqueeze(-1)
        p_A, h_A = self.A(x_seq)
        p_B, h_B = self.B(x_att)
        p_final, g = self.gate(x, h_A, h_B, p_A, p_B)
        return {'p_A': p_A, 'p_B': p_B, 'p_final': p_final, 'gate': g}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# train / cv
# ============================================================
def train_one(Xtr, ytr, Xva, yva, device, alpha, beta, gamma,
              epochs=150, batch_size=32, patience=15, lr=1e-3, wd=1e-3):
    model = MiniPRDENet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt = torch.FloatTensor(Xtr).to(device)
    yt = torch.FloatTensor(ytr).to(device)
    Xv = torch.FloatTensor(Xva).to(device)
    best_f1 = -1
    best_state = None
    best_m = None
    pc = 0
    n = Xt.shape[0]
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            out = model(Xt[idx])
            p_A = out['p_A'].squeeze(-1).clamp(1e-7, 1 - 1e-7)
            p_B = out['p_B'].squeeze(-1).clamp(1e-7, 1 - 1e-7)
            p_f = out['p_final'].squeeze(-1).clamp(1e-7, 1 - 1e-7)
            y = yt[idx]
            L_A = F.binary_cross_entropy(p_A, y)
            L_B = F.binary_cross_entropy(1 - p_B, 1 - y)
            L_f = F.binary_cross_entropy(p_f, y)
            loss = alpha * L_A + beta * L_B + gamma * L_f
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        model.eval()
        with torch.no_grad():
            out_v = model(Xv)
            p_v = out_v['p_final'].squeeze(-1).cpu().numpy()
        m = evaluate(yva, (p_v > 0.5).astype(int), p_v)
        if m['f1'] > best_f1:
            best_f1 = m['f1']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_m = m
            pc = 0
        else:
            pc += 1
            if pc >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_v = model(Xv)
        pf = out_v['p_final'].squeeze(-1).cpu().numpy()
        pa = out_v['p_A'].squeeze(-1).cpu().numpy()
        pb = out_v['p_B'].squeeze(-1).cpu().numpy()
    return {'metrics': best_m, 'p_final': pf, 'p_A': pa, 'p_B': pb}


def run_one_config(X, y, name, alpha, beta, gamma, device):
    out_dir = f'outputs/pr_de_net/{name}'
    if os.path.exists(os.path.join(out_dir, 'results.json')):
        print(f"  [{name}] cache hit")
        return json.load(open(os.path.join(out_dir, 'results.json')))

    set_seed(42)
    os.makedirs(out_dir, exist_ok=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_p = np.zeros(len(y))
    all_pA = np.zeros(len(y))
    all_pB = np.zeros(len(y))
    all_fi = np.zeros(len(y), dtype=np.int64)
    fold_results = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]).astype(np.float32)
        Xva = sc.transform(X[va]).astype(np.float32)
        out = train_one(Xtr, y[tr], Xva, y[va], device, alpha, beta, gamma)
        m = out['metrics']
        print(f"  [{name} fold {fold+1}] F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} AUC={m['auc']:.4f}")
        fold_results.append(m)
        all_p[va] = out['p_final']
        all_pA[va] = out['p_A']
        all_pB[va] = out['p_B']
        all_fi[va] = fold
    summary = summarize_fold_results(fold_results)
    print_results_table(f"PR-DE-Net ({name})", summary)
    np.save(os.path.join(out_dir, 'probs.npy'), all_p)
    np.save(os.path.join(out_dir, 'probs_A.npy'), all_pA)
    np.save(os.path.join(out_dir, 'probs_B.npy'), all_pB)
    np.save(os.path.join(out_dir, 'labels.npy'), y.astype(np.int64))
    np.save(os.path.join(out_dir, 'fold_idx.npy'), all_fi)
    sj = {
        'model': f'PR-DE-Net ({name})',
        'alpha': alpha, 'beta': beta, 'gamma': gamma,
        'cv_results': {
            'accuracy_mean': summary['accuracy_mean'],
            'accuracy_std': summary['accuracy_std'],
            'precision_mean': summary['precision_mean'],
            'precision_std': summary['precision_std'],
            'recall_mean': summary['recall_mean'],
            'recall_std': summary['recall_std'],
            'f1_mean': summary['f1_mean'],
            'f1_std': summary['f1_std'],
            'auc_mean': summary['auc_mean'],
            'auc_std': summary['auc_std'],
        },
        'fold_details': fold_results,
        'n_params': MiniPRDENet().count_parameters(),
    }
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(sj, f, indent=2)
    return sj


if __name__ == '__main__':
    print("Mini-PR-DE-Net params:", MiniPRDENet().count_parameters())
    ide_logs, passed_df = load_ide_logs()
    X, y_p, _ = build_feature_matrix(ide_logs, passed_df)
    y = 1 - y_p
    device = get_device()
    configs = [
        ('mini_full',        1.0, 1.0, 2.0),
        ('mini_alpha_high',  2.0, 0.5, 1.5),
        ('mini_balanced',    1.5, 1.5, 2.0),
        ('mini_gamma_low',   1.0, 1.0, 1.0),
    ]
    rows = []
    for name, a, b, g in configs:
        print(f"\n=== {name}  alpha={a} beta={b} gamma={g} ===")
        sj = run_one_config(X, y, name, a, b, g, device)
        rows.append((name, sj['cv_results']))
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"{'name':20s} {'F1':>10s} {'P':>10s} {'R':>10s} {'AUC':>10s}")
    for name, cv in rows:
        print(f"{name:20s} {cv['f1_mean']:.4f}±{cv['f1_std']:.3f} "
              f"{cv['precision_mean']:.4f}±{cv['precision_std']:.3f} "
              f"{cv['recall_mean']:.4f}±{cv['recall_std']:.3f} "
              f"{cv['auc_mean']:.4f}±{cv['auc_std']:.3f}")