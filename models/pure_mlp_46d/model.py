#!/usr/bin/env python3
"""
消融实验: Pure MLP-46D vs LSTM-Seq-7D vs LSTM-MLP-46D

目的: 验证 "赢的关键是 46d 信息密度" 而非 "LSTM 的 gating"

实验设计:
- Pure MLP-46D: 纯 MLP,无 LSTM,无 attention
  46 → Linear(64) → BN → ReLU → Dropout → Linear(64) → BN → ReLU → Dropout → Linear(1)
- LSTM-MLP-46D (已有): Linear(46→64) → unsqueeze(1) → LSTM(seq_len=1) → Linear→Linear
- LSTM-Seq-7D (已有): 真时序事件序列模型, max_seq_len=500

如果 Pure MLP-46D ≈ LSTM-MLP-46D >> LSTM-Seq-7D:
  → 证明 "信息密度" 是关键, LSTM 的 gating 贡献微弱
如果 LSTM-MLP-46D >> Pure MLP-46D:
  → 证明 LSTM 的 gating 也有贡献 (但仍然可能 << 信息密度)

数据集: CS1, n=473, 5-fold StratifiedKFold (random_state=42, failed=1)
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)

OUT_DIR = '/home/ubuntu/CodeEMO/outputs/unified_compare/pure_mlp_46d'
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CACHE_DIR = '/tmp/codeemo_features'
SEED = 42


class PureMLPClassifier(nn.Module):
    """Pure MLP on 46-dim hand-crafted features (消融对照).

    与 LSTM-MLP-46D 的关键区别: 没有 LSTM 包装,没有 gating.
    纯粹的 2 层 MLP,BatchNorm + Dropout.
    """
    def __init__(self, input_dim=46, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one_fold(X_tr, y_tr, X_va, y_va,
                   epochs=120, batch_size=32, lr=1e-3, weight_decay=1e-4,
                   patience=15, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = PureMLPClassifier(input_dim=X_tr.shape[1]).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss()

    Xt = torch.FloatTensor(X_tr).to(DEVICE)
    yt = torch.FloatTensor(y_tr).to(DEVICE)
    Xv = torch.FloatTensor(X_va).to(DEVICE)
    yv = torch.FloatTensor(y_va).to(DEVICE)

    n = len(y_tr)
    best_v = float('inf')
    best_state = None
    pc = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits = model(Xt[idx])
            loss = crit(logits, yt[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            v_loss = crit(model(Xv), yv).item()
        if v_loss < best_v:
            best_v = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1
            if pc >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(Xv).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def metric(y_true, y_pred, y_prob):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_true, y_prob)),
    }


def summary(folds):
    keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    out = {'n_folds': len(folds)}
    for k in keys:
        v = [m[k] for m in folds]
        out[f'{k}_mean'] = float(np.mean(v))
        out[f'{k}_std'] = float(np.std(v))
    return out


def run():
    print(f'=== Pure MLP-46D 消融对照实验 ===')
    print(f'Device: {DEVICE}, seed: {SEED}')
    print(f'输出目录: {OUT_DIR}')

    # 加载 46d 特征
    X46 = np.load(os.path.join(CACHE_DIR, 'X_46d.npy'))
    y_pass = np.load(os.path.join(CACHE_DIR, 'y.npy'))
    y_failed = 1 - y_pass
    n = len(y_failed)
    print(f'X46: {X46.shape}, y_failed pos_rate: {y_failed.mean():.4f}')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_idx = np.zeros(n, dtype=int)
    oof = np.zeros(n, dtype=np.float64)
    fold_metrics = []

    for fi, (tr, va) in enumerate(skf.split(X46, y_failed)):
        fold_idx[va] = fi
        sc = StandardScaler().fit(X46[tr])
        Xtr_s = sc.transform(X46[tr]).astype(np.float32)
        Xva_s = sc.transform(X46[va]).astype(np.float32)

        t0 = time.time()
        probs = train_one_fold(
            Xtr_s, y_failed[tr].astype(np.float32),
            Xva_s, y_failed[va].astype(np.float32),
            seed=SEED + fi,
        )
        elapsed = time.time() - t0
        oof[va] = probs
        pred = (probs > 0.5).astype(int)
        m = metric(y_failed[va], pred, probs)
        m['elapsed_seconds'] = round(elapsed, 2)
        fold_metrics.append(m)
        print(f'Fold {fi+1}: Acc={m["accuracy"]:.4f} F1={m["f1"]:.4f} '
              f'Prec={m["precision"]:.4f} Rec={m["recall"]:.4f} AUC={m["auc"]:.4f} '
              f'({elapsed:.1f}s)')

    cv = summary(fold_metrics)
    print()
    print('=== 5-fold CV 汇总 ===')
    print(f'F1: {cv["f1_mean"]:.4f} ± {cv["f1_std"]:.4f}')
    print(f'AUC: {cv["auc_mean"]:.4f} ± {cv["auc_std"]:.4f}')
    print(f'Accuracy: {cv["accuracy_mean"]:.4f} ± {cv["accuracy_std"]:.4f}')
    print(f'Precision: {cv["precision_mean"]:.4f}')
    print(f'Recall: {cv["recall_mean"]:.4f}')

    np.save(os.path.join(OUT_DIR, 'probs.npy'), oof)
    np.save(os.path.join(OUT_DIR, 'labels.npy'), y_failed.astype(np.int8))
    np.save(os.path.join(OUT_DIR, 'fold_idx.npy'), fold_idx)

    report = {
        'model': 'PureMLP-46D',
        'description': '消融实验: 纯 MLP (无 LSTM, 无 attention) on 46d hand-crafted features',
        'architecture': {
            'type': 'Pure MLP',
            'layers': '46 -> Linear(64) -> BN -> ReLU -> Dropout -> Linear(64) -> BN -> ReLU -> Dropout -> Linear(1)',
            'n_params': 46*64 + 64 + 64*64 + 64 + 64 + 1,  # ≈ 7841
            'activation': 'ReLU',
            'normalization': 'BatchNorm1d',
            'dropout': 0.3,
        },
        'training': {
            'optimizer': 'Adam',
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'batch_size': 32,
            'epochs': 120,
            'patience': 15,
            'gradient_clip': 1.0,
            'cv': '5-fold StratifiedKFold (random_state=42)',
        },
        'dataset': {
            'name': 'CS1',
            'n_samples': int(n),
            'n_failed': int(y_failed.sum()),
            'n_passed': int((1 - y_failed).sum()),
            'pos_rate_failed': float(y_failed.mean()),
        },
        'cv_results': {k: v for k, v in cv.items() if k != 'folds'},
        'fold_details': fold_metrics,
        'label_convention': 'y=1=failed',
        'comparison_target': {
            'LSTM-Seq-7D': {
                'F1': 0.8062, 'AUC': 0.7259,
                'note': '真时序事件序列模型, max_seq_len=500'
            },
            'LSTM-MLP-46D': {
                'F1': 0.8622, 'AUC': 0.9170,
                'note': '现有 LSTM-46D, 实际是 gated MLP (seq_len=1)'
            },
        },
    }

    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\n[SAVED] {OUT_DIR}/results.json')


if __name__ == '__main__':
    t0 = time.time()
    run()
    print(f'\nTotal time: {time.time()-t0:.1f}s')