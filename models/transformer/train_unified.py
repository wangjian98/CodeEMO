"""
Transformer 统一训练脚本 - 输出到 unified_compare 格式

支持 7-dim（事件计数，每个事件类型当一个时间步）/ 46-dim（手工特征，4 段×11 维）
统一输出 failed=1 口径。

设计：
  - 7d → n_segments=7, segment_size=1（每种事件类型为一个时间步）
  - 46d → n_segments=4, segment_size=11（默认，沿用原模型配置）
  - 训练目标 y_eval = 1 - passed（默认）

用法:
    python models/transformer/train_unified.py --features 7d
    python models/transformer/train_unified.py --features 46d
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, set_seed
from common.feature_engineering import build_feature_matrix
from common.evaluator import evaluate, summarize_fold_results, print_results_table

EVENT_TYPES = [
    'text_insert', 'text_remove', 'text_paste',
    'focus_gained', 'focus_lost', 'run', 'submit'
]

# Segments plan: 7d -> 7×1, 46d -> 4×11
SEGMENT_PLAN = {
    '7d':  (7, 1),
    '46d': (4, 11),
}


def build_7dim_features(ide_logs, students):
    counts = (ide_logs
              .groupby(['student', 'eventType'])
              .size()
              .unstack(fill_value=0))
    counts = counts.reindex(columns=EVENT_TYPES, fill_value=0)
    counts = counts.reindex(index=students, fill_value=0)
    return counts.values.astype(np.float32)


class FlexTransformer(nn.Module):
    """Flexible-segment Transformer, same logic as models/transformer/model.py
    but with configurable n_segments × segment_size."""

    def __init__(self, input_dim, n_segments, d_model=64, nhead=4,
                 num_layers=3, dropout=0.2):
        super().__init__()
        seg_size = input_dim // n_segments
        self.n_segments = n_segments
        self.seg_size = seg_size
        self.proj = nn.Linear(seg_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_segments, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=128, dropout=dropout,
            activation='relu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        b = x.size(0)
        usable = self.n_segments * self.seg_size
        seg = (x[:, :usable]
               .view(b, self.n_segments, self.seg_size))
        h = self.proj(seg) + self.pos
        h = self.encoder(h)
        return self.head(h.mean(dim=1)).squeeze(-1)


def metric_dict(y_true, y_pred, y_prob):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_true, y_prob)),
    }


def per_fold(probs, labels, fold_idx, n_folds=5, thr=0.5):
    out = []
    for fi in range(n_folds):
        m = fold_idx == fi
        if m.sum() == 0:
            continue
        yi = labels[m]; pi = probs[m]
        yh = (pi > thr).astype(int)
        out.append(metric_dict(yi, yh, pi))
    return out


def summary(metrics):
    keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    out = {'n_folds_used': len(metrics)}
    for k in keys:
        vals = [m[k] for m in metrics]
        out[f'{k}_mean'] = float(np.mean(vals))
        out[f'{k}_std']  = float(np.std(vals))
    return out


def train_fold(X_tr, y_tr, X_va, y_va, device, n_segments, seg_size,
               epochs=80, batch_size=32, patience=10, lr=1e-3,
               weight_decay=1e-5, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    input_dim = X_tr.shape[1]
    model = FlexTransformer(input_dim, n_segments).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=device)

    n = len(X_tr)
    best_loss = float('inf'); best_state = None; pat_cnt = 0

    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(n)
        # mini-batch
        for i in range(0, n, batch_size):
            b_idx = idx[i:i+batch_size]
            xb = X_tr_t[b_idx]; yb = y_tr_t[b_idx]
            logit = model(xb)
            loss = bce(logit, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v_logit = model(X_va_t)
            v_loss = bce(v_logit, torch.tensor(y_va, dtype=torch.float32, device=device)).item()
        if v_loss < best_loss:
            best_loss = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat_cnt = 0
        else:
            pat_cnt += 1
        if pat_cnt >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        v_logit = model(X_va_t).cpu().numpy()
    v_prob = 1.0 / (1.0 + np.exp(-v_logit))
    return v_prob


def train_unified(features='46d', label_conv='failed1', folds=5,
                  epochs=80, batch_size=32, patience=10,
                  output_dir=None, random_state=42, device=None):
    assert features in ('7d', '46d')
    assert label_conv in ('failed1', 'passed1')

    suffix = '7dim' if features == '7d' else '46d'
    if output_dir is None:
        output_dir = os.path.join(_PROJECT_ROOT, 'outputs', 'unified_compare',
                                   f'transformer_{suffix}')
    os.makedirs(output_dir, exist_ok=True)

    n_segments, seg_size = SEGMENT_PLAN[features]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_seed(random_state)

    print('=' * 72)
    print(f'  Transformer — features={features} ({n_segments}×{seg_size}), '
          f'label_conv={label_conv}, device={device}')
    print('=' * 72)

    ide_logs, passed_df = load_ide_logs()
    student_ids = passed_df['student'].values
    y_passed = passed_df['passed'].values.astype(int)
    n = len(student_ids)
    print(f'n_students={n}, passed={y_passed.sum()}, failed={n-y_passed.sum()}')

    if features == '7d':
        X = build_7dim_features(ide_logs, student_ids)
    else:
        X_full, _, _ = build_feature_matrix(ide_logs, passed_df)
        X = X_full
    print(f'feature_dim={X.shape[1]}')

    y_eval = (1 - y_passed) if label_conv == 'failed1' else y_passed.copy()

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    fold_idx_arr = np.zeros(n, dtype=int)
    all_probs = np.zeros(n, dtype=np.float64)
    fold_metrics = []

    for fi, (tr, va) in enumerate(skf.split(X, y_eval), 1):
        fold_idx_arr[va] = fi - 1
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xva = scaler.transform(X[va])
        probs_va = train_fold(Xtr, y_eval[tr], Xva, y_eval[va], device,
                              n_segments, seg_size,
                              epochs=epochs, batch_size=batch_size,
                              patience=patience, lr=1e-3,
                              weight_decay=1e-5, seed=random_state)
        all_probs[va] = probs_va
        yhat = (probs_va > 0.5).astype(int)
        m = metric_dict(y_eval[va], yhat, probs_va)
        fold_metrics.append(m)
        print(f'  Fold {fi}/{folds}  Acc={m["accuracy"]:.4f} '
              f'F1={m["f1"]:.4f} AUC={m["auc"]:.4f}')

    cv = summary(fold_metrics)
    print_results_table(f'Transformer ({features}, label={label_conv})', {
        **{k: cv[k] for k in cv if k.endswith('_mean') or k.endswith('_std')},
        'folds': fold_metrics
    })

    np.save(os.path.join(output_dir, 'probs.npy'), all_probs)
    np.save(os.path.join(output_dir, 'labels.npy'), y_eval.astype(np.int8))
    np.save(os.path.join(output_dir, 'fold_idx.npy'), fold_idx_arr)

    out = {
        'model': 'Transformer',
        'features': features,
        'n_segments': n_segments, 'segment_size': seg_size,
        'feature_dim': int(X.shape[1]),
        'hyperparameters': {
            'd_model': 64, 'nhead': 4, 'num_layers': 3, 'dropout': 0.2,
            'epochs': epochs, 'batch_size': batch_size,
            'patience': patience, 'lr': 1e-3, 'weight_decay': 1e-5
        },
        'label_convention': 'y=1=failed' if label_conv == 'failed1' else 'y=1=passed',
        'n_folds': folds,
        'cv_results': {k: cv[k] for k in cv if k != 'n_folds_used'},
        'fold_details': fold_metrics,
        'n_samples': n,
        'n_failed': int(y_eval.sum()),
        'n_passed': int(n - y_eval.sum())
    }
    with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)

    print(f'\nSaved to: {output_dir}')
    print(f'Unified summary  F1={cv["f1_mean"]:.4f}±{cv["f1_std"]:.4f}  '
          f'AUC={cv["auc_mean"]:.4f}±{cv["auc_std"]:.4f}')
    return cv


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Transformer unified trainer')
    p.add_argument('--features', choices=['7d', '46d'], default='46d')
    p.add_argument('--label-conv', choices=['failed1', 'passed1'], default='failed1')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--output-dir', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default=None,
                   help='"cuda" / "cpu" / None (auto)')
    args = p.parse_args()
    dev = None if args.device is None else torch.device(args.device)
    train_unified(features=args.features, label_conv=args.label_conv,
                  folds=args.folds, epochs=args.epochs,
                  batch_size=args.batch_size, patience=args.patience,
                  output_dir=args.output_dir, random_state=args.seed,
                  device=dev)
