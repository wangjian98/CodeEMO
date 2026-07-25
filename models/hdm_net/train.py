"""
HDM-Net 训练脚本 - 5-fold CV, failed=1 口径
==============================================
数据准备:
  * x_tree (n, 7): 7-dim event counts
  * tree_probs (n, 2): per-student RF 概率 (frozen feature, 来自同一 fold)
  * x_seq (n, 4, 11): 46-dim reshaped to 4×11
  * x_att (n, 7, 1): 7-dim reshaped to 7×1
  * y (n,): 0/1, failed=1

训练:
  * EarlyStopping (patience=10, BCEWithLogitsLoss)
  * AdamW, lr=3e-3, weight_decay=1e-3
  * CosineAnnealingLR T_max=80
  * 5-fold StratifiedKFold, seed=42 (与现有 unified 一致)
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)
from sklearn.linear_model import LogisticRegression

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, set_seed
from common.feature_engineering import build_feature_matrix
from common.evaluator import evaluate, summarize_fold_results, print_results_table

from models.hdm_net.model import HDMNet, count_parameters

EVENT_TYPES = [
    'text_insert', 'text_remove', 'text_paste',
    'focus_gained', 'focus_lost', 'run', 'submit'
]


def build_7dim(ide_logs, students):
    counts = (ide_logs
              .groupby(['student', 'eventType'])
              .size()
              .unstack(fill_value=0))
    counts = counts.reindex(columns=EVENT_TYPES, fill_value=0)
    counts = counts.reindex(index=students, fill_value=0)
    return counts.values.astype(np.float32)


def make_tree_oof_probs(X7, y, n_splits=5, seed=42):
    """Per-fold OOF RF probs: for each fold, train RF on train, predict on val+test.
    返回 (n, 2) per-student probs, indexed same as X7.
    """
    n = len(y)
    oof = np.zeros((n, 2), dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(X7, y):
        clf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                      random_state=seed, n_jobs=-1)
        clf.fit(X7[tr], y[tr])
        oof[va] = clf.predict_proba(X7[va])
    return oof


def metric_dict(y_true, y_pred, y_prob):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_true, y_prob)),
    }


def summary(metrics):
    keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    out = {'n_folds_used': len(metrics)}
    for k in keys:
        vals = [m[k] for m in metrics]
        out[f'{k}_mean'] = float(np.mean(vals))
        out[f'{k}_std'] = float(np.std(vals))
    return out


def train_one_fold(model, optimizer, scheduler, bce, device,
                    x_tree_tr, tree_probs_tr, x_seq_tr, x_att_tr, y_tr,
                    x_tree_va, tree_probs_va, x_seq_va, x_att_va, y_va,
                    epochs=80, batch_size=32, patience=10, fold_idx=None):
    best_vloss = float('inf')
    best_state = None
    pat = 0

    # to torch
    x_tree_tr_t = torch.tensor(x_tree_tr, dtype=torch.float32, device=device)
    tree_probs_tr_t = torch.tensor(tree_probs_tr, dtype=torch.float32, device=device)
    x_seq_tr_t = torch.tensor(x_seq_tr, dtype=torch.float32, device=device)
    x_att_tr_t = torch.tensor(x_att_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)

    x_tree_va_t = torch.tensor(x_tree_va, dtype=torch.float32, device=device)
    tree_probs_va_t = torch.tensor(tree_probs_va, dtype=torch.float32, device=device)
    x_seq_va_t = torch.tensor(x_seq_va, dtype=torch.float32, device=device)
    x_att_va_t = torch.tensor(x_att_va, dtype=torch.float32, device=device)
    y_va_t = torch.tensor(y_va, dtype=torch.float32, device=device)

    n = len(y_tr)
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(n)
        for i in range(0, n, batch_size):
            b = idx[i:i+batch_size]
            logits = model(x_tree_tr_t[b], tree_probs_tr_t[b],
                            x_seq_tr_t[b], x_att_tr_t[b])
            logits = torch.clamp(logits, min=-30, max=30)
            loss = bce(logits, y_tr_t[b])
            if torch.isnan(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        # validation
        model.eval()
        with torch.no_grad():
            v_logits = model(x_tree_va_t, tree_probs_va_t,
                              x_seq_va_t, x_att_va_t)
            v_loss = bce(v_logits, y_va_t).item()

        if v_loss < best_vloss:
            best_vloss = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        v_logits = model(x_tree_va_t, tree_probs_va_t,
                          x_seq_va_t, x_att_va_t).cpu().numpy()
    v_probs = 1.0 / (1.0 + np.exp(-v_logits))
    return v_probs, best_vloss


def main(output_dir=None, folds=5, epochs=80, batch_size=32, patience=10, seed=42):
    if output_dir is None:
        output_dir = os.path.join(_PROJECT_ROOT, 'outputs', 'unified_compare', 'hdm_net')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 72)
    print("  HDM-Net: Heterogeneous Decoder Mixture Network")
    print(f"  output: {output_dir}")
    print("=" * 72)

    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    # ---- data ----
    ide_logs, passed_df = load_ide_logs()
    students = passed_df['student'].values
    y_passed = passed_df['passed'].values.astype(int)
    n = len(students)

    X7 = build_7dim(ide_logs, students)
    X_full, _, _ = build_feature_matrix(ide_logs, passed_df)  # (n, 46)
    y_eval = 1 - y_passed  # failed=1

    # 准备三种 view 输入
    x_att_all = X7.reshape(n, 7, 1).astype(np.float32)
    x_seq_all = X_full.reshape(n, 46, 1).astype(np.float32)  # 46 time-steps x 1-dim
    x_tree_all = X7.astype(np.float32)

    # ---- Step 1: 5-fold OOF RF probs (作为 Tree branch 固定特征) ----
    print("\n[Step 1] Computing 5-fold OOF RF probs...")
    tree_probs_all = make_tree_oof_probs(X7, y_eval, n_splits=folds, seed=seed)
    print(f"  RF probs shape={tree_probs_all.shape}, "
          f"col0 mean={tree_probs_all[:, 0].mean():.4f} (= P(passed=1)), "
          f"col1 mean={tree_probs_all[:, 1].mean():.4f} (= P(failed=1))")

    # ---- Step 2: HDM-Net 5-fold CV ----
    print(f"\n[Step 2] HDM-Net 5-fold CV (epochs={epochs}, batch={batch_size}, patience={patience})")
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_idx_arr = np.zeros(n, dtype=int)
    all_probs = np.zeros(n, dtype=np.float64)
    fold_metrics = []

    for fi, (tr, va) in enumerate(skf.split(X7, y_eval), 1):
        # 标准化 seq & tree
        scaler = StandardScaler()
        x_seq_tr = scaler.fit_transform(x_seq_all[tr].reshape(-1, 1)).reshape(-1, 46, 1).astype(np.float32)
        x_seq_va = scaler.transform(x_seq_all[va].reshape(-1, 1)).reshape(-1, 46, 1).astype(np.float32)
        scaler2 = StandardScaler()
        x_tree_tr = scaler2.fit_transform(x_tree_all[tr]).astype(np.float32)
        x_tree_va = scaler2.transform(x_tree_all[va]).astype(np.float32)
        x_att_tr = x_att_all[tr]; x_att_va = x_att_all[va]
        # RF probs 已经是 [0,1] 区间概率向量，不需要标准化
        tree_probs_tr = tree_probs_all[tr].astype(np.float32)
        tree_probs_va = tree_probs_all[va].astype(np.float32)

        # 初始化 model
        torch.manual_seed(seed + fi); np.random.seed(seed + fi)
        model = HDMNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                        weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        bce = nn.BCEWithLogitsLoss()

        # train
        probs_va, vloss = train_one_fold(
            model, optimizer, scheduler, bce, device,
            x_tree_tr, tree_probs_tr, x_seq_tr, x_att_tr, y_eval[tr],
            x_tree_va, tree_probs_va, x_seq_va, x_att_va, y_eval[va],
            epochs=epochs, batch_size=batch_size, patience=patience,
            fold_idx=fi,
        )

        all_probs[va] = probs_va
        fold_idx_arr[va] = fi - 1
        yhat = (probs_va > 0.5).astype(int)
        m = metric_dict(y_eval[va], yhat, probs_va)
        fold_metrics.append(m)
        print(f"  Fold {fi}/{folds}  Acc={m['accuracy']:.4f} "
              f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
              f"F1={m['f1']:.4f}  AUC={m['auc']:.4f}  vloss={vloss:.4f}")

    cv = summary(fold_metrics)
    print_results_table('HDM-Net (label=failed1)', {
        **{k: cv[k] for k in cv if k.endswith('_mean') or k.endswith('_std')},
        'folds': fold_metrics
    })

    # ---- save ----
    np.save(os.path.join(output_dir, 'probs.npy'), all_probs)
    np.save(os.path.join(output_dir, 'labels.npy'), y_eval.astype(np.int8))
    np.save(os.path.join(output_dir, 'fold_idx.npy'), fold_idx_arr)

    out = {
        'model': 'HDM-Net',
        'label_convention': 'y=1=failed',
        'n_folds': folds,
        'architecture': {
            'n_params': int(count_parameters(HDMNet())),
            'tree_branch': '7-dim events + 2-dim RF probs -> MLP -> 32-d',
            'seq_branch':  'BiLSTM on 4×11 segments -> 32-d',
            'attn_branch': 'Transformer on 7×1 segments -> 32-d',
            'fusion': 'XCA (cross-view cross-attn) + PIG (per-instance gating)',
        },
        'cv_results': {k: cv[k] for k in cv if k != 'n_folds_used'},
        'fold_details': fold_metrics,
        'n_samples': int(n),
        'n_failed': int(y_eval.sum()),
        'n_passed': int(n - y_eval.sum()),
    }
    with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)

    print(f"\n[OK] Saved to: {output_dir}")
    print(f"\nUnified comparison summary:")
    print(f"  Accuracy:  {cv['accuracy_mean']:.4f} ± {cv['accuracy_std']:.4f}")
    print(f"  Precision: {cv['precision_mean']:.4f} ± {cv['precision_std']:.4f}")
    print(f"  Recall:    {cv['recall_mean']:.4f} ± {cv['recall_std']:.4f}")
    print(f"  F1:        {cv['f1_mean']:.4f} ± {cv['f1_std']:.4f}")
    print(f"  AUC:       {cv['auc_mean']:.4f} ± {cv['auc_std']:.4f}")
    print(f"  Parameters: {count_parameters(HDMNet()):,}")
    return cv


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='HDM-Net unified trainer')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--output-dir', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    main(output_dir=args.output_dir, folds=args.folds, epochs=args.epochs,
         batch_size=args.batch_size, patience=args.patience, seed=args.seed)
