"""
方案 1: 数据增强对比实验

训练 LSTM_46d,对比 4 种策略:
  A. baseline (无增强)
  B. Gaussian Noise (sigma=0.08)
  C. Feature Dropout (p=0.1)
  D. SMOTE-like (同层插值)
  E. Combined (3x 数据, 噪+丢+SMOTE)
  F. Combined (5x 数据, 噪+丢+SMOTE)

每种策略都做 5 折交叉验证,保存指标并输出对比表
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, set_seed
from common.feature_engineering import build_feature_matrix
from common.evaluator import evaluate
from models.lstm.model import LSTMClassifier
from augment.augmentations import gaussian_noise, feature_dropout, smote_like, combined_augment


def train_one_fold(X_train, y_train, X_val, y_val, device,
                   input_dim=46, epochs=100, batch_size=32,
                   patience=10, lr=1e-3):
    """训练单折"""
    model = LSTMClassifier(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    Xt = torch.FloatTensor(X_train).to(device)
    yt = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    Xv = torch.FloatTensor(X_val).to(device)
    yv = torch.FloatTensor(y_val).unsqueeze(1).to(device)

    best_v = float('inf')
    best_state = None
    pc = 0
    n = Xt.shape[0]
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            loss = criterion(model(Xt[idx]), yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            v_loss = criterion(model(Xv), yv).item()
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
        probs = model(Xv).squeeze(-1).cpu().numpy()
    preds = (probs > 0.5).astype(int)
    return preds, probs


def run_cv(X, y, name, device, n_folds=5, seed=42, augment_fn=None,
           epochs=100, batch_size=32, lr=1e-3):
    """5 折交叉验证,可选数据增强"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics = []

    print(f"\n{'='*60}")
    print(f"实验: {name}")
    print(f"原始数据量: {len(X)}, y=1: {(y==1).sum()}, y=0: {(y==0).sum()}")
    print(f"{'='*60}")

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train_raw, X_test = X[train_idx], X[test_idx]
        y_train_raw, y_test = y[train_idx], y[test_idx]

        # 增强只在训练集上做
        if augment_fn is not None:
            X_train_aug, y_train_aug = augment_fn(X_train_raw, y_train_raw)
            print(f"\n--- Fold {fold_idx}/{n_folds} | train_aug={len(X_train_aug)} test={len(X_test)} ---")
        else:
            X_train_aug, y_train_aug = X_train_raw, y_train_raw
            print(f"\n--- Fold {fold_idx}/{n_folds} | train={len(X_train_aug)} test={len(X_test)} ---")

        # 标准化
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_aug)
        X_test_s = scaler.transform(X_test)

        y_pred, y_prob = train_one_fold(
            X_train_s, y_train_aug, X_test_s, y_test, device,
            input_dim=X.shape[1], epochs=epochs, batch_size=batch_size,
            patience=10, lr=lr,
        )
        m = evaluate(y_test, y_pred, y_prob)
        fold_metrics.append(m)
        print(f"  Fold {fold_idx}: Acc={m['accuracy']:.4f} P={m['precision']:.4f} "
              f"R={m['recall']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f}")

    # 汇总
    keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    summary = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        summary[f'{k}_mean'] = float(np.mean(vals))
        summary[f'{k}_std'] = float(np.std(vals))

    print(f"\n>>> {name} 5折均值:")
    print(f"    Acc={summary['accuracy_mean']:.4f}±{summary['accuracy_std']:.4f} "
          f"P={summary['precision_mean']:.4f}±{summary['precision_std']:.4f} "
          f"R={summary['recall_mean']:.4f}±{summary['recall_std']:.4f} "
          f"F1={summary['f1_mean']:.4f}±{summary['f1_std']:.4f} "
          f"AUC={summary['auc_mean']:.4f}±{summary['auc_std']:.4f}")

    return summary, fold_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='outputs/aug_compare')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")

    # 加载数据
    ide_logs, passed = load_ide_logs()
    X, y, student_ids = build_feature_matrix(ide_logs, passed)
    print(f"\n数据集: {X.shape}, passed={int((y==1).sum())}, failed={int((y==0).sum())}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    results = {}

    # A. Baseline
    s, _ = run_cv(X, y, "A. Baseline (无增强)", device,
                  n_folds=args.folds, seed=args.seed,
                  augment_fn=None, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    results['A_baseline'] = s

    # B. Gaussian Noise
    s, _ = run_cv(X, y, "B. Gaussian Noise (σ=0.08)", device,
                  n_folds=args.folds, seed=args.seed,
                  augment_fn=lambda Xt, yt: gaussian_noise(Xt, yt, sigma=0.08, seed=args.seed),
                  epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    results['B_noise'] = s

    # C. Feature Dropout
    s, _ = run_cv(X, y, "C. Feature Dropout (p=0.1)", device,
                  n_folds=args.folds, seed=args.seed,
                  augment_fn=lambda Xt, yt: feature_dropout(Xt, yt, p=0.1, seed=args.seed),
                  epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    results['C_dropout'] = s

    # D. SMOTE-like
    s, _ = run_cv(X, y, "D. SMOTE-like (同层插值)", device,
                  n_folds=args.folds, seed=args.seed,
                  augment_fn=lambda Xt, yt: smote_like(Xt, yt, n_samples=len(Xt), seed=args.seed),
                  epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    results['D_smote'] = s

    # E. Combined 3x
    s, _ = run_cv(X, y, "E. Combined Aug (3x 数据)", device,
                  n_folds=args.folds, seed=args.seed,
                  augment_fn=lambda Xt, yt: combined_augment(Xt, yt, factor=3, seed=args.seed),
                  epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    results['E_combined3x'] = s

    # F. Combined 5x
    s, _ = run_cv(X, y, "F. Combined Aug (5x 数据)", device,
                  n_folds=args.folds, seed=args.seed,
                  augment_fn=lambda Xt, yt: combined_augment(Xt, yt, factor=5, seed=args.seed),
                  epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    results['F_combined5x'] = s

    # 保存 JSON
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 输出对比表
    print("\n\n")
    print("=" * 100)
    print(" " * 35 + "方案 1 - 数据增强对比表 (5 折均值)")
    print("=" * 100)
    print(f"{'策略':<28} {'Acc':>9} {'P':>9} {'R':>9} {'F1':>9} {'AUC':>9}    {'vs Baseline F1':>15}")
    print("-" * 100)
    base_f1 = results['A_baseline']['f1_mean']
    label_map = {
        'A_baseline': 'A. Baseline',
        'B_noise': 'B. Gaussian Noise',
        'C_dropout': 'C. Feature Dropout',
        'D_smote': 'D. SMOTE-like',
        'E_combined3x': 'E. Combined 3x ⭐',
        'F_combined5x': 'F. Combined 5x',
    }
    best_name, best_f1 = None, 0
    for k, v in results.items():
        delta = v['f1_mean'] - base_f1
        sign = '+' if delta >= 0 else ''
        print(f"{label_map[k]:<28} "
              f"{v['accuracy_mean']:.4f}   "
              f"{v['precision_mean']:.4f}   "
              f"{v['recall_mean']:.4f}   "
              f"{v['f1_mean']:.4f}   "
              f"{v['auc_mean']:.4f}    "
              f"{sign}{delta:.4f}")
        if v['f1_mean'] > best_f1:
            best_f1 = v['f1_mean']
            best_name = label_map[k]
    print("-" * 100)
    print(f"⭐ 最优策略: {best_name}, F1={best_f1:.4f}")
    print(f"   相对 Baseline F1 提升: {best_f1 - base_f1:+.4f}")
    print(f"   7 路 Late Fusion 对比: F1=0.9013 (来自之前实验)")
    print("=" * 100)


if __name__ == '__main__':
    main()