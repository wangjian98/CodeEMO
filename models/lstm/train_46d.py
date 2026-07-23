"""
LSTM 46 维手工特征训练脚本

输入: 46维手工聚合特征 (从 common.feature_engineering.build_feature_matrix)
输出: LSTMClassifier 单向 LSTM, P(passed=1) → 翻转为 P(failed=1) 给融合用

用法:
    python models/lstm/train_46d.py
    python models/lstm/train_46d.py --folds 5 --output-dir outputs/lstm_46d
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, set_seed
from common.feature_engineering import build_feature_matrix
from common.evaluator import evaluate, summarize_fold_results, print_results_table
from models.lstm.model import LSTMClassifier


def train_one_fold(X_train, y_train, X_val, y_val, device,
                   input_dim=46, epochs=100, batch_size=32,
                   patience=10, lr=1e-3):
    """训练单折 (复用 bilstm_save_probs 模式)"""
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


def main():
    parser = argparse.ArgumentParser(description='LSTM 46d 手工特征训练')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='outputs/lstm_46d')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    ide_logs, passed = load_ide_logs()
    X, y, student_ids = build_feature_matrix(ide_logs, passed)
    print(f"X shape: {X.shape}, passed={int((y==1).sum())}, failed={int((y==0).sum())}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_probs, fold_metrics = [], []
    all_idx = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        print(f"\n=== Fold {fold_idx}/{args.folds} | train={len(train_idx)} test={len(test_idx)} ===")
        y_pred, y_prob = train_one_fold(
            X_train_s, y_train, X_test_s, y_test, device,
            input_dim=X.shape[1], epochs=args.epochs, batch_size=args.batch_size,
            patience=args.patience, lr=args.lr,
        )
        m = evaluate(y_test, y_pred, y_prob)
        fold_metrics.append(m)
        fold_probs.append(y_prob)
        all_idx.append(test_idx)
        print(f"  Fold {fold_idx}: Acc={m['accuracy']:.4f} P={m['precision']:.4f} "
              f"R={m['recall']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f}")

    print("\n========== LSTM 46d 5 折汇总 ==========")
    summary = summarize_fold_results(fold_metrics)
    print_results_table("LSTM 46-dim", summary)

    os.makedirs(args.output_dir, exist_ok=True)
    all_probs = np.zeros(len(y))
    for probs, idx in zip(fold_probs, all_idx):
        all_probs[idx] = probs
    # labels: y=1=passed (build_feature_matrix 约定)
    np.save(os.path.join(args.output_dir, 'probs.npy'), all_probs)
    np.save(os.path.join(args.output_dir, 'labels.npy'), y)
    np.save(os.path.join(args.output_dir, 'fold_idx.npy'),
            np.concatenate([np.full(len(idx), i, dtype=int) for i, idx in enumerate(all_idx)]))
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump({
            'model': 'LSTM 46-dim',
            'config': {'input_dim': 46, 'hidden_dim': 64, 'num_layers': 2,
                       'epochs': args.epochs, 'batch_size': args.batch_size,
                       'patience': args.patience, 'lr': args.lr,
                       'device': str(device)},
            'cv_results': {
                'accuracy_mean': float(summary['accuracy_mean']),
                'accuracy_std':  float(summary['accuracy_std']),
                'precision_mean': float(summary['precision_mean']),
                'precision_std':  float(summary['precision_std']),
                'recall_mean':    float(summary['recall_mean']),
                'recall_std':     float(summary['recall_std']),
                'f1_mean':        float(summary['f1_mean']),
                'f1_std':         float(summary['f1_std']),
                'auc_mean':       float(summary['auc_mean']),
                'auc_std':        float(summary['auc_std']),
            },
            'label_convention': 'y=1=passed (与 B46 一致, 融合时需 1-probs 翻转)',
            'fold_details': fold_metrics,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n概率已保存到 {args.output_dir}/probs.npy")


if __name__ == '__main__':
    main()