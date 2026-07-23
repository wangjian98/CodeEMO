"""
BiLSTM 46 维 path: 5 折交叉验证, 同时保存每折预测概率到 disk
为后续集成用.

用法:
    python models/bilstm_save_probs.py
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, get_device, set_seed
from common.feature_engineering import build_feature_matrix
from common.evaluator import evaluate
from models.bilstm.model import create_model


def train_one_model(X_train, y_train, X_val, y_val, device,
                    input_dim=46, epochs=100, batch_size=32,
                    patience=10, lr=1e-3):
    model = create_model(input_dim=input_dim)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    Xt = torch.FloatTensor(X_train).to(device)
    yt = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    Xv = torch.FloatTensor(X_val).to(device)
    yv = torch.FloatTensor(y_val).unsqueeze(1).to(device)

    n = Xt.shape[0]
    best_v = float('inf')
    best_state = None
    pc = 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = Xt[idx], yt[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            v_pred = model(Xv)
            v_loss = criterion(v_pred, yv).item()
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
        v_probs = model(Xv).squeeze(-1).cpu().numpy()
    v_preds = (v_probs > 0.5).astype(int)
    return v_preds, v_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='outputs/bilstm_save_probs')
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"设备: {device}")

    ide_logs, passed = load_ide_logs()
    X, y, student_ids = build_feature_matrix(ide_logs, passed)
    print(f"X shape: {X.shape}, passed={int((y==0).sum())}, failed={int((y==1).sum())}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_probs, fold_labels, fold_metrics = [], [], []
    all_idx = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        print(f"\n=== Fold {fold_idx}/{args.folds} | train={len(train_idx)} test={len(test_idx)} ===", flush=True)
        y_pred, y_prob = train_one_model(
            X_train_s, y_train, X_test_s, y_test, device,
            input_dim=X.shape[1],
            epochs=args.epochs, batch_size=args.batch_size,
            patience=args.patience, lr=args.lr,
        )
        m = evaluate(y_test, y_pred, y_prob)
        fold_metrics.append(m)
        fold_probs.append(y_prob)
        fold_labels.append(y_test)
        all_idx.append(test_idx)
        print(f"  Fold {fold_idx}: Acc={m['accuracy']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f}", flush=True)

    # 汇总
    print("\n========== BiLSTM 46 维 5 折汇总 ==========")
    for k in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        ms = [m[k] for m in fold_metrics]
        print(f"  {k}: {np.mean(ms):.4f} ± {np.std(ms):.4f}")

    # 保存所有概率（用于集成）
    os.makedirs(args.output_dir, exist_ok=True)
    # 拼接时按 test_idx 排序保证顺序
    all_probs = np.zeros(len(y))
    for probs, idx in zip(fold_probs, all_idx):
        all_probs[idx] = probs
    np.save(os.path.join(args.output_dir, 'probs.npy'), all_probs)
    np.save(os.path.join(args.output_dir, 'labels.npy'), y)
    np.save(os.path.join(args.output_dir, 'fold_idx.npy'),
            np.concatenate([np.full(len(idx), i, dtype=int)
                          for i, idx in enumerate(all_idx)]))
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump({
            'model': 'BiLSTM 46-dim (with probs saved)',
            'cv_metrics': {
                k: {'mean': float(np.mean([m[k] for m in fold_metrics])),
                    'std': float(np.std([m[k] for m in fold_metrics]))}
                for k in ['accuracy', 'precision', 'recall', 'f1', 'auc']
            },
            'fold_details': fold_metrics,
        }, f, indent=2)
    print(f"\n概率已保存到 {args.output_dir}/probs.npy")


if __name__ == '__main__':
    main()
