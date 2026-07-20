"""
方案 2: 特征扩展 + Attention Pooling 对比实验

4 个实验:
  A. 46d + mean pooling (重跑 baseline)
  B. 46d + attention pooling
  C. 扩展(~136d) + mean pooling
  D. 扩展(~136d) + attention pooling ⭐

每组 5 折交叉验证,输出完整对比表
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, set_seed
from common.feature_engineering import build_feature_matrix
from common.evaluator import evaluate
from augment.feature_engineering_v2 import build_extended_features, combine_features
from models.lstm.model_attn import LSTMClassifierWithAttention, LSTMClassifierMeanPool


def train_one_fold(model_cls, X_train, y_train, X_val, y_val, device,
                   epochs=100, batch_size=32, patience=15, lr=1e-3, **model_kwargs):
    """训练单折"""
    model = model_cls(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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


def run_cv(X, y, name, device, n_folds=5, seed=42, epochs=100,
           batch_size=32, lr=1e-3, model_cls=None, **model_kwargs):
    """5 折交叉验证"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics = []

    print(f"\n{'='*70}")
    print(f"实验: {name} | 输入维度: {X.shape[1]}")
    print(f"样本: {len(X)}, y=1: {(y==1).sum()}, y=0: {(y==0).sum()}")
    print(f"{'='*70}")

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        print(f"\n--- Fold {fold_idx}/{n_folds} | train={len(train_idx)} test={len(test_idx)} ---")
        y_pred, y_prob = train_one_fold(
            model_cls, X_train_s, y_train, X_test_s, y_test, device,
            epochs=epochs, batch_size=batch_size, lr=lr, **model_kwargs
        )
        m = evaluate(y_test, y_pred, y_prob)
        fold_metrics.append(m)
        print(f"  Fold {fold_idx}: Acc={m['accuracy']:.4f} P={m['precision']:.4f} "
              f"R={m['recall']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f}")

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
    parser.add_argument('--output-dir', type=str, default='outputs/v2_compare')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")

    # 加载原始 46d 特征
    print("[1/2] 加载 46d 原特征...")
    ide_logs, passed = load_ide_logs()
    X_orig, y, student_ids = build_feature_matrix(ide_logs, passed)
    print(f"      X_orig: {X_orig.shape}, passed={int((y==1).sum())}, failed={int((y==0).sum())}")

    # 加载扩展特征
    print("\n[2/2] 计算扩展特征 (~136d)...")
    X_ext, ext_names, ext_sids = build_extended_features(ide_logs)
    print(f"      X_ext: {X_ext.shape}, 新增特征数: {len(ext_names)}")

    # 对齐 (确保 student 顺序一致)
    sid_to_ext = dict(zip(ext_sids, X_ext))
    X_ext_aligned = np.array([sid_to_ext.get(sid, np.zeros(X_ext.shape[1])) for sid in student_ids])

    # 拼接
    X_combined = combine_features(X_orig, X_ext_aligned)
    print(f"\n合并后: X_combined: {X_combined.shape}")
    print(f"原 46d + 扩展 {X_ext.shape[1]}d = 总 {X_combined.shape[1]}d")

    os.makedirs(args.output_dir, exist_ok=True)

    results = {}

    # A. 46d + mean pooling (用原 LSTM model 验证 baseline)
    print("\n\n" + "=" * 70)
    print("【A 组】46d + Mean Pooling (重跑 baseline)")
    print("=" * 70)
    sys.path.insert(0, _PROJECT_ROOT)
    from models.lstm.model import LSTMClassifier as OrigLSTM
    s, _ = run_cv(X_orig, y, "A. 46d + Mean Pooling", device,
                  n_folds=args.folds, seed=args.seed,
                  epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                  model_cls=OrigLSTM, input_dim=X_orig.shape[1])
    results['A_46d_mean'] = s

    # B. 46d + Attention Pooling
    s, _ = run_cv(X_orig, y, "B. 46d + Attention Pooling", device,
                  n_folds=args.folds, seed=args.seed,
                  epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                  model_cls=LSTMClassifierMeanPool, input_dim=X_orig.shape[1])
    results['B_46d_attn'] = s

    # C. 扩展 ~136d + Mean Pooling
    s, _ = run_cv(X_combined, y, f"C. {X_combined.shape[1]}d + Mean Pooling", device,
                  n_folds=args.folds, seed=args.seed,
                  epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                  model_cls=LSTMClassifierMeanPool, input_dim=X_combined.shape[1])
    results['C_ext_mean'] = s

    # D. 扩展 ~136d + Attention Pooling ⭐
    s, _ = run_cv(X_combined, y, f"D. {X_combined.shape[1]}d + Attention Pooling ⭐", device,
                  n_folds=args.folds, seed=args.seed,
                  epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                  model_cls=LSTMClassifierWithAttention, input_dim=X_combined.shape[1])
    results['D_ext_attn'] = s

    # 保存 JSON
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 输出对比表
    print("\n\n")
    print("=" * 100)
    print(" " * 30 + "方案 2 - 特征扩展 + Attention Pooling 对比表")
    print("=" * 100)
    print(f"{'策略':<32} {'Acc':>8} {'P':>8} {'R':>8} {'F1':>8} {'AUC':>8}  {'ΔF1':>8}")
    print("-" * 100)
    base_f1 = results['A_46d_mean']['f1_mean']
    base_auc = results['A_46d_mean']['auc_mean']
    label_map = {
        'A_46d_mean': 'A. 46d + Mean Pool',
        'B_46d_attn': 'B. 46d + Attn Pool',
        'C_ext_mean': f'C. {X_combined.shape[1]}d + Mean Pool',
        'D_ext_attn': f'D. {X_combined.shape[1]}d + Attn Pool ⭐',
    }
    best_name, best_f1 = 'A_46d_mean', base_f1
    for k, v in results.items():
        delta = v['f1_mean'] - base_f1
        sign = '+' if delta >= 0 else ''
        delta_auc = v['auc_mean'] - base_auc
        sign_auc = '+' if delta_auc >= 0 else ''
        print(f"{label_map[k]:<32} "
              f"{v['accuracy_mean']:.4f}  "
              f"{v['precision_mean']:.4f}  "
              f"{v['recall_mean']:.4f}  "
              f"{v['f1_mean']:.4f}  "
              f"{v['auc_mean']:.4f}  "
              f"{sign}{delta:.4f}")
        if v['f1_mean'] > best_f1:
            best_f1 = v['f1_mean']
            best_name = label_map[k]
    print("-" * 100)
    print(f"⭐ 最优策略: {best_name}")
    print(f"   F1 = {best_f1:.4f} (相对 baseline: {best_f1 - base_f1:+.4f})")
    print(f"")
    print(f"   方案 1 最佳 (SMOTE):    F1=0.7654 (vs 46d baseline)")
    print(f"   方案 2 最佳 (本次):     F1={best_f1:.4f} (vs 46d baseline)")
    print(f"   7 路 Late Fusion (参考): F1=0.9013")
    print("=" * 100)


if __name__ == '__main__':
    main()