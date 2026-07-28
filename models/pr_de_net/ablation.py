"""
PR-DE-Net 消融脚本
  - no_gate:     不用 gate, p_final = 0.5*p_A + 0.5*p_B
  - single_loss: 只用 γ·BCE(p_final, y)，证明三段式 loss 的价值

跑完会输出统一对比表，方便和 full 比较。
"""
import os, sys, json, time
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
from models.pr_de_net.model import PRDENet


def load_features():
    ide_logs, passed_df = load_ide_logs()
    X, y_passed, student_ids = build_feature_matrix(ide_logs, passed_df)
    y = 1 - y_passed
    return X, y


def train_one_fold_ablation(X_train, y_train, X_val, y_val, device,
                            ablation='no_gate',
                            alpha=1.0, beta=1.0, gamma=2.0,
                            epochs=150, batch_size=32, patience=15, lr=1e-3,
                            weight_decay=1e-3):
    model = PRDENet(dropout=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    Xt = torch.FloatTensor(X_train).to(device)
    yt = torch.FloatTensor(y_train).to(device)
    Xv = torch.FloatTensor(X_val).to(device)
    yv = torch.FloatTensor(y_val).to(device)

    best_val_f1 = -1.0
    best_state = None
    best_metrics = None
    pc = 0
    n = Xt.shape[0]

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            out = model(Xt[idx])

            p_A = out['p_A'].squeeze(-1).clamp(1e-7, 1 - 1e-7)
            p_B = out['p_B'].squeeze(-1).clamp(1e-7, 1 - 1e-7)
            p_f = out['p_final'].squeeze(-1).clamp(1e-7, 1 - 1e-7)
            y_f = yt[idx]

            # rebuild p_final according to ablation
            if ablation == 'no_gate':
                p_f = 0.5 * p_A + 0.5 * p_B

            if ablation == 'single_loss':
                # only fused loss, alpha=beta=0
                loss = F.binary_cross_entropy(p_f, y_f)
            else:
                L_A = F.binary_cross_entropy(p_A, y_f)
                L_B = F.binary_cross_entropy(1 - p_B, 1 - y_f)
                L_fuse = F.binary_cross_entropy(p_f, y_f)
                loss = alpha * L_A + beta * L_B + gamma * L_fuse

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # validation: re-compute p_final according to ablation
        model.eval()
        with torch.no_grad():
            out_v = model(Xv)
            p_A_v = out_v['p_A'].squeeze(-1)
            p_B_v = out_v['p_B'].squeeze(-1)
            if ablation == 'no_gate':
                p_v = (0.5 * p_A_v + 0.5 * p_B_v).cpu().numpy()
            else:
                p_v = out_v['p_final'].squeeze(-1).cpu().numpy()
        y_pred = (p_v > 0.5).astype(int)
        metrics = evaluate(y_val, y_pred, p_v)

        if metrics['f1'] > best_val_f1:
            best_val_f1 = metrics['f1']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_metrics = metrics
            pc = 0
        else:
            pc += 1
            if pc >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_v = model(Xv)
        p_A_v = out_v['p_A'].squeeze(-1).cpu().numpy()
        p_B_v = out_v['p_B'].squeeze(-1).cpu().numpy()
        p_f_v = out_v['p_final'].squeeze(-1).cpu().numpy()
        if ablation == 'no_gate':
            p_v = 0.5 * p_A_v + 0.5 * p_B_v
        else:
            p_v = p_f_v
    return {'metrics': best_metrics, 'p_final': p_v, 'p_A': p_A_v, 'p_B': p_B_v}


def run_cv(X, y, output_dir, ablation, **kwargs):
    if os.path.exists(os.path.join(output_dir, 'results.json')):
        print(f"  [{ablation}] cache hit, skip")
        return json.load(open(os.path.join(output_dir, 'results.json')))

    device = get_device()
    set_seed(42)
    os.makedirs(output_dir, exist_ok=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_p_final = np.zeros(len(y))
    all_p_A = np.zeros(len(y))
    all_p_B = np.zeros(len(y))
    all_fold_idx = np.zeros(len(y), dtype=np.int64)

    fold_results = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr).astype(np.float32)
        Xva_s = scaler.transform(Xva).astype(np.float32)

        out = train_one_fold_ablation(
            Xtr_s, ytr, Xva_s, yva, device, ablation=ablation, **kwargs
        )
        m = out['metrics']
        print(f"  [{ablation} fold {fold+1}] F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} AUC={m['auc']:.4f}")
        fold_results.append(m)
        all_p_final[va_idx] = out['p_final']
        all_p_A[va_idx] = out['p_A']
        all_p_B[va_idx] = out['p_B']
        all_fold_idx[va_idx] = fold

    summary = summarize_fold_results(fold_results)
    print_results_table(f"PR-DE-Net ({ablation})", summary)

    np.save(os.path.join(output_dir, 'probs.npy'), all_p_final)
    np.save(os.path.join(output_dir, 'probs_A.npy'), all_p_A)
    np.save(os.path.join(output_dir, 'probs_B.npy'), all_p_B)
    np.save(os.path.join(output_dir, 'labels.npy'), y.astype(np.int64))
    np.save(os.path.join(output_dir, 'fold_idx.npy'), all_fold_idx)

    summary_json = {
        'model': f'PR-DE-Net ({ablation})',
        'ablation': ablation,
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
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(summary_json, f, indent=2)
    return summary_json


if __name__ == '__main__':
    X, y = load_features()
    for ablation in ['no_gate', 'single_loss']:
        out_dir = f'outputs/pr_de_net/{ablation}'
        print(f"\n=== ablation: {ablation} ===")
        run_cv(X, y, out_dir, ablation)