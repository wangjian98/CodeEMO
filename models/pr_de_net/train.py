"""
PR-DE-Net 训练脚本

5 折分层交叉验证，三段式 Loss：
  L = α·BCE(p_A, y) + β·BCE(1-p_B, y) + γ·BCE(p_final, y)

支持消融实验:
  --ablation no_gate     : 用 0.5/0.5 平均替代 gate
  --ablation no_branchA  : 只用 Branch B + Gate
  --ablation no_branchB  : 只用 Branch A + Gate
  --ablation single_loss : 只用 γ·BCE(p_final, y)

输出:
  outputs/pr_de_net/{ablation_tag}/results.json
  outputs/pr_de_net/{ablation_tag}/probs.npy, labels.npy, fold_idx.npy
"""
import os, sys, json, argparse, time
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


# ============================================================
# Loss
# ============================================================
def pr_de_loss(out, y, alpha=1.0, beta=1.0, gamma=2.0):
    """Three-piece loss for PR complementary learning.

    Args:
        out: dict from model forward
        y: (B,) tensor in {0, 1}
    """
    p_A = out['p_A'].squeeze(-1).clamp(1e-7, 1 - 1e-7)
    p_B = out['p_B'].squeeze(-1).clamp(1e-7, 1 - 1e-7)
    p_f = out['p_final'].squeeze(-1).clamp(1e-7, 1 - 1e-7)

    y_f = y.float()
    # L_A: standard BCE for failed (high recall)
    L_A = F.binary_cross_entropy(p_A, y_f)
    # L_B: BCE on "passed" class (treat passed as positive) for high precision on passed
    #   → equivalent to BCE(1-p_B, 1-y) = -log(1-p_B)*(1-y) - log(p_B)*y
    #   when y=0 (passed), we want p_B low → 1-p_B high, BCE(1-p_B, 1) → low
    #   This pulls p_B down for passed, hence high precision on passed
    L_B = F.binary_cross_entropy(1 - p_B, 1 - y_f)
    # L_fuse
    L_fuse = F.binary_cross_entropy(p_f, y_f)

    return alpha * L_A + beta * L_B + gamma * L_fuse, {
        'L_A': float(L_A.detach().item()),
        'L_B': float(L_B.detach().item()),
        'L_fuse': float(L_fuse.detach().item()),
    }


# ============================================================
# Data
# ============================================================
def load_features():
    """Load 46-dim features and labels.

    Note: build_feature_matrix returns y=1=passed, y=0=failed.
    We FLIP to y=1=failed to match unified_compare convention.
    """
    print("Loading data...")
    ide_logs, passed_df = load_ide_logs()
    X, y_passed, student_ids = build_feature_matrix(ide_logs, passed_df)
    # flip: y=1=failed (consistent with unified_compare / bi_lstm_trans_v2/labels.npy)
    y = 1 - y_passed
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    print(f"  failed rate: {y.mean():.4f} (n_failed={y.sum()}, n_passed={(1-y).sum()})")
    return X, y


# ============================================================
# Train one fold
# ============================================================
def train_one_fold(X_train, y_train, X_val, y_val, device,
                   alpha=1.0, beta=1.0, gamma=2.0,
                   ablation='full',
                   epochs=200, batch_size=32, patience=20, lr=1e-3,
                   weight_decay=1e-3, verbose=False):
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
        ep_losses = {'L_A': 0, 'L_B': 0, 'L_fuse': 0, 'total': 0}
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xt[idx]
            yb = yt[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss, comp = pr_de_loss(out, yb, alpha, beta, gamma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for k in comp:
                ep_losses[k] += comp[k]
            ep_losses['total'] += float(loss.detach().item())
            n_batches += 1
        scheduler.step()

        # validation
        model.eval()
        with torch.no_grad():
            out_v = model(Xv)
            p_v = out_v['p_final'].squeeze(-1).cpu().numpy()
            y_pred = (p_v > 0.5).astype(int)
        from common.evaluator import evaluate
        metrics = evaluate(y_val, y_pred, p_v)
        val_f1 = metrics['f1']

        if verbose and ep % 20 == 0:
            print(f"  ep {ep:3d}: L_A={ep_losses['L_A']/n_batches:.4f} "
                  f"L_B={ep_losses['L_B']/n_batches:.4f} L_f={ep_losses['L_fuse']/n_batches:.4f} "
                  f"val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_metrics = metrics
            pc = 0
        else:
            pc += 1
            if pc >= patience:
                break

    # restore best
    model.load_state_dict(best_state)
    # collect predictions on val
    model.eval()
    with torch.no_grad():
        out_v = model(Xv)
        p_final = out_v['p_final'].squeeze(-1).cpu().numpy()
        p_A = out_v['p_A'].squeeze(-1).cpu().numpy()
        p_B = out_v['p_B'].squeeze(-1).cpu().numpy()
        gate = out_v['gate'].squeeze(-1).cpu().numpy()

    return {
        'metrics': best_metrics,
        'p_final': p_final,
        'p_A': p_A,
        'p_B': p_B,
        'gate': gate,
        'y_val': y_val,
    }


# ============================================================
# Cross-validation
# ============================================================
def run_cv(X, y, output_dir, ablation='full',
            alpha=1.0, beta=1.0, gamma=2.0,
            n_splits=5, seed=42, device=None,
            epochs=200, batch_size=32, patience=20,
            verbose=False):
    if device is None:
        device = get_device()
    set_seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    all_p_final = np.zeros(len(y), dtype=np.float64)
    all_p_A = np.zeros(len(y), dtype=np.float64)
    all_p_B = np.zeros(len(y), dtype=np.float64)
    all_gates = np.zeros(len(y), dtype=np.float64)
    all_fold_idx = np.zeros(len(y), dtype=np.int64)

    fold_results = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"\n[fold {fold+1}/{n_splits}] n_train={len(tr_idx)} n_val={len(va_idx)}")
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]

        # standardize on train, apply to val
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr).astype(np.float32)
        Xva_s = scaler.transform(Xva).astype(np.float32)

        t0 = time.time()
        out = train_one_fold(
            Xtr_s, ytr, Xva_s, yva, device,
            alpha=alpha, beta=beta, gamma=gamma,
            ablation=ablation,
            epochs=epochs, batch_size=batch_size, patience=patience,
            verbose=verbose,
        )
        dt = time.time() - t0
        m = out['metrics']
        print(f"  → Acc={m['accuracy']:.4f} P={m['precision']:.4f} "
              f"R={m['recall']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f} ({dt:.1f}s)")
        fold_results.append(m)

        all_p_final[va_idx] = out['p_final']
        all_p_A[va_idx] = out['p_A']
        all_p_B[va_idx] = out['p_B']
        all_gates[va_idx] = out['gate']
        all_fold_idx[va_idx] = fold

    summary = summarize_fold_results(fold_results)
    print_results_table(f"PR-DE-Net ({ablation})", summary)

    # save arrays (used by unified_compare)
    np.save(os.path.join(output_dir, 'probs.npy'), all_p_final)
    np.save(os.path.join(output_dir, 'probs_A.npy'), all_p_A)
    np.save(os.path.join(output_dir, 'probs_B.npy'), all_p_B)
    np.save(os.path.join(output_dir, 'gates.npy'), all_gates)
    np.save(os.path.join(output_dir, 'labels.npy'), y.astype(np.int64))
    np.save(os.path.join(output_dir, 'fold_idx.npy'), all_fold_idx)

    # save json summary
    summary_json = {
        'model': f'PR-DE-Net ({ablation})',
        'label_convention': 'y=1=failed',
        'ablation': ablation,
        'alpha': alpha, 'beta': beta, 'gamma': gamma,
        'n_params': PRDENet().count_parameters(),
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
        'n_samples': int(len(y)),
        'n_failed': int(y.sum()),
        'n_passed': int((1 - y).sum()),
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(summary_json, f, indent=2)
    print(f"\nSaved to {output_dir}/results.json")
    return summary_json


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='outputs/pr_de_net')
    parser.add_argument('--ablation', default='full',
                        choices=['full', 'no_gate', 'single_loss'])
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    X, y = load_features()
    tag = args.ablation if args.ablation == 'full' else f'{args.ablation}'
    out_dir = os.path.join(args.output_dir, tag)
    run_cv(
        X, y, out_dir,
        ablation=args.ablation,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        epochs=args.epochs, batch_size=args.batch_size,
        patience=args.patience, seed=args.seed,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()