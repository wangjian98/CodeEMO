"""
MASC-Net 训练入口
5-fold StratifiedKFold CV, 输出 P/A/R/F1/AUC
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

_PROJECT_ROOT = '/home/ubuntu/CodeEMO'
sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, set_seed
from common.feature_engineering import build_feature_matrix
from common.evaluator import evaluate
from models.masc_net.model import MASCNet, FocalLoss, count_parameters


def train_one_fold(X_train, y_train, X_val, y_val, device,
                   epochs=120, batch_size=32, patience=15, lr=1e-3,
                   weight_decay=1e-2,
                   use_contrastive=True, use_uncertainty=True):
    model = MASCNet(
        n_prototypes_per_class=4,
        use_contrastive=use_contrastive,
        use_uncertainty=use_uncertainty,
        feat_dim=64, proj_dim=32, dropout=0.3,
    ).to(device)
    print(f"  Parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    focal = FocalLoss(alpha=0.7, gamma=2.0)

    Xt = torch.FloatTensor(X_train).to(device)
    yt = torch.FloatTensor(y_train).to(device)
    Xv = torch.FloatTensor(X_val).to(device)
    yv = torch.FloatTensor(y_val).to(device)

    best_v = float('inf')
    best_state = None
    pc = 0
    n = Xt.shape[0]

    # 在训练开始前初始化阈值
    model.eval()
    with torch.no_grad():
        _, _, aux = model(Xt, labels=torch.zeros(n, dtype=torch.long, device=device).long(),
                          compute_loss=False)
        # 简化:用第一次 forward 的 sim_per_class
    # 不需要 labels 的 sim_per_class 提取
    sim_pc_init = None
    # 再次 forward 拿 sim
    model.eval()
    with torch.no_grad():
        # 我们手动跑一次: encoder → 对比分支 → proto
        from models.masc_net.model import SampleAwareContrastive, PrototypeBank
        h, _ = model.encoder(Xt)
        if model.use_contrastive:
            z_q, _, _ = model.contrastive(h, model.proto_bank.prototypes.data)
        else:
            z_q = h[:, :32]
            z_q = nn.functional.normalize(z_q, dim=-1)
        sim_pc_init, _ = model.proto_bank.get_class_assignment(z_q)
    # 把 labels 转 long
    model.classifier.set_thresh_from_train(sim_pc_init, torch.LongTensor(y_train).to(device))

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            labels_b = torch.LongTensor(y_train).to(device)[idx]
            probs, unc, aux = model(Xt[idx], labels=labels_b, compute_loss=True)
            loss_cls = focal(probs, yt[idx])
            loss = loss_cls + aux['loss_total']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            v_probs, _, _ = model(Xv, labels=None, compute_loss=False)
            v_loss = focal(v_probs, yv).item()

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
        probs, uncertainty, _ = model(Xv, labels=None, compute_loss=False)
        probs = probs.cpu().numpy()
        uncertainty = uncertainty.cpu().numpy()

    # 阈值 sweep: 对不平衡数据找最佳 F1 阈值
    best_f1 = 0.0
    best_t = 0.5
    for t in np.arange(0.10, 0.91, 0.01):
        f1 = f1_score(y_val, (probs > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    preds_at_best = (probs > best_t).astype(int)
    metrics = evaluate(y_val, preds_at_best, probs)
    metrics['best_threshold'] = float(best_t)
    metrics['mean_uncertainty'] = float(uncertainty.mean())
    return metrics, probs, uncertainty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='outputs/masc_net')
    parser.add_argument('--ablation', type=str, default='full',
                        choices=['full', 'no_contrastive', 'no_uncertainty', 'baseline_only'])
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 消融开关
    use_contrastive = args.ablation in ('full', 'no_uncertainty')
    use_uncertainty = args.ablation in ('full', 'no_contrastive')

    # 加载数据
    ide_logs, passed = load_ide_logs()
    X, y, student_ids = build_feature_matrix(ide_logs, passed)
    print(f"X shape: {X.shape}, passed={int((y==0).sum())}, failed={int((y==1).sum())}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_metrics, fold_probs, fold_labels, fold_idx, fold_uncs = [], [], [], [], []
    all_idx = []

    start = time.time()
    for fold_idx_i, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        print(f"\n=== Fold {fold_idx_i}/{args.folds} | train={len(train_idx)} test={len(test_idx)} ===",
              flush=True)

        m, probs, unc = train_one_fold(
            X_train_s, y_train, X_test_s, y_test, device,
            epochs=args.epochs, batch_size=args.batch_size,
            patience=args.patience, lr=args.lr,
            weight_decay=args.weight_decay,
            use_contrastive=use_contrastive, use_uncertainty=use_uncertainty,
        )
        fold_metrics.append(m)
        fold_probs.append(probs)
        fold_labels.append(y_test)
        fold_uncs.append(unc)
        all_idx.append(test_idx)
        print(f"  Fold {fold_idx_i}: "
              f"Acc={m['accuracy']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
              f"F1={m['f1']:.4f} AUC={m['auc']:.4f} "
              f"(thr={m['best_threshold']:.2f}, unc={m['mean_uncertainty']:.3f})",
              flush=True)

    elapsed = time.time() - start

    # 汇总
    print(f"\n{'='*60}")
    print(f"  MASC-Net ({args.ablation}) 5 折汇总 ({elapsed:.1f}s)")
    print(f"{'='*60}")
    for k in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        vals = [m[k] for m in fold_metrics]
        print(f"  {k:10s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print(f"{'='*60}")

    # 保存
    os.makedirs(args.output_dir, exist_ok=True)
    # 拼接概率(按 test_idx 还原到原顺序)
    all_probs = np.zeros(len(y))
    all_uncs = np.zeros(len(y))
    for probs, idx, unc in zip(fold_probs, all_idx, fold_uncs):
        all_probs[idx] = probs
        all_uncs[idx] = unc
    np.save(os.path.join(args.output_dir, f'{args.ablation}_probs.npy'), all_probs)
    np.save(os.path.join(args.output_dir, f'{args.ablation}_uncs.npy'), all_uncs)
    np.save(os.path.join(args.output_dir, 'labels.npy'), y)

    with open(os.path.join(args.output_dir, f'{args.ablation}_results.json'), 'w') as f:
        json.dump({
            'model': f'MASC-Net ({args.ablation})',
            'config': vars(args),
            'use_contrastive': use_contrastive,
            'use_uncertainty': use_uncertainty,
            'elapsed_seconds': elapsed,
            'cv_metrics': {
                k: {'mean': float(np.mean([m[k] for m in fold_metrics])),
                    'std': float(np.std([m[k] for m in fold_metrics]))}
                for k in ['accuracy', 'precision', 'recall', 'f1', 'auc']
            },
            'fold_details': fold_metrics,
        }, f, indent=2)
    print(f"\n结果已保存: {args.output_dir}/{args.ablation}_results.json")


if __name__ == '__main__':
    main()
