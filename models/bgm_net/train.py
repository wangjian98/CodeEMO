"""
BGM-Net 训练脚本

支持消融实验:
    python models/bgm_net/train.py                          # 完整模型
    python models/bgm_net/train.py --no-gate                # 去掉Behavior Gate (H1)
    python models/bgm_net/train.py --no-entropy-attn        # 去掉熵注意力 (H2)
    python models/bgm_net/train.py --no-cross               # 去掉比率交叉项 (H3)
    python models/bgm_net/train.py --no-all                 # 全部去掉 (退化baseline)

    python models/bgm_net/train.py --all-variants           # 跑全部5个变体

输出:
    results/bgm_net/{variant}/results.json
    results/bgm_net/{variant}/gate_analysis.json (完整模型)
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, set_seed
from common.feature_engineering import build_feature_matrix
from common.evaluator import evaluate, summarize_fold_results, print_results_table
from models.bgm_net.model import BGMNet, count_parameters


def train_one_fold(X_train, y_train, X_val, y_val, device,
                   use_gate=True, use_entropy_attention=True,
                   use_cross_interaction=True,
                   epochs=150, batch_size=32, patience=15, lr=1e-3,
                   weight_decay=1e-2):
    """训练单折"""
    model = BGMNet(
        use_gate=use_gate,
        use_entropy_attention=use_entropy_attention,
        use_cross_interaction=use_cross_interaction,
        dropout=0.3,
    ).to(device)

    print(f"  Parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.BCELoss()

    # 类别平衡: pos_weight
    n_pos = float((y_train == 1).sum())
    n_neg = float((y_train == 0).sum())
    pos_weight = n_neg / (n_pos + 1e-8)

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
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()

            preds = model(Xt[idx])
            # 加权BCE
            weight = torch.where(yt[idx] == 1,
                                 torch.full_like(yt[idx], pos_weight),
                                 torch.ones_like(yt[idx]))
            loss = (criterion(preds, yt[idx]) * weight).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

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

    preds_05 = (probs > 0.5).astype(int)

    # 阈值sweep找最佳F1
    best_f1 = 0.0
    best_threshold = 0.5
    best_preds = preds_05
    for t in np.arange(0.05, 0.96, 0.01):
        p = (probs > t).astype(int)
        f1 = f1_score(y_val, p, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_preds = p

    return best_preds, probs, best_threshold, model


def run_variant(variant_name, use_gate, use_entropy_attention, use_cross_interaction,
                X, y, folds=5, seed=42, output_dir='results/bgm_net'):
    """运行一个变体的完整5折CV"""
    from copy import deepcopy

    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  BGM-Net Variant: {variant_name}")
    print(f"  gate={use_gate}, entropy_attn={use_entropy_attention}, cross={use_cross_interaction}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    config = {
        'use_gate': use_gate,
        'use_entropy_attention': use_entropy_attention,
        'use_cross_interaction': use_cross_interaction,
    }

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_metrics_05 = []
    fold_metrics_best = []
    all_fold_probs = []
    all_fold_gate_values = []
    all_fold_attn_weights = []
    all_fold_indices = []
    best_thresholds = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        print(f"\n--- Fold {fold_idx}/{folds} | train={len(train_idx)} test={len(test_idx)} ---")

        preds, probs, best_thr, trained_model = train_one_fold(
            X_train_s, y_train, X_test_s, y_test, device,
            use_gate=use_gate,
            use_entropy_attention=use_entropy_attention,
            use_cross_interaction=use_cross_interaction,
        )

        # F1@0.5
        preds_05 = (probs > 0.5).astype(int)
        m_05 = evaluate(y_test, preds_05, probs)

        # F1@best
        m_best = evaluate(y_test, preds, probs)

        fold_metrics_05.append(m_05)
        fold_metrics_best.append(m_best)
        all_fold_probs.append(probs)
        all_fold_indices.append(test_idx)
        best_thresholds.append(best_thr)

        print(f"  Fold {fold_idx}: F1@0.5={m_05['f1']:.4f} F1@best={m_best['f1']:.4f} "
              f"(thr={best_thr:.2f}) AUC={m_best['auc']:.4f}")

        # 保存最后一折的gate和attention用于分析
        if fold_idx == folds:
            with torch.no_grad():
                Xv_tensor = torch.FloatTensor(X_test_s).to(device)
                _ = trained_model(Xv_tensor)
                if hasattr(trained_model, 'get_gate_values') and trained_model.get_gate_values() is not None:
                    all_fold_gate_values.append(
                        trained_model.get_gate_values().cpu().numpy())
                if hasattr(trained_model, 'get_attention_weights') and trained_model.get_attention_weights() is not None:
                    all_fold_attn_weights.append(
                        trained_model.get_attention_weights().cpu().numpy())

    # 汇总
    summary_05 = summarize_fold_results(fold_metrics_05)
    summary_best = summarize_fold_results(fold_metrics_best)
    avg_thr = float(np.mean(best_thresholds))

    print_results_table(f"BGM-Net [{variant_name}] F1@0.5", summary_05)
    print_results_table(f"BGM-Net [{variant_name}] F1@best", summary_best)
    print(f"  Avg best threshold: {avg_thr:.2f}")

    # 保存结果
    out_dir = os.path.join(output_dir, variant_name)
    os.makedirs(out_dir, exist_ok=True)

    result = {
        'variant': variant_name,
        'config': config,
        'f1_at_05': summary_05,
        'f1_at_best': summary_best,
        'avg_best_threshold': avg_thr,
        'best_thresholds_per_fold': best_thresholds,
    }

    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(result, f, indent=2)

    # 保存gate和attention分析数据
    if all_fold_gate_values:
        np.save(os.path.join(out_dir, 'gate_values.npy'), all_fold_gate_values[0])
    if all_fold_attn_weights:
        np.save(os.path.join(out_dir, 'attention_weights.npy'), all_fold_attn_weights[0])

    return result


def main():
    parser = argparse.ArgumentParser(description='BGM-Net 训练 + 消融实验')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='results/bgm_net')
    parser.add_argument('--no-gate', action='store_true', help='去掉Behavior Gate (H1)')
    parser.add_argument('--no-entropy-attn', action='store_true', help='去掉熵注意力 (H2)')
    parser.add_argument('--no-cross', action='store_true', help='去掉比率交叉项 (H3)')
    parser.add_argument('--no-all', action='store_true', help='全部去掉 (退化baseline)')
    parser.add_argument('--all-variants', action='store_true', help='跑全部5个变体')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载数据
    ide_logs, passed = load_ide_logs()
    X, y, student_ids = build_feature_matrix(ide_logs, passed)
    print(f"X shape: {X.shape}, passed={int((y==1).sum())}, failed={int((y==0).sum())}")

    if args.all_variants:
        # 跑全部5个变体
        variants = [
            ('full',         True,  True,  True),
            ('no_gate',      False, True,  True),
            ('no_entropy',   True,  False, True),
            ('no_cross',     True,  True,  False),
            ('baseline',     False, False, False),
        ]
    else:
        # 单次运行
        use_gate = not args.no_gate and not args.no_all
        use_entropy = not args.no_entropy_attn and not args.no_all
        use_cross = not args.no_cross and not args.no_all
        variant_name = 'custom'
        if args.no_all:
            variant_name = 'baseline'
        elif args.no_gate:
            variant_name = 'no_gate'
        elif args.no_entropy_attn:
            variant_name = 'no_entropy'
        elif args.no_cross:
            variant_name = 'no_cross'
        elif not (args.no_gate or args.no_entropy_attn or args.no_cross):
            variant_name = 'full'
        variants = [(variant_name, use_gate, use_entropy, use_cross)]

    all_results = {}
    for name, gate, entropy, cross in variants:
        result = run_variant(
            name, gate, entropy, cross,
            X, y, folds=args.folds, seed=args.seed,
            output_dir=args.output_dir,
        )
        all_results[name] = result

    # 打印汇总对比表
    print(f"\n\n{'='*80}")
    print(f"  BGM-Net 消融实验汇总 ({len(variants)} 变体)")
    print(f"{'='*80}")
    print(f"  {'Variant':<15} {'F1@0.5':>10} {'F1@best':>10} {'AUC':>10} {'Prec':>10} {'Recall':>10}")
    print(f"  {'-'*65}")
    for name, _, _, _ in variants:
        r = all_results[name]
        s5 = r['f1_at_05']
        sb = r['f1_at_best']
        print(f"  {name:<15} {s5['f1_mean']:.4f}±{s5['f1_std']:.3f} "
              f"{sb['f1_mean']:.4f}±{sb['f1_std']:.3f} "
              f"{sb['auc_mean']:.4f}±{sb['auc_std']:.3f} "
              f"{sb['precision_mean']:.4f} "
              f"{sb['recall_mean']:.4f}")
    print(f"{'='*80}")

    # 保存汇总
    summary_path = os.path.join(args.output_dir, 'ablation_summary.json')
    os.makedirs(args.output_dir, exist_ok=True)
    compact = {}
    for name in all_results:
        s5 = all_results[name]['f1_at_05']
        sb = all_results[name]['f1_at_best']
        compact[name] = {
            'f1_at_05': {k: v for k, v in s5.items() if k != 'folds'},
            'f1_at_best': {k: v for k, v in sb.items() if k != 'folds'},
            'avg_best_threshold': all_results[name]['avg_best_threshold'],
        }
    with open(summary_path, 'w') as f:
        json.dump(compact, f, indent=2)
    print(f"\n汇总已保存到: {summary_path}")


if __name__ == '__main__':
    main()
