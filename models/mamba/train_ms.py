"""
多尺度 Mamba 训练脚本 - 支持 --max-seq-len 参数 + 输出 probs/labels/fold_idx

用法:
  python models/mamba/train_ms.py --max-seq-len 50   --output-dir outputs/mamba_sht
  python models/mamba/train_ms.py --max-seq-len 500  --output-dir outputs/mamba_mid
  python models/mamba/train_ms.py --max-seq-len 2000 --output-dir outputs/mamba_long

与 train_gpu.py 的区别:
  1. --max-seq-len 参数化 max_events 和 FullMambaStudent.max_seq_len
  2. step5 微调时, 收集每个 fold 的 probs/y_true/fold_idx
  3. 最终把全部样本的 probs/labels/fold_idx 保存为 .npy, 供 late fusion 使用
  4. 短序列时减少 pretrain/fintune epochs (节省时间)
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, set_seed
from common.evaluator import summarize_fold_results, print_results_table
from models.mamba.model import FullMambaStudent
from models.mamba.steps.step1_preprocessing import preprocess
from models.mamba.steps.step2_pretrain import pretrain
from models.mamba.steps.step3_multiscale import extract_representations
from models.mamba.steps.step4_prototype import run_kmeans
from models.mamba.steps.step5_finetune import (
    focal_loss, _freeze_backbone, collate_for_finetune
)
from models.mamba.steps.step6_interpret import run_interpretability

from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold


def finetune_cv_with_probs(pretrained_state, dataset, student_ids, y, device,
                           n_folds=5, n_epochs=8, batch_size=8, max_seq_len=2000,
                           d_model=64, n_layers=6, d_state=16):
    """与 step5.finetune_cv 类似, 但额外输出每个样本的 (probs, fold_idx)"""
    print(f"\n[Step 5 MS] 预测微调 - {n_folds}折交叉验证 (max_seq={max_seq_len})")
    print(f"  样本数: {len(dataset)}, 标签: pass={np.sum(y==0)}, fail={np.sum(y==1)}")
    print(f"  训练轮数/折: {n_epochs}, 批大小: {batch_size}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # 全量收集: 每个样本的 P(fail) 和 fold_idx
    all_probs = np.zeros(len(dataset), dtype=np.float32)
    all_fold_idx = np.full(len(dataset), -1, dtype=np.int32)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(student_ids, y)):
        print(f"\n  --- Fold {fold_idx+1}/{n_folds} ---  train={len(train_idx)} test={len(test_idx)}")

        model = FullMambaStudent(
            n_event_types=7, d_model=d_model, n_layers=n_layers, d_state=d_state,
            n_prototypes=4, max_seq_len=max_seq_len
        )
        model.to(device)
        model.load_state_dict(pretrained_state)
        _freeze_backbone(model)

        train_samples = [dataset[i] for i in train_idx]
        test_samples = [dataset[i] for i in test_idx]

        train_loader = DataLoader(
            train_samples, batch_size=batch_size, shuffle=True,
            collate_fn=collate_for_finetune, drop_last=True  # 防止 BN 在最后 1 样本 batch 报错
        )
        test_loader = DataLoader(
            test_samples, batch_size=batch_size, shuffle=False,
            collate_fn=collate_for_finetune, drop_last=False
        )

        # 分层学习率
        head_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and ('risk_head' in n or 'se_gate' in n)]
        backbone_params = [p for n, p in model.named_parameters()
                           if p.requires_grad
                           and not any(k in n for k in ['risk_head', 'se_gate'])]
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': 5e-5},
            {'params': head_params,     'lr': 5e-4},
        ], weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs * len(train_loader), eta_min=1e-6
        )

        y_train = np.array([s['risk'] for s in train_samples])
        n_pass = max(int((y_train == 0).sum()), 1)
        n_fail = max(int((y_train == 1).sum()), 1)
        alpha = torch.tensor(
            [len(y_train) / (2 * n_pass), len(y_train) / (2 * n_fail)],
            dtype=torch.float32, device=device
        ).clamp(max=2.0)

        # 训练
        model.train()
        for epoch in range(n_epochs):
            total_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                batch_input = {
                    'event_types': batch['event_types'].to(device),
                    'time_intervals': batch['time_intervals'].to(device),
                    'deadline_dists': batch['deadline_dists'].to(device),
                    'part_ids': batch['part_ids'].to(device),
                }
                targets = batch['risk'].to(device)

                optimizer.zero_grad()
                outputs = model(batch_input)
                logits = outputs['risk']

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    optimizer.zero_grad()
                    continue
                loss = focal_loss(logits, targets, alpha=alpha,
                                  gamma=2.0, label_smoothing=0.05)
                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad()
                    continue
                loss.backward()

                bad_grad = any(
                    (p.grad is not None) and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                    for p in model.parameters() if p.requires_grad
                )
                if bad_grad:
                    optimizer.zero_grad()
                    continue

                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=0.5
                )
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                n_batches += 1
            print(f"    Epoch {epoch+1}/{n_epochs} - Loss: {total_loss/max(n_batches,1):.4f}")

        # 评估 + 收集 probs
        model.eval()
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                batch_input = {
                    'event_types': batch['event_types'].to(device),
                    'time_intervals': batch['time_intervals'].to(device),
                    'deadline_dists': batch['deadline_dists'].to(device),
                    'part_ids': batch['part_ids'].to(device),
                }
                targets = batch['risk'].to(device)
                outputs = model(batch_input)
                logits = outputs['risk']
                probs = torch.softmax(logits, dim=-1)

                y_true.extend(targets.cpu().numpy())
                y_pred.extend(logits.argmax(dim=-1).cpu().numpy())
                y_prob.extend(probs[:, 1].cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        # 用通用 evaluator
        from common.evaluator import evaluate
        metrics = evaluate(y_true, y_pred, y_prob)
        fold_results.append(metrics)

        # 收集 probs 到全量数组
        all_probs[test_idx] = y_prob
        all_fold_idx[test_idx] = fold_idx

        print(f"    Fold {fold_idx+1}: Acc={metrics['accuracy']:.4f} "
              f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
              f"F1={metrics['f1']:.4f} AUC={metrics['auc']:.4f}")

    summary = summarize_fold_results(fold_results)
    print_results_table(f"Mamba-SSM MS (max_seq={max_seq_len})", summary)

    return all_probs, all_fold_idx, fold_results, summary


def main():
    parser = argparse.ArgumentParser(description='多尺度 Mamba 训练 (输出 probs 供 late fusion)')
    parser.add_argument('--max-seq-len', type=int, required=True,
                        help='最大事件序列长度 (50 / 500 / 2000)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='输出目录 (例如 outputs/mamba_sht)')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrain-epochs', type=int, default=3)
    parser.add_argument('--finetune-epochs', type=int, default=8)
    parser.add_argument('--pretrain-batch-size', type=int, default=16)
    parser.add_argument('--finetune-batch-size', type=int, default=16)
    parser.add_argument('--d-model', type=int, default=64, help='模型维度 (默认 64, OOM 时降到 48)')
    parser.add_argument('--n-layers', type=int, default=6, help='Mamba 层数 (默认 6, OOM 时降到 4)')
    parser.add_argument('--d-state', type=int, default=16, help='SSM 状态维度 (默认 16, OOM 时降到 12)')
    args = parser.parse_args()

    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
    else:
        device = torch.device('cpu')
        gpu_name = None

    print("=" * 60)
    print(f"  多尺度 Mamba 训练 - max_seq_len={args.max_seq_len}")
    print(f"  模型: FullMambaStudent (d_model=64, n_layers=6)")
    print(f"  设备: {device}" + (f" ({gpu_name})" if gpu_name else ""))
    print(f"  输出: {args.output_dir}")
    print("=" * 60)

    output_dir = os.path.join(_PROJECT_ROOT, args.output_dir) \
        if not os.path.isabs(args.output_dir) else args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: 预处理 (max_events = max_seq_len)
    ide_logs_df, passed_df = load_ide_logs()
    samples, student_ids, labels = preprocess(
        ide_logs_df, passed_df, max_events=args.max_seq_len
    )

    # Step 2: 预训练
    model = FullMambaStudent(
        n_event_types=7, d_model=args.d_model, n_layers=args.n_layers, d_state=args.d_state,
        n_prototypes=4, max_seq_len=args.max_seq_len
    )
    model.to(device)

    pretrained_state = pretrain(
        model, samples, device,
        epochs=args.pretrain_epochs, batch_size=args.pretrain_batch_size
    )

    # Step 3+4: 特征提取 + KMeans (不影响结果, 可跳过以省时间, 但保留以贴近原流程)
    model.load_state_dict(pretrained_state)
    try:
        representations, repr_labels = extract_representations(
            model, samples, device, batch_size=args.pretrain_batch_size
        )
        kmeans, cluster_assignments = run_kmeans(
            representations, repr_labels, n_clusters=4
        )
    except Exception as e:
        print(f"  [warn] step3/4 跳过: {e}")

    # Step 5: 微调 + 收集 probs
    all_probs, all_fold_idx, fold_results, summary = finetune_cv_with_probs(
        pretrained_state, samples, student_ids, labels, device,
        n_folds=args.folds, n_epochs=args.finetune_epochs,
        batch_size=args.finetune_batch_size,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model, n_layers=args.n_layers, d_state=args.d_state,
    )

    # 保存 npy 给 late fusion
    np.save(os.path.join(output_dir, 'probs.npy'), all_probs)
    np.save(os.path.join(output_dir, 'labels.npy'), labels.astype(np.int32))
    np.save(os.path.join(output_dir, 'fold_idx.npy'), all_fold_idx)
    print(f"\n  保存 npy: probs.npy / labels.npy / fold_idx.npy ({len(all_probs)} 样本)")

    # 保存 results.json (跟原版兼容)
    results = {
        'model': 'FullMambaStudent',
        'config': {
            'd_model': 64,
            'n_layers': 6,
            'd_state': 16,
            'n_prototypes': 4,
            'max_seq_len': args.max_seq_len,
            'device': str(device),
            'gpu_name': gpu_name,
        },
        'pipeline': {
            'pretrain_epochs': args.pretrain_epochs,
            'finetune_epochs': args.finetune_epochs,
            'n_folds': args.folds,
            'pretrain_batch_size': args.pretrain_batch_size,
            'finetune_batch_size': args.finetune_batch_size,
        },
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
        'n_samples': len(samples),
        'n_passed': int((labels == 0).sum()),
        'n_failed': int((labels == 1).sum()),
        'fold_details': fold_results,
    }

    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  保存 results.json: {results_path}")

    # 最终总结
    print("\n" + "=" * 60)
    print(f"  Mamba 多尺度 (max_seq={args.max_seq_len}) 训练完成")
    print(f"  Accuracy: {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")
    print(f"  Precision: {summary['precision_mean']:.4f} +/- {summary['precision_std']:.4f}")
    print(f"  Recall: {summary['recall_mean']:.4f} +/- {summary['recall_std']:.4f}")
    print(f"  F1 Score: {summary['f1_mean']:.4f} +/- {summary['f1_std']:.4f}")
    print(f"  AUC:      {summary['auc_mean']:.4f} +/- {summary['auc_std']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()