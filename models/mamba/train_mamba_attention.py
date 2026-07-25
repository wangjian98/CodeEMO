"""
Mamba-Attention 融合模型 - GPU 训练脚本

流程:
  Step 1: 7 维事件编码 (复用 step1_preprocessing.preprocess)
  Step 2: 自监督预训练 - 下一事件预测 (复用 step2_pretrain.pretrain)
  Step 3: 监督微调 - 5 折交叉验证, 计算 P/R/A/F1/AUC

输出:
  outputs/mamba_attention/results.json
  outputs/mamba_attention/comparison.csv (5 折详细结果)

用法:
    python models/mamba/train_mamba_attention.py
    python models/mamba/train_mamba_attention.py --folds 5 --finetune-epochs 12 --batch-size 32
"""

import os
import sys
import json
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

# sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, set_seed
from common.evaluator import summarize_fold_results, print_results_table
from models.mamba.mamba_attention import create_model
from models.mamba.steps.step1_preprocessing import preprocess
from models.mamba.steps.step2_pretrain import pretrain as pretrain_nextevent
from models.mamba.model import EVENT_TYPES


# =====================================================================
# Collate: 7 维事件序列 → batch 张量
# =====================================================================
def collate_finetune(samples, max_events=2000):
    L = min(max(s['n_events'] for s in samples), max_events)

    et_list, ti_list, dd_list, pi_list, risk_list = [], [], [], [], []
    for s in samples:
        n = s['n_events']
        et = s['event_types'][:n][:L]
        ti = s['time_intervals'][:n][:L]
        dd = s['deadline_dists'][:n][:L]
        pi = s['part_ids'][:n][:L]

        if et.shape[0] < L:
            pad = L - et.shape[0]
            et = F.pad(et, (0, pad), value=0)
            ti = F.pad(ti, (0, pad), value=0.0)
            dd = F.pad(dd, (0, pad), value=0.0)
            pi = F.pad(pi, (0, pad), value=0)

        et_list.append(et)
        ti_list.append(ti)
        dd_list.append(dd)
        pi_list.append(pi)
        risk_list.append(s['risk'])

    return {
        'event_types':    torch.stack(et_list),
        'time_intervals': torch.stack(ti_list),
        'deadline_dists': torch.stack(dd_list),
        'part_ids':       torch.stack(pi_list),
        'risk':           torch.LongTensor(risk_list),
    }


# =====================================================================
# Focal Loss (沿用 step5 设计)
# =====================================================================
def focal_loss(logits, targets, alpha=None, gamma=2.0, label_smoothing=0.05):
    C = logits.shape[-1]
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(label_smoothing / C)
        true_dist.scatter_(1, targets.unsqueeze(1),
                           1.0 - label_smoothing + label_smoothing / C)
    oh = F.one_hot(targets, C).float()
    p_t = (probs * oh).sum(dim=-1)
    fw = (1.0 - p_t) ** gamma
    a_t = alpha[targets] if alpha is not None else 1.0
    loss = -(a_t * fw * (log_probs * true_dist).sum(dim=-1))
    return loss.mean()


# =====================================================================
# 解冻策略
# =====================================================================
def freeze_for_finetune(model):
    """全参数解冻 (优化A): 端到端训练"""
    for p in model.parameters():
        p.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数: {n_train:,} / {n_total:,} (100.00% 全参数解冻)")



# =====================================================================
# 主流程
# =====================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--pretrain-epochs', type=int, default=3)
    p.add_argument('--finetune-epochs', type=int, default=15)
    p.add_argument('--pretrain-batch-size', type=int, default=16)
    p.add_argument('--finetune-batch-size', type=int, default=16)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--n-mamba-layers', type=int, default=3)
    p.add_argument('--n-attn-layers', type=int, default=2)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--d-state', type=int, default=16)
    p.add_argument('--downsample-steps', type=int, default=512)
    p.add_argument('--max-events', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', type=str,
                   default='outputs/mamba_attention')
    p.add_argument('--no-pretrain', action='store_true',
                   help='跳过预训练 (调试用)')
    args = p.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        gpu_name = None; gpu_mem = None

    print("=" * 70)
    print("  Mamba-Attention 融合模型 - GPU 训练")
    print(f"  d_model={args.d_model}  mamba_layers={args.n_mamba_layers}  attn_layers={args.n_attn_layers}")
    print(f"  heads={args.n_heads}  d_state={args.d_state}  downsample_steps={args.downsample_steps}")
    print("=" * 70)
    if gpu_name:
        print(f"  设备: {device} ({gpu_name}, {gpu_mem:.1f} GB)")
    else:
        print(f"  设备: {device} (回退CPU)")
    print(f"  折数={args.folds}, pretrain_epochs={args.pretrain_epochs}, finetune_epochs={args.finetune_epochs}")
    print(f"  pretrain_bs={args.pretrain_batch_size}, finetune_bs={args.finetune_batch_size}")
    print()

    out_dir = args.output_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(_PROJECT_ROOT, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------ Step 1: 数据 ------------------------
    print("[Step 1] 7维事件编码...")
    samples, student_ids, labels = preprocess(max_events=args.max_events)
    print(f"  样本数: {len(samples)}, passed={(labels==0).sum()}, failed={(labels==1).sum()}")
    print()

    # ------------------------ 模型 ------------------------
    model = create_model(
        d_model=args.d_model,
        n_mamba_layers=args.n_mamba_layers,
        n_attn_layers=args.n_attn_layers,
        n_heads=args.n_heads,
        d_state=args.d_state,
        max_seq_len=args.max_events,
        downsample_steps=args.downsample_steps,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params / 1e6:.2f} M")
    print()

    # ------------------------ Step 2: 预训练 ------------------------
    if args.no_pretrain:
        print("[Step 2] 跳过预训练")
        pretrained_state = model.state_dict()
    else:
        print("[Step 2] 自监督预训练 - 下一事件预测")
        pretrained_state = pretrain_nextevent(
            model, samples, device,
            epochs=args.pretrain_epochs,
            batch_size=args.pretrain_batch_size,
        )
    print()

    # ------------------------ Step 3: 微调 (5 fold CV) ------------------------
    print(f"[Step 3] 微调 - {args.folds} 折交叉验证")
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(student_ids, labels)):
        print(f"\n  ─── Fold {fold_idx + 1}/{args.folds} ───")
        print(f"    训练: {len(train_idx)}, 测试: {len(test_idx)}")

        # 模型 + 预训练权重
        m = create_model(
            d_model=args.d_model,
            n_mamba_layers=args.n_mamba_layers,
            n_attn_layers=args.n_attn_layers,
            n_heads=args.n_heads,
            d_state=args.d_state,
            max_seq_len=args.max_events,
            downsample_steps=args.downsample_steps,
        ).to(device)
        m.load_state_dict(pretrained_state)
        freeze_for_finetune(m)

        train_samples = [samples[i] for i in train_idx]
        test_samples  = [samples[i] for i in test_idx]

        train_loader = DataLoader(
            train_samples, batch_size=args.finetune_batch_size, shuffle=True,
            collate_fn=lambda b: collate_finetune(b, args.max_events),
            drop_last=False,
        )
        test_loader = DataLoader(
            test_samples, batch_size=args.finetune_batch_size, shuffle=False,
            collate_fn=lambda b: collate_finetune(b, args.max_events),
            drop_last=False,
        )

        # 分组学习率
        head_keys = ('risk_head', 'fusion', 'attn_path.encoder',
                     'attn_path.cls_token', 'attn_path.pos_embed',
                     'attn_path.norm')
        head_params, backbone_params = [], []
        for n, pp in m.named_parameters():
            if not pp.requires_grad:
                continue
            (head_params if any(k in n for k in head_keys) else backbone_params).append(pp)

        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': 2e-5},
            {'params': head_params,     'lr': 1e-4},
        ], weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.finetune_epochs * len(train_loader), eta_min=1e-6
        )

        # Focal alpha
        y_train = np.array([s['risk'] for s in train_samples])
        n_pass = max(int((y_train == 0).sum()), 1)
        n_fail = max(int((y_train == 1).sum()), 1)
        alpha = torch.tensor(
            [len(y_train) / (2 * n_pass), len(y_train) / (2 * n_fail)],
            device=device, dtype=torch.float32
        ).clamp(max=2.0)
        print(f"    Focal alpha: passed={alpha[0].item():.3f}, failed={alpha[1].item():.3f}")

        # ---- 训练循环 ----
        m.train()
        for epoch in range(args.finetune_epochs):
            tot_loss, nb = 0.0, 0
            for batch in train_loader:
                x = {k: v.to(device) for k, v in batch.items() if k != 'risk'}
                y = batch['risk'].to(device)

                optimizer.zero_grad()
                logits = m(x)['risk']

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    continue
                loss = focal_loss(logits, y, alpha=alpha, gamma=2.0, label_smoothing=0.05)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                bad = any(
                    (pp.grad is not None) and
                    (torch.isnan(pp.grad).any() or torch.isinf(pp.grad).any())
                    for pp in m.parameters() if pp.requires_grad
                )
                if bad:
                    optimizer.zero_grad()
                    continue
                torch.nn.utils.clip_grad_norm_(
                    [pp for pp in m.parameters() if pp.requires_grad], max_norm=0.5
                )
                optimizer.step()
                sched.step()

                tot_loss += loss.item(); nb += 1
            avg = tot_loss / max(nb, 1)
            print(f"    Epoch {epoch+1}/{args.finetune_epochs} - Loss: {avg:.4f}")

        # ---- 评估 ----
        m.eval()
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                x = {k: v.to(device) for k, v in batch.items() if k != 'risk'}
                y = batch['risk'].to(device)
                logits = m(x)['risk']
                probs = torch.softmax(logits, dim=-1)

                y_true.extend(y.cpu().numpy())
                y_pred.extend(logits.argmax(dim=-1).cpu().numpy())
                y_prob.extend(probs[:, 1].cpu().numpy())

        from common.evaluator import evaluate
        y_true = np.array(y_true); y_pred = np.array(y_pred); y_prob = np.array(y_prob)
        metrics = evaluate(y_true, y_pred, y_prob)
        fold_results.append(metrics)
        print(f"    Acc: {metrics['accuracy']:.4f}  P: {metrics['precision']:.4f}  "
              f"R: {metrics['recall']:.4f}  F1: {metrics['f1']:.4f}  AUC: {metrics['auc']:.4f}")

    # ------------------------ 汇总 ------------------------
    summary = summarize_fold_results(fold_results)
    print_results_table("Mamba-Attention Fusion (GPU)", summary)

    # 保存
    results = {
        'model': 'MambaAttentionFusion',
        'config': {
            'd_model': args.d_model,
            'n_mamba_layers': args.n_mamba_layers,
            'n_attn_layers': args.n_attn_layers,
            'n_heads': args.n_heads,
            'd_state': args.d_state,
            'downsample_steps': args.downsample_steps,
            'max_events': args.max_events,
            'device': str(device),
            'gpu_name': gpu_name,
            'n_params_M': round(n_params / 1e6, 3),
        },
        'pipeline': {
            'pretrain_epochs': args.pretrain_epochs,
            'finetune_epochs': args.finetune_epochs,
            'n_folds': args.folds,
            'pretrain_batch_size': args.pretrain_batch_size,
            'finetune_batch_size': args.finetune_batch_size,
            'seed': args.seed,
        },
        'cv_results': {
            'accuracy_mean':  summary['accuracy_mean'],
            'accuracy_std':   summary['accuracy_std'],
            'precision_mean': summary['precision_mean'],
            'precision_std':  summary['precision_std'],
            'recall_mean':    summary['recall_mean'],
            'recall_std':     summary['recall_std'],
            'f1_mean':        summary['f1_mean'],
            'f1_std':         summary['f1_std'],
            'auc_mean':       summary['auc_mean'],
            'auc_std':        summary['auc_std'],
        },
        'fold_details': fold_results,
        'n_samples': len(samples),
        'n_passed':  int((labels == 0).sum()),
        'n_failed':  int((labels == 1).sum()),
    }
    json_path = os.path.join(out_dir, 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # CSV 详细
    csv_path = os.path.join(out_dir, 'comparison.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('fold,accuracy,precision,recall,f1,auc\n')
        for i, fr in enumerate(fold_results, 1):
            f.write(f"{i},{fr['accuracy']:.4f},{fr['precision']:.4f},"
                    f"{fr['recall']:.4f},{fr['f1']:.4f},{fr['auc']:.4f}\n")
        f.write(f"mean,{summary['accuracy_mean']:.4f},"
                f"{summary['precision_mean']:.4f},{summary['recall_mean']:.4f},"
                f"{summary['f1_mean']:.4f},{summary['auc_mean']:.4f}\n")
        f.write(f"std,{summary['accuracy_std']:.4f},"
                f"{summary['precision_std']:.4f},{summary['recall_std']:.4f},"
                f"{summary['f1_std']:.4f},{summary['auc_std']:.4f}\n")

    print()
    print("=" * 70)
    print("  ✅ Mamba-Attention 训练完成")
    print(f"  结果: {json_path}")
    print(f"  CSV:  {csv_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
