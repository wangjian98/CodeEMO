"""
Step 5: 预测微调 - 5折交叉验证风险分类 (v2 强化版)

升级内容 (相比 baseline):
  - 解冻策略: risk_head + 最后一层 MambaBlock + final_norm
  - 分层学习率: head=5e-4, backbone=5e-5
  - 损失函数: Focal Loss (gamma=2) + class_weight
  - CosineAnnealing 学习率调度
  - NaN/Inf 防御: 跳过坏 batch, 收紧梯度裁剪到 0.5
  - Label smoothing 0.05

标签说明:
  - risk=0: 通过 (passed)
  - risk=1: 挂科 (at-risk/failed)
"""

import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

# ============================================================
# sys.path 设置
# ============================================================
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.evaluator import evaluate, summarize_fold_results, print_results_table
from models.mamba.model import create_model


def collate_for_finetune(samples):
    """为风险分类任务构建batch (带填充)"""
    max_len = min(max(s['n_events'] for s in samples), 2000)

    batch_event_types = []
    batch_time_intervals = []
    batch_deadline_dists = []
    batch_part_ids = []
    batch_risk = []

    for s in samples:
        n = s['n_events']
        et = s['event_types'][:n]
        ti = s['time_intervals'][:n]
        dd = s['deadline_dists'][:n]
        pi = s['part_ids'][:n]

        if len(et) < max_len:
            pad_len = max_len - len(et)
            et = F.pad(et, (0, pad_len), value=0)
            ti = F.pad(ti, (0, pad_len), value=0.0)
            dd = F.pad(dd, (0, pad_len), value=0.0)
            pi = F.pad(pi, (0, pad_len), value=0)

        batch_event_types.append(et[:max_len])
        batch_time_intervals.append(ti[:max_len])
        batch_deadline_dists.append(dd[:max_len])
        batch_part_ids.append(pi[:max_len])
        batch_risk.append(s['risk'])

    return {
        'event_types': torch.stack(batch_event_types),
        'time_intervals': torch.stack(batch_time_intervals),
        'deadline_dists': torch.stack(batch_deadline_dists),
        'part_ids': torch.stack(batch_part_ids),
        'risk': torch.LongTensor(batch_risk),
    }


def focal_loss(logits, targets, alpha=None, gamma=2.0, label_smoothing=0.05):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        logits: (B, C)
        targets: (B,)
        alpha: (C,) 类别权重, 缓解不平衡
        gamma: 聚焦参数, 默认 2.0
        label_smoothing: 标签平滑
    """
    num_classes = logits.shape[-1]
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    # label smoothing
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(label_smoothing / num_classes)
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing + label_smoothing / num_classes)

    # focal 权重
    target_one_hot = F.one_hot(targets, num_classes).float()
    p_t = (probs * target_one_hot).sum(dim=-1)        # (B,)
    focal_weight = (1.0 - p_t) ** gamma                # (B,)

    # alpha 权重 (按类别)
    if alpha is not None:
        alpha_t = alpha[targets]                        # (B,)
    else:
        alpha_t = 1.0

    loss = -(alpha_t * focal_weight * (log_probs * true_dist).sum(dim=-1))
    return loss.mean()


def _freeze_backbone(model):
    """冻结骨干网络, 仅解冻 risk_head + 最后一层 Mamba + final_norm + se_gate"""
    frozen_params = [
        'event_embed', 'time_embed', 'deadline_embed',
        'input_proj',
        'mamba.layers.0', 'mamba.layers.1', 'mamba.layers.2',
        'mamba.layers.3', 'mamba.layers.4',     # 冻结前 5 层
        'part_attn', 'prototype_centers', 'event_head',
        'fine_proj', 'medium_proj', 'coarse_proj',
        'cross_attn', 'scale_fusion',
    ]
    for name, param in model.named_parameters():
        should_freeze = any(name.startswith(fp) for fp in frozen_params)
        if should_freeze:
            param.requires_grad = False
        else:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"    可训练参数: {trainable}/{total} ({trainable/total*100:.2f}%)")


def finetune_cv(pretrained_state, dataset, student_ids, y, device,
                n_folds=5, n_epochs=10, batch_size=8, seed=42):
    """Step 5 v3: 强化版 - γ=3, 加风险头 BN, 多 seed 稳定评估"""
    """Step 5 v2: 强化版微调交叉验证"""
    print(f"\n[Step 5 v2] 预测微调 - {n_folds}折交叉验证 (Focal Loss)")
    print(f"  样本数: {len(dataset)}")
    print(f"  标签分布: 通过={np.sum(y==0)}, 挂科={np.sum(y==1)}")
    print(f"  训练轮数/折: {n_epochs}")
    print(f"  批大小: {batch_size}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(student_ids, y)):
        print(f"\n  --- Fold {fold_idx+1}/{n_folds} ---")
        print(f"    训练集: {len(train_idx)}, 测试集: {len(test_idx)}")

        # 新模型 + 预训练权重
        model = create_model(device)
        model.load_state_dict(pretrained_state)
        _freeze_backbone(model)

        # 数据
        train_samples = [dataset[i] for i in train_idx]
        test_samples = [dataset[i] for i in test_idx]

        train_loader = DataLoader(
            train_samples, batch_size=batch_size, shuffle=True,
            collate_fn=collate_for_finetune, drop_last=False
        )
        test_loader = DataLoader(
            test_samples, batch_size=batch_size, shuffle=False,
            collate_fn=collate_for_finetune, drop_last=False
        )

        # 分层学习率
        head_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and ('risk_head' in n or 'se_gate' in n)]
        backbone_params = [p for n, p in model.named_parameters()
                           if p.requires_grad and n not in [pn for pn in
                                                            [n for n, _ in model.named_parameters()
                                                             if 'risk_head' in n or 'se_gate' in n]]]
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': 5e-5},
            {'params': head_params,     'lr': 5e-4},
        ], weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs * len(train_loader), eta_min=1e-6
        )

        # Focal Loss 类别权重: 平衡正负样本
        y_train = np.array([s['risk'] for s in train_samples])
        n_pass = max(int((y_train == 0).sum()), 1)
        n_fail = max(int((y_train == 1).sum()), 1)
        # 让少数类 (passed) 权重略高, 但不过分 (alpha <= 2.0)
        alpha = torch.tensor(
            [len(y_train) / (2 * n_pass), len(y_train) / (2 * n_fail)],
            dtype=torch.float32, device=device
        ).clamp(max=2.0)
        print(f"    Focal alpha: passed={alpha[0].item():.3f}, failed={alpha[1].item():.3f}")

        # 训练
        model.train()
        best_loss = float('inf')
        for epoch in range(n_epochs):
            gamma = 2.0 + (epoch / max(n_epochs - 1, 1))   # 2.0 → 3.0 渐进, 防止初期过聚焦
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

                # 数值防御
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    optimizer.zero_grad()
                    continue

                loss = focal_loss(logits, targets, alpha=alpha,
                                  gamma=gamma, label_smoothing=0.05)

                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad()
                    continue

                loss.backward()

                # 梯度 NaN/Inf 防御
                bad_grad = any(
                    (p.grad is not None) and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                    for p in model.parameters() if p.requires_grad
                )
                if bad_grad:
                    optimizer.zero_grad()
                    continue

                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=0.5
                )
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            print(f"    Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.4f}")

        # 评估
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

        metrics = evaluate(y_true, y_pred, y_prob)
        fold_results.append(metrics)

        print(f"    Acc: {metrics['accuracy']:.4f}, "
              f"P: {metrics['precision']:.4f}, "
              f"R: {metrics['recall']:.4f}, "
              f"F1: {metrics['f1']:.4f}, "
              f"AUC: {metrics['auc']:.4f}")

    summary = summarize_fold_results(fold_results)
    print_results_table("Mamba-SSM v2 (Focal + Hand-crafted)", summary)
    return fold_results


if __name__ == '__main__':
    from models.mamba.steps.step1_preprocessing import preprocess
    from models.mamba.steps.step2_pretrain import pretrain
    from common.data_loader import get_device, set_seed

    set_seed(42)
    device = get_device()

    samples, student_ids, labels = preprocess()
    model = create_model(device)
    pretrained_state = pretrain(model, samples, device, epochs=2)
    fold_results = finetune_cv(pretrained_state, samples, student_ids, labels, device)
    print(f"\n微调测试完成: {len(fold_results)} 折")