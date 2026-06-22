"""
Step 5: 预测微调 - 5折交叉验证风险分类

在预训练权重基础上, 冻结骨干网络 (仅训练 risk_head),
通过5折分层交叉验证评估模型性能。

标签说明:
  - risk=0: 通过 (passed)
  - risk=1: 挂科 (at-risk/failed)
  - risk_head 输出2类: class 0=passed, class 1=at-risk

训练配置:
  - 冻结: event_embed, time_embed, deadline_embed, input_proj, mamba, final_norm,
          part_attn, prototype_centers, event_head
  - 仅训练: risk_head
  - 每折训练5个epoch
  - 损失函数: CrossEntropyLoss
  - 优化器: AdamW(lr=1e-3, weight_decay=0.01)
  - 梯度裁剪: max_norm=1.0
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
    """
    为风险分类任务构建batch (带填充)

    将所有样本的序列填充到batch内最大长度 (上限2000)。

    Returns:
        dict with:
            event_types: (batch, max_len)
            time_intervals: (batch, max_len)
            deadline_dists: (batch, max_len)
            part_ids: (batch, max_len)
            risk: (batch,) LongTensor of labels
    """
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

        # 填充到 max_len
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


def _freeze_backbone(model):
    """冻结骨干网络参数, 仅保留 risk_head 可训练"""
    frozen_params = [
        'event_embed', 'time_embed', 'deadline_embed',
        'input_proj', 'mamba', 'final_norm',
        'part_attn', 'prototype_centers', 'event_head',
        'fine_proj', 'medium_proj', 'coarse_proj',
        'cross_attn', 'scale_fusion',
    ]
    for name, param in model.named_parameters():
        # 检查参数名是否以任何冻结模块开头
        should_freeze = any(name.startswith(fp) for fp in frozen_params)
        if should_freeze:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"    可训练参数: {trainable}/{total} ({trainable/total*100:.1f}%)")


def finetune_cv(pretrained_state, dataset, student_ids, y, device,
                n_folds=5, n_epochs=5, batch_size=16):
    """
    Step 5: 5折交叉验证微调

    Args:
        pretrained_state: 预训练的模型 state_dict
        dataset: Step 1 输出的 samples 列表
        student_ids: np.array of student IDs
        y: np.array of risk labels (0=passed, 1=at-risk)
        device: 计算设备
        n_folds: 交叉验证折数
        n_epochs: 每折训练轮数
        batch_size: 批大小

    Returns:
        fold_results: list of dict (每折的评估指标)
    """
    print(f"\n[Step 5] 预测微调 - {n_folds}折交叉验证")
    print(f"  样本数: {len(dataset)}")
    print(f"  标签分布: 通过={np.sum(y==0)}, 挂科={np.sum(y==1)}")
    print(f"  训练轮数/折: {n_epochs}")
    print(f"  批大小: {batch_size}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(student_ids, y)):
        print(f"\n  --- Fold {fold_idx+1}/{n_folds} ---")
        print(f"    训练集: {len(train_idx)}, 测试集: {len(test_idx)}")

        # 创建新模型并加载预训练权重
        model = create_model(device)
        model.load_state_dict(pretrained_state)

        # 冻结骨干网络
        _freeze_backbone(model)

        # 准备数据
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

        # 仅优化可训练参数 (risk_head)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

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

                loss = criterion(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            if (epoch + 1) % 1 == 0:
                print(f"    Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.4f}")

        # 评估
        model.eval()
        y_true = []
        y_pred = []
        y_prob = []

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
                y_prob.extend(probs[:, 1].cpu().numpy())  # P(at-risk)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        metrics = evaluate(y_true, y_pred, y_prob)
        fold_results.append(metrics)

        print(f"    Accuracy: {metrics['accuracy']:.4f}, "
              f"Precision: {metrics['precision']:.4f}, "
              f"Recall: {metrics['recall']:.4f}, "
              f"F1: {metrics['f1']:.4f}, "
              f"AUC: {metrics['auc']:.4f}")

    # 汇总结果
    summary = summarize_fold_results(fold_results)
    print_results_table("Mamba-SSM", summary)

    return fold_results


if __name__ == '__main__':
    from models.mamba.steps.step1_preprocessing import preprocess
    from models.mamba.steps.step2_pretrain import pretrain
    from common.data_loader import get_device, set_seed

    set_seed(42)
    device = get_device()

    # Step 1: 加载数据
    samples, student_ids, labels = preprocess()

    # Step 2: 预训练
    model = create_model(device)
    pretrained_state = pretrain(model, samples, device, epochs=2)

    # Step 5: 微调交叉验证
    fold_results = finetune_cv(pretrained_state, samples, student_ids, labels, device)

    print(f"\n微调测试完成: {len(fold_results)} 折")
