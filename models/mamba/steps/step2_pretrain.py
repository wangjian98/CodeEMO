"""
Step 2: Mamba 预训练 - 下一事件预测 (自监督)

使用下一事件预测任务 (Next-Event Prediction) 进行自监督预训练。
模型接收前 n-1 个事件, 预测第 n 个事件类型。

训练配置:
  - batch_size=16
  - optimizer: AdamW(lr=1e-3, weight_decay=0.01)
  - loss: CrossEntropyLoss
  - gradient clipping: max_norm=1.0
  - epochs: 默认2 (CPU) / 3 (GPU)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ============================================================
# sys.path 设置
# ============================================================
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.mamba.model import EVENT_TYPES


def collate_for_pretrain(samples):
    """
    为下一事件预测任务构建batch

    输入: 前 n-1 个事件
    目标: 第 n 个事件类型 (最后一个事件)

    Returns:
        dict with:
            event_types: (batch, seq_len-1) 前 n-1 个事件类型
            time_intervals: (batch, seq_len-1)
            deadline_dists: (batch, seq_len-1)
            part_ids: (batch, seq_len-1)
            next_events: (batch,) 目标事件类型 (LongTensor)
    """
    max_len = min(max(s['n_events'] for s in samples), 2000)
    input_len = max_len - 1

    batch_event_types = []
    batch_time_intervals = []
    batch_deadline_dists = []
    batch_part_ids = []
    batch_next_events = []

    for s in samples:
        n = s['n_events']
        et = s['event_types'][:n-1]
        ti = s['time_intervals'][:n-1]
        dd = s['deadline_dists'][:n-1]
        pi = s['part_ids'][:n-1]
        target = s['event_types'][n-1]

        # 填充到 input_len
        if len(et) < input_len:
            pad_len = input_len - len(et)
            et = F.pad(et, (0, pad_len), value=0)
            ti = F.pad(ti, (0, pad_len), value=0.0)
            dd = F.pad(dd, (0, pad_len), value=0.0)
            pi = F.pad(pi, (0, pad_len), value=0)

        batch_event_types.append(et[:input_len])
        batch_time_intervals.append(ti[:input_len])
        batch_deadline_dists.append(dd[:input_len])
        batch_part_ids.append(pi[:input_len])
        batch_next_events.append(target)

    return {
        'event_types': torch.stack(batch_event_types),
        'time_intervals': torch.stack(batch_time_intervals),
        'deadline_dists': torch.stack(batch_deadline_dists),
        'part_ids': torch.stack(batch_part_ids),
        'next_events': torch.stack(batch_next_events),
    }


def pretrain(model, dataset, device, epochs=2, batch_size=16):
    """
    Step 2: Mamba 自监督预训练 - 下一事件预测

    Args:
        model: Mamba学生模型 (SimplifiedMambaStudent 或 FullMambaStudent)
        dataset: Step 1 输出的 samples 列表
        device: 计算设备
        epochs: 预训练轮数
        batch_size: 批大小

    Returns:
        pretrained_state: 预训练后的模型 state_dict
    """
    print(f"\n[Step 2] Mamba 预训练 - 下一事件预测 (自监督)")
    print(f"  样本数: {len(dataset)}")
    print(f"  训练轮数: {epochs}")
    print(f"  批大小: {batch_size}")

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_for_pretrain, drop_last=False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        correct = 0
        total = 0

        for batch in dataloader:
            # 将batch数据移至设备
            batch_input = {
                'event_types': batch['event_types'].to(device),
                'time_intervals': batch['time_intervals'].to(device),
                'deadline_dists': batch['deadline_dists'].to(device),
                'part_ids': batch['part_ids'].to(device),
            }
            targets = batch['next_events'].to(device)

            optimizer.zero_grad()

            outputs = model(batch_input)
            logits = outputs['event']  # (batch, n_event_types)

            loss = criterion(logits, targets)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        avg_loss = total_loss / max(n_batches, 1)
        acc = correct / max(total, 1)
        print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

    print(f"  预训练完成")

    return model.state_dict()


if __name__ == '__main__':
    # 独立运行测试
    from models.mamba.steps.step1_preprocessing import preprocess
    from models.mamba.model import create_model
    from common.data_loader import get_device, set_seed

    set_seed(42)
    device = get_device()

    # Step 1: 加载数据
    samples, student_ids, labels = preprocess()

    # 创建模型
    model = create_model(device)

    # Step 2: 预训练
    pretrained_state = pretrain(model, samples, device, epochs=2)

    print(f"\n预训练测试完成, state_dict keys 数: {len(pretrained_state)}")
