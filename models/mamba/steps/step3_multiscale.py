"""
Step 3: 多尺度特征提取

此模块主要包含文档说明和特征提取辅助函数。

多尺度特征提取已集成在模型 forward 传播中:
  - SimplifiedMambaStudent: 全局均值 + 最后状态 + 分部均值 (3尺度融合)
  - FullMambaStudent: 细粒度窗口 + 中粒度全局 + 粗粒度分部 (3尺度交叉注意力融合)

本模块提供 extract_representations() 函数, 用于提取所有学生的多尺度表示,
供后续原型发现 (Step 4) 使用。

尺度说明:
  - Fine-grained (细粒度): 每100事件窗口的均值, 捕捉短期行为模式
  - Medium-grained (中粒度): 全局均值, 捕捉整体行为特征
  - Coarse-grained (粗粒度): 按题目部分(part)分组均值, 捕捉跨部分学习模式
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ============================================================
# sys.path 设置
# ============================================================
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.mamba.model import EVENT_TYPES


def collate_for_extract(samples):
    """
    为特征提取构建batch (带填充)

    与 collate_for_finetune 相同的填充逻辑,
    但不包含risk标签 (仅用于特征提取)
    """
    max_len = min(max(s['n_events'] for s in samples), 2000)

    batch_event_types = []
    batch_time_intervals = []
    batch_deadline_dists = []
    batch_part_ids = []

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

    return {
        'event_types': torch.stack(batch_event_types),
        'time_intervals': torch.stack(batch_time_intervals),
        'deadline_dists': torch.stack(batch_deadline_dists),
        'part_ids': torch.stack(batch_part_ids),
    }


def extract_representations(model, dataset, device, batch_size=16):
    """
    Step 3: 多尺度特征提取

    运行模型在所有数据上, 收集多尺度表示。

    模型内部的 forward 方法已集成多尺度特征提取:
      - SimplifiedMambaStudent: 3尺度 (全局/最后/分部) 直接平均
      - FullMambaStudent: 3尺度 (细/中/粗) 交叉注意力融合

    Args:
        model: 预训练后的 Mamba 模型
        dataset: Step 1 输出的 samples 列表
        device: 计算设备
        batch_size: 批大小

    Returns:
        tuple: (representations, labels)
            - representations: np.array (n_samples, d_model)
            - labels: np.array (n_samples,) risk labels
    """
    print(f"\n[Step 3] 多尺度特征提取")
    print(f"  样本数: {len(dataset)}")
    print(f"  多尺度策略: 全局均值 + 最后状态 + 分部均值 (融合)")

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_for_extract, drop_last=False
    )

    model.eval()
    representations = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch_input = {
                'event_types': batch['event_types'].to(device),
                'time_intervals': batch['time_intervals'].to(device),
                'deadline_dists': batch['deadline_dists'].to(device),
                'part_ids': batch['part_ids'].to(device),
            }

            outputs = model(batch_input, return_repr=True)
            repr_vec = outputs['repr']  # (batch, d_model)
            representations.append(repr_vec.cpu().numpy())

    # 收集标签 (按dataloader顺序, 但shuffle=False所以与dataset顺序一致)
    for s in dataset:
        labels.append(s['risk'])

    representations = np.concatenate(representations, axis=0)
    labels = np.array(labels)

    print(f"  特征维度: {representations.shape}")
    print(f"  标签分布: 通过(risk=0)={np.sum(labels==0)}, 挂科(risk=1)={np.sum(labels==1)}")

    return representations, labels


if __name__ == '__main__':
    from models.mamba.steps.step1_preprocessing import preprocess
    from models.mamba.model import create_model
    from common.data_loader import get_device, set_seed

    set_seed(42)
    device = get_device()

    # Step 1: 加载数据
    samples, student_ids, labels = preprocess()

    # 创建并初始化模型
    model = create_model(device)

    # Step 3: 提取表示
    reprs, lbls = extract_representations(model, samples, device)

    print(f"\n特征提取测试完成: {reprs.shape}")
