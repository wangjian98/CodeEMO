"""
Step 6: 可解释性分析

利用模型的 get_interpretability() 方法提取可解释性信息:

1. 事件类型重要性: 哪些事件类型对预测贡献最大
   - focus_gained / focus_lost: 注意力切换频率
   - text_insert / text_remove: 编码活动强度
   - text_paste: 复制粘贴行为 (可能的作弊指标)
   - run / submit: 调试和提交行为

2. 时间模式: 学生行为在时间维度上的分布
   - 近期行为 vs 早期行为的比率
   - 识别"临时抱佛脚"模式

3. 原型分布: 学生被分配到各行为原型的概率
   - 哪些原型与高风险相关
   - 学生群体聚类特征
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
from models.mamba.steps.step3_multiscale import collate_for_extract


def run_interpretability(model, dataset, device, batch_size=16):
    """
    Step 6: 可解释性分析

    Args:
        model: 训练好的 Mamba 模型
        dataset: Step 1 输出的 samples 列表
        device: 计算设备
        batch_size: 批大小

    Returns:
        interpret_dict: dict with keys:
            - event_importance: (n_event_types,) 事件类型重要性
            - event_type_names: list of event type names
            - temporal_patterns: dict with temporal analysis
            - proto_distribution: (n_samples, n_prototypes) 原型分布
            - proto_labels: (n_samples,) 主要原型分配
            - risk_correlation: 原型与风险的相关性
    """
    print(f"\n[Step 6] 可解释性分析")
    print(f"  样本数: {len(dataset)}")

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_for_extract, drop_last=False
    )

    model.eval()
    all_event_importance = []
    all_temporal = []
    all_proto_weights = []
    all_proto_ids = []
    all_risk = []

    with torch.no_grad():
        for batch in dataloader:
            batch_input = {
                'event_types': batch['event_types'].to(device),
                'time_intervals': batch['time_intervals'].to(device),
                'deadline_dists': batch['deadline_dists'].to(device),
                'part_ids': batch['part_ids'].to(device),
            }

            interp = model.get_interpretability(batch_input)

            all_event_importance.append(interp['event_importance'])
            if 'temporal_ratio' in interp:
                all_temporal.append(interp['temporal_ratio'])
            elif 'temporal_importance' in interp:
                all_temporal.append(interp['temporal_importance'])
            all_proto_weights.append(interp['proto_weights'])
            all_proto_ids.append(interp['proto_id'])

    # 收集风险标签
    for s in dataset:
        all_risk.append(s['risk'])

    # 聚合结果
    event_importance = np.mean(all_event_importance, axis=0)
    event_type_names = EVENT_TYPES

    temporal = np.concatenate(all_temporal, axis=0).flatten()
    proto_weights_all = np.concatenate(all_proto_weights, axis=0)
    proto_ids_all = np.concatenate(all_proto_ids, axis=0)
    risk_labels = np.array(all_risk)

    # 1. 打印事件类型重要性
    print(f"\n  1. 事件类型重要性:")
    print(f"  {'事件类型':<20s} | {'重要性':>8s}")
    print(f"  {'-'*20}-+-{'-'*8}")
    sorted_idx = np.argsort(event_importance)[::-1]
    for idx in sorted_idx:
        bar = '#' * int(event_importance[idx] * 50)
        print(f"  {event_type_names[idx]:<20s} | {event_importance[idx]:.4f} {bar}")

    # 2. 打印时间模式
    print(f"\n  2. 时间模式分析:")
    print(f"    时间模式均值: {temporal.mean():.4f}")
    print(f"    时间模式标准差: {temporal.std():.4f}")
    print(f"    时间模式范围: [{temporal.min():.4f}, {temporal.max():.4f}]")

    # 按风险分组比较
    passed_temporal = temporal[risk_labels == 0]
    failed_temporal = temporal[risk_labels == 1]
    if len(passed_temporal) > 0:
        print(f"    通过学生时间模式均值: {passed_temporal.mean():.4f}")
    if len(failed_temporal) > 0:
        print(f"    挂科学生时间模式均值: {failed_temporal.mean():.4f}")

    # 3. 打印原型分布
    n_prototypes = proto_weights_all.shape[1]
    print(f"\n  3. 原型分布:")
    print(f"  {'原型':>6s} | {'样本数':>6s} | {'占比':>6s} | {'平均风险':>8s} | {'平均权重':>8s}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}")

    proto_risk_correlation = []
    for p in range(n_prototypes):
        mask = (proto_ids_all == p)
        n_in_proto = mask.sum()
        if n_in_proto > 0:
            avg_risk = risk_labels[mask].mean()
            avg_weight = proto_weights_all[:, p].mean()
        else:
            avg_risk = 0.0
            avg_weight = proto_weights_all[:, p].mean()

        pct = n_in_proto / len(risk_labels) * 100
        print(f"  {p:>6d} | {n_in_proto:>6d} | {pct:>5.1f}% | {avg_risk:>8.4f} | {avg_weight:>8.4f}")
        proto_risk_correlation.append({
            'prototype': p,
            'n_samples': int(n_in_proto),
            'avg_risk': float(avg_risk),
            'avg_weight': float(avg_weight),
        })

    # 4. 原型-风险相关性总结
    print(f"\n  4. 原型-风险关联:")
    for prc in proto_risk_correlation:
        level = "高风险" if prc['avg_risk'] > 0.5 else ("中等风险" if prc['avg_risk'] > 0.2 else "低风险")
        print(f"    原型 {prc['prototype']}: {level} (avg_risk={prc['avg_risk']:.3f}, n={prc['n_samples']})")

    interpret_dict = {
        'event_importance': event_importance,
        'event_type_names': event_type_names,
        'temporal_mean': float(temporal.mean()),
        'temporal_std': float(temporal.std()),
        'proto_weights': proto_weights_all,
        'proto_ids': proto_ids_all,
        'proto_risk_correlation': proto_risk_correlation,
        'n_prototypes': n_prototypes,
    }

    print(f"\n  可解释性分析完成")

    return interpret_dict


if __name__ == '__main__':
    from models.mamba.steps.step1_preprocessing import preprocess
    from models.mamba.model import create_model
    from common.data_loader import get_device, set_seed

    set_seed(42)
    device = get_device()

    # Step 1: 加载数据
    samples, student_ids, labels = preprocess()

    # 创建模型
    model = create_model(device)

    # Step 6: 可解释性
    interp = run_interpretability(model, samples, device)

    print(f"\n可解释性测试完成")
    print(f"事件重要性: {interp['event_importance']}")
