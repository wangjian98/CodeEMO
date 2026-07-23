"""
Step 1: 数据预处理 - 7维事件编码

将每个学生的IDE日志事件序列编码为模型可用的张量格式。

每个事件编码为以下维度:
  - event_type: 事件类型索引 (0-6, 共7类)
  - time_interval: 距上一事件的时间间隔 (log-normalized)
  - deadline_dist: 距截止时间距离 (归一化)
  - part_id: 题目部分编号 (1-7)

标签: risk=0 表示通过(passed), risk=1 表示挂科(at-risk/failed)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

# ============================================================
# sys.path 设置: 确保 common/ 和项目根目录可导入
# ============================================================
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_MODELS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _MODELS_ROOT not in sys.path:
    sys.path.insert(0, _MODELS_ROOT)

from common.data_loader import load_ide_logs
from models.mamba.model import encode_events, EVENT_TYPES


MAX_EVENTS = 2000


def preprocess(ide_logs_df=None, passed_df=None, max_events=MAX_EVENTS):
    """
    Step 1: 数据预处理 - 加载并编码所有学生的事件序列

    Args:
        ide_logs_df: IDE日志DataFrame, 若为None则自动加载
        passed_df: 标签DataFrame, 若为None则自动加载
        max_events: 最大事件序列长度 (截断)

    Returns:
        tuple: (samples, student_ids, labels)
            - samples: list of dict, 每个包含编码后的张量和risk标签
            - student_ids: np.array of student IDs
            - labels: np.array of risk labels (0=passed, 1=at-risk)
    """
    # 加载数据
    if ide_logs_df is None or passed_df is None:
        ide_logs_df, passed_df = load_ide_logs()

    print("\n[Step 1] 数据预处理 - 7维事件编码")
    print(f"  IDE日志数: {len(ide_logs_df)}")
    print(f"  学生数: {passed_df['student'].nunique()}")

    # 构建标签映射: student -> passed (bool)
    # risk = 1 - passed (passed=True -> risk=0, passed=False -> risk=1)
    passed_map = dict(zip(passed_df['student'], passed_df['passed']))

    # 按学生分组
    students = passed_df['student'].unique()
    samples = []
    student_ids = []
    labels = []

    skipped = 0
    for student_id in students:
        student_df = ide_logs_df[ide_logs_df['student'] == student_id].copy()

        if len(student_df) == 0:
            skipped += 1
            continue

        # 编码事件序列
        encoded = encode_events(student_df, max_events=max_events)

        # 获取风险标签
        passed = passed_map.get(student_id, True)
        risk = 0 if passed else 1  # risk=0: 通过, risk=1: 挂科

        sample = {
            'student_id': student_id,
            'event_types': encoded['event_types'],
            'time_intervals': encoded['time_intervals'],
            'deadline_dists': encoded['deadline_dists'],
            'part_ids': encoded['part_ids'],
            'n_events': encoded['n_events'],
            'risk': risk,
        }
        samples.append(sample)
        student_ids.append(student_id)
        labels.append(risk)

    student_ids = np.array(student_ids)
    labels = np.array(labels)

    # 打印统计信息
    print(f"  有效学生数: {len(samples)} (跳过 {skipped} 个无日志学生)")
    print(f"  事件序列统计:")
    event_counts = [s['n_events'] for s in samples]
    print(f"    最小事件数: {min(event_counts)}")
    print(f"    最大事件数: {max(event_counts)}")
    print(f"    平均事件数: {np.mean(event_counts):.1f}")
    print(f"    中位数: {np.median(event_counts):.1f}")
    print(f"  标签分布:")
    n_passed = (labels == 0).sum()
    n_failed = (labels == 1).sum()
    print(f"    通过 (risk=0): {n_passed} ({n_passed/len(labels)*100:.1f}%)")
    print(f"    挂科 (risk=1): {n_failed} ({n_failed/len(labels)*100:.1f}%)")
    print(f"  事件类型: {EVENT_TYPES}")

    return samples, student_ids, labels


if __name__ == '__main__':
    samples, student_ids, labels = preprocess()
    print(f"\n预处理完成: {len(samples)} 个样本")
    print(f"示例样本 keys: {list(samples[0].keys())}")
    print(f"示例 event_types shape: {samples[0]['event_types'].shape}")
