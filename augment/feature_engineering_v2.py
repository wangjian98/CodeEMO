"""
特征扩展模块 - 在 46 维基础上扩展到约 120 维
新增 3 类特征:
  1. 行为比率 (~20d)  - 跨特征交互,捕获业务直觉
  2. 高阶统计 (~20d)  - skew/kurtosis/percentiles/IQR
  3. 时序动态  (~30d)  - 前/中/后时段分组特征

注意:这些特征基于原始事件日志计算,不是从 46d 衍生而来,
因此能引入 46d 完全缺失的信息。
"""
import numpy as np
import pandas as pd
from scipy.stats import entropy as shannon_entropy
from typing import List


EVENT_TYPES = ['text_insert', 'text_remove', 'text_paste',
               'focus_gained', 'focus_lost', 'run', 'submit']


def _per_student_event_stats(df_student: pd.DataFrame) -> dict:
    """对单个学生的事件日志计算统计特征"""
    feats = {}

    # 1. 行为比率特征 (跨事件交互)
    n_insert = (df_student.eventType == 'text_insert').sum()
    n_remove = (df_student.eventType == 'text_remove').sum()
    n_paste = (df_student.eventType == 'text_paste').sum()
    n_focus_gained = (df_student.eventType == 'focus_gained').sum()
    n_focus_lost = (df_student.eventType == 'focus_lost').sum()
    n_run = (df_student.eventType == 'run').sum()
    n_submit = (df_student.eventType == 'submit').sum()
    n_total = len(df_student)

    # 编辑效率 (有效编辑比例)
    feats['edit_ratio'] = (n_insert + n_paste) / (n_total + 1)
    feats['remove_ratio'] = n_remove / (n_total + 1)
    feats['paste_ratio'] = n_paste / (n_total + 1)
    # 焦点平衡 (专注度)
    feats['focus_balance'] = n_focus_gained / (n_focus_lost + 1)
    feats['focus_ratio'] = (n_focus_gained + n_focus_lost) / (n_total + 1)
    # 编译/提交效率
    feats['run_per_submit'] = n_run / (n_submit + 1)
    feats['submit_rate'] = n_submit / (n_total + 1)
    feats['run_rate'] = n_run / (n_total + 1)
    # 调试强度 (运行多但提交少 = 反复试错)
    feats['debug_intensity'] = n_run / (n_insert + n_remove + 1)
    # 提交成功率(占事件比例)
    feats['commit_ratio'] = n_submit / (n_total + 1)
    # 编辑频率
    feats['edit_per_focus'] = (n_insert + n_remove + n_paste) / (n_focus_gained + 1)
    # 删除/插入比 (净效果)
    feats['remove_insert_ratio'] = n_remove / (n_insert + 1)
    # paste vs insert (依赖粘贴)
    feats['paste_dependency'] = n_paste / (n_insert + n_paste + 1)
    # focus 事件密度
    feats['focus_event_density'] = (n_focus_gained + n_focus_lost) / (n_total + 1)
    # run / focus (每专注一次跑几次)
    feats['run_per_focus'] = n_run / (n_focus_gained + 1)
    # submit / run (一次运行一次提交 = 高效)
    feats['submit_per_run'] = n_submit / (n_run + 1)
    # 综合编码强度
    feats['coding_intensity'] = (n_insert + n_remove) / (n_total + 1)
    # 行为多样性 (shannon entropy of event types)
    type_counts = [n_insert, n_remove, n_paste, n_focus_gained, n_focus_lost, n_run, n_submit]
    total = sum(type_counts) + 1e-9
    probs = [c / total for c in type_counts if c > 0]
    feats['event_diversity'] = shannon_entropy(probs) if probs else 0.0
    # 焦点切换频率
    feats['focus_switch_freq'] = (n_focus_gained + n_focus_lost) / (n_total + 1)
    # 写-跑-提 周期
    feats['cycle_efficiency'] = (n_insert + n_remove) / (n_run + n_submit + 1)
    # 删除率 (草稿/探索)
    feats['exploration_ratio'] = n_remove / (n_insert + n_remove + 1)

    # 2. 高阶统计特征 (基于 timestamp 间隔)
    if len(df_student) > 1:
        ts = df_student.timestamp.sort_values().values
        intervals = np.diff(ts).astype('timedelta64[s]').astype(float)
        # 移除异常值 (>2小时 = 7200s)
        intervals = intervals[intervals < 7200]
        if len(intervals) > 0:
            feats['interval_median'] = float(np.median(intervals))
            feats['interval_iqr'] = float(np.percentile(intervals, 75) - np.percentile(intervals, 25))
            feats['interval_p10'] = float(np.percentile(intervals, 10))
            feats['interval_p90'] = float(np.percentile(intervals, 90))
            feats['interval_p95'] = float(np.percentile(intervals, 95))
            feats['interval_skew'] = float(((intervals - intervals.mean()) ** 3).mean() / (intervals.std() + 1e-9) ** 3)
            feats['interval_kurtosis'] = float(((intervals - intervals.mean()) ** 4).mean() / (intervals.std() + 1e-9) ** 4 - 3)
            feats['interval_cv'] = float(intervals.std() / (intervals.mean() + 1e-9))
            feats['interval_max'] = float(intervals.max())
            feats['interval_min'] = float(intervals.min())
            # burst 模式 (前 10% 间隔极短)
            sorted_int = np.sort(intervals)
            feats['burst_ratio'] = float(np.percentile(intervals, 10) / (np.percentile(intervals, 90) + 1e-9))
            # session count (间隔 > 30 min 视为新会话)
            gap_thresh = 1800
            feats['session_count'] = int(np.sum(intervals > gap_thresh)) + 1
            # 长间隔次数 (>5min)
            feats['long_gap_count'] = int(np.sum(intervals > 300))
            # 短间隔次数 (<10s)
            feats['short_gap_count'] = int(np.sum(intervals < 10))
            # 时间跨度
            feats['time_span'] = float((ts[-1] - ts[0]).astype('timedelta64[s]').astype(float))
            # 早期 vs 后期活跃度对比
            mid = len(intervals) // 2
            if mid > 0:
                early_rate = mid / (intervals[:mid].sum() + 1e-9)
                late_rate = (len(intervals) - mid) / (intervals[mid:].sum() + 1e-9)
                feats['early_late_ratio'] = float(early_rate / (late_rate + 1e-9))
            else:
                feats['early_late_ratio'] = 1.0
            # 节奏变化 (后段 std / 前段 std)
            if mid > 1:
                early_std = intervals[:mid].std()
                late_std = intervals[mid:].std()
                feats['rhythm_change'] = float(late_std / (early_std + 1e-9))
            else:
                feats['rhythm_change'] = 1.0
            # 平均事件间隔 (重命名避免重复)
            feats['mean_interval_v2'] = float(intervals.mean())
        else:
            for k in ['interval_median', 'interval_iqr', 'interval_p10', 'interval_p90',
                      'interval_p95', 'interval_skew', 'interval_kurtosis', 'interval_cv',
                      'interval_max', 'interval_min', 'burst_ratio', 'session_count',
                      'long_gap_count', 'short_gap_count', 'time_span',
                      'early_late_ratio', 'rhythm_change', 'mean_interval_v2']:
                feats[k] = 0.0
    else:
        for k in ['interval_median', 'interval_iqr', 'interval_p10', 'interval_p90',
                  'interval_p95', 'interval_skew', 'interval_kurtosis', 'interval_cv',
                  'interval_max', 'interval_min', 'burst_ratio', 'session_count',
                  'long_gap_count', 'short_gap_count', 'time_span',
                  'early_late_ratio', 'rhythm_change', 'mean_interval_v2']:
            feats[k] = 0.0

    # 3. 时序动态特征 - 分 3 个时段
    if len(df_student) > 2:
        ts_sorted = df_student.sort_values('timestamp').reset_index(drop=True)
        n = len(ts_sorted)
        q1, q2 = n // 3, 2 * n // 3

        for period_name, period_df in [
            ('early', ts_sorted.iloc[:q1 if q1 > 0 else 1]),
            ('mid', ts_sorted.iloc[q1:q2 if q2 > q1 else q1 + 1]),
            ('late', ts_sorted.iloc[q2:])
        ]:
            if len(period_df) == 0:
                continue
            feats[f'{period_name}_event_count'] = len(period_df)
            feats[f'{period_name}_insert_rate'] = (period_df.eventType == 'text_insert').mean()
            feats[f'{period_name}_remove_rate'] = (period_df.eventType == 'text_remove').mean()
            feats[f'{period_name}_focus_rate'] = (period_df.eventType == 'focus_gained').mean()
            feats[f'{period_name}_run_rate'] = (period_df.eventType == 'run').mean()
            # 净代码量 (insert - remove)
            n_ins = (period_df.eventType == 'text_insert').sum()
            n_rem = (period_df.eventType == 'text_remove').sum()
            feats[f'{period_name}_net_code'] = (n_ins - n_rem) / (len(period_df) + 1)
            # 该时段提交/运行
            feats[f'{period_name}_submit_rate'] = (period_df.eventType == 'submit').mean()
            # 该时段的事件间隔均值
            if len(period_df) > 1:
                p_ts = period_df.timestamp.values
                p_int = np.diff(p_ts).astype('timedelta64[s]').astype(float)
                p_int = p_int[p_int < 7200]
                feats[f'{period_name}_avg_interval'] = float(p_int.mean()) if len(p_int) > 0 else 0.0
                feats[f'{period_name}_std_interval'] = float(p_int.std()) if len(p_int) > 0 else 0.0
            else:
                feats[f'{period_name}_avg_interval'] = 0.0
                feats[f'{period_name}_std_interval'] = 0.0
    else:
        for p in ['early', 'mid', 'late']:
            for k in ['event_count', 'insert_rate', 'remove_rate', 'focus_rate',
                      'run_rate', 'net_code', 'submit_rate', 'avg_interval', 'std_interval']:
                feats[f'{p}_{k}'] = 0.0

    return feats


def build_extended_features(ide_logs: pd.DataFrame) -> tuple:
    """构建扩展特征矩阵 (46d 原特征 + ~90d 新特征 ≈ 136d)

    Returns:
        X: (n_students, n_extended_features)
        feature_names: list of feature names
    """
    print("Building extended features (~136d)...")
    grouped = ide_logs.groupby('student')
    rows = []
    student_ids = []
    for sid, df_s in grouped:
        feats = _per_student_event_stats(df_s)
        rows.append(feats)
        student_ids.append(sid)

    X_new = pd.DataFrame(rows).fillna(0.0)
    # 替换 inf
    X_new = X_new.replace([np.inf, -np.inf], 0.0)
    return X_new.values, X_new.columns.tolist(), student_ids


def combine_features(X_original: np.ndarray, X_extended: np.ndarray) -> np.ndarray:
    """拼接 46d 原特征 + 扩展特征"""
    return np.hstack([X_original, X_extended])