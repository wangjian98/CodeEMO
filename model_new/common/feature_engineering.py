"""
特征工程模块 - 生成46维特征

特征组成:
  1. 事件基础统计 (28维): 7种事件类型 x 4统计量(均值/标准差/变异系数/香农熵)
  2. 行为轨迹 (10维): improvement/consistency/trend/mean_interval/std_interval/
     min_interval/max_interval/duration_per_event/median_interval/iqr_interval
  3. 情绪复合特征 (6维): edit_ratio_mean/std, delete_ratio_mean/std, focus_ratio_mean/std
  4. 元信息 (2维): num_problems, total_events
"""
import numpy as np
import pandas as pd
from scipy.stats import entropy as shannon_entropy


EVENT_TYPES = [
    'text_insert', 'text_remove', 'text_paste',
    'focus_gained', 'focus_lost', 'run', 'submit'
]


def _safe_float(val, default=0.0):
    """安全转换为float"""
    if isinstance(val, (int, float)):
        return float(val) if np.isfinite(val) else default
    if isinstance(val, np.ndarray):
        return float(val.flat[0]) if val.size > 0 and np.isfinite(val.flat[0]) else default
    if hasattr(val, '__len__') and len(val) == 1:
        return _safe_float(val[0], default)
    return default


def _compute_shannon_entropy(counts):
    if len(counts) == 0:
        return 0.0
    counts = np.array(counts, dtype=float)
    if counts.sum() == 0:
        return 0.0
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def extract_features_for_student(student_df):
    """为单个学生提取46维特征，返回np.array(46,)"""
    features = np.zeros(46, dtype=np.float32)
    idx = 0

    # 1. 事件基础统计 (28维)
    for et in EVENT_TYPES:
        et_events = student_df[student_df['eventType'] == et]['timestamp']
        if len(et_events) > 0:
            times = (et_events - et_events.min()).dt.total_seconds().values
            if len(times) < 2:
                times = np.array([0.0, 0.0])
        else:
            times = np.array([0.0, 0.0])

        features[idx] = _safe_float(np.mean(times)); idx += 1
        features[idx] = _safe_float(np.std(times)); idx += 1
        features[idx] = _safe_float(np.std(times) / (np.mean(times) + 1e-10)); idx += 1

        bins = min(10, max(1, len(times) // 10))
        if bins > 1:
            hist, _ = np.histogram(times, bins=bins)
            features[idx] = _safe_float(_compute_shannon_entropy(hist + 1e-10)); idx += 1
        else:
            features[idx] = 0.0; idx += 1

    # 2. 行为轨迹 (10维)
    all_times = student_df['timestamp']
    if len(all_times) > 0:
        all_times_numeric = (all_times - all_times.min()).dt.total_seconds().values
    else:
        all_times_numeric = np.array([0.0])

    if len(all_times_numeric) < 2:
        idx += 10
    else:
        intervals = np.diff(all_times_numeric)
        if len(intervals) == 0:
            idx += 10
        else:
            x = np.arange(len(intervals))
            features[idx] = _safe_float(np.polyfit(x, intervals, 1)[0]) if len(intervals) >= 2 else 0.0; idx += 1

            mean_int = np.mean(intervals)
            features[idx] = _safe_float(np.std(intervals) / (mean_int + 1e-10)) if mean_int > 0 else 0.0; idx += 1

            x2 = np.arange(len(all_times_numeric))
            features[idx] = _safe_float(np.polyfit(x2, all_times_numeric, 1)[0]) if len(all_times_numeric) >= 2 else 0.0; idx += 1

            features[idx] = _safe_float(np.mean(intervals)); idx += 1
            features[idx] = _safe_float(np.std(intervals)); idx += 1
            features[idx] = _safe_float(np.min(intervals)); idx += 1
            features[idx] = _safe_float(np.max(intervals)); idx += 1

            duration = all_times_numeric[-1] - all_times_numeric[0]
            features[idx] = _safe_float(duration / (len(all_times_numeric) + 1e-10)); idx += 1

            features[idx] = _safe_float(np.median(intervals)); idx += 1

            q75, q25 = np.percentile(intervals, [75, 25])
            features[idx] = _safe_float(q75 - q25); idx += 1

    # 3. 情绪复合特征 (6维)
    edit_counts = student_df[student_df['eventType'] == 'text_insert'].groupby('exercise').size() if len(student_df) > 0 else pd.Series()
    delete_counts = student_df[student_df['eventType'] == 'text_remove'].groupby('exercise').size() if len(student_df) > 0 else pd.Series()
    focus_counts = student_df[student_df['eventType'] == 'focus_gained'].groupby('exercise').size() if len(student_df) > 0 else pd.Series()

    if len(edit_counts) > 0 and len(delete_counts) > 0:
        edit_ratios = edit_counts / (edit_counts + delete_counts + 1e-10)
        features[idx] = _safe_float(edit_ratios.mean()); idx += 1
        features[idx] = _safe_float(edit_ratios.std()); idx += 1
        delete_ratios = delete_counts / (edit_counts + delete_counts + 1e-10)
        features[idx] = _safe_float(delete_ratios.mean()); idx += 1
        features[idx] = _safe_float(delete_ratios.std()); idx += 1
    else:
        idx += 4

    total_events = student_df.groupby('exercise').size() if len(student_df) > 0 else pd.Series([1])
    if len(focus_counts) > 0:
        focus_ratios = focus_counts / (total_events + 1e-10)
        features[idx] = _safe_float(focus_ratios.mean()); idx += 1
        features[idx] = _safe_float(focus_ratios.std()); idx += 1
    else:
        idx += 2

    # 4. 元信息 (2维)
    features[idx] = float(student_df['exercise'].nunique()) if len(student_df) > 0 else 0.0; idx += 1
    features[idx] = float(len(student_df)); idx += 1

    return features


def build_feature_matrix(ide_logs, passed):
    """构建46维特征矩阵

    Returns:
        X: np.array (n_students, 46)
        y: np.array (n_students,) binary labels (1=passed, 0=failed)
        student_ids: list
    """
    print("Building 46-dim feature matrix...")
    data = ide_logs.merge(passed, left_on='student', right_on='student')

    feature_list = []
    labels = []
    student_ids = []

    for student_id, group in data.groupby('student'):
        feat = extract_features_for_student(group)
        feature_list.append(feat)
        labels.append(1 if group['passed'].iloc[0] in [True, 'True'] else 0)
        student_ids.append(student_id)

    X = np.array(feature_list, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(labels, dtype=np.int64)

    print(f"  Feature matrix: {X.shape}")
    print(f"  Passed: {sum(y)}, Failed: {len(y) - sum(y)}")

    return X, y, student_ids
