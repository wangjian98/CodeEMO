"""AUTOML 模型定义 - TSFRESH 特征 + Random Forest

本模块使用 TSFRESH（Time Series FeatuRe Extraction on basis of Scalable
Hypothesis tests）自动从学生 IDE 事件时间序列中提取大量特征，
作为对照手工 46 维特征的 AutoML 基线。

原理:
  TSFRESH 会计算 ~700 种统计 / 频谱 / 复杂度算子，对每种事件类型生成完整
  的特征向量，再用 Benjamini-Yekutieli FDR 校正的多重假设检验过滤掉与
  目标变量无关的特征。最终保留的特征子集喂给 RF 分类器。

输入输出:
  输入: 学生事件日志 (long format: id, time, kind, value)
  输出: predict() -> y_pred, predict_proba() -> P(passed=1)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# 7 种事件类型，与手工 46 维特征保持一致
EVENT_TYPES = [
    'text_insert', 'text_remove', 'text_paste',
    'focus_gained', 'focus_lost', 'run', 'submit'
]


def create_classifier(n_estimators=200, max_depth=10, random_state=42):
    """创建用于 TSFRESH 特征的 Random Forest 分类器

    Args:
        n_estimators: 决策树数量
        max_depth: 树的最大深度
        random_state: 随机种子

    Returns:
        RandomForestClassifier 实例
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'  # 处理 33.6% 不平衡
    )


def build_long_format(ide_logs, max_events_per_student=5000):
    """将 IDE 日志转换为 TSFRESH 所需的 long format

    TSFRESH 要求输入为长格式 DataFrame:
      - id:  时间序列标识 (student id)
      - time: 时间戳 (秒，相对每名学生起始时间)
      - kind: 多变量时间序列的变量名 (event type)
      - value: 该时刻的数值

    为避免 28M 事件对 TSFRESH 造成计算压力，对每名学生的事件数做截断。
    默认保留每个学生前 max_events_per_student 个事件（按时间排序）。

    Args:
        ide_logs: DataFrame, columns=[student, part, exercise, eventType,
                    timestamp, timeToDeadline]
        max_events_per_student: 每个学生保留的最大事件数

    Returns:
        pd.DataFrame: long format with columns=[id, time, kind, value]
    """
    print(f"  Building TSFRESH long-format from {len(ide_logs)} events ...")

    # 转换时间戳为相对秒数（按 student 分组，各自归零）
    ide_logs = ide_logs.copy()
    ide_logs['timestamp'] = pd.to_datetime(ide_logs['timestamp'])
    ide_logs = ide_logs.sort_values(['student', 'timestamp'])

    # 计算每名学生的相对时间
    student_min_time = ide_logs.groupby('student')['timestamp'].transform('min')
    ide_logs['rel_time'] = (
        ide_logs['timestamp'] - student_min_time
    ).dt.total_seconds()

    # 长格式
    long_df = pd.DataFrame({
        'id': ide_logs['student'].values,
        'time': ide_logs['rel_time'].values,
        'kind': ide_logs['eventType'].values,
        'value': 1.0  # 每个事件计为 1
    })

    # 限制每个学生的事件数（按时间排序取前 N 个）
    long_df = long_df.sort_values(['id', 'time'])
    long_df = long_df.groupby('id').head(max_events_per_student).reset_index(drop=True)

    print(f"  Long-format shape: {long_df.shape}")
    print(f"  Unique students (ids): {long_df['id'].nunique()}")
    print(f"  Event kinds: {long_df['kind'].unique().tolist()}")

    return long_df


def extract_tsfresh_features(long_df, n_jobs=8, kind_to_fc_params=None):
    """使用 TSFRESH 提取时间序列特征

    对每种事件类型 ('kind') 提取传入的特征计算参数集下的所有算子。
    输出形状: (n_students, n_features_total)，其中 n_features_total 取决于
    特征提取设置和学生数。

    为控制计算量，本实现默认对每个 event kind 使用 MinimalFCParameters
    （约 10 个算子），再对全量时间戳序列（不区分 kind）使用 EfficientFCParameters
    的子集。如果需要更全面的特征，可切换到 ComprehensiveFCParameters。

    Args:
        long_df: TSFRESH long-format DataFrame
        n_jobs: 并行核数
        kind_to_fc_params: 各 event kind 的特征提取参数，默认用 MinimalFCParameters

    Returns:
        pd.DataFrame: (n_students, n_features) 特征矩阵
    """
    from tsfresh import extract_features
    from tsfresh.feature_extraction import MinimalFCParameters

    if kind_to_fc_params is None:
        kind_to_fc_params = MinimalFCParameters()

    print(f"  Extracting TSFRESH features (kind-aware) ...")
    print(f"  Event kinds: {long_df['kind'].nunique()}")

    features = extract_features(
        long_df,
        column_id='id',
        column_sort='time',
        column_kind='kind',
        column_value='value',
        default_fc_parameters=kind_to_fc_params,
        n_jobs=n_jobs,
        disable_progressbar=False
    )

    # 处理 NaN 和 Inf（TSFRESH 在某些算子上会产生）
    features = features.replace([np.inf, -np.inf], np.nan)
    n_before = features.shape[1]
    # 删除全为 NaN 的特征
    features = features.dropna(axis=1, how='all')
    # 用中位数填充剩余 NaN
    features = features.fillna(features.median(numeric_only=True))
    # 用 0 填充仍存在的 NaN（如果某列全为 NaN 则 median 也是 NaN）
    features = features.fillna(0)

    n_after = features.shape[1]
    print(f"  Raw features: {n_before}, after cleaning: {n_after}")
    print(f"  Shape: {features.shape}")

    return features


def select_features_by_target(features, y, fdr_level=0.05):
    """基于 FDR 校正的多重假设检验选择与目标相关的特征

    Args:
        features: TSFRESH 输出的特征矩阵
        y: 目标标签 (0/1)
        fdr_level: FDR 控制水平（越小保留越少特征）

    Returns:
        pd.DataFrame: 筛选后的特征矩阵

    Notes:
        若 FDR 选择后无特征保留（典型场景：小样本 + 严格 FDR 水平），
        自动回退到保留所有特征。
    """
    from tsfresh import select_features
    import warnings as _w

    print(f"  Selecting features via FDR-corrected hypothesis testing (fdr_level={fdr_level}) ...")
    print(f"  Before selection: {features.shape[1]} features")

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        features_selected = select_features(features, y, fdr_level=fdr_level)

    n_selected = features_selected.shape[1]
    if n_selected == 0:
        print(f"  WARNING: FDR selection retained 0 features; falling back to all features.")
        print(f"  Consider relaxing fdr_level (e.g., 0.1/0.2) or using more samples.")
        return features
    else:
        print(f"  After selection:  {n_selected} features")
        print(f"  Retention rate:   {100.0 * n_selected / features.shape[1]:.1f}%")
        return features_selected