"""
特征工程模块 - 生成46维特征
"""
import pandas as pd
import numpy as np
from scipy.stats import entropy, linregress
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def compute_shannon_entropy(counts):
    """计算香农熵"""
    if len(counts) == 0:
        return 0.0
    counts = np.array(counts, dtype=float)
    if counts.sum() == 0:
        return 0.0
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def compute_cv(values):
    """计算变异系数"""
    if len(values) == 0 or np.mean(values) == 0:
        return 0.0
    return np.std(values) / (np.mean(values) + 1e-10)

def compute_improvement(event_times, event_types):
    """计算改进趋势指标"""
    if len(event_times) < 2:
        return 0.0
    
    # 计算事件间隔
    intervals = np.diff(event_times)
    if len(intervals) == 0 or intervals.mean() == 0:
        return 0.0
    
    # 使用线性回归计算趋势
    x = np.arange(len(intervals))
    try:
        slope, _, _, _, _ = linregress(x, intervals)
        return slope
    except:
        return 0.0

def compute_consistency(event_times):
    """计算一致性指标"""
    if len(event_times) < 2:
        return 0.0
    
    intervals = np.diff(event_times)
    if len(intervals) == 0 or np.mean(intervals) == 0:
        return 0.0
    
    # 间隔的标准差除以均值
    return np.std(intervals) / (np.mean(intervals) + 1e-10)

def compute_trend(event_times):
    """计算时间趋势"""
    if len(event_times) < 2:
        return 0.0
    
    x = np.arange(len(event_times))
    try:
        slope, _, _, _, _ = linregress(x, event_times)
        return slope
    except:
        return 0.0

def compute_behavior_trajectory(event_times, event_types):
    """计算行为轨迹特征（10维）"""
    features = []
    
    if len(event_times) < 2:
        return [0.0] * 10
    
    intervals = np.diff(event_times)
    
    # 1. improvement: 改进趋势
    x = np.arange(len(intervals))
    try:
        slope, _, _, _, _ = linregress(x, intervals)
        features.append(slope)
    except:
        features.append(0.0)
    
    # 2. consistency: 一致性
    if np.mean(intervals) > 0:
        features.append(np.std(intervals) / (np.mean(intervals) + 1e-10))
    else:
        features.append(0.0)
    
    # 3. trend: 整体趋势
    try:
        slope, _, _, _, _ = linregress(x, event_times)
        features.append(slope)
    except:
        features.append(0.0)
    
    # 4-10. 其他行为轨迹特征
    features.append(np.mean(intervals))  # 平均间隔
    features.append(np.std(intervals))   # 间隔标准差
    features.append(np.min(intervals))   # 最小间隔
    features.append(np.max(intervals))   # 最大间隔
    
    # 活跃时间段
    first_time = event_times[0]
    last_time = event_times[-1]
    duration = last_time - first_time
    features.append(duration / (len(event_times) + 1e-10))  # 每事件平均时长
    
    features.append(np.median(intervals))  # 中位数间隔
    
    # 间隔的四分位距
    if len(intervals) > 0:
        q75, q25 = np.percentile(intervals, [75, 25])
        features.append(q75 - q25)
    else:
        features.append(0.0)
    
    # 最后一个间隔与第一个间隔的比值
    if len(intervals) > 0 and intervals[0] > 0:
        features.append(intervals[-1] / (intervals[0] + 1e-10))
    else:
        features.append(0.0)
    
    return features[:10]

def compute_emotion_composite_features(student_df):
    """计算情绪复合特征（6维）"""
    features = []
    
    # 按题目分组计算比率
    edit_counts = student_df[student_df['eventType'] == 'text_insert'].groupby('exercise').size()
    delete_counts = student_df[student_df['eventType'] == 'text_remove'].groupby('exercise').size()
    focus_counts = student_df[student_df['eventType'] == 'focus_gained'].groupby('exercise').size()
    
    # 计算编辑比率
    edit_ratios = edit_counts / (edit_counts + delete_counts + 1e-10)
    features.append(edit_ratios.mean())   # edit_ratio_mean
    features.append(edit_ratios.std())    # edit_ratio_std
    
    # 计算删除比率
    delete_ratios = delete_counts / (edit_counts + delete_counts + 1e-10)
    features.append(delete_ratios.mean())  # delete_ratio_mean
    features.append(delete_ratios.std())   # delete_ratio_std
    
    # 计算焦点比率
    total_events = student_df.groupby('exercise').size()
    focus_ratios = focus_counts / (total_events + 1e-10)
    features.append(focus_ratios.mean())   # focus_ratio_mean
    features.append(focus_ratios.std())    # focus_ratio_std
    
    return features

def extract_features(student_df, student_id):
    """为单个学生提取46维特征"""
    features = {}
    
    # 事件类型列表
    event_types = ['text_insert', 'text_remove', 'text_paste', 
                  'focus_gained', 'focus_lost', 'run', 'submit']
    
    # ===== 1. 事件基础统计（28维）=====
    for et in event_types:
        et_events = student_df[student_df['eventType'] == et]['timestamp']
        
        # 转换时间戳为数值（秒）
        times = (et_events - et_events.min()).dt.total_seconds() if len(et_events) > 0 else np.array([0])
        
        if len(times) < 2:
            times = np.array([0, 0])
        
        # 均值
        features[f'{et}_mean'] = np.mean(times)
        # 标准差
        features[f'{et}_std'] = np.std(times)
        # 变异系数
        features[f'{et}_cv'] = np.std(times) / (np.mean(times) + 1e-10)
        # 香农熵（将时间分段计算）
        bins = min(10, max(1, len(times) // 10))
        if bins > 1:
            hist, _ = np.histogram(times, bins=bins)
            features[f'{et}_entropy'] = compute_shannon_entropy(hist)
        else:
            features[f'{et}_entropy'] = 0.0
    
    # ===== 2. 行为轨迹（10维）=====
    all_times = student_df['timestamp']
    all_times_numeric = (all_times - all_times.min()).dt.total_seconds().values
    
    trajectory_features = compute_behavior_trajectory(
        all_times_numeric, 
        student_df['eventType'].values
    )
    
    trajectory_names = ['improvement', 'consistency', 'trend', 'mean_interval',
                       'std_interval', 'min_interval', 'max_interval',
                       'duration_per_event', 'median_interval', 'iqr_interval']
    # Note: interval_ratio is computed but not returned (10-dim output)
    
    for i, name in enumerate(trajectory_names[:10]):
        features[f'trajectory_{name}'] = trajectory_features[i]
    
    # ===== 3. 情绪复合特征（6维）=====
    emotion_features = compute_emotion_composite_features(student_df)
    emotion_names = ['edit_ratio_mean', 'edit_ratio_std', 'delete_ratio_mean',
                     'delete_ratio_std', 'focus_ratio_mean', 'focus_ratio_std']
    
    for i, name in enumerate(emotion_names):
        features[f'emotion_{name}'] = emotion_features[i]
    
    # ===== 4. 元信息（2维）=====
    features['num_problems'] = student_df['exercise'].nunique()
    features['total_events'] = len(student_df)
    
    return features

def build_feature_matrix(ide_logs, passed):
    """构建特征矩阵"""
    print("Building feature matrix...")
    
    # 确保时间戳是datetime类型
    ide_logs['timestamp'] = pd.to_datetime(ide_logs['timestamp'])
    
    # 合并标签
    data = ide_logs.merge(passed, left_on='student', right_on='student')
    
    # 按学生提取特征
    feature_list = []
    labels = []
    student_ids = []
    
    for student_id, group in data.groupby('student'):
        features = extract_features(group, student_id)
        feature_list.append(features)
        labels.append(1 if group['passed'].iloc[0] == True or group['passed'].iloc[0] == 'True' else 0)
        student_ids.append(student_id)
    
    # 转换为DataFrame
    feature_df = pd.DataFrame(feature_list)
    
    # 处理无穷大和NaN
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.fillna(0)
    
    print(f"Feature matrix shape: {feature_df.shape}")
    print(f"Positive samples: {sum(labels)}, Negative samples: {len(labels) - sum(labels)}")
    
    return feature_df, np.array(labels), student_ids

if __name__ == "__main__":
    from data_processing import load_data
    
    ide_logs, passed = load_data(
        '/tmp/IDE_logs/IDE_logs.csv',
        '/tmp/IDE_logs/passed.csv'
    )
    
    X, y, student_ids = build_feature_matrix(ide_logs, passed)
    print(f"Features: {X.columns.tolist()}")
    print(f"Feature shape: {X.shape}")
