"""
Featuretools AutoML baseline - 与手工 46 维对比

原理:
  Featuretools 用 Deep Feature Synthesis (DFS) 自动从关系型数据（students / events
  两张表 + 主外键关系）生成特征。本质上是把人工写 ratio / aggregate 的过程
  自动化。

与 TSFRESH 的区别:
  - TSFRESH 处理单变量时间序列（per-event-type 的时间间隔序列）
  - Featuretools 处理多表关系数据（students + events 的跨表聚合）

输入输出:
  输入: students 表 (student_id, passed) + events 表 (event_id, student_id,
         timestamp, event_type, code_chars)
  输出: 学生级特征矩阵 (n_students × n_features) -> RF 分类器

用法:
    python models/automl/evaluate_featuretools.py
    python models/automl/evaluate_featuretools.py --folds 5 --max-depth 2
"""
import os
import sys
import json
import time
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, set_seed
from common.feature_engineering import build_feature_matrix
from common.evaluator import evaluate, summarize_fold_results, print_results_table
from models.automl.model import create_classifier, EVENT_TYPES


def build_featuretools_entityset(ide_logs, passed, max_events_per_student=5000):
    """构建 Featuretools 所需的 EntitySet（students + events 两张表）

    Args:
        ide_logs: 原始 IDE 日志 DataFrame
        passed: 标签 DataFrame
        max_events_per_student: 每名学生保留的最大事件数

    Returns:
        (es, students_df, events_df, student_ids): Featuretools EntitySet 和相关 DataFrame
    """
    import featuretools as ft

    print(f"  Building Featuretools EntitySet from {len(ide_logs)} events ...")

    ide_logs = ide_logs.copy()
    ide_logs['timestamp'] = pd.to_datetime(ide_logs['timestamp'])

    # 每名学生的事件截断（避免 Featuretools 在 28M 行上爆掉）
    ide_logs = ide_logs.sort_values(['student', 'timestamp'])
    ide_logs['_seq'] = ide_logs.groupby('student').cumcount()
    ide_logs_trim = ide_logs[ide_logs['_seq'] < max_events_per_student].copy()
    print(f"  After trim: {len(ide_logs_trim)} events")

    # 给每个事件分配唯一 ID
    ide_logs_trim = ide_logs_trim.reset_index(drop=True)
    ide_logs_trim['event_id'] = ide_logs_trim.index

    # === Students 表（target entity）===
    students_df = passed[['student', 'passed']].copy()
    students_df.columns = ['student_id', 'passed']
    students_df['passed'] = students_df['passed'].astype(int)

    # === Events 表（child entity）===
    events_df = ide_logs_trim[['event_id', 'student', 'timestamp', 'eventType']].copy()
    events_df.columns = ['event_id', 'student_id', 'timestamp', 'event_type']
    # 删除 student_id 缺失的行（防御性）
    events_df = events_df.dropna(subset=['student_id'])
    events_df['student_id'] = events_df['student_id'].astype(int)

    # 构建 EntitySet
    es = ft.EntitySet(id='codeemo')
    es.add_dataframe(
        dataframe=students_df,
        dataframe_name='students',
        index='student_id'
    )
    es.add_dataframe(
        dataframe=events_df,
        dataframe_name='events',
        index='event_id',
        time_index='timestamp'
    )
    es.add_relationship(
        ft.Relationship(es['students']['student_id'], es['events']['student_id'])
    )

    print(f"  EntitySet: students={len(students_df)}, events={len(events_df)}")
    print(f"  Relationships: students → events (via student_id)")

    return es, students_df, events_df


def run_dfs(es, max_depth=2):
    """运行 Deep Feature Synthesis (DFS)

    Args:
        es: Featuretools EntitySet
        max_depth: DFS 深度（1 = 仅当前实体聚合, 2 = 跨 1 张子表）

    Returns:
        (feature_matrix, feature_definitions)
    """
    import featuretools as ft

    default_agg_primitives = [
        'sum', 'mean', 'min', 'max', 'std', 'median',
        'count', 'mode', 'num_unique', 'skew'
    ]
    default_trans_primitives = [
        'day', 'month', 'weekday', 'hour', 'is_weekend'
    ]

    print(f"  Running DFS (max_depth={max_depth}) ...")
    print(f"  Agg primitives: {default_agg_primitives}")
    print(f"  Trans primitives: {default_trans_primitives}")

    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name='students',
        agg_primitives=default_agg_primitives,
        trans_primitives=default_trans_primitives,
        max_depth=max_depth,
        n_jobs=1,  # Featuretools 多线程不稳定，用 1 更安全
        verbose=False,
        features_only=False
    )

    # 删除全 NaN 列
    feature_matrix = feature_matrix.dropna(axis=1, how='all')
    # 填充剩余 NaN
    for col in feature_matrix.columns:
        if feature_matrix[col].dtype.kind in 'biufc':
            feature_matrix[col] = feature_matrix[col].fillna(feature_matrix[col].median())
        else:
            feature_matrix[col] = feature_matrix[col].fillna(0)

    print(f"  DFS output: {feature_matrix.shape} = ({len(feature_matrix)} students × "
          f"{len(feature_matrix.columns)} features)")

    return feature_matrix, feature_defs


def select_features_by_mutual_info(X_df, y, top_k=80, random_state=42):
    """用 mutual information 选择 top-k 特征（替代 FDR，因为 Featuretools 输出的混合类型）

    Args:
        X_df: 特征 DataFrame (n_samples, n_features)
        y: 标签
        top_k: 保留 top-k 特征

    Returns:
        pd.DataFrame: 选择后的特征
    """
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder

    # 处理非数值列
    X_encoded = X_df.copy()
    for col in X_encoded.columns:
        if X_encoded[col].dtype == object or X_encoded[col].dtype.name == 'category':
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

    X_arr = X_encoded.values

    print(f"  Selecting top-{top_k} features by mutual information ...")
    print(f"  Before selection: {X_arr.shape[1]} features")

    mi_scores = mutual_info_classif(X_arr, y, random_state=random_state)
    top_idx = np.argsort(mi_scores)[::-1][:min(top_k, len(mi_scores))]
    X_selected = X_df.iloc[:, top_idx]

    print(f"  After selection: {X_selected.shape[1]} features")
    print(f"  Top-5 features by MI:")
    for i, idx in enumerate(top_idx[:5]):
        print(f"    {i+1}. {X_df.columns[idx]} (MI={mi_scores[idx]:.4f})")

    return X_selected


def cv_train_evaluate(X, y, folds=5, n_estimators=200, max_depth=10, random_state=42):
    """5 折交叉验证训练 + 评估"""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = create_classifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_val_scaled)
        y_prob = model.predict_proba(X_val_scaled)[:, 1]

        metrics = evaluate(y_val, y_pred, y_prob)
        fold_results.append(metrics)

    return summarize_fold_results(fold_results)


def evaluate_featuretools_vs_handcrafted(
        folds=5, output_dir='outputs/automl_featuretools',
        max_events_per_student=5000, max_depth=2, top_k_features=80):
    """主评估流程"""
    set_seed(42)

    print("=" * 70)
    print("  Featuretools AutoML Baseline vs 手工 46 维")
    print("=" * 70)

    t_total = time.time()

    # === 加载数据 ===
    print("\n[Step 1] 加载 IDE 日志 ...")
    ide_logs, passed = load_ide_logs()

    # === 手工 46 维特征 ===
    print("\n[Step 2] 构建手工 46 维特征 ...")
    X_hand, y_hand, student_ids_hand = build_feature_matrix(ide_logs, passed)
    print(f"  Handcrafted: X.shape = {X_hand.shape}, y dist = {np.bincount(y_hand)}")

    # === Featuretools EntitySet ===
    print("\n[Step 3] 构建 Featuretools EntitySet ...")
    es, students_df, events_df = build_featuretools_entityset(
        ide_logs, passed, max_events_per_student=max_events_per_student
    )

    # === DFS 特征提取 ===
    print("\n[Step 4] 运行 DFS (Deep Feature Synthesis) ...")
    ft_matrix, ft_defs = run_dfs(es, max_depth=max_depth)

    # === 特征选择 ===
    print("\n[Step 5] Mutual Information 特征选择 ...")
    student_ids_ft = sorted(ft_matrix.index.tolist())
    y_ft = students_df.set_index('student_id').loc[student_ids_ft, 'passed'].values

    ft_selected = select_features_by_mutual_info(
        ft_matrix.loc[student_ids_ft], y_ft, top_k=top_k_features
    )

    # 对齐 common students
    common_students = sorted(set(student_ids_hand) & set(student_ids_ft))
    print(f"\n  Common students: {len(common_students)}")

    X_ft = ft_selected.loc[common_students].values
    common_idx = [student_ids_hand.index(sid) for sid in common_students]
    y_common = y_hand[common_idx]
    X_hand_common = X_hand[common_idx]

    n_common = len(common_students)

    # === 评估两组特征 ===
    print(f"\n[Step 6] {folds} 折交叉验证对比 ...")

    print("\n>>> 评估组 1: 手工 46 维 + RF")
    summary_hand = cv_train_evaluate(
        X_hand_common, y_common, folds=folds,
        n_estimators=100, max_depth=10
    )
    print_results_table("Handcrafted 46d + RF", summary_hand)

    print("\n>>> 评估组 2: Featuretools 自动特征 + RF")
    summary_ft = cv_train_evaluate(
        X_ft, y_common, folds=folds,
        n_estimators=200, max_depth=10
    )
    print_results_table(f"Featuretools (depth={max_depth}) + RF", summary_ft)

    # === 对比输出 ===
    print("\n" + "=" * 70)
    print("  对比结果汇总 (5-fold CV mean ± std)")
    print("=" * 70)
    comparison_rows = []
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        hand_m = summary_hand[f'{metric}_mean']
        hand_s = summary_hand[f'{metric}_std']
        ft_m = summary_ft[f'{metric}_mean']
        ft_s = summary_ft[f'{metric}_std']
        delta = ft_m - hand_m
        comparison_rows.append({
            'metric': metric.upper(),
            'handcrafted_46d_mean': hand_m,
            'handcrafted_46d_std': hand_s,
            'featuretools_mean': ft_m,
            'featuretools_std': ft_s,
            'delta_featuretools_minus_hand': delta
        })
        marker = "+" if delta > 0 else ("-" if delta < 0 else "=")
        print(f"  {metric.upper():<10}  "
              f"Handcrafted: {hand_m:.4f} ± {hand_s:.4f}   "
              f"Featuretools: {ft_m:.4f} ± {ft_s:.4f}   "
              f"d={delta:+.4f} {marker}")

    print("\n" + "=" * 70)
    print("  结论")
    print("=" * 70)
    f1_delta = summary_ft['f1_mean'] - summary_hand['f1_mean']
    if f1_delta > 0.005:
        verdict = "Featuretools 反超手工 (F1 > 0.5%)"
    elif f1_delta < -0.005:
        verdict = "手工特征显著优于 Featuretools (F1 > 0.5%)"
    else:
        verdict = "两组特征性能基本持平 (F1 < 0.5%)"
    print(f"  ★ {verdict}")
    print(f"  ★ 手工 46 维特征数: {X_hand_common.shape[1]}")
    print(f"  ★ Featuretools 特征数: 原始 {ft_matrix.shape[1]}, 选择后 {X_ft.shape[1]}")

    elapsed = time.time() - t_total
    print(f"\n  总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")

    # === 保存结果 ===
    os.makedirs(output_dir, exist_ok=True)
    eval_path = os.path.join(output_dir, 'evaluation.json')
    eval_result = {
        'experiment': 'AUTOML_FEATURETOOLS_vs_HANDCRAFTED',
        'hyperparameters': {
            'featuretools_max_depth': max_depth,
            'max_events_per_student': max_events_per_student,
            'top_k_features': top_k_features,
            'rf_handcrafted': {'n_estimators': 100, 'max_depth': 10},
            'rf_featuretools': {'n_estimators': 200, 'max_depth': 10},
            'random_state': 42,
            'n_folds': folds,
            'n_common_students': int(n_common)
        },
        'feature_counts': {
            'handcrafted_46d': int(X_hand_common.shape[1]),
            'featuretools_raw': int(ft_matrix.shape[1]),
            'featuretools_selected': int(X_ft.shape[1])
        },
        'handcrafted_46d': {k: v for k, v in summary_hand.items() if k != 'folds'},
        'featuretools': {k: v for k, v in summary_ft.items() if k != 'folds'},
        'comparison': comparison_rows,
        'verdict': verdict,
        'elapsed_seconds': elapsed,
        'sample_feature_definitions': [str(d) for d in ft_defs[:30]]
    }
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)

    print(f"\n评估结果已保存至: {eval_path}")

    return eval_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Featuretools AutoML baseline')
    parser.add_argument('--folds', type=int, default=5,
                        help='交叉验证折数 (默认: 5)')
    parser.add_argument('--output-dir', type=str, default='outputs/automl_featuretools',
                        help='结果输出目录')
    parser.add_argument('--max-events', type=int, default=5000,
                        help='每名学生保留的最大事件数 (默认: 5000)')
    parser.add_argument('--max-depth', type=int, default=2,
                        help='DFS 深度 (默认: 2)')
    parser.add_argument('--top-k', type=int, default=80,
                        help='Mutual Information top-k 特征数 (默认: 80)')
    args = parser.parse_args()

    evaluate_featuretools_vs_handcrafted(
        folds=args.folds,
        output_dir=args.output_dir,
        max_events_per_student=args.max_events,
        max_depth=args.max_depth,
        top_k_features=args.top_k
    )