"""
AUTOML 模型评估脚本 (TSFRESH vs 手工 46 维对比)

对比两组特征在相同评估框架下的表现:
  - AUTOML (TSFRESH 自动提取的特征 + RF)
  - Handcrafted (手工 46 维特征 + RF，与 models/rf/train.py 同等设置)

输出 4 项核心指标: Accuracy / Precision / Recall / F1（以及 AUC）
便于论文 Section 4 中直接作为 Table 2-bis 引用。

用法:
    python models/automl/evaluate.py
    python models/automl/evaluate.py --folds 5 --fc minimal
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
from models.automl.model import (
    create_classifier,
    build_long_format,
    extract_tsfresh_features,
    select_features_by_target,
    EVENT_TYPES
)
from models.automl.train import get_fc_parameters


def cv_train_evaluate(X, y, folds=5, n_estimators=200, max_depth=10, random_state=42):
    """通用的 5 折交叉验证训练 + 评估（RF）

    Args:
        X: 特征矩阵 (n_samples, n_features)
        y: 标签
        folds: 折数
        n_estimators: 树数量
        max_depth: 最大深度
        random_state: 随机种子

    Returns:
        summary: dict with metric_mean/metric_std for accuracy, precision,
                  recall, f1, auc
    """
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


def evaluate_comparison(folds=5, output_dir='outputs/automl', fc_name='minimal',
                         max_events_per_student=5000, n_jobs=8, fdr_level=0.05):
    """评估并对比 TSFRESH 与手工 46 维特征

    Args:
        folds: 交叉验证折数
        output_dir: 结果输出目录
        fc_name: TSFRESH 特征提取参数集
        max_events_per_student: 每名学生保留的最大事件数
        n_jobs: 并行核数
        fdr_level: FDR 校正水平

    Returns:
        dict: 对比结果
    """
    set_seed(42)

    print("=" * 70)
    print("  AUTOML (TSFRESH) vs 手工 46 维 - 评估对比")
    print("=" * 70)

    # === 加载数据 ===
    print("\n[Step 1] 加载 IDE 日志 ...")
    ide_logs, passed = load_ide_logs()
    passed_dict = dict(zip(passed['student'], passed['passed'].astype(int)))

    # === 手工 46 维特征 ===
    print("\n[Step 2] 构建手工 46 维特征 ...")
    X_hand, y_hand, student_ids = build_feature_matrix(ide_logs, passed)
    print(f"  Handcrafted: X.shape = {X_hand.shape}, y dist = {np.bincount(y_hand)}")

    # === TSFRESH 特征 ===
    print("\n[Step 3] 构建 TSFRESH long format ...")
    long_df = build_long_format(ide_logs, max_events_per_student=max_events_per_student)

    print("\n[Step 4] 提取 TSFRESH 特征 ...")
    fc_params = get_fc_parameters(fc_name)
    features_all = extract_tsfresh_features(
        long_df, n_jobs=n_jobs, kind_to_fc_params=fc_params
    )

    print("\n[Step 5] FDR 特征选择 ...")
    student_ids_sorted = sorted(features_all.index.tolist())
    y_ts = np.array([passed_dict.get(sid, 0) for sid in student_ids_sorted])
    X_ts_all = features_all.loc[student_ids_sorted].values

    # 确保与手工特征用相同的学生子集
    common_students = sorted(set(student_ids) & set(student_ids_sorted))
    print(f"  Common students (handcrafted ∩ TSFRESH): {len(common_students)}")

    features_selected = select_features_by_target(
        pd.DataFrame(X_ts_all, index=student_ids_sorted), y_ts, fdr_level=fdr_level
    )
    X_ts = features_selected.loc[common_students].values

    # 对齐 y
    common_idx = [student_ids.index(sid) for sid in common_students]
    y_common = y_hand[common_idx]
    y_common_ts = np.array([y_ts[student_ids_sorted.index(sid)] for sid in common_students])
    assert np.array_equal(y_common, y_common_ts), "y mismatch between feature sets!"

    # 手工特征也对齐到 common students
    X_hand_common = X_hand[common_idx]
    n_common = len(common_students)
    print(f"\n  对齐后样本数: {n_common}")

    # === 评估两组特征 ===
    print(f"\n[Step 6] {folds} 折交叉验证对比 ...")

    print("\n>>> 评估组 1: 手工 46 维特征 + RF")
    summary_hand = cv_train_evaluate(
        X_hand_common, y_common, folds=folds,
        n_estimators=100, max_depth=10  # 与 models/rf 完全一致
    )
    print_results_table("Handcrafted 46d + RF", summary_hand)

    print("\n>>> 评估组 2: TSFRESH 自动特征 + RF")
    summary_ts = cv_train_evaluate(
        X_ts, y_common, folds=folds,
        n_estimators=200, max_depth=10
    )
    print_results_table(f"TSFRESH ({fc_name}) + RF", summary_ts)

    # === 对比表格输出 ===
    print("\n" + "=" * 70)
    print("  对比结果汇总 (5-fold CV mean ± std)")
    print("=" * 70)
    comparison_rows = []
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        hand_m = summary_hand[f'{metric}_mean']
        hand_s = summary_hand[f'{metric}_std']
        ts_m = summary_ts[f'{metric}_mean']
        ts_s = summary_ts[f'{metric}_std']
        delta = ts_m - hand_m
        comparison_rows.append({
            'metric': metric.upper(),
            'handcrafted_46d_mean': hand_m,
            'handcrafted_46d_std': hand_s,
            'tsfresh_mean': ts_m,
            'tsfresh_std': ts_s,
            'delta_tsfresh_minus_hand': delta
        })
        marker = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"  {metric.upper():<10}  "
              f"Handcrafted: {hand_m:.4f} ± {hand_s:.4f}   "
              f"TSFRESH: {ts_m:.4f} ± {ts_s:.4f}   "
              f"Δ={delta:+.4f} {marker}")

    # 总结赢家
    print("\n" + "=" * 70)
    print("  结论")
    print("=" * 70)
    f1_delta = summary_ts['f1_mean'] - summary_hand['f1_mean']
    if f1_delta > 0.005:
        verdict = "TSFRESH 反超手工特征 (F1 高出 >0.5%)"
    elif f1_delta < -0.005:
        verdict = "手工特征显著优于 TSFRESH (F1 高出 >0.5%)"
    else:
        verdict = "两组特征性能基本持平 (F1 差距 <0.5%)"
    print(f"  ★ {verdict}")
    print(f"  ★ 手工 46 维特征数: {X_hand_common.shape[1]}")
    print(f"  ★ TSFRESH 特征数:  原始 {features_all.shape[1]}, "
          f"选择后 {X_ts.shape[1]}")

    # === 保存结果 ===
    os.makedirs(output_dir, exist_ok=True)
    eval_path = os.path.join(output_dir, 'evaluation.json')
    eval_result = {
        'experiment': 'AUTOML_TSFRESH_vs_HANDCRAFTED',
        'hyperparameters': {
            'tsfresh_fc_parameters': fc_name,
            'max_events_per_student': max_events_per_student,
            'fdr_level': fdr_level,
            'rf_handcrafted': {'n_estimators': 100, 'max_depth': 10},
            'rf_tsfresh': {'n_estimators': 200, 'max_depth': 10},
            'random_state': 42,
            'n_folds': folds,
            'n_common_students': int(n_common)
        },
        'feature_counts': {
            'handcrafted_46d': int(X_hand_common.shape[1]),
            'tsfresh_raw': int(features_all.shape[1]),
            'tsfresh_selected': int(X_ts.shape[1])
        },
        'handcrafted_46d': {k: v for k, v in summary_hand.items() if k != 'folds'},
        'tsfresh': {k: v for k, v in summary_ts.items() if k != 'folds'},
        'comparison': comparison_rows,
        'verdict': verdict
    }
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)

    print(f"\n评估结果已保存至: {eval_path}")

    return eval_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AUTOML (TSFRESH) vs 手工 46 维评估')
    parser.add_argument('--folds', type=int, default=5,
                        help='交叉验证折数 (默认: 5)')
    parser.add_argument('--output-dir', type=str, default='outputs/automl',
                        help='结果输出目录 (默认: outputs/automl)')
    parser.add_argument('--fc', type=str, default='minimal',
                        choices=['minimal', 'efficient', 'comprehensive'],
                        help='TSFRESH 特征提取参数集 (默认: minimal)')
    parser.add_argument('--max-events', type=int, default=5000,
                        help='每名学生保留的最大事件数 (默认: 5000)')
    parser.add_argument('--n-jobs', type=int, default=8,
                        help='并行核数 (默认: 8)')
    parser.add_argument('--fdr-level', type=float, default=0.05,
                        help='FDR 校正水平 (默认: 0.05)')
    args = parser.parse_args()

    evaluate_comparison(
        folds=args.folds,
        output_dir=args.output_dir,
        fc_name=args.fc,
        max_events_per_student=args.max_events,
        n_jobs=args.n_jobs,
        fdr_level=args.fdr_level
    )