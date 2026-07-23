"""
AutoML baseline using autofeat (Horn et al., 2020) - algebraic feature engineering.

autofeat takes an existing feature matrix and automatically generates
non-linear combinations (log, sqrt, +, -, *, /) and selects informative
features via Lasso.

This is the "third AutoML tool" to complement:
  - TSFRESH (time-series feature extraction)
  - autofeat (algebraic transformations)

Note: Featuretools (DFS) was dropped because woodwork 0.31 has API bugs
that prevent proper initialization in the current environment.

Usage:
    python models/automl/evaluate_autofeat.py
    python models/automl/evaluate_autofeat.py --folds 5 --feateng-steps 2
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
from models.automl.model import create_classifier


def run_autofeat(X, y, feateng_steps=2, n_jobs=8):
    """使用 autofeat 在已有特征矩阵上做代数变换特征工程

    Args:
        X: 已有特征矩阵 (n_samples, n_features)
        y: 标签
        feateng_steps: 变换阶数 (1=only unary, 2=binary combos)
        n_jobs: 并行核数

    Returns:
        X_new: 变换后的特征矩阵
        new_feature_names: 列表，新生成的特征名
    """
    from autofeat import AutoFeatClassifier

    print(f"  Running autofeat with feateng_steps={feateng_steps}, n_jobs={n_jobs}")
    print(f"  Input shape: {X.shape}")

    af = AutoFeatClassifier(
        feateng_steps=feateng_steps,
        n_jobs=n_jobs,
        verbose=1
    )

    t0 = time.time()
    X_new = af.fit_transform(X, y)
    elapsed = time.time() - t0

    n_new_features = len(af.new_feat_cols_) if hasattr(af, 'new_feat_cols_') else X_new.shape[1] - X.shape[1]

    print(f"  Output shape: {X_new.shape}")
    print(f"  Added {n_new_features} engineered features in {elapsed:.1f}s")

    if hasattr(af, 'new_feat_cols_'):
        print(f"  Sample new features (first 10):")
        for f in af.new_feat_cols_[:10]:
            print(f"    {f}")

    return X_new, getattr(af, 'new_feat_cols_', [])


def cv_train_evaluate(X, y, folds=5, n_estimators=200, max_depth=10, random_state=42):

    if hasattr(X, "values"):
        X = X.values
    if hasattr(X, "columns"):
        X = np.asarray(X, dtype=float)
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


def evaluate_autofeat_vs_handcrafted(
        folds=5, output_dir='outputs/automl_autofeat',
        feateng_steps=2, n_jobs=8):
    """主评估流程"""
    set_seed(42)

    print("=" * 70)
    print("  autofeat AutoML Baseline vs 手工 46 维")
    print("=" * 70)

    t_total = time.time()

    # === 加载数据 ===
    print("\n[Step 1] 加载 IDE 日志 ...")
    ide_logs, passed = load_ide_logs()

    # === 手工 46 维特征 ===
    print("\n[Step 2] 构建手工 46 维特征 ...")
    X_hand, y_hand, student_ids = build_feature_matrix(ide_logs, passed)
    print(f"  Handcrafted: X.shape = {X_hand.shape}, y dist = {np.bincount(y_hand)}")

    # === autofeat 特征工程 ===
    print("\n[Step 3] 运行 autofeat 特征工程 ...")
    # autofeat 需要处理 NaN/Inf
    X_hand_clean = np.nan_to_num(X_hand, nan=0.0, posinf=0.0, neginf=0.0)
    X_auto, new_feats = run_autofeat(
        X_hand_clean, y_hand, feateng_steps=feateng_steps, n_jobs=n_jobs
    )

    # === 评估两组特征 ===
    print(f"\n[Step 4] {folds} 折交叉验证对比 ...")

    print("\n>>> 评估组 1: 手工 46 维 + RF")
    summary_hand = cv_train_evaluate(
        X_hand, y_hand, folds=folds,
        n_estimators=100, max_depth=10
    )
    print_results_table("Handcrafted 46d + RF", summary_hand)

    print("\n>>> 评估组 2: autofeat 自动特征 + RF")
    summary_auto = cv_train_evaluate(
        X_auto, y_hand, folds=folds,
        n_estimators=200, max_depth=10
    )
    print_results_table(f"autofeat (steps={feateng_steps}) + RF", summary_auto)

    # === 对比输出 ===
    print("\n" + "=" * 70)
    print("  对比结果汇总 (5-fold CV mean ± std)")
    print("=" * 70)
    comparison_rows = []
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        hand_m = summary_hand[f'{metric}_mean']
        hand_s = summary_hand[f'{metric}_std']
        auto_m = summary_auto[f'{metric}_mean']
        auto_s = summary_auto[f'{metric}_std']
        delta = auto_m - hand_m
        comparison_rows.append({
            'metric': metric.upper(),
            'handcrafted_46d_mean': hand_m,
            'handcrafted_46d_std': hand_s,
            'autofeat_mean': auto_m,
            'autofeat_std': auto_s,
            'delta_autofeat_minus_hand': delta
        })
        marker = "+" if delta > 0 else ("-" if delta < 0 else "=")
        print(f"  {metric.upper():<10}  "
              f"Handcrafted: {hand_m:.4f} ± {hand_s:.4f}   "
              f"autofeat: {auto_m:.4f} ± {auto_s:.4f}   "
              f"d={delta:+.4f} {marker}")

    print("\n" + "=" * 70)
    print("  结论")
    print("=" * 70)
    f1_delta = summary_auto['f1_mean'] - summary_hand['f1_mean']
    if f1_delta > 0.005:
        verdict = "autofeat 反超手工 (F1 > 0.5%)"
    elif f1_delta < -0.005:
        verdict = "手工特征显著优于 autofeat (F1 > 0.5%)"
    else:
        verdict = "两组特征性能基本持平 (F1 < 0.5%)"
    print(f"  ★ {verdict}")
    print(f"  ★ 手工 46 维特征数: {X_hand.shape[1]}")
    print(f"  ★ autofeat 特征数: 原始 {X_hand.shape[1]}, 增强后 {X_auto.shape[1]} (+{X_auto.shape[1] - X_hand.shape[1]})")

    elapsed = time.time() - t_total
    print(f"\n  总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")

    # === 保存结果 ===
    os.makedirs(output_dir, exist_ok=True)
    eval_path = os.path.join(output_dir, 'evaluation.json')
    eval_result = {
        'experiment': 'AUTOML_AUTOFEAT_vs_HANDCRAFTED',
        'hyperparameters': {
            'autofeat_feateng_steps': feateng_steps,
            'n_jobs': n_jobs,
            'rf_handcrafted': {'n_estimators': 100, 'max_depth': 10},
            'rf_autofeat': {'n_estimators': 200, 'max_depth': 10},
            'random_state': 42,
            'n_folds': folds,
        },
        'feature_counts': {
            'handcrafted_46d': int(X_hand.shape[1]),
            'autofeat_input': int(X_hand.shape[1]),
            'autofeat_output': int(X_auto.shape[1]),
            'autofeat_added': int(X_auto.shape[1] - X_hand.shape[1]),
        },
        'handcrafted_46d': {k: v for k, v in summary_hand.items() if k != 'folds'},
        'autofeat': {k: v for k, v in summary_auto.items() if k != 'folds'},
        'comparison': comparison_rows,
        'verdict': verdict,
        'elapsed_seconds': elapsed,
        'sample_new_features': new_feats[:30] if new_feats else []
    }
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)

    print(f"\n评估结果已保存至: {eval_path}")

    return eval_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='autofeat AutoML baseline')
    parser.add_argument('--folds', type=int, default=5,
                        help='交叉验证折数 (默认: 5)')
    parser.add_argument('--output-dir', type=str, default='outputs/automl_autofeat',
                        help='结果输出目录')
    parser.add_argument('--feateng-steps', type=int, default=2,
                        help='autofeat 特征工程阶数 (默认: 2)')
    parser.add_argument('--n-jobs', type=int, default=8,
                        help='并行核数 (默认: 8)')
    args = parser.parse_args()

    evaluate_autofeat_vs_handcrafted(
        folds=args.folds,
        output_dir=args.output_dir,
        feateng_steps=args.feateng_steps,
        n_jobs=args.n_jobs
    )