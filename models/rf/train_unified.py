"""
Random Forest 统一训练脚本 - 输出到 unified_compare 格式

支持两种特征维度：7-dim（事件计数）/ 46-dim（手工特征）
统一输出 failed=1 口径的 probs/labels/fold_idx/results.json

用法:
    python models/rf/train_unified.py --features 7d
    python models/rf/train_unified.py --features 46d
    python models/rf/train_unified.py --features 46d --label-conv passed1   # 不推荐，已默认 failed=1

数据约定:
    y=1 表示通过(passed), y=0 表示未通过(failed)。本脚本默认按 failed=1 输出
    所有 probs / labels / metrics，即 y=1 ↔ P(failed)=1 ↔ predicted failed。
"""
import os
import sys
import json
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)

warnings.filterwarnings("ignore")

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, set_seed
from common.feature_engineering import build_feature_matrix
from common.evaluator import evaluate, summarize_fold_results, print_results_table


EVENT_TYPES = [
    'text_insert', 'text_remove', 'text_paste',
    'focus_gained', 'focus_lost', 'run', 'submit'
]


def build_7dim_features(ide_logs: pd.DataFrame, students: np.ndarray) -> np.ndarray:
    """7-dim feature: each student's count per event type."""
    counts = (ide_logs
              .groupby(['student', 'eventType'])
              .size()
              .unstack(fill_value=0))
    counts = counts.reindex(columns=EVENT_TYPES, fill_value=0)
    counts = counts.reindex(index=students, fill_value=0)
    return counts.values.astype(np.float32)


def metric_dict(y_true, y_pred, y_prob):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_true, y_prob)),
    }


def per_fold(probs, labels, fold_idx, n_folds=5, thr=0.5):
    out = []
    for fi in range(n_folds):
        m = fold_idx == fi
        if m.sum() == 0:
            continue
        yi = labels[m]
        pi = probs[m]
        yh = (pi > thr).astype(int)
        out.append(metric_dict(yi, yh, pi))
    return out


def summary(metrics):
    keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    out = {'n_folds_used': len(metrics)}
    for k in keys:
        vals = [m[k] for m in metrics]
        out[f'{k}_mean'] = float(np.mean(vals))
        out[f'{k}_std'] = float(np.std(vals))
    return out


def train_unified(features='46d', label_conv='failed1', folds=5,
                  output_dir=None, random_state=42):
    """统一训练入口."""
    assert features in ('7d', '46d')
    assert label_conv in ('failed1', 'passed1')

    suffix = '7dim' if features == '7d' else '46d'
    if output_dir is None:
        output_dir = os.path.join(_PROJECT_ROOT, 'outputs', 'unified_compare',
                                   f'rf_{suffix}')
    os.makedirs(output_dir, exist_ok=True)

    set_seed(random_state)

    print("=" * 72)
    print(f"  Random Forest — features={features}, label_conv={label_conv}")
    print("=" * 72)

    # ---------------- load ----------------
    ide_logs, passed_df = load_ide_logs()
    student_ids = passed_df['student'].values
    y_passed = passed_df['passed'].values.astype(int)
    n = len(student_ids)
    print(f"n_students = {n}, passed (y=1) = {y_passed.sum()}, failed = {n - y_passed.sum()}")

    if features == '7d':
        X = build_7dim_features(ide_logs, student_ids)
    else:
        X_full, _, _ = build_feature_matrix(ide_logs, passed_df)
        X = X_full
    print(f"feature_dim = {X.shape[1]}")

    # ---------------- label conversion ----------------
    if label_conv == 'failed1':
        y_eval = 1 - y_passed  # y_eval = 1 iff failed
    else:
        y_eval = y_passed.copy()

    # ---------------- 5-fold CV ----------------
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    # split on y_eval so folds are stratified by the eval label
    fold_idx = np.zeros(n, dtype=int)
    all_probs = np.zeros(n, dtype=np.float64)
    fold_metrics = []

    for fi, (tr, va) in enumerate(skf.split(X, y_eval)):
        fold_idx[va] = fi
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xva = scaler.transform(X[va])
        clf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                      random_state=random_state, n_jobs=-1)
        clf.fit(Xtr, y_eval[tr])
        probs_va = clf.predict_proba(Xva)[:, 1]
        all_probs[va] = probs_va
        yhat = (probs_va > 0.5).astype(int)
        m = metric_dict(y_eval[va], yhat, probs_va)
        fold_metrics.append(m)
        print(f"  Fold {fi + 1}/{folds}  "
              f"Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  AUC={m['auc']:.4f}")

    cv = summary(fold_metrics)
    print_results_table(f"Random Forest ({features}, label={label_conv})", {
        **{k: cv[k] for k in cv if k.endswith('_mean') or k.endswith('_std')},
        'folds': fold_metrics
    })

    # ---------------- save artifacts ----------------
    np.save(os.path.join(output_dir, 'probs.npy'), all_probs)
    np.save(os.path.join(output_dir, 'labels.npy'), y_eval.astype(np.int8))
    np.save(os.path.join(output_dir, 'fold_idx.npy'), fold_idx)

    # ---------------- save results.json (unified_compare shape) ----------------
    cfg = {
        'model': 'RandomForest',
        'features': features,
        'feature_dim': int(X.shape[1]),
        'hyperparameters': {'n_estimators': 200, 'max_depth': 12,
                            'random_state': random_state, 'n_jobs': -1},
        'label_convention': 'y=1=failed' if label_conv == 'failed1' else 'y=1=passed',
        'n_folds': folds,
    }
    out = {**cfg,
           'cv_results': {k: cv[k] for k in cv if k != 'n_folds_used'},
           'fold_details': fold_metrics,
           'n_samples': n,
           'n_failed': int(y_eval.sum()),
           'n_passed': int(n - y_eval.sum())}
    with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {output_dir}")
    print(f"  probs.npy  labels.npy  fold_idx.npy  results.json")
    print(f"\nUnified comparison summary:")
    print(f"  F1 = {cv['f1_mean']:.4f} ± {cv['f1_std']:.4f}")
    print(f"  AUC = {cv['auc_mean']:.4f} ± {cv['auc_std']:.4f}")
    return cv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Forest unified trainer')
    parser.add_argument('--features', choices=['7d', '46d'], default='46d',
                        help='Feature dimensionality (default 46d)')
    parser.add_argument('--label-conv', choices=['failed1', 'passed1'],
                        default='failed1', help='Label convention (default failed1)')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train_unified(features=args.features, label_conv=args.label_conv,
                  folds=args.folds, output_dir=args.output_dir,
                  random_state=args.seed)
