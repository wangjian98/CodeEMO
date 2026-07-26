"""
OST-Forest: Out-of-fold Self-distilled Tree Forest
CodeEMO CS1 (n=473, failed=1)

Architecture (3 stages):
  1) H-Forest: 20 RF(7dim) with heterogeneous seed+depth → 5-fold OOF (20-d)
  2) M-Stack: 7-d handcrafted + 20-d OOF + 6-d session stat = 33-d
  3) G-Head:  LightGBM (max_depth=4, n_est=300, λ=1.0) + self-distillation soft label

Label convention: y=1 ↔ failed (consistent with unified_compare)
"""

import os
import sys
import json
import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
import lightgbm as lgb

warnings.filterwarnings("ignore")

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, set_seed
from common.feature_engineering import build_feature_matrix
from models.rf.train_unified import build_7dim_features


# ============================================================================
# Default configuration
# ============================================================================

# Stage-1: H-Forest — heterogeneous RF ensemble
DEFAULT_SEEDS = [42, 7, 123, 2024, 88, 1997, 314, 271, 161, 13,
                 100, 555, 1, 999, 333, 777, 2025, 666, 888, 17]
DEFAULT_DEPTHS = [3, 5, 7, 4, 6, 8, 10, 5, 7, 4,
                  9, 6, 12, 5, 8, 4, 11, 7, 5, 6]
DEFAULT_N_RFS = 20
RF_N_ESTIMATORS = 200

# Stage-3: G-Head LightGBM hyperparameters
LGB_PARAMS = dict(
    objective='binary',
    metric='binary_logloss',
    learning_rate=0.05,
    num_leaves=15,
    max_depth=4,
    n_estimators=300,
    reg_lambda=1.0,
    min_child_samples=20,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    verbosity=-1,
)

# Self-distillation
DEFAULT_SOFT_ALPHA = 0.4   # weight on soft-label samples
DEFAULT_FOLD = 5
DEFAULT_RANDOM_STATE = 42
DEFAULT_OUTPUT_DIR = 'outputs/ost_forest'


# ============================================================================
# Stage 1: H-Forest
# ============================================================================

def fit_h_forest(X, y, fold_idx, n_folds=DEFAULT_FOLD,
                 n_rfs=DEFAULT_N_RFS,
                 seeds=None, depths=None):
    """Train 20 heterogeneous RFs and produce (n, n_rfs) OOF probability matrix.

    fold_idx is reused across RFs so that all 20 models share the same fold split
    — this is critical: combining OOF probabilities from different splits would
    leak information.
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS
    if depths is None:
        depths = DEFAULT_DEPTHS
    n = len(y)
    oof_matrix = np.zeros((n, n_rfs), dtype=np.float64)
    t0 = time.time()
    for i in range(n_rfs):
        seed, depth = seeds[i], depths[i]
        for fi in range(n_folds):
            va = fold_idx == fi
            tr = ~va
            rf = RandomForestClassifier(
                n_estimators=RF_N_ESTIMATORS,
                max_depth=depth,
                random_state=seed,
                n_jobs=-1,
            )
            rf.fit(X[tr], y[tr])
            oof_matrix[va, i] = rf.predict_proba(X[va])[:, 1]
        elapsed = time.time() - t0
        eta = elapsed * (n_rfs - i - 1) / (i + 1) if i > 0 else 0
        print(f"  [H-Forest] RF {i+1:2d}/{n_rfs} (seed={seed:>5}, depth={depth:>2}) "
              f"| mean prob = {oof_matrix[:, i].mean():.4f} | "
              f"elapsed {elapsed:.1f}s, ETA {eta:.1f}s",
              flush=True)
    return oof_matrix


# ============================================================================
# Stage 2: M-Stack (feature concatenation)
# ============================================================================

def build_meta_features(X7, oof_matrix, X46=None):
    """Build 33-d meta features: [7-d handcrafted | 20-d OOF | 6-d session stat]."""
    if X46 is not None:
        # 6-d session statistics over 46-d raw features
        stat = np.zeros((X46.shape[0], 6), dtype=np.float32)
        stat[:, 0] = X46.mean(axis=1)
        stat[:, 1] = X46.std(axis=1)
        stat[:, 2] = X46.max(axis=1)
        stat[:, 3] = X46.min(axis=1)
        # skew, kurt via pandas (more robust)
        df = pd.DataFrame(X46)
        stat[:, 4] = df.skew(axis=1).values
        stat[:, 5] = df.kurtosis(axis=1).values
    else:
        stat = np.zeros((X7.shape[0], 6), dtype=np.float32)
    return np.concatenate([X7.astype(np.float32),
                           oof_matrix.astype(np.float32),
                           stat], axis=1)


# ============================================================================
# Stage 3: G-Head (LightGBM with optional self-distillation)
# ============================================================================

def fit_g_head(X_meta, y, fold_idx, oof_matrix=None, alpha=0.0,
               params=None, n_folds=DEFAULT_FOLD):
    """Train G-Head LightGBM with optional self-distillation.

    Self-distillation: duplicate samples; weight=(1-α) for hard, α for soft.
    Soft label for sample i = mean OOF probability across the 20 RFs of stage 1.
    """
    if params is None:
        params = LGB_PARAMS
    n = len(y)
    oof_probs = np.zeros(n, dtype=np.float64)
    fold_metrics = []

    for fi in range(n_folds):
        va = fold_idx == fi
        tr = ~va
        Xtr, Xva = X_meta[tr], X_meta[va]
        ytr, yva = y[tr], y[va]
        lgb_params = dict(params)
        n_est = lgb_params.pop('n_estimators', 300)

        if alpha > 0 and oof_matrix is not None:
            soft = oof_matrix[tr].mean(axis=1)
            # Duplicate samples: hard + soft, with respective weights
            X_aug = np.vstack([Xtr, Xtr])
            # LightGBM binary objective requires y in {0,1}; binarize soft
            soft_bin = (soft > 0.5).astype(int)
            y_aug = np.hstack([ytr, soft_bin])
            w_aug = np.hstack([np.ones(len(ytr)) * (1.0 - alpha),
                               np.ones(len(ytr)) * alpha])
        else:
            X_aug, y_aug, w_aug = Xtr, ytr, np.ones(len(ytr), dtype=np.float64)

        # Native API to accept sample_weight
        train_data = lgb.Dataset(X_aug, label=y_aug, weight=w_aug)
        booster = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=n_est,
        )
        probs_va = booster.predict(Xva)
        oof_probs[va] = probs_va
        yhat = (probs_va > 0.5).astype(int)
        m = _metric_dict(yva, yhat, probs_va)
        fold_metrics.append(m)

    return oof_probs, fold_metrics


# ============================================================================
# Metrics helpers
# ============================================================================

def _metric_dict(y_true, y_pred, y_prob):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_true, y_prob)),
    }


def _summarize(fold_metrics):
    keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    out = {'n_folds_used': len(fold_metrics)}
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        out[f'{k}_mean'] = float(np.mean(vals))
        out[f'{k}_std'] = float(np.std(vals))
    return out


# ============================================================================
# Main pipeline
# ============================================================================

def run_pipeline(X7, X46, y, n_rfs=DEFAULT_N_RFS,
                 alpha=DEFAULT_SOFT_ALPHA,
                 n_folds=DEFAULT_FOLD,
                 random_state=DEFAULT_RANDOM_STATE,
                 output_dir=None,
                 seeds=None, depths=None,
                 lgb_params=None,
                 save_artifacts=True):
    set_seed(random_state)
    if output_dir is None:
        output_dir = os.path.join(_PROJECT_ROOT, DEFAULT_OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*72}\n"
          f"  OST-Forest pipeline  |  n={len(y)}  failed_rate={y.mean():.4f}\n"
          f"  H-Forest: {n_rfs} RFs, alpha(soft)={alpha}\n"
          f"{'='*72}\n", flush=True)

    # ---- share fold split across stages ----
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_idx = np.zeros(len(y), dtype=int)
    for fi, (_, va) in enumerate(skf.split(X7, y)):
        fold_idx[va] = fi

    # ---- Stage 1: H-Forest ----
    t0 = time.time()
    oof_matrix = fit_h_forest(
        X7, y, fold_idx, n_folds=n_folds, n_rfs=n_rfs,
        seeds=seeds, depths=depths,
    )
    t1 = time.time()
    print(f"\n  [H-Forest] OOF matrix: {oof_matrix.shape}, mean={oof_matrix.mean():.4f}, "
          f"std={oof_matrix.std():.4f} | elapsed {t1-t0:.1f}s\n", flush=True)

    # ---- Stage 2: M-Stack ----
    X_meta = build_meta_features(X7, oof_matrix, X46=X46)
    print(f"  [M-Stack] meta features: {X_meta.shape}\n", flush=True)

    # ---- Stage 3: G-Head ----
    final_lgb_params = dict(LGB_PARAMS)
    if lgb_params:
        final_lgb_params.update(lgb_params)
    oof_probs, fold_metrics = fit_g_head(
        X_meta, y, fold_idx, oof_matrix=oof_matrix,
        alpha=alpha, params=final_lgb_params, n_folds=n_folds,
    )
    t2 = time.time()
    summary = _summarize(fold_metrics)
    summary['elapsed_seconds'] = t2 - t0
    summary['config'] = {
        'n_rfs': n_rfs,
        'alpha_soft': alpha,
        'lgb_params': {k: v for k, v in final_lgb_params.items()
                       if k in ('learning_rate', 'num_leaves', 'max_depth',
                                'reg_lambda', 'min_child_samples',
                                'subsample', 'colsample_bytree')},
        'rf_n_estimators': RF_N_ESTIMATORS,
    }

    print(f"\n{'='*72}\n  OST-Forest 5-fold CV summary\n{'='*72}")
    for k in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        print(f"  {k:>10s}  {summary[f'{k}_mean']:.4f} ± {summary[f'{k}_std']:.4f}")
    print(f"{'='*72}\n", flush=True)

    if save_artifacts:
        np.save(os.path.join(output_dir, 'oof_matrix.npy'), oof_matrix)
        np.save(os.path.join(output_dir, 'oof_probs.npy'), oof_probs)
        np.save(os.path.join(output_dir, 'fold_idx.npy'), fold_idx)
        np.save(os.path.join(output_dir, 'labels.npy'), y)
        results = {
            'model': 'OST-Forest',
            'n_samples': int(len(y)),
            'fail_rate': float(y.mean()),
            'cv_results': {k: summary[f'{k}_mean'] for k in
                           ['accuracy', 'precision', 'recall', 'f1', 'auc']},
            'cv_std': {k: summary[f'{k}_std'] for k in
                       ['accuracy', 'precision', 'recall', 'f1', 'auc']},
            'fold_details': fold_metrics,
            'elapsed_seconds': summary['elapsed_seconds'],
            'config': summary['config'],
        }
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  artifacts saved to {output_dir}\n", flush=True)
    return oof_probs, fold_metrics, summary, oof_matrix


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-rfs', type=int, default=DEFAULT_N_RFS)
    p.add_argument('--alpha', type=float, default=DEFAULT_SOFT_ALPHA)
    p.add_argument('--folds', type=int, default=DEFAULT_FOLD)
    p.add_argument('--seed', type=int, default=DEFAULT_RANDOM_STATE)
    p.add_argument('--output-dir', type=str, default=None)
    args = p.parse_args()

    ide_logs, passed_df = load_ide_logs()
    student_ids = passed_df['student'].values
    y_passed = passed_df['passed'].values.astype(int)
    y_eval = 1 - y_passed  # failed=1

    X7 = build_7dim_features(ide_logs, student_ids)
    X46_full, _, _ = build_feature_matrix(ide_logs, passed_df)
    n = len(y_eval)
    X46 = X46_full  # full 46-d for stat

    print(f"n={n}, failed={y_eval.sum()}, passed={n - y_eval.sum()}")
    print(f"X7={X7.shape}, X46={X46.shape}")

    run_pipeline(X7, X46, y_eval,
                 n_rfs=args.n_rfs, alpha=args.alpha,
                 n_folds=args.folds, random_state=args.seed,
                 output_dir=args.output_dir)


if __name__ == '__main__':
    main()
