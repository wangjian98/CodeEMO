"""
OST-Forest refinement runs after ablation analysis.

Insight: 33-d + LightGBM overfits; LR + 13-d (no AOOF) is better.
Try combinations:
  F1: 13-d + LR + alpha=0.4 (combine A and C)
  F2: 13-d + LR + no distillation (alpha=0)
  F3: 33-d + LR + alpha=0.4 (just swap LGBM->LR)
  F4: 33-d + LR + no distillation (alpha=0)
  F5: 13-d + LR + alpha=0.2 (smaller soft weight)
  F6: alpha sweep (0.0, 0.2, 0.4, 0.6) on best 13-d LR config
"""
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.path.insert(0, '/home/ubuntu/CodeEMO')
from common.data_loader import load_ide_logs, set_seed
from common.feature_engineering import build_feature_matrix
from models.rf.train_unified import build_7dim_features
sys.path.insert(0, '/home/ubuntu/CodeEMO/models/ost_forest')
from train import (build_meta_features, fit_h_forest, _metric_dict, _summarize,
                   DEFAULT_SEEDS, DEFAULT_DEPTHS, LGB_PARAMS)


def fit_lr_head(X_meta, y, fold_idx, oof_matrix=None, alpha=0.0,
                use_aoof=True, n_folds=5):
    n = len(y)
    oof_probs = np.zeros(n, dtype=np.float64)
    fold_metrics = []
    for fi in range(n_folds):
        va = fold_idx == fi
        tr = ~va
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_meta[tr])
        Xva = sc.transform(X_meta[va])
        if alpha > 0 and oof_matrix is not None:
            soft = oof_matrix[tr].mean(axis=1)
            soft_bin = (soft > 0.5).astype(int)
            X_aug = np.vstack([Xtr, Xtr])
            y_aug = np.hstack([y[tr], soft_bin])
            w_aug = np.hstack([np.ones(len(y[tr])) * (1.0 - alpha),
                               np.ones(len(y[tr])) * alpha])
            lr = LogisticRegression(C=1.0, max_iter=500, random_state=42)
            lr.fit(X_aug, y_aug, sample_weight=w_aug)
        else:
            lr = LogisticRegression(C=1.0, max_iter=500, random_state=42)
            lr.fit(Xtr, y[tr])
        probs_va = lr.predict_proba(Xva)[:, 1]
        oof_probs[va] = probs_va
        yhat = (probs_va > 0.5).astype(int)
        fold_metrics.append(_metric_dict(y[va], yhat, probs_va))
    return oof_probs, fold_metrics


def build_13d(X7, X46):
    stat = np.zeros((X46.shape[0], 6), dtype=np.float32)
    stat[:, 0] = X46.mean(axis=1)
    stat[:, 1] = X46.std(axis=1)
    stat[:, 2] = X46.max(axis=1)
    stat[:, 3] = X46.min(axis=1)
    df = pd.DataFrame(X46)
    stat[:, 4] = df.skew(axis=1).values
    stat[:, 5] = df.kurtosis(axis=1).values
    return np.concatenate([X7.astype(np.float32), stat], axis=1)


def main():
    ide_logs, passed_df = load_ide_logs()
    students = passed_df['student'].values
    y_passed = passed_df['passed'].values.astype(int)
    y = 1 - y_passed
    X7 = build_7dim_features(ide_logs, students)
    X46, _, _ = build_feature_matrix(ide_logs, passed_df)

    set_seed(42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_idx = np.zeros(len(y), dtype=int)
    for fi, (_, va) in enumerate(skf.split(X7, y)):
        fold_idx[va] = fi

    print("\n=== Reusing cached 20-RF OOF ===", flush=True)
    cached_oof = np.load('/home/ubuntu/CodeEMO/outputs/ost_forest/main_oof.npy')
    print(f"  OOF shape: {cached_oof.shape}, mean={cached_oof.mean():.4f}", flush=True)

    # 13-d
    X_13 = build_13d(X7, X46)
    print(f"\n13-d feature shape: {X_13.shape}", flush=True)
    # 33-d
    X_33 = build_meta_features(X7, cached_oof, X46=X46)
    print(f"33-d feature shape: {X_33.shape}", flush=True)

    print('\n' + '=' * 72 + '\n  Refinement Runs (LR + soft-label combinations)\n' + '=' * 72)

    configs = [
        ('F1', 13, True,  0.4),   # no AOOF + LR + alpha=0.4
        ('F2', 13, True,  0.0),   # no AOOF + LR + alpha=0
        ('F3', 33, True,  0.4),   # full + LR + alpha=0.4 (= Ablation C)
        ('F4', 33, True,  0.0),   # full + LR + alpha=0
        ('F5', 13, True,  0.2),   # no AOOF + LR + alpha=0.2
        ('F6', 13, True,  0.6),   # no AOOF + LR + alpha=0.6
    ]
    results = []
    for name, d, use_aoof, alpha in configs:
        X_meta = X_13 if d == 13 else X_33
        oof = cached_oof if use_aoof else None
        oof_probs, fm = fit_lr_head(X_meta, y, fold_idx, oof_matrix=oof,
                                    alpha=alpha)
        s = _summarize(fm)
        results.append((name, d, use_aoof, alpha, s))
        print(f"  {name}  d={d:2d}  aoof={use_aoof}  alpha={alpha:.1f}  "
              f"F1={s['f1_mean']:.4f}±{s['f1_std']:.4f}  "
              f"AUC={s['auc_mean']:.4f}±{s['auc_std']:.4f}  "
              f"P={s['precision_mean']:.4f}  R={s['recall_mean']:.4f}",
              flush=True)
        # save
        out = {
            'config': name, 'd': d, 'use_aoof': use_aoof, 'alpha': alpha,
            'cv_results': {k: s[f'{k}_mean'] for k in
                           ['accuracy', 'precision', 'recall', 'f1', 'auc']},
            'cv_std': {k: s[f'{k}_std'] for k in
                       ['accuracy', 'precision', 'recall', 'f1', 'auc']},
            'fold_details': fm,
        }
        with open(f'/home/ubuntu/CodeEMO/outputs/ost_forest/refine_{name}.json', 'w') as f:
            json.dump(out, f, indent=2)

    # Print summary
    print('\n' + '=' * 72 + '\n  Refinement Summary (sorted by F1)\n' + '=' * 72)
    for name, d, use_aoof, alpha, s in sorted(results, key=lambda x: -x[4]['f1_mean']):
        print(f"  {name}: F1={s['f1_mean']:.4f}±{s['f1_std']:.4f}  "
              f"AUC={s['auc_mean']:.4f}±{s['auc_std']:.4f}  "
              f"d={d} alpha={alpha}")

    # Best config
    best_name, best_d, best_aoof, best_alpha, best_s = max(results, key=lambda x: x[4]['f1_mean'])
    print(f'\n>>> BEST: {best_name} F1={best_s["f1_mean"]:.4f} AUC={best_s["auc_mean"]:.4f}')

    # Save best to outputs/ost_forest/results.json (overwriting main)
    np.save('/home/ubuntu/CodeEMO/outputs/ost_forest/refine_oof_probs.npy', np.zeros(1))  # placeholder
    return results


if __name__ == '__main__':
    main()
