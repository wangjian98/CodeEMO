"""
OST-Forest ablations (reuse cached H-Forest OOF to save time).

Ablations:
  A — drop AOOF (use only 13-d: 7-d handcrafted + 6-d session stat)
  B — drop self-distillation (alpha=0)
  C — replace G-Head LightGBM with Logistic Regression
  D — 20 RF -> 5 RF (heterogeneity saturation)
  E — 7-dim -> 46-dim handcrafted (feature dimensionality)
"""
import os, sys, json, argparse, warnings, time
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
from train import (build_meta_features, fit_h_forest, fit_g_head,
                   _metric_dict, _summarize,
                   DEFAULT_SEEDS, DEFAULT_DEPTHS, DEFAULT_N_RFS,
                   RF_N_ESTIMATORS, LGB_PARAMS)

ABLATION_CONFIG = {
    'A':  dict(name='no_AOOF',          use_aoof=False, alpha=0.4, lgb=True),
    'B':  dict(name='no_self_distill',  use_aoof=True,  alpha=0.0, lgb=True),
    'C':  dict(name='LR_head',          use_aoof=True,  alpha=0.4, lgb=False),
    'D':  dict(name='only5_RF',         use_aoof=True,  alpha=0.4, lgb=True, n_rfs=5),
    'E':  dict(name='use46d',           use_aoof=False, alpha=0.4, lgb=True, use_46d=True),
}


def run_ablation(abl_key, X7, X46, y, fold_idx,
                 cached_oof=None, n_rfs_main=20,
                 output_dir='/home/ubuntu/CodeEMO/outputs/ost_forest/ablation'):
    cfg = ABLATION_CONFIG[abl_key]
    set_seed(42)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*72}\n"
          f"  Ablation {abl_key}: {cfg['name']}\n"
          f"  use_aoof={cfg.get('use_aoof', True)}, alpha={cfg.get('alpha', 0.4)}, "
          f"lgb={cfg.get('lgb', True)}, n_rfs={cfg.get('n_rfs', n_rfs_main)}, "
          f"use_46d={cfg.get('use_46d', False)}"
          f"\n{'='*72}", flush=True)

    n_rfs = cfg.get('n_rfs', n_rfs_main)
    use_aoof = cfg.get('use_aoof', True)
    use_46d = cfg.get('use_46d', False)

    # Build OOF matrix
    if n_rfs == n_rfs_main and cached_oof is not None:
        oof_matrix = cached_oof
        print(f"  reusing cached OOF matrix ({oof_matrix.shape})", flush=True)
    else:
        X_for_forest = X46 if use_46d else X7
        depths = DEFAULT_DEPTHS[:n_rfs] if n_rfs <= len(DEFAULT_DEPTHS) \
            else (DEFAULT_DEPTHS * ((n_rfs // len(DEFAULT_DEPTHS)) + 1))[:n_rfs]
        seeds = DEFAULT_SEEDS[:n_rfs] if n_rfs <= len(DEFAULT_SEEDS) \
            else (DEFAULT_SEEDS * ((n_rfs // len(DEFAULT_SEEDS)) + 1))[:n_rfs]
        oof_matrix = fit_h_forest(
            X_for_forest, y, fold_idx, n_rfs=n_rfs,
            seeds=seeds, depths=depths,
        )

    # Build meta features
    X7_for_meta = X46 if use_46d else X7
    if use_aoof:
        X_meta = build_meta_features(X7_for_meta, oof_matrix, X46=X46)
    else:
        # only 7-d + 6-d session stat (no AOOF)
        X_meta = build_meta_features(X7_for_meta,
                                     np.zeros((len(y), 1), dtype=np.float64),
                                     X46=X46)  # zeros get filtered below? not really
        # actually: build 13-d by hand
        stat = np.zeros((X46.shape[0], 6), dtype=np.float32)
        stat[:, 0] = X46.mean(axis=1)
        stat[:, 1] = X46.std(axis=1)
        stat[:, 2] = X46.max(axis=1)
        stat[:, 3] = X46.min(axis=1)
        df = pd.DataFrame(X46)
        stat[:, 4] = df.skew(axis=1).values
        stat[:, 5] = df.kurtosis(axis=1).values
        X_meta = np.concatenate([X7_for_meta.astype(np.float32), stat], axis=1)
        # reset oof_matrix to None for non-AOOF runs
        oof_matrix = None

    print(f"  [M-Stack] meta features: {X_meta.shape}", flush=True)

    # Fit head
    if cfg.get('lgb', True):
        params = dict(LGB_PARAMS)
        oof_probs, fold_metrics = fit_g_head(
            X_meta, y, fold_idx, oof_matrix=oof_matrix,
            alpha=cfg.get('alpha', 0.4), params=params,
        )
    else:
        # Logistic regression with scaling
        n = len(y)
        oof_probs = np.zeros(n, dtype=np.float64)
        fold_metrics = []
        for fi in range(int(fold_idx.max() + 1)):
            va = fold_idx == fi
            tr = ~va
            sc = StandardScaler()
            Xtr = sc.fit_transform(X_meta[tr])
            Xva = sc.transform(X_meta[va])
            # for self-distillation in LR: do augmentation via duplication + weight
            if cfg.get('alpha', 0.4) > 0 and oof_matrix is not None:
                soft = oof_matrix[tr].mean(axis=1)
                soft_bin = (soft > 0.5).astype(int)
                X_aug = np.vstack([Xtr, Xtr])
                y_aug = np.hstack([y[tr], soft_bin])
                # use sample_weight
                lr = LogisticRegression(C=1.0, max_iter=200, n_jobs=-1,
                                         random_state=42)
                lr.fit(X_aug, y_aug,
                       sample_weight=np.hstack([
                           np.ones(len(y[tr])) * (1.0 - cfg.get('alpha', 0.4)),
                           np.ones(len(y[tr])) * cfg.get('alpha', 0.4)]))
            else:
                lr = LogisticRegression(C=1.0, max_iter=200, n_jobs=-1,
                                         random_state=42)
                lr.fit(Xtr, y[tr])
            probs_va = lr.predict_proba(Xva)[:, 1]
            oof_probs[va] = probs_va
            yhat = (probs_va > 0.5).astype(int)
            fold_metrics.append(_metric_dict(y[va], yhat, probs_va))

    summary = _summarize(fold_metrics)
    print(f"\n  Ablation {abl_key} ({cfg['name']}) summary:")
    for k in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        print(f"    {k:>10s}  {summary[f'{k}_mean']:.4f} ± {summary[f'{k}_std']:.4f}",
              flush=True)

    # Save
    out_path = os.path.join(output_dir, f'ablation_{abl_key}_{cfg["name"]}_results.json')
    with open(out_path, 'w') as f:
        json.dump({
            'ablation': cfg['name'],
            'cv_results': {k: summary[f'{k}_mean'] for k in
                           ['accuracy', 'precision', 'recall', 'f1', 'auc']},
            'cv_std': {k: summary[f'{k}_std'] for k in
                       ['accuracy', 'precision', 'recall', 'f1', 'auc']},
            'fold_details': fold_metrics,
        }, f, indent=2)
    return summary, oof_matrix


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=str, default='ABCDE',
                   help='Ablations to run (any subset of ABCDE)')
    p.add_argument('--n-rfs-main', type=int, default=20,
                   help='Number of RFs in main (cached)')
    args = p.parse_args()

    # Load data
    ide_logs, passed_df = load_ide_logs()
    students = passed_df['student'].values
    y_passed = passed_df['passed'].values.astype(int)
    y = 1 - y_passed
    X7 = build_7dim_features(ide_logs, students)
    X46, _, _ = build_feature_matrix(ide_logs, passed_df)

    set_seed(42)
    fold_idx = np.zeros(len(y), dtype=int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fi, (_, va) in enumerate(skf.split(X7, y)):
        fold_idx[va] = fi

    # Cache main OOF
    print("=== Computing main 20-RF OOF (cache for ablations A,B,C,D) ===", flush=True)
    cached_oof = fit_h_forest(X7, y, fold_idx, n_rfs=args.n_rfs_main)
    np.save('/home/ubuntu/CodeEMO/outputs/ost_forest/main_oof.npy', cached_oof)

    # Run each requested ablation
    results = {}
    for k in args.run.upper():
        if k not in ABLATION_CONFIG:
            print(f"  Skipping unknown ablation '{k}'", flush=True)
            continue
        s, _ = run_ablation(k, X7, X46, y, fold_idx,
                            cached_oof=cached_oof,
                            n_rfs_main=args.n_rfs_main)
        results[k] = (ABLATION_CONFIG[k]['name'],
                      {kk: s[f'{kk}_mean'] for kk in
                       ['accuracy', 'precision', 'recall', 'f1', 'auc']})

    print(f"\n{'='*72}\n  Ablation Summary\n{'='*72}")
    for k, (name, m) in results.items():
        print(f"  {k} {name:>20s}  "
              f"F1={m['f1']:.4f}  AUC={m['auc']:.4f}  "
              f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
              f"Acc={m['accuracy']:.4f}", flush=True)
    return results


if __name__ == '__main__':
    main()
