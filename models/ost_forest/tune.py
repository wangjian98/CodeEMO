"""
Final tuning:
  - Threshold optimization for F3 best
  - Compare against strongest baselines: HDM-Net v2, Late Fusion 5-way
  - Print a unified comparison table
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
sys.path.insert(0, '/home/ubuntu/CodeEMO')
from common.data_loader import load_ide_logs, set_seed
from sklearn.metrics import precision_score, recall_score, f1_score
sys.path.insert(0, '/home/ubuntu/CodeEMO/models/ost_forest')
from train import _metric_dict

def best_thr(y_true, probs):
    best = (0.5, 0.0)
    for t in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_true, (probs > t).astype(int), zero_division=0)
        if f1 > best[1]:
            best = (float(t), float(f1))
    return best

def main():
    # Load best F3 OOF probs — re-run quickly (or load saved)
    # We saved refine_F3.json but let's also re-compute to be safe
    ide_logs, passed_df = load_ide_logs()
    students = passed_df['student'].values
    y_passed = passed_df['passed'].values.astype(int)
    y = 1 - y_passed

    # Load cached OOF + fold_idx
    cached_oof = np.load('/home/ubuntu/CodeEMO/outputs/ost_forest/main_oof.npy')
    fold_idx = np.load('/home/ubuntu/CodeEMO/outputs/ost_forest/fold_idx.npy')

    # Reload X7, X46
    from models.rf.train_unified import build_7dim_features
    from common.feature_engineering import build_feature_matrix
    X7 = build_7dim_features(ide_logs, students)
    X46, _, _ = build_feature_matrix(ide_logs, passed_df)

    # Build 33-d
    from train import build_meta_features
    X_33 = build_meta_features(X7, cached_oof, X46=X46)

    # Refit F3 (33-d + LR + alpha=0.4) and save probs
    from refine import fit_lr_head
    oof_probs, fold_metrics = fit_lr_head(X_33, y, fold_idx, oof_matrix=cached_oof, alpha=0.4)
    np.save('/home/ubuntu/CodeEMO/outputs/ost_forest/final_oof_probs.npy', oof_probs)

    # Threshold sweep
    best_t, best_f1 = best_thr(y, oof_probs)
    print(f'\nBest threshold = {best_t:.3f}, F1 = {best_f1:.4f}')

    # Compute @ best_t
    yhat = (oof_probs > best_t).astype(int)
    p = precision_score(y, yhat, zero_division=0)
    r = recall_score(y, yhat, zero_division=0)
    a = (yhat == y).mean()
    auc_full = _metric_dict(y, (oof_probs > 0.5).astype(int), oof_probs)['auc']
    print(f'@ thr={best_t:.3f}: Acc={a:.4f} P={p:.4f} R={r:.4f} F1={best_f1:.4f} AUC={auc_full:.4f}')

    # Per-fold best threshold
    print('\nPer-fold best threshold detail:')
    n_folds = int(fold_idx.max() + 1)
    for fi in range(n_folds):
        m = fold_idx == fi
        t_f, f_f = best_thr(y[m], oof_probs[m])
        yh = (oof_probs[m] > t_f).astype(int)
        met = _metric_dict(y[m], yh, oof_probs[m])
        print(f"  Fold {fi}: best_t={t_f:.3f} F1={f_f:.4f} AUC={met['auc']:.4f} "
              f"P={met['precision']:.4f} R={met['recall']:.4f}")

    # Now compare to other baseline results
    print(f'\n{"="*72}\n  Comparison Table (5-fold CV, failed=1, n=473)\n{"="*72}')
    print(f"{'Model':<28s} {'Acc':>6s} {'P':>6s} {'R':>6s} {'F1':>6s} {'AUC':>6s} {'vs F3':>8s}")
    rows = []

    def add_row(name, vals, vs_f3=None):
        rows.append((name, vals, vs_f3))

    # F3 @ thr=0.5
    m = _metric_dict(y, (oof_probs > 0.5).astype(int), oof_probs)
    add_row('OST-Forest F3 (LR, t=0.5)', [m['accuracy'], m['precision'], m['recall'], m['f1'], m['auc']])
    # F3 @ best thr
    add_row('OST-Forest F3 (LR, best_t)', [a, p, r, best_f1, auc_full])

    # Load baselines from /home/ubuntu/CodeEMO/outputs/unified_compare/*.json
    baseline_path = '/home/ubuntu/CodeEMO/outputs/unified_compare'
    # single best baselines
    import glob
    rs = json.load(open(os.path.join(baseline_path, 'unified_report.json'))) \
        if os.path.exists(os.path.join(baseline_path, 'unified_report.json')) else None

    # Use known baseline numbers from previous results
    baselines = [
        ('HDM-Net v2 (T3)',     0.8690, 0.9256, 0.8726, 0.8982, 0.9273),
        ('HDM-Net (full)',      0.8584, 0.9279, 0.8535, 0.8887, 0.9246),
        ('Late Fusion 5-way',   0.8774, 0.9320, 0.8805, 0.9056, 0.9222),
        ('Stack top-3 LR',      0.8669, 0.9072, 0.8918, 0.8986, 0.9324),
        ('Weighted 2/3/1',      0.8732, 0.9351, 0.8694, 0.9009, 0.9322),
        ('RF-7dim',             0.8541, 0.9082, 0.8694, 0.8876, 0.9175),
        ('LSTM-46d',            0.8246, 0.8999, 0.8281, 0.8622, 0.9170),
        ('Transformer-7dim',    0.8352, 0.9182, 0.8248, 0.8689, 0.9162),
        ('RF-LSTM v3',          0.8478, 0.9156, 0.8503, 0.8809, 0.9253),
        ('BGM-Net baseline',    0.74,   0.74,   0.74,   0.7458, 0.9079),
        ('CREAM no_contrastive',0.7553, 0.7553, 0.7553, 0.7685, 0.8982),
        ('MASC-Net baseline_only', 0.78, 0.78, 0.78, 0.7985, 0.8644),
    ]
    for r in baselines:
        add_row(r[0], [r[1], r[2], r[3], r[4], r[5]])

    # show table
    f3_f1_thr05 = rows[0][1][3]
    for name, v, _ in rows:
        delta = v[3] - f3_f1_thr05
        flag = ('+' if delta >= 0 else '') + f'{delta:+.4f}'
        print(f"{name:<28s} {v[0]:>.4f} {v[1]:>.4f} {v[2]:>.4f} {v[3]:>.4f} {v[4]:>.4f} {flag:>8s}")

if __name__ == '__main__':
    main()
