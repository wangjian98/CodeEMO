"""
SHAP Analysis for RF-7dim on CS1 dataset.
"""
import os, sys, json, warnings, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)

warnings.filterwarnings("ignore")

import shap

PROJECT_ROOT = "/home/ubuntu/CodeEMO"
sys.path.insert(0, PROJECT_ROOT)
from common.data_loader import load_ide_logs, set_seed

EVENT_TYPES = [
    'text_insert', 'text_remove', 'text_paste',
    'focus_gained', 'focus_lost', 'run', 'submit'
]

OUTDIR = os.path.join(PROJECT_ROOT, "results", "random_forest", "shap")
os.makedirs(OUTDIR, exist_ok=True)


def build_7dim_features(ide_logs, students):
    counts = (ide_logs
              .groupby(['student', 'eventType'])
              .size()
              .unstack(fill_value=0))
    counts = counts.reindex(columns=EVENT_TYPES, fill_value=0)
    counts = counts.reindex(index=students, fill_value=0)
    return counts.values.astype(np.float32)


set_seed(42)
print("Loading data...")
ide_logs, passed_df = load_ide_logs()
student_ids = passed_df['student'].values
y_passed = passed_df['passed'].values.astype(int)
y_eval = 1 - y_passed
n = len(student_ids)
print(f"n_students = {n}, failed = {y_eval.sum()}, passed = {n - y_eval.sum()}")

X = build_7dim_features(ide_logs, student_ids)
print(f"X shape: {X.shape}")

print("\n[1/4] Running 5-fold CV + per-fold SHAP ...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_X, all_y, all_proba, all_shap = [], [], [], []
all_expected_per_fold = []
fold_metrics = []

t0 = time.time()
for fi, (tr, va) in enumerate(skf.split(X, y_eval)):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[tr])
    Xva = scaler.transform(X[va])
    clf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                  random_state=42, n_jobs=-1)
    clf.fit(Xtr, y_eval[tr])
    proba = clf.predict_proba(Xva)[:, 1]

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Xva)
    exp_val = explainer.expected_value
    if isinstance(sv, list):
        sv1 = sv[1]
        if isinstance(exp_val, (list, np.ndarray)) and hasattr(exp_val, '__len__') and len(exp_val) > 1:
            exp_val1 = exp_val[1]
        else:
            exp_val1 = exp_val
    else:
        if sv.ndim == 3 and sv.shape[-1] == 2:
            sv1 = sv[:, :, 1]
            exp_val1 = exp_val[1]
        else:
            sv1 = sv
            exp_val1 = exp_val

    all_X.append(Xva)
    all_y.append(y_eval[va])
    all_proba.append(proba)
    all_shap.append(sv1)
    all_expected_per_fold.append(float(exp_val1))

    pred = (proba > 0.5).astype(int)
    m = {
        'accuracy':  float(accuracy_score(y_eval[va], pred)),
        'precision': float(precision_score(y_eval[va], pred, zero_division=0)),
        'recall':    float(recall_score(y_eval[va], pred, zero_division=0)),
        'f1':        float(f1_score(y_eval[va], pred, zero_division=0)),
        'auc':       float(roc_auc_score(y_eval[va], proba)),
    }
    fold_metrics.append(m)
    print(f"  Fold {fi+1}/5  acc={m['accuracy']:.3f}  f1={m['f1']:.3f}  auc={m['auc']:.3f}  "
          f"mean P(fail)={proba.mean():.3f}  expected={exp_val1:.3f}  "
          f"shap_sum_avg={sv1.sum(axis=1).mean():+.4f}, exp_val={exp_val1:.3f}")

elapsed = time.time() - t0
print(f"  CV+SHAP done in {elapsed:.1f}s")

def logit_single(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))

X_va_full = np.concatenate(all_X)
y_full = np.concatenate(all_y)
proba_full = np.concatenate(all_proba)
shap_full = np.concatenate(all_shap)
expected_val = float(np.mean(all_expected_per_fold))
print(f"\nAggregated: n_val={len(y_full)}, mean P(fail)={proba_full.mean():.3f}, avg expected_val={expected_val:.3f}")

mean_abs = np.mean(np.abs(shap_full), axis=0)
order = np.argsort(mean_abs)[::-1]
print("\nMean |SHAP| per feature (ranked):")
for i in order:
    print(f"  {EVENT_TYPES[i]:15s}: {mean_abs[i]:.4f}")

with open(os.path.join(OUTDIR, "shap_summary.json"), "w") as f:
    json.dump({
        "feature_names": EVENT_TYPES,
        "mean_abs_shap": mean_abs.tolist(),
        "expected_value_per_fold": all_expected_per_fold,
        "global_expected_value": expected_val,
        "cv_fold_metrics": fold_metrics,
        "n_val_samples": int(len(y_full)),
        "n_students_total": int(n),
        "class_balance_failed_passed": [int(y_eval.sum()), int(n - y_eval.sum())],
        "runtime_seconds": elapsed,
    }, f, indent=2)

print("\n[2/4] Producing plots ...")
plt.figure(figsize=(9, 5))
shap.summary_plot(shap_full, X_va_full, feature_names=EVENT_TYPES, plot_type="dot", show=False, max_display=7)
plt.title("SHAP Beeswarm — RF-7dim on CS1\n(red = high feature value, blue = low; right of 0 = ↑ P(fail))", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "summary_beeswarm.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ beeswarm saved")

plt.figure(figsize=(8, 4))
shap.summary_plot(shap_full, X_va_full, feature_names=EVENT_TYPES, plot_type="bar", show=False, max_display=7)
plt.title("Mean |SHAP| Feature Importance — RF-7dim", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "importance_bar.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅ bar plot saved")

# Force plots top-3
print("\n[3/4] Force plots top-3 P(fail) students ...")
top_idx = np.argsort(proba_full)[-3:][::-1]
for rank, idx in enumerate(top_idx, 1):
    plt.figure(figsize=(11, 3.5))
    shap.force_plot(expected_val, shap_full[idx], X_va_full[idx],
                    feature_names=EVENT_TYPES, matplotlib=True,
                    show=False, contribution_threshold=0.05)
    trues = "FAILED" if y_full[idx] == 1 else "PASSED"
    plt.title(f"#{rank}: P(fail)={proba_full[idx]:.3f}  Truth={trues}  (blue↓risk, red↑risk)")
    plt.tight_layout()
    # original index in raw X (not scaled) for naming
    orig_idx = idx + int(len(X_va_full)/5) * (idx // (len(y_full) // 5))  # rough; we just use idx
    fname = f"force_rank{rank}_prob{proba_full[idx]:.2f}_truth{trues}.png"
    plt.savefig(os.path.join(OUTDIR, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ force #{rank} saved ({fname})")

# Statistical decomposition
print("\n[4/4] Per-class raw event-count stats ...")
df = pd.DataFrame(X, columns=EVENT_TYPES)
df['true_label'] = np.where(y_eval == 1, 'failed', 'passed')
group_means = df.groupby('true_label')[EVENT_TYPES].mean().round(2)
print("\nMean event counts (failed vs passed):")
print(group_means.to_string())
group_means.to_csv(os.path.join(OUTDIR, "mean_counts_by_class.csv"))

mean_shap_failed = shap_full[y_full == 1].mean(axis=0)
mean_shap_passed = shap_full[y_full == 0].mean(axis=0)
df_shap = pd.DataFrame({
    'feature': EVENT_TYPES,
    'mean_shap_when_failed': mean_shap_failed.round(4),
    'mean_shap_when_passed': mean_shap_passed.round(4),
    'mean_abs_total': mean_abs.round(4),
    'direction_for_failed_class': ['↑ risk' if v > 0 else '↓ risk' for v in mean_shap_failed],
}).sort_values('mean_abs_total', ascending=False)
df_shap.to_csv(os.path.join(OUTDIR, "shap_per_feature.csv"), index=False)
print("\nSHAP per feature (sorted by mean |SHAP|):")
print(df_shap.to_string(index=False))

np.savez(os.path.join(OUTDIR, "shap_values.npz"),
         X_va=X_va_full, y=y_full, proba=proba_full,
         shap_values=shap_full, expected_value=expected_val,
         feature_names=EVENT_TYPES)

print(f"\n✅ All artifacts saved to: {OUTDIR}")
