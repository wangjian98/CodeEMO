"""
SHAP Analysis [A] + [B] for RF on CS1.

A: Compare 7-dim vs 46-dim feature RF (both via SHAP)
B: Time-window SHAP: 24h / 3d / 7d / full — how feature importance shifts
   as we restrict data to events closer to the deadline.
"""
import os, sys, json, warnings, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

import shap

warnings.filterwarnings("ignore")

PROJECT_ROOT = "/home/ubuntu/CodeEMO"
sys.path.insert(0, PROJECT_ROOT)
from common.data_loader import load_ide_logs, set_seed
from common.feature_engineering import build_feature_matrix

EVENT_TYPES = [
    'text_insert', 'text_remove', 'text_paste',
    'focus_gained', 'focus_lost', 'run', 'submit'
]

OUTDIR = os.path.join(PROJECT_ROOT, "results", "random_forest", "shap_compare")
os.makedirs(OUTDIR, exist_ok=True)


def build_7dim(ide_logs, students, time_window_seconds=None):
    df = ide_logs
    if time_window_seconds is not None:
        df = df[df['timeToDeadline'] <= time_window_seconds]
    counts = (df.groupby(['student', 'eventType']).size().unstack(fill_value=0))
    counts = counts.reindex(columns=EVENT_TYPES, fill_value=0)
    counts = counts.reindex(index=students, fill_value=0)
    return counts.values.astype(np.float32)


def logit(p, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def shap_cv_analysis(X, y, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_X, all_y, all_proba, all_shap = [], [], [], []
    expected_per_fold = []
    fold_metrics = []

    for fi, (tr, va) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xva = scaler.transform(X[va])
        clf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                      random_state=random_state, n_jobs=-1)
        clf.fit(Xtr, y[tr])
        proba = clf.predict_proba(Xva)[:, 1]

        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xva)
        exp_val = explainer.expected_value
        if isinstance(sv, list):
            sv1 = sv[1]
            exp_val1 = exp_val[1] if isinstance(exp_val, (list, np.ndarray)) and hasattr(exp_val, '__len__') and len(exp_val) > 1 else exp_val
        else:
            if sv.ndim == 3 and sv.shape[-1] == 2:
                sv1 = sv[:, :, 1]
                exp_val1 = exp_val[1]
            else:
                sv1 = sv
                exp_val1 = exp_val

        all_X.append(Xva)
        all_y.append(y[va])
        all_proba.append(proba)
        all_shap.append(sv1)
        expected_per_fold.append(float(exp_val1))

        pred = (proba > 0.5).astype(int)
        fold_metrics.append({
            'acc': float(accuracy_score(y[va], pred)),
            'f1':  float(f1_score(y[va], pred, zero_division=0)),
            'auc': float(roc_auc_score(y[va], proba)),
        })

    return {
        'X_va':     np.concatenate(all_X),
        'y':        np.concatenate(all_y),
        'proba':    np.concatenate(all_proba),
        'shap':     np.concatenate(all_shap),
        'expected': float(np.mean(expected_per_fold)),
        'fold_metrics': fold_metrics,
    }


# ============ Setup ============
set_seed(42)
print("Loading data...")
ide_logs, passed_df = load_ide_logs()
student_ids = passed_df['student'].values
y_passed = passed_df['passed'].values.astype(int)
y_eval = 1 - y_passed
n = len(student_ids)
print(f"n_students = {n}, failed = {y_eval.sum()}, passed = {n - y_eval.sum()}\n")


# ============ A: 7-dim vs 46-dim ============
print("=" * 60)
print("[A] 7-dim vs 46-dim SHAP comparison")
print("=" * 60)

X_7 = build_7dim(ide_logs, student_ids, time_window_seconds=None)

print("Building 46-dim features...")
X_46_full, _, _ = build_feature_matrix(ide_logs, passed_df)
# Some students may produce NaN rows; handle
X_46_full = np.nan_to_num(X_46_full.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
print(f"  X_7 shape:   {X_7.shape}")
print(f"  X_46 shape:  {X_46_full.shape}")

print("\n[A1] Running SHAP CV on 7-dim ...")
t0 = time.time()
r7 = shap_cv_analysis(X_7, y_eval)
print(f"  Done in {time.time()-t0:.1f}s   CV F1={np.mean([m['f1'] for m in r7['fold_metrics']]):.4f}  AUC={np.mean([m['auc'] for m in r7['fold_metrics']]):.4f}")

print("\n[A2] Running SHAP CV on 46-dim ...")
t0 = time.time()
r46 = shap_cv_analysis(X_46_full, y_eval)
print(f"  Done in {time.time()-t0:.1f}s   CV F1={np.mean([m['f1'] for m in r46['fold_metrics']]):.4f}  AUC={np.mean([m['auc'] for m in r46['fold_metrics']]):.4f}")

mean_abs_7 = np.mean(np.abs(r7['shap']), axis=0)
mean_abs_46 = np.mean(np.abs(r46['shap']), axis=0)

# Generic names for 46-dim
dim_46_names = [f"f{i:02d}" for i in range(X_46_full.shape[1])]

order_7 = np.argsort(mean_abs_7)[::-1]
order_46 = np.argsort(mean_abs_46)[::-1]

# Plot A1: side-by-side top-7 vs top-15 features
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

top7 = order_7[:7]
axes[0].barh(range(len(top7)-1, -1, -1), mean_abs_7[top7], color='steelblue')
axes[0].set_yticks(range(len(top7)-1, -1, -1))
axes[0].set_yticklabels([EVENT_TYPES[i] for i in top7])
axes[0].set_xlabel('Mean |SHAP|')
axes[0].set_title('RF-7dim: Top-7 Features\nF1=%.3f AUC=%.3f (5-fold mean)' %
                   (np.mean([m['f1'] for m in r7['fold_metrics']]),
                    np.mean([m['auc'] for m in r7['fold_metrics']])), fontsize=10)
axes[0].grid(axis='x', linestyle=':', alpha=0.4)

top15 = order_46[:15]
axes[1].barh(range(len(top15)-1, -1, -1), mean_abs_46[top15], color='darkorange')
axes[1].set_yticks(range(len(top15)-1, -1, -1))
axes[1].set_yticklabels([dim_46_names[i] for i in top15], fontsize=8)
axes[1].set_xlabel('Mean |SHAP|')
axes[1].set_title('RF-46dim: Top-15 Features (out of 46)\nF1=%.3f AUC=%.3f (5-fold mean)' %
                   (np.mean([m['f1'] for m in r46['fold_metrics']]),
                    np.mean([m['auc'] for m in r46['fold_metrics']])), fontsize=10)
axes[1].grid(axis='x', linestyle=':', alpha=0.4)

plt.suptitle('[A] 7-dim vs 46-dim — feature importance by SHAP', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "A_7vs46_top_features.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved A_7vs46_top_features.png")

# Plot A2: SHAP concentration curve (how many features to reach 80% of importance)
def importance_concentration(mean_abs):
    """Return list of (k, cum_importance_share) for top-k features."""
    sorted_v = np.sort(mean_abs)[::-1]
    total = sorted_v.sum()
    cum = np.cumsum(sorted_v) / total
    return cum

cum_7 = importance_concentration(mean_abs_7)
cum_46 = importance_concentration(mean_abs_46)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(1, len(cum_7)+1), cum_7, marker='o', linewidth=2, label='7-dim', color='steelblue')
ax.plot(range(1, len(cum_46)+1), cum_46, marker='s', linewidth=2, label='46-dim', color='darkorange')
ax.axhline(0.80, linestyle='--', color='gray', alpha=0.5, label='80% threshold')
ax.set_xlabel('Top-K features (sorted by |SHAP|)')
ax.set_ylabel('Cumulative share of SHAP importance')
ax.set_title('[A] Importance Concentration — How concentrated is the model?')
ax.legend()
ax.grid(axis='y', linestyle=':', alpha=0.4)

# Mark where 80% is reached
k80_7 = int(np.searchsorted(cum_7, 0.80)) + 1
k80_46 = int(np.searchsorted(cum_46, 0.80)) + 1
ax.axvline(k80_7, color='steelblue', linestyle=':', alpha=0.6)
ax.axvline(k80_46, color='darkorange', linestyle=':', alpha=0.6)
ax.text(k80_7, 0.50, f'7-dim: {k80_7} feats', color='steelblue', fontsize=10)
ax.text(k80_46, 0.40, f'46-dim: {k80_46} feats', color='darkorange', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "A_concentration_curve.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved A_concentration_curve.png (k80: 7={k80_7}, 46={k80_46})")

# Save A artifacts
with open(os.path.join(OUTDIR, "A_summary.json"), "w") as f:
    json.dump({
        "7d": {
            "feature_names": EVENT_TYPES,
            "mean_abs_shap": mean_abs_7.tolist(),
            "ranked_order": [EVENT_TYPES[i] for i in order_7.tolist()],
            "k80_features": int(k80_7),
            "cv_mean_f1": float(np.mean([m['f1'] for m in r7['fold_metrics']])),
            "cv_mean_auc": float(np.mean([m['auc'] for m in r7['fold_metrics']])),
            "cv_mean_acc": float(np.mean([m['acc'] for m in r7['fold_metrics']])),
        },
        "46d": {
            "feature_names": dim_46_names,
            "mean_abs_shap": mean_abs_46.tolist(),
            "ranked_order": [dim_46_names[i] for i in order_46.tolist()],
            "k80_features": int(k80_46),
            "cv_mean_f1": float(np.mean([m['f1'] for m in r46['fold_metrics']])),
            "cv_mean_auc": float(np.mean([m['auc'] for m in r46['fold_metrics']])),
            "cv_mean_acc": float(np.mean([m['acc'] for m in r46['fold_metrics']])),
        }
    }, f, indent=2)


# ============ B: Time-window SHAP ============
print("\n" + "=" * 60)
print("[B] Time-window SHAP analysis")
print("=" * 60)

# timeToDeadline is in seconds
windows = [
    ('24h', 24 * 3600),
    ('3d',  3 * 24 * 3600),
    ('7d',  7 * 24 * 3600),
    ('full', None),
]

results_b = {}
window_data_summary = []

for name, threshold_seconds in windows:
    print(f"\n--- Window: {name} ---")
    if threshold_seconds is None:
        events_in_window = len(ide_logs)
        X = build_7dim(ide_logs, student_ids, time_window_seconds=None)
        label = "all events"
    else:
        events_in_window = int((ide_logs['timeToDeadline'] <= threshold_seconds).sum())
        X = build_7dim(ide_logs, student_ids, time_window_seconds=threshold_seconds)
        label = f"≤{name} before deadline ({events_in_window} events)"

    print(f"  Filter: {label}")
    print(f"  X shape: {X.shape}, total events counted: {int(X.sum()):,}")
    print(f"  Coverage: {(X > 0).sum() / X.size:.1%} non-zero cells")

    t0 = time.time()
    r = shap_cv_analysis(X, y_eval)
    elapsed = time.time() - t0

    mean_abs = np.mean(np.abs(r['shap']), axis=0)
    order = np.argsort(mean_abs)[::-1]

    f1 = float(np.mean([m['f1'] for m in r['fold_metrics']]))
    auc = float(np.mean([m['auc'] for m in r['fold_metrics']]))
    acc = float(np.mean([m['acc'] for m in r['fold_metrics']]))

    print(f"  Done in {elapsed:.1f}s  | F1={f1:.4f}  AUC={auc:.4f}  Acc={acc:.4f}")
    print(f"  Top-3 features:")
    for i in order[:3]:
        print(f"    {EVENT_TYPES[i]:15s}: |SHAP|={mean_abs[i]:.4f}")

    results_b[name] = {
        "threshold_seconds": threshold_seconds,
        "events_in_window": events_in_window,
        "n_features": X.shape[1],
        "non_zero_share": float((X > 0).sum() / X.size),
        "feature_mean_abs": mean_abs.tolist(),
        "ranked_order": [EVENT_TYPES[i] for i in order.tolist()],
        "cv_f1_mean": f1,
        "cv_auc_mean": auc,
        "cv_acc_mean": acc,
        "runtime": elapsed,
    }

# Save B summary
with open(os.path.join(OUTDIR, "B_summary.json"), "w") as f:
    json.dump(results_b, f, indent=2)

# Plot B1: line chart
fig, ax = plt.subplots(figsize=(11, 6))
colors = plt.cm.Set1(np.linspace(0, 1, len(EVENT_TYPES)))

for fi, feat in enumerate(EVENT_TYPES):
    vals = [results_b[w[0]]['feature_mean_abs'][EVENT_TYPES.index(feat)] for w in windows]
    ax.plot([w[0] for w in windows], vals, marker='o', linewidth=2, label=feat, color=colors[fi])

ax.set_xlabel('Time window (data within X of deadline)')
ax.set_ylabel('Mean |SHAP|')
ax.set_title('[B] SHAP Feature Importance Across Time Windows (RF-7dim)')
ax.legend(loc='best', fontsize=9, ncol=2)
ax.grid(axis='y', linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "B_importance_vs_time.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved B_importance_vs_time.png")

# Plot B2: heatmap (raw share)
window_labels = [w[0] for w in windows]
matrix = np.zeros((len(EVENT_TYPES), len(windows)))
for wi, w in enumerate(windows):
    for fi in range(len(EVENT_TYPES)):
        matrix[fi, wi] = results_b[w[0]]['feature_mean_abs'][fi]
matrix_norm = matrix / matrix.sum(axis=0)

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(matrix_norm, annot=matrix, fmt='.3f',
            xticklabels=window_labels, yticklabels=EVENT_TYPES,
            cmap='YlGnBu', ax=ax, cbar_kws={'label': 'Share of SHAP importance'},
            annot_kws={'size': 8})
ax.set_title('[B] SHAP Importance Heatmap (rows: feature, cols: time window)\nraw values annotated')
ax.set_xlabel('Time window')
ax.set_ylabel('Feature')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "B_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved B_heatmap.png")

# Plot B3: F1 / AUC vs time
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
ws = [w[0] for w in windows]
axes[0].plot(ws, [results_b[w]['cv_f1_mean'] for w in ws], marker='o', linewidth=2, color='darkblue')
axes[0].set_ylabel('CV mean F1')
axes[0].set_title('[B] RF-7dim F1 across time windows')
axes[0].grid(axis='y', linestyle=':', alpha=0.4)
axes[0].set_ylim(0.65, 0.95)
for x, y in zip(ws, [results_b[w]['cv_f1_mean'] for w in ws]):
    axes[0].annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)

axes[1].plot(ws, [results_b[w]['cv_auc_mean'] for w in ws], marker='o', linewidth=2, color='darkred')
axes[1].set_ylabel('CV mean AUC')
axes[1].set_title('[B] RF-7dim AUC across time windows')
axes[1].grid(axis='y', linestyle=':', alpha=0.4)
axes[1].set_ylim(0.65, 0.95)
for x, y in zip(ws, [results_b[w]['cv_auc_mean'] for w in ws]):
    axes[1].annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "B_F1_AUC_vs_time.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved B_F1_AUC_vs_time.png")

# Save OVERALL summary
with open(os.path.join(OUTDIR, "OVERALL_summary.json"), "w") as f:
    json.dump({
        "A_46vs7": {
            "7d": {
                "n_features": 7,
                "cv_f1": float(np.mean([m['f1'] for m in r7['fold_metrics']])),
                "cv_auc": float(np.mean([m['auc'] for m in r7['fold_metrics']])),
                "k80_features": int(k80_7),
                "top3": [EVENT_TYPES[i] for i in order_7[:3].tolist()],
            },
            "46d": {
                "n_features": 46,
                "cv_f1": float(np.mean([m['f1'] for m in r46['fold_metrics']])),
                "cv_auc": float(np.mean([m['auc'] for m in r46['fold_metrics']])),
                "k80_features": int(k80_46),
                "top3": [dim_46_names[i] for i in order_46[:3].tolist()],
            },
        },
        "B_time_windows": {
            w: {
                "F1": results_b[w]["cv_f1_mean"],
                "AUC": results_b[w]["cv_auc_mean"],
                "events": results_b[w]["events_in_window"],
                "top3": results_b[w]["ranked_order"][:3],
            } for w in [w[0] for w in windows]
        },
    }, f, indent=2)

print(f"\n✅ All artifacts saved to: {OUTDIR}")
print("✅ Done!")
