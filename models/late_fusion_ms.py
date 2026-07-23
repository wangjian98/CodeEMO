"""
Late Fusion 多尺度权重网格搜索

3 路 P(fail) 融合:
  - short:  max_seq=50  (BiLSTM 7d, 新训练)
  - mid:    max_seq=500 (BiLSTM 7d, 已有)
  - long:   max_seq=2000(BiLSTM+Trans v2, 已有)

最终概率 = a*P_short + b*P_mid + c*P_long,  a+b+c=1, step=0.05
评估: 5-fold F1 / AUC, 复用 ensemble_final 已对齐的 fold_idx
"""
import os
import json
import sys
import numpy as np
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score)

sys.path.insert(0, "/home/ubuntu/CodeEMO")


# ---- 加载 3 个模型在 473 个学生上的 P(fail) 概率 ----
P_short = np.load("/home/ubuntu/CodeEMO/outputs/bilstm_7dim_max50_gpu/probs.npy")
P_mid   = np.load("/home/ubuntu/CodeEMO/outputs/bilstm_7dim_gpu/probs.npy")
P_long  = np.load("/home/ubuntu/CodeEMO/outputs/bi_lstm_trans_v2/probs.npy")

# 用 D_v2 的 labels 作为 ground truth (最权威 + 与训练分布对齐)
labels = np.load("/home/ubuntu/CodeEMO/outputs/bi_lstm_trans_v2/labels.npy")
fold_idx = np.load("/home/ubuntu/CodeEMO/outputs/bi_lstm_trans_v2/fold_idx.npy")

print(f"P_short shape={P_short.shape}, range=[{P_short.min():.4f},{P_short.max():.4f}]")
print(f"P_mid   shape={P_mid.shape},   range=[{P_mid.min():.4f},{P_mid.max():.4f}]")
print(f"P_long  shape={P_long.shape},  range=[{P_long.min():.4f},{P_long.max():.4f}]")
print(f"labels: pass={int((labels==0).sum())}, fail={int((labels==1).sum())}")
print(f"folds: {np.unique(fold_idx, return_counts=True)}")
print()


def per_fold_metrics(probs, labels, fold_idx, thr=0.5):
    out = []
    for f in range(5):
        m = fold_idx == f
        y = labels[m]
        p = probs[m]
        out.append({
            'accuracy':  float(accuracy_score(y, (p > thr).astype(int))),
            'precision': float(precision_score(y, (p > thr).astype(int), zero_division=0)),
            'recall':    float(recall_score(y, (p > thr).astype(int), zero_division=0)),
            'f1':        float(f1_score(y, (p > thr).astype(int), zero_division=0)),
            'auc':       float(roc_auc_score(y, p)),
        })
    return out


def summarize(ms):
    return {k: f"{np.mean([m[k] for m in ms]):.4f} ± {np.std([m[k] for m in ms]):.4f}"
            for k in ['accuracy', 'precision', 'recall', 'f1', 'auc']}


print("=" * 65)
print("单模型 5 折表现 (P(fail))")
print("=" * 65)
for name, P in [('short max=50', P_short),
                ('mid   max=500', P_mid),
                ('long  max=2000', P_long)]:
    ms = per_fold_metrics(P, labels, fold_idx)
    print(f"\n{name}: {summarize(ms)}")
    for i, m in enumerate(ms, 1):
        print(f"  Fold {i}: F1={m['f1']:.4f}  AUC={m['auc']:.4f}")


print()
print("=" * 65)
print("Late Fusion 网格 (a*P_short + b*P_mid + c*P_long), step=0.05")
print("=" * 65)

step = 0.05
results = []
for a in np.arange(0, 1.0 + 1e-9, step):
    for b in np.arange(0, 1.0 + 1e-9 - a + 1e-9, step):
        c = 1.0 - a - b
        if c < -1e-9:
            continue
        c = max(0.0, c)
        P = a * P_short + b * P_mid + c * P_long
        global_auc = roc_auc_score(labels, P)
        ms = per_fold_metrics(P, labels, fold_idx)
        f1_mean = float(np.mean([m['f1'] for m in ms]))
        f1_std  = float(np.std([m['f1'] for m in ms]))
        auc_mean = float(np.mean([m['auc'] for m in ms]))
        results.append({
            'a': round(a, 3), 'b': round(b, 3), 'c': round(c, 3),
            'global_auc': float(global_auc),
            'f1_mean': f1_mean, 'f1_std': f1_std, 'auc_mean': auc_mean,
        })

results.sort(key=lambda x: -x['global_auc'])
print(f"\n搜索点数: {len(results)}")
print("\nTop 10 (按 global AUC 降序):")
print(f"{'rank':>4} {'a':>5} {'b':>5} {'c':>5}  {'global_auc':>10}  {'F1_mean':>8} +/- {'F1_std':<6}  {'AUC_mean':>8}")
for i, r in enumerate(results[:10], 1):
    print(f"{i:>4} {r['a']:>5.2f} {r['b']:>5.2f} {r['c']:>5.2f}  "
          f"{r['global_auc']:>10.4f}  {r['f1_mean']:>8.4f} +/- {r['f1_std']:.4f}  {r['auc_mean']:>8.4f}")

results_f1 = sorted(results, key=lambda x: -x['f1_mean'])
print("\nTop 10 (按 F1_mean 降序):")
print(f"{'rank':>4} {'a':>5} {'b':>5} {'c':>5}  {'global_auc':>10}  {'F1_mean':>8} +/- {'F1_std':<6}  {'AUC_mean':>8}")
for i, r in enumerate(results_f1[:10], 1):
    print(f"{i:>4} {r['a']:>5.2f} {r['b']:>5.2f} {r['c']:>5.2f}  "
          f"{r['global_auc']:>10.4f}  {r['f1_mean']:>8.4f} +/- {r['f1_std']:.4f}  {r['auc_mean']:>8.4f}")

best_auc = results[0]
best_f1 = results_f1[0]
print()
print("=" * 65)
print("[最佳 AUC 配置]")
print(f"   a={best_auc['a']:.2f}, b={best_auc['b']:.2f}, c={best_auc['c']:.2f}")
print(f"   F1 = {best_auc['f1_mean']:.4f} +/- {best_auc['f1_std']:.4f}")
print(f"   AUC = {best_auc['global_auc']:.4f} (global) / {best_auc['auc_mean']:.4f} (5折均值)")
print()
print("[最佳 F1 配置]")
print(f"   a={best_f1['a']:.2f}, b={best_f1['b']:.2f}, c={best_f1['c']:.2f}")
print(f"   F1 = {best_f1['f1_mean']:.4f} +/- {best_f1['f1_std']:.4f}")
print(f"   AUC = {best_f1['global_auc']:.4f} (global) / {best_f1['auc_mean']:.4f} (5折均值)")

baseline_f1 = 0.8754
print()
print("=" * 65)
print(f"vs baseline (ensemble_final F1=0.8754, AUC=0.9234)")
print(f"   最佳 AUC vs baseline: {best_auc['global_auc'] - 0.9234:+.4f} (global)")
print(f"   最佳 F1  vs baseline: {best_f1['f1_mean']  - baseline_f1:+.4f}")
print("=" * 65)

out_dir = "/home/ubuntu/CodeEMO/outputs/late_fusion_ms_v1"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'results.json'), 'w') as f:
    json.dump({
        'note': 'Late Fusion MVP: a*P_short + b*P_mid + c*P_long',
        'step': step,
        'n_grid': len(results),
        'best_auc': best_auc,
        'best_f1': best_f1,
        'top10_auc': results[:10],
        'top10_f1': results_f1[:10],
        'baseline': {
            'f1': baseline_f1,
            'auc': 0.9234,
            'note': 'BiLSTM(46d) + D_v2 alpha=0.5',
        },
    }, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存: {out_dir}/results.json")
