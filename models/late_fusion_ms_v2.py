"""
Late Fusion v2 — 4 路融合，把 baseline 已有的 BiLSTM-46d 加进来
  - BiLSTM_46d        (高维手工特征)
  - bi_lstm_trans_v2  (max=2000, raw 序列 + 30d 手工 + SE)
  - bilstm_7d_500     (max=500,  raw 序列)
  - bilstm_7d_50      (max=50,   raw 序列 — 新训练)
网格搜索 4 个权重, 和=1, step=0.10
"""
import os, json, sys
import numpy as np
from itertools import product
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score)

sys.path.insert(0, "/home/ubuntu/CodeEMO")

P_b46 = (1 - np.load("/home/ubuntu/CodeEMO/outputs/bilstm_save_probs/probs.npy"))  # 翻转
P_dv2 = np.load("/home/ubuntu/CodeEMO/outputs/bi_lstm_trans_v2/probs.npy")
P_mid = np.load("/home/ubuntu/CodeEMO/outputs/bilstm_7dim_gpu/probs.npy")
P_short = np.load("/home/ubuntu/CodeEMO/outputs/bilstm_7dim_max50_gpu/probs.npy")
labels = np.load("/home/ubuntu/CodeEMO/outputs/bi_lstm_trans_v2/labels.npy")
fold_idx = np.load("/home/ubuntu/CodeEMO/outputs/bi_lstm_trans_v2/fold_idx.npy")

# 验证 lengths
for n, p in [('B46', P_b46), ('DV2', P_dv2), ('MID', P_mid), ('SHT', P_short)]:
    print(f"  {n}: shape={p.shape}, range=[{p.min():.4f},{p.max():.4f}]")
print(f"  labels: {labels.shape}, pos_rate={labels.mean():.3f}\n")


def per_fold(probs, thr=0.5):
    out = []
    for f in range(5):
        m = fold_idx == f
        y = labels[m]; p = probs[m]
        out.append({
            'f1':  float(f1_score(y, (p>thr).astype(int), zero_division=0)),
            'auc': float(roc_auc_score(y, p)),
            'acc': float(accuracy_score(y, (p>thr).astype(int))),
        })
    return out


def summary(ms):
    return {k: f"{np.mean([m[k] for m in ms]):.4f} ± {np.std([m[k] for m in ms]):.4f}"
            for k in ['f1','auc','acc']}


print("="*70)
print("单模型复测")
print("="*70)
for name, P in [('B46',P_b46),('DV2',P_dv2),('MID',P_mid),('SHT',P_short)]:
    print(f"{name}: {summary(per_fold(P))}\n")


# 4 路网格, step=0.10, 和=1
step = 0.10
results = []
n_steps = int(round(1.0/step)) + 1
for w in product(np.arange(0, 1+1e-9, step), repeat=4):
    s = sum(w)
    if abs(s - 1.0) > 1e-6:
        continue
    a, b, c, d = w
    P = a*P_b46 + b*P_dv2 + c*P_mid + d*P_short
    ms = per_fold(P)
    f1m = float(np.mean([m['f1'] for m in ms]))
    auc = float(roc_auc_score(labels, P))
    results.append({
        'a':round(float(a),2), 'b':round(float(b),2),
        'c':round(float(c),2), 'd':round(float(d),2),
        'f1':f1m, 'auc':auc,
    })
print(f"\n4路网格搜索点数: {len(results)} (step={step})\n")

res_f1 = sorted(results, key=lambda r: -r['f1'])
res_auc = sorted(results, key=lambda r: -r['auc'])

print("Top 15 by F1:")
print(f"{'rank':>4} {'a=B46':>7} {'b=DV2':>7} {'c=MID':>7} {'d=SHT':>7}  {'F1':>8}  {'AUC':>8}")
for i,r in enumerate(res_f1[:15], 1):
    print(f"{i:>4} {r['a']:>7.2f} {r['b']:>7.2f} {r['c']:>7.2f} {r['d']:>7.2f}  {r['f1']:>8.4f}  {r['auc']:>8.4f}")

print("\nTop 15 by AUC:")
print(f"{'rank':>4} {'a=B46':>7} {'b=DV2':>7} {'c=MID':>7} {'d=SHT':>7}  {'F1':>8}  {'AUC':>8}")
for i,r in enumerate(res_auc[:15], 1):
    print(f"{i:>4} {r['a']:>7.2f} {r['b']:>7.2f} {r['c']:>7.2f} {r['d']:>7.2f}  {r['f1']:>8.4f}  {r['auc']:>8.4f}")

base_f1, base_auc = 0.8754, 0.9234
b = res_f1[0]
print()
print("="*70)
print(f"baseline: F1={base_f1}, AUC={base_auc}")
print(f"MVP v1 最佳 F1 (3路): 0.8479 (-0.0275)")
print(f"MVP v2 最佳 F1 (4路): {b['f1']:.4f} ({b['f1']-base_f1:+.4f}) 配比={b['a']:.2f}/{b['b']:.2f}/{b['c']:.2f}/{b['d']:.2f}")
b2 = res_auc[0]
print(f"MVP v2 最佳 AUC(4路): {b2['auc']:.4f} ({b2['auc']-base_auc:+.4f})")
print("="*70)

out_dir = "/home/ubuntu/CodeEMO/outputs/late_fusion_ms_v2"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir,'results.json'),'w') as f:
    json.dump({
        'note':'4路 Late Fusion',
        'step':step,
        'n_grid':len(results),
        'best_f1':res_f1[0], 'best_auc':res_auc[0],
        'top15_f1':res_f1[:15], 'top15_auc':res_auc[:15],
        'baseline':{'f1':base_f1,'auc':base_auc},
    }, f, indent=2, ensure_ascii=False)
print(f"\n保存到: {out_dir}/results.json")
