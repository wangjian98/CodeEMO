"""
5 路 Late Fusion 网格 (B46 + DV2 + MID + SHT + B2K)
step=0.10, 和=1, 715 组. 用最佳阈值匹配 OOF F1.
"""
import os, json, sys
import numpy as np
from itertools import product
from sklearn.metrics import roc_auc_score, f1_score

sys.path.insert(0, "/home/ubuntu/CodeEMO")

P_b46  = (1 - np.load("/home/ubuntu/CodeEMO/outputs/bilstm_save_probs/probs.npy"))
P_dv2  = np.load("/home/ubuntu/CodeEMO/outputs/bi_lstm_trans_v2/probs.npy")
P_mid  = np.load("/home/ubuntu/CodeEMO/outputs/bilstm_7dim_gpu/probs.npy")
P_sht  = np.load("/home/ubuntu/CodeEMO/outputs/bilstm_7dim_max50_gpu/probs.npy")
P_b2k  = np.load("/home/ubuntu/CodeEMO/outputs/bilstm_7dim_max2000_gpu/probs.npy")
labels = np.load("/home/ubuntu/CodeEMO/outputs/bi_lstm_trans_v2/labels.npy")
fold_idx = np.load("/home/ubuntu/CodeEMO/outputs/bi_lstm_trans_v2/fold_idx.npy")

arr = [P_b46, P_dv2, P_mid, P_sht, P_b2k]
names = ['B46','DV2','MID','SHT','B2K']

print("5路网格搜索 (step=0.10, 715 组)")
step = 0.10
results = []
all_w = list(product(np.arange(0, 1+1e-9, step), repeat=5))
all_w = [w for w in all_w if abs(sum(w)-1) < 1e-6]
print(f"有效配比: {len(all_w)}")

for w in all_w:
    a,b,c,d,e = w
    P_ens = a*arr[0] + b*arr[1] + c*arr[2] + d*arr[3] + e*arr[4]
    ms = []
    for f in range(5):
        m = fold_idx == f
        y = labels[m]; p = P_ens[m]
        for thr in [0.45, 0.48, 0.50, 0.52, 0.55]:
            ms.append((f, thr, f1_score(y, (p>thr).astype(int), zero_division=0)))
    # 对每个 f 选最佳 thr, 再平均
    f1s = []
    for f in range(5):
        cands = [s for s in ms if s[0]==f]
        f1s.append(max(s[2] for s in cands))
    f1m = np.mean(f1s)
    f1s_std = np.std(f1s)
    auc = roc_auc_score(labels, P_ens)
    results.append({
        'a':round(a,2),'b':round(b,2),'c':round(c,2),'d':round(d,2),'e':round(e,2),
        'f1':float(f1m),'f1_std':float(f1s_std),'auc':float(auc),
    })

results.sort(key=lambda r: -r['f1'])
print("\nTop 10 (F1, 每折最优阈值):")
print(f"{'rank':>4} {'B46':>5} {'DV2':>5} {'MID':>5} {'SHT':>5} {'B2K':>5}   {'F1':>8} ± {'std':>6}   {'AUC':>8}")
for i,r in enumerate(results[:10], 1):
    print(f"{i:>4} {r['a']:>5.2f} {r['b']:>5.2f} {r['c']:>5.2f} {r['d']:>5.2f} {r['e']:>5.2f}   {r['f1']:>8.4f} ± {r['f1_std']:.4f}   {r['auc']:>8.4f}")

results.sort(key=lambda r: -r['auc'])
print("\nTop 10 (AUC):")
for i,r in enumerate(results[:10], 1):
    print(f"{i:>4} {r['a']:>5.2f} {r['b']:>5.2f} {r['c']:>5.2f} {r['d']:>5.2f} {r['e']:>5.2f}   {r['f1']:>8.4f} ± {r['f1_std']:.4f}   {r['auc']:>8.4f}")

best = results[0]
print()
print(f"baseline: F1=0.8754  AUC=0.9234")
print(f"v2 4路网格 (step=0.10): F1=0.8941  AUC=0.9232")
print(f"5路 LR-stacking: F1=0.8907  AUC=0.9194")
print(f"5路 grid 最佳: F1={best['f1']:.4f}  AUC={best['auc']:.4f}  配比=B46:{best['a']}/DV2:{best['b']}/MID:{best['c']}/SHT:{best['d']}/B2K:{best['e']}")

out_dir = "/home/ubuntu/CodeEMO/outputs/late_fusion_5way_v1"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir,'results.json'),'w') as f:
    json.dump({'note':'5路 grid, 每折最优thr', 'step':step, 'best':results[0],
               'top10_f1': sorted(results, key=lambda r:-r['f1'])[:10],
               'baseline':{'f1':0.8754,'auc':0.9234}}, f, indent=2, ensure_ascii=False)
print(f"\n保存: {out_dir}/results.json")
