"""
完整 Late Fusion + Stacking v2.5
  4 路 OOF probs → 元学习器 (LR + 软阈值搜索 + logit-space 加权)
"""
import os, json, sys, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             precision_score, recall_score, f1_score)

sys.path.insert(0, "/home/ubuntu/CodeEMO")


def load_model_probs(path, flip=False):
    p = np.load(path)
    return (1 - p) if flip else p


def per_fold_metrics(probs, labels, fold_idx, thr=0.5):
    out = []
    for f in range(5):
        m = fold_idx == f
        y = labels[m]; p = probs[m]
        out.append({
            'f1':  float(f1_score(y, (p>thr).astype(int), zero_division=0)),
            'auc': float(roc_auc_score(y, p)),
            'acc': float(accuracy_score(y, (p>thr).astype(int))),
            'p':   float(precision_score(y, (p>thr).astype(int), zero_division=0)),
            'r':   float(recall_score(y, (p>thr).astype(int), zero_division=0)),
        })
    return out


def avg(metric_list):
    return {k: f"{np.mean([m[k] for m in metric_list]):.4f} ± {np.std([m[k] for m in metric_list]):.4f}"
            for k in ['acc','p','r','f1','auc']}


def find_best_threshold(probs, labels, fold_idx):
    """按 5-fold OOF 找最优阈值（最大化 F1）"""
    best_thr, best_f1 = 0.5, 0.0
    # 加 OOF 拼接后扫阈值, 然后把阈值应用到每折求真实 F1 均值 (一致性更好)
    order = np.argsort(-probs)
    thr_grid = np.unique(np.concatenate([
        np.linspace(0.05, 0.95, 19),
        probs[order[:50]],  # top-50 边界
    ]))
    for thr in thr_grid:
        ms = per_fold_metrics(probs, labels, fold_idx, thr=thr)
        f1m = np.mean([m['f1'] for m in ms])
        if f1m > best_f1:
            best_f1 = f1m
            best_thr = float(thr)
    return best_thr, best_f1


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1-eps)
    return np.log(p / (1 - p))


def main():
    # ---- 加载 4 路 probs (P(fail) 已对齐 labels, fold_idx) ----
    P = {
        'B46':  load_model_probs("/home/ubuntu/CodeEMO/outputs/bilstm_save_probs/probs.npy", flip=True),
        'DV2':  load_model_probs("/home/ubuntu/CodeEMO/outputs/bi_lstm_trans_v2/probs.npy"),
        'MID':  load_model_probs("/home/ubuntu/CodeEMO/outputs/bilstm_7dim_gpu/probs.npy"),
        'SHT':  load_model_probs("/home/ubuntu/CodeEMO/outputs/bilstm_7dim_max50_gpu/probs.npy"),
        # 'B2K': 可选 — 若训练完成
    }
    # 可选 max=2000 (若已训练好)
    b2k_path = "/home/ubuntu/CodeEMO/outputs/bilstm_7dim_max2000_gpu/probs.npy"
    have_b2k = os.path.exists(b2k_path)
    if have_b2k:
        P['B2K'] = load_model_probs(b2k_path)
        print(f"[+] 5路启用: 加 B2K (max=2000 BiLSTM)")
    else:
        print(f"[ ] B2K 还没训完, 用 4 路")

    labels = np.load("/home/ubuntu/CodeEMO/outputs/bi_lstm_trans_v2/labels.npy")
    fold_idx = np.load("/home/ubuntu/CodeEMO/outputs/bi_lstm_trans_v2/fold_idx.npy")
    keys = list(P.keys())

    print(f"Models in stack: {keys}")
    print(f"labels: pass={int((labels==0).sum())}, fail={int((labels==1).sum())}")
    print()

    # ---- 单模型 5 折复测 ----
    print("="*70)
    print("单模型 5 折 (P(fail), threshold=0.5)")
    print("="*70)
    for k in keys:
        ms = per_fold_metrics(P[k], labels, fold_idx)
        print(f"{k}: {avg(ms)}")
    print()

    # ---- A. 简单 Late Fusion 网格 (4 路, step=0.05, 和=1) ----
    print("="*70)
    print("A. 简单 Late Fusion 网格 (线性加权, step=0.05)")
    print("="*70)
    if len(keys) == 4:
        from itertools import product
        step = 0.05
        grid = []
        for w in product(np.arange(0, 1+1e-9, step), repeat=4):
            s = sum(w)
            if abs(s-1) > 1e-6: continue
            a,b,c,d = w
            P_ens = a*P['B46'] + b*P['DV2'] + c*P['MID'] + d*P['SHT']
            auc = float(roc_auc_score(labels, P_ens))
            thr, f1 = find_best_threshold(P_ens, labels, fold_idx)
            grid.append({'a':round(a,2),'b':round(b,2),'c':round(c,2),'d':round(d,2),
                         'thr':thr, 'f1':f1, 'auc':auc})
        grid.sort(key=lambda x: -x['f1'])
        print(f"\n网格点数: {len(grid)}")
        print("\nTop 5 by F1 (含 OOF 最佳阈值):")
        for i,r in enumerate(grid[:5],1):
            print(f"  {i}: a={r['a']:.2f} b={r['b']:.2f} c={r['c']:.2f} d={r['d']:.2f}"
                  f"  thr={r['thr']:.3f}  F1={r['f1']:.4f}  AUC={r['auc']:.4f}")

    # ---- B. Stacking — LR 元学习器 ----
    print()
    print("="*70)
    print("B. Stacking — LR 元学习器 (5-fold OOF)")
    print("="*70)
    X_logit = np.stack([logit(P[k]) for k in keys], axis=1)
    X_prob  = np.stack([P[k]  for k in keys], axis=1)

    # LR 5-fold OOF
    skf_idx = []
    for f in range(5):
        skf_idx.append(np.where(fold_idx == f)[0])

    def stack_oof(X, y, model_fn):
        from sklearn.model_selection import StratifiedKFold
        oof = np.zeros(len(y))
        for f in range(5):
            tr = np.concatenate([skf_idx[i] for i in range(5) if i != f])
            te = skf_idx[f]
            clf = model_fn()
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        return oof

    # B1. LR on logit-probs
    oof_lr1 = stack_oof(X_logit, labels, lambda: LogisticRegression(C=1.0, max_iter=200))
    thr_lr1, f1_lr1 = find_best_threshold(oof_lr1, labels, fold_idx)
    print(f"\nLR(logit)       thr={thr_lr1:.3f}  F1={f1_lr1:.4f}  AUC={roc_auc_score(labels, oof_lr1):.4f}")

    # B2. LR on raw probs
    oof_lr2 = stack_oof(X_prob, labels, lambda: LogisticRegression(C=1.0, max_iter=200))
    thr_lr2, f1_lr2 = find_best_threshold(oof_lr2, labels, fold_idx)
    print(f"LR(prob)        thr={thr_lr2:.3f}  F1={f1_lr2:.4f}  AUC={roc_auc_score(labels, oof_lr2):.4f}")

    # B3. LR with regularization sweep
    print("\nLR 正则扫描 (logit-probs, 5-fold):")
    best = (-1, None, None, None)
    for C in [0.1, 0.3, 1.0, 3.0, 10.0]:
        oof = stack_oof(X_logit, labels, lambda C=C: LogisticRegression(C=C, max_iter=300))
        thr, f1 = find_best_threshold(oof, labels, fold_idx)
        auc = roc_auc_score(labels, oof)
        print(f"  C={C:>4}: thr={thr:.3f}  F1={f1:.4f}  AUC={auc:.4f}")
        if f1 > best[0]:
            best = (f1, C, thr, auc)

    # B4. 用每折预测均值(非 OOF) — 参考用
    # B5. logit-space 加权再 sigmoid
    print("\nlogit-space 加权融合:")
    # 用网格 LR 系数做 logit 加权
    # 用最优线性组合: w_logit = inv(X'X) X' y
    from numpy.linalg import lstsq
    w, *_ = lstsq(X_logit, labels.astype(float), rcond=None)
    P_logit_lin = 1 / (1 + np.exp(-X_logit @ w))
    thr_l, f1_l = find_best_threshold(P_logit_lin, labels, fold_idx)
    print(f"logit-LS (raw): weights={w.round(3).tolist()}, thr={thr_l:.3f}, F1={f1_l:.4f}, AUC={roc_auc_score(labels, P_logit_lin):.4f}")

    # ---- 总结 ----
    print()
    print("="*70)
    print("📊 全局对比")
    print("="*70)
    base_f1, base_auc = 0.8754, 0.9234
    print(f"Baseline (BiLSTM46 + D_v2 α=0.5)      F1={base_f1:.4f}  AUC={base_auc:.4f}")
    print(f"原 Step1 v2 最佳 (4 路网格)          F1=0.8941  AUC=0.9232")
    print(f"Step2 Stacking LR(logit, C={best[1]})  F1={best[0]:.4f}  AUC={best[3]:.4f}")
    print(f"Step2 logit-LS                         F1={f1_l:.4f}  AUC={roc_auc_score(labels, P_logit_lin):.4f}")

    # 保存
    out_dir = "/home/ubuntu/CodeEMO/outputs/late_fusion_stacking"
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        'models': keys,
        'have_b2k': have_b2k,
        'best_lr': {'C': best[1], 'thr': best[2], 'f1': best[0], 'auc': best[3]},
        'logit_lstsq': {'weights': w.tolist(), 'thr': thr_l, 'f1': f1_l,
                        'auc': float(roc_auc_score(labels, P_logit_lin))},
        'baseline': {'f1': base_f1, 'auc': base_auc},
    }
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n结果: {out_dir}/results.json")


if __name__ == '__main__':
    main()
