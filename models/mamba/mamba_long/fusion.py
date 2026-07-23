"""
Mamba-Long Late Fusion - 步骤 5

把 Mamba-Long 加入多模型融合:
  - 7 路融合: 6 个统一对比模型 + Mamba-Long
  - 网格搜索最佳权重组合
  - 输出 fusion 报告
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             precision_score, recall_score, f1_score)

OUT = "/home/ubuntu/CodeEMO/outputs/unified_compare"
MLONG = "/home/ubuntu/CodeEMO/outputs/mamba_long"
UNIFIED_Y = "/home/ubuntu/CodeEMO/outputs/bi_lstm_trans_v2/labels.npy"
UNIFIED_F = "/home/ubuntu/CodeEMO/outputs/bi_lstm_trans_v2/fold_idx.npy"

COMBOS_7 = [
    ('LSTM',  '7dim', 'lstm_7dim'),
    ('LSTM',  '46d',  'lstm_46d'),
    ('BiLSTM','7dim', 'bilstm_7dim'),
    ('BiLSTM','46d',  'bilstm_46d'),
    ('Mamba', '7dim', 'mamba_7dim'),
    ('Mamba', '46d',  'mamba_46d'),
    ('MambaLong','7d+micro','mamba_long'),
]


def per_fold(probs, labels, fold_idx, thr=0.5):
    out = []
    for fi in range(5):
        m = fold_idx == fi
        if m.sum() == 0:
            continue
        yi = labels[m]
        pi = probs[m]
        yh = (pi > thr).astype(int)
        out.append({
            'accuracy': float(accuracy_score(yi, yh)),
            'precision': float(precision_score(yi, yh, zero_division=0)),
            'recall': float(recall_score(yi, yh, zero_division=0)),
            'f1': float(f1_score(yi, yh, zero_division=0)),
            'auc': float(roc_auc_score(yi, pi)),
        })
    return out


def summary(metrics):
    keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    out = {}
    for k in keys:
        out[f'{k}_mean'] = float(np.mean([m[k] for m in metrics]))
        out[f'{k}_std'] = float(np.std([m[k] for m in metrics]))
    return out


def load_combo(model, features, dir_name, is_46d=False):
    if dir_name == 'mamba_long':
        base = MLONG
    else:
        base = os.path.join(OUT, dir_name)
    p = np.load(os.path.join(base, 'probs.npy'))
    if is_46d:
        p = 1.0 - p
    y = np.load(UNIFIED_Y)
    f = np.load(UNIFIED_F)
    return p, y, f


def main():
    print("=" * 80)
    print("  Mamba-Long Late Fusion (步骤 5)")
    print("=" * 80)

    y = np.load(UNIFIED_Y)
    f = np.load(UNIFIED_F)

    # 加载 7 个模型
    print("\n=== 加载 7 模型 ===")
    probs_dict = {}
    for model, features, dir_name in COMBOS_7:
        is_46d = (features == '46d')
        try:
            p, _, _ = load_combo(model, features, dir_name, is_46d)
            name = f"{model}_{features}"
            probs_dict[name] = p
            ms = per_fold(p, y, f)
            s = summary(ms)
            print(f"  ✓ {name:<22} F1={s['f1_mean']:.4f}  AUC={s['auc_mean']:.4f}")
        except Exception as e:
            print(f"  ✗ {model}_{features}: {e}")

    # ===== Late Fusion 实验 =====
    print(f"\n=== Late Fusion 实验 (7 模型候选) ===")

    names = list(probs_dict.keys())
    n = len(names)

    # 代表性候选
    candidates = [
        ('equal_7way',       [1/n]*n),
        ('b46_only',         [1 if 'BiLSTM_46d' in x else 0 for x in names]),
        ('b46+lstm46+mamba_long',
            [0.4 if 'BiLSTM_46d' in x else (0.3 if 'LSTM_46d' in x else (0.3 if 'MambaLong' in x else 0)) for x in names]),
        ('mamba_long_heavy',
            [0.6 if 'MambaLong' in x else (0.4 if 'BiLSTM_46d' in x else 0) for x in names]),
        ('all46d_+_mamba_long',
            [0.4 if '46d' in x and 'LSTM' not in x else (0.2 if 'MambaLong' in x else 0) for x in names]),
        ('best_top3_avg',
            [1/3 if x in ('BiLSTM_46d', 'LSTM_46d', 'MambaLong_7d+micro') else 0 for x in names]),
        ('top4_equal',
            [0.25 if x in ('BiLSTM_46d', 'LSTM_46d', 'Mamba_46d', 'MambaLong_7d+micro') else 0 for x in names]),
    ]
    # 归一
    candidates = [(name, [w/sum(ws) for w in ws]) for name, ws in candidates if sum(ws) > 0]

    rows = []
    for cname, ws in candidates:
        P = sum(w * probs_dict[n] for w, n in zip(ws, names))
        ms = per_fold(P, y, f)
        s = summary(ms)
        rows.append((cname, s['f1_mean'], s['auc_mean'], s['accuracy_mean'], s['precision_mean'], s['recall_mean']))
    rows.sort(key=lambda x: -x[1])

    print(f"\n  {'方案':<30} {'F1':<8} {'AUC':<8} {'Acc':<8} {'Prec':<8} {'Rec':<8}")
    print("  " + "-" * 75)
    for r in rows:
        print(f"  {r[0]:<30} {r[1]:.4f}   {r[2]:.4f}   {r[3]:.4f}   {r[4]:.4f}   {r[5]:.4f}")

    # ===== 6 路 vs 7 路对比 =====
    print(f"\n=== 6 路 vs 7 路 (加入 Mamba-Long) ===")
    if 'MambaLong_7d+micro' in probs_dict:
        # 6 路 = 不含 MambaLong
        names_6 = [n for n in names if 'MambaLong' not in n]
        # 用 best 6-way 权重 (来自 multi_scale_mamba_fusion_v2: a=0.1, b=0, c=0.4, d=0.3, e=0.2)
        ws6 = [0.1, 0, 0.4, 0.3, 0, 0.2]  # LSTM7=0.1, LSTM46=0, BiLSTM7=0.4, BiLSTM46=0.3, Mamba7=0, Mamba46=0.2
        if len(names_6) == 6:
            P6 = sum(w * probs_dict[n] for w, n in zip(ws6, names_6))
            ms6 = per_fold(P6, y, f)
            s6 = summary(ms6)
            print(f"  6 路 (best from fusion_v2): F1={s6['f1_mean']:.4f}  AUC={s6['auc_mean']:.4f}")

            # 7 路 = 6 路 + MambaLong 0.1
            ws7 = ws6 + [0.1]
            names_7 = names_6 + ['MambaLong_7d+micro']
            ws7 = [w/sum(ws7) for w in ws7]
            P7 = sum(w * probs_dict[n] for w, n in zip(ws7, names_7))
            ms7 = per_fold(P7, y, f)
            s7 = summary(ms7)
            print(f"  7 路 (+ MambaLong 0.1):    F1={s7['f1_mean']:.4f}  AUC={s7['auc_mean']:.4f}")

            # 7 路最优网格搜索 (coarse step=0.10)
            best_7 = None
            best_f1 = 0
            for ws in product(np.arange(0, 1.01, 0.10), repeat=7):
                if abs(sum(ws) - 1.0) > 1e-6:
                    continue
                P = sum(w * probs_dict[n] for w, n in zip(ws, names))
                ms = per_fold(P, y, f)
                s = summary(ms)
                if s['f1_mean'] > best_f1:
                    best_f1 = s['f1_mean']
                    best_7 = (list(ws), s)
            if best_7:
                print(f"\n  ★ 7 路网格最优: F1={best_7[1]['f1_mean']:.4f}  AUC={best_7[1]['auc_mean']:.4f}")
                print(f"     weights: {dict(zip(names, [round(w,2) for w in best_7[0]]))}")

    # ===== 保存 fusion 报告 =====
    out_dir = os.path.join(MLONG, 'fusion')
    os.makedirs(out_dir, exist_ok=True)
    report = {
        'models_in_fusion': names,
        'candidates': [
            {'name': r[0], 'f1': r[1], 'auc': r[2], 'accuracy': r[3],
             'precision': r[4], 'recall': r[5]}
            for r in rows
        ],
    }
    with open(os.path.join(out_dir, 'fusion_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ===== 可视化 =====
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r[1] for r in rows], 0.4, label='F1', color='steelblue')
    ax.bar(x + 0.2, [r[2] for r in rows], 0.4, label='AUC', color='darkorange')
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=15, ha='right')
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel('Score')
    ax.set_title('Late Fusion with Mamba-Long (步骤 5)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out_fig = os.path.join(out_dir, 'fusion_combo.png')
    plt.savefig(out_fig, dpi=120)
    plt.close()
    print(f"\n  📊 {out_fig}")
    print(f"\n{'='*80}\n  步骤 5 完成\n{'='*80}")


if __name__ == '__main__':
    main()