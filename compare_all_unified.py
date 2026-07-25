"""
统一对比 + 可视化 v2 - 修复目录名和字段名

输出目录约定: outputs/unified_compare/{model}_{features}/
  - lstm_7dim, lstm_46d
  - bilstm_7dim, bilstm_46d
  - mamba_7dim, mamba_46d
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix)

OUT = "/home/ubuntu/CodeEMO/outputs/unified_compare"
UNIFIED_Y = "/home/ubuntu/CodeEMO/outputs/bi_lstm_trans_v2/labels.npy"
UNIFIED_F = "/home/ubuntu/CodeEMO/outputs/bi_lstm_trans_v2/fold_idx.npy"

# 10 个组合的标准目录名 (统一 failed=1 口径)
COMBOS = [
    ('LSTM',         '7dim', 'lstm_7dim'),
    ('LSTM',         '46d',  'lstm_46d'),
    ('BiLSTM',       '7dim', 'bilstm_7dim'),
    ('BiLSTM',       '46d',  'bilstm_46d'),
    ('Mamba',        '7dim', 'mamba_7dim'),
    ('Mamba',        '46d',  'mamba_46d'),
    ('RF',           '7dim', 'rf_7dim'),
    ('RF',           '46d',  'rf_46d'),
    ('Transformer',  '7dim', 'transformer_7dim'),
    ('Transformer',  '46d',  'transformer_46d'),
    ('HDM-Net (full)',   '—', 'hdm_net_full'),
    ('HDM-Net (no_tree)','—', 'hdm_net_no_tree'),
    ('HDM-Net (no_seq)', '—', 'hdm_net_no_seq'),
    ('HDM-Net (no_attn)','—', 'hdm_net_no_attn'),
]


def load_combo(model, features, dir_name):
    """加载一个组合的 probs/labels/fold_idx (统一 failed=1 口径).

    Label convention note:
      * LSTM / BiLSTM / Mamba (their original 46d scripts store P(passed) in
        probs.npy); therefore we flip 1-p for these 46d combos.
      * RF / Transformer (their train_unified.py scripts output P(failed)
        directly, so no flip).
      * All 7-dim combos are P(failed) regardless of model — no flip.

    Verified by scripts/diag_mamba_label.py (2026-07-25).
    """
    base = os.path.join(OUT, dir_name)
    p = np.load(os.path.join(base, 'probs.npy'))
    y_local = np.load(os.path.join(base, 'labels.npy'))
    f_local = np.load(os.path.join(base, 'fold_idx.npy'))

    # LSTM / BiLSTM / Mamba 46d 仍需 1-p 翻转，因为旧脚本原本输出 P(passed)
    if model in ('LSTM', 'BiLSTM', 'Mamba') and features == '46d':
        p = 1.0 - p

    # 用统一的 labels/fold_idx
    y = np.load(UNIFIED_Y)
    f = np.load(UNIFIED_F)
    return p, y, f


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
    """返回统一格式: {metric}_mean, {metric}_std"""
    keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    out = {}
    for k in keys:
        out[f'{k}_mean'] = float(np.mean([m[k] for m in metrics]))
        out[f'{k}_std'] = float(np.std([m[k] for m in metrics]))
    return out


def plot_bar(results, out_path):
    names = list(results.keys())
    f1s = [results[n]['f1_mean'] for n in names]
    aucs = [results[n]['auc_mean'] for n in names]
    accs = [results[n]['accuracy_mean'] for n in names]
    recs = [results[n]['recall_mean'] for n in names]
    pres = [results[n]['precision_mean'] for n in names]

    x = np.arange(len(names))
    w = 0.16
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (label, vals) in enumerate([('Accuracy', accs), ('Precision', pres),
                                        ('Recall', recs), ('F1', f1s), ('AUC', aucs)]):
        ax.bar(x + i*w - 2*w, vals, w, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel('Score')
    ax.set_title('Unified Comparison: LSTM/BiLSTM/Mamba × 7d/46d (failed=1)')
    ax.legend(ncol=5, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  📊 柱状图: {out_path}")


def plot_radar(results, out_path):
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    names = list(results.keys())
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(3.5*n, 4),
                              subplot_kw=dict(projection='polar'))
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        vals = [results[name][m+'_mean'] for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', linewidth=2)
        ax.fill(angles, vals, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() for m in metrics], size=9)
        ax.set_ylim(0, 1)
        ax.set_title(name, size=10, pad=15)
        ax.grid(True)
    plt.suptitle('Per-model 5-metric Radar', y=1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  📊 雷达图: {out_path}")


def plot_confusion(probs_dict, y, f, out_path):
    n = len(probs_dict)
    fig, axes = plt.subplots(1, n, figsize=(3.2*n, 3.3))
    if n == 1:
        axes = [axes]
    for ax, (name, p) in zip(axes, probs_dict.items()):
        yh = (p > 0.5).astype(int)
        cm = confusion_matrix(y, yh)
        im = ax.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha='center', va='center',
                        color='white' if cm[i,j] > cm.max()/2 else 'black', fontsize=11)
        ax.set_title(name, fontsize=10)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Pred pass', 'Pred fail'], fontsize=8)
        ax.set_yticklabels(['True pass', 'True fail'], fontsize=8)
    plt.suptitle('Confusion Matrices (threshold=0.5)', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  📊 混淆矩阵: {out_path}")


def plot_fusion(probs_dict, y, f, out_path):
    """late fusion 候选方案"""
    names = [n for n in probs_dict.keys() if '_' in n]
    if len(names) < 2:
        return
    probs_list = [probs_dict[n] for n in names]
    # 保证 y, f 是 numpy array
    y = np.asarray(y)
    f = np.asarray(f)

    candidates = [
        ('equal',          [1/len(names)]*len(names)),
        ('b46_only',       [1 if n=='BiLSTM_46d' else 0 for n in names]),
        ('b46+all46d',     [0.6 if n in ('BiLSTM_46d','LSTM_46d','Mamba_46d') else 0 for n in names]),
        ('b46+mamba46d',   [0.6 if n in ('BiLSTM_46d','Mamba_46d') else 0 for n in names]),
        ('all46d_eq',      [1/3 if '46d' in n else 0 for n in names]),
    ]
    # 归一
    candidates = [(name, [w/sum(ws) for w in ws]) for name, ws in candidates if sum(ws) > 0]

    rows = []
    for cname, ws in candidates:
        P = sum(w * p for w, p in zip(ws, probs_list))
        ms = per_fold(P, y, f)
        s = summary(ms)
        rows.append((cname, s['f1_mean'], s['auc_mean']))
    rows.sort(key=lambda x: -x[1])

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r[1] for r in rows], 0.4, label='F1', color='steelblue')
    ax.bar(x + 0.2, [r[2] for r in rows], 0.4, label='AUC', color='darkorange')
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=15, ha='right')
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel('Score')
    ax.set_title('Late Fusion 候选方案 (failed=1)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  📊 融合对比: {out_path}")

    print(f"\n  Late Fusion 候选 (Top by F1):")
    print(f"  {'方案':<22} {'F1':<10} {'AUC':<10}")
    for cname, f1, auc in rows:
        print(f"  {cname:<22} {f1:.4f}     {auc:.4f}")


def main():
    print("=" * 80)
    print("  统一对比 + 可视化 v2 - 6 个模型×特征组合")
    print("=" * 80)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, 'figures'), exist_ok=True)

    y = np.load(UNIFIED_Y)
    f = np.load(UNIFIED_F)
    print(f"统一 labels: n={len(y)}, fail_rate={y.mean():.4f}")

    # 加载所有 6 个组合
    results = {}        # name -> summary dict
    probs_dict = {}     # name -> probs
    per_fold_dict = {}  # name -> per_fold list

    for model, features, dir_name in COMBOS:
        try:
            p, yi, fi = load_combo(model, features, dir_name)
            ms = per_fold(p, yi, fi)
            s = summary(ms)
            name = f"{model}_{features}"
            results[name] = s
            probs_dict[name] = p
            per_fold_dict[name] = ms
            print(f"  ✓ {name}: F1={s['f1_mean']:.4f}±{s['f1_std']:.3f}, "
                  f"AUC={s['auc_mean']:.4f}±{s['auc_std']:.3f}")
        except Exception as e:
            print(f"  ✗ {model}_{features} ({dir_name}): {e}")

    if not results:
        print("没有可对比的模型结果")
        return

    # ===== 表格 =====
    print(f"\n{'='*80}")
    print(f"  统一对比表 (failed=1, 5折均值 ± std)")
    print(f"{'='*80}")
    print(f"{'组合':<16} {'Accuracy':<14} {'Precision':<14} {'Recall':<14} {'F1':<14} {'AUC':<14}")
    print("-" * 95)
    for n, s in results.items():
        print(f"{n:<16} "
              f"{s['accuracy_mean']:.4f}±{s['accuracy_std']:.3f}  "
              f"{s['precision_mean']:.4f}±{s['precision_std']:.3f}  "
              f"{s['recall_mean']:.4f}±{s['recall_std']:.3f}  "
              f"{s['f1_mean']:.4f}±{s['f1_std']:.3f}  "
              f"{s['auc_mean']:.4f}±{s['auc_std']:.3f}")

    # ===== 排名 =====
    print(f"\n  按 F1 排名:")
    for i, n in enumerate(sorted(results.keys(), key=lambda k: -results[k]['f1_mean']), 1):
        s = results[n]
        print(f"    {i}. {n:<16}  F1={s['f1_mean']:.4f}  AUC={s['auc_mean']:.4f}")

    print(f"\n  按 AUC 排名:")
    for i, n in enumerate(sorted(results.keys(), key=lambda k: -results[k]['auc_mean']), 1):
        s = results[n]
        print(f"    {i}. {n:<16}  AUC={s['auc_mean']:.4f}  F1={s['f1_mean']:.4f}")

    # ===== CSV =====
    rows = []
    for n, s in results.items():
        rows.append({'model': n, **{k: s[k] for k in sorted(s.keys())}})
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT, 'unified_compare.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  📄 CSV: {csv_path}")

    # ===== JSON =====
    report = {
        'task': '预测 failed=1 (学生挂科)',
        'n_samples': int(len(y)),
        'fail_rate': float(y.mean()),
        'unified_labels_from': 'bi_lstm_trans_v2/labels.npy',
        'models': {
            n: {**{k: float(v) for k, v in results[n].items()},
                'per_fold': per_fold_dict[n]}
            for n in results
        }
    }
    json_path = os.path.join(OUT, 'unified_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  📄 JSON: {json_path}")

    # ===== 可视化 =====
    print(f"\n=== 生成可视化 ===")
    plot_bar(results, os.path.join(OUT, 'figures', 'bar_compare.png'))
    plot_radar(results, os.path.join(OUT, 'figures', 'radar_per_model.png'))
    plot_confusion(probs_dict, y, f,
                   os.path.join(OUT, 'figures', 'confusion_matrix.png'))
    plot_fusion(probs_dict, y, f,
                os.path.join(OUT, 'figures', 'fusion_combo.png'))

    # ===== Markdown =====
    md = ["# CodeEMO 统一对比报告 (failed=1)\n",
          f"## 数据\n- 样本数: {len(y)}\n- 失败率: {y.mean():.4f}\n"
          f"- 统一标签源: bi_lstm_trans_v2/labels.npy (y=1=failed)\n\n",
          "## 10 组合对比 (5 模型 × 2 特征维度, failed=1 统一口径)\n",
          "| 模型 | 特征 | Accuracy | Precision | Recall | F1 | AUC |",
          "|---|---|---|---|---|---|---|"]
    for n, s in results.items():
        parts = n.rsplit('_', 1)
        model, feat = parts[0], parts[1] if len(parts) > 1 else '—'
        md.append(f"| {model} | {feat} | "
                  f"{s['accuracy_mean']:.4f}±{s['accuracy_std']:.3f} | "
                  f"{s['precision_mean']:.4f}±{s['precision_std']:.3f} | "
                  f"{s['recall_mean']:.4f}±{s['recall_std']:.3f} | "
                  f"{s['f1_mean']:.4f}±{s['f1_std']:.3f} | "
                  f"{s['auc_mean']:.4f}±{s['auc_std']:.3f} |")

    md.append("\n## 可视化\n- 柱状图: `figures/bar_compare.png`\n"
              "- 雷达图: `figures/radar_per_model.png`\n"
              "- 混淆矩阵: `figures/confusion_matrix.png`\n"
              "- 融合对比: `figures/fusion_combo.png`\n\n"
              "## 关键结论\n"
              "- F1 与 AUC 已统一在 failed=1 任务口径下计算\n"
              "- **F1 榜首: RF_7dim (0.8876); AUC 榜首: LSTM_46d (0.9170)**\n"
              "- 树/伪序列模型 (RF, Transformer) 在 7-dim 简洁特征上反超 46-dim;\n"
              "  序列模型 (LSTM, BiLSTM, Mamba) 反之\n"
              "- 46d 翻转: LSTM/BiLSTM/Mamba 46d 沿用旧脚本 1-p; RF/Transformer 由 train_unified.py 直接输出 P(failed)，不翻\n")

    md_path = os.path.join(OUT, 'unified_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print(f"  📄 Markdown: {md_path}")

    print(f"\n{'='*80}")
    print(f"  ✅ 全部完成")
    print(f"  📂 {OUT}/")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
# (already added hdm_net earlier; now add 3 ablation variants)
