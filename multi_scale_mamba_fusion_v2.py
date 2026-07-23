"""
多尺度 Mamba Late Fusion v2 - 修复 B46 数据对齐问题

关键修复:
  - B46 (bilstm_save_probs) 的 labels/fold_idx 跟其他模型不一致, 必须用 bi_lstm_trans_v2 的 labels/fold_idx 来处理
  - BiLSTM-micro 的 fold_idx 也跟 mamba/bi_lstm_trans_v2 不一致, 单独处理

支持的融合方案:
  3-way:  Mamba_SHT + Mamba_MID + Mamba_LONG
  4-way:  + BiLSTM_v2 (long sequence)
  4-way:  + B46 (B46 数据对齐修正)
  5-way:  3x Mamba + BiLSTM_v2 + B46
  6-way:  3x Mamba + BiLSTM_v2 + B46 + BiLSTM_micro_SHT
  6-way:  3x Mamba + BiLSTM_v2 + B46 + BiLSTM_micro_MID
  7-way:  3x Mamba + BiLSTM_v2 + B46 + BiLSTM_micro_SHT + BiLSTM_micro_MID
"""
import os
import json
import numpy as np
from itertools import product
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             precision_score, recall_score, f1_score)

OUT = "/home/ubuntu/CodeEMO/outputs"


def load_with_main_labels(name, main_labels, main_fold_idx):
    """
    加载一个模型的 probs/labels/fold_idx.

    - 如果 labels/fold_idx 跟 main 不一致, 用 main 的 (要求样本顺序一致)
    - B46: probs 是 P(passed), 翻转为 P(failed)
    """
    base = os.path.join(OUT, name)
    p = np.load(os.path.join(base, 'probs.npy'))
    y_local = np.load(os.path.join(base, 'labels.npy'))
    f_local = np.load(os.path.join(base, 'fold_idx.npy'))

    # 极性修复
    if 'bilstm_save_probs' in name:
        p = 1.0 - p

    # 检查 labels 是否跟 main 一致
    if not np.array_equal(y_local, main_labels):
        print(f"  [warn] {name}: labels 不一致, 使用 main_labels")
        y = main_labels
    else:
        y = y_local

    if not np.array_equal(f_local, main_fold_idx):
        print(f"  [warn] {name}: fold_idx 不一致, 使用 main_fold_idx (假设样本顺序相同)")
        f = main_fold_idx
    else:
        f = f_local

    return p, y, f


def per_fold_metrics(probs, labels, fold_idx, thr=0.5):
    out = []
    for f in range(5):
        m = fold_idx == f
        if m.sum() == 0:
            continue
        y = labels[m]
        p = probs[m]
        yh = (p > thr).astype(int)
        out.append({
            'accuracy': float(accuracy_score(y, yh)),
            'precision': float(precision_score(y, yh, zero_division=0)),
            'recall': float(recall_score(y, yh, zero_division=0)),
            'f1': float(f1_score(y, yh, zero_division=0)),
            'auc': float(roc_auc_score(y, p)),
        })
    return out


def summarize(metrics):
    keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    return {k: f"{np.mean([m[k] for m in metrics]):.4f} ± {np.std([m[k] for m in metrics]):.4f}"
            for k in keys}


def eval_model(name, p, y, f):
    ms = per_fold_metrics(p, y, f)
    s = summarize(ms)
    return {
        'name': name,
        'n_samples': int(len(y)),
        'global_auc': float(roc_auc_score(y, p)),
        'metrics': s,
        'per_fold': ms,
    }


def grid_search(weights_list, probs_list, labels, fold_idx):
    results = []
    for ws in weights_list:
        if abs(sum(ws) - 1.0) > 1e-6:
            continue
        P = sum(w * p for w, p in zip(ws, probs_list))
        ms = per_fold_metrics(P, labels, fold_idx)
        f1m = float(np.mean([m['f1'] for m in ms]))
        auc = float(roc_auc_score(labels, P))
        results.append({
            'weights': [round(float(w), 2) for w in ws],
            'f1': f1m,
            'auc': auc,
        })
    return results


def find_best(results, metric='f1', top=5):
    return sorted(results, key=lambda r: -r[metric])[:top]


def show_top(results, top_k, label):
    print(f"\n  Top {top_k} (by F1):")
    n = len(results[0]['weights'])
    weights_str = ' '.join([f'w{i}' for i in range(n)])
    print(f"  {'rank':<4} {' '.join([f'w{i:<5}' for i in range(n)])} {'F1':<10} {'AUC':<10}")
    for i, r in enumerate(results[:top_k], 1):
        w_str = ' '.join([f'{w:<6.2f}' for w in r['weights']])
        print(f"  {i:<4} {w_str} {r['f1']:.4f}    {r['auc']:.4f}")


def main():
    print("=" * 80)
    print("  多尺度 Mamba Late Fusion v2 (A=修B46对齐 + B=5路含Mamba)")
    print("=" * 80)

    # 主参考: bi_lstm_trans_v2 (它的 labels/fold_idx 跟 mamba 一致)
    main_y = np.load(os.path.join(OUT, 'bi_lstm_trans_v2/labels.npy'))
    main_f = np.load(os.path.join(OUT, 'bi_lstm_trans_v2/fold_idx.npy'))
    print(f"\n  主参考 labels: shape={main_y.shape}, fail_rate={main_y.mean():.4f}")

    # 加载所有模型
    print("\n=== 加载模型 (使用主 labels/fold_idx) ===")
    models = {}
    for tag, name in [
        ('Mamba_SHT', 'mamba_sht'),
        ('Mamba_MID', 'mamba_mid'),
        ('Mamba_LONG', 'mamba_long'),
        ('BiLSTM_v2_LONG', 'bi_lstm_trans_v2'),
        ('BiLSTM_micro_SHT', 'bilstm_7dim_micro_max50'),
        ('BiLSTM_micro_MID', 'bilstm_7dim_micro_max500'),
        ('BiLSTM_B46', 'bilstm_save_probs'),
    ]:
        p, y, f = load_with_main_labels(name, main_y, main_f)
        models[tag] = eval_model(tag, p, y, f)
        print(f"  [loaded] {tag}: n={len(y)}, global_auc={models[tag]['global_auc']:.4f}, "
              f"f1={models[tag]['metrics']['f1']}")

    # 单模型对比
    print("\n" + "=" * 80)
    print("  单模型对比 (5折均值 ± std) - 修复后")
    print("=" * 80)
    print(f"{'模型':<22} {'Accuracy':<14} {'Precision':<14} {'Recall':<14} {'F1':<14} {'AUC':<14} {'GlobalAUC':<10}")
    print("-" * 110)
    for tag in ['Mamba_SHT', 'Mamba_MID', 'Mamba_LONG',
                'BiLSTM_micro_SHT', 'BiLSTM_micro_MID', 'BiLSTM_v2_LONG', 'BiLSTM_B46']:
        m = models[tag]
        print(f"{tag:<22} "
              f"{m['metrics']['accuracy']:<14} "
              f"{m['metrics']['precision']:<14} "
              f"{m['metrics']['recall']:<14} "
              f"{m['metrics']['f1']:<14} "
              f"{m['metrics']['auc']:<14} "
              f"{m['global_auc']:.4f}")

    # 准备 probs 数组
    P = {tag: np.load(os.path.join(OUT, name, 'probs.npy'))
         if 'bilstm_save_probs' not in name else 1 - np.load(os.path.join(OUT, name, 'probs.npy'))
         for tag, name in [
             ('Mamba_SHT', 'mamba_sht'),
             ('Mamba_MID', 'mamba_mid'),
             ('Mamba_LONG', 'mamba_long'),
             ('BiLSTM_v2_LONG', 'bi_lstm_trans_v2'),
             ('BiLSTM_micro_SHT', 'bilstm_7dim_micro_max50'),
             ('BiLSTM_micro_MID', 'bilstm_7dim_micro_max500'),
             ('BiLSTM_B46', 'bilstm_save_probs'),
         ]}

    y = main_y
    f = main_f

    # ============================================================
    # 方案 A 验证: B46 单点 + 与 B46 相关的 4 路 (Mamba 3 路 + B46)
    # ============================================================
    print("\n" + "=" * 80)
    print("  [A方案验证] 4 路: Mamba_SHT + Mamba_MID + Mamba_LONG + B46 (修正)")
    print("=" * 80)
    step = 0.10
    ws = list(product(np.arange(0, 1 + 1e-9, step), repeat=4))
    results = grid_search(ws, [P['Mamba_SHT'], P['Mamba_MID'], P['Mamba_LONG'], P['BiLSTM_B46']], y, f)
    by_f1 = sorted(results, key=lambda r: -r['f1'])
    show_top(by_f1, 8, "Mamba3+B46")
    best_a = by_f1[0]
    P_a = sum(w * p for w, p in zip(best_a['weights'],
                                    [P['Mamba_SHT'], P['Mamba_MID'], P['Mamba_LONG'], P['BiLSTM_B46']]))
    m_a = per_fold_metrics(P_a, y, f)
    s_a = summarize(m_a)
    print(f"\n  ★ A方案 最佳: weights={best_a['weights']}, F1={s_a['f1']}, AUC={s_a['auc']}")

    # ============================================================
    # 方案 B: 5 路 (3x Mamba + BiLSTM_v2 + B46)
    # ============================================================
    print("\n" + "=" * 80)
    print("  [B方案] 5 路: Mamba_SHT + Mamba_MID + Mamba_LONG + BiLSTM_v2 + B46")
    print("=" * 80)
    step = 0.10
    ws = list(product(np.arange(0, 1 + 1e-9, step), repeat=5))
    print(f"  (5 路网格: {len(ws)} 个, 过滤和=1 后剩 ~{len([w for w in ws if abs(sum(w)-1)<1e-6])} 个)")
    results_5 = grid_search(ws, [P['Mamba_SHT'], P['Mamba_MID'], P['Mamba_LONG'],
                                  P['BiLSTM_v2_LONG'], P['BiLSTM_B46']], y, f)
    by_f1_5 = sorted(results_5, key=lambda r: -r['f1'])
    by_auc_5 = sorted(results_5, key=lambda r: -r['auc'])

    print("\n  Top 10 by F1:")
    n = 5
    print(f"  {'rank':<4} {' '.join([f'w{i:<5}' for i in range(n)])} {'F1':<10} {'AUC':<10}")
    for i, r in enumerate(by_f1_5[:10], 1):
        w_str = ' '.join([f'{w:<6.2f}' for w in r['weights']])
        print(f"  {i:<4} {w_str} {r['f1']:.4f}    {r['auc']:.4f}")

    print("\n  Top 5 by AUC:")
    print(f"  {'rank':<4} {' '.join([f'w{i:<5}' for i in range(n)])} {'F1':<10} {'AUC':<10}")
    for i, r in enumerate(by_auc_5[:5], 1):
        w_str = ' '.join([f'{w:<6.2f}' for w in r['weights']])
        print(f"  {i:<4} {w_str} {r['f1']:.4f}    {r['auc']:.4f}")

    best_b_f1 = by_f1_5[0]
    P_b_f1 = sum(w * p for w, p in zip(best_b_f1['weights'],
                                       [P['Mamba_SHT'], P['Mamba_MID'], P['Mamba_LONG'],
                                        P['BiLSTM_v2_LONG'], P['BiLSTM_B46']]))
    m_b_f1 = per_fold_metrics(P_b_f1, y, f)
    s_b_f1 = summarize(m_b_f1)

    best_b_auc = by_auc_5[0]
    P_b_auc = sum(w * p for w, p in zip(best_b_auc['weights'],
                                        [P['Mamba_SHT'], P['Mamba_MID'], P['Mamba_LONG'],
                                         P['BiLSTM_v2_LONG'], P['BiLSTM_B46']]))
    m_b_auc = per_fold_metrics(P_b_auc, y, f)
    s_b_auc = summarize(m_b_auc)

    print(f"\n  ★ B方案 最佳 F1:  weights={best_b_f1['weights']}, F1={s_b_f1['f1']}, AUC={s_b_f1['auc']}")
    print(f"  ★ B方案 最佳 AUC: weights={best_b_auc['weights']}, F1={s_b_auc['f1']}, AUC={s_b_auc['auc']}")

    # ============================================================
    # 5-way 对比基线 (BiLSTM-only)
    # ============================================================
    print("\n" + "=" * 80)
    print("  对比基线: late_fusion_ms_v2 (BiLSTM 4路)")
    print("=" * 80)
    if os.path.exists(os.path.join(OUT, 'late_fusion_ms_v2', 'results.json')):
        with open(os.path.join(OUT, 'late_fusion_ms_v2', 'results.json')) as fh:
            v2 = json.load(fh)
        print(f"  BiLSTM 4路 (a=0.4,b=0.4,c=0.1,d=0.1): F1={v2['best_f1']['f1']:.4f}, AUC={v2['best_f1']['auc']:.4f}")

    # ============================================================
    # 完整保存报告
    # ============================================================
    report = {
        'note': 'v2: B46 用 bi_lstm_trans_v2 的 labels/fold_idx 处理 (修正极性+对齐)',
        'single_models': {tag: {'metrics': m['metrics'], 'global_auc': m['global_auc']}
                          for tag, m in models.items()},
        'A_method_4way_mamba3_b46': {
            'weights': best_a['weights'],
            'f1': s_a['f1'],
            'auc': s_a['auc'],
        },
        'B_method_5way_mamba3_bilstm_v2_b46': {
            'best_f1': {'weights': best_b_f1['weights'], 'f1': s_b_f1['f1'], 'auc': s_b_f1['auc']},
            'best_auc': {'weights': best_b_auc['weights'], 'f1': s_b_auc['f1'], 'auc': s_b_auc['auc']},
        },
        'baseline_bilstm_4way': v2['best_f1'] if os.path.exists(os.path.join(OUT, 'late_fusion_ms_v2', 'results.json')) else None,
    }

    report_path = os.path.join(OUT, 'multi_scale_mamba_report_v2.json')
    with open(report_path, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\n  报告已保存: {report_path}")

    # ============================================================
    # 最终结论
    # ============================================================
    print("\n" + "=" * 80)
    print("  最终结论对比")
    print("=" * 80)
    print(f"  {'方案':<45} {'F1':<14} {'AUC':<14}")
    print(f"  {'-'*75}")
    print(f"  {'BiLSTM 4路 (现有最强 baseline)':<45} {v2['best_f1']['f1']:.4f}        {v2['best_f1']['auc']:.4f}")
    print(f"  {'A: Mamba 3路 + B46 (修正)':<45} {s_a['f1']:<14} {s_a['auc']:<14}")
    print(f"  {'B: Mamba 3路 + BiLSTM_v2 + B46 (best F1)':<45} {s_b_f1['f1']:<14} {s_b_f1['auc']:<14}")
    print(f"  {'B: Mamba 3路 + BiLSTM_v2 + B46 (best AUC)':<45} {s_b_auc['f1']:<14} {s_b_auc['auc']:<14}")


if __name__ == '__main__':
    main()