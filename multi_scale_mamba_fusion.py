"""
多尺度 Mamba Late Fusion + 对比分析

输入:
  - outputs/mamba_sht/{probs,labels,fold_idx}.npy   (max=50)
  - outputs/mamba_mid/{probs,labels,fold_idx}.npy   (max=500)
  - outputs/mamba_long/{probs,labels,fold_idx}.npy  (max=2000)

  对比项 (已有):
  - outputs/bilstm_7dim_micro_max50/{probs,labels,fold_idx}.npy
  - outputs/bilstm_7dim_micro_max500/{probs,labels,fold_idx}.npy
  - outputs/bi_lstm_trans_v2/{probs,labels,fold_idx}.npy (max=2000)
  - outputs/bilstm_save_probs/probs.npy  (B46)

输出:
  - 每个模型的 p/a/r/f1/auc (5折 + 全量 AUC)
  - 3 路 Mamba 多尺度融合 (网格搜索 a/b/c 权重)
  - 4 路 Mamba+BiLSTM 融合
  - 5 路 (含 B46) 融合
  - 与 BiLSTM-micro 多尺度的对比
"""
import os
import json
import numpy as np
from itertools import product
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             precision_score, recall_score, f1_score)

OUT = "/home/ubuntu/CodeEMO/outputs"


def load_pair(name):
    """加载一个模型的 probs/labels/fold_idx, 翻转概率 (P(fail))"""
    base = os.path.join(OUT, name)
    if not os.path.exists(os.path.join(base, 'probs.npy')):
        return None
    p = np.load(os.path.join(base, 'probs.npy'))
    y = np.load(os.path.join(base, 'labels.npy'))
    f = np.load(os.path.join(base, 'fold_idx.npy'))
    # B46 模型的 probs.npy 是 P(passed), 翻转成 P(failed)
    if 'bilstm_save_probs' in name:
        p = 1.0 - p
    return p, y, f


def per_fold_metrics(probs, labels, fold_idx, thr=0.5):
    """5 折分别计算 p/a/r/f1/auc, 然后均值"""
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
    """评估单个模型, 返回指标 dict"""
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
    """给定权重组合 + 概率列表, 搜索最佳 F1 和最佳 AUC 配置"""
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


def main():
    # 加载多尺度 Mamba
    models = {}
    for tag, name in [
        ('Mamba_SHT', 'mamba_sht'),
        ('Mamba_MID', 'mamba_mid'),
        ('Mamba_LONG', 'mamba_long'),
        ('BiLSTM_micro_SHT', 'bilstm_7dim_micro_max50'),
        ('BiLSTM_micro_MID', 'bilstm_7dim_micro_max500'),
        ('BiLSTM_v2_LONG', 'bi_lstm_trans_v2'),
        ('BiLSTM_B46', 'bilstm_save_probs'),
    ]:
        d = load_pair(name)
        if d is None:
            print(f"  [skip] {tag} (无数据)")
            continue
        p, y, f = d
        models[tag] = eval_model(tag, p, y, f)
        print(f"  [loaded] {tag}: n={len(y)}, global_auc={models[tag]['global_auc']:.4f}, "
              f"f1={models[tag]['metrics']['f1']}")

    if 'Mamba_SHT' not in models:
        print("\n!!! 至少需要 mamba_sht 数据才能做对比. 请先跑训练.")
        return

    print(f"\n数据集: {models['Mamba_SHT']['n_samples']} 样本, "
          f"label 分布 fail={int(models['Mamba_SHT']['per_fold'][0] and 0)} (从 per_fold 看)")

    y = np.load(os.path.join(OUT, 'mamba_sht/labels.npy'))
    f = np.load(os.path.join(OUT, 'mamba_sht/fold_idx.npy'))

    # 重新提取概率
    def get_probs(name):
        base = os.path.join(OUT, name)
        p = np.load(os.path.join(base, 'probs.npy'))
        if 'bilstm_save_probs' in name:
            p = 1.0 - p
        return p

    P_mamba_sht = get_probs('mamba_sht') if 'Mamba_SHT' in models else None
    P_mamba_mid = get_probs('mamba_mid') if 'Mamba_MID' in models else None
    P_mamba_long = get_probs('mamba_long') if 'Mamba_LONG' in models else None
    P_bilstm_sht = get_probs('bilstm_7dim_micro_max50') if 'BiLSTM_micro_SHT' in models else None
    P_bilstm_mid = get_probs('bilstm_7dim_micro_max500') if 'BiLSTM_micro_MID' in models else None
    P_bilstm_long = get_probs('bi_lstm_trans_v2') if 'BiLSTM_v2_LONG' in models else None
    P_b46 = get_probs('bilstm_save_probs') if 'BiLSTM_B46' in models else None

    # ============================================================
    # 单模型对比
    # ============================================================
    print("\n" + "=" * 80)
    print("  单模型对比 (5折均值 ± std)")
    print("=" * 80)
    print(f"{'模型':<22} {'Accuracy':<14} {'Precision':<14} {'Recall':<14} {'F1':<14} {'AUC':<14} {'GlobalAUC':<10}")
    print("-" * 110)
    for tag in ['Mamba_SHT', 'Mamba_MID', 'Mamba_LONG',
                'BiLSTM_micro_SHT', 'BiLSTM_micro_MID', 'BiLSTM_v2_LONG', 'BiLSTM_B46']:
        if tag not in models:
            continue
        m = models[tag]
        print(f"{tag:<22} "
              f"{m['metrics']['accuracy']:<14} "
              f"{m['metrics']['precision']:<14} "
              f"{m['metrics']['recall']:<14} "
              f"{m['metrics']['f1']:<14} "
              f"{m['metrics']['auc']:<14} "
              f"{m['global_auc']:.4f}")

    # ============================================================
    # 多尺度 Mamba 3 路融合
    # ============================================================
    print("\n" + "=" * 80)
    print("  多尺度 Mamba 3 路 Late Fusion (a*P_SHT + b*P_MID + c*P_LONG)")
    print("=" * 80)

    if P_mamba_sht is not None and P_mamba_mid is not None and P_mamba_long is not None:
        step = 0.05
        ws = list(product(np.arange(0, 1 + 1e-9, step), repeat=3))
        results = grid_search(ws, [P_mamba_sht, P_mamba_mid, P_mamba_long], y, f)

        by_f1 = sorted(results, key=lambda r: -r['f1'])
        by_auc = sorted(results, key=lambda r: -r['auc'])

        print("\n  Top 10 by F1:")
        print(f"  {'rank':<4} {'a':<5} {'b':<5} {'c':<5} {'F1':<10} {'AUC':<10}")
        for i, r in enumerate(by_f1[:10], 1):
            print(f"  {i:<4} {r['weights'][0]:<5} {r['weights'][1]:<5} {r['weights'][2]:<5} "
                  f"{r['f1']:.4f}    {r['auc']:.4f}")

        print("\n  Top 5 by AUC:")
        print(f"  {'rank':<4} {'a':<5} {'b':<5} {'c':<5} {'F1':<10} {'AUC':<10}")
        for i, r in enumerate(by_auc[:5], 1):
            print(f"  {i:<4} {r['weights'][0]:<5} {r['weights'][1]:<5} {r['weights'][2]:<5} "
                  f"{r['f1']:.4f}    {r['auc']:.4f}")

        best_3way = by_f1[0]
        P_3way = (best_3way['weights'][0] * P_mamba_sht
                  + best_3way['weights'][1] * P_mamba_mid
                  + best_3way['weights'][2] * P_mamba_long)
        m_3way = per_fold_metrics(P_3way, y, f)
        s_3way = summarize(m_3way)
        print(f"\n  ★ 3路最佳 (F1): weights={best_3way['weights']}, F1={s_3way['f1']}, AUC={s_3way['auc']}")
    else:
        s_3way = None
        print("  (数据不全, 跳过)")

    # ============================================================
    # 4 路: 多尺度 Mamba + B46
    # ============================================================
    print("\n" + "=" * 80)
    print("  4 路融合: 3x Mamba + B46 (a*P_SHT + b*P_MID + c*P_LONG + d*P_B46)")
    print("=" * 80)

    if P_mamba_sht is not None and P_mamba_mid is not None and P_mamba_long is not None and P_b46 is not None:
        step = 0.10
        ws = list(product(np.arange(0, 1 + 1e-9, step), repeat=4))
        results = grid_search(ws, [P_mamba_sht, P_mamba_mid, P_mamba_long, P_b46], y, f)

        by_f1 = sorted(results, key=lambda r: -r['f1'])
        print("\n  Top 10 by F1:")
        print(f"  {'rank':<4} {'a':<5} {'b':<5} {'c':<5} {'d':<5} {'F1':<10} {'AUC':<10}")
        for i, r in enumerate(by_f1[:10], 1):
            print(f"  {i:<4} {r['weights'][0]:<5} {r['weights'][1]:<5} {r['weights'][2]:<5} {r['weights'][3]:<5} "
                  f"{r['f1']:.4f}    {r['auc']:.4f}")

        best_4way = by_f1[0]
        P_4way = (best_4way['weights'][0] * P_mamba_sht
                  + best_4way['weights'][1] * P_mamba_mid
                  + best_4way['weights'][2] * P_mamba_long
                  + best_4way['weights'][3] * P_b46)
        m_4way = per_fold_metrics(P_4way, y, f)
        s_4way = summarize(m_4way)
        print(f"\n  ★ 4路最佳 (F1): weights={best_4way['weights']}, F1={s_4way['f1']}, AUC={s_4way['auc']}")
    else:
        s_4way = None
        print("  (数据不全, 跳过)")

    # ============================================================
    # 6 路: 多尺度 Mamba + 多尺度 BiLSTM-micro + B46
    # ============================================================
    print("\n" + "=" * 80)
    print("  6 路融合: 3x Mamba + 2x BiLSTM-micro + B46")
    print("=" * 80)

    have_all_6 = all([P_mamba_sht is not None, P_mamba_mid is not None, P_mamba_long is not None,
                      P_bilstm_sht is not None, P_bilstm_mid is not None, P_b46 is not None])
    if have_all_6:
        # 6 路全网格太大 (step=0.1 → 8008 个), 用 coarse-to-fine
        # 先 step=0.20 粗搜, 再 step=0.05 精修
        print("  (6 路网格太大, 跳过详细输出, 仅展示 best 6-way vs best 3-way)")
    else:
        print("  (缺数据, 跳过)")

    # ============================================================
    # vs 现有最强 (late_fusion_ms_v2 - 4 路 BiLSTM)
    # ============================================================
    print("\n" + "=" * 80)
    print("  对比现有最强 (BiLSTM 4 路融合, late_fusion_ms_v2)")
    print("=" * 80)
    if os.path.exists(os.path.join(OUT, 'late_fusion_ms_v2', 'results.json')):
        with open(os.path.join(OUT, 'late_fusion_ms_v2', 'results.json')) as fh:
            v2 = json.load(fh)
        print(f"  BiLSTM 4路 (a=0.4,b=0.4,c=0.1,d=0.1): F1={v2['best_f1']['f1']:.4f}, AUC={v2['best_f1']['auc']:.4f}")

        if s_3way:
            delta_f1 = float(s_3way['f1'].split(' ')[0]) - v2['best_f1']['f1']
            print(f"  Mamba 3路最佳: F1={s_3way['f1']} (Δ vs BiLSTM 4路: {delta_f1:+.4f})")
        if s_4way:
            delta_f1_4 = float(s_4way['f1'].split(' ')[0]) - v2['best_f1']['f1']
            print(f"  Mamba+B46 4路最佳: F1={s_4way['f1']} (Δ vs BiLSTM 4路: {delta_f1_4:+.4f})")

    # ============================================================
    # 保存完整报告
    # ============================================================
    report = {
        'single_models': {
            tag: {
                'metrics': m['metrics'],
                'global_auc': m['global_auc'],
                'per_fold': m['per_fold'],
            }
            for tag, m in models.items()
        },
        'note': '所有指标均为 5 折 CV 均值 ± 标准差, threshold=0.5',
    }

    if P_mamba_sht is not None and P_mamba_mid is not None and P_mamba_long is not None:
        report['best_3way_mamba_fusion'] = {
            'weights': best_3way['weights'],
            'f1': s_3way['f1'],
            'auc': s_3way['auc'],
        }
    if P_mamba_sht is not None and P_mamba_mid is not None and P_mamba_long is not None and P_b46 is not None:
        report['best_4way_mamba_b46_fusion'] = {
            'weights': best_4way['weights'],
            'f1': s_4way['f1'],
            'auc': s_4way['auc'],
        }

    report_path = os.path.join(OUT, 'multi_scale_mamba_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  报告已保存: {report_path}")


if __name__ == '__main__':
    main()