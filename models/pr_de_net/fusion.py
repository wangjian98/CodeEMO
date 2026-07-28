"""
PR-DE-Net 融合分析:
  1. 把 PR-DE-Net 加到 Weighted 1/3/1 融合中
  2. 在 HDM-Net v2 + RF-7d + PR-DE-Net 三方融合上找最优权重
  3. 输出和现有最强方案 (0.9009) 的对比
"""
import os, sys, json, numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.evaluator import evaluate, summarize_fold_results, print_results_table

# load labels & fold_idx (consistent across all)
labels = np.load('outputs/pr_de_net/full/labels.npy')
fold_idx = np.load('outputs/pr_de_net/full/fold_idx.npy')


def load_probs(path):
    if os.path.exists(path):
        return np.load(path)
    return None


def load_fold_results(name):
    """Load fold-level metrics from a results.json"""
    p = os.path.join('outputs/pr_de_net', name, 'results.json')
    if os.path.exists(p):
        d = json.load(open(p))
        return d.get('cv_results', {}), d.get('fold_details', [])
    return None, None


def fusion_score(probs_dict, weights, labels, fold_idx, threshold=0.5):
    """Weighted average of probs, evaluated per-fold.

    Note: LSTM-46d / BiLSTM-46d / Mamba-46d probs are P(passed) → flip to P(failed).
    Everything else: PR-DE-Net, RF, Transformer-7d, HDM-Net are already P(failed).
    """
    p = np.zeros_like(labels, dtype=float)
    for k, w in weights.items():
        p_k = probs_dict[k]
        if k in ('lstm_46d', 'bilstm_46d', 'mamba_46d'):
            p += w * (1 - p_k)   # flip P(passed) -> P(failed)
        else:
            p += w * p_k
    s = sum(weights.values())
    p /= s
    fold_results = []
    for f in range(5):
        m = evaluate(labels[fold_idx == f],
                     (p[fold_idx == f] > threshold).astype(int),
                     p[fold_idx == f])
        fold_results.append(m)
    return p, fold_results


def search_3way(probs_dict, labels, fold_idx, names):
    """Grid search 3-way weights {a, b, c} (step 0.5)."""
    best = None
    for a in np.arange(0.0, 4.5, 0.5):
        for b in np.arange(0.0, 4.5, 0.5):
            for c in np.arange(0.0, 4.5, 0.5):
                if a + b + c == 0:
                    continue
                weights = {names[0]: a, names[1]: b, names[2]: c}
                p, fr = fusion_score(probs_dict, weights, labels, fold_idx)
                f1 = np.mean([r['f1'] for r in fr])
                auc = np.mean([r['auc'] for r in fr])
                if best is None or f1 > best['f1']:
                    best = {'weights': weights, 'f1': f1, 'auc': auc, 'fold_results': fr}
    return best


def main():
    # collect probs from various sources
    probs_dict = {}

    # PR-DE-Net (full)
    probs_dict['pr_de'] = load_probs('outputs/pr_de_net/full/probs.npy')

    # From unified_compare (need to check directory structure)
    print("Searching for unified_compare results...")
    uc_dir = 'outputs/unified_compare'
    if os.path.isdir(uc_dir):
        for name in os.listdir(uc_dir):
            sub = os.path.join(uc_dir, name)
            if not os.path.isdir(sub):
                continue
            if os.path.exists(os.path.join(sub, 'probs.npy')):
                probs_dict[name] = np.load(os.path.join(sub, 'probs.npy'))

    print(f"Loaded {len(probs_dict)} prob arrays:")
    for k, v in probs_dict.items():
        print(f"  {k}: shape={v.shape}, mean={v.mean():.4f}")

    # evaluate each individually as baseline
    print("\n" + "=" * 70)
    print("Individual model performance on unified labels:")
    print(f"{'model':25s} {'F1':>10s} {'P':>10s} {'R':>10s} {'AUC':>10s}")
    rows = {}
    for name, p in probs_dict.items():
        fr = []
        for f in range(5):
            fr.append(evaluate(labels[fold_idx == f],
                               (p[fold_idx == f] > 0.5).astype(int),
                               p[fold_idx == f]))
        summary = summarize_fold_results(fr)
        rows[name] = summary
        print(f"{name:25s} {summary['f1_mean']:.4f}±{summary['f1_std']:.3f} "
              f"{summary['precision_mean']:.4f}±{summary['precision_std']:.3f} "
              f"{summary['recall_mean']:.4f}±{summary['recall_std']:.3f} "
              f"{summary['auc_mean']:.4f}±{summary['auc_std']:.3f}")

    # 3-way search: RF_7d + HDM-Net v2 + PR-DE-Net
    candidates = ['pr_de']
    for name in ['rf_7dim', 'hdm_net_v2', 'lstm_46d', 'transformer_7dim', 'weighted_1_3_1']:
        if name in probs_dict:
            candidates.append(name)

    if 'pr_de' in probs_dict and 'rf_7dim' in probs_dict and 'hdm_net_v2' in probs_dict:
        print("\n=== 3-way search: RF_7dim + HDM-Net v2 + PR-DE-Net ===")
        best3 = search_3way(probs_dict, labels, fold_idx,
                            ['rf_7dim', 'hdm_net_v2', 'pr_de'])
        print(f"Best 3-way: weights={best3['weights']} F1={best3['f1']:.4f} AUC={best3['auc']:.4f}")
        print_results_table("3-way fusion", summarize_fold_results(best3['fold_results']))

        # save
        with open('outputs/pr_de_net/fusion_3way.json', 'w') as f:
            json.dump({'best_3way': best3}, f, indent=2)

    # 4-way
    if 'pr_de' in probs_dict and 'lstm_46d' in probs_dict:
        print("\n=== 4-way search: RF_7dim + HDM-Net v2 + LSTM_46d + PR-DE-Net (step=1) ===")
        best4 = None
        for a in np.arange(0.0, 5.0, 1.0):
            for b in np.arange(0.0, 5.0, 1.0):
                for c in np.arange(0.0, 5.0, 1.0):
                    for d in np.arange(0.0, 5.0, 1.0):
                        if a + b + c + d == 0:
                            continue
                        weights = {'rf_7dim': a, 'hdm_net_v2': b, 'lstm_46d': c, 'pr_de': d}
                        p, fr = fusion_score(probs_dict, weights, labels, fold_idx)
                        f1 = np.mean([r['f1'] for r in fr])
                        auc = np.mean([r['auc'] for r in fr])
                        if best4 is None or f1 > best4['f1']:
                            best4 = {'weights': weights, 'f1': f1, 'auc': auc, 'fold_results': fr}
        print(f"Best 4-way: weights={best4['weights']} F1={best4['f1']:.4f} AUC={best4['auc']:.4f}")
        print_results_table("4-way fusion", summarize_fold_results(best4['fold_results']))
        with open('outputs/pr_de_net/fusion_4way.json', 'w') as f:
            json.dump({'best_4way': best4}, f, indent=2)

    # Save per-fold comparison summary
    summary_path = 'outputs/pr_de_net/comparison_with_baselines.json'
    out = {
        'pr_de_net_full': rows.get('pr_de', {}),
        'rf_7dim': rows.get('rf_7dim', {}),
        'hdm_net_v2': rows.get('hdm_net_v2', {}),
        'lstm_46d': rows.get('lstm_46d', {}),
        'transformer_7dim': rows.get('transformer_7dim', {}),
    }
    with open(summary_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved comparison to {summary_path}")


if __name__ == '__main__':
    main()