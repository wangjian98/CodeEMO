"""
LSTM 7-dim (GPU) - 与 bilstm_7dim_gpu.py 完全一致，仅 bidirectional=False
"""
import os, sys, json, time
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

_PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _PROJECT)
from common.data_loader import load_ide_logs, set_seed, get_device
from common.evaluator import evaluate, summarize_fold_results
from models.mamba.steps.step1_preprocessing import preprocess as preprocess_seq
import importlib.util
spec = importlib.util.spec_from_file_location('cmp7', os.path.join(_PROJECT, 'models', 'bilstm_7dim_gpu.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
train_one_fold_gpu = mod.train_one_fold_gpu

# 复用 compare_lstm_bilstm_7dim.py 里的 SeqClassifier
spec2 = importlib.util.spec_from_file_location('seq7', os.path.join(_PROJECT, 'models', 'compare_lstm_bilstm_7dim.py'))
mod2 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(mod2)
SeqClassifier = mod2.SeqClassifier
collate_seq = mod2.collate_seq


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-seq-len', type=int, default=500)
    parser.add_argument('--output-dir', type=str, default='outputs/lstm_7dim_gpu')
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f'设备: {device}', flush=True)

    ide_logs, passed = load_ide_logs()
    samples, student_ids, labels = preprocess_seq(ide_logs, passed, max_events=args.max_seq_len)
    labels_arr = np.array(labels)
    print(f'样本数: {len(samples)} (pass={int((labels_arr==0).sum())}, fail={int((labels_arr==1).sum())})', flush=True)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_metrics, fold_probs, test_indices = [], [], []
    for fold, (train_idx, test_idx) in enumerate(skf.split(samples, labels_arr), start=1):
        train_samples = [samples[i] for i in train_idx]
        test_samples = [samples[i] for i in test_idx]
        y_test = np.array([labels[i] for i in test_idx])
        print(f'\n=== Fold {fold}/{args.folds} | train={len(train_idx)} test={len(test_idx)} ===', flush=True)
        torch.manual_seed(args.seed + fold)
        model = SeqClassifier(n_event_types=7, d_model=64, hidden_dim=64,
                              num_layers=2, dropout=0.3, bidirectional=False)
        t0 = time.time()
        probs, y_true = train_one_fold_gpu(
            model, train_samples, test_samples, device,
            epochs=args.epochs, batch_size=args.batch_size,
            patience=args.patience, lr=args.lr)
        elapsed = time.time() - t0
        preds = (probs > 0.5).astype(int)
        m = evaluate(y_true, preds, probs)
        m['time_s'] = round(elapsed, 1)
        fold_metrics.append(m)
        fold_probs.append(probs)
        test_indices.append(test_idx)
        print(f'  Fold {fold}: Acc={m["accuracy"]:.4f} P={m["precision"]:.4f} R={m["recall"]:.4f} F1={m["f1"]:.4f} AUC={m["auc"]:.4f} ({elapsed:.0f}s)', flush=True)

    summary = summarize_fold_results(fold_metrics)
    print('\n========== LSTM 7-dim GPU 5-fold summary ==========', flush=True)
    for k in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        mm = summary.get(k + '_mean', 0)
        ss = summary.get(k + '_std', 0)
        print(f'  {k}: {mm:.4f} +/- {ss:.4f}', flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    all_probs = np.zeros(len(samples))
    for p, idx in zip(fold_probs, test_indices):
        all_probs[idx] = p
    np.save(os.path.join(args.output_dir, 'probs.npy'), all_probs)
    np.save(os.path.join(args.output_dir, 'labels.npy'), labels_arr)
    np.save(os.path.join(args.output_dir, 'fold_idx.npy'),
            np.concatenate([np.full(len(idx), i, dtype=int) for i, idx in enumerate(test_indices)]))

    out = {
        'model': 'LSTM 7-dim (GPU version)',
        'config': vars(args),
        'cv_results': {k.replace('_mean', ''): v for k, v in summary.items() if k.endswith('_mean')},
        'cv_results_std': {k.replace('_mean', '_std'): v for k, v in summary.items() if k.endswith('_std')},
        'fold_details': fold_metrics,
        'n_samples': len(samples),
        'n_passed': int((labels_arr==0).sum()),
        'n_failed': int((labels_arr==1).sum()),
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {args.output_dir}/results.json', flush=True)


if __name__ == '__main__':
    main()
