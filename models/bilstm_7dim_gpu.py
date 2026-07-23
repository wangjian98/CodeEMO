"""
BiLSTM 7 维 (GPU 加速版): 与 compare_lstm_bilstm_7dim.py 的 BiLSTM 等价的 GPU 实现
  - 设备: cuda
  - 5 折 StratifiedKFold(seed=42)
  - max_seq_len=500
  - 100 epoch, patience=15, batch=32, lr=1e-3
  - EventEncoder(7->64) -> BiLSTM(64, hidden=64, 2 layers, bidirectional) -> last+avg -> 128 -> Linear(64) -> ReLU -> Linear(1) -> Sigmoid
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, set_seed, get_device
from common.evaluator import evaluate, summarize_fold_results
from models.mamba.steps.step1_preprocessing import preprocess as preprocess_seq


# -- 直接复用 compare_lstm_bilstm_7dim.py 内的类 ----------------------
sys.path.insert(0, _PROJECT_ROOT)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "compare7dim",
    os.path.join(_PROJECT_ROOT, "models", "compare_lstm_bilstm_7dim.py")
)
mod = importlib.util.module_from_spec(spec)
# 防止模块内 main() 执行
spec.loader.exec_module(mod)

EventEncoder = mod.EventEncoder
SeqClassifier = mod.SeqClassifier
collate_seq = mod.collate_seq


def train_one_fold_gpu(model, train_samples, val_samples, device,
                       epochs=100, batch_size=32, patience=15, lr=1e-3):
    train_loader = DataLoader(train_samples, batch_size=batch_size, shuffle=True, collate_fn=collate_seq)
    val_loader = DataLoader(val_samples, batch_size=batch_size, shuffle=False, collate_fn=collate_seq)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    model = model.to(device)
    best_val_loss = float('inf')
    best_state = None
    pc = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0; n_b = 0
        for batch in train_loader:
            inp = {k: v.to(device) for k, v in batch.items() if k != 'risk'}
            tgt = batch['risk'].float().to(device)
            optimizer.zero_grad()
            out = model(inp).squeeze(-1)
            loss = criterion(out, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item(); n_b += 1

        model.eval()
        v_loss = 0; vb = 0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                inp = {k: v.to(device) for k, v in batch.items() if k != 'risk'}
                tgt = batch['risk'].float().to(device)
                out = model(inp).squeeze(-1)
                v_loss += criterion(out, tgt).item(); vb += 1
                all_probs.extend(out.cpu().numpy())
                all_labels.extend(tgt.cpu().numpy())
        avg_v = v_loss / max(vb, 1)
        if avg_v < best_val_loss:
            best_val_loss = avg_v
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1
            if pc >= patience:
                break

    # 加载 best
    model.load_state_dict(best_state)
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            inp = {k: v.to(device) for k, v in batch.items() if k != 'risk'}
            tgt = batch['risk'].float().to(device)
            out = model(inp).squeeze(-1)
            all_probs.extend(out.cpu().numpy())
            all_labels.extend(tgt.cpu().numpy())
    return np.array(all_probs), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-seq-len', type=int, default=500)
    parser.add_argument('--output-dir', type=str, default='outputs/bilstm_7dim_gpu')
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"设备: {device}", flush=True)

    ide_logs, passed = load_ide_logs()
    samples, student_ids, labels = preprocess_seq(ide_logs, passed, max_events=args.max_seq_len)
    labels_arr = np.array(labels)
    print(f"样本数: {len(samples)} (pass={int((labels_arr==0).sum())}, fail={int((labels_arr==1).sum())})", flush=True)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_metrics, fold_probs, test_indices = [], [], []
    for fold, (train_idx, test_idx) in enumerate(skf.split(samples, labels_arr), start=1):
        train_samples = [samples[i] for i in train_idx]
        test_samples = [samples[i] for i in test_idx]
        y_test = np.array([labels[i] for i in test_idx])
        print(f"\n=== Fold {fold}/{args.folds} | train={len(train_idx)} test={len(test_idx)} ===", flush=True)
        torch.manual_seed(args.seed + fold)
        model = SeqClassifier(n_event_types=7, d_model=64, hidden_dim=64, num_layers=2, dropout=0.3, bidirectional=True)
        t0 = time.time()
        probs, y_true = train_one_fold_gpu(
            model, train_samples, test_samples, device,
            epochs=args.epochs, batch_size=args.batch_size,
            patience=args.patience, lr=args.lr,
        )
        elapsed = time.time() - t0
        preds = (probs > 0.5).astype(int)
        m = evaluate(y_true, preds, probs)
        m['time_s'] = round(elapsed, 1)
        fold_metrics.append(m)
        fold_probs.append(probs)
        test_indices.append(test_idx)
        print(f"  Fold {fold}: Acc={m['accuracy']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f} ({elapsed:.0f}s)", flush=True)

    summary = summarize_fold_results(fold_metrics)
    print("\n========== BiLSTM 7维 GPU 5折汇总 ==========", flush=True)
    for k in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        m = summary.get(k + '_mean', 0)
        s = summary.get(k + '_std', 0)
        print(f"  {k}: {m:.4f} ± {s:.4f}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    # 保存 probs 完整数组
    all_probs = np.zeros(len(samples))
    for p, idx in zip(fold_probs, test_indices):
        all_probs[idx] = p
    np.save(os.path.join(args.output_dir, 'probs.npy'), all_probs)
    np.save(os.path.join(args.output_dir, 'labels.npy'), labels_arr)
    np.save(os.path.join(args.output_dir, 'fold_idx.npy'),
            np.concatenate([np.full(len(idx), i, dtype=int)
                          for i, idx in enumerate(test_indices)]))

    out = {
        'model': 'BiLSTM 7-dim (GPU version)',
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
    print(f"\n结果已保存: {args.output_dir}/results.json", flush=True)


if __name__ == '__main__':
    main()
