"""
BiLSTM 7d + micro-behaviour 手工程特征 (12 维)
  - input:  7d 事件序列 (max=N) + 12d micro 特征
  - 模型:   EventEncoder -> BiLSTM -> [last; micro] -> FC
  - 训练:   5-fold CV, 输出 P(fail)
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, set_seed, get_device
from common.evaluator import evaluate, summarize_fold_results
from models.mamba.steps.step1_preprocessing import preprocess as preprocess_seq

# -- 复用 compare_lstm_bilstm_7dim 的基础结构 -------------------
import importlib.util
spec = importlib.util.spec_from_file_location(
    "compare7dim",
    os.path.join(_PROJECT_ROOT, "models", "compare_lstm_bilstm_7dim.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
EventEncoder = mod.EventEncoder
collate_seq  = mod.collate_seq


# ─── 12 维 Micro-Behaviour 特征 ───────────────────────────────
EVENT_TYPES_7 = ['focus_gained', 'focus_lost', 'text_insert',
                 'text_remove', 'text_paste', 'run', 'submit']


def compute_micro_features(ide_logs_df, student_ids, n_first=30):
    """
    对每个学生从最早的 n_first 个事件中提取 12 维 micro-behaviour 特征。

    特征定义:
      0  focus_gain_rate        前 n_first 内 focus_gained 占比
      1  focus_lose_rate        前 n_first 内 focus_lost 占比
      2  focus_gl_ratio         gain/lose 比 (无 lose 时 = gain_count)
      3  edit_density           前 n_first 内 text_insert 占比
      4  delete_density         前 n_first 内 text_remove 占比
      5  edit_delete_ratio      ins/rem 比 (无 rem 时 = ins_count)
      6  submit_rate            前 n_first 内 submit 占比
      7  early_tightness        前 n_first 内事件平均间隔 (归一化)
      8  early_deadline_prox    前 n_first 内 deadline_dist 均值
      9  early_event_count      前 n_first 内事件数 (归一化 n_first=30)
      10 intro_burst_score      前 10 个事件数 / 前 n_first 总事件数 (开局密集度)
      11 paste_density          前 n_first 内 text_paste 占比

    Returns:  (n_students, 12)  float32
    """
    n = len(student_ids)
    feats = np.zeros((n, 12), dtype=np.float32)
    for i, sid in enumerate(student_ids):
        df = ide_logs_df[ide_logs_df['student'] == sid].sort_values('timestamp')
        head = df.head(n_first)
        if len(head) == 0:
            continue
        n_e = len(head)
        cnts = head['eventType'].value_counts()

        # 0, 1, 2: focus gain/lose
        g = cnts.get('focus_gained', 0)
        l = cnts.get('focus_lost', 0)
        feats[i, 0] = g / n_e
        feats[i, 1] = l / n_e
        feats[i, 2] = (g / l) if l > 0 else float(g)

        # 3, 4, 5: edit/delete
        ins = cnts.get('text_insert', 0)
        rem = cnts.get('text_remove', 0)
        feats[i, 3] = ins / n_e
        feats[i, 4] = rem / n_e
        feats[i, 5] = (ins / rem) if rem > 0 else float(ins)

        # 6: submit
        feats[i, 6] = cnts.get('submit', 0) / n_e

        # 7: 平均事件间隔
        if n_e > 1:
            ts = (head['timestamp'] - head['timestamp'].min()).dt.total_seconds().values
            iv = np.diff(ts)
            feats[i, 7] = float(np.mean(iv)) / 3600.0   # 归一化到小时
        else:
            feats[i, 7] = 0.0

        # 8: deadline 接近度 (avg distance to deadline)
        feats[i, 8] = float(head['timeToDeadline'].clip(0, 1).mean())

        # 9: 前 n_first 内事件数 (归一化)
        feats[i, 9] = n_e / float(n_first)

        # 10: 开局密集度 (前 10 个事件数 / 前 n_first 总数)
        feats[i, 10] = min(len(df.head(10)), 10) / 10.0

        # 11: paste 密度
        feats[i, 11] = cnts.get('text_paste', 0) / n_e

    return feats


# ─── 模型: BiLSTM + Micro Concat ────────────────────────────────

class BiLSTMMicro(nn.Module):
    def __init__(self, n_event_types=7, d_model=64, hidden_dim=64,
                 num_layers=2, dropout=0.3, micro_dim=12):
        super().__init__()
        self.encoder = EventEncoder(n_event_types, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        lstm_out_dim = hidden_dim * 2  # bidirectional
        # concat 12d micro
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim + micro_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, batch):
        et = batch['event_types']
        ti = batch['time_intervals']
        dd = batch['deadline_dists']
        micro = batch['micro']                              # (B, 12)
        encoded = self.encoder(et, ti, dd)                  # (B, L, d_model)
        lstm_out, _ = self.lstm(encoded)
        last = lstm_out[:, -1, :]                           # (B, hidden*2)
        fused = torch.cat([last, micro], dim=-1)            # (B, hidden*2+12)
        return self.classifier(fused)


# ─── Collate (在原基础上加 micro) ──────────────────────────────

def collate_seq_micro(batch_samples):
    base = collate_seq(batch_samples)
    base['micro'] = torch.stack([
        torch.from_numpy(s['micro']) for s in batch_samples
    ]).float()
    return base


def add_micro_to_samples(samples, micro_feats, student_ids_map):
    """把 micro_feats (按 preprocess_seq 返回顺序) 注入每个 sample"""
    sid_to_idx = {s: i for i, s in enumerate(student_ids_map)}
    for s in samples:
        s['micro'] = micro_feats[sid_to_idx[s['student_id']]]
    return samples


# ─── 训练一折 (GPU) ──────────────────────────────────────────────

def train_one_fold_gpu(model, train_samples, val_samples, device,
                       epochs=50, batch_size=32, patience=10, lr=1e-3):
    train_loader = DataLoader(train_samples, batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_seq_micro)
    val_loader   = DataLoader(val_samples, batch_size=batch_size,
                              shuffle=False, collate_fn=collate_seq_micro)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    model = model.to(device)
    best_val_loss = float('inf'); best_state = None; pc = 0
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0; n_b = 0
        for batch in train_loader:
            inp = {k: v.to(device) for k, v in batch.items() if k != 'risk'}
            tgt = batch['risk'].float().to(device)
            optimizer.zero_grad()
            out = model(inp).squeeze(-1)
            loss = criterion(out, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item(); n_b += 1

        model.eval()
        v_loss = 0; vb = 0
        with torch.no_grad():
            for batch in val_loader:
                inp = {k: v.to(device) for k, v in batch.items() if k != 'risk'}
                tgt = batch['risk'].float().to(device)
                out = model(inp).squeeze(-1)
                v_loss += criterion(out, tgt).item(); vb += 1
        avg_v = v_loss / max(vb, 1)
        if avg_v < best_val_loss:
            best_val_loss = avg_v
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1
            if pc >= patience: break

    if best_state: model.load_state_dict(best_state)
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


# ─── 主流程 ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-seq-len', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--output-dir', type=str, default='outputs/bilstm_7dim_micro_max50')
    parser.add_argument('--micro-n-first', type=int, default=30)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"设备: {device}", flush=True)

    ide_logs, passed = load_ide_logs()
    samples, student_ids, labels = preprocess_seq(
        ide_logs, passed, max_events=args.max_seq_len)
    labels_arr = np.array(labels)

    # 计算 micro features (按原 student_ids 顺序)
    student_ids_arr = np.array(student_ids)
    micro = compute_micro_features(ide_logs, student_ids_arr,
                                   n_first=args.micro_n_first)
    print(f"Micro features: shape={micro.shape}, "
          f"mean={micro.mean(0).round(3).tolist()}, "
          f"std={micro.std(0).round(3).tolist()}", flush=True)
    samples = add_micro_to_samples(samples, micro, student_ids_arr)

    print(f"样本数: {len(samples)} (pass={int((labels_arr==0).sum())}, "
          f"fail={int((labels_arr==1).sum())})", flush=True)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True,
                          random_state=args.seed)
    fold_metrics, fold_probs, test_indices = [], [], []
    for fold, (train_idx, test_idx) in enumerate(
            skf.split(samples, labels_arr), start=1):
        train_samples = [samples[i] for i in train_idx]
        test_samples  = [samples[i] for i in test_idx]
        y_test = np.array([labels[i] for i in test_idx])
        print(f"\n=== Fold {fold}/{args.folds} "
              f"| train={len(train_idx)} test={len(test_idx)} ===", flush=True)
        torch.manual_seed(args.seed + fold)
        model = BiLSTMMicro()
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
        print(f"  Fold {fold}: Acc={m['accuracy']:.4f} P={m['precision']:.4f}"
              f" R={m['recall']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f}"
              f" ({elapsed:.0f}s)", flush=True)

    summary = summarize_fold_results(fold_metrics)
    print("\n========== BiLSTM 7d + Micro 5折汇总 ==========", flush=True)
    for k in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        m = summary.get(k + '_mean', 0); s = summary.get(k + '_std', 0)
        print(f"  {k}: {m:.4f} ± {s:.4f}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    all_probs = np.zeros(len(samples))
    for p, idx in zip(fold_probs, test_indices):
        all_probs[idx] = p
    np.save(os.path.join(args.output_dir, 'probs.npy'), all_probs)
    np.save(os.path.join(args.output_dir, 'labels.npy'), labels_arr)
    np.save(os.path.join(args.output_dir, 'fold_idx.npy'),
            np.concatenate([
                np.full(len(idx), i, dtype=int)
                for i, idx in enumerate(test_indices)
            ]))

    out = {
        'model': 'BiLSTM 7d + Micro (12d)',
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
