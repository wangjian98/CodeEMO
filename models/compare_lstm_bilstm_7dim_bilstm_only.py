"""
BiLSTM 7维序列实验 - 快速单独运行
复用 compare_lstm_bilstm_7dim.py 的相同配置
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

from common.data_loader import load_ide_logs, set_seed
from common.evaluator import evaluate, summarize_fold_results, print_results_table
from models.mamba.steps.step1_preprocessing import preprocess as preprocess_seq


class EventEncoder(nn.Module):
    def __init__(self, n_event_types=7, d_model=64):
        super().__init__()
        self.ev_embed = nn.Embedding(n_event_types, 16)
        self.time_embed = nn.Linear(1, 8)
        self.dl_embed = nn.Linear(1, 8)
        self.proj = nn.Linear(16 + 8 + 8, d_model)

    def forward(self, event_types, time_intervals, deadline_dists):
        ev = self.ev_embed(event_types)
        te = self.time_embed(time_intervals.unsqueeze(-1))
        de = self.dl_embed(deadline_dists.unsqueeze(-1))
        return self.proj(torch.cat([ev, te, de], dim=-1))


class SeqClassifier(nn.Module):
    def __init__(self, bidirectional=True, d_model=64, hidden_dim=64,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.bidirectional = bidirectional
        self.encoder = EventEncoder(7, d_model)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True, bidirectional=bidirectional)
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, batch):
        et = batch['event_types']
        ti = batch['time_intervals']
        dd = batch['deadline_dists']
        encoded = self.encoder(et, ti, dd)
        lstm_out, (hn, _) = self.lstm(encoded)
        last = lstm_out[:, -1, :] if self.bidirectional else hn[-1, :, :]
        return self.classifier(last)


def collate_seq(batch_samples):
    max_len = max(s['n_events'] for s in batch_samples)
    bet, bti, bdd, bpi, br = [], [], [], [], []
    for s in batch_samples:
        n = s['n_events']
        et, ti, dd, pi = s['event_types'][:n], s['time_intervals'][:n], \
                         s['deadline_dists'][:n], s['part_ids'][:n]
        if len(et) < max_len:
            p = max_len - len(et)
            et = F.pad(et, (0, p)); ti = F.pad(ti, (0, p))
            dd = F.pad(dd, (0, p)); pi = F.pad(pi, (0, p))
        bet.append(et[:max_len]); bti.append(ti[:max_len])
        bdd.append(dd[:max_len]); bpi.append(pi[:max_len])
        br.append(s['risk'])
    return {
        'event_types': torch.stack(bet),
        'time_intervals': torch.stack(bti),
        'deadline_dists': torch.stack(bdd),
        'part_ids': torch.stack(bpi),
        'risk': torch.LongTensor(br),
    }


def train_one_fold(model, train_samples, val_samples, device,
                   epochs=80, batch_size=32, patience=20, lr=1e-3):
    model = model.to(device)
    train_loader = DataLoader(train_samples, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_seq)
    val_loader = DataLoader(val_samples, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_seq)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    best_val_loss = float('inf')
    best_state = None
    patience_cnt = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            inp = {k: v.to(device) for k, v in batch.items() if k != 'risk'}
            tgt = batch['risk'].float().to(device)
            optimizer.zero_grad()
            out = model(inp).squeeze(-1)
            loss = criterion(out, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        val_loss = 0; vb = 0
        with torch.no_grad():
            for batch in val_loader:
                inp = {k: v.to(device) for k, v in batch.items() if k != 'risk'}
                tgt = batch['risk'].float().to(device)
                val_loss += criterion(model(inp).squeeze(-1), tgt).item(); vb += 1
        avg_val = val_loss / max(vb, 1)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            inp = {k: v.to(device) for k, v in batch.items() if k != 'risk'}
            out = model(inp).squeeze(-1)
            all_probs.extend(out.cpu().numpy())
            all_labels.extend(batch['risk'].numpy())
    return np.array(all_probs), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-seq-len', type=int, default=200)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cpu')
    print(f"BiLSTM on 7-dim sequences | folds={args.folds} max_seq={args.max_seq_len}")

    t0 = time.time()
    ide_logs, passed = load_ide_logs()
    samples, student_ids, labels = preprocess_seq(ide_logs, passed,
                                                  max_events=args.max_seq_len)
    print(f"Samples: {len(samples)}, loaded in {time.time()-t0:.1f}s")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(np.array(student_ids), labels), 1):
        t_fold = time.time()
        model = SeqClassifier(bidirectional=True, d_model=64,
                              hidden_dim=64, num_layers=2, dropout=0.3)
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        val_probs, val_labels = train_one_fold(
            model, train_samples, val_samples, device,
            epochs=args.epochs, batch_size=args.batch_size,
            patience=args.patience, lr=args.lr)
        val_pred = (val_probs >= 0.5).astype(int)
        metrics = evaluate(val_labels, val_pred, val_probs)
        fold_results.append(metrics)
        print(f"  Fold {fold_idx}: Acc={metrics['accuracy']:.4f}  "
              f"F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}  ({time.time()-t_fold:.0f}s)")

    summary = summarize_fold_results(fold_results)
    print(f"\nBiLSTM (7-dim)汇总:")
    print(f"  Acc={summary['accuracy_mean']:.4f}  Prec={summary['precision_mean']:.4f}  "
          f"Recall={summary['recall_mean']:.4f}")
    print(f"  F1={summary['f1_mean']:.4f}  AUC={summary['auc_mean']:.4f}")

    out_dir = os.path.join(_PROJECT_ROOT, 'outputs', 'compare_lstm_bilstm_7dim')
    os.makedirs(out_dir, exist_ok=True)
    # 追加 BiLSTM 结果
    existing = {}
    path = os.path.join(out_dir, 'results.json')
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    existing['bilstm'] = {
        'input': '7-dim event sequences',
        'params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'config': {'d_model': 64, 'hidden_dim': 64, 'num_layers': 2,
                   'bidirectional': True, 'max_seq_len': args.max_seq_len},
        'cv_results': {k: float(v) for k, v in summary.items()},
        'fold_details': fold_results,
    }
    with open(path, 'w') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"Saved: {path}")


if __name__ == '__main__':
    main()
