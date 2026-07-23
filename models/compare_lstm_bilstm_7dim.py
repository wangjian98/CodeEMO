"""
LSTM vs BiLSTM 对比实验 - 统一使用 7维事件序列

两种模型共享同一个事件编码器，差异仅在 LSTM 的 bidirectional 参数：
  - LSTM:   单向，hidden_dim * num_layers = 128 维最后状态
  - BiLSTM: 双向，hidden_dim * 2 = 128 维最后状态

输入: 7维事件序列 (event_type + time_interval + deadline_dist)
      经过 EventEncoder 编码为 (batch, seq_len, hidden_dim)

用法:
    python models/compare_lstm_bilstm_7dim.py
    python models/compare_lstm_bilstm_7dim.py --folds 5 --max-seq-len 500
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


# ─── 共享事件编码器 ────────────────────────────────────────────

class EventEncoder(nn.Module):
    """7维事件编码器: event_type embedding + time + deadline → d_model"""
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


# ─── 统一序列模型基类 ──────────────────────────────────────────

class SeqClassifier(nn.Module):
    """
    通用序列分类器: EventEncoder → LSTM/BiLSTM → 分类头

    Args:
        bidirectional: True=BiLSTM, False=LSTM
    """
    def __init__(self, n_event_types=7, d_model=64, hidden_dim=64,
                 num_layers=2, dropout=0.3, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.encoder = EventEncoder(n_event_types, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.output_dim = lstm_out_dim

    def forward(self, batch):
        et = batch['event_types']
        ti = batch['time_intervals']
        dd = batch['deadline_dists']
        encoded = self.encoder(et, ti, dd)      # (B, L, d_model)
        lstm_out, (hn, _) = self.lstm(encoded)  # lstm_out: (B, L, hidden*2 or hidden)

        if self.bidirectional:
            # 取最后时刻的正向+反向最后状态拼接 (即 lstm_out[:, -1, :])
            last = lstm_out[:, -1, :]           # (B, hidden*2)
        else:
            # 单向 LSTM: 取最后一层最后时刻 hidden state
            last = hn[-1, :, :]                 # (B, hidden)
        return self.classifier(last)            # (B, 1)


# ─── Batch 填充 ────────────────────────────────────────────────

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


# ─── 单折训练 ──────────────────────────────────────────────────

def train_one_fold(model, train_samples, val_samples, device,
                   epochs=100, batch_size=32, patience=15, lr=1e-3):
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
        epoch_loss = 0; n_batches = 0
        for batch in train_loader:
            inp = {k: v.to(device) for k, v in batch.items() if k != 'risk'}
            tgt = batch['risk'].float().to(device)
            optimizer.zero_grad()
            out = model(inp).squeeze(-1)
            loss = criterion(out, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item(); n_batches += 1

        # 验证
        model.eval()
        val_loss = 0; vb = 0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                inp = {k: v.to(device) for k, v in batch.items() if k != 'risk'}
                tgt = batch['risk'].float().to(device)
                out = model(inp).squeeze(-1)
                val_loss += criterion(out, tgt).item(); vb += 1
                all_probs.extend(out.cpu().numpy())
                all_labels.extend(tgt.cpu().numpy())

        avg_val = val_loss / max(vb, 1)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    # 最优权重推理
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


# ─── 主函数 ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-seq-len', type=int, default=500)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cpu')

    print("=" * 62)
    print("  LSTM vs BiLSTM 对比实验  |  输入: 7维事件序列")
    print("=" * 62)
    print(f"  设备={device}  折数={args.folds}  max_seq={args.max_seq_len}")
    print(f"  batch={args.batch_size}  lr={args.lr}  epochs={args.epochs}")
    print("=" * 62)

    t0 = time.time()

    # 加载数据
    print("\n[数据加载] 7维事件序列 ...")
    ide_logs, passed = load_ide_logs()
    samples, student_ids, labels = preprocess_seq(ide_logs, passed,
                                                  max_events=args.max_seq_len)
    print(f"  样本数={len(samples)}  通过={sum(labels==0)}  挂科={sum(labels==1)}")
    print(f"  加载耗时: {time.time()-t0:.1f}s")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    student_arr = np.array(student_ids)

    results = {}

    for model_name, bidirectional in [('LSTM', False), ('BiLSTM', True)]:
        print(f"\n{'='*50}")
        print(f"  模型: {model_name} (bidirectional={bidirectional})")
        print(f"{'='*50}")

        fold_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(
                skf.split(student_arr, labels), 1):

            t_fold = time.time()
            train_samples = [samples[i] for i in train_idx]
            val_samples = [samples[i] for i in val_idx]

            model = SeqClassifier(bidirectional=bidirectional, d_model=64,
                                  hidden_dim=64, num_layers=2, dropout=0.3)

            val_probs, val_labels = train_one_fold(
                model, train_samples, val_samples, device,
                epochs=args.epochs, batch_size=args.batch_size,
                patience=args.patience, lr=args.lr)

            val_pred = (val_probs >= 0.5).astype(int)
            metrics = evaluate(val_labels, val_pred, val_probs)
            fold_results.append(metrics)
            print(f"  Fold {fold_idx}: Acc={metrics['accuracy']:.4f}  "
                  f"F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}  "
                  f"({time.time()-t_fold:.0f}s)")

        summary = summarize_fold_results(fold_results)
        results[model_name] = {'summary': summary, 'folds': fold_results,
                               'params': sum(p.numel() for p in model.parameters() if p.requires_grad)}
        print(f"\n  {model_name} 汇总:")
        print(f"    Acc={summary['accuracy_mean']:.4f}  "
              f"Prec={summary['precision_mean']:.4f}  "
              f"Recall={summary['recall_mean']:.4f}")
        print(f"    F1={summary['f1_mean']:.4f}  AUC={summary['auc_mean']:.4f}")

    # 对比表格
    print("\n" + "=" * 62)
    print("  对比结果汇总")
    print("=" * 62)
    print_results_table("LSTM (7-dim)", results['LSTM']['summary'])
    print_results_table("BiLSTM (7-dim)", results['BiLSTM']['summary'])

    # 保存
    out_dir = os.path.join(_PROJECT_ROOT, 'outputs', 'compare_lstm_bilstm_7dim')
    os.makedirs(out_dir, exist_ok=True)
    save_data = {
        'lstm': {
            'input': '7-dim event sequences',
            'params': results['LSTM']['params'],
            'config': {'d_model': 64, 'hidden_dim': 64, 'num_layers': 2,
                       'bidirectional': False, 'max_seq_len': args.max_seq_len},
            'cv_results': {k: float(v) for k, v in results['LSTM']['summary'].items()},
            'fold_details': results['LSTM']['folds'],
        },
        'bilstm': {
            'input': '7-dim event sequences',
            'params': results['BiLSTM']['params'],
            'config': {'d_model': 64, 'hidden_dim': 64, 'num_layers': 2,
                       'bidirectional': True, 'max_seq_len': args.max_seq_len},
            'cv_results': {k: float(v) for k, v in results['BiLSTM']['summary'].items()},
            'fold_details': results['BiLSTM']['folds'],
        },
        'n_samples': len(samples),
        'n_folds': args.folds,
    }
    with open(os.path.join(out_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {out_dir}/results.json")
    print(f"总耗时: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
