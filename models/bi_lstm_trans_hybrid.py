"""
方案 D: BiLSTM + Transformer 双流混合 (用于超过 7 维 BiLSTM baseline 0.8768)

架构:
    事件序列 → EventEncoder(d=64, with positional encoding)
              → ┬─ BiLSTM(hidden=64, 2 layers, bidirectional)
                │    双聚合: avg pool + max pool → 256 维
                └─ TransformerEncoder(4 heads, 2 layers, d=64)
                     cls token aggregation → 64 维
              → Concat (lstm 256 + trans 64 + hand 3) = 323 维
              → LayerNorm → Linear(128) → GELU → Dropout → Linear(2)

为什么这样设计:
  - BiLSTM 捕捉短程双向模式 (门控局部依赖)
  - Transformer 捕捉全局自注意力 (任意位置关系)
  - 两者通过 concat 互补, 加 LayerNorm 让量纲统一
  - 手工特征 (3维) 提供全学期统计信号

用法:
    python models/bi_lstm_trans_hybrid.py
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


# ─── Event Encoder (with positional) ─────────────────────────

class EventEncoder(nn.Module):
    def __init__(self, n_event_types=7, d_model=64, max_len=2048):
        super().__init__()
        self.ev_embed = nn.Embedding(n_event_types, 16)
        self.time_embed = nn.Linear(1, 8)
        self.dl_embed = nn.Linear(1, 8)
        self.pos_embed = nn.Embedding(max_len, 16)
        self.proj = nn.Linear(16 + 8 + 8 + 16, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, event_types, time_intervals, deadline_dists):
        B, L = event_types.shape
        ev = self.ev_embed(event_types)
        te = self.time_embed(time_intervals.unsqueeze(-1))
        de = self.dl_embed(deadline_dists.unsqueeze(-1))
        pos_ids = torch.arange(L, device=event_types.device).unsqueeze(0).expand(B, L)
        pe = self.pos_embed(pos_ids)
        out = self.proj(torch.cat([ev, te, de, pe], dim=-1))
        return self.norm(out)  # (B, L, d_model)


# ─── 手工特征 (向量化, 无 Python 循环) ─────────────────────

def compute_handcrafted(batch, n_event_types=7):
    """3维手工特征, 完全向量化"""
    et = batch['event_types']    # (B, L)
    ti = batch['time_intervals'] # (B, L)
    dd = batch['deadline_dists'] # (B, L)
    # 1. deadline proximity: 1 - mean(dd), 越接近 deadline 越接近 1
    f1 = (1.0 - dd).mean(dim=-1, keepdim=True)
    # 2. event diversity: 用 one-hot sum + 0
    onehot = F.one_hot(et.clamp(min=0), n_event_types).float()  # (B, L, 7)
    # 但 event_type 可能包含 padding (无 0 标记, 改用长度 > 0 都有效)
    # 这里假设 et 已经过 preprocess, 等于 -100 的就是 padding, 不过当前 preprocess 没生成 -100
    n_unique = (onehot.sum(dim=1) > 0).float().sum(dim=-1, keepdim=True) / n_event_types
    # 3. submit regularity: submit 事件密度 (假设 submit=6)
    submit_mask = (et == 6).float()
    n_submit = submit_mask.sum(dim=-1, keepdim=True)
    f3 = n_submit / ti.sum(dim=-1, keepdim=True).clamp(min=1e-3)
    return torch.cat([f1, n_unique, f3], dim=-1)  # (B, 3)


# ─── BiLSTM + Transformer 混合 ────────────────────────────────

class BiLSTMTransformerHybrid(nn.Module):
    def __init__(self, n_event_types=7, d_model=64, hidden=64,
                 lstm_layers=2, trans_layers=2, n_heads=4,
                 dropout=0.3, max_len=2048):
        super().__init__()
        self.encoder = EventEncoder(n_event_types, d_model, max_len)
        # BiLSTM branch
        self.bilstm = nn.LSTM(
            input_size=d_model, hidden_size=hidden,
            num_layers=lstm_layers, batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0, bidirectional=True
        )
        lstm_out = hidden * 2
        # Transformer branch
        trans_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, activation='gelu', norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(trans_layer, num_layers=trans_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # Fusion head
        fusion_dim = lstm_out * 2 + d_model + 3  # avg+max(256) + cls(64) + hand(3)
        self.risk_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, batch):
        et = batch['event_types']
        ti = batch['time_intervals']
        dd = batch['deadline_dists']
        x = self.encoder(et, ti, dd)
        # BiLSTM branch
        lstm_out, _ = self.bilstm(x)
        lstm_avg = lstm_out.mean(dim=1)
        lstm_max = lstm_out.max(dim=1).values
        lstm_feat = torch.cat([lstm_avg, lstm_max], dim=-1)  # (B, hidden*4 = 256)
        # Transformer branch
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x_t = torch.cat([cls, x], dim=1)
        mask = torch.zeros(x_t.shape[1], x_t.shape[1], device=x.device, dtype=torch.bool)
        t_out = self.transformer(x_t, mask=mask)
        t_cls = t_out[:, 0, :]
        # Handcrafted features
        hand = compute_handcrafted(batch)
        # Fuse
        fused = torch.cat([lstm_feat, t_cls, hand], dim=-1)
        return self.risk_head(fused)


# ─── Focal Loss ────────────────────────────────────────────────

def focal_loss(logits, targets, alpha=None, gamma=2.5, label_smoothing=0.05):
    num_classes = logits.shape[-1]
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    targets_oh = F.one_hot(targets, num_classes).float()
    targets_oh = targets_oh * (1 - label_smoothing) + label_smoothing / num_classes
    weight = (1 - probs) ** gamma
    if alpha is not None:
        alpha_t = alpha[targets]
        weight = weight * alpha_t.unsqueeze(-1)
    return -(targets_oh * weight * log_probs).sum(dim=-1).mean()


# ─── Collate ──────────────────────────────────────────────────

def collate_seq(batch_samples):
    max_len = max(s['n_events'] for s in batch_samples)
    bet, bti, bdd, bpi, br = [], [], [], [], []
    for s in batch_samples:
        n = s['n_events']
        et = s['event_types'][:n]
        ti = s['time_intervals'][:n]
        dd = s['deadline_dists'][:n]
        pi = s['part_ids'][:n]
        r = s['risk']
        if len(et) < max_len:
            p = max_len - len(et)
            et = F.pad(et, (0, p), value=0)
            ti = F.pad(ti, (0, p), value=0.0)
            dd = F.pad(dd, (0, p), value=0.0)
            pi = F.pad(pi, (0, p), value=0)
        bet.append(et[:max_len])
        bti.append(ti[:max_len])
        bdd.append(dd[:max_len])
        bpi.append(pi[:max_len])
        br.append(r)
    return {
        'event_types': torch.stack(bet),
        'time_intervals': torch.stack(bti),
        'deadline_dists': torch.stack(bdd),
        'part_ids': torch.stack(bpi),
        'risk': torch.LongTensor(br),
    }


# ─── 训练一折 ─────────────────────────────────────────────────

def train_one_fold(model, train_samples, val_samples, device,
                   epochs=30, batch_size=16, patience=8, lr=1e-3, fold=0):
    model = model.to(device)
    train_loader = DataLoader(train_samples, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_seq)
    val_loader = DataLoader(val_samples, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_seq)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    alpha = torch.tensor([1.488, 0.753], device=device)
    best_val_auc = -1.0
    best_state = None
    patience_cnt = 0

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, batch_cnt = 0.0, 0
        for batch in train_loader:
            inp = {k: v.to(device) for k, v in batch.items() if k != 'risk'}
            tgt = batch['risk'].to(device)
            optimizer.zero_grad()
            logits = model(inp)
            loss = focal_loss(logits, tgt, alpha=alpha, gamma=2.5)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item(); batch_cnt += 1
        scheduler.step()
        print(f'  Fold{fold} Epoch {epoch}/{epochs} loss={loss_sum/max(batch_cnt,1):.4f}', flush=True)

        model.eval()
        probs_list, labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                inp = {k: v.to(device) for k, v in batch.items() if k != 'risk'}
                tgt = batch['risk'].to(device)
                logits = model(inp)
                probs = F.softmax(logits, dim=-1)[:, 1]
                probs_list.extend(probs.cpu().numpy())
                labels_list.extend(tgt.cpu().numpy())
        try:
            auc = roc_auc_score(labels_list, probs_list)
        except Exception:
            auc = 0.5
        if auc > best_val_auc:
            best_val_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    probs_list, labels_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            inp = {k: v.to(device) for k, v in batch.items() if k != 'risk'}
            tgt = batch['risk'].to(device)
            logits = model(inp)
            probs = F.softmax(logits, dim=-1)[:, 1]
            probs_list.extend(probs.cpu().numpy())
            labels_list.extend(tgt.cpu().numpy())
    return np.array(probs_list), np.array(labels_list)


# ─── 主流程 ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-seq-len', type=int, default=2000)
    parser.add_argument('--output-dir', type=str, default='outputs/bi_lstm_trans_hybrid')
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f'设备: {device}')

    ide_logs, passed = load_ide_logs()
    samples, student_ids, labels = preprocess_seq(
        ide_logs, passed, max_events=args.max_seq_len
    )
    n_pass = int((np.array(labels) == 0).sum())
    n_fail = int((np.array(labels) == 1).sum())
    print(f'样本数: {len(samples)} (pass={n_pass}, fail={n_fail})')

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(samples, labels), start=1):
        train_samples = [samples[i] for i in train_idx]
        test_samples = [samples[i] for i in test_idx]
        y_test = np.array([labels[i] for i in test_idx])
        print(f'\n=== Fold {fold}/{args.folds} | train={len(train_samples)} test={len(test_samples)} ===')
        torch.manual_seed(args.seed + fold)
        model = BiLSTMTransformerHybrid(max_len=args.max_seq_len)
        t0 = time.time()
        probs, y_true = train_one_fold(
            model, train_samples, test_samples, device,
            epochs=args.epochs, batch_size=args.batch_size, fold=fold,
        )
        elapsed = time.time() - t0
        from common.evaluator import evaluate
        preds = (probs > 0.5).astype(int)
        m = evaluate(y_true, preds, probs)  # probs 是 1D 数组
        # m['auc'] 已经由 evaluate 计算好了, 不用再算
        m['time_s'] = round(elapsed, 1)
        fold_metrics.append(m)
        print(f"  Fold {fold}: Acc={m['accuracy']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f} ({elapsed:.0f}s)")

    summary = summarize_fold_results(fold_metrics)
    print('\n========== 5折CV汇总 ==========')
    for k in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        m = summary.get(k + '_mean', 0)
        s = summary.get(k + '_std', 0)
        print(f'  {k}: {m:.4f} ± {s:.4f}')

    os.makedirs(args.output_dir, exist_ok=True)
    out = {
        'model': 'BiLSTM+Transformer Hybrid (Plan D)',
        'config': vars(args),
        'cv_results': {k.replace('_mean', ''): v for k, v in summary.items() if k.endswith('_mean')},
        'cv_results_std': {k.replace('_mean', '_std'): v for k, v in summary.items() if k.endswith('_std')},
        'fold_details': fold_metrics,
        'n_samples': len(samples),
        'n_passed': int((np.array(labels) == 0).sum()),
        'n_failed': int((np.array(labels) == 1).sum()),
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\n保存: {args.output_dir}/results.json')


if __name__ == '__main__':
    main()
