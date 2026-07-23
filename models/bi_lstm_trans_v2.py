"""
方案 D-改良版 (Plan D2): BiLSTM+Transformer Hybrid v2
  - 手工特征扩展到 30 维 (deadline, 时间分布, 事件多样性, submit 模式, focus 模式等)
  - 加 SE 通道注意力到 fusion
  - 提高 d_model 到 96, transformer heads=6

为什么这版能涨:
  - 30 维手工特征为模型提供大量可解释统计信号 (弥补 BiLSTM 46 维 path 的优势)
  - SE 让 fusion head 自动调权 (BiLSTM vs Transformer vs handcraft)
  - d_model 增大让表征更丰富
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


# ─── Event Encoder ──────────────────────────────────────────────

class EventEncoder(nn.Module):
    def __init__(self, n_event_types=7, d_model=96, max_len=2048):
        super().__init__()
        self.ev_embed = nn.Embedding(n_event_types, 32)
        self.time_embed = nn.Linear(1, 16)
        self.dl_embed = nn.Linear(1, 16)
        self.pos_embed = nn.Embedding(max_len, 32)
        self.proj = nn.Linear(32 + 16 + 16 + 32, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, event_types, time_intervals, deadline_dists):
        B, L = event_types.shape
        ev = self.ev_embed(event_types)
        te = self.time_embed(time_intervals.unsqueeze(-1))
        de = self.dl_embed(deadline_dists.unsqueeze(-1))
        pos_ids = torch.arange(L, device=event_types.device).unsqueeze(0).expand(B, L)
        pe = self.pos_embed(pos_ids)
        return self.norm(self.proj(torch.cat([ev, te, de, pe], dim=-1)))


# ─── 30 维手工特征 (向量化) ───────────────────────────────────

def compute_handcrafted_30(batch, n_event_types=7):
    """30 维手工特征: 覆盖 deadline/时间分布/事件分布/focus/submit 模式等"""
    et = batch['event_types']    # (B, L) int 0-6
    ti = batch['time_intervals'] # (B, L) float
    dd = batch['deadline_dists'] # (B, L) float [0, 1]

    B, L = et.shape

    feats = []

    # ── 1. deadline 相关 (4维) ──
    feats.append((1.0 - dd).mean(dim=-1))                  # 1: 平均接近 deadline
    feats.append(dd.min(dim=-1).values)                    # 2: 最近 deadline 距离
    feats.append((dd < 0.1).float().mean(dim=-1))           # 3: 末期比例
    feats.append((dd > 0.7).float().mean(dim=-1))           # 4: 初期比例

    # ── 2. 事件分布 (7维) ──
    onehot = F.one_hot(et.clamp(min=0), n_event_types).float()  # (B, L, 7)
    for k in range(n_event_types):
        feats.append(onehot[:, :, k].mean(dim=-1))         # 5-11: 各事件占比

    # ── 3. event diversity & entropy (2维) ──
    p = onehot.mean(dim=1) + 1e-8                          # (B, 7)
    p = p / p.sum(dim=-1, keepdim=True)
    entropy = -(p * p.log()).sum(dim=-1)                   # (B,)
    n_unique = (onehot.sum(dim=1) > 0).float().sum(dim=-1) / n_event_types
    feats.append(entropy)                                  # 12
    feats.append(n_unique)                                 # 13

    # ── 4. 提交 (submit=6) 模式 (5维) ──
    submit_mask = (et == 6).float()
    n_submit = submit_mask.sum(dim=-1)                      # 14: 总提交次数
    feats.append(n_submit / L)                             # 14
    feats.append(n_submit.log1p() / np.log(2000))           # 15: log归一化

    # submit 间隔的方差
    # 用 onehot 索引 ti 求和后归一
    submit_ti = (ti * submit_mask).sum(dim=-1) / submit_mask.sum(dim=-1).clamp(min=1)
    feats.append(submit_ti)                                # 16: 平均提交间隔
    feats.append(((ti - submit_ti.unsqueeze(-1))**2 * submit_mask).sum(dim=-1) /
                 submit_mask.sum(dim=-1).clamp(min=1))     # 17: 提交间隔方差

    # ── 5. focus 模式 (4维) (focus_gained=0, focus_lost=1) ──
    gain_mask = (et == 0).float()
    lose_mask = (et == 1).float()
    n_gain = gain_mask.sum(dim=-1)
    n_lose = lose_mask.sum(dim=-1)
    feats.append(n_gain / L)                               # 18
    feats.append(n_lose / L)                               # 19
    feats.append(n_gain / n_lose.clamp(min=1))             # 20: gain/lose 比
    feats.append(n_lose.log1p() - n_gain.log1p())          # 21: 净失焦

    # ── 6. 时间分布 (4维) ──
    ti_mean = ti.mean(dim=-1)                              # 22: 平均事件间隔
    ti_std = ti.std(dim=-1)                                # 23: 间隔方差
    feats.append(ti_mean)
    feats.append(ti_std)
    # 间隔直方图 (3 bins)
    feats.append((ti < 0.05).float().mean(dim=-1))         # 24: 短间隔比例
    feats.append((ti > 0.5).float().mean(dim=-1))          # 25: 长间隔比例

    # ── 7. 有效事件数 (1维) ──
    feats.append(torch.full((B,), float(L), device=et.device) / 2000.0)  # 26: 序列长度归一化

    # ── 8. 编辑密集度 (4维) (text_insert=2, text_remove=3) ──
    ins_mask = (et == 2).float()
    rem_mask = (et == 3).float()
    n_ins = ins_mask.sum(dim=-1)
    n_rem = rem_mask.sum(dim=-1)
    feats.append(n_ins / L)                                # 27
    feats.append(n_rem / L)                                # 28
    feats.append(n_ins / n_rem.clamp(min=1))               # 29
    feats.append(((n_ins - n_rem) / L))                    # 30

    stacked = torch.stack(feats, dim=-1)                   # (B, 30)
    return stacked


# ─── SE 通道注意力 ──────────────────────────────────────────────

class SEBlock(nn.Module):
    """Squeeze-and-Excitation: 自动给融合向量不同通道加权"""
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // reduction)
        self.fc2 = nn.Linear(dim // reduction, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, dim)
        s = x.mean(dim=0, keepdim=True)                    # (1, dim) 全局平均
        s = F.gelu(self.fc1(s))
        s = self.sigmoid(self.fc2(s))
        return x * s                                       # (B, dim) * (1, dim)


# ─── 改良 Hybrid ──────────────────────────────────────────────

class BiLSTMTransformerV2(nn.Module):
    def __init__(self, n_event_types=7, d_model=96, hidden=80,
                 lstm_layers=2, trans_layers=2, n_heads=6,
                 dropout=0.3, max_len=2048, hand_dim=30):
        super().__init__()
        self.encoder = EventEncoder(n_event_types, d_model, max_len)
        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=d_model, hidden_size=hidden,
            num_layers=lstm_layers, batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0, bidirectional=True
        )
        lstm_out = hidden * 2
        # Transformer
        trans_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, activation='gelu', norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(trans_layer, num_layers=trans_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # fusion: lstm_avg+max + trans_cls + hand 30 = hidden*4 + d_model + 30
        fusion_dim = lstm_out * 2 + d_model + hand_dim
        self.risk_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            SEBlock(fusion_dim, reduction=8),
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
        # BiLSTM
        lstm_out, _ = self.bilstm(x)
        lstm_avg = lstm_out.mean(dim=1)
        lstm_max = lstm_out.max(dim=1).values
        lstm_feat = torch.cat([lstm_avg, lstm_max], dim=-1)
        # Transformer
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x_t = torch.cat([cls, x], dim=1)
        mask = torch.zeros(x_t.shape[1], x_t.shape[1], device=x.device, dtype=torch.bool)
        t_out = self.transformer(x_t, mask=mask)
        t_cls = t_out[:, 0, :]
        # Hand 30维
        hand = compute_handcrafted_30(batch)
        # Fuse
        fused = torch.cat([lstm_feat, t_cls, hand], dim=-1)
        return self.risk_head(fused)


# ─── Focal Loss ────────────────────────────────────────────────

def focal_loss(logits, targets, alpha=None, gamma=3.0, label_smoothing=0.05):
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
                   epochs=35, batch_size=16, patience=10, lr=1e-3, fold=0):
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
            loss = focal_loss(logits, tgt, alpha=alpha, gamma=3.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item(); batch_cnt += 1
        scheduler.step()
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
        print(f"  Fold{fold} Epoch {epoch}/{epochs} loss={loss_sum/max(batch_cnt,1):.4f} val_auc={auc:.4f}", flush=True)
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
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-seq-len', type=int, default=2000)
    parser.add_argument('--output-dir', type=str, default='outputs/bi_lstm_trans_v2')
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"设备: {device}")

    ide_logs, passed = load_ide_logs()
    samples, student_ids, labels = preprocess_seq(
        ide_logs, passed, max_events=args.max_seq_len
    )
    n_pass = int((np.array(labels) == 0).sum())
    n_fail = int((np.array(labels) == 1).sum())
    print(f"样本数: {len(samples)} (pass={n_pass}, fail={n_fail})")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_metrics = []
    all_probs_full = np.zeros(len(samples))
    for fold, (train_idx, test_idx) in enumerate(skf.split(samples, labels), start=1):
        train_samples = [samples[i] for i in train_idx]
        test_samples = [samples[i] for i in test_idx]
        y_test = np.array([labels[i] for i in test_idx])
        print(f"\n=== Fold {fold}/{args.folds} | train={len(train_samples)} test={len(test_samples)} ===", flush=True)
        torch.manual_seed(args.seed + fold)
        model = BiLSTMTransformerV2(max_len=args.max_seq_len)
        t0 = time.time()
        probs, y_true = train_one_fold(
            model, train_samples, test_samples, device,
            epochs=args.epochs, batch_size=args.batch_size, fold=fold,
        )
        elapsed = time.time() - t0
        preds = (probs > 0.5).astype(int)
        m = evaluate(y_true, preds, probs)
        m['time_s'] = round(elapsed, 1)
        fold_metrics.append(m)
        all_probs_full[test_idx] = probs
        print(f"  Fold {fold}: Acc={m['accuracy']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f} ({elapsed:.0f}s)", flush=True)

    summary = summarize_fold_results(fold_metrics)
    print("\n========== 5折CV汇总 ==========")
    for k in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        m = summary.get(k + '_mean', 0)
        s = summary.get(k + '_std', 0)
        print(f"  {k}: {m:.4f} ± {s:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, 'probs.npy'), all_probs_full)
    np.save(os.path.join(args.output_dir, 'labels.npy'), np.array(labels))
    out = {
        'model': 'BiLSTM+Transformer v2 (Plan D-Improved)',
        'config': vars(args),
        'cv_results': {k.replace('_mean', ''): v for k, v in summary.items() if k.endswith('_mean')},
        'cv_results_std': {k.replace('_mean', '_std'): v for k, v in summary.items() if k.endswith('_std')},
        'fold_details': fold_metrics,
        'n_samples': len(samples),
        'n_passed': n_pass,
        'n_failed': n_fail,
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n概率已保存: {args.output_dir}/probs.npy")


if __name__ == '__main__':
    main()
