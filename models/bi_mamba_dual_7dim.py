"""
BiLSTM + Mamba 双塔融合模型 - 仅使用 7维事件序列

架构:
  7维事件序列 ──→ 共享 EventEncoder (7→64)
                      ↓
          ┌───────────┴───────────┐
          ↓                       ↓
    BiLSTM Tower (128维)    Mamba Tower (48维)
          └───────────┬───────────┘
                      ↓ concat (176维)
                   FC → 2类

与 TripleTower 的区别:
  - 无 46维统计特征塔（纯 7维序列）
  - 共享 EventEncoder（节省参数）
  - BiLSTM 用 last_hidden + mean 融合

运行:
    python models/bi_mamba_dual_7dim.py
    python models/bi_mamba_dual_7dim.py --folds 5 --epochs 80
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

class SharedEventEncoder(nn.Module):
    """7维事件编码器: event_type embedding + time + deadline → d_model"""
    def __init__(self, n_event_types=7, d_model=64):
        super().__init__()
        self.ev = nn.Embedding(n_event_types, 16)
        self.te = nn.Linear(1, 8)
        self.de = nn.Linear(1, 8)
        self.proj = nn.Linear(16 + 8 + 8, d_model)

    def forward(self, et, ti, dd):
        return self.proj(torch.cat([
            self.ev(et),
            self.te(ti.unsqueeze(-1)),
            self.de(dd.unsqueeze(-1))
        ], dim=-1))


# ─── BiLSTM Tower ──────────────────────────────────────────────

class BiLSTMTower(nn.Module):
    """
    Tower A: 7维序列 → 双向 LSTM → 128维表征

    融合策略: last_output(128维) 直接取，双向 LSTM 的最后时刻输出
    已经包含了正向最后+反向最后的拼接。
    """
    def __init__(self, d_model=64, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0,
            bidirectional=True
        )
        self.output_dim = hidden * 2  # 128
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoded):
        """
        encoded: (batch, seq_len, d_model)
        Returns: (batch, 128)
        """
        out, _ = self.lstm(encoded)           # (B, L, hidden*2)
        last = out[:, -1, :]                  # (B, hidden*2) = (B, 128)
        mean = out.mean(dim=1)                 # (B, hidden*2)
        return self.dropout(last + mean)       # (B, 128)


# ─── Mamba Tower ───────────────────────────────────────────────

class MambaTower(nn.Module):
    """
    Tower B: 7维序列 → S6 选择性扫描 → 48维表征

    使用 2个 S6Block，串行扫描（CPU 友好）。
    多尺度池化: 全局均值 + 最后状态 + part均值
    """
    def __init__(self, d_model=48, layers=2, d_state=8, dropout=0.2,
                 encoder_d_model=64):
        super().__init__()

        class S6Block(nn.Module):
            def __init__(self, dm, ds=8):
                super().__init__()
                self.ip = nn.Linear(dm, int(2*dm))
                self.c1d = nn.Conv1d(int(2*dm), int(2*dm), 4, padding=3,
                                     groups=int(2*dm))
                self.dt_rank = max(1, int(2*dm)//16)
                self.xp = nn.Linear(int(2*dm), self.dt_rank + ds*2, bias=False)
                self.dti = nn.Parameter(torch.empty(self.dt_rank))
                nn.init.uniform_(self.dti, -1.0, 0.0)
                A = -torch.abs(torch.arange(1, ds+1, dtype=torch.float32))
                self.A = nn.Parameter(A.unsqueeze(0).expand(int(2*dm), -1).clone())
                self.D = nn.Parameter(torch.ones(int(2*dm)))
                self.op = nn.Linear(int(2*dm), dm, bias=False)
                self.act = nn.SiLU()
                self.norm = nn.RMSNorm(dm)

            def sel_scan(self, x, dt, A, B, C, D):
                B2, SL, DI = x.shape
                DS = A.shape[1]
                dt = F.silu(dt)
                dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)).clamp(max=1-1e-6)
                dB = (dt.unsqueeze(-1) * B.unsqueeze(2)).clamp(max=1e6)
                h = torch.zeros(B2, DI, DS, device=x.device, dtype=x.dtype)
                ys = []
                for t in range(SL):
                    h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
                    ys.append(torch.bmm(h, C[:, t].unsqueeze(-1)).squeeze(-1))
                return torch.stack(ys, 1) + x * D

            def forward(self, x):
                B2, SL, _ = x.shape
                xi = self.act(self.ip(x))
                xc = self.c1d(xi.transpose(1, 2))[:, :, :SL].transpose(1, 2)
                xc = self.act(xc)
                xp = self.xp(xc.reshape(-1, xc.size(-1)))
                dt, Bs, Cs = torch.split(xp, [self.dt_rank, self.A.shape[1], self.A.shape[1]], dim=-1)
                dt = dt.reshape(B2, SL, self.dt_rank)
                Bs = Bs.reshape(B2, SL, -1)
                Cs = Cs.reshape(B2, SL, -1)
                if self.dt_rank < xc.size(-1):
                    p = torch.zeros(B2, SL, xc.size(-1), device=dt.device, dtype=dt.dtype)
                    p[:, :, :self.dt_rank] = dt
                    dt = p
                y = self.sel_scan(xc, dt, self.A, Bs, Cs, self.D)
                return self.op(y)

        class MBlock(nn.Module):
            def __init__(self, dm, ds=8):
                super().__init__()
                self.mixer = S6Block(dm, ds)
                self.norm = nn.RMSNorm(dm)

            def forward(self, x):
                return self.mixer(self.norm(x)) + x

        self.layers = nn.ModuleList([MBlock(d_model, d_state) for _ in range(layers)])
        self.final_norm = nn.RMSNorm(d_model)
        self.input_proj = nn.Linear(encoder_d_model, d_model)
        self.output_dim = d_model  # 48

    def forward(self, encoded, part_ids):
        x = self.input_proj(encoded)  # (B, L, 48)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)  # (B, L, 48)

        seq_mean = x.mean(dim=1)      # (B, 48)
        seq_last = x[:, -1, :]        # (B, 48)

        # 按 part 分组均值
        part_means = []
        for p in range(1, 8):
            mask = (part_ids == p)
            if mask.any():
                pm = (x * mask.unsqueeze(-1)).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
            else:
                pm = torch.zeros_like(seq_mean)
            part_means.append(pm)
        part_r = torch.stack(part_means, dim=1).mean(dim=1)  # (B, 48)

        return (seq_mean + seq_last + part_r) / 3  # (B, 48)


# ─── 双塔融合模型 ─────────────────────────────────────────────

class BiLSTMMambaDualTower(nn.Module):
    """
    BiLSTM + Mamba 双塔融合 - 仅使用 7维事件序列

    设计:
      - 共享 EventEncoder (7→64)，两个 tower 共用
      - BiLSTM Tower: 双向建模，128维输出
      - Mamba Tower: 选择性扫描，48维输出
      - concat: 176维 → FC → 2类
    """
    def __init__(self,
                 encoder_d_model=64,
                 bilstm_hidden=64,
                 bilstm_layers=2,
                 bilstm_dropout=0.3,
                 mamba_d_model=48,
                 mamba_layers=2,
                 mamba_d_state=8,
                 mamba_dropout=0.2,
                 fusion_hidden=96,
                 dropout=0.3):
        super().__init__()

        # 共享编码器
        self.encoder = SharedEventEncoder(n_event_types=7, d_model=encoder_d_model)

        # BiLSTM Tower
        self.bilstm_tower = BiLSTMTower(
            d_model=encoder_d_model,
            hidden=bilstm_hidden,
            layers=bilstm_layers,
            dropout=bilstm_dropout
        )

        # Mamba Tower
        self.mamba_tower = MambaTower(
            d_model=mamba_d_model,
            layers=mamba_layers,
            d_state=mamba_d_state,
            dropout=mamba_dropout,
            encoder_d_model=encoder_d_model
        )

        # 融合层
        combined_dim = self.bilstm_tower.output_dim + self.mamba_tower.output_dim  # 128+48=176
        self.fusion = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_dim, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 2)
        )

        # Xavier 初始化
        for m in self.fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch):
        et = batch['event_types']
        ti = batch['time_intervals']
        dd = batch['deadline_dists']
        pi = batch.get('part_ids', torch.zeros_like(et))

        # 编码: (B, L, encoder_d_model=64)
        encoded = self.encoder(et, ti, dd)

        # BiLSTM tower: 128维
        bilstm_repr = self.bilstm_tower(encoded)

        # Mamba tower: 48维
        mamba_repr = self.mamba_tower(encoded, pi)

        # concat: 176维
        combined = torch.cat([bilstm_repr, mamba_repr], dim=-1)

        return self.fusion(combined)  # (B, 2)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Batch 填充 ────────────────────────────────────────────────

def collate(batch_samples, max_len=None):
    if max_len is None:
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

def train_fold(model, tr_s, va_s, dev, epochs=80, bs=32, pat=20, lr=1e-3, label_smooth=0.05):
    model = model.to(dev)
    trl = DataLoader(tr_s, bs, shuffle=True, collate_fn=collate)
    val = DataLoader(va_s, bs, shuffle=False, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    best_vl, best_st, pc = float('inf'), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for b in trl:
            inp = {k: v.to(dev) for k, v in b.items() if k != 'risk'}
            tgt = b['risk'].to(dev)
            opt.zero_grad()
            loss = crit(model(inp), tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        model.eval(); vl, vb = 0, 0; ps, ls = [], []
        with torch.no_grad():
            for b in val:
                inp = {k: v.to(dev) for k, v in b.items() if k != 'risk'}
                tgt = b['risk'].to(dev)
                logits = model(inp)
                vl += crit(logits, tgt).item(); vb += 1
                ps.extend(torch.softmax(logits, -1)[:, 1].cpu().numpy())
                ls.extend(tgt.cpu().numpy())
        avl = vl / max(vb, 1)
        if avl < best_vl:
            best_vl = avl; best_st = {k: v.clone() for k, v in model.state_dict().items()}; pc = 0
        else:
            pc += 1
            if pc >= pat:
                print(f"    Epoch {ep}: val={avl:.4f}  [早停 patience={pat}]")
                break

    model.load_state_dict(best_st); model.eval()
    ps, ls = [], []
    with torch.no_grad():
        for b in val:
            inp = {k: v.to(dev) for k, v in b.items() if k != 'risk'}
            logits = model(inp)
            ps.extend(torch.softmax(logits, -1)[:, 1].cpu().numpy())
            ls.extend(b['risk'].numpy())
    return np.array(ps), np.array(ls)


# ─── 主函数 ────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser(description='BiLSTM + Mamba 双塔融合 (7维序列)')
    pa.add_argument('--folds', type=int, default=5)
    pa.add_argument('--epochs', type=int, default=80)
    pa.add_argument('--batch-size', type=int, default=32)
    pa.add_argument('--patience', type=int, default=20)
    pa.add_argument('--lr', type=float, default=1e-3)
    pa.add_argument('--seed', type=int, default=42)
    pa.add_argument('--max-seq-len', type=int, default=200)
    pa.add_argument('--label-smoothing', type=float, default=0.05)
    args = pa.parse_args()

    set_seed(args.seed)
    dev = torch.device('cpu')
    t0 = time.time()

    print("=" * 60)
    print("  BiLSTM + Mamba 双塔融合  |  输入: 7维事件序列")
    print("=" * 60)
    print(f"  设备={dev}  折数={args.folds}  max_seq={args.max_seq_len}")
    print(f"  batch={args.batch_size}  lr={args.lr}  epochs={args.epochs}")
    print(f"  label_smoothing={args.label_smoothing}  patience={args.patience}")
    print("=" * 60)

    # 加载数据
    print("\n[数据加载] 7维事件序列 ...")
    ide_logs, passed = load_ide_logs()
    samples, sids, labels = preprocess_seq(ide_logs, passed, max_events=args.max_seq_len)
    print(f"  样本={len(samples)}  通过={sum(labels==0)}  挂科={sum(labels==1)}")
    print(f"  加载耗时: {time.time()-t0:.1f}s")

    skf = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    fold_results = []

    for fi, (tri, vai) in enumerate(skf.split(np.array(sids), labels), 1):
        tf = time.time()
        model = BiLSTMMambaDualTower()
        ps, ls = train_fold(model,
                            [samples[i] for i in tri],
                            [samples[i] for i in vai],
                            dev, epochs=args.epochs, bs=args.batch_size,
                            pat=args.patience, lr=args.lr,
                            label_smooth=args.label_smoothing)
        pred = (ps >= 0.5).astype(int)
        m = evaluate(ls, pred, ps)
        fold_results.append(m)
        print(f"  Fold {fi}: Acc={m['accuracy']:.4f}  Prec={m['precision']:.4f}  "
              f"Recall={m['recall']:.4f}  F1={m['f1']:.4f}  AUC={m['auc']:.4f}  "
              f"({time.time()-tf:.0f}s)")

    summary = summarize_fold_results(fold_results)
    print(f"\n  双塔汇总:")
    print(f"    Acc={summary['accuracy_mean']:.4f}  "
          f"Prec={summary['precision_mean']:.4f}  "
          f"Recall={summary['recall_mean']:.4f}")
    print(f"    F1={summary['f1_mean']:.4f}  AUC={summary['auc_mean']:.4f}")

    # 保存
    out = os.path.join(_PROJECT_ROOT, 'outputs', 'bi_mamba_dual_7dim')
    os.makedirs(out, exist_ok=True)
    results = {
        'model': 'BiLSTM+Mamba DualTower (7-dim only)',
        'architecture': 'Shared EventEncoder(7→64) → BiLSTM(128) + Mamba(48) → concat(176) → FC → 2',
        'config': {
            'encoder_d_model': 64,
            'bilstm_hidden': 64, 'bilstm_layers': 2, 'bilstm_dropout': 0.3,
            'mamba_d_model': 48, 'mamba_layers': 2, 'mamba_d_state': 8,
            'max_seq_len': args.max_seq_len,
            'label_smoothing': args.label_smoothing,
        },
        'training': {
            'n_folds': args.folds, 'epochs': args.epochs,
            'batch_size': args.batch_size, 'patience': args.patience, 'lr': args.lr,
        },
        'cv_results': {k: float(v) for k, v in summary.items()},
        'fold_details': fold_results,
        'n_samples': len(samples),
        'params': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    with open(os.path.join(out, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out}/results.json  总耗时: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
