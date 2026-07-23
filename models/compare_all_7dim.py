"""
CodeEMO 全模型 7维事件序列对比实验

模型:
  1. LSTM      (单向, d_model=64, hidden=64, layers=2)
  2. BiLSTM    (双向, d_model=64, hidden=64, layers=2)
  3. Mamba     (S6, d_model=48, layers=2, d_state=8)
  4. Transformer (4头, 3层, d_model=64)

输入: 7维事件序列 (event_type + time_interval + deadline_dist)

用法:
    python models/compare_all_7dim.py
    python models/compare_all_7dim.py --folds 5 --max-seq-len 200
    # 分别单独运行:
    python models/compare_all_7dim.py --model mamba
    python models/compare_all_7dim.py --model transformer
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


# ─── 共享编码器 ────────────────────────────────────────────────

class EventEncoder(nn.Module):
    """7维事件编码器"""
    def __init__(self, n_event_types=7, d_model=64):
        super().__init__()
        self.ev = nn.Embedding(n_event_types, 16)
        self.te = nn.Linear(1, 8)
        self.de = nn.Linear(1, 8)
        self.proj = nn.Linear(16 + 8 + 8, d_model)

    def forward(self, et, ti, dd):
        return self.proj(torch.cat([self.ev(et), self.te(ti.unsqueeze(-1)), self.de(dd.unsqueeze(-1))], dim=-1))


# ─── 模型定义 ──────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """单向 LSTM"""
    def __init__(self, d_model=64, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.encoder = EventEncoder(7, d_model)
        self.lstm = nn.LSTM(d_model, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0, bidirectional=False)
        self.cls = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 32),
                                 nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, batch):
        x = self.encoder(batch['event_types'], batch['time_intervals'], batch['deadline_dists'])
        _, (hn, _) = self.lstm(x)
        return self.cls(hn[-1])


class BiLSTMModel(nn.Module):
    """双向 LSTM"""
    def __init__(self, d_model=64, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.encoder = EventEncoder(7, d_model)
        self.lstm = nn.LSTM(d_model, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0, bidirectional=True)
        self.cls = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, 64),
                                 nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, batch):
        x = self.encoder(batch['event_types'], batch['time_intervals'], batch['deadline_dists'])
        out, _ = self.lstm(x)
        return self.cls(out[:, -1, :])


class MambaModel(nn.Module):
    """Mamba S6 模型"""
    def __init__(self, d_model=48, layers=2, d_state=8, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.encoder = EventEncoder(7, d_model)

        class S6Block(nn.Module):
            def __init__(self, dm, ds=8):
                super().__init__()
                self.ip = nn.Linear(dm, int(2*dm)); self.c1d = nn.Conv1d(int(2*dm), int(2*dm), 4, padding=3, groups=int(2*dm))
                self.dt_rank = max(1, int(2*dm)//16)
                self.xp = nn.Linear(int(2*dm), self.dt_rank + ds*2, bias=False)
                self.dti = nn.Parameter(torch.empty(self.dt_rank)); nn.init.uniform_(self.dti, -1, 0)
                A = -torch.abs(torch.arange(1, ds+1, dtype=torch.float32))
                self.A = nn.Parameter(A.unsqueeze(0).expand(int(2*dm), -1).clone())
                self.D = nn.Parameter(torch.ones(int(2*dm)))
                self.op = nn.Linear(int(2*dm), dm, bias=False)
                self.act = nn.SiLU()
                self.norm = nn.RMSNorm(dm)

            def sel_scan(self, x, dt, A, B, C, D):
                B2, SL, DI = x.shape; DS = A.shape[1]
                dt = F.silu(dt)
                dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)).clamp(max=1-1e-6)
                dB = (dt.unsqueeze(-1) * B.unsqueeze(2)).clamp(max=1e6)
                h = torch.zeros(B2, DI, DS, device=x.device, dtype=x.dtype)
                ys = []
                for t in range(SL):
                    h = dA[:,t]*h + dB[:,t]*x[:,t].unsqueeze(-1)
                    ys.append(torch.bmm(h, C[:,t].unsqueeze(-1)).squeeze(-1))
                return torch.stack(ys,1) + x*D

            def forward(self, x):
                B2, SL, _ = x.shape
                xi = self.act(self.ip(x))
                xc = self.c1d(xi.transpose(1,2))[:,:,:SL].transpose(1,2)
                xc = self.act(xc)
                xp = self.xp(xc.reshape(-1, xc.size(-1)))
                dt, B2s, C2s = torch.split(xp, [self.dt_rank, self.A.shape[1], self.A.shape[1]], dim=-1)
                dt = dt.reshape(B2, SL, self.dt_rank)
                B2s = B2s.reshape(B2, SL, -1); C2s = C2s.reshape(B2, SL, -1)
                if self.dt_rank < xc.size(-1):
                    p = torch.zeros(B2, SL, xc.size(-1), device=dt.device, dtype=dt.dtype)
                    p[:,:,:self.dt_rank] = dt; dt = p
                y = self.sel_scan(xc, dt, self.A, B2s, C2s, self.D)
                return self.op(y)

        class MBlock(nn.Module):
            def __init__(self, dm, ds=8):
                super().__init__()
                self.m = S6Block(dm, ds); self.n = nn.RMSNorm(dm)
            def forward(self, x): return self.m(self.n(x)) + x

        self.layers = nn.ModuleList([MBlock(d_model, d_state) for _ in range(layers)])
        self.final_norm = nn.RMSNorm(d_model)
        self.cls = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 32),
                                 nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, batch):
        x = self.encoder(batch['event_types'], batch['time_intervals'], batch['deadline_dists'])
        for layer in self.layers: x = layer(x)
        x = self.final_norm(x)
        seq_mean = x.mean(1); seq_last = x[:, -1, :]
        # part 分组
        pi = batch.get('part_ids', torch.zeros_like(batch['event_types']))
        part_means = []
        for p in range(1, 8):
            mask = (pi == p)
            if mask.any():
                pm = (x * mask.unsqueeze(-1)).sum(1) / (mask.sum(1, keepdim=True) + 1e-8)
            else:
                pm = torch.zeros_like(seq_mean)
            part_means.append(pm)
        part_r = torch.stack(part_means, 1).mean(1)
        repr = (seq_mean + seq_last + part_r) / 3
        return self.cls(repr)


class TransformerModel(nn.Module):
    """Transformer 编码器 (支持 7维序列)"""
    def __init__(self, d_model=64, nhead=4, layers=3, dropout=0.2):
        super().__init__()
        self.encoder = EventEncoder(7, d_model)
        self.pos = nn.Parameter(torch.randn(1, 500, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128,
                                         dropout=dropout, batch_first=True)
        self.tf = nn.TransformerEncoder(enc, layers)
        self.cls = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 32),
                                 nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, batch, max_len=None):
        et = batch['event_types']
        ti = batch['time_intervals']
        dd = batch['deadline_dists']
        if max_len:
            et, ti, dd = et[:, :max_len], ti[:, :max_len], dd[:, :max_len]
        x = self.encoder(et, ti, dd)  # (B, L, d)
        L = x.size(1)
        x = x + self.pos[:, :L, :]
        out = self.tf(x)  # (B, L, d)
        pooled = out.mean(1)  # (B, d)
        return self.cls(pooled)


MODEL_REGISTRY = {
    'lstm':        (LSTMModel,        {'d_model': 64, 'hidden': 64, 'layers': 2, 'dropout': 0.3}),
    'bilstm':      (BiLSTMModel,      {'d_model': 64, 'hidden': 64, 'layers': 2, 'dropout': 0.3}),
    'mamba':       (MambaModel,       {'d_model': 48, 'layers': 2, 'd_state': 8, 'dropout': 0.2}),
    'transformer': (TransformerModel, {'d_model': 64, 'nhead': 4, 'layers': 3, 'dropout': 0.2}),
}


# ─── Batch collation ────────────────────────────────────────────

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


# ─── 训练 ──────────────────────────────────────────────────────

def train_fold(model, tr_s, va_s, dev, epochs=80, bs=32, pat=20, lr=1e-3, max_len=None):
    model = model.to(dev)
    trl = DataLoader(tr_s, bs, shuffle=True, collate_fn=lambda x: collate(x, max_len))
    val = DataLoader(va_s, bs, shuffle=False, collate_fn=lambda x: collate(x, max_len))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCELoss()
    best_vl, best_st, pc = float('inf'), None, 0
    for ep in range(1, epochs+1):
        model.train()
        for b in trl:
            inp = {k: v.to(dev) for k, v in b.items() if k != 'risk'}
            tgt = b['risk'].float().to(dev)
            opt.zero_grad()
            loss = crit(model(inp).squeeze(-1), tgt)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval(); vl, vb = 0, 0; ps, ls = [], []
        with torch.no_grad():
            for b in val:
                inp = {k: v.to(dev) for k, v in b.items() if k != 'risk'}
                tgt = b['risk'].float().to(dev)
                vl += crit(model(inp).squeeze(-1), tgt).item(); vb += 1
                ps.extend(model(inp).squeeze(-1).cpu().numpy())
                ls.extend(tgt.cpu().numpy())
        avl = vl / max(vb, 1)
        if avl < best_vl:
            best_vl = avl; best_st = {k: v.clone() for k, v in model.state_dict().items()}; pc = 0
        else:
            pc += 1
            if pc >= pat: break
    model.load_state_dict(best_st); model.eval()
    ps, ls = [], []
    with torch.no_grad():
        for b in val:
            inp = {k: v.to(dev) for k, v in b.items() if k != 'risk'}
            ps.extend(model(inp).squeeze(-1).cpu().numpy())
            ls.extend(b['risk'].numpy())
    return np.array(ps), np.array(ls)


# ─── 主函数 ────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--folds', type=int, default=5)
    pa.add_argument('--epochs', type=int, default=80)
    pa.add_argument('--batch-size', type=int, default=32)
    pa.add_argument('--patience', type=int, default=20)
    pa.add_argument('--lr', type=float, default=1e-3)
    pa.add_argument('--seed', type=int, default=42)
    pa.add_argument('--max-seq-len', type=int, default=200)
    pa.add_argument('--model', type=str, default='all',
                    choices=['all', 'lstm', 'bilstm', 'mamba', 'transformer'])
    args = pa.parse_args()

    set_seed(args.seed)
    dev = torch.device('cpu')
    t0 = time.time()

    print("加载数据 ...")
    ide_logs, passed = load_ide_logs()
    samples, sids, labels = preprocess_seq(ide_logs, passed, max_events=args.max_seq_len)
    print(f"样本={len(samples)} 通过={sum(labels==0)} 挂科={sum(labels==1)}")

    skf = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    all_results = {}
    models_to_run = MODEL_REGISTRY if args.model == 'all' else {args.model: MODEL_REGISTRY[args.model]}

    for mname, (mclass, mkwargs) in models_to_run.items():
        print(f"\n{'='*50}\n  {mname.upper()} (7-dim sequences)\n{'='*50}")
        fr = []
        for fi, (tri, vai) in enumerate(skf.split(np.array(sids), labels), 1):
            tf = time.time()
            model = mclass(**mkwargs)
            ps, ls = train_fold(model,
                                [samples[i] for i in tri], [samples[i] for i in vai],
                                dev, epochs=args.epochs, bs=args.batch_size,
                                pat=args.patience, lr=args.lr, max_len=args.max_seq_len)
            pred = (ps >= 0.5).astype(int)
            m = evaluate(ls, pred, ps)
            fr.append(m)
            print(f"  Fold {fi}: Acc={m['accuracy']:.4f} F1={m['f1']:.4f} "
                  f"AUC={m['auc']:.4f} ({time.time()-tf:.0f}s)")
        sm = summarize_fold_results(fr)
        all_results[mname] = {'summary': sm, 'folds': fr,
                              'params': sum(p.numel() for p in model.parameters() if p.requires_grad)}
        print(f"\n  {mname} → Acc={sm['accuracy_mean']:.4f} F1={sm['f1_mean']:.4f} AUC={sm['auc_mean']:.4f}")

    # 保存
    out = os.path.join(_PROJECT_ROOT, 'outputs', 'compare_all_7dim')
    os.makedirs(out, exist_ok=True)
    save = {}
    for k, v in all_results.items():
        save[k] = {
            'input': '7-dim event sequences',
            'params': v['params'],
            'cv_results': {x: float(y) if not isinstance(y, list) else y
                           for x, y in v['summary'].items()},
            'fold_details': v['folds'],
        }
    with open(os.path.join(out, 'results.json'), 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\n已保存: {out}/results.json  总耗时: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
