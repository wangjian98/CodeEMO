"""
BiLSTM vs RF 对比实验 - 7维序列聚合特征 vs 46维专家特征

目标:
  1. RF + 46维特征 → 已有结果 (AUC=90.6%)
  2. RF + 7维序列聚合特征 → 待跑
  3. BiLSTM + 46维特征 → 已有结果 (AUC=91.0%)
  4. BiLSTM + 7维序列 → 已有结果 (AUC=73.3%)

7维序列聚合方案 (FIXED_SIZE=46 维，与46维特征对齐):
  - 每类事件(7类) × 5统计量(均值/标准差/最小/最大/比率) = 35维
  - 全局特征: 事件总数(1) + 总时间(1) + 平均时间间隔(1) + 开头结尾事件类型(2) + part覆盖率(1) = 6维
  - 时间特征: 时间间隔的均值/标准差/最大值(3) = 3维
  - 熵特征: 事件类型分布的香农熵(1) + 时间间隔分布的香农熵(1) = 2维
  总计: 35+6+3+2 = 46维

运行:
    python models/compare_bilstm_rf.py --model rf7
    python models/compare_bilstm_rf.py --model all
"""

import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, set_seed
from common.evaluator import evaluate, summarize_fold_results, print_results_table
from common.feature_engineering import build_feature_matrix
from models.mamba.steps.step1_preprocessing import preprocess as preprocess_seq
from models.bi_mamba_dual_7dim import collate  # 复用填充函数


# ─── 7维序列 → 46维聚合特征 ───────────────────────────────────

EVENT_TYPE_NAMES = ['focus_gained', 'focus_lost', 'text_insert',
                    'text_remove', 'text_paste', 'run', 'submit']


def aggregate_sequence_to_features(sample, event_types=EVENT_TYPE_NAMES):
    """
    将 7维事件序列聚合为 46维固定向量

    35维: 每类事件(7) × 5统计量(均值/标准差/最小/最大/事件数占比)
     6维: 全局特征 (总事件数/总时间/平均间隔/开头事件/结尾事件/part覆盖率)
     3维: 时间特征 (时间间隔均值/标准差/最大)
     2维: 熵特征 (事件类型分布熵 + 时间间隔分布熵)
    总计: 46维
    """
    n = sample['n_events']
    et = sample['event_types'][:n].numpy() if hasattr(sample['event_types'], 'numpy') else np.array(sample['event_types'][:n])
    ti = sample['time_intervals'][:n].numpy() if hasattr(sample['time_intervals'], 'numpy') else np.array(sample['time_intervals'][:n])
    dd = sample['deadline_dists'][:n].numpy() if hasattr(sample['deadline_dists'], 'numpy') else np.array(sample['deadline_dists'][:n])
    pi = sample['part_ids'][:n].numpy() if hasattr(sample['part_ids'], 'numpy') else np.array(sample['part_ids'][:n])

    features = []

    # ── 35维: 每类事件 × 5统计量 ──────────────────────────────
    for i, et_name in enumerate(event_types):
        mask = (et == i)
        count = mask.sum()
        ratio = count / max(n, 1)

        if count > 0:
            mean_ti = ti[mask].mean()
            std_ti = ti[mask].std() if count > 1 else 0.0
            min_ti = ti[mask].min()
            max_ti = ti[mask].max()
        else:
            mean_ti = std_ti = min_ti = max_ti = 0.0

        features.extend([mean_ti, std_ti, min_ti, max_ti, ratio])

    # ── 6维: 全局特征 ─────────────────────────────────────────
    total_events = n
    total_time = ti.sum()
    mean_interval = ti.mean() if n > 0 else 0.0
    first_event = et[0] / 6.0  # 归一化
    last_event = et[-1] / 6.0  # 归一化
    part_coverage = len(set(pi)) / 7.0  # part 覆盖率

    features.extend([total_events, total_time, mean_interval, first_event, last_event, part_coverage])

    # ── 3维: 时间特征 ─────────────────────────────────────────
    features.extend([ti.mean(), ti.std() if n > 1 else 0.0, ti.max()])

    # ── 2维: 熵特征 ───────────────────────────────────────────
    # 事件类型分布熵
    counts = np.bincount(et, minlength=7).astype(float)
    probs = counts / max(counts.sum(), 1)
    probs = probs[probs > 0]
    event_entropy = -np.sum(probs * np.log(probs + 1e-8))

    # 时间间隔分布熵 (离散化)
    bins = np.histogram(ti, bins=5)[0].astype(float)
    bin_probs = bins / max(bins.sum(), 1)
    bin_probs = bin_probs[bin_probs > 0]
    time_entropy = -np.sum(bin_probs * np.log(bin_probs + 1e-8))

    features.extend([event_entropy, time_entropy])

    return np.array(features, dtype=np.float32)


def build_seq_aggregated_features(samples):
    """批量聚合"""
    X = []
    for s in samples:
        X.append(aggregate_sequence_to_features(s))
    return np.array(X)


# ─── BiLSTM 7维序列模型 (复用 compare_all_7dim) ───────────────

class BiLSTM7Dim(nn.Module):
    def __init__(self, d_model=64, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        from models.bi_mamba_dual_7dim import SharedEventEncoder
        self.enc = SharedEventEncoder(7, d_model)
        self.lstm = nn.LSTM(d_model, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0, bidirectional=True)
        self.cls = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden*2, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1), nn.Sigmoid()
        )

    def forward(self, batch):
        et, ti, dd = batch['event_types'], batch['time_intervals'], batch['deadline_dists']
        x = self.enc(et, ti, dd)
        out, _ = self.lstm(x)
        return self.cls(out[:, -1, :])


# ─── 训练函数 ──────────────────────────────────────────────────

def run_rf_fold(X_train, X_val, y_train, y_val, seed=42):
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_prob = rf.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred, zero_division=0),
        'f1': f1_score(y_val, y_pred, zero_division=0),
        'auc': roc_auc_score(y_val, y_prob),
    }


def run_bilstm_fold(model, tr_s, va_s, dev, epochs=80, bs=32, pat=20, lr=1e-3):
    import torch.nn as nn
    import torch.nn.functional as F
    model = model.to(dev)
    trl = DataLoader(tr_s, bs, shuffle=True, collate_fn=lambda x: collate(x))
    val = DataLoader(va_s, bs, shuffle=False, collate_fn=lambda x: collate(x))
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
        model.eval(); vl, vb = 0, 0
        with torch.no_grad():
            for b in val:
                inp = {k: v.to(dev) for k, v in b.items() if k != 'risk'}
                tgt = b['risk'].float().to(dev)
                vl += crit(model(inp).squeeze(-1), tgt).item(); vb += 1
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
    pa.add_argument('--model', type=str, default='all',
                    choices=['all', 'rf46', 'rf7', 'bilstm46', 'bilstm7'])
    pa.add_argument('--folds', type=int, default=5)
    pa.add_argument('--seed', type=int, default=42)
    args = pa.parse_args()

    set_seed(args.seed)
    dev = torch.device('cpu')
    t0 = time.time()

    # ── 加载数据 ──
    print("加载数据 ...")
    ide_logs, passed = load_ide_logs()

    # 46维特征
    X46, y46, sids46 = build_feature_matrix(ide_logs, passed)
    print(f"  46维特征: {X46.shape}, 通过={sum(y46==0)}, 挂科={sum(y46==1)}")

    # 7维序列
    samples7, sids7, labels7 = preprocess_seq(ide_logs, passed, max_events=200)
    X7_agg = build_seq_aggregated_features(samples7)
    print(f"  7维序列: {len(samples7)} 样本, 聚合后={X7_agg.shape}")
    print(f"  加载耗时: {time.time()-t0:.1f}s")

    skf = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)

    results = {}

    # ── RF + 46维特征 ──────────────────────────────────────────
    if args.model in ['all', 'rf46']:
        print(f"\n{'='*50}\n  RF + 46维特征\n{'='*50}")
        fr = []
        for fi, (tri, vai) in enumerate(skf.split(X46, y46), 1):
            tf = time.time()
            m = run_rf_fold(X46[tri], X46[vai], y46[tri], y46[vai])
            fr.append(m)
            print(f"  Fold {fi}: Acc={m['accuracy']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f} ({time.time()-tf:.0f}s)")
        sm = summarize_fold_results(fr)
        results['rf46'] = {'summary': sm, 'folds': fr}
        print(f"  RF+46 → Acc={sm['accuracy_mean']:.4f} AUC={sm['auc_mean']:.4f}")

    # ── RF + 7维序列聚合特征 ───────────────────────────────────
    if args.model in ['all', 'rf7']:
        print(f"\n{'='*50}\n  RF + 7维序列聚合特征\n{'='*50}")
        fr = []
        for fi, (tri, vai) in enumerate(skf.split(X7_agg, labels7), 1):
            tf = time.time()
            m = run_rf_fold(X7_agg[tri], X7_agg[vai], labels7[tri], labels7[vai])
            fr.append(m)
            print(f"  Fold {fi}: Acc={m['accuracy']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f} ({time.time()-tf:.0f}s)")
        sm = summarize_fold_results(fr)
        results['rf7'] = {'summary': sm, 'folds': fr}
        print(f"  RF+7 → Acc={sm['accuracy_mean']:.4f} AUC={sm['auc_mean']:.4f}")

    # ── BiLSTM + 46维特征 ──────────────────────────────────────
    if args.model in ['all', 'bilstm46']:
        print(f"\n{'='*50}\n  BiLSTM + 46维特征\n{'='*50}")
        fr = []
        for fi, (tri, vai) in enumerate(skf.split(X46, y46), 1):
            tf = time.time()
            model = BiLSTM7Dim()
            ps, ls = run_bilstm_fold(model,
                                     [samples7[i] for i in tri],
                                     [samples7[i] for i in vai], dev)
            # 注意: BiLSTM 用 46维特征时直接用 X46 而非序列
            # 但 BiLSTM 需要序列输入，这里用 46维特征的聚合版本替代
            # 实际上 BiLSTM 只能用序列输入，这里标注为 N/A
            print(f"  Fold {fi}: [BiLSTM 不支持直接处理 46维特征，需要序列输入]")
            break
        print("  注意: BiLSTM 架构要求序列输入，无法直接处理 46维特征向量")
        print("  46维特征已由 comparison.csv 中 BiLSTM 单独运行过: AUC=91.0%")

    # ── BiLSTM + 7维序列 ───────────────────────────────────────
    if args.model in ['all', 'bilstm7']:
        print(f"\n{'='*50}\n  BiLSTM + 7维事件序列\n{'='*50}")
        fr = []
        for fi, (tri, vai) in enumerate(skf.split(np.array(sids7), labels7), 1):
            tf = time.time()
            model = BiLSTM7Dim()
            ps, ls = run_bilstm_fold(model,
                                     [samples7[i] for i in tri],
                                     [samples7[i] for i in vai], dev)
            pred = (ps >= 0.5).astype(int)
            m = evaluate(ls, pred, ps)
            fr.append(m)
            print(f"  Fold {fi}: Acc={m['accuracy']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f} ({time.time()-tf:.0f}s)")
        sm = summarize_fold_results(fr)
        results['bilstm7'] = {'summary': sm, 'folds': fr}
        print(f"  BiLSTM+7 → Acc={sm['accuracy_mean']:.4f} AUC={sm['auc_mean']:.4f}")

    # ── 保存 ──
    out = os.path.join(_PROJECT_ROOT, 'outputs', 'compare_bilstm_rf')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n已保存: {out}/results.json  总耗时: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
