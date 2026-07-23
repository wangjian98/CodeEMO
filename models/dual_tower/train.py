"""
双塔模型训练脚本 - 端到端联合训练

两种输入:
  1. 46维统计特征 (common/feature_engineering.py)
  2. 7维事件序列  (models/mamba/steps/step1_preprocessing.py)

三塔融合: BiLSTM(46-dim) + Mamba(7-dim序列) + MLP(46-dim)

训练策略:
  - 端到端联合优化，不冻结任何部分
  - 5折分层交叉验证 + Early stopping (patience=20)
  - Label Smoothing (0.05) 防止过拟合
  - 梯度裁剪 (max_norm=1.0)
  - 优化: AdamW(lr=5e-4, weight_decay=0.01)

用法:
    python models/dual_tower/train.py
    python models/dual_tower/train.py --folds 5 --epochs 100
    python models/dual_tower/train.py --max-seq-len 500 --batch-size 32
"""

import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, set_seed
from common.evaluator import evaluate, summarize_fold_results, print_results_table
from common.feature_engineering import build_feature_matrix
from models.dual_tower.model import DualTowerModel
from models.mamba.steps.step1_preprocessing import preprocess as preprocess_seq


# ─── Batch collation ────────────────────────────────────────────

def collate_for_dual_tower(samples):
    """构建 Mamba Tower 的 batch (带填充)"""
    max_len = max(s['n_events'] for s in samples)
    batch_et, batch_ti, batch_dd, batch_pi = [], [], [], []
    for s in samples:
        n = s['n_events']
        et = s['event_types'][:n]
        ti = s['time_intervals'][:n]
        dd = s['deadline_dists'][:n]
        pi = s['part_ids'][:n]
        if len(et) < max_len:
            pad = max_len - len(et)
            et = F.pad(et, (0, pad), value=0)
            ti = F.pad(ti, (0, pad), value=0.0)
            dd = F.pad(dd, (0, pad), value=0.0)
            pi = F.pad(pi, (0, pad), value=0)
        batch_et.append(et[:max_len])
        batch_ti.append(ti[:max_len])
        batch_dd.append(dd[:max_len])
        batch_pi.append(pi[:max_len])
    return {
        'event_types': torch.stack(batch_et),
        'time_intervals': torch.stack(batch_ti),
        'deadline_dists': torch.stack(batch_dd),
        'part_ids': torch.stack(batch_pi),
    }


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items() if k != 'risk'}


# ─── 单折训练 ──────────────────────────────────────────────────

def train_one_fold(model, train_feat, train_seq, train_y,
                   val_feat, val_seq, val_y,
                   device, epochs=100, batch_size=32, patience=20, lr=5e-4,
                   label_smoothing=0.05):
    """
    Args:
        train_feat: (n_train, 46) np.array
        train_seq: list of sample dicts
        train_y: (n_train,) np.array
        val_feat, val_seq, val_y: 同理
    """
    model = model.to(device)

    # 标准化 (在训练集上 fit)
    scaler = StandardScaler()
    train_feat_s = scaler.fit_transform(train_feat).astype(np.float32)
    val_feat_s = scaler.transform(val_feat).astype(np.float32)

    train_feat_t = torch.FloatTensor(train_feat_s).to(device)
    val_feat_t = torch.FloatTensor(val_feat_s).to(device)
    train_y_t = torch.LongTensor(train_y).to(device)
    val_y_t = torch.LongTensor(val_y).to(device)

    train_seq_samples = list(train_seq)
    val_seq_samples = list(val_seq)

    train_loader = DataLoader(
        list(range(len(train_feat_t))), batch_size=batch_size,
        shuffle=True, drop_last=False)
    val_loader = DataLoader(
        list(range(len(val_feat_t))), batch_size=batch_size,
        shuffle=False, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_val_loss = float('inf')
    best_state = None
    patience_cnt = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx in train_loader:
            idx = batch_idx
            batch_feat = train_feat_t[idx]
            batch_y = train_y_t[idx]

            # 收集对应的序列样本
            batch_seq = [train_seq_samples[i] for i in idx.cpu().numpy()]
            batch_seq_dict = collate_for_dual_tower(batch_seq)
            batch_seq_dict = to_device(batch_seq_dict, device)

            optimizer.zero_grad()
            logits = model(batch_feat, batch_seq_dict)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = epoch_loss / max(n_batches, 1)

        # 验证
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_probs, all_labels = [], []

        with torch.no_grad():
            for batch_idx in val_loader:
                batch_feat = val_feat_t[batch_idx]
                batch_seq = [val_seq_samples[i] for i in batch_idx.cpu().numpy()]
                batch_seq_dict = collate_for_dual_tower(batch_seq)
                batch_seq_dict = to_device(batch_seq_dict, device)

                logits = model(batch_feat, batch_seq_dict)
                loss = criterion(logits, val_y_t[batch_idx])
                val_loss += loss.item()
                val_batches += 1

                probs = torch.softmax(logits, dim=-1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(val_y_t[batch_idx].cpu().numpy())

        avg_val = val_loss / max(val_batches, 1)
        elapsed = time.time() - t0

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"    Epoch {epoch:3d}  train={avg_train:.4f}  val={avg_val:.4f}  "
                      f"{elapsed:.1f}s  [早停 patience={patience}]")
                break

        print(f"    Epoch {epoch:3d}  train={avg_train:.4f}  val={avg_val:.4f}  "
              f"{elapsed:.1f}s  lr={scheduler.get_last_lr()[0]:.2e}")

    # 最优权重推理
    model.load_state_dict(best_state)
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch_idx in val_loader:
            batch_feat = val_feat_t[batch_idx]
            batch_seq = [val_seq_samples[i] for i in batch_idx.cpu().numpy()]
            batch_seq_dict = collate_for_dual_tower(batch_seq)
            batch_seq_dict = to_device(batch_seq_dict, device)
            logits = model(batch_feat, batch_seq_dict)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(val_y_t[batch_idx].cpu().numpy())

    return np.array(all_probs), np.array(all_labels)


# ─── 主函数 ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='outputs/dual_tower')
    parser.add_argument('--max-seq-len', type=int, default=500)
    parser.add_argument('--label-smoothing', type=float, default=0.05)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cpu')

    print("=" * 62)
    print("  BiLSTM + Mamba 三塔融合模型")
    print("  输入: 46维统计特征 + 7维事件序列")
    print("=" * 62)
    print(f"  设备={device}  折数={args.folds}  max_seq={args.max_seq_len}")
    print(f"  batch={args.batch_size}  lr={args.lr}  epochs={args.epochs}")
    print(f"  label_smoothing={args.label_smoothing}  patience={args.patience}")
    print("=" * 62)

    t0 = time.time()

    # ── 加载 46维统计特征 ──
    print("\n[数据加载] 46维统计特征 ...")
    ide_logs, passed = load_ide_logs()
    X, y, student_ids = build_feature_matrix(ide_logs, passed)
    print(f"  X={X.shape}  通过={sum(y==0)}  挂科={sum(y==1)}")

    # ── 加载 7维事件序列 ──
    print("[数据加载] 7维事件序列 ...")
    samples, _, labels_seq = preprocess_seq(ide_logs, passed, max_events=args.max_seq_len)
    print(f"  序列样本={len(samples)}  序列上限={args.max_seq_len}")

    print(f"  总加载耗时: {time.time()-t0:.1f}s")

    # ── 5折交叉验证 ──
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(np.array(student_ids), y), 1):

        print(f"\n{'='*50}")
        print(f"  Fold {fold_idx}/{args.folds}  train={len(train_idx)}  val={len(val_idx)}")
        print(f"{'='*50}")

        t_fold = time.time()
        train_feat, val_feat = X[train_idx], X[val_idx]
        train_y, val_y = y[train_idx], y[val_idx]
        train_seq = [samples[i] for i in train_idx]
        val_seq = [samples[i] for i in val_idx]

        model = DualTowerModel(max_seq_len=args.max_seq_len)
        print(f"  参数量: {model.count_parameters():,}  (样本数={len(train_idx)}, 比={len(train_idx)/model.count_parameters()*1000:.1f}‰)")

        val_probs, val_labels = train_one_fold(
            model, train_feat, train_seq, train_y,
            val_feat, val_seq, val_y,
            device, epochs=args.epochs, batch_size=args.batch_size,
            patience=args.patience, lr=args.lr,
            label_smoothing=args.label_smoothing)

        val_pred = (val_probs >= 0.5).astype(int)
        metrics = evaluate(val_labels, val_pred, val_probs)
        fold_results.append(metrics)
        print(f"  → Acc={metrics['accuracy']:.4f}  Prec={metrics['precision']:.4f}  "
              f"Recall={metrics['recall']:.4f}  F1={metrics['f1']:.4f}  "
              f"AUC={metrics['auc']:.4f}  ({time.time()-t_fold:.0f}s)")

    # ── 汇总 ──
    summary = summarize_fold_results(fold_results)
    print_results_table("BiLSTM+Mamba TripleTower (46+7)", summary)

    # ── 保存 ──
    out_dir = os.path.join(_PROJECT_ROOT, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    results = {
        'model': 'BiLSTM+Mamba TripleTower',
        'inputs': ['46-dim statistical features', '7-dim event sequences'],
        'config': {
            'bilstm_hidden': 64, 'bilstm_layers': 2,
            'mamba_d_model': 48, 'mamba_layers': 2, 'mamba_d_state': 8,
            'feat_hidden': 64, 'max_seq_len': args.max_seq_len,
            'dropout': 0.3, 'label_smoothing': args.label_smoothing,
        },
        'training': {
            'n_folds': args.folds, 'epochs': args.epochs,
            'batch_size': args.batch_size, 'patience': args.patience, 'lr': args.lr,
        },
        'cv_results': {k: float(v) for k, v in summary.items()},
        'fold_details': fold_results,
        'n_samples': len(samples),
    }
    with open(os.path.join(out_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {out_dir}/results.json")
    print(f"总耗时: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
