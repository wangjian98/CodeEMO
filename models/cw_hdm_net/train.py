#!/usr/bin/env python3
"""
CW-HDM-Net: Class-Weighted HDM-Net with Stochastic View Dropout
================================================================
基于 HDM-Net v2, 加 4 个针对性改动:

1. Class Weight (passed=2.0, failed=1.0) - 解决不平衡
2. Soft Head only focal loss (避免 hard head 过度预测 failed)
3. Stochastic View Dropout (每 batch 随机丢 1 个视图, 减少过拟合)
4. InfoNCE 权重降到 0.05 (辅助, 不喧宾夺主)

保留 HDM-Net v2 的 PIG, 保留 M-AAE-Net 验证有效的 KD 和 Dual Head.
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _PROJECT_ROOT)
from common.data_loader import load_ide_logs, set_seed
from common.feature_engineering import build_feature_matrix, EVENT_TYPES
from models.hdm_net.model import TreeHead, SeqBranch, AttnBranch


# ============================================================
# CW-HDM-Net 模型
# ============================================================
class CWHDMNet(nn.Module):
    """HDM-Net v2 + class weight + stochastic view dropout + dual head"""
    def __init__(self, d=32, tree_depth=3, tree_width=64, dropout=0.3):
        super().__init__()
        self.tree_branch = TreeHead(in_dim=11, d=d, depth=tree_depth,
                                     dropout=dropout, use_skip=True, use_bn=True)
        self.seq_branch = SeqBranch(in_dim_per_step=1, d=d, num_layers=1, dropout=0.1)
        self.att_branch = AttnBranch(d=d, nhead=4, dropout=0.1)

        # PIG (per-instance gating)
        self.pig = nn.Sequential(
            nn.Linear(3 * d, d), nn.ReLU(),
            nn.Linear(d, 3)
        )

        # 96-d 嵌入
        self.embed = nn.Linear(d * 3, 96)

        # Dual Head: Hard (CE) + Soft (focal)
        self.hard_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d // 2, 1)
        )
        self.soft_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d // 2),
            nn.Linear(d // 2, 1)
        )

    def forward(self, x_tree, tree_probs, x_seq, x_att, view_drop=None):
        """
        view_drop: 可选, [tree_drop, seq_drop, att_drop] (0/1) 强制丢视图
        """
        # 每个视图可被强制置零 (stochastic view dropout)
        if view_drop is None:
            view_drop = [0, 0, 0]

        h_tree = self.tree_branch(x_tree, tree_probs)
        h_seq = self.seq_branch(x_seq)
        h_att = self.att_branch(x_att)

        if view_drop[0]:
            h_tree = h_tree * 0
        if view_drop[1]:
            h_seq = h_seq * 0
        if view_drop[2]:
            h_att = h_att * 0

        h_concat = torch.cat([h_tree, h_seq, h_att], dim=-1)

        # PIG
        g = torch.softmax(self.pig(h_concat), dim=-1)
        h_fused = g[:, 0:1] * h_tree + g[:, 1:2] * h_seq + g[:, 2:3] * h_att

        embedding = self.embed(h_concat)

        hard_logit = self.hard_head(h_fused).squeeze(-1)
        soft_logit = self.soft_head(h_fused).squeeze(-1)

        return {
            'hard_logit': hard_logit,
            'soft_logit': soft_logit,
            'embedding': embedding,
            'h_fused': h_fused,
        }


# ============================================================
# Loss functions
# ============================================================
def infonce_loss(embeddings, labels, temperature=0.1):
    embeddings = F.normalize(embeddings, dim=-1)
    sim = torch.matmul(embeddings, embeddings.T) / temperature
    labels_v = labels.contiguous().view(-1, 1)
    mask_pos = torch.eq(labels_v, labels_v.T).float().to(embeddings.device)
    mask_neg = 1 - mask_pos
    mask_self = torch.eye(embeddings.size(0)).to(embeddings.device)
    mask_pos = mask_pos - mask_self
    sim_max = sim.max(dim=1, keepdim=True).values.detach()
    sim = sim - sim_max
    exp_sim = torch.exp(sim)
    pos = (exp_sim * mask_pos).sum(dim=1)
    neg = (exp_sim * (mask_neg + mask_self)).sum(dim=1)
    valid = (mask_pos.sum(dim=1) > 0).float()
    loss = -torch.log(pos / (neg + 1e-9) + 1e-9)
    return (loss * valid).sum() / (valid.sum() + 1e-9)


def kd_loss(student_logits, teacher_probs, T=3.0):
    s2 = torch.stack([-student_logits, student_logits], dim=-1)
    t2 = torch.stack([1 - teacher_probs, teacher_probs], dim=-1)
    p_s = F.log_softmax(s2 / T, dim=-1)
    p_t = F.softmax(t2 / T, dim=-1)
    return F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)


def focal_loss_with_weight(logits, labels, alpha=0.7, gamma=2.0, smoothing=0.1, class_weight=None):
    """Focal Loss with class weight"""
    n_classes = 2
    smooth_labels = torch.full((logits.size(0), n_classes),
                                     smoothing / (n_classes - 1),
                                     device=logits.device)
    smooth_labels.scatter_(1, labels.unsqueeze(1), 1 - smoothing)

    p1 = torch.sigmoid(logits)
    p = torch.stack([1 - p1, p1], dim=-1)

    pt = (p * smooth_labels).sum(dim=-1)
    focal_weight = (1 - pt) ** gamma
    # class weight: passed=2x, failed=1x
    if class_weight is not None:
        cw = torch.where(labels == 0, class_weight[0], class_weight[1]).float()
    else:
        cw = torch.where(labels == 1, alpha, 1 - alpha).float()

    target_oh = torch.zeros_like(p)
    target_oh.scatter_(1, labels.unsqueeze(1), 1.0)
    bce = F.binary_cross_entropy_with_logits(
        torch.stack([-logits, logits], dim=-1),
        target_oh,
        reduction='none'
    ).sum(dim=-1)
    return (cw * focal_weight * bce).mean()


# ============================================================
# 数据准备
# ============================================================
def prepare_data(n=473):
    set_seed(42)
    ide_logs, passed = load_ide_logs()
    X_46d, y_passed, student_ids = build_feature_matrix(ide_logs, passed)
    y = 1 - y_passed

    df = ide_logs.groupby(['student', 'eventType']).size().unstack(fill_value=0)
    df = df.reindex(columns=EVENT_TYPES, fill_value=0)
    df = df.reindex(index=student_ids, fill_value=0)
    x_tree_7d = df.values.astype(np.float32)

    x_tree_11d = np.column_stack([
        x_tree_7d,
        x_tree_7d.sum(axis=1, keepdims=True) / (x_tree_7d.sum(axis=1, keepdims=True).max() + 1),
        np.std(x_tree_7d, axis=1, keepdims=True),
    ])

    x_seq = X_46d.reshape(-1, 46, 1).astype(np.float32)
    x_att = x_tree_7d.reshape(-1, 7, 1).astype(np.float32)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_probs = np.zeros(n, dtype=np.float32)
    for tr, va in skf.split(x_tree_11d, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(x_tree_11d[tr])
        Xva = scaler.transform(x_tree_11d[va])
        rf.fit(Xtr, y[tr])
        rf_probs[va] = rf.predict_proba(Xva)[:, 1]
    tree_probs = np.column_stack([1 - rf_probs, rf_probs]).astype(np.float32)

    return x_tree_11d, tree_probs, x_seq, x_att, y


def train_one_fold(x_tree_tr, tree_probs_tr, x_seq_tr, x_att_tr, y_tr,
                    x_tree_va, tree_probs_va, x_seq_va, x_att_va, y_va,
                    epochs=80, batch_size=32, patience=15, device='cuda'):
    model = CWHDMNet(d=32, tree_depth=3, tree_width=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))  # passed=2x weight

    n_tr = len(y_tr)
    best_v = float('inf')
    best_state = None
    pc = 0

    for ep in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(n_tr)
        total_loss = 0
        for i in range(0, n_tr, batch_size):
            idx = perm[i:i + batch_size]
            xt = torch.FloatTensor(x_tree_tr[idx]).to(device)
            tp = torch.FloatTensor(tree_probs_tr[idx]).to(device)
            xs = torch.FloatTensor(x_seq_tr[idx]).to(device)
            xa = torch.FloatTensor(x_att_tr[idx]).to(device)
            yt = torch.FloatTensor(y_tr[idx]).to(device)
            lt = torch.LongTensor(y_tr[idx]).to(device)

            # 创新 3: Stochastic view dropout - 每个 batch 随机丢一个
            view_to_drop = np.random.randint(3)
            view_drop = [0, 0, 0]
            view_drop[view_to_drop] = 1

            out = model(xt, tp, xs, xa, view_drop=view_drop)
            hard_logit = out['hard_logit']
            soft_logit = out['soft_logit']
            embed = out['embedding']

            # 创新 1: Class weight (passed=2x in hard head BCE)
            loss_hard = bce(hard_logit, yt)
            # 创新 2: Focal only on soft head
            loss_soft = focal_loss_with_weight(soft_logit, lt, alpha=0.7, gamma=2.0,
                                                 smoothing=0.1,
                                                 class_weight=torch.tensor([2.0, 1.0]).to(device))
            # 创新 4: InfoNCE 降到 0.05
            loss_info = infonce_loss(embed, lt, temperature=0.1)
            # KD (保留)
            loss_kd = kd_loss(hard_logit, tp[:, 1], T=3.0)

            # 权重: 0.5 hard + 0.3 soft + 0.05 InfoNCE + 0.15 KD
            loss = 0.5 * loss_hard + 0.3 * loss_soft + 0.05 * loss_info + 0.15 * loss_kd

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(idx)
        scheduler.step()
        total_loss /= n_tr

        model.eval()
        with torch.no_grad():
            xt = torch.FloatTensor(x_tree_va).to(device)
            tp = torch.FloatTensor(tree_probs_va).to(device)
            xs = torch.FloatTensor(x_seq_va).to(device)
            xa = torch.FloatTensor(x_att_va).to(device)
            yt = torch.FloatTensor(y_va).to(device)
            out = model(xt, tp, xs, xa)
            v_loss = bce(out['hard_logit'], yt).item()

        if v_loss < best_v:
            best_v = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1
            if pc >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_probs(model, x_tree, tree_probs, x_seq, x_att, device='cuda', batch_size=64):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(x_tree), batch_size):
            xt = torch.FloatTensor(x_tree[i:i+batch_size]).to(device)
            tp = torch.FloatTensor(tree_probs[i:i+batch_size]).to(device)
            xs = torch.FloatTensor(x_seq[i:i+batch_size]).to(device)
            xa = torch.FloatTensor(x_att[i:i+batch_size]).to(device)
            out = model(xt, tp, xs, xa)
            p_hard = torch.sigmoid(out['hard_logit']).cpu().numpy()
            p_soft = torch.sigmoid(out['soft_logit']).cpu().numpy()
            probs.append(0.5 * p_hard + 0.5 * p_soft)
    return np.concatenate(probs)


def evaluate(probs, y, threshold=0.5):
    yh = (probs > threshold).astype(int)
    return {
        'accuracy': float(accuracy_score(y, yh)),
        'precision': float(precision_score(y, yh, zero_division=0)),
        'recall': float(recall_score(y, yh, zero_division=0)),
        'f1': float(f1_score(y, yh, zero_division=0)),
        'auc': float(roc_auc_score(y, probs)),
        'failed_F1': float(f1_score(y, yh, pos_label=1, zero_division=0)),
        'passed_F1': float(f1_score(y, yh, pos_label=0, zero_division=0)),
        'failed_P': float(precision_score(y, yh, pos_label=1, zero_division=0)),
        'failed_R': float(recall_score(y, yh, pos_label=1, zero_division=0)),
        'passed_P': float(precision_score(y, yh, pos_label=0, zero_division=0)),
        'passed_R': float(recall_score(y, yh, pos_label=0, zero_division=0)),
        'cm': confusion_matrix(y, yh).flatten().tolist(),
    }


def main(output_dir='outputs/unified_compare/cw_hdm_net', epochs=80, batch_size=32):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}", flush=True)
    os.makedirs(output_dir, exist_ok=True)

    x_tree, tree_probs, x_seq, x_att, y = prepare_data()
    n = len(y)
    print(f"数据: n={n}, fail_rate={y.mean():.4f}", flush=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_probs = np.zeros(n)
    fold_metrics = []
    start = time.time()

    for fi, (tr, va) in enumerate(skf.split(x_tree, y)):
        print(f"\n--- Fold {fi+1}/5 ---", flush=True)
        model = train_one_fold(
            x_tree[tr], tree_probs[tr], x_seq[tr], x_att[tr], y[tr],
            x_tree[va], tree_probs[va], x_seq[va], x_att[va], y[va],
            epochs=epochs, batch_size=batch_size, device=device
        )
        probs_va = predict_probs(model, x_tree[va], tree_probs[va],
                                  x_seq[va], x_att[va], device=device)
        all_probs[va] = probs_va
        m = evaluate(probs_va, y[va])
        print(f"  F1={m['f1']:.4f}  AUC={m['auc']:.4f}  acc={m['accuracy']:.4f}  failed_F1={m['failed_F1']:.4f}  passed_F1={m['passed_F1']:.4f}", flush=True)
        fold_metrics.append(m)

    elapsed = time.time() - start
    print(f"\n总耗时: {elapsed:.1f}s", flush=True)

    cv = {}
    for k in ['accuracy', 'precision', 'recall', 'f1', 'auc',
              'failed_F1', 'passed_F1', 'failed_P', 'failed_R', 'passed_P', 'passed_R']:
        cv[f"{k}_mean"] = float(np.mean([m[k] for m in fold_metrics]))
        cv[f"{k}_std"] = float(np.std([m[k] for m in fold_metrics]))

    np.save(os.path.join(output_dir, 'probs.npy'), all_probs)
    np.save(os.path.join(output_dir, 'labels.npy'), y)
    report = {
        'model': 'CW-HDM-Net',
        'n_samples': n,
        'n_failed': int(y.sum()),
        'n_passed': int((1-y).sum()),
        'n_params': sum(p.numel() for p in CWHDMNet().parameters()),
        'elapsed_seconds': elapsed,
        'architecture': {
            'base': 'HDM-Net v2 (3 views + PIG)',
            'innovation_1': 'Class Weight (passed=2x, failed=1x) - 解决不平衡',
            'innovation_2': 'Focal Loss only on Soft Head - 避免过度预测 failed',
            'innovation_3': 'Stochastic View Dropout (每 batch 丢 1 视图)',
            'innovation_4': 'InfoNCE 权重降到 0.05 (辅助)',
            'kept_1': 'Dual Head (验证有效, M-AAE-Net AUC +1pp)',
            'kept_2': 'KD from RF_7dim (验证有效)',
        },
        'cv_results': cv,
        'fold_details': fold_metrics,
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*72}", flush=True)
    print(f"CW-HDM-Net CV 结果:", flush=True)
    for k in ['failed_F1', 'passed_F1', 'accuracy', 'auc', 'f1']:
        print(f"  {k}: {cv[f'{k}_mean']:.4f} ± {cv[f'{k}_std']:.4f}", flush=True)
    print(f"  n_params: {report['n_params']:,}", flush=True)
    print(f"  输出: {output_dir}/", flush=True)
    return cv


if __name__ == '__main__':
    main()