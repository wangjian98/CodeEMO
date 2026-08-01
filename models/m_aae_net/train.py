#!/usr/bin/env python3
"""
M-AAE-Net: Multi-view Adaptive Asymmetric Ensemble Network
============================================================
基于 HDM-Net v2 的演进, 加 4 个创新点:

1. Class-aware Gating (双门控)
2. Dual Head (双 head)
3. InfoNCE Contrastive Loss (对比损失)
4. Knowledge Distillation from RF_7dim
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
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
from common.feature_engineering import build_feature_matrix
from models.hdm_net.model import TreeHead, SeqBranch, AttnBranch


# ============================================================
# 创新 1: Class-aware Gating (双门控)
# ============================================================
class ClassAwareGating(nn.Module):
    """两个独立的门控: failed 路由 + passed 路由"""
    def __init__(self, in_dim=192, hidden_dim=64):
        super().__init__()
        self.gate_failed = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.gate_passed = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, h_tree, h_seq, h_att):
        h_concat = torch.cat([h_tree, h_seq, h_att], dim=-1)
        feat = torch.cat([h_tree, h_seq, h_att, h_concat], dim=-1)
        g_f = self.gate_failed(feat)
        g_p = self.gate_passed(feat)
        return g_f.squeeze(-1), g_p.squeeze(-1)


# ============================================================
# 创新 2: Dual Head (双 head)
# ============================================================
class DualHead(nn.Module):
    def __init__(self, feat_dim=96, dropout=0.3):
        super().__init__()
        self.hard_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, 1)
        )
        self.soft_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(feat_dim // 2),
            nn.Linear(feat_dim // 2, 1)
        )

    def forward(self, x):
        return self.hard_head(x).squeeze(-1), self.soft_head(x).squeeze(-1)


# ============================================================
# 完整模型: M-AAE-Net
# ============================================================
class MAAENet(nn.Module):
    def __init__(self, d=32, tree_depth=3, tree_width=64, dropout=0.3):
        super().__init__()
        self.tree_branch = TreeHead(in_dim=12, d=d, depth=tree_depth,
                                     dropout=dropout, use_skip=True, use_bn=True)
        self.seq_branch = SeqBranch(in_dim_per_step=1, d=d, num_layers=1, dropout=0.1)
        self.att_branch = AttnBranch(d=d, nhead=4, dropout=0.1)

        self.embed = nn.Linear(d * 3, 96)
        self.gating = ClassAwareGating(in_dim=192, hidden_dim=64)
        self.dual_head = DualHead(feat_dim=96, dropout=dropout)

    def forward(self, x_tree, tree_probs, x_seq, x_att):
        h_tree = self.tree_branch(x_tree, tree_probs)
        h_seq = self.seq_branch(x_seq)
        h_att = self.att_branch(x_att)

        h_concat = torch.cat([h_tree, h_seq, h_att], dim=-1)
        embedding = self.embed(h_concat)

        g_f, g_p = self.gating(h_tree, h_seq, h_att)
        hard_logit, soft_logit = self.dual_head(embedding)

        return {
            'hard_logit': hard_logit,
            'soft_logit': soft_logit,
            'embedding': embedding,
            'gate_failed': g_f,
            'gate_passed': g_p,
        }


# ============================================================
# 创新 3: InfoNCE 对比损失
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


# ============================================================
# 创新 4: KL 蒸馏损失
# ============================================================
def kd_loss(student_logits, teacher_probs, T=3.0):
    """student_logits: (B,); teacher_probs: (B,) P(failed=1)"""
    s2 = torch.stack([-student_logits, student_logits], dim=-1)
    t2 = torch.stack([1 - teacher_probs, teacher_probs], dim=-1)
    p_s = F.log_softmax(s2 / T, dim=-1)
    p_t = F.softmax(t2 / T, dim=-1)
    return F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)


# ============================================================
# Focal Loss + Label Smoothing
# ============================================================
def focal_loss(logits, labels, alpha=0.7, gamma=2.0, smoothing=0.1):
    n_classes = 2
    smooth_labels = torch.full((logits.size(0), n_classes),
                                     smoothing / (n_classes - 1),
                                     device=logits.device)
    smooth_labels.scatter_(1, labels.unsqueeze(1), 1 - smoothing)

    p1 = torch.sigmoid(logits)
    p = torch.stack([1 - p1, p1], dim=-1)

    pt = (p * smooth_labels).sum(dim=-1)
    focal_weight = (1 - pt) ** gamma
    alpha_w = torch.where(labels == 1, alpha, 1 - alpha).float()

    target_oh = torch.zeros_like(p)
    target_oh.scatter_(1, labels.unsqueeze(1), 1.0)
    bce = F.binary_cross_entropy_with_logits(
        torch.stack([-logits, logits], dim=-1),
        target_oh,
        reduction='none'
    ).sum(dim=-1)
    return (alpha_w * focal_weight * bce).mean()


# ============================================================
# 数据准备
# ============================================================
def prepare_data(n=473):
    set_seed(42)
    ide_logs, passed = load_ide_logs()
    X_46d, y_passed, student_ids = build_feature_matrix(ide_logs, passed)
    y = 1 - y_passed

    from common.feature_engineering import EVENT_TYPES
    df = ide_logs.groupby(['student', 'eventType']).size().unstack(fill_value=0)
    df = df.reindex(columns=EVENT_TYPES, fill_value=0)
    df = df.reindex(index=student_ids, fill_value=0)
    x_tree_7d = df.values.astype(np.float32)

    x_tree_11d = np.column_stack([
        x_tree_7d,
        x_tree_7d.sum(axis=1, keepdims=True) / (x_tree_7d.sum(axis=1, keepdims=True).max() + 1),
        np.std(x_tree_7d, axis=1, keepdims=True),
        np.std(x_tree_7d, axis=1, keepdims=True) / (x_tree_7d.mean(axis=1, keepdims=True) + 1e-9),
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


# ============================================================
# 5-fold CV 训练
# ============================================================
def train_one_fold(x_tree_tr, tree_probs_tr, x_seq_tr, x_att_tr, y_tr,
                    x_tree_va, tree_probs_va, x_seq_va, x_att_va, y_va,
                    epochs=80, batch_size=32, patience=10, device='cuda'):
    model = MAAENet(d=32, tree_depth=3, tree_width=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    bce_hard = nn.BCEWithLogitsLoss()

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

            out = model(xt, tp, xs, xa)
            hard_logit = out['hard_logit']
            soft_logit = out['soft_logit']
            embed = out['embedding']

            loss_hard = bce_hard(hard_logit, yt)
            loss_soft = focal_loss(soft_logit, lt, alpha=0.7, gamma=2.0, smoothing=0.1)
            loss_info = infonce_loss(embed, lt, temperature=0.1)
            loss_kd = kd_loss(hard_logit, tp[:, 1], T=3.0)

            loss = 0.4 * loss_hard + 0.3 * loss_soft + 0.2 * loss_info + 0.1 * loss_kd

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
            v_loss = bce_hard(out['hard_logit'], yt).item()

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
            probs.append(0.4 * p_hard + 0.6 * p_soft)
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


def main(output_dir='outputs/unified_compare/m_aae_net', epochs=80, batch_size=32):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    os.makedirs(output_dir, exist_ok=True)

    x_tree, tree_probs, x_seq, x_att, y = prepare_data()
    n = len(y)
    print(f"数据: n={n}, fail_rate={y.mean():.4f}")

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
        print(f"  F1={m['f1']:.4f}  AUC={m['auc']:.4f}  acc={m['accuracy']:.4f}", flush=True)
        fold_metrics.append(m)

    elapsed = time.time() - start
    print(f"\n总耗时: {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

    cv = {}
    for k in ['accuracy', 'precision', 'recall', 'f1', 'auc',
              'failed_F1', 'passed_F1', 'failed_P', 'failed_R', 'passed_P', 'passed_R']:
        cv[f"{k}_mean"] = float(np.mean([m[k] for m in fold_metrics]))
        cv[f"{k}_std"] = float(np.std([m[k] for m in fold_metrics]))

    np.save(os.path.join(output_dir, 'probs.npy'), all_probs)
    np.save(os.path.join(output_dir, 'labels.npy'), y)
    report = {
        'model': 'M-AAE-Net',
        'n_samples': n,
        'n_failed': int(y.sum()),
        'n_passed': int((1-y).sum()),
        'n_params': sum(p.numel() for p in MAAENet().parameters()),
        'elapsed_seconds': elapsed,
        'architecture': {
            'tree_branch': 'TreeHead(7d + RF probs + 2 handcraft, depth=3, width=64, skip+LN)',
            'seq_branch': 'BiLSTM(46x1)',
            'att_branch': 'Pre-norm Transformer(7x1, 4 heads)',
            'innovation_1': 'Class-aware Gating (failed/passed dual route)',
            'innovation_2': 'Dual Head (Hard CE + Soft Focal+LS, 0.4:0.6)',
            'innovation_3': 'InfoNCE Loss (96-d embedding, temp=0.1, weight=0.2)',
            'innovation_4': 'KD from RF_7dim (T=3.0, KL weight=0.1)',
        },
        'cv_results': cv,
        'fold_details': fold_metrics,
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*72}", flush=True)
    print(f"M-AAE-Net CV 结果 (5-fold):", flush=True)
    print(f"  failed_F1: {cv['failed_F1_mean']:.4f} ± {cv['failed_F1_std']:.4f}", flush=True)
    print(f"  passed_F1: {cv['passed_F1_mean']:.4f} ± {cv['passed_F1_std']:.4f}", flush=True)
    print(f"  accuracy:  {cv['accuracy_mean']:.4f} ± {cv['accuracy_std']:.4f}", flush=True)
    print(f"  AUC:       {cv['auc_mean']:.4f} ± {cv['auc_std']:.4f}", flush=True)
    print(f"  F1:        {cv['f1_mean']:.4f} ± {cv['f1_std']:.4f}", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"  n_params: {report['n_params']:,}", flush=True)
    print(f"  输出: {output_dir}/", flush=True)
    return cv


if __name__ == '__main__':
    main()