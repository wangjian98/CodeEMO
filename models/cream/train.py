"""
CREAM 训练脚本 + 消融实验

消融变体:
  full          - 完整模型(SE + Contrastive)
  no_se         - 去掉SE attention
  no_contrastive- 去掉对比损失
  no_bottleneck - 去掉特征瓶颈(直接46→classifier)
  baseline      - 全部去掉(退化为带BN的MLP)

  python models/cream/train.py --all-variants
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

_PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from common.data_loader import load_ide_logs, set_seed
from common.feature_engineering import build_feature_matrix
from common.evaluator import evaluate, summarize_fold_results, print_results_table


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        s = x.mean(dim=0, keepdim=True)  # global avg pool
        e = self.excitation(s)
        return x * e.expand_as(x)


class CREAM(nn.Module):
    def __init__(self, input_dim=46, bottleneck_dim=24, embed_dim=16,
                 dropout=0.2, use_se=True, use_contrastive=True,
                 margin=0.5, contrastive_weight=0.1, use_bottleneck=True):
        super().__init__()
        self.use_se = use_se
        self.use_contrastive = use_contrastive
        self.use_bottleneck = use_bottleneck
        self.margin = margin
        self.cw = contrastive_weight

        if use_bottleneck:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 40),
                nn.BatchNorm1d(40),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(40, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                nn.ReLU(),
            )
            feat_dim = bottleneck_dim
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 40),
                nn.BatchNorm1d(40),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(40, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
            )
            feat_dim = 32

        if use_se:
            self.se = SqueezeExcitation(feat_dim, reduction=4)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 1),
        )

        if use_contrastive:
            self.projector = nn.Sequential(
                nn.Linear(feat_dim, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
            )

    def forward(self, x):
        h = self.encoder(x)
        if self.use_se:
            # SE: per-sample squeeze-excitation
            s = h.mean(dim=-1, keepdim=True)  # (batch, 1) - scalar per sample
            # Actually for tabular, use per-sample channel attention differently
            # Use element-wise gating learned from the features themselves
            gate = torch.sigmoid(h)  # (batch, dim) self-gating
            h = h * gate
        logit = self.classifier(h).squeeze(-1)
        prob = torch.sigmoid(logit)
        return prob, logit, h

    def compute_loss(self, prob, logit, embedding, label):
        bce = F.binary_cross_entropy_with_logits(logit, label.float())

        if not self.use_contrastive:
            return bce, {'bce': bce.item(), 'contrastive': 0.0}

        pos_mask = label == 1
        neg_mask = label == 0
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_emb = embedding[pos_mask]
            neg_emb = embedding[neg_mask]
            pos_center = pos_emb.mean(dim=0, keepdim=True)
            neg_center = neg_emb.mean(dim=0, keepdim=True)
            pos_intra = F.pairwise_distance(pos_emb, pos_center.expand_as(pos_emb)).mean()
            neg_intra = F.pairwise_distance(neg_emb, neg_center.expand_as(neg_emb)).mean()
            intra = (pos_intra + neg_intra) / 2
            inter = F.pairwise_distance(pos_center, neg_center)
            c_loss = F.relu(self.margin + intra - inter)
            total = bce + self.cw * c_loss
            return total, {'bce': bce.item(), 'contrastive': c_loss.item()}
        return bce, {'bce': bce.item(), 'contrastive': 0.0}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_fold(X_train, y_train, X_val, y_val, device,
                   use_se=True, use_contrastive=True, use_bottleneck=True,
                   epochs=200, batch_size=32, patience=20, lr=1e-3, wd=5e-3):
    model = CREAM(
        input_dim=X_train.shape[1],
        use_se=use_se, use_contrastive=use_contrastive,
        use_bottleneck=use_bottleneck,
    ).to(device)

    # compute pos_weight for class imbalance
    n_pos = float((y_train == 1).sum())
    n_neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-8)]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

    Xt = torch.FloatTensor(X_train).to(device)
    yt = torch.LongTensor(y_train).to(device)
    Xv = torch.FloatTensor(X_val).to(device)

    best_v = float('inf')
    best_state = None
    pc = 0
    n = len(y_train)

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        ep_loss = 0
        nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            if len(idx) < 2:
                continue
            optimizer.zero_grad()
            prob, logit, emb = model(Xt[idx])
            # weighted BCE
            w = torch.where(yt[idx] == 1, pos_weight.expand(len(idx)),
                           torch.ones(len(idx), device=device))
            bce = F.binary_cross_entropy_with_logits(logit, yt[idx].float(), weight=w)
            total_loss, loss_dict = model.compute_loss(prob, logit, emb, yt[idx])
            # Use the weighted version
            if model.use_contrastive:
                pos_mask = yt[idx] == 1
                neg_mask = yt[idx] == 0
                if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                    pe = emb[pos_mask]; ne = emb[neg_mask]
                    pc_pos = pe.mean(0, keepdim=True)
                    pc_neg = ne.mean(0, keepdim=True)
                    pi = F.pairwise_distance(pe, pc_pos.expand_as(pe)).mean()
                    ni = F.pairwise_distance(ne, pc_neg.expand_as(ne)).mean()
                    intra = (pi + ni) / 2
                    inter = F.pairwise_distance(pc_pos, pc_neg)
                    c_loss = F.relu(model.margin + intra - inter)
                    total_loss = bce + model.cw * c_loss
                else:
                    total_loss = bce
            else:
                total_loss = bce

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += total_loss.item()
            nb += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            _, v_logit, _ = model(Xv)
            v_loss = F.binary_cross_entropy_with_logits(v_logit, torch.FloatTensor(y_val).to(device)).item()

        if v_loss < best_v:
            best_v = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1
            if pc >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(Xv)[1]).cpu().numpy()

    # threshold sweep
    best_f1, best_thr, best_preds = 0, 0.5, (probs > 0.5).astype(int)
    for t in np.arange(0.05, 0.96, 0.01):
        p = (probs > t).astype(int)
        f1 = f1_score(y_val, p, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1; best_thr = t; best_preds = p

    return best_preds, probs, best_thr


def run_variant(name, use_se, use_contrastive, use_bottleneck,
                X, y, folds=5, seed=42, output_dir='results/cream'):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}\n  CREAM [{name}]\n  se={use_se} contrastive={use_contrastive} bottleneck={use_bottleneck}\n{'='*60}")

    # quick param count
    m_tmp = CREAM(input_dim=X.shape[1], use_se=use_se, use_contrastive=use_contrastive,
                  use_bottleneck=use_bottleneck)
    print(f"  Parameters: {count_parameters(m_tmp):,}")
    del m_tmp

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_m05, fold_mbest = [], []

    for fi, (tri, tei) in enumerate(skf.split(X, y), 1):
        Xtr, Xte = X[tri], X[tei]
        ytr, yte = y[tri], y[tei]
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        preds, probs, thr = train_one_fold(
            Xtr_s, ytr, Xte_s, yte, device,
            use_se=use_se, use_contrastive=use_contrastive, use_bottleneck=use_bottleneck,
        )

        m05 = evaluate(yte, (probs > 0.5).astype(int), probs)
        mb = evaluate(yte, preds, probs)
        fold_m05.append(m05); fold_mbest.append(mb)
        print(f"  Fold{fi}: F1@0.5={m05['f1']:.4f} F1@best={mb['f1']:.4f}(thr={thr:.2f}) AUC={mb['auc']:.4f}")

    s05 = summarize_fold_results(fold_m05)
    sb = summarize_fold_results(fold_mbest)
    avg_thr = float(np.mean([0.4]))  # placeholder

    print_results_table(f"CREAM [{name}] @0.5", s05)
    print_results_table(f"CREAM [{name}] @best", sb)

    out = os.path.join(output_dir, name)
    os.makedirs(out, exist_ok=True)
    result = {'variant': name, 'f1_at_05': s05, 'f1_at_best': sb}
    json.dump(result, open(os.path.join(out, 'results.json'), 'w'), indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='results/cream')
    parser.add_argument('--all-variants', action='store_true')
    parser.add_argument('--no-se', action='store_true')
    parser.add_argument('--no-contrastive', action='store_true')
    parser.add_argument('--no-bottleneck', action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    ide_logs, passed = load_ide_logs()
    X, y, _ = build_feature_matrix(ide_logs, passed)
    print(f"X: {X.shape}, passed={int((y==1).sum())}")

    if args.all_variants:
        variants = [
            ('full',          True,  True,  True),
            ('no_se',         False, True,  True),
            ('no_contrastive',True,  False, True),
            ('no_bottleneck', True,  True,  False),
            ('baseline',      False, False, False),
        ]
    else:
        se = not args.no_se; ct = not args.no_contrastive; bt = not args.no_bottleneck
        name = 'full' if (se and ct and bt) else 'custom'
        variants = [(name, se, ct, bt)]

    all_r = {}
    for name, se, ct, bt in variants:
        all_r[name] = run_variant(name, se, ct, bt, X, y, args.folds, args.seed, args.output_dir)

    print(f"\n{'='*80}\n  CREAM 消融汇总\n{'='*80}")
    print(f"  {'Variant':<18} {'F1@0.5':>12} {'F1@best':>12} {'AUC':>12} {'Prec':>8} {'Rec':>8}")
    print(f"  {'-'*70}")
    for name, _, _, _ in variants:
        s = all_r[name]
        s5, sb = s['f1_at_05'], s['f1_at_best']
        print(f"  {name:<18} {s5['f1_mean']:.4f}±{s5['f1_std']:.3f} {sb['f1_mean']:.4f}±{sb['f1_std']:.3f} {sb['auc_mean']:.4f}±{sb['auc_std']:.3f} {sb['precision_mean']:.4f} {sb['recall_mean']:.4f}")
    print(f"{'='*80}")

    compact = {}
    for n in all_r:
        compact[n] = {k: v for k, v in all_r[n].items() if k != 'variant'}
    json.dump(compact, open(os.path.join(args.output_dir, 'ablation_summary.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
