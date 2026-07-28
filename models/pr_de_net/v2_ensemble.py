"""
PR-DE-Net v2: 基于 full (164K) 调优 + 多模型 ensemble

候选:
  v2_a: alpha=1.0 beta=1.0 gamma=3.0  (强化融合 loss)
  v2_b: alpha=1.5 beta=1.0 gamma=2.0  (强化 A 分支 Recall)
  v2_c: alpha=1.0 beta=1.5 gamma=2.0  (强化 B 分支 Precision)
  v2_d: alpha=0.5 beta=0.5 gamma=2.0  (弱化独立 loss)
然后 ensemble: average(v2_a, v2_b, v2_c, v2_d)
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, get_device, set_seed
from common.feature_engineering import build_feature_matrix
from common.evaluator import evaluate, summarize_fold_results, print_results_table
from models.pr_de_net.model import PRDENet


def train_one(Xtr, ytr, Xva, yva, device, alpha, beta, gamma,
              epochs=200, batch_size=32, patience=20, lr=1e-3, wd=1e-3):
    model = PRDENet(dropout=0.3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt = torch.FloatTensor(Xtr).to(device)
    yt = torch.FloatTensor(ytr).to(device)
    Xv = torch.FloatTensor(Xva).to(device)
    best_f1, best_state, best_m, pc = -1, None, None, 0
    n = Xt.shape[0]
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            out = model(Xt[idx])
            p_A = out['p_A'].squeeze(-1).clamp(1e-7, 1 - 1e-7)
            p_B = out['p_B'].squeeze(-1).clamp(1e-7, 1 - 1e-7)
            p_f = out['p_final'].squeeze(-1).clamp(1e-7, 1 - 1e-7)
            y = yt[idx]
            L_A = F.binary_cross_entropy(p_A, y)
            L_B = F.binary_cross_entropy(1 - p_B, 1 - y)
            L_f = F.binary_cross_entropy(p_f, y)
            loss = alpha * L_A + beta * L_B + gamma * L_f
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            p_v = model(Xv)['p_final'].squeeze(-1).cpu().numpy()
        m = evaluate(yva, (p_v > 0.5).astype(int), p_v)
        if m['f1'] > best_f1:
            best_f1, best_state, best_m = m['f1'], {k: v.clone() for k, v in model.state_dict().items()}, m
            pc = 0
        else:
            pc += 1
            if pc >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_v = model(Xv)
        pf = out_v['p_final'].squeeze(-1).cpu().numpy()
        pa = out_v['p_A'].squeeze(-1).cpu().numpy()
        pb = out_v['p_B'].squeeze(-1).cpu().numpy()
        gate = out_v['gate'].squeeze(-1).cpu().numpy()
    return {'metrics': best_m, 'p_final': pf, 'p_A': pa, 'p_B': pb, 'gate': gate}


def run_one(X, y, name, alpha, beta, gamma, device):
    out_dir = f'outputs/pr_de_net/{name}'
    if os.path.exists(os.path.join(out_dir, 'results.json')):
        print(f"  [{name}] cache hit")
        return json.load(open(os.path.join(out_dir, 'results.json')))
    set_seed(42)
    os.makedirs(out_dir, exist_ok=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_p, all_pA, all_pB, all_g, all_fi = (
        np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y)),
        np.zeros(len(y)), np.zeros(len(y), dtype=np.int64),
    )
    fold_results = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]).astype(np.float32)
        Xva = sc.transform(X[va]).astype(np.float32)
        out = train_one(Xtr, y[tr], Xva, y[va], device, alpha, beta, gamma)
        m = out['metrics']
        print(f"  [{name} fold {fold+1}] F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} AUC={m['auc']:.4f}")
        fold_results.append(m)
        all_p[va] = out['p_final']
        all_pA[va] = out['p_A']
        all_pB[va] = out['p_B']
        all_g[va] = out['gate']
        all_fi[va] = fold
    summary = summarize_fold_results(fold_results)
    print_results_table(f"PR-DE-Net ({name})", summary)
    np.save(os.path.join(out_dir, 'probs.npy'), all_p)
    np.save(os.path.join(out_dir, 'probs_A.npy'), all_pA)
    np.save(os.path.join(out_dir, 'probs_B.npy'), all_pB)
    np.save(os.path.join(out_dir, 'gates.npy'), all_g)
    np.save(os.path.join(out_dir, 'labels.npy'), y.astype(np.int64))
    np.save(os.path.join(out_dir, 'fold_idx.npy'), all_fi)
    sj = {
        'model': f'PR-DE-Net ({name})',
        'alpha': alpha, 'beta': beta, 'gamma': gamma,
        'cv_results': {k: summary[k] for k in [
            'accuracy_mean', 'accuracy_std', 'precision_mean', 'precision_std',
            'recall_mean', 'recall_std', 'f1_mean', 'f1_std', 'auc_mean', 'auc_std'
        ]},
        'fold_details': fold_results,
        'n_params': PRDENet().count_parameters(),
    }
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(sj, f, indent=2)
    return sj


if __name__ == '__main__':
    ide_logs, passed_df = load_ide_logs()
    X, y_p, _ = build_feature_matrix(ide_logs, passed_df)
    y = 1 - y_p
    device = get_device()
    configs = [
        ('v2_a_gamma3',   1.0, 1.0, 3.0),
        ('v2_b_alpha15',  1.5, 1.0, 2.0),
        ('v2_c_beta15',   1.0, 1.5, 2.0),
        ('v2_d_alpha05',  0.5, 0.5, 2.0),
    ]
    rows = []
    for name, a, b, g in configs:
        print(f"\n=== {name} a={a} b={b} g={g} ===")
        sj = run_one(X, y, name, a, b, g, device)
        rows.append((name, sj['cv_results']))

    # ensemble: average of 4
    print("\n=== Ensemble (avg of 4 v2 configs) ===")
    ens_dir = 'outputs/pr_de_net/v2_ensemble'
    os.makedirs(ens_dir, exist_ok=True)
    probs_list = [np.load(f'outputs/pr_de_net/{n}/probs.npy') for n, _, _, _ in configs]
    p_ens = np.mean(probs_list, axis=0)
    np.save(os.path.join(ens_dir, 'probs.npy'), p_ens)
    labels = np.load('outputs/pr_de_net/v2_a_gamma3/labels.npy')
    fold_idx = np.load('outputs/pr_de_net/v2_a_gamma3/fold_idx.npy')
    np.save(os.path.join(ens_dir, 'labels.npy'), labels)
    np.save(os.path.join(ens_dir, 'fold_idx.npy'), fold_idx)

    fold_metrics = []
    for f in range(5):
        m = fold_metrics_local = evaluate(
            labels[fold_idx == f],
            (p_ens[fold_idx == f] > 0.5).astype(int),
            p_ens[fold_idx == f],
        )
        fold_metrics.append(m)
        print(f"  [ensemble fold {f+1}] F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} AUC={m['auc']:.4f}")
    summary = summarize_fold_results(fold_metrics)
    print_results_table("PR-DE-Net (v2 ensemble)", summary)
    sj = {
        'model': 'PR-DE-Net (v2 ensemble)',
        'cv_results': {k: summary[k] for k in [
            'accuracy_mean', 'accuracy_std', 'precision_mean', 'precision_std',
            'recall_mean', 'recall_std', 'f1_mean', 'f1_std', 'auc_mean', 'auc_std'
        ]},
        'fold_details': fold_metrics,
        'components': [n for n, _, _, _ in configs],
    }
    with open(os.path.join(ens_dir, 'results.json'), 'w') as f:
        json.dump(sj, f, indent=2)

    print("\n" + "=" * 80)
    print(f"{'name':18s} {'F1':>10s} {'P':>10s} {'R':>10s} {'AUC':>10s}")
    for name, cv in rows:
        print(f"{name:18s} {cv['f1_mean']:.4f}±{cv['f1_std']:.3f} "
              f"{cv['precision_mean']:.4f}±{cv['precision_std']:.3f} "
              f"{cv['recall_mean']:.4f}±{cv['recall_std']:.3f} "
              f"{cv['auc_mean']:.4f}±{cv['auc_std']:.3f}")
    ens_cv = sj['cv_results']
    print(f"{'v2_ensemble':18s} {ens_cv['f1_mean']:.4f}±{ens_cv['f1_std']:.3f} "
          f"{ens_cv['precision_mean']:.4f}±{ens_cv['precision_std']:.3f} "
          f"{ens_cv['recall_mean']:.4f}±{ens_cv['recall_std']:.3f} "
          f"{ens_cv['auc_mean']:.4f}±{ens_cv['auc_std']:.3f}")