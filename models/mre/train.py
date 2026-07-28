"""
Multi-Route Expert (MRE) 训练脚本 - CS1 数据集

5 折分层交叉验证，统一折拆分 (random_state=42, stratified on failed=1)

训练流程：
  1. 在每折内独立训练 RF Expert (X_7d) 和 LSTM Expert (X_46d)
  2. 用 OOF probs 训练 Gating Network
  3. 三种融合策略都跑：soft / confidence / hard
  4. 与单模型 baseline 对比

输出：
  outputs/unified_compare/mre/{mode}/probs.npy, labels.npy, fold_idx.npy, results.json
  outputs/unified_compare/mre/all_results.json
"""
import os, sys, json, time, pickle, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)

_PROJECT_ROOT = '/home/ubuntu/CodeEMO'
sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import set_seed
from common.evaluator import evaluate, summarize_fold_results, print_results_table

from models.mre.mre_model import MREFusion

CACHE_DIR = '/tmp/codeemo_features'
OUT_DIR = '/home/ubuntu/CodeEMO/outputs/unified_compare/mre'
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EVENT_TYPES = ['text_insert', 'text_remove', 'text_paste',
               'focus_gained', 'focus_lost', 'run', 'submit']


def load_features():
    X7 = np.load(os.path.join(CACHE_DIR, 'X_7d.npy'))
    X46 = np.load(os.path.join(CACHE_DIR, 'X_46d.npy'))
    y_pass = np.load(os.path.join(CACHE_DIR, 'y.npy'))  # y=1 means passed
    return X7, X46, y_pass


# ----------------------- 简单 LSTM 模型 -----------------------
class SimpleLSTM(nn.Module):
    """LSTM on 46-dim -> P(failed)"""
    def __init__(self, input_dim=46, hidden=32, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=num_layers,
                            batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (B, 46) -> (B, 1, 46) single-step sequence
        x = x.unsqueeze(1)
        h, _ = self.lstm(x)
        return self.fc(self.dropout(h[:, -1, :])).squeeze(-1)


def train_lstm_one_fold(X_tr, y_tr, X_va, y_va,
                         epochs=80, batch_size=32, lr=1e-3, patience=10,
                         device=DEVICE, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SimpleLSTM(input_dim=X_tr.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()
    Xt = torch.FloatTensor(X_tr).to(device)
    yt = torch.FloatTensor(y_tr).to(device)
    Xv = torch.FloatTensor(X_va).to(device)
    yv = torch.FloatTensor(y_va).to(device)
    n = len(y_tr)
    best_v = float('inf')
    best_state = None
    pc = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits = model(Xt[idx])
            loss = crit(logits, yt[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v_loss = crit(model(Xv), yv).item()
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
        logits = model(Xv).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def train_gate_one_fold(rf_p_tr, lstm_p_tr, raw_tr, y_tr,
                         rf_p_va, lstm_p_va, raw_va, y_va,
                         fusion_mode='soft', epochs=300, lr=3e-3,
                         batch_size=64, seed=0, device=DEVICE):
    """训练门控网络 (MREFusion)"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MREFusion(raw_dim=raw_tr.shape[1], fusion_mode=fusion_mode).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # 输入是 probs, gate 网络需要 sigmoid 后的概率
    def to_prob(p):
        return np.clip(p, 1e-6, 1 - 1e-6).astype(np.float32)

    rf_tr_t = torch.FloatTensor(to_prob(rf_p_tr)).to(device)
    lstm_tr_t = torch.FloatTensor(to_prob(lstm_p_tr)).to(device)

    rf_va_t = torch.FloatTensor(to_prob(rf_p_va)).to(device)
    lstm_va_t = torch.FloatTensor(to_prob(lstm_p_va)).to(device)

    # 标准化 raw features
    sc = StandardScaler()
    raw_tr_s = sc.fit_transform(raw_tr).astype(np.float32)
    raw_va_s = sc.transform(raw_va).astype(np.float32)
    raw_tr_t = torch.FloatTensor(raw_tr_s).to(device)
    raw_va_t = torch.FloatTensor(raw_va_s).to(device)

    y_tr_t = torch.FloatTensor(y_tr.astype(np.float32)).to(device)
    y_va_t = torch.FloatTensor(y_va.astype(np.float32)).to(device)

    n = len(y_tr)
    best_v_loss = float('inf')
    best_state = None
    pc = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            fused_prob, _ = model(rf_tr_t[idx], lstm_tr_t[idx], raw_tr_t[idx])
            eps = 1e-6
            fused_clip = torch.clamp(fused_prob, eps, 1 - eps)
            loss = -(y_tr_t[idx] * torch.log(fused_clip) +
                     (1 - y_tr_t[idx]) * torch.log(1 - fused_clip)).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            fused_va, _ = model(rf_va_t, lstm_va_t, raw_va_t)
            eps = 1e-6
            fused_clip = torch.clamp(fused_va, eps, 1 - eps)
            v_loss = -(y_va_t * torch.log(fused_clip) +
                       (1 - y_va_t) * torch.log(1 - fused_clip)).mean().item()
        if v_loss < best_v_loss:
            best_v_loss = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1
            if pc >= 20:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        fused_va, gate_w = model(rf_va_t, lstm_va_t, raw_va_t)
    return fused_va.cpu().numpy(), gate_w.cpu().numpy(), best_state


# ----------------------- 主流程 -----------------------
def metric(y_true, y_pred, y_prob):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_true, y_prob)),
    }


def summary_metrics(folds):
    keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    out = {'n_folds': len(folds)}
    for k in keys:
        vals = [m[k] for m in folds]
        out[f'{k}_mean'] = float(np.mean(vals))
        out[f'{k}_std'] = float(np.std(vals))
    return out


def run_full_cv(seed=42):
    print(f'\n=== Multi-Route Expert Fusion ===')
    print(f'Device: {DEVICE}, seed={seed}')

    X7, X46, y_pass = load_features()
    y_failed = 1 - y_pass
    print(f'X7: {X7.shape}, X46: {X46.shape}, y_failed pos_rate: {y_failed.mean():.4f}')

    n = len(y_failed)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_idx = np.zeros(n, dtype=int)

    # 第一遍:训练 RF Expert 和 LSTM Expert,收集 OOF probs
    print('\n--- Training RF & LSTM experts (per fold) ---')
    rf_oof = np.zeros(n, dtype=np.float64)
    lstm_oof = np.zeros(n, dtype=np.float64)
    rf_metrics = []
    lstm_metrics = []
    for fi, (tr, va) in enumerate(skf.split(X7, y_failed)):
        fold_idx[va] = fi
        sc7 = StandardScaler().fit(X7[tr])
        sc46 = StandardScaler().fit(X46[tr])

        # RF Expert (on 7d)
        clf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                      random_state=seed, n_jobs=-1)
        clf.fit(sc7.transform(X7[tr]), y_failed[tr])
        rf_p = clf.predict_proba(sc7.transform(X7[va]))[:, 1]
        rf_oof[va] = rf_p
        rf_pred = (rf_p > 0.5).astype(int)
        rf_m = metric(y_failed[va], rf_pred, rf_p)
        rf_metrics.append(rf_m)
        print(f'  Fold {fi+1} RF: Acc={rf_m["accuracy"]:.4f} F1={rf_m["f1"]:.4f} '
              f'Prec={rf_m["precision"]:.4f} Rec={rf_m["recall"]:.4f} AUC={rf_m["auc"]:.4f}')

        # LSTM Expert (on 46d)
        lstm_p = train_lstm_one_fold(sc46.transform(X46[tr]), y_failed[tr].astype(np.float32),
                                      sc46.transform(X46[va]), y_failed[va].astype(np.float32),
                                      seed=seed + fi)
        lstm_oof[va] = lstm_p
        lstm_pred = (lstm_p > 0.5).astype(int)
        lstm_m = metric(y_failed[va], lstm_pred, lstm_p)
        lstm_metrics.append(lstm_m)
        print(f'  Fold {fi+1} LSTM: Acc={lstm_m["accuracy"]:.4f} F1={lstm_m["f1"]:.4f} '
              f'Prec={lstm_m["precision"]:.4f} Rec={lstm_m["recall"]:.4f} AUC={lstm_m["auc"]:.4f}')

    print('\n--- RF Expert 5-fold summary ---')
    rf_sum = summary_metrics(rf_metrics)
    print_results_table('RF Expert (7d)', {**rf_sum, 'folds': rf_metrics})

    print('\n--- LSTM Expert 5-fold summary ---')
    lstm_sum = summary_metrics(lstm_metrics)
    print_results_table('LSTM Expert (46d)', {**lstm_sum, 'folds': lstm_metrics})

    np.save(os.path.join(OUT_DIR, 'rf_expert_oof.npy'), rf_oof)
    np.save(os.path.join(OUT_DIR, 'lstm_expert_oof.npy'), lstm_oof)
    np.save(os.path.join(OUT_DIR, 'labels.npy'), y_failed.astype(np.int8))
    np.save(os.path.join(OUT_DIR, 'fold_idx.npy'), fold_idx)

    # 简单基线:平均
    avg_oof = 0.5 * (rf_oof + lstm_oof)
    avg_pred = (avg_oof > 0.5).astype(int)
    avg_metrics = [metric(y_failed[fold_idx == fi], avg_pred[fold_idx == fi],
                           avg_oof[fold_idx == fi])
                   for fi in range(5)]
    avg_sum = summary_metrics(avg_metrics)

    # 网格搜索最优线性权重
    print('\n--- Grid search optimal linear weight (RF*w + LSTM*(1-w)) ---')
    best_f1 = -1; best_w = None; best_metrics = None
    for w in np.linspace(0, 1, 21):
        fused = w * rf_oof + (1 - w) * lstm_oof
        preds = (fused > 0.5).astype(int)
        m_folds = [metric(y_failed[fold_idx == fi], preds[fold_idx == fi],
                           fused[fold_idx == fi]) for fi in range(5)]
        ms = summary_metrics(m_folds)
        if ms['f1_mean'] > best_f1:
            best_f1 = ms['f1_mean']; best_w = w; best_metrics = ms
    print(f'  best w_rf={best_w:.2f}, F1={best_f1:.4f}')
    grid_sum = best_metrics

    # 第二遍:训练门控网络 (per fold)
    print('\n--- Training Gating Networks (per fold) ---')
    gate_results = {}
    gate_model_states = {}  # 保存每个 mode 的 5 个 fold 模型
    for mode in ['soft', 'confidence', 'hard']:
        print(f'\n  [{mode}]')
        model_states = []
        oof_fused = np.zeros(n)
        oof_w_rf = np.zeros(n)
        fold_metrics = []
        for fi, (tr, va) in enumerate(skf.split(X7, y_failed)):
            fused_p, gw, model_state = train_gate_one_fold(
                rf_oof[tr], lstm_oof[tr], X7[tr], y_failed[tr],
                rf_oof[va], lstm_oof[va], X7[va], y_failed[va],
                fusion_mode=mode, seed=seed + fi * 7)
            model_states.append(model_state)
            oof_fused[va] = fused_p
            oof_w_rf[va] = gw[:, 0]
            pred = (fused_p > 0.5).astype(int)
            m = metric(y_failed[va], pred, fused_p)
            fold_metrics.append(m)
            print(f'    Fold {fi+1}: Acc={m["accuracy"]:.4f} F1={m["f1"]:.4f} '
                  f'Prec={m["precision"]:.4f} Rec={m["recall"]:.4f} AUC={m["auc"]:.4f} '
                  f'alpha_rf_mean={gw[:,0].mean():.3f}')

        ms = summary_metrics(fold_metrics)
        gate_results[mode] = ms
        print_results_table(f'MRE Fusion ({mode})', {**ms, 'folds': fold_metrics})
        gate_model_states[mode] = model_states
        sub = os.path.join(OUT_DIR, mode)
        os.makedirs(sub, exist_ok=True)
        np.save(os.path.join(sub, 'probs.npy'), oof_fused.astype(np.float64))
        np.save(os.path.join(sub, 'labels.npy'), y_failed.astype(np.int8))
        np.save(os.path.join(sub, 'fold_idx.npy'), fold_idx)
        np.save(os.path.join(sub, 'gate_w_rf.npy'), oof_w_rf)
        with open(os.path.join(sub, 'results.json'), 'w') as f:
            json.dump({
                'model': f'MultiRouteExpert-{mode}',
                'cv_results': {k: v for k, v in ms.items() if k != 'folds'},
                'fold_details': fold_metrics,
                'n_samples': int(n),
                'n_failed': int(y_failed.sum()),
                'n_passed': int((1 - y_failed).sum()),
                'label_convention': 'y=1=failed',
                'expert_a': 'RandomForest (7d, n_estimators=200, max_depth=12)',
                'expert_b': 'LSTM (46d, hidden=32, 1-layer)',
                'gate': 'MLP(6+7->32->16->2) softmax over (alpha_rf, alpha_lstm)',
            }, f, indent=2)

    report = {
        'dataset': {
            'name': 'CS1',
            'n_samples': int(n),
            'n_failed': int(y_failed.sum()),
            'n_passed': int((1 - y_failed).sum()),
            'pos_rate_failed': float(y_failed.mean()),
            'features': '7-dim event counts + 46-dim hand-crafted',
        },
        'experts': {
            'rf_7d': rf_sum,
            'lstm_46d': lstm_sum,
        },
        'baselines': {
            'avg_50_50': avg_sum,
            'grid_best_w_rf': {**grid_sum, 'best_w_rf': float(best_w)},
        },
        'mre_fusion': {
            mode: gate_results[mode] for mode in ['soft', 'confidence', 'hard']
        },
    }
    # 保存 gate 模型状态用于 SHAP 分析
    model_dir = os.path.join(OUT_DIR, 'gate_models')
    os.makedirs(model_dir, exist_ok=True)
    for mode, states in gate_model_states.items():
        for fi, st in enumerate(states):
            torch.save(st, os.path.join(model_dir, f'{mode}_fold{fi+1}.pt'))
    print(f'[SAVED] gate models -> {model_dir}')

    with open(os.path.join(OUT_DIR, 'all_results.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\n[SAVED] all_results.json -> {OUT_DIR}')

    return report


if __name__ == '__main__':
    set_seed(42)
    t0 = time.time()
    report = run_full_cv(seed=42)
    print(f'\nTotal time: {time.time()-t0:.1f}s')