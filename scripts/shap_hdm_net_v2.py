#!/usr/bin/env python3
"""
HDM-Net v2 + SHAP 分析
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

_PROJECT_ROOT = "/home/ubuntu/CodeEMO"
sys.path.insert(0, _PROJECT_ROOT)
from common.data_loader import load_ide_logs, set_seed
from common.feature_engineering import build_feature_matrix, EVENT_TYPES
from models.hdm_net.model import HDMNet

import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/CodeEMO/outputs/unified_compare/hdm_net_v2_shap"
os.makedirs(OUT_DIR, exist_ok=True)


def prepare_data_local():
    """复用 HDM-Net v2 训练脚本的数据准备"""
    set_seed(42)
    ide_logs, passed = load_ide_logs()
    X_46d, y_passed, student_ids = build_feature_matrix(ide_logs, passed)
    y = 1 - y_passed

    df = ide_logs.groupby(['student', 'eventType']).size().unstack(fill_value=0)
    df = df.reindex(columns=EVENT_TYPES, fill_value=0)
    df = df.reindex(index=student_ids, fill_value=0)
    x_tree_7d = df.values.astype(np.float32)

    x_tree = x_tree_7d

    x_seq = X_46d.reshape(-1, 46, 1).astype(np.float32)
    x_att = x_tree_7d.reshape(-1, 7, 1).astype(np.float32)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_probs = np.zeros(len(y), dtype=np.float32)
    for tr, va in skf.split(x_tree, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(x_tree[tr])
        Xva = scaler.transform(x_tree[va])
        rf.fit(Xtr, y[tr])
        rf_probs[va] = rf.predict_proba(Xva)[:, 1]
    tree_probs = np.column_stack([1 - rf_probs, rf_probs]).astype(np.float32)

    return x_tree, tree_probs, x_seq, x_att, y


def train_fold_and_save(x_tree, tree_probs, x_seq, x_att, y,
                        device='cuda', epochs=60, batch_size=32, patience=10, seed=42):
    """重训 1 个 fold 保存权重"""
    set_seed(seed)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_models = []

    for fi, (tr, va) in enumerate(skf.split(x_tree, y)):
        model = HDMNet(d=32, tree_depth=3, tree_width=64).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
        bce = nn.BCEWithLogitsLoss()

        best_v = float('inf')
        best_state = None
        pc = 0

        for ep in range(1, epochs + 1):
            model.train()
            for i in range(0, len(tr), batch_size):
                idx = tr[i:i + batch_size]
                xt = torch.FloatTensor(x_tree[idx]).to(device)
                tp = torch.FloatTensor(tree_probs[idx]).to(device)
                xs = torch.FloatTensor(x_seq[idx]).to(device)
                xa = torch.FloatTensor(x_att[idx]).to(device)
                yt = torch.FloatTensor(y[idx]).to(device)
                logits = model(xt, tp, xs, xa, return_gate=False)
                loss = bce(logits, yt)
                optim.zero_grad()
                loss.backward()
                optim.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                xt = torch.FloatTensor(x_tree[va]).to(device)
                tp = torch.FloatTensor(tree_probs[va]).to(device)
                xs = torch.FloatTensor(x_seq[va]).to(device)
                xa = torch.FloatTensor(x_att[va]).to(device)
                yt = torch.FloatTensor(y[va]).to(device)
                logits = model(xt, tp, xs, xa, return_gate=False)
                v_loss = bce(logits, yt).item()
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
        fold_models.append(model.cpu())
        torch.save(best_state, os.path.join(OUT_DIR, f"fold_{fi}.pt"))
        print(f"  Fold {fi+1}/5 训练完成", flush=True)
    return fold_models


class WrappedModel(nn.Module):
    """包装 HDM-Net: 接受 66-d flattened input"""
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model

    def forward(self, X):
        X_t = torch.FloatTensor(X) if not torch.is_tensor(X) else X
        xt = X_t[:, :9]
        tp = X_t[:, 9:11]
        xs = X_t[:, 11:57].unsqueeze(-1)
        xa = X_t[:, 57:64].unsqueeze(-1)
        logits = self.model(xt, tp, xs, xa, return_gate=False)
        return torch.sigmoid(logits)


def compute_shap(model, X_combined, background_size=30, test_size=60):
    """KernelExplainer (model-agnostic)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model = model.to(device)

    background = X_combined[:background_size]
    test_samples = X_combined[background_size:background_size + test_size]

    print(f"  背景: {background.shape}, 测试: {test_samples.shape}", flush=True)

    def f(X):
        X_t = torch.FloatTensor(X).to(device)
        xt = X_t[:, :7]
        tp = X_t[:, 7:9]
        xs = X_t[:, 9:55].unsqueeze(-1)
        xa = X_t[:, 55:62].unsqueeze(-1)
        with torch.no_grad():
            logits = model(xt, tp, xs, xa, return_gate=False)
            probs = torch.sigmoid(logits)
        return probs.cpu().numpy()

    test_out = f(X_combined[:2])
    print(f"  f 输出: {test_out.shape}", flush=True)

    explainer = shap.KernelExplainer(f, background)
    shap_values = explainer.shap_values(test_samples, nsamples=50, silent=True)

    if isinstance(shap_values, list):
        shap_values = np.array(shap_values[0])
    return shap_values, test_samples, f, explainer


def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}", flush=True)

    print("加载数据...", flush=True)
    x_tree, tree_probs, x_seq, x_att, y = prepare_data_local()
    n = len(y)
    print(f"数据: n={n}, fail_rate={y.mean():.4f}", flush=True)

    print("\n=== 步骤 1: 重训 HDM-Net v2 5-fold 保存权重 ===", flush=True)
    start = time.time()
    fold_models = train_fold_and_save(x_tree, tree_probs, x_seq, x_att, y, device=device)
    print(f"训练耗时: {time.time()-start:.1f}s", flush=True)

    print("\n=== 步骤 2: 计算 SHAP (GradientExplainer) ===", flush=True)
    x_seq_flat = x_seq.mean(axis=2)
    x_att_flat = x_att.mean(axis=2)
    X_combined = np.concatenate([x_tree, tree_probs, x_seq_flat, x_att_flat], axis=1)
    print(f"  Total features: {X_combined.shape[1]} = 9 (tree) + 2 (RF) + 46 (seq) + 7 (att)")
    print(f"  简化输入维度: {X_combined.shape}", flush=True)

    feat_names = (
        EVENT_TYPES +
        ['rf_p_passed', 'rf_p_failed'] +
        [f'seq_dim_{i}' for i in range(46)] +
        [f'att_dim_{i}' for i in range(7)]
    )

    shap_values, test_samples, wrapped, explainer = compute_shap(fold_models[0], X_combined)

    print(f"  SHAP shape: {shap_values.shape}", flush=True)

    np.savez(os.path.join(OUT_DIR, 'shap_values.npz'),
             shap_values=shap_values, X=test_samples,
             feature_names=np.array(feat_names, dtype=object))

    print("\n=== 步骤 3: 出图 ===", flush=True)

    # Summary beeswarm
    plt.figure(figsize=(12, 8))
    try:
        shap.summary_plot(shap_values, test_samples, feature_names=feat_names,
                          show=False, max_display=20)
    except Exception as e:
        print(f"  summary_plot 失败: {e}, 退化为 bar", flush=True)
        shap.summary_plot(shap_values, test_samples, feature_names=feat_names,
                          plot_type='bar', show=False, max_display=20)
    plt.title('HDM-Net v2 SHAP Summary (top 20 features)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'summary_beeswarm.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print("  summary_beeswarm.png", flush=True)

    # Bar
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, test_samples, feature_names=feat_names,
                      plot_type='bar', show=False, max_display=20)
    plt.title('HDM-Net v2 Mean |SHAP| (top 20 features)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'importance_bar.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print("  importance_bar.png", flush=True)

    # Force plots (top 3 high-confidence failed)
    model = fold_models[0]
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    with torch.no_grad():
        xt = torch.FloatTensor(x_tree).to(device)
        tp = torch.FloatTensor(tree_probs).to(device)
        xs = torch.FloatTensor(x_seq).to(device)
        xa = torch.FloatTensor(x_att).to(device)
        logits = model(xt, tp, xs, xa, return_gate=False)
        probs = torch.sigmoid(logits).cpu().numpy()

    top_idx = np.argsort(probs)[::-1][:3]
    for rank, idx in enumerate(top_idx, 1):
        plt.figure(figsize=(14, 3))
        try:
            # KernelExplainer 期望值
            try:
                base = float(explainer.expected_value)
                if isinstance(base, (list, np.ndarray)):
                    base = float(base[0]) if len(base) > 0 else 0.5
            except Exception:
                base = 0.5
            shap.force_plot(base, shap_values[idx], test_samples[idx],
                             feature_names=feat_names, matplotlib=True, show=False)
            plt.title(f"Force rank{rank} prob={probs[idx]:.3f} truth={'FAIL' if y[idx]==1 else 'PASS'}",
                      fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f'force_rank{rank}_idx{idx}.png'), dpi=120, bbox_inches='tight')
            plt.close()
            print(f"  force_rank{rank}_idx{idx}.png (prob={probs[idx]:.3f})", flush=True)
        except Exception as e:
            print(f"  force rank{rank} 失败: {e}", flush=True)
            plt.close()

    # Top 15 特征排名
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:15]
    print("\n=== 步骤 4: Top 15 特征 (mean |SHAP|) ===", flush=True)
    for i, idx in enumerate(top_idx, 1):
        print(f"  {i:2d}. {feat_names[idx]:30s}  {mean_abs[idx]:.4f}", flush=True)

    summary = {
        'model': 'HDM-Net v2 + SHAP (GradientExplainer)',
        'n_samples': int(n),
        'n_features': 64,
        'feature_breakdown': {
            '7_events': 7, '2_rf_probs': 2, '46_seq_dims': 46, '7_att_dims': 7,
        },
        'top_15_features': [(feat_names[i], float(mean_abs[i])) for i in top_idx],
        'shap_stats': {
            'mean_abs_shap': float(mean_abs.mean()),
            'std_abs_shap': float(mean_abs.std()),
            'max_abs_shap': float(mean_abs.max()),
        },
        'elapsed_seconds': time.time() - start,
    }
    with open(os.path.join(OUT_DIR, 'shap_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*72}", flush=True)
    print(f"HDM-Net v2 + SHAP 完成, 总耗时: {time.time()-start:.1f}s", flush=True)
    print(f"输出: {OUT_DIR}/", flush=True)


if __name__ == '__main__':
    main()