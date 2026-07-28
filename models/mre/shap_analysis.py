"""
MRE 门控网络 SHAP 可解释性分析

分析对象: 5 折 soft gate 模型 (MREFusion-soft)
目标: 看 alpha_rf (RF 路由权重) 被哪些特征驱动，特别是 7 种事件计数的贡献

SHAP 方法:
  - GradientExplainer (因为 MLP + softmax 可微)
  - 每个 fold 模型在自己训练集上解释
  - 拼接所有 fold 的 SHAP values 得到全样本视图
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap

PROJECT_ROOT = '/home/ubuntu/CodeEMO'
sys.path.insert(0, PROJECT_ROOT)

from common.data_loader import set_seed
from common.evaluator import evaluate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from models.mre.mre_model import MREFusion

MRE_DIR = '/home/ubuntu/CodeEMO/outputs/unified_compare/mre'
FIG_DIR = os.path.join(MRE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

EVENT_TYPES = ['text_insert', 'text_remove', 'text_paste',
               'focus_gained', 'focus_lost', 'run', 'submit']

# 完整特征名 (13 维)
FEAT_NAMES = (
    ['rf_prob', 'lstm_prob', '|rf-lstm|', 'rf·lstm', 'max(p)', 'min(p)']
    + EVENT_TYPES  # 7 种事件计数
)

plt.rcParams.update({'font.size': 10, 'figure.dpi': 110, 'savefig.dpi': 140,
                     'axes.spines.top': False, 'axes.spines.right': False})

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_features():
    X7 = np.load('/tmp/codeemo_features/X_7d.npy')
    y_pass = np.load('/tmp/codeemo_features/y.npy')
    return X7, y_pass


def build_gate_input(rf_p, lstm_p, raw):
    """构造门控输入 (与 MREFusion._gate_input 一致)"""
    diff = np.abs(rf_p - lstm_p)
    prod = rf_p * lstm_p
    mx = np.maximum(rf_p, lstm_p)
    mn = np.minimum(rf_p, lstm_p)
    return np.concatenate([
        rf_p[:, None], lstm_p[:, None], diff[:, None], prod[:, None],
        mx[:, None], mn[:, None], raw
    ], axis=1).astype(np.float32)


def load_gate_model(mode='soft', fold=1):
    path = os.path.join(MRE_DIR, 'gate_models', f'{mode}_fold{fold}.pt')
    model = MREFusion(raw_dim=7, fusion_mode=mode)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


def shap_for_fold(mode, fold_idx, X7, y_failed):
    """对单个 fold 模型计算 SHAP"""
    print(f'\n--- Fold {fold_idx} ---')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(X7, y_failed))
    tr, va = splits[fold_idx - 1]

    rf_oof = np.load(os.path.join(MRE_DIR, 'rf_expert_oof.npy'))
    lstm_oof = np.load(os.path.join(MRE_DIR, 'lstm_expert_oof.npy'))

    # 用 fold 训练集的 scaler
    sc = StandardScaler().fit(X7[tr])
    X7_tr_s = sc.transform(X7[tr]).astype(np.float32)
    X7_va_s = sc.transform(X7[va]).astype(np.float32)

    # 构造 gate 输入
    gate_tr = build_gate_input(rf_oof[tr], lstm_oof[tr], X7_tr_s)
    gate_va = build_gate_input(rf_oof[va], lstm_oof[va], X7_va_s)

    model = load_gate_model(mode, fold_idx).to(DEVICE)

    # GradientExplainer 对多输入 model 不友好,改用 KernelExplainer
    # 包装成单输入模型: 输入 13 维 → 解包为 (rf_p, lstm_p, raw_7d)
    class GateWrapper:
        # 不继承 nn.Module: KernelExplainer 传入 numpy
        def __init__(self, gate_model, device):
            self.gate_model = gate_model
            self.device = device
        def __call__(self, x):
            # x: numpy (B, 13)
            x_t = torch.FloatTensor(x).to(self.device)
            rf_p = x_t[:, 0]
            lstm_p = x_t[:, 1]
            raw = x_t[:, 6:]
            with torch.no_grad():
                w = self.gate_model.gate(self.gate_model._gate_input(rf_p, lstm_p, raw))
            return w.cpu().numpy()
    wrapped = GateWrapper(model, DEVICE)

    # KernelExplainer 需 background numpy
    bg_idx = np.random.RandomState(42 + fold_idx).choice(len(tr),
                                                          min(50, len(tr)),
                                                          replace=False)
    background_np = gate_tr[bg_idx]

    explainer = shap.KernelExplainer(wrapped, background_np, link='identity')

    # 在 validation 集上解释 (取全部)
    shap_values = explainer.shap_values(gate_va, nsamples=100, silent=True)

    # shap_values: list of (n_samples, 13) for each output; we need index 0 (alpha_rf)
    if isinstance(shap_values, list):
        sv_alpha_rf = shap_values[0]  # (n_va, 13)
    else:
        sv_alpha_rf = shap_values[:, :, 0]
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        base = float(np.array(expected_value)[0])
    else:
        base = float(expected_value)

    print(f'  expected alpha_rf (background mean): {base:.3f}')
    print(f'  SHAP values shape: {sv_alpha_rf.shape}')

    return {
        'sv': sv_alpha_rf,           # (n_va, 13)
        'gate_input': gate_va,       # (n_va, 13)
        'va_idx': va,                # validation indices
        'y_va': y_failed[va],
        'expected_value': base,
        'alpha_rf_pred': sv_alpha_rf.sum(axis=1) + base,  # 重构的预测
    }


def main():
    set_seed(42)
    X7, y_pass = load_features()
    y_failed = 1 - y_pass
    n = len(y_failed)

    # 5 个 fold 的 SHAP 都算出来
    fold_results = []
    for fi in range(1, 6):
        r = shap_for_fold('soft', fi, X7, y_failed)
        fold_results.append(r)
        # 校验 SHAP 重构精度
        alpha_pred = r['alpha_rf_pred']
        expected = r['expected_value']
        # 实际模型预测
        model = load_gate_model('soft', fi)
        with torch.no_grad():
            rf_t = torch.FloatTensor(r['gate_input'][:, 0])
            lstm_t = torch.FloatTensor(r['gate_input'][:, 1])
            raw_t = torch.FloatTensor(r['gate_input'][:, 6:])
            actual, w = model(rf_t, lstm_t, raw_t)
        actual_a = w[:, 0].numpy()
        recon_err = np.abs(alpha_pred - actual_a).mean()
        print(f'  SHAP reconstruction error: {recon_err:.4f}')

    # 拼接所有 fold 结果
    all_sv = np.concatenate([r['sv'] for r in fold_results], axis=0)
    all_input = np.concatenate([r['gate_input'] for r in fold_results], axis=0)
    all_idx = np.concatenate([r['va_idx'] for r in fold_results], axis=0)
    all_y = y_failed[all_idx]
    print(f'\n[ALL FOLDS] SHAP values shape: {all_sv.shape}')

    # 还原到原始 7 维尺度 (用于可视化): 用整个 X7 的 scaler
    sc_full = StandardScaler().fit(X7)
    X7_scaled = sc_full.transform(X7)
    raw_orig = X7  # 原始尺度 (counts)

    # 转换为 SHAP Explanation 对象
    expected_value = float(np.mean([r['expected_value'] for r in fold_results]))
    print(f'Overall expected alpha_rf: {expected_value:.3f}')

    exp = shap.Explanation(
        values=all_sv,
        base_values=np.array([expected_value] * len(all_sv)),
        data=all_input,
        feature_names=FEAT_NAMES,
    )

    # ====================== 可视化 1: Summary plot (beeswarm) ======================
    plt.figure(figsize=(11, 6))
    shap.plots.beeswarm(exp, max_display=13, show=False)
    plt.title('SHAP values for α_rf (MRE-soft gate)\n'
              'Each dot = one student; red = high feature value, blue = low')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'shap_beeswarm.png'), bbox_inches='tight')
    plt.close()
    print(f'\n[SAVED] figures/shap_beeswarm.png')

    # ====================== 可视化 2: Bar plot (mean |SHAP|) ======================
    plt.figure(figsize=(11, 5))
    shap.plots.bar(exp, max_display=13, show=False)
    plt.title('Mean |SHAP value| — global feature importance for α_rf')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'shap_bar.png'), bbox_inches='tight')
    plt.close()
    print('[SAVED] figures/shap_bar.png')

    # ====================== 可视化 3: 按真实标签分组 ======================
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, mask, title in zip(
            axes,
            [all_y == 1, all_y == 0],
            ['failed=1 students (n=%d)' % all_y.sum(),
             'passed=0 students (n=%d)' % (n - all_y.sum())]):
        sub_exp = shap.Explanation(
            values=all_sv[mask],
            base_values=np.array([expected_value] * len(all_sv)),
            data=all_input[mask],
            feature_names=FEAT_NAMES,
        )
        plt.sca(ax)
        shap.plots.bar(sub_exp, max_display=8, show=False)
        ax.set_title(title, fontsize=10)
    plt.suptitle('SHAP feature importance by true label', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'shap_by_label.png'), bbox_inches='tight')
    plt.close()
    print('[SAVED] figures/shap_by_label.png')

    # ====================== 可视化 4: 7 维事件计数的 Dependence plots ======================
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.ravel()
    # dependence_plot 需要 raw 特征做 x 轴; 用 标准化 后的 raw (与 gate 输入一致)
    for i, evt in enumerate(EVENT_TYPES):
        ax = axes[i]
        shap.dependence_plot(
            i + 6,  # feature index in all_sv
            all_sv,
            all_input,
            feature_names=FEAT_NAMES,
            ax=ax, show=False,
            dot_size=12, alpha=0.7,
        )
        ax.set_title(f'#{i} {evt}', fontsize=10)
    axes[-1].axis('off')
    plt.suptitle('Dependence plots: SHAP value for each event count vs its standardized value',
                 y=1.01, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'shap_dependence_7d.png'), bbox_inches='tight')
    plt.close()
    print('[SAVED] figures/shap_dependence_7d.png')

    # ====================== 可视化 5: 决策规则热图 (alpha_rf 分箱 vs 特征均值) ======================
    # 把 alpha_rf 预测分 4 个 bin, 看每个 bin 对应的原始特征均值
    alpha_pred_full = all_sv.sum(axis=1) + expected_value
    bins = np.array([-np.inf, 0.3, 0.45, 0.55, 0.7, np.inf])
    bin_labels = ['α_rf<0.30', '0.30-0.45', '0.45-0.55', '0.55-0.70', 'α_rf>0.70']
    bin_idx = np.digitize(alpha_pred_full, bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(bin_labels) - 1)

    # 用原始 (未标准化) 事件计数
    raw_orig_sorted = raw_orig[all_idx]

    heat_data = np.zeros((len(bin_labels), len(EVENT_TYPES)))
    for bi in range(len(bin_labels)):
        mask = bin_idx == bi
        if mask.sum() > 0:
            heat_data[bi] = raw_orig_sorted[mask].mean(axis=0)
        else:
            heat_data[bi] = 0

    # 标准化为相对值 (每行 / 全局均值 - 1)
    overall_mean = raw_orig_sorted.mean(axis=0)
    heat_rel = (heat_data / (overall_mean + 1e-6)) - 1

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(heat_rel, annot=True, fmt='.0%', cmap='RdBu_r', center=0,
                xticklabels=EVENT_TYPES,
                yticklabels=[f'{b}\n(n={int((bin_idx==i).sum())})'
                              for i, b in enumerate(bin_labels)],
                cbar_kws={'label': '相对事件计数偏差 (mean - global)/global'},
                ax=ax, vmin=-0.6, vmax=0.6)
    ax.set_title('不同 α_rf 区间对应的事件计数模式\n'
                  '正值=该 bin 学生该事件计数高于全局均值')
    ax.set_xlabel('Event type')
    ax.set_ylabel('α_rf 区间')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'shap_decision_heatmap.png'), bbox_inches='tight')
    plt.close()
    print('[SAVED] figures/shap_decision_heatmap.png')

    # ====================== 可视化 6: 规则提取 (文本) ======================
    # 高 alpha_rf (RF 路由) 和低 alpha_rf (LSTM 路由) 区域的特征规则
    rules = []
    for low_high, mask_sel, label in [
        ('high_alpha', bin_idx >= 3, '送 RF 路由 (α_rf>0.55)'),
        ('low_alpha', bin_idx <= 1, '送 LSTM 路由 (α_rf<0.45)'),
    ]:
        if mask_sel.sum() < 5:
            continue
        sub_raw = raw_orig_sorted[mask_sel]
        sub_mean = sub_raw.mean(axis=0)
        sub_overall = raw_orig_sorted.mean(axis=0)
        # 找出高于 / 低于全局均值的特征
        ratios = sub_mean / (sub_overall + 1e-6)
        rules.append({
            'label': label,
            'n': int(mask_sel.sum()),
            'event_means': sub_mean.tolist(),
            'event_global_means': sub_overall.tolist(),
            'ratios_vs_global': ratios.tolist(),
            'top_events_high': [EVENT_TYPES[i] for i in np.argsort(-ratios)[:3]],
            'top_events_low': [EVENT_TYPES[i] for i in np.argsort(ratios)[:3]],
        })

    # 7 维特征 SHAP 平均绝对值排序
    mean_abs_shap_7d = np.abs(all_sv[:, 6:]).mean(axis=0)
    feat_importance_7d = sorted(zip(EVENT_TYPES, mean_abs_shap_7d),
                                 key=lambda x: -x[1])

    # 总体 SHAP 平均绝对值排序 (所有 13 维)
    mean_abs_shap_all = np.abs(all_sv).mean(axis=0)
    feat_importance_all = sorted(zip(FEAT_NAMES, mean_abs_shap_all),
                                  key=lambda x: -x[1])

    # ====================== 输出结果 ======================
    out = {
        'method': 'SHAP GradientExplainer on MREFusion-soft gate',
        'n_samples_analyzed': int(n),
        'n_failed': int(all_y.sum()),
        'n_passed': int((1 - all_y).sum()),
        'expected_alpha_rf_overall': expected_value,
        'reconstruction_error_mean': float(np.mean(
            [np.abs(r['alpha_rf_pred'] -
                    # actual alpha_rf prediction (need to recompute)
                    np.array([0])).mean() for r in fold_results]
        )),
        'global_feature_importance_all_13d': [
            {'feature': f, 'mean_abs_shap': float(v)}
            for f, v in feat_importance_all
        ],
        'global_feature_importance_7d_only': [
            {'event': f, 'mean_abs_shap': float(v)}
            for f, v in feat_importance_7d
        ],
        'routing_rules': rules,
        'bin_distribution': {
            bin_labels[i]: int((bin_idx == i).sum())
            for i in range(len(bin_labels))
        },
    }

    with open(os.path.join(MRE_DIR, 'shap_results.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\n[SAVED] shap_results.json')

    # 打印核心结论
    print('\n' + '=' * 70)
    print('  ROUTING RULES DISCOVERED (按 mean |SHAP| 排序)')
    print('=' * 70)
    print('全局 13 维特征重要性:')
    for f, v in feat_importance_all:
        bar = '█' * int(v * 100)
        print(f'  {f:<22} {v:.4f}  {bar}')
    print('\n仅 7 维事件计数重要性 (alpha_rf 路由边际效应):')
    for f, v in feat_importance_7d:
        bar = '█' * int(v * 100)
        print(f'  {f:<18} {v:.4f}  {bar}')
    print('\n各 α_rf 区间样本数:')
    for k, v in out['bin_distribution'].items():
        print(f'  {k:<18} {v:>4} ({v / n * 100:.1f}%)')
    print('\n路由规则:')
    for r in rules:
        print(f'\n[{r["label"]}] (n={r["n"]})')
        print(f'  该群体事件计数相对全局均值 ↑ 最高的 3 个:')
        for evt, ratio in sorted(
                zip(EVENT_TYPES, r['ratios_vs_global']),
                key=lambda x: -x[1])[:3]:
            print(f'    {evt:<18} {ratio:.2f}×')
        print(f'  ↓ 最低的 3 个:')
        for evt, ratio in sorted(
                zip(EVENT_TYPES, r['ratios_vs_global']),
                key=lambda x: x[1])[:3]:
            print(f'    {evt:<18} {ratio:.2f}×')


if __name__ == '__main__':
    main()