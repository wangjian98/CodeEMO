"""
SHAP Deep Dive - 失败/通过学生的路由差异 + Permutation importance 对照

比较两组人群的 SHAP 行为:
  - failed=1 学生 (n=314): SHAP values, α_rf 分布
  - passed=0 学生 (n=159): SHAP values, α_rf 分布

对照实验: Permutation importance
  - 把每个 raw_7d 维度打乱, 看 α_rf 预测损失多少
  - 与 SHAP 重要性交叉验证
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = '/home/ubuntu/CodeEMO'
sys.path.insert(0, PROJECT_ROOT)

from common.data_loader import set_seed
from models.mre.mre_model import MREFusion

MRE_DIR = '/home/ubuntu/CodeEMO/outputs/unified_compare/mre'
FIG_DIR = os.path.join(MRE_DIR, 'figures')

EVENT_TYPES = ['text_insert', 'text_remove', 'text_paste',
               'focus_gained', 'focus_lost', 'run', 'submit']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.rcParams.update({'font.size': 10, 'figure.dpi': 110, 'savefig.dpi': 140,
                     'axes.spines.top': False, 'axes.spines.right': False})


def build_gate_input(rf_p, lstm_p, raw):
    diff = np.abs(rf_p - lstm_p)
    prod = rf_p * lstm_p
    mx = np.maximum(rf_p, lstm_p)
    mn = np.minimum(rf_p, lstm_p)
    return np.concatenate([
        rf_p[:, None], lstm_p[:, None], diff[:, None], prod[:, None],
        mx[:, None], mn[:, None], raw
    ], axis=1).astype(np.float32)


def main():
    set_seed(42)
    X7 = np.load('/tmp/codeemo_features/X_7d.npy')
    y_pass = np.load('/tmp/codeemo_features/y.npy')
    y_failed = 1 - y_pass
    n = len(y_failed)

    rf_oof = np.load(os.path.join(MRE_DIR, 'rf_expert_oof.npy'))
    lstm_oof = np.load(os.path.join(MRE_DIR, 'lstm_expert_oof.npy'))

    # 5 折 soft gate 模型
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(X7, y_failed))

    # 收集所有 fold 的 alpha_rf 预测 + SHAP values
    # 简单重跑: 直接前向传播获取 alpha_rf
    all_alpha = np.zeros(n)
    for fi in range(5):
        tr, va = splits[fi]
        sc = StandardScaler().fit(X7[tr])
        X7_va_s = sc.transform(X7[va]).astype(np.float32)
        gate_va = build_gate_input(rf_oof[va], lstm_oof[va], X7_va_s)

        model = MREFusion(raw_dim=7, fusion_mode='soft')
        model.load_state_dict(torch.load(
            os.path.join(MRE_DIR, 'gate_models', f'soft_fold{fi+1}.pt'),
            map_location='cpu'))
        model.eval()

        with torch.no_grad():
            rf_t = torch.FloatTensor(gate_va[:, 0])
            lstm_t = torch.FloatTensor(gate_va[:, 1])
            raw_t = torch.FloatTensor(gate_va[:, 6:])
            _, w = model(rf_t, lstm_t, raw_t)
        all_alpha[va] = w[:, 0].numpy()

    print(f'Overall α_rf: mean={all_alpha.mean():.3f}, std={all_alpha.std():.3f}')

    # ==================================================
    # 1. 按真实标签分析 α_rf 分布差异
    # ==================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    bins = np.linspace(0, 1, 21)
    for ax_i, (mask, label, color) in enumerate([
            (y_failed == 1, f'failed=1 (n={int(y_failed.sum())})', '#d62728'),
            (y_failed == 0, f'passed=0 (n={int((1-y_failed).sum())})', '#1f77b4')]):
        ax = axes[ax_i]
        ax.hist(all_alpha[mask], bins=bins, alpha=0.7, color=color,
                density=True, label=label)
        ax.axvline(all_alpha[mask].mean(), color=color, ls='--', lw=1.5,
                    label=f'mean={all_alpha[mask].mean():.3f}')
        ax.set_xlabel('α_rf')
        ax.set_ylabel('Density')
        ax.set_title(f'{label}')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    axes[2].axis('off')
    plt.suptitle('α_rf 分布: failed vs passed 学生')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'alpha_by_label.png'), bbox_inches='tight')
    plt.close()
    print('[SAVED] figures/alpha_by_label.png')

    # ==================================================
    # 2. 路由偏好 vs 真实标签 的 2x2 分析
    # ==================================================
    # 离散化: 4 个路由区间 x 2 个真实标签
    bins_alpha = np.array([0, 0.3, 0.5, 0.7, 1.0])
    alpha_bin = np.digitize(all_alpha, bins_alpha) - 1
    alpha_bin = np.clip(alpha_bin, 0, 3)
    bin_labels = ['α<0.30\n(强LSTM)', '0.30-0.50\n(略LSTM)',
                   '0.50-0.70\n(略RF)', 'α≥0.70\n(强RF)']

    fig, ax = plt.subplots(figsize=(10, 5))
    cross = np.zeros((4, 2))
    for i in range(4):
        for j in range(2):
            cross[i, j] = ((alpha_bin == i) & (y_failed == j)).sum()
    cross_norm = cross / cross.sum(axis=0, keepdims=True) * 100

    x = np.arange(4)
    w = 0.35
    ax.bar(x - w/2, cross_norm[:, 1], w, color='#d62728',
            label=f'failed=1 (n={int(y_failed.sum())})')
    ax.bar(x + w/2, cross_norm[:, 0], w, color='#1f77b4',
            label=f'passed=0 (n={int((1-y_failed).sum())})')
    for i in range(4):
        ax.text(i - w/2, cross_norm[i, 1] + 1, f'{int(cross[i, 1])}',
                 ha='center', fontsize=9)
        ax.text(i + w/2, cross_norm[i, 0] + 1, f'{int(cross[i, 0])}',
                 ha='center', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_ylabel('本类标签内占比 (%)')
    ax.set_title('α_rf 路由偏好 × 真实标签\n'
                  '(failed 学生更倾向哪条路由? passed 学生更倾向哪条?)')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'routing_cross_label.png'), bbox_inches='tight')
    plt.close()
    print('[SAVED] figures/routing_cross_label.png')

    # 统计检验: failed 学生的 alpha_rf 是否显著不同于 passed
    from scipy.stats import mannwhitneyu, ttest_ind
    u_stat, p_mw = mannwhitneyu(all_alpha[y_failed == 1],
                                 all_alpha[y_failed == 0],
                                 alternative='two-sided')
    t_stat, p_tt = ttest_ind(all_alpha[y_failed == 1],
                              all_alpha[y_failed == 0])
    print(f'\nMann-Whitney U test: U={u_stat:.0f}, p={p_mw:.4f}')
    print(f't-test: t={t_stat:.3f}, p={p_tt:.4f}')

    # ==================================================
    # 3. Permutation importance 对照
    # ==================================================
    print('\n--- Permutation importance (cross-check) ---')
    perm_results = {}
    for fi in range(5):
        tr, va = splits[fi]
        sc = StandardScaler().fit(X7[tr])
        X7_va_s = sc.transform(X7[va]).astype(np.float32)
        gate_va = build_gate_input(rf_oof[va], lstm_oof[va], X7_va_s)

        model = MREFusion(raw_dim=7, fusion_mode='soft')
        model.load_state_dict(torch.load(
            os.path.join(MRE_DIR, 'gate_models', f'soft_fold{fi+1}.pt'),
            map_location='cpu'))
        model.eval()

        with torch.no_grad():
            rf_t = torch.FloatTensor(gate_va[:, 0])
            lstm_t = torch.FloatTensor(gate_va[:, 1])
            raw_t = torch.FloatTensor(gate_va[:, 6:])
            _, baseline_w = model(rf_t, lstm_t, raw_t)
            baseline_alpha = baseline_w[:, 0].numpy()  # (n_va,) alpha_rf
        baseline_var = float(np.var(baseline_alpha))

        per_feat_var = []
        for fi_feat in range(7):
            perm_input = gate_va.copy()
            perm_input[:, 6 + fi_feat] = np.random.RandomState(42 + fi).permutation(
                perm_input[:, 6 + fi_feat])
            with torch.no_grad():
                rf_t = torch.FloatTensor(perm_input[:, 0])
                lstm_t = torch.FloatTensor(perm_input[:, 1])
                raw_t = torch.FloatTensor(perm_input[:, 6:])
                _, perm_w = model(rf_t, lstm_t, raw_t)
                perm_alpha = perm_w[:, 0].numpy()
            per_feat_var.append(float(np.var(perm_alpha)))

        # 重要性 = (baseline_var - perm_var) / baseline_var
        importances = [(baseline_var - v) / (baseline_var + 1e-12) for v in per_feat_var]
        perm_results[fi] = importances
        print(f'  Fold {fi+1}: {[f"{x:.3f}" for x in importances]}')

    perm_mean = np.mean(list(perm_results.values()), axis=0)
    perm_std = np.std(list(perm_results.values()), axis=0)

    # SHAP 值 (mean abs) - 重新加载之前计算的
    with open(os.path.join(MRE_DIR, 'shap_results.json')) as f:
        shap_res = json.load(f)
    shap_means = {d['event']: d['mean_abs_shap']
                   for d in shap_res['global_feature_importance_7d_only']}
    shap_arr = np.array([shap_means[e] for e in EVENT_TYPES])

    # 对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(7)
    axes[0].barh(x, shap_arr, color='#d62728', alpha=0.7, label='SHAP (mean |val|)')
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(EVENT_TYPES)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Importance')
    axes[0].set_title('SHAP mean |value|')
    axes[0].grid(alpha=0.3, axis='x')

    axes[1].barh(x, perm_mean, xerr=perm_std, color='#1f77b4',
                  alpha=0.7, label='Permutation')
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(EVENT_TYPES)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('ΔVar (baseline - permuted)')
    axes[1].set_title('Permutation importance (5-fold mean ± std)')
    axes[1].grid(alpha=0.3, axis='x')

    plt.suptitle('两种可解释性方法的交叉验证: raw_7d 对 α_rf 的重要性')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'shap_vs_permutation.png'), bbox_inches='tight')
    plt.close()
    print('[SAVED] figures/shap_vs_permutation.png')

    # 相关性
    from scipy.stats import spearmanr, pearsonr
    sp_r, sp_p = spearmanr(shap_arr, perm_mean)
    pe_r, pe_p = pearsonr(shap_arr, perm_mean)
    print(f'\nSHAP vs Permutation 相关性:')
    print(f'  Spearman: ρ={sp_r:.3f}, p={sp_p:.4f}')
    print(f'  Pearson:  r={pe_r:.3f}, p={pe_p:.4f}')

    # ==================================================
    # 4. 路由规则文本化 (按真实标签分组)
    # ==================================================
    print('\n' + '=' * 70)
    print('  ROUTING RULES BY LABEL')
    print('=' * 70)
    X7_orig = X7
    for mask, label in [(y_failed == 1, 'failed=1'),
                         (y_failed == 0, 'passed=0')]:
        sub_alpha = all_alpha[mask]
        sub_X = X7_orig[mask]
        sub_X_global = X7_orig
        # 分 4 区间
        for low, high, route_name in [(0, 0.3, '强LSTM'), (0.7, 1.0, '强RF')]:
            sub_mask = (sub_alpha >= low) & (sub_alpha < high)
            if sub_mask.sum() < 5:
                continue
            ratio = sub_X[sub_mask].mean(axis=0) / (sub_X.mean(axis=0) + 1e-6)
            print(f'\n[{label} 走 {route_name}] (n={sub_mask.sum()})')
            for i, e in enumerate(EVENT_TYPES):
                bar = '+' * int(max(0, ratio[i] - 1) * 20)
                neg_bar = '-' * int(max(0, 1 - ratio[i]) * 20)
                sign = '+' if ratio[i] > 1 else '-'
                print(f'    {e:<14} {ratio[i]:.2f}×  {sign}{bar}{neg_bar}')

    # 保存扩展结果
    extra = {
        'mann_whitney': {'U': float(u_stat), 'p': float(p_mw)},
        't_test': {'t': float(t_stat), 'p': float(p_tt)},
        'permutation_importance_5fold_mean': {
            EVENT_TYPES[i]: float(perm_mean[i]) for i in range(7)
        },
        'permutation_importance_5fold_std': {
            EVENT_TYPES[i]: float(perm_std[i]) for i in range(7)
        },
        'shap_vs_permutation': {
            'spearman_rho': float(sp_r), 'spearman_p': float(sp_p),
            'pearson_r': float(pe_r), 'pearson_p': float(pe_p),
        },
        'alpha_rf_by_label': {
            'failed_mean': float(all_alpha[y_failed == 1].mean()),
            'failed_std': float(all_alpha[y_failed == 1].std()),
            'passed_mean': float(all_alpha[y_failed == 0].mean()),
            'passed_std': float(all_alpha[y_failed == 0].std()),
        },
    }
    with open(os.path.join(MRE_DIR, 'shap_extra_results.json'), 'w') as f:
        json.dump(extra, f, indent=2, ensure_ascii=False)
    print(f'\n[SAVED] shap_extra_results.json')


if __name__ == '__main__':
    main()