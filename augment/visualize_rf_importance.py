"""
RF 特征重要性可视化
生成 4 张图:
  1. Top 20 特征重要性横向条形图
  2. 4 类特征贡献饼图
  3. 4 类特征贡献对比柱状图 (总和 + 平均)
  4. 每类 Top 5 特征对比 (堆叠条形图)
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无头模式
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体 - 强制使用 Noto Sans CJK SC
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 特征名 (与 rf_feature_importance.py 一致)
FEATURE_NAMES = (
    [f'{et}_{stat}' for et in ['text_insert', 'text_remove', 'text_paste',
                                'focus_gained', 'focus_lost', 'run', 'submit']
     for stat in ['mean', 'std', 'cv', 'entropy']] +
    ['improvement', 'consistency', 'trend',
     'mean_interval', 'std_interval', 'min_interval',
     'max_interval', 'duration_per_event',
     'median_interval', 'iqr_interval'] +
    ['edit_ratio_mean', 'edit_ratio_std',
     'delete_ratio_mean', 'delete_ratio_std',
     'focus_ratio_mean', 'focus_ratio_std'] +
    ['num_problems', 'total_events']
)

CAT_RANGES = {
    'Cat1\n(事件统计, 28d)': (0, 28),
    'Cat2\n(行为轨迹, 10d)': (28, 38),
    'Cat3\n(情绪复合, 6d)': (38, 44),
    'Cat4\n(元信息, 2d)': (44, 46),
}
CAT_COLORS = {
    'Cat1\n(事件统计, 28d)': '#1f77b4',
    'Cat2\n(行为轨迹, 10d)': '#ff7f0e',
    'Cat3\n(情绪复合, 6d)': '#2ca02c',
    'Cat4\n(元信息, 2d)': '#d62728',
}


def find_cat(idx):
    for cname, (s, e) in CAT_RANGES.items():
        if s <= idx < e:
            return cname
    return 'Unknown'


def main():
    # 加载已有的特征重要性数据
    with open('outputs/ablation/feature_importance.json') as f:
        data = json.load(f)

    importances = np.array([fe['importance'] for fe in data['feature_importances']])
    # 注意: data['feature_importances'] 是按重要性降序的
    # 我们需要按原始 idx 顺序重建
    importance_by_idx = np.zeros(46)
    for fe in data['feature_importances']:
        importance_by_idx[fe['idx']] = fe['importance']

    sorted_idx = np.argsort(importance_by_idx)[::-1]
    os.makedirs('outputs/ablation/figures', exist_ok=True)

    # ============ 图 1: Top 20 特征重要性 ============
    fig, ax = plt.subplots(figsize=(12, 8))
    top_n = 20
    top_idx = sorted_idx[:top_n]
    top_names = [FEATURE_NAMES[i] for i in top_idx]
    top_imps = importance_by_idx[top_idx]
    top_cats = [find_cat(i) for i in top_idx]
    colors = [CAT_COLORS[c] for c in top_cats]

    y_pos = np.arange(top_n)
    bars = ax.barh(y_pos, top_imps, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (Gini)', fontsize=12)
    ax.set_title(f'Top {top_n} Features by RF Importance (5-Fold CV Average)', fontsize=14, fontweight='bold')

    # 添加数值标签
    for i, (bar, imp) in enumerate(zip(bars, top_imps)):
        ax.text(imp + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{imp:.4f}', va='center', fontsize=9)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, edgecolor='black', label=cat.split('\n')[0])
                       for cat, color in CAT_COLORS.items()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('outputs/ablation/figures/top20_importance.png', dpi=120, bbox_inches='tight')
    plt.close()
    print('✓ Figure 1 saved: top20_importance.png')

    # ============ 图 2: 4 类贡献饼图 ============
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    cat_sums = {}
    for cat_name, (s, e) in CAT_RANGES.items():
        cat_sums[cat_name] = float(importance_by_idx[s:e].sum())

    cats = list(cat_sums.keys())
    sizes = list(cat_sums.values())
    colors_list = [CAT_COLORS[c] for c in cats]

    # 左: 占比饼图
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=cats, colors=colors_list,
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 11}, pctdistance=0.75,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    ax1.set_title('Importance Distribution\nby Feature Category', fontsize=13, fontweight='bold')

    # 右: 平均每维信息密度
    cat_density = {cat: cat_sums[cat] / (CAT_RANGES[cat][1] - CAT_RANGES[cat][0])
                   for cat in cats}
    densities = [cat_density[c] for c in cats]
    bars = ax2.bar(range(len(cats)), densities, color=colors_list, edgecolor='black')
    ax2.set_xticks(range(len(cats)))
    ax2.set_xticklabels([c.split('\n')[0] for c in cats], fontsize=11)
    ax2.set_ylabel('Average Importance per Dimension', fontsize=12)
    ax2.set_title('Information Density\n(Importance / #Dimensions)', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, d in zip(bars, densities):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f'{d:.4f}', ha='center', fontsize=11, fontweight='bold')

    plt.suptitle('RF Feature Importance Analysis - Category Summary', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('outputs/ablation/figures/category_distribution.png', dpi=120, bbox_inches='tight')
    plt.close()
    print('✓ Figure 2 saved: category_distribution.png')

    # ============ 图 3: 每类 Top 5 特征 (网格图) ============
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax, (cat_name, (s, e)) in zip(axes, CAT_RANGES.items()):
        cat_imp = importance_by_idx[s:e]
        # 该类内的 Top 5 (或全部, 如果少于5个)
        top_k = min(5, len(cat_imp))
        local_idx = np.argsort(cat_imp)[::-1][:top_k]
        names = [FEATURE_NAMES[s + i] for i in local_idx]
        imps = cat_imp[local_idx]
        colors_local = [CAT_COLORS[cat_name]] * top_k

        bars = ax.barh(range(top_k), imps, color=colors_local, edgecolor='black')
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(names, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=10)
        ax.set_title(f'{cat_name.split(chr(10))[0]}\nTop {top_k} Features',
                     fontsize=11, fontweight='bold')
        for i, (bar, imp) in enumerate(zip(bars, imps)):
            ax.text(imp + 0.0005, bar.get_y() + bar.get_height() / 2,
                    f'{imp:.4f}', va='center', fontsize=9)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.suptitle('Top Features within Each Category', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/ablation/figures/per_category_top.png', dpi=120, bbox_inches='tight')
    plt.close()
    print('✓ Figure 3 saved: per_category_top.png')

    # ============ 图 4: 全部 46 维特征重要性 (带类别着色) ============
    fig, ax = plt.subplots(figsize=(18, 7))
    x_pos = np.arange(46)
    # 按类别分段着色
    colors_full = []
    for idx in range(46):
        cat = find_cat(idx)
        colors_full.append(CAT_COLORS[cat])
    # 按重要性排序后再画
    sorted_x = sorted_idx  # 已经是降序
    sorted_imps = importance_by_idx[sorted_idx]
    sorted_colors = [colors_full[i] for i in sorted_idx]

    ax.bar(x_pos, sorted_imps, color=sorted_colors, edgecolor='black', linewidth=0.3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([FEATURE_NAMES[i] for i in sorted_idx], rotation=90, fontsize=7)
    ax.set_ylabel('Feature Importance', fontsize=12)
    ax.set_title('All 46 Features Ranked by RF Importance (colored by category)', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    legend_elements = [Patch(facecolor=color, edgecolor='black', label=cat.split('\n')[0])
                       for cat, color in CAT_COLORS.items()]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig('outputs/ablation/figures/all46_ranked.png', dpi=120, bbox_inches='tight')
    plt.close()
    print('✓ Figure 4 saved: all46_ranked.png')

    # ============ 图 5: 累计重要性曲线 ============
    fig, ax = plt.subplots(figsize=(12, 6))
    cumulative = np.cumsum(importance_by_idx[sorted_idx])
    cumulative_pct = cumulative / cumulative[-1] * 100
    ax.plot(range(1, 47), cumulative_pct, 'b-o', markersize=4, linewidth=2)
    ax.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% threshold')
    ax.axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='95% threshold')
    # 找达到 80% 和 95% 的特征数
    n_80 = int(np.searchsorted(cumulative_pct, 80)) + 1
    n_95 = int(np.searchsorted(cumulative_pct, 95)) + 1
    ax.axvline(x=n_80, color='red', linestyle=':', alpha=0.5)
    ax.axvline(x=n_95, color='orange', linestyle=':', alpha=0.5)
    ax.text(n_80 + 0.5, 50, f'{n_80} features\ncover 80%', color='red', fontsize=10)
    ax.text(n_95 + 0.5, 30, f'{n_95} features\ncover 95%', color='orange', fontsize=10)
    ax.set_xlabel('Number of Top Features', fontsize=12)
    ax.set_ylabel('Cumulative Importance (%)', fontsize=12)
    ax.set_title('Cumulative Feature Importance Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(0, 46)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig('outputs/ablation/figures/cumulative_importance.png', dpi=120, bbox_inches='tight')
    plt.close()
    print('✓ Figure 5 saved: cumulative_importance.png')

    print('\n所有图片已保存到 outputs/ablation/figures/')


if __name__ == '__main__':
    main()