"""
按类别消融实验有效性可视化
3 张图:
  1. 类别重要性 vs 类别删除影响 (双重条形图)
  2. 跨模型 ΔF1 对比 (grouped bar)
  3. 雷达图: 4 个类别 × 3 个模型的 F1 衰减
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
fm = font_manager.fontManager
fm.findfont('Noto Sans CJK SC', rebuild_if_missing=True)
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 数据 - 来自 ablation_compare_v2.py 的结果
CATS = ['Cat1\n事件统计\n(28d)', 'Cat2\n行为轨迹\n(10d)', 'Cat3\n情绪复合\n(6d)', 'Cat4\n元信息\n(2d)']

# 类别重要性百分比 (来自 feature_importance.json)
importance_pct = [56.63, 9.92, 23.89, 9.56]

# ΔF1@best vs Full 46d (负数表示删除后下降)
delta_rf = [-0.0322, -0.0006, -0.0261, -0.0106]
delta_lstm = [-0.0300, +0.0048, -0.0267, -0.0049]
delta_bilstm = [-0.0155, +0.0266, -0.0228, -0.0005]

CAT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def plot_double_bar():
    """图 1: 类别重要性 + 删除后 ΔF1 (双轴)"""
    fig, ax1 = plt.subplots(figsize=(13, 7))

    x = np.arange(len(CATS))
    width = 0.35

    # 左轴: 重要性百分比
    bars1 = ax1.bar(x - width / 2, importance_pct, width,
                     color=CAT_COLORS, alpha=0.7, edgecolor='black', label='重要性占比 (%)')
    ax1.set_ylabel('重要性占比 (%)', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(CATS, fontsize=11)
    ax1.set_ylim(0, 70)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    for bar, v in zip(bars1, importance_pct):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')

    # 右轴: RF 删除 ΔF1
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, delta_rf, width,
                     color=CAT_COLORS, alpha=0.4, edgecolor='black', hatch='//',
                     label='RF 删除后 ΔF1@best')
    ax2.set_ylabel('RF 删除后 ΔF1@best (负=下降)', fontsize=13)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_ylim(-0.05, 0.05)

    for bar, v in zip(bars2, delta_rf):
        offset = 0.002 if v >= 0 else -0.002
        va = 'bottom' if v >= 0 else 'top'
        ax2.text(bar.get_x() + bar.get_width() / 2, v + offset,
                 f'{v:+.4f}', ha='center', va=va, fontsize=10, fontweight='bold')

    plt.title('特征类别: 重要性占比 vs 删除后性能下降 (RF 模型)', fontsize=14, fontweight='bold')
    fig.legend(loc='upper right', bbox_to_anchor=(0.88, 0.92), fontsize=11)
    plt.tight_layout()
    plt.savefig('outputs/ablation/figures/category_effectiveness.png', dpi=120, bbox_inches='tight')
    plt.close()
    print('✓ Figure 1: category_effectiveness.png')


def plot_cross_model_delta():
    """图 2: 跨模型 ΔF1 对比 (grouped bar)"""
    fig, ax = plt.subplots(figsize=(13, 7))

    x = np.arange(len(CATS))
    width = 0.27

    bars1 = ax.bar(x - width, delta_rf, width, label='RF', color='#1f77b4', edgecolor='black')
    bars2 = ax.bar(x, delta_lstm, width, label='LSTM', color='#2ca02c', edgecolor='black')
    bars3 = ax.bar(x + width, delta_bilstm, width, label='BiLSTM', color='#d62728', edgecolor='black')

    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(CATS, fontsize=11)
    ax.set_ylabel('删除该类别后 ΔF1@best', fontsize=13)
    ax.set_title('跨模型消融影响 (3 模型 × 4 类别, 负=下降, 正=提升)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            v = bar.get_height()
            offset = 0.002 if v >= 0 else -0.002
            va = 'bottom' if v >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                    f'{v:+.3f}', ha='center', va=va, fontsize=8)

    plt.tight_layout()
    plt.savefig('outputs/ablation/figures/cross_model_delta.png', dpi=120, bbox_inches='tight')
    plt.close()
    print('✓ Figure 2: cross_model_delta.png')


def plot_radar():
    """图 3: 雷达图 - 类别有效性"""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # 角度 (4 个类别)
    angles = np.linspace(0, 2 * np.pi, len(CATS), endpoint=False).tolist()
    angles += angles[:1]

    # 数据: 类别删除造成的 F1 损失 (绝对值, 越大说明类别越重要)
    # 取 RF / LSTM / BiLSTM 平均
    avg_impact = [(abs(d_r) + abs(d_l) + abs(d_b)) / 3
                  for d_r, d_l, d_b in zip(delta_rf, delta_lstm, delta_bilstm)]

    # 三模型单独
    rf_impact = [abs(d) for d in delta_rf]
    lstm_impact = [abs(d) for d in delta_lstm]
    bilstm_impact = [abs(d) for d in delta_bilstm]

    rf_impact += rf_impact[:1]
    lstm_impact += lstm_impact[:1]
    bilstm_impact += bilstm_impact[:1]
    avg_impact += avg_impact[:1]

    ax.plot(angles, rf_impact, 'o-', linewidth=2, label='RF', color='#1f77b4')
    ax.fill(angles, rf_impact, alpha=0.15, color='#1f77b4')
    ax.plot(angles, lstm_impact, 'o-', linewidth=2, label='LSTM', color='#2ca02c')
    ax.fill(angles, lstm_impact, alpha=0.15, color='#2ca02c')
    ax.plot(angles, bilstm_impact, 'o-', linewidth=2, label='BiLSTM', color='#d62728')
    ax.fill(angles, bilstm_impact, alpha=0.15, color='#d62728')
    ax.plot(angles, avg_impact, 'o--', linewidth=3, label='3 模型平均', color='black')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(CATS, fontsize=11)
    ax.set_ylim(0, 0.04)
    ax.set_yticks([0.01, 0.02, 0.03])
    ax.set_yticklabels(['0.01', '0.02', '0.03'], fontsize=9)
    ax.set_title('类别有效性雷达图 (面积越大 = 删除后影响越大 = 类别越重要)',
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=11)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/ablation/figures/category_radar.png', dpi=120, bbox_inches='tight')
    plt.close()
    print('✓ Figure 3: category_radar.png')


def plot_summary_table():
    """图 4: 综合分类摘要 (大表格 + 关键指标)"""
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.axis('off')

    # 表头
    columns = ['类别', '维度', '特征示例', '理论依据', '重要性%', 'ΔF1 (RF/LSTM/BiLSTM)', '有效性']
    rows = [
        ['Cat1\n事件统计', '28d', 'submit_entropy\ntext_paste_entropy\ntext_remove_mean',
         '信息论 (Shannon Entropy)\n+ 多维统计量',
         '56.6%', '−0.032 / −0.030 / −0.016', '🟢 必要'],
        ['Cat2\n行为轨迹', '10d', 'improvement\nconsistency\nmean_interval',
         '时序分析 (Trend + Cadence)',
         '9.9%', '−0.001 / +0.005 / +0.027', '🔴 可删 (噪声)'],
        ['Cat3\n情绪复合', '6d', 'focus_ratio_mean\ndelete_ratio_std\nedit_ratio_mean',
         '比率特征 (跨事件交互)\n业务: 行为意图',
         '23.9%', '−0.026 / −0.027 / −0.023', '🟢 关键 (创新)'],
        ['Cat4\n元信息', '2d', 'total_events\nnum_problems',
         '归一化基数\n(Piech et al. 2015)',
         '9.6%', '−0.011 / −0.005 / −0.001', '🟡 仅 total_events'],
    ]

    table = ax.table(cellText=rows, colLabels=columns,
                     loc='center', cellLoc='center',
                     colColours=['#4472C4'] * len(columns))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # 表头样式
    for i in range(len(columns)):
        cell = table[(0, i)]
        cell.set_text_props(weight='bold', color='white', fontsize=11)
        cell.set_height(0.12)

    # 行颜色
    for i, color in enumerate(['#E8F0FE', '#FFF4E5', '#E6F4EA', '#FCE8E6']):
        for j in range(len(columns)):
            cell = table[(i + 1, j)]
            cell.set_facecolor(color)
            cell.set_height(0.15)

    plt.title('46d 特征分类摘要表 (含创新性 + 有效性评估)', fontsize=15, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('outputs/ablation/figures/summary_table.png', dpi=120, bbox_inches='tight')
    plt.close()
    print('✓ Figure 4: summary_table.png')


if __name__ == '__main__':
    os.makedirs('outputs/ablation/figures', exist_ok=True)
    plot_double_bar()
    plot_cross_model_delta()
    plot_radar()
    plot_summary_table()
    print('\n✅ 4 张可视化图全部生成完成')