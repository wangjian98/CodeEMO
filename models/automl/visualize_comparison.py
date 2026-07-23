"""
生成 AutoML baseline 对比的可视化图表，用于论文 Figure 5。

输出:
  1. 3-way F1/AUC bar chart (Handcrafted vs TSFRESH minimal vs TSFRESH efficient vs Featuretools)
  2. Category importance heat map (Cat1/2/3/4 × RF/LSTM/BiLSTM ablation ΔF1)
  3. Combined figure (paper-ready)

用法:
    python models/automl/visualize_comparison.py
    python models/automl/visualize_comparison.py --output-dir outputs/automl/figures
"""
import os
import sys
import json
import argparse

import matplotlib
matplotlib.use('Agg')  # 无 display 环境
import matplotlib.pyplot as plt
import numpy as np

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def load_eval(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_metric_comparison_4way(eval_paths, output_dir, output_basename='fig5_automl_comparison'):
    """绘制 4-way 对比柱状图：Handcrafted + TSFRESH (minimal/efficient) + Featuretools

    Args:
        eval_paths: dict of {label: path}
            e.g. {
                'Handcrafted 46d': 'outputs/automl/evaluation.json',
                'TSFRESH (minimal)': 'outputs/automl/evaluation.json',
                'TSFRESH (efficient)': 'outputs/automl_efficient/evaluation.json',
                'Featuretools': 'outputs/automl_featuretools/evaluation.json'
            }
        output_dir: 输出目录
        output_basename: 文件名前缀
    """
    os.makedirs(output_dir, exist_ok=True)

    # 提取数据
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    labels = list(eval_paths.keys())
    means = {}
    stds = {}

    for label, path in eval_paths.items():
        eval_data = load_eval(path)
        # 从 evaluation.json 找 handcrafted + 各 AutoML 数据
        if label == 'Handcrafted 46d':
            means[label] = {m: eval_data['handcrafted_46d'][f'{m}_mean'] for m in metrics}
            stds[label] = {m: eval_data['handcrafted_46d'][f'{m}_std'] for m in metrics}
        elif label.startswith('TSFRESH'):
            means[label] = {m: eval_data['tsfresh'][f'{m}_mean'] for m in metrics}
            stds[label] = {m: eval_data['tsfresh'][f'{m}_std'] for m in metrics}
        elif label == 'Featuretools':
            means[label] = {m: eval_data['featuretools'][f'{m}_mean'] for m in metrics}
            stds[label] = {m: eval_data['featuretools'][f'{m}_std'] for m in metrics}
        elif label == 'autofeat':
            means[label] = {m: eval_data['autofeat'][f'{m}_mean'] for m in metrics}
            stds[label] = {m: eval_data['autofeat'][f'{m}_std'] for m in metrics}

    # 绘制 5 个子图（每个指标一个）
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

    colors = ['#2E7D32', '#1976D2', '#F57C00', '#C62828']  # green, blue, orange, red

    for i, metric in enumerate(metrics):
        ax = axes[i]
        x = np.arange(len(labels))
        vals = [means[l][metric] for l in labels]
        errs = [stds[l][metric] for l in labels]

        bars = ax.bar(x, vals, yerr=errs, capsize=4,
                       color=colors[:len(labels)], alpha=0.85,
                       edgecolor='black', linewidth=0.8)

        # 标注数值
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        ax.set_title(metric.upper(), fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.upper(), fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.suptitle('Figure 5: Handcrafted 46-dim vs AutoML Baselines (5-fold CV, mean ± std)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f'{output_basename}.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved 4-way comparison figure: {out_path}")

    return out_path


def plot_f1_only_bars(eval_paths, output_dir):
    """简化版：仅 F1 指标的横向 bar chart（更适合论文主图）"""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    labels = list(eval_paths.keys())
    f1_means = []
    f1_stds = []
    for label, path in eval_paths.items():
        eval_data = load_eval(path)
        if label == 'Handcrafted 46d':
            f1_means.append(eval_data['handcrafted_46d']['f1_mean'])
            f1_stds.append(eval_data['handcrafted_46d']['f1_std'])
        elif label.startswith('TSFRESH'):
            f1_means.append(eval_data['tsfresh']['f1_mean'])
            f1_stds.append(eval_data['tsfresh']['f1_std'])
        elif label == 'Featuretools':
            f1_means.append(eval_data['featuretools']['f1_mean'])
            f1_stds.append(eval_data['featuretools']['f1_std'])
        elif label == 'autofeat':
            f1_means.append(eval_data['autofeat']['f1_mean'])
            f1_stds.append(eval_data['autofeat']['f1_std'])

    colors = ['#2E7D32', '#1976D2', '#F57C00', '#C62828', '#7B1FA2'][:len(labels)]
    y = np.arange(len(labels))
    bars = ax.barh(y, f1_means, xerr=f1_stds, capsize=4,
                    color=colors, alpha=0.85,
                    edgecolor='black', linewidth=0.8)

    for bar, mean, std in zip(bars, f1_means, f1_stds):
        ax.text(mean + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{mean:.4f} ± {std:.4f}', va='center', fontsize=10)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('F1 Score (5-fold CV)', fontsize=11)
    ax.set_title('Figure 5: F1 Score — Handcrafted vs AutoML Baselines',
                  fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()  # Handcrafted 在最上面
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'fig5_f1_bars.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved F1 bars: {out_path}")
    return out_path


def plot_category_ablation_heatmap(output_dir):
    """Cat1/2/3/4 × RF/LSTM/BiLSTM 的 ΔF1 heat map

    数据来自 paper-draft.md Table 3 (手工特征 ablation)。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 数据来自 paper-draft Table 3 (Variant B/C/D/E)
    categories = ['Cat1 (Events)', 'Cat2 (Trajectory)', 'Cat3 (Ratio)', 'Cat4 (Meta)']
    models = ['RF', 'LSTM', 'BiLSTM']

    # 手工 46 维 → 删除各类别的 ΔF1
    # B. -Cat1: RF -0.0322, LSTM -0.0300, BiLSTM -0.0155
    # C. -Cat2: RF -0.0006, LSTM +0.0048, BiLSTM +0.0266
    # D. -Cat3: RF -0.0261, LSTM -0.0267, BiLSTM -0.0228
    # E. -Cat4: RF -0.0106, LSTM -0.0049, BiLSTM -0.0005
    delta_f1 = np.array([
        [-0.0322, -0.0300, -0.0155],  # Cat1
        [-0.0006, +0.0048, +0.0266],  # Cat2
        [-0.0261, -0.0267, -0.0228],  # Cat3
        [-0.0106, -0.0049, -0.0005],  # Cat4
    ])

    fig, ax = plt.subplots(figsize=(7, 5))

    # 红-白-绿色板（负-中-正）
    vmax = max(abs(delta_f1.min()), abs(delta_f1.max()))
    im = ax.imshow(delta_f1, cmap='RdYlGn', aspect='auto',
                   vmin=-vmax, vmax=vmax)

    # 标注数值
    for i in range(len(categories)):
        for j in range(len(models)):
            val = delta_f1[i, j]
            color = 'white' if abs(val) > 0.015 else 'black'
            ax.text(j, i, f'{val:+.3f}', ha='center', va='center',
                    color=color, fontweight='bold', fontsize=11)

    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, fontsize=11)
    ax.set_yticks(np.arange(len(categories)))
    ax.set_yticklabels(categories, fontsize=11)
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Deleted Category', fontsize=11)
    ax.set_title('Figure 4 (paper): Ablation Impact ΔF1 on Handcrafted Features\n'
                 '(red = harmful to remove, green = helpful to remove)',
                 fontsize=11, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label('ΔF1 vs Full 46-dim', fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'fig4_ablation_heatmap.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved ablation heatmap: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description='生成 AutoML baseline 对比可视化')
    parser.add_argument('--output-dir', type=str,
                        default='outputs/automl/figures',
                        help='图片输出目录')
    parser.add_argument('--hand-path', type=str,
                        default='outputs/automl/evaluation.json',
                        help='Handcrafted 46d 评估结果路径')
    parser.add_argument('--tsfresh-min-path', type=str,
                        default='outputs/automl/evaluation.json',
                        help='TSFRESH (minimal) 评估结果路径')
    parser.add_argument('--tsfresh-eff-path', type=str,
                        default='outputs/automl_efficient/evaluation.json',
                        help='TSFRESH (efficient) 评估结果路径')
    parser.add_argument('--featuretools-path', type=str,
                        default='outputs/automl_featuretools/evaluation.json',
                        help='Featuretools 评估结果路径')
    parser.add_argument('--autofeat-path', type=str,
                        default='outputs/automl_autofeat/evaluation.json',
                        help='autofeat 评估结果路径')
    args = parser.parse_args()

    # 检查文件存在
    eval_paths = {
        'Handcrafted 46d': args.hand_path,
        'TSFRESH (minimal)': args.tsfresh_min_path,
        'TSFRESH (efficient)': args.tsfresh_eff_path,
    }

    # 如果 Featuretools 存在，加入
    ft_label = 'Featuretools'
    if os.path.exists(args.featuretools_path):
        eval_paths[ft_label] = args.featuretools_path
        print(f"Found Featuretools results, including in plots")
    else:
        print(f"Featuretools results not found at {args.featuretools_path}, skipping")

    # 如果 autofeat 存在，加入
    af_label = 'autofeat'
    if os.path.exists(args.autofeat_path):
        eval_paths[af_label] = args.autofeat_path
        print(f"Found autofeat results, including in plots")
    else:
        print(f"autofeat results not found at {args.autofeat_path}, skipping")

    print("=" * 70)
    print("  Visualization: AutoML baseline comparison")
    print("=" * 70)

    print(f"\n[1/3] Plotting {len(eval_paths)}-way comparison ...")
    plot_metric_comparison_4way(eval_paths, args.output_dir)

    print(f"\n[2/3] Plotting F1-only bars ...")
    plot_f1_only_bars(eval_paths, args.output_dir)

    print(f"\n[3/3] Plotting category ablation heatmap ...")
    plot_category_ablation_heatmap(args.output_dir)

    print(f"\n所有图表已保存至: {args.output_dir}")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith(('.png', '.pdf')):
            full = os.path.join(args.output_dir, f)
            size_kb = os.path.getsize(full) / 1024
            print(f"  {f} ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()