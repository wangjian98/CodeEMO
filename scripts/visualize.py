#!/usr/bin/env python3
"""
统一可视化脚本 - 生成模型对比图表

用法:
  python scripts/visualize.py                    # 可视化所有结果
  python scripts/visualize.py --output outputs/comparison.png
"""

import sys
import os
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def load_all_results(output_dir='outputs'):
    """加载所有模型结果"""
    results = {}
    output_path = Path(output_dir)

    model_dirs = ['rf', 'lstm', 'bilstm', 'transformer', 'mamba']

    for model_name in model_dirs:
        results_file = output_path / model_name / 'results.json'
        if results_file.exists():
            with open(results_file) as f:
                results[model_name] = json.load(f)

    return results


def create_comparison_plot(results, output_path):
    """生成对比柱状图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib 未安装，跳过可视化")
        return

    models = list(results.keys())
    if not models:
        print("没有找到结果数据")
        return

    metrics = ['accuracy', 'f1', 'auc']
    metric_labels = ['Accuracy', 'F1 Score', 'AUC']
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]

        means = [results[m].get(f'{metric}_mean', 0) for m in models]
        stds = [results[m].get(f'{metric}_std', 0) for m in models]

        x = np.arange(len(models))
        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=[colors[i % len(colors)] for i in range(len(models))],
                      alpha=0.85, edgecolor='black', linewidth=0.5)

        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in models], rotation=30, ha='right')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

        # 在柱子上标注数值
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('CodeEMO 模型对比 - 5折交叉验证', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"对比图已保存至: {output_path}")


def create_radar_plot(results, output_path):
    """生成雷达图对比"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    models = list(results.keys())
    if not models:
        return

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']

    for i, model in enumerate(models):
        values = [results[model].get(f'{m}_mean', 0) for m in metrics]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model.upper(),
                color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.set_title('模型性能雷达图', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"雷达图已保存至: {output_path}")


def print_ascii_table(results):
    """打印ASCII对比表格"""
    print(f"\n{'='*80}")
    print("  CodeEMO 模型对比结果")
    print(f"{'='*80}")

    header = f"  {'Model':<15} {'Accuracy':<18} {'Precision':<18} {'Recall':<18} {'F1':<18} {'AUC':<15}"
    print(header)
    print(f"  {'-'*95}")

    for name, res in results.items():
        acc = f"{res.get('accuracy_mean', 0):.4f}±{res.get('accuracy_std', 0):.4f}"
        prec = f"{res.get('precision_mean', 0):.4f}±{res.get('precision_std', 0):.4f}"
        rec = f"{res.get('recall_mean', 0):.4f}±{res.get('recall_std', 0):.4f}"
        f1 = f"{res.get('f1_mean', 0):.4f}±{res.get('f1_std', 0):.4f}"
        auc = f"{res.get('auc_mean', 0):.4f}±{res.get('auc_std', 0):.4f}"

        print(f"  {name:<15} {acc:<18} {prec:<18} {rec:<18} {f1:<18} {auc:<15}")

    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='CodeEMO 可视化工具')
    parser.add_argument('--input-dir', type=str, default='outputs',
                        help='结果输入目录')
    parser.add_argument('--output', type=str, default='outputs/comparison.png',
                        help='对比图输出路径')
    parser.add_argument('--radar', type=str, default='outputs/radar.png',
                        help='雷达图输出路径')

    args = parser.parse_args()

    results = load_all_results(args.input_dir)

    if not results:
        print(f"未找到结果文件，请先运行实验")
        sys.exit(1)

    print_ascii_table(results)
    create_comparison_plot(results, args.output)
    create_radar_plot(results, args.radar)


if __name__ == '__main__':
    main()
