#!/usr/bin/env python3
"""
CodeEMO 统一运行入口

用法:
  python main.py --model rf           # 运行随机森林
  python main.py --model lstm         # 运行LSTM
  python main.py --model bilstm       # 运行BiLSTM
  python main.py --model transformer  # 运行Transformer
  python main.py --model mamba        # 运行Mamba (CPU版)
  python main.py --model mamba_gpu    # 运行Mamba (GPU版)
  python main.py --model all          # 运行所有模型

选项:
  --model       模型选择 (默认: all)
  --folds       交叉验证折数 (默认: 5)
  --output-dir  输出目录 (默认: outputs)
"""

import sys
import os
import json
import argparse
import subprocess
import time
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.resolve()


def run_model(model_name, folds=5, output_dir='outputs'):
    """运行单个模型"""
    print(f"\n{'#'*60}")
    print(f"# 运行模型: {model_name}")
    print(f"{'#'*60}\n")

    start_time = time.time()

    if model_name == 'mamba':
        script = PROJECT_ROOT / 'models' / 'mamba' / 'train_cpu.py'
    elif model_name == 'mamba_gpu':
        script = PROJECT_ROOT / 'models' / 'mamba' / 'train_gpu.py'
    else:
        script = PROJECT_ROOT / 'models' / model_name / 'train.py'

    if not script.exists():
        print(f"错误: 找不到脚本 {script}")
        return None

    cmd = [
        sys.executable, str(script),
        '--folds', str(folds),
        '--output-dir', str(PROJECT_ROOT / output_dir / model_name.replace('_gpu', ''))
    ]

    print(f"命令: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    elapsed = time.time() - start_time

    # 读取结果
    model_dir = model_name.replace('_gpu', '')
    results_file = PROJECT_ROOT / output_dir / model_dir / 'results.json'

    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        results['_elapsed_seconds'] = elapsed
        return results
    else:
        print(f"警告: 结果文件未找到 {results_file}")
        return None


def generate_comparison_table(all_results):
    """生成ASCII对比表格"""
    print(f"\n{'='*80}")
    print("  CodeEMO 全模型对比结果")
    print(f"{'='*80}")

    header = f"  {'Model':<15} {'Accuracy':<18} {'F1':<18} {'AUC':<18}"
    print(header)
    print(f"  {'-'*65}")

    csv_lines = ['model,accuracy_mean,accuracy_std,f1_mean,f1_std,auc_mean,auc_std']

    for name, results in all_results.items():
        if results is None:
            continue

        acc_mean = results.get('accuracy_mean', 0)
        acc_std = results.get('accuracy_std', 0)
        f1_mean = results.get('f1_mean', 0)
        f1_std = results.get('f1_std', 0)
        auc_mean = results.get('auc_mean', 0)
        auc_std = results.get('auc_std', 0)

        row = f"  {name:<15} {acc_mean:.4f}±{acc_std:.4f}   {f1_mean:.4f}±{f1_std:.4f}   {auc_mean:.4f}±{auc_std:.4f}"
        print(row)

        csv_lines.append(
            f"{name},{acc_mean:.4f},{acc_std:.4f},{f1_mean:.4f},{f1_std:.4f},{auc_mean:.4f},{auc_std:.4f}"
        )

    print(f"{'='*80}")

    return '\n'.join(csv_lines)


def main():
    parser = argparse.ArgumentParser(description='CodeEMO 统一运行入口')
    parser.add_argument('--model', type=str, default='all',
                        choices=['rf', 'lstm', 'bilstm', 'transformer', 'mamba', 'mamba_gpu', 'all'],
                        help='选择模型 (默认: all)')
    parser.add_argument('--folds', type=int, default=5,
                        help='交叉验证折数 (默认: 5)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='输出目录 (默认: outputs)')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  CodeEMO: 融合编程行为情绪表征的学业早期风险预测")
    print(f"{'='*60}")
    print(f"  模型: {args.model}")
    print(f"  折数: {args.folds}")
    print(f"  输出: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.model == 'all':
        model_list = ['rf', 'lstm', 'bilstm', 'transformer', 'mamba']
    else:
        model_list = [args.model]

    all_results = {}
    total_start = time.time()

    for model_name in model_list:
        print(f"\n{'='*60}")
        print(f"  [{model_list.index(model_name)+1}/{len(model_list)}] {model_name}")
        print(f"{'='*60}")
        results = run_model(model_name, args.folds, args.output_dir)
        all_results[model_name] = results

    total_elapsed = time.time() - total_start

    # 生成对比表格
    if len(all_results) > 1:
        csv_content = generate_comparison_table(all_results)

        csv_path = Path(args.output_dir) / 'comparison.csv'
        with open(csv_path, 'w') as f:
            f.write(csv_content + '\n')
        print(f"\n对比结果已保存至: {csv_path}")

        # 生成分析报告
        analysis_path = Path(args.output_dir) / 'analysis.md'
        generate_analysis_md(all_results, analysis_path, total_elapsed)

    # 打印总时间
    print(f"\n总耗时: {total_elapsed/60:.1f} 分钟")

    return all_results


def generate_analysis_md(all_results, path, total_elapsed):
    """生成Markdown分析报告"""
    lines = [
        "# CodeEMO 实验结果分析\n",
        f"## 总览\n",
        f"- 总耗时: {total_elapsed/60:.1f} 分钟\n",
        f"- 模型数量: {len([r for r in all_results.values() if r is not None])}\n",
        f"\n## 结果对比\n",
        f"| 模型 | Accuracy | F1 | AUC | 耗时(秒) |",
        f"|------|----------|----|-----|----------|",
    ]

    for name, results in all_results.items():
        if results is None:
            lines.append(f"| {name} | N/A | N/A | N/A | N/A |")
            continue

        acc = f"{results.get('accuracy_mean', 0):.4f}±{results.get('accuracy_std', 0):.4f}"
        f1 = f"{results.get('f1_mean', 0):.4f}±{results.get('f1_std', 0):.4f}"
        auc = f"{results.get('auc_mean', 0):.4f}±{results.get('auc_std', 0):.4f}"
        elapsed = f"{results.get('_elapsed_seconds', 0):.0f}"

        lines.append(f"| {name} | {acc} | {f1} | {auc} | {elapsed} |")

    lines.append(f"\n## 结论\n")

    # 找最佳模型
    valid = {k: v for k, v in all_results.items() if v is not None}
    if valid:
        best_f1 = max(valid.items(), key=lambda x: x[1].get('f1_mean', 0))
        best_acc = max(valid.items(), key=lambda x: x[1].get('accuracy_mean', 0))
        best_auc = max(valid.items(), key=lambda x: x[1].get('auc_mean', 0))

        lines.append(f"- **最高 F1**: {best_f1[0]} ({best_f1[1].get('f1_mean', 0):.4f})")
        lines.append(f"- **最高 Accuracy**: {best_acc[0]} ({best_acc[1].get('accuracy_mean', 0):.4f})")
        lines.append(f"- **最高 AUC**: {best_auc[0]} ({best_auc[1].get('auc_mean', 0):.4f})")

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"分析报告已保存至: {path}")


if __name__ == '__main__':
    main()
