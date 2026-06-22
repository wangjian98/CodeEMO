"""
Mamba-SSM 评估脚本

独立的评估流程:
  1. 加载数据并编码
  2. 预训练模型
  3. 5折交叉验证评估 (冻结骨干, 仅训练risk_head)
  4. 可解释性分析
  5. 保存评估结果至 outputs/mamba/evaluation.json

用法:
    python models/mamba/evaluate.py
    python models/mamba/evaluate.py --folds 10
    python models/mamba/evaluate.py --device gpu
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

# ============================================================
# sys.path 设置
# ============================================================
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, set_seed, get_device
from common.evaluator import summarize_fold_results, print_results_table
from models.mamba.model import create_model, SimplifiedMambaStudent, FullMambaStudent
from models.mamba.steps.step1_preprocessing import preprocess
from models.mamba.steps.step2_pretrain import pretrain
from models.mamba.steps.step3_multiscale import extract_representations
from models.mamba.steps.step4_prototype import run_kmeans
from models.mamba.steps.step5_finetune import finetune_cv
from models.mamba.steps.step6_interpret import run_interpretability


def main():
    parser = argparse.ArgumentParser(description='Mamba-SSM 评估')
    parser.add_argument('--folds', type=int, default=5, help='交叉验证折数 (默认: 5)')
    parser.add_argument('--output-dir', type=str, default='outputs/mamba',
                        help='输出目录 (默认: outputs/mamba)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子 (默认: 42)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'gpu'],
                        help='计算设备 (默认: auto)')
    parser.add_argument('--pretrain-epochs', type=int, default=2,
                        help='预训练轮数 (默认: 2)')
    parser.add_argument('--finetune-epochs', type=int, default=5,
                        help='每折微调轮数 (默认: 5)')
    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 确定设备
    if args.device == 'auto':
        device = get_device()
    elif args.device == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    use_gpu = device.type == 'cuda'

    print("=" * 60)
    print("  Mamba-SSM 评估")
    print("=" * 60)
    print(f"  设备: {device}")
    print(f"  模型: {'FullMambaStudent' if use_gpu else 'SimplifiedMambaStudent'}")
    print(f"  交叉验证折数: {args.folds}")
    print(f"  输出目录: {args.output_dir}")
    print("=" * 60)

    # 创建输出目录
    output_dir = os.path.join(_PROJECT_ROOT, args.output_dir) \
        if not os.path.isabs(args.output_dir) else args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # Step 1: 数据预处理
    # ============================================================
    ide_logs_df, passed_df = load_ide_logs()
    samples, student_ids, labels = preprocess(
        ide_logs_df, passed_df, max_events=2000
    )

    # ============================================================
    # Step 2: 预训练
    # ============================================================
    if use_gpu:
        model = FullMambaStudent(
            n_event_types=7, d_model=64, n_layers=6, d_state=16,
            n_prototypes=4, max_seq_len=2000
        )
        batch_size_pretrain = 32
    else:
        model = SimplifiedMambaStudent(
            n_event_types=7, d_model=32, n_layers=2, d_state=8,
            n_prototypes=4, max_seq_len=2000
        )
        batch_size_pretrain = 16

    model.to(device)

    pretrained_state = pretrain(
        model, samples, device,
        epochs=args.pretrain_epochs, batch_size=batch_size_pretrain
    )

    # ============================================================
    # Step 3: 多尺度特征提取
    # ============================================================
    model.load_state_dict(pretrained_state)
    representations, repr_labels = extract_representations(
        model, samples, device, batch_size=batch_size_pretrain
    )

    # ============================================================
    # Step 4: 原型发现
    # ============================================================
    kmeans, cluster_assignments = run_kmeans(
        representations, repr_labels, n_clusters=4
    )

    # ============================================================
    # Step 5: 5折交叉验证评估
    # ============================================================
    fold_results = finetune_cv(
        pretrained_state, samples, student_ids, labels, device,
        n_folds=args.folds, n_epochs=args.finetune_epochs, batch_size=16
    )

    # 汇总结果
    summary = summarize_fold_results(fold_results)
    print_results_table("Mamba-SSM Evaluation", summary)

    # ============================================================
    # Step 6: 可解释性分析
    # ============================================================
    # 使用预训练模型 (未微调) 进行全局可解释性
    interp_model = create_model(device)
    interp_model.load_state_dict(pretrained_state)

    interpret_dict = run_interpretability(interp_model, samples, device)

    # ============================================================
    # 保存评估结果
    # ============================================================
    evaluation = {
        'model': 'FullMambaStudent' if use_gpu else 'SimplifiedMambaStudent',
        'device': str(device),
        'n_samples': len(samples),
        'n_passed': int((labels == 0).sum()),
        'n_failed': int((labels == 1).sum()),
        'n_folds': args.folds,
        'metrics': {
            'accuracy': {
                'mean': summary['accuracy_mean'],
                'std': summary['accuracy_std'],
            },
            'precision': {
                'mean': summary['precision_mean'],
                'std': summary['precision_std'],
            },
            'recall': {
                'mean': summary['recall_mean'],
                'std': summary['recall_std'],
            },
            'f1': {
                'mean': summary['f1_mean'],
                'std': summary['f1_std'],
            },
            'auc': {
                'mean': summary['auc_mean'],
                'std': summary['auc_std'],
            },
        },
        'fold_details': fold_results,
        'cluster_info': {
            'n_clusters': 4,
            'inertia': float(kmeans.inertia_),
            'cluster_sizes': [
                int((cluster_assignments == c).sum()) for c in range(4)
            ],
            'cluster_risk_ratios': [
                float(repr_labels[cluster_assignments == c].mean())
                if (cluster_assignments == c).sum() > 0 else 0.0
                for c in range(4)
            ],
        },
        'interpretability': {
            'event_importance': interpret_dict['event_importance'].tolist(),
            'event_type_names': interpret_dict['event_type_names'],
            'temporal_mean': interpret_dict['temporal_mean'],
            'temporal_std': interpret_dict['temporal_std'],
            'proto_risk_correlation': interpret_dict['proto_risk_correlation'],
        },
    }

    eval_path = os.path.join(output_dir, 'evaluation.json')
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)
    print(f"\n评估结果已保存至: {eval_path}")

    # 打印最终总结
    print("\n" + "=" * 60)
    print("  Mamba-SSM 评估完成")
    print("=" * 60)
    print(f"  模型: {evaluation['model']}")
    print(f"  设备: {device}")
    print(f"  样本数: {len(samples)} (通过: {(labels==0).sum()}, 挂科: {(labels==1).sum()})")
    print(f"  Accuracy: {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")
    print(f"  Precision: {summary['precision_mean']:.4f} +/- {summary['precision_std']:.4f}")
    print(f"  Recall: {summary['recall_mean']:.4f} +/- {summary['recall_std']:.4f}")
    print(f"  F1 Score: {summary['f1_mean']:.4f} +/- {summary['f1_std']:.4f}")
    print(f"  AUC: {summary['auc_mean']:.4f} +/- {summary['auc_std']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
