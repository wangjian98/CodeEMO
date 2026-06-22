"""
Mamba-SSM GPU 完整训练流程 (6步)

使用 FullMambaStudent (d_model=64, n_layers=6) 在GPU上运行完整的6步流程。
自动检测GPU, 若不可用则回退到CPU。

与 CPU 版本的区别:
  - 模型: FullMambaStudent (d_model=64, n_layers=6, d_state=16)
  - 预训练: 3 epochs, batch_size=32
  - 微调: batch_size=16
  - 更强的多尺度特征提取 (交叉注意力融合)

用法:
    python models/mamba/train_gpu.py
    python models/mamba/train_gpu.py --folds 10
    python models/mamba/train_gpu.py --output-dir outputs/mamba
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
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, set_seed
from common.evaluator import summarize_fold_results, print_results_table
from models.mamba.model import FullMambaStudent
from models.mamba.steps.step1_preprocessing import preprocess
from models.mamba.steps.step2_pretrain import pretrain
from models.mamba.steps.step3_multiscale import extract_representations
from models.mamba.steps.step4_prototype import run_kmeans
from models.mamba.steps.step5_finetune import finetune_cv
from models.mamba.steps.step6_interpret import run_interpretability


def main():
    parser = argparse.ArgumentParser(description='Mamba-SSM GPU 训练流程')
    parser.add_argument('--folds', type=int, default=5, help='交叉验证折数 (默认: 5)')
    parser.add_argument('--output-dir', type=str, default='outputs/mamba',
                        help='输出目录 (默认: outputs/mamba)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子 (默认: 42)')
    parser.add_argument('--pretrain-epochs', type=int, default=3,
                        help='预训练轮数 (默认: 3)')
    parser.add_argument('--finetune-epochs', type=int, default=5,
                        help='每折微调轮数 (默认: 5)')
    parser.add_argument('--pretrain-batch-size', type=int, default=32,
                        help='预训练批大小 (默认: 32)')
    parser.add_argument('--finetune-batch-size', type=int, default=16,
                        help='微调批大小 (默认: 16)')
    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设备: 自动检测GPU, 回退到CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    else:
        device = torch.device('cpu')
        gpu_name = None
        gpu_mem = None

    print("=" * 60)
    print("  Mamba-SSM GPU 训练流程")
    print("  模型: FullMambaStudent (d_model=64, n_layers=6)")
    print("=" * 60)
    if gpu_name:
        print(f"  设备: {device} ({gpu_name}, {gpu_mem:.1f} GB)")
    else:
        print(f"  设备: {device} (GPU不可用, 回退到CPU)")
    print(f"  交叉验证折数: {args.folds}")
    print(f"  预训练轮数: {args.pretrain_epochs}")
    print(f"  预训练批大小: {args.pretrain_batch_size}")
    print(f"  微调轮数/折: {args.finetune_epochs}")
    print(f"  微调批大小: {args.finetune_batch_size}")
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
    # Step 2: 预训练 - 下一事件预测
    # ============================================================
    model = FullMambaStudent(
        n_event_types=7, d_model=64, n_layers=6, d_state=16,
        n_prototypes=4, max_seq_len=2000
    )
    model.to(device)

    pretrained_state = pretrain(
        model, samples, device,
        epochs=args.pretrain_epochs, batch_size=args.pretrain_batch_size
    )

    # ============================================================
    # Step 3: 多尺度特征提取
    # ============================================================
    model.load_state_dict(pretrained_state)
    representations, repr_labels = extract_representations(
        model, samples, device, batch_size=args.pretrain_batch_size
    )

    # ============================================================
    # Step 4: 原型发现 - K-Means 聚类
    # ============================================================
    kmeans, cluster_assignments = run_kmeans(
        representations, repr_labels, n_clusters=4
    )

    # ============================================================
    # Step 5: 预测微调 - 5折交叉验证
    # ============================================================
    fold_results = finetune_cv(
        pretrained_state, samples, student_ids, labels, device,
        n_folds=args.folds, n_epochs=args.finetune_epochs,
        batch_size=args.finetune_batch_size
    )

    # 汇总结果
    summary = summarize_fold_results(fold_results)
    print_results_table("Mamba-SSM (GPU)", summary)

    # ============================================================
    # Step 6: 可解释性分析
    # ============================================================
    best_fold_idx = np.argmax([f['f1'] for f in fold_results])
    print(f"\n  使用第 {best_fold_idx+1} 折 (最佳F1) 的模型进行可解释性分析")

    interp_model = FullMambaStudent(
        n_event_types=7, d_model=64, n_layers=6, d_state=16,
        n_prototypes=4, max_seq_len=2000
    )
    interp_model.to(device)
    interp_model.load_state_dict(pretrained_state)

    interpret_dict = run_interpretability(interp_model, samples, device)

    # ============================================================
    # 保存结果
    # ============================================================
    results = {
        'model': 'FullMambaStudent',
        'config': {
            'd_model': 64,
            'n_layers': 6,
            'd_state': 16,
            'n_prototypes': 4,
            'max_seq_len': 2000,
            'device': str(device),
            'gpu_name': gpu_name,
        },
        'pipeline': {
            'pretrain_epochs': args.pretrain_epochs,
            'finetune_epochs': args.finetune_epochs,
            'n_folds': args.folds,
            'pretrain_batch_size': args.pretrain_batch_size,
            'finetune_batch_size': args.finetune_batch_size,
        },
        'cv_results': {
            'accuracy_mean': summary['accuracy_mean'],
            'accuracy_std': summary['accuracy_std'],
            'precision_mean': summary['precision_mean'],
            'precision_std': summary['precision_std'],
            'recall_mean': summary['recall_mean'],
            'recall_std': summary['recall_std'],
            'f1_mean': summary['f1_mean'],
            'f1_std': summary['f1_std'],
            'auc_mean': summary['auc_mean'],
            'auc_std': summary['auc_std'],
        },
        'n_samples': len(samples),
        'n_passed': int((labels == 0).sum()),
        'n_failed': int((labels == 1).sum()),
        'interpretability': {
            'event_importance': interpret_dict['event_importance'].tolist(),
            'event_type_names': interpret_dict['event_type_names'],
            'temporal_mean': interpret_dict['temporal_mean'],
            'temporal_std': interpret_dict['temporal_std'],
            'proto_risk_correlation': interpret_dict['proto_risk_correlation'],
        },
        'fold_details': fold_results,
    }

    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至: {results_path}")

    # 打印综合总结
    print("\n" + "=" * 60)
    print("  Mamba-SSM GPU 训练完成 - 综合总结")
    print("=" * 60)
    print(f"  模型: FullMambaStudent")
    print(f"  设备: {device}" + (f" ({gpu_name})" if gpu_name else ""))
    print(f"  样本数: {len(samples)} (通过: {(labels==0).sum()}, 挂科: {(labels==1).sum()})")
    print(f"  交叉验证: {args.folds} 折")
    print(f"  Accuracy: {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")
    print(f"  F1 Score: {summary['f1_mean']:.4f} +/- {summary['f1_std']:.4f}")
    print(f"  AUC:      {summary['auc_mean']:.4f} +/- {summary['auc_std']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
