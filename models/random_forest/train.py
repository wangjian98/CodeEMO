"""
Random Forest 训练入口
可单独运行: python models/random_forest/train.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from common.data_loader import load_data
from common.feature_engineering import build_feature_matrix
from common.evaluator import print_metrics
from models.random_forest.evaluator import evaluate_rf
from models.random_forest.visualize import visualize_all


def run(feature_set='7'):
    """运行 RF 实验

    Args:
        feature_set: '7' 使用7维特征, '46' 使用46维特征

    Returns:
        dict: 实验结果摘要
    """
    print(f"\n{'='*60}")
    print(f"  Random Forest ({feature_set}特征) 实验")
    print(f"{'='*60}")

    ide_logs, passed = load_data()
    X_7, X_46, X_cs, y, student_ids = build_feature_matrix(ide_logs, passed)

    if feature_set == '7':
        X = X_7
    elif feature_set == '46':
        X = X_46
    elif feature_set in ('cs', 'cross-scale'):
        X = X_cs
    else:
        raise ValueError(f"Unknown feature_set={feature_set}")
    print(f"  特征维度: {X.shape[1]}, 样本数: {len(y)}")

    summary = evaluate_rf(X, y)
    fold_details = summary.pop('fold_details')

    result = {
        'accuracy': summary['accuracy'][0],
        'precision': summary['precision'][0],
        'recall': summary['recall'][0],
        'f1': summary['f1'][0],
        'auc': summary['auc'][0],
        'accuracy_std': summary['accuracy'][1],
        'f1_std': summary['f1'][1],
        'auc_std': summary['auc'][1],
    }
    print_metrics(result, f'Random Forest ({feature_set}feat)')

    save_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'results', 'random_forest')
    os.makedirs(save_dir, exist_ok=True)
    visualize_all(summary, fold_details, save_dir)

    key_suffix = feature_set if feature_set == 'cs' else f'{feature_set}feat'
    return {f'rf_{key_suffix}': {
        'accuracy': {'mean': summary['accuracy'][0], 'std': summary['accuracy'][1]},
        'precision': {'mean': summary['precision'][0], 'std': summary['precision'][1]},
        'recall': {'mean': summary['recall'][0], 'std': summary['recall'][1]},
        'f1': {'mean': summary['f1'][0], 'std': summary['f1'][1]},
        'auc': {'mean': summary['auc'][0], 'std': summary['auc'][1]},
    }}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--feature-set', default='7', choices=['7', '46', 'cs'])
    _args = ap.parse_args()
    run(_args.feature_set)
