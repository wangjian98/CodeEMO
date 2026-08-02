"""
评估模块 - 共享评估指标
"""
import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score,
    recall_score, roc_auc_score
)


def evaluate(y_true, y_pred, y_prob=None):
    """计算全套评估指标

    Args:
        y_true: 真实标签 (0/1)
        y_pred: 预测标签 (0/1)
        y_prob: 预测为正类的概率 (用于AUC)，若为None则用y_pred

    Returns:
        dict: {accuracy, precision, recall, f1, auc}
    """
    if y_prob is None:
        y_prob = y_pred.astype(float)

    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
    }


def summarize_fold_results(fold_results):
    """将多折结果汇总为均值±标准差

    Args:
        fold_results: list of dict (each from evaluate())

    Returns:
        dict: {metric_mean, metric_std} for each metric, plus 'folds' list
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    summary = {}

    for m in metrics:
        values = [r[m] for r in fold_results]
        summary[f'{m}_mean'] = float(np.mean(values))
        summary[f'{m}_std'] = float(np.std(values))

    summary['folds'] = fold_results
    return summary


def print_results_table(model_name, summary):
    """打印结果汇总"""
    print(f"\n{'='*50}")
    print(f"  {model_name} - 5折交叉验证结果")
    print(f"{'='*50}")
    print(f"  Accuracy:  {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    print(f"  Precision: {summary['precision_mean']:.4f} ± {summary['precision_std']:.4f}")
    print(f"  Recall:    {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}")
    print(f"  F1 Score:  {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    print(f"  AUC:       {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
    print(f"{'='*50}")
