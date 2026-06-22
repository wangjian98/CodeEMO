"""
随机森林模型评估脚本

在全数据集上重新训练模型，并通过5折交叉验证进行全面评估。
标签约定: y=1 表示通过(passed), y=0 表示未通过/有风险(failed/at-risk)

用法:
    python models/rf/evaluate.py
    python models/rf/evaluate.py --folds 10
    python models/rf/evaluate.py --output-dir outputs/rf
"""
import sys
import os
import json
import argparse

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

# 将项目根目录加入 sys.path，以便导入 common 模块
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, set_seed
from common.feature_engineering import build_feature_matrix
from common.evaluator import evaluate, summarize_fold_results, print_results_table
from models.rf.model import create_model


def evaluate_model(folds=5, output_dir='outputs/rf'):
    """在全数据集上重新训练模型，并通过交叉验证进行全面评估

    评估流程:
      1. 对全部数据做 StandardScaler 标准化
      2. 在全量数据上训练最终模型，记录训练集拟合表现
      3. 使用5折交叉验证获取泛化性能的逐折指标

    Args:
        folds: 交叉验证折数
        output_dir: 结果输出目录
    """
    set_seed(42)

    print("=" * 60)
    print("  随机森林模型评估")
    print("=" * 60)

    # 加载数据
    ide_logs, passed = load_ide_logs()
    X, y, student_ids = build_feature_matrix(ide_logs, passed)

    print(f"\n样本总数: {len(y)}")
    print(f"  通过 (y=1): {int(y.sum())}")
    print(f"  未通过 (y=0): {int((y == 0).sum())}")
    print(f"特征维度: {X.shape[1]}")

    # 全量数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 在全量数据上训练最终模型
    print("\n在全量数据上训练最终模型...")
    final_model = create_model(n_estimators=100, max_depth=10, random_state=42)
    final_model.fit(X_scaled, y)

    # 训练集拟合表现
    y_train_pred = final_model.predict(X_scaled)
    y_train_prob = final_model.predict_proba(X_scaled)[:, 1]
    train_metrics = evaluate(y, y_train_pred, y_train_prob)

    print(f"\n训练集拟合表现:")
    print(f"  Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"  Precision: {train_metrics['precision']:.4f}")
    print(f"  Recall:    {train_metrics['recall']:.4f}")
    print(f"  F1 Score:  {train_metrics['f1']:.4f}")
    print(f"  AUC:       {train_metrics['auc']:.4f}")

    # 特征重要性
    feature_importance = final_model.feature_importances_
    top_k = min(10, len(feature_importance))
    top_indices = np.argsort(feature_importance)[::-1][:top_k]
    print(f"\nTop-{top_k} 重要特征:")
    for rank, idx in enumerate(top_indices):
        print(f"  {rank + 1}. 特征#{idx:02d} - 重要性: {feature_importance[idx]:.4f}")

    # 5折交叉验证评估泛化性能
    print(f"\n开始 {folds} 折交叉验证评估...")
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        fold_scaler = StandardScaler()
        X_train_scaled = fold_scaler.fit_transform(X_train)
        X_val_scaled = fold_scaler.transform(X_val)

        fold_model = create_model(n_estimators=100, max_depth=10, random_state=42)
        fold_model.fit(X_train_scaled, y_train)

        y_pred = fold_model.predict(X_val_scaled)
        y_prob = fold_model.predict_proba(X_val_scaled)[:, 1]

        metrics = evaluate(y_val, y_pred, y_prob)
        fold_results.append(metrics)

        print(f"  Fold {fold_idx + 1}/{folds} - "
              f"Acc: {metrics['accuracy']:.4f}, "
              f"F1: {metrics['f1']:.4f}, "
              f"AUC: {metrics['auc']:.4f}")

    # 汇总交叉验证结果
    cv_summary = summarize_fold_results(fold_results)
    print_results_table("随机森林 (Random Forest) - 交叉验证", cv_summary)

    # 保存评估结果
    os.makedirs(output_dir, exist_ok=True)
    eval_path = os.path.join(output_dir, 'evaluation.json')
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model': 'RandomForest',
            'hyperparameters': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            'n_samples': int(len(y)),
            'n_features': int(X.shape[1]),
            'n_folds': folds,
            'train_full_metrics': train_metrics,
            'cv_summary': {k: v for k, v in cv_summary.items() if k != 'folds'},
            'cv_fold_details': fold_results,
            'feature_importance': {
                f'feature_{idx:02d}': float(feature_importance[idx])
                for idx in range(len(feature_importance))
            },
            'top_features': [
                {'feature': int(idx), 'importance': float(feature_importance[idx])}
                for idx in top_indices
            ],
        }, f, ensure_ascii=False, indent=2)

    print(f"\n评估结果已保存至: {eval_path}")

    return cv_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='随机森林模型评估')
    parser.add_argument('--folds', type=int, default=5,
                        help='交叉验证折数 (默认: 5)')
    parser.add_argument('--output-dir', type=str, default='outputs/rf',
                        help='结果输出目录 (默认: outputs/rf)')
    args = parser.parse_args()

    evaluate_model(folds=args.folds, output_dir=args.output_dir)
