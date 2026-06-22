"""
随机森林模型训练脚本

使用5折分层交叉验证训练随机森林模型，评估学生早期风险预测性能。
标签约定: y=1 表示通过(passed), y=0 表示未通过/有风险(failed/at-risk)

用法:
    python models/rf/train.py
    python models/rf/train.py --folds 10
    python models/rf/train.py --output-dir outputs/rf
"""
import sys
import os
import json
import argparse

import numpy as np
from sklearn.model_selection import StratifiedKFold
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


def train(folds=5, output_dir='outputs/rf'):
    """执行5折交叉验证训练随机森林模型

    Args:
        folds: 交叉验证折数
        output_dir: 结果输出目录
    """
    # 设置随机种子，保证实验可复现
    set_seed(42)

    # 加载数据
    print("=" * 60)
    print("  随机森林模型训练")
    print("=" * 60)

    ide_logs, passed = load_ide_logs()
    X, y, student_ids = build_feature_matrix(ide_logs, passed)

    print(f"\n样本总数: {len(y)}")
    print(f"  通过 (y=1): {int(y.sum())}")
    print(f"  未通过 (y=0): {int((y == 0).sum())}")
    print(f"特征维度: {X.shape[1]}")

    # 5折分层交叉验证
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    fold_results = []

    print(f"\n开始 {folds} 折交叉验证...\n")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # 创建并训练模型
        model = create_model(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train_scaled, y_train)

        # 预测
        y_pred = model.predict(X_val_scaled)
        y_prob = model.predict_proba(X_val_scaled)[:, 1]

        # 评估
        metrics = evaluate(y_val, y_pred, y_prob)
        fold_results.append(metrics)

        print(f"  Fold {fold_idx + 1}/{folds} - "
              f"Acc: {metrics['accuracy']:.4f}, "
              f"F1: {metrics['f1']:.4f}, "
              f"AUC: {metrics['auc']:.4f}")

    # 汇总结果
    summary = summarize_fold_results(fold_results)
    print_results_table("随机森林 (Random Forest)", summary)

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model': 'RandomForest',
            'hyperparameters': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            'n_folds': folds,
            'n_samples': int(len(y)),
            'n_features': int(X.shape[1]),
            'summary': {k: v for k, v in summary.items() if k != 'folds'},
            'fold_details': fold_results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存至: {results_path}")

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='随机森林模型训练')
    parser.add_argument('--folds', type=int, default=5,
                        help='交叉验证折数 (默认: 5)')
    parser.add_argument('--output-dir', type=str, default='outputs/rf',
                        help='结果输出目录 (默认: outputs/rf)')
    args = parser.parse_args()

    train(folds=args.folds, output_dir=args.output_dir)
