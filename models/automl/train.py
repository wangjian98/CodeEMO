"""
AUTOML 模型训练脚本 (TSFRESH-only 基线)

使用 TSFRESH 自动从 IDE 事件时间序列中提取特征，
再训练 Random Forest 分类器在 5 折分层交叉验证下评估。

对比基准: models/rf 在手工 46 维特征上的结果。
标签约定: y=1 表示通过(passed), y=0 表示未通过/有风险(failed/at-risk)

用法:
    # 默认配置（5折, MinimalFCParameters, 5000 events/student）
    python models/automl/train.py

    # 更全面的特征提取 (ComprehensiveFCParameters)
    python models/automl/train.py --fc comprehensive

    # 调整事件采样数
    python models/automl/train.py --max-events 10000
"""
import os
import sys
import json
import time
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# 将项目根目录加入 sys.path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, set_seed
from common.evaluator import evaluate, summarize_fold_results, print_results_table
from models.automl.model import (
    create_classifier,
    build_long_format,
    extract_tsfresh_features,
    select_features_by_target,
    EVENT_TYPES
)


def get_fc_parameters(fc_name):
    """根据名称选择 TSFRESH 特征提取参数集"""
    from tsfresh.feature_extraction import (
        MinimalFCParameters, EfficientFCParameters,
        ComprehensiveFCParameters
    )
    fc_map = {
        'minimal': MinimalFCParameters(),
        'efficient': EfficientFCParameters(),
        'comprehensive': ComprehensiveFCParameters()
    }
    if fc_name not in fc_map:
        raise ValueError(f"Unknown FC parameters: {fc_name}. "
                         f"Choose from {list(fc_map.keys())}")
    return fc_map[fc_name]


def train(folds=5, output_dir='outputs/automl', fc_name='minimal',
          max_events_per_student=5000, n_jobs=8, fdr_level=0.05):
    """执行 5 折交叉验证训练 AUTOML (TSFRESH) 模型

    流程:
      1. 加载 IDE 日志
      2. 转换为 TSFRESH long format
      3. 提取 TSFRESH 特征（可选 Minimal / Efficient / Comprehensive）
      4. FDR 校正的特征选择
      5. 在选择后的特征上训练 RF，5 折交叉验证
      6. 输出 Acc/Precision/Recall/F1/AUC 指标

    Args:
        folds: 交叉验证折数
        output_dir: 结果输出目录
        fc_name: TSFRESH 特征提取参数集
            ('minimal' / 'efficient' / 'comprehensive')
        max_events_per_student: 每名学生保留的最大事件数
        n_jobs: 并行核数
        fdr_level: FDR 校正水平
    """
    # 设置随机种子
    set_seed(42)

    print("=" * 70)
    print(f"  AUTOML 模型训练 (TSFRESH + RF)")
    print(f"  FC Parameters: {fc_name}, max_events/student: {max_events_per_student}")
    print("=" * 70)

    t_total_start = time.time()

    # === 1. 加载数据 ===
    print("\n[1/5] 加载 IDE 日志 ...")
    ide_logs, passed = load_ide_logs()
    # passed 是 DataFrame(student, passed)；转成 dict 方便查找
    passed_dict = dict(zip(passed['student'], passed['passed'].astype(int)))
    n_students = ide_logs['student'].nunique()
    print(f"  总学生数: {n_students}")

    # === 2. 转换为 TSFRESH long format ===
    print("\n[2/5] 构建 TSFRESH long format ...")
    long_df = build_long_format(ide_logs, max_events_per_student=max_events_per_student)

    # === 3. 提取 TSFRESH 特征 ===
    print("\n[3/5] 提取 TSFRESH 特征 ...")
    fc_params = get_fc_parameters(fc_name)
    features_all = extract_tsfresh_features(
        long_df, n_jobs=n_jobs, kind_to_fc_params=fc_params
    )

    # === 4. FDR 特征选择 ===
    print("\n[4/5] FDR 校正的特征选择 ...")
    # 按 student 排序 passed 标签，与 features_all 的索引对齐
    student_ids = sorted(features_all.index.tolist())
    y = np.array([passed_dict.get(sid, 0) for sid in student_ids])
    X = features_all.loc[student_ids].values

    print(f"  Before selection: X.shape = {X.shape}, y distribution: {np.bincount(y)}")

    features_selected = select_features_by_target(
        pd.DataFrame(X, index=student_ids), y, fdr_level=fdr_level
    )
    X_selected = features_selected.values

    # === 5. 5 折交叉验证训练 ===
    print(f"\n[5/5] {folds} 折分层交叉验证训练 RF ...")
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_selected, y)):
        X_train, X_val = X_selected[train_idx], X_selected[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # 训练 RF
        model = create_classifier(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X_train_scaled, y_train)

        # 预测
        y_pred = model.predict(X_val_scaled)
        y_prob = model.predict_proba(X_val_scaled)[:, 1]

        # 评估
        metrics = evaluate(y_val, y_pred, y_prob)
        fold_results.append(metrics)

        print(f"  Fold {fold_idx + 1}/{folds} - "
              f"Acc: {metrics['accuracy']:.4f}, "
              f"F1:  {metrics['f1']:.4f}, "
              f"AUC: {metrics['auc']:.4f}")

    # === 汇总结果 ===
    summary = summarize_fold_results(fold_results)
    print_results_table("AUTOML (TSFRESH + RF)", summary)

    elapsed = time.time() - t_total_start
    print(f"\n总耗时: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分钟)")

    # === 保存结果 ===
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model': 'AUTOML_TSFRESH_RF',
            'methodology': 'TSFRESH time-series feature extraction + Random Forest',
            'hyperparameters': {
                'tsfresh_fc_parameters': fc_name,
                'max_events_per_student': max_events_per_student,
                'fdr_level': fdr_level,
                'rf_n_estimators': 200,
                'rf_max_depth': 10,
                'random_state': 42
            },
            'n_students': int(n_students),
            'n_features_raw': int(features_all.shape[1]),
            'n_features_selected': int(X_selected.shape[1]),
            'n_folds': folds,
            'summary': {k: v for k, v in summary.items() if k != 'folds'},
            'fold_details': fold_results,
            'elapsed_seconds': elapsed,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存至: {results_path}")

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AUTOML (TSFRESH) 模型训练')
    parser.add_argument('--folds', type=int, default=5,
                        help='交叉验证折数 (默认: 5)')
    parser.add_argument('--output-dir', type=str, default='outputs/automl',
                        help='结果输出目录 (默认: outputs/automl)')
    parser.add_argument('--fc', type=str, default='minimal',
                        choices=['minimal', 'efficient', 'comprehensive'],
                        help='TSFRESH 特征提取参数集 (默认: minimal)')
    parser.add_argument('--max-events', type=int, default=5000,
                        help='每名学生保留的最大事件数 (默认: 5000)')
    parser.add_argument('--n-jobs', type=int, default=8,
                        help='并行核数 (默认: 8)')
    parser.add_argument('--fdr-level', type=float, default=0.05,
                        help='FDR 校正水平 (默认: 0.05)')
    args = parser.parse_args()

    train(
        folds=args.folds,
        output_dir=args.output_dir,
        fc_name=args.fc,
        max_events_per_student=args.max_events,
        n_jobs=args.n_jobs,
        fdr_level=args.fdr_level
    )