"""
Transformer模型评估脚本 - 学生早期风险预测

在全量数据上重新训练, 使用5折分层交叉验证进行全面评估,
保存详细的评估结果。

用法:
    python models/transformer/evaluate.py
    python models/transformer/evaluate.py --folds 10
    python models/transformer/evaluate.py --output-dir outputs/transformer
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ---- 路径设置: 定位 common/ 模块 ----
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, get_device, set_seed
from common.feature_engineering import build_feature_matrix
from common.evaluator import evaluate, summarize_fold_results, print_results_table

from models.transformer.model import create_model
from models.transformer.train import train_one_epoch, predict


def train_full_model(X, y, device, epochs=100, batch_size=32,
                     patience=10, seed=42):
    """在全量数据上训练最终模型

    Returns:
        model: 训练完成的模型
        scaler: 拟合好的StandardScaler
    """
    set_seed(seed)

    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # 转为Tensor
    X_tensor = torch.from_numpy(X_scaled)
    y_tensor = torch.from_numpy(y.astype(np.float32))

    # 构建DataLoader
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # 创建模型
    model = create_model(input_dim=X.shape[1])
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # 训练 (无验证集, 使用全部数据, 基于训练损失早停)
    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, dataloader, criterion, optimizer, device)

        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={train_loss:.4f}  "
                  f"best={best_loss:.4f}  patience={patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"  早停触发 (epoch {epoch}), 最佳训练损失: {best_loss:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler


def cross_validate(X, y, device, folds=5, epochs=100, batch_size=32,
                   patience=10, seed=42):
    """执行K折交叉验证评估

    Returns:
        fold_results: 每折的评估指标列表
        all_y_true: 所有折拼接的真实标签
        all_y_prob: 所有折拼接的预测概率
    """
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_results = []
    all_y_true = []
    all_y_prob = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- 评估 Fold {fold_idx}/{folds} ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        print(f"  训练集: {len(y_train)} (正: {int(y_train.sum())}), "
              f"验证集: {len(y_val)} (正: {int(y_val.sum())})")

        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
        X_val_scaled = scaler.transform(X_val).astype(np.float32)

        # 转为Tensor
        X_train_tensor = torch.from_numpy(X_train_scaled)
        y_train_tensor = torch.from_numpy(y_train.astype(np.float32))
        X_val_tensor = torch.from_numpy(X_val_scaled)

        # 构建DataLoader
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        # 创建并训练模型
        set_seed(seed)
        model = create_model(input_dim=X.shape[1])
        model.to(device)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        # 训练循环 (带早停)
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

            # 验证损失
            val_prob = predict(model, X_val_tensor, device)
            val_loss = criterion(
                torch.from_numpy(val_prob),
                torch.from_numpy(y_val.astype(np.float32)).unsqueeze(1),
            ).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        # 加载最佳模型
        if best_state is not None:
            model.load_state_dict(best_state)

        # 最终预测
        y_prob = predict(model, X_val_tensor, device).flatten()
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = evaluate(y_val, y_pred, y_prob)
        fold_results.append(metrics)
        all_y_true.extend(y_val.tolist())
        all_y_prob.extend(y_prob.tolist())

        print(f"  Fold {fold_idx} 结果: "
              f"Acc={metrics['accuracy']:.4f}, "
              f"F1={metrics['f1']:.4f}, "
              f"AUC={metrics['auc']:.4f}")

    return fold_results, np.array(all_y_true), np.array(all_y_prob)


def main():
    parser = argparse.ArgumentParser(description='Transformer模型评估 - 学生早期风险预测')
    parser.add_argument('--folds', type=int, default=5, help='交叉验证折数 (默认5)')
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(_PROJECT_ROOT, 'outputs', 'transformer'),
                        help='结果输出目录')
    parser.add_argument('--epochs', type=int, default=100, help='最大训练轮数 (默认100)')
    parser.add_argument('--batch-size', type=int, default=32, help='批大小 (默认32)')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值 (默认10)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子 (默认42)')
    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 获取计算设备
    device = get_device()
    print(f"计算设备: {device}")

    # 加载数据
    print("\n=== 加载数据 ===")
    ide_logs, passed = load_ide_logs()
    X, y, student_ids = build_feature_matrix(ide_logs, passed)
    print(f"样本数: {len(y)}, 特征维度: {X.shape[1]}")
    print(f"正样本(通过): {int(y.sum())}, 负样本(未通过): {int(len(y) - y.sum())}")

    # 第一步: 5折交叉验证评估
    print(f"\n=== {args.folds}折分层交叉验证评估 ===")
    fold_results, all_y_true, all_y_prob = cross_validate(
        X, y, device,
        folds=args.folds, epochs=args.epochs, batch_size=args.batch_size,
        patience=args.patience, seed=args.seed,
    )

    # 汇总交叉验证结果
    cv_summary = summarize_fold_results(fold_results)
    print_results_table('Transformer (交叉验证)', cv_summary)

    # 第二步: 在全量数据上重新训练最终模型
    print(f"\n=== 在全量数据上训练最终模型 ===")
    final_model, final_scaler = train_full_model(
        X, y, device,
        epochs=args.epochs, batch_size=args.batch_size,
        patience=args.patience, seed=args.seed,
    )

    # 在全量数据上的训练集表现 (参考用, 非泛化性能)
    X_scaled = final_scaler.transform(X).astype(np.float32)
    X_tensor = torch.from_numpy(X_scaled)
    full_prob = predict(final_model, X_tensor, device).flatten()
    full_pred = (full_prob >= 0.5).astype(int)
    full_metrics = evaluate(y, full_pred, full_prob)

    print(f"\n=== 全量数据训练集表现 (参考) ===")
    print(f"  Accuracy:  {full_metrics['accuracy']:.4f}")
    print(f"  Precision: {full_metrics['precision']:.4f}")
    print(f"  Recall:    {full_metrics['recall']:.4f}")
    print(f"  F1 Score:  {full_metrics['f1']:.4f}")
    print(f"  AUC:       {full_metrics['auc']:.4f}")

    # 保存评估结果
    os.makedirs(args.output_dir, exist_ok=True)
    eval_path = os.path.join(args.output_dir, 'evaluation.json')
    output = {
        'model': 'Transformer',
        'evaluation_type': '5-fold stratified cross-validation + full retrain',
        'folds': args.folds,
        'hyperparameters': {
            'd_model': 64,
            'nhead': 4,
            'num_layers': 3,
            'dropout': 0.2,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'patience': args.patience,
            'lr': 1e-3,
            'weight_decay': 1e-5,
        },
        'cross_validation': {k: v for k, v in cv_summary.items() if k != 'folds'},
        'cv_fold_details': cv_summary['folds'],
        'full_retrain_metrics': full_metrics,
    }
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n评估结果已保存至: {eval_path}")


if __name__ == '__main__':
    main()
