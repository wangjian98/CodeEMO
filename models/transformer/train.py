"""
Transformer模型训练脚本 - 学生早期风险预测

使用5折分层交叉验证训练Transformer分类器,
对IDE编程日志的46维特征进行学生通过/不通过的二元分类。

用法:
    python models/transformer/train.py
    python models/transformer/train.py --folds 10
    python models/transformer/train.py --output-dir outputs/transformer
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
# 当前文件: models/transformer/train.py
# 项目根目录: 向上3级
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, get_device, set_seed
from common.feature_engineering import build_feature_matrix
from common.evaluator import evaluate, summarize_fold_results, print_results_table

from models.transformer.model import create_model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def predict(model, X_tensor, device, batch_size=64):
    """在评估模式下进行预测"""
    model.eval()
    preds_list = []

    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i:i + batch_size].to(device)
        preds_list.append(model(batch).cpu().numpy())

    return np.vstack(preds_list)


def train_fold(X_train, y_train, X_val, y_val, device, epochs=100,
               batch_size=32, patience=10, seed=42):
    """训练单折, 返回验证集上的预测结果

    使用早停机制: 当验证损失连续patience轮未改善时停止训练。

    Returns:
        y_pred: 验证集预测标签 (0/1)
        y_prob: 验证集预测概率
    """
    # 设置随机种子以确保可复现
    set_seed(seed)

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

    # 创建模型
    model = create_model(input_dim=X_train.shape[1])
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # 早停
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # 验证集损失
        val_prob = predict(model, X_val_tensor, device)
        val_loss = criterion(
            torch.from_numpy(val_prob),
            torch.from_numpy(y_val.astype(np.float32)).unsqueeze(1),
        ).item()

        # 早停判断
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"best={best_val_loss:.4f}  patience={patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"    早停触发 (epoch {epoch}), 最佳验证损失: {best_val_loss:.4f}")
            break

    # 加载最佳模型权重
    if best_state is not None:
        model.load_state_dict(best_state)

    # 最终预测
    y_prob = predict(model, X_val_tensor, device).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    return y_pred, y_prob


def main():
    parser = argparse.ArgumentParser(description='Transformer模型训练 - 学生早期风险预测')
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

    # 5折分层交叉验证
    print(f"\n=== {args.folds}折分层交叉验证训练 ===")
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold_idx}/{args.folds} ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        print(f"  训练集: {len(y_train)} (正: {int(y_train.sum())}), "
              f"验证集: {len(y_val)} (正: {int(y_val.sum())})")

        y_pred, y_prob = train_fold(
            X_train, y_train, X_val, y_val, device,
            epochs=args.epochs, batch_size=args.batch_size,
            patience=args.patience, seed=args.seed,
        )

        metrics = evaluate(y_val, y_pred, y_prob)
        fold_results.append(metrics)
        print(f"  Fold {fold_idx} 结果: "
              f"Acc={metrics['accuracy']:.4f}, "
              f"F1={metrics['f1']:.4f}, "
              f"AUC={metrics['auc']:.4f}")

    # 汇总结果
    summary = summarize_fold_results(fold_results)
    print_results_table('Transformer', summary)

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, 'results.json')
    output = {
        'model': 'Transformer',
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
        'summary': {k: v for k, v in summary.items() if k != 'folds'},
        'fold_details': summary['folds'],
    }
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至: {results_path}")


if __name__ == '__main__':
    main()
