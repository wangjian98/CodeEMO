"""
LSTM模型评估脚本 - 学生早期风险预测

在全量数据上重新训练LSTM模型，并通过5折交叉验证进行全面评估。
与train.py的区别: 本脚本专注于模型评估，保存更详细的评估报告。

用法:
    python models/lstm/evaluate.py
    python models/lstm/evaluate.py --folds 5 --output-dir outputs/lstm
"""
import sys
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# 定位项目根目录和common模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.data_loader import load_ide_logs, get_device, set_seed
from common.feature_engineering import build_feature_matrix
from common.evaluator import evaluate, summarize_fold_results, print_results_table
from models.lstm.model import LSTMClassifier


def create_dataloader(X, y, batch_size=32, shuffle=True):
    """创建PyTorch DataLoader"""
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate_model(model, X, y, device):
    """在验证集上评估模型，返回评估指标dict"""
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    outputs = model(X_tensor)
    y_prob = outputs.squeeze().cpu().numpy()
    y_pred = (y_prob > 0.5).astype(int)
    return evaluate(y, y_pred, y_prob)


def train_fold(X_train, y_train, X_val, y_val, device, params):
    """训练单个折的LSTM模型，带早停"""
    input_dim = X_train.shape[1]
    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_dim=params['hidden_dim'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    train_loader = create_dataloader(X_train, y_train, batch_size=params['batch_size'], shuffle=True)

    best_val_f1 = -1.0
    best_state = None
    best_metrics = None
    patience_counter = 0
    epoch_losses = []

    for epoch in range(params['epochs']):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        epoch_losses.append(train_loss)

        val_metrics = evaluate_model(model, X_val, y_val, device)
        val_f1 = val_metrics['f1']

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_metrics = val_metrics
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= params['patience']:
            break

    # 恢复最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)

    # 最终评估
    final_metrics = evaluate_model(model, X_val, y_val, device)
    return model, final_metrics, epoch_losses


def train_full_model(X, y, device, params):
    """在全量数据上训练LSTM模型

    用于训练最终部署模型 (无验证集，固定epoch数)
    """
    input_dim = X.shape[1]
    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_dim=params['hidden_dim'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    train_loader = create_dataloader(X, y, batch_size=params['batch_size'], shuffle=True)

    print("  在全量数据上训练最终模型...")
    for epoch in range(params['epochs']):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d}/{params['epochs']}  loss={train_loss:.4f}")

    return model


def main():
    parser = argparse.ArgumentParser(description='LSTM模型评估 - 学生早期风险预测')
    parser.add_argument('--folds', type=int, default=5, help='交叉验证折数 (默认5)')
    parser.add_argument('--output-dir', type=str, default='outputs/lstm', help='输出目录')
    args = parser.parse_args()

    # 设置随机种子
    set_seed(42)

    # 获取设备
    device = get_device()
    print(f"设备: {device}")

    # 加载数据
    print("\n" + "=" * 60)
    print("  LSTM模型评估 - 学生早期风险预测")
    print("=" * 60)

    ide_logs, passed = load_ide_logs()
    X, y, student_ids = build_feature_matrix(ide_logs, passed)

    print(f"\n数据集: {X.shape[0]} 个学生, {X.shape[1]} 维特征")
    print(f"标签分布: 通过={sum(y)}, 未通过={len(y) - sum(y)}")

    # 超参数
    params = {
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.3,
        'lr': 0.001,
        'epochs': 100,
        'patience': 10,
        'batch_size': 32,
    }
    print(f"\n超参数: {params}")

    # === 第一步: 5折交叉验证评估 ===
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    fold_results = []

    print(f"\n开始 {args.folds} 折交叉验证评估...\n")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold_idx + 1}/{args.folds} ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # StandardScaler标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        print(f"  训练集: {len(X_train)} (通过={sum(y_train)})")
        print(f"  验证集: {len(X_val)} (通过={sum(y_val)})")

        model, metrics, epoch_losses = train_fold(
            X_train_scaled, y_train, X_val_scaled, y_val, device, params
        )
        fold_results.append(metrics)

        print(f"  结果: Acc={metrics['accuracy']:.4f}  Prec={metrics['precision']:.4f}  "
              f"Rec={metrics['recall']:.4f}  F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}")

    # 汇总交叉验证结果
    summary = summarize_fold_results(fold_results)
    print_results_table('LSTM (交叉验证)', summary)

    # === 第二步: 在全量数据上重新训练最终模型 ===
    print("\n在全量数据上重新训练最终模型...")
    scaler_full = StandardScaler()
    X_scaled_full = scaler_full.fit_transform(X)
    full_model = train_full_model(X_scaled_full, y, device, params)

    # 全量数据上的自评估 (仅供参考，非泛化性能)
    full_metrics = evaluate_model(full_model, X_scaled_full, y, device)
    print(f"\n全量数据自评估: Acc={full_metrics['accuracy']:.4f}  "
          f"F1={full_metrics['f1']:.4f}  AUC={full_metrics['auc']:.4f}")
    print("  (注意: 全量数据自评估仅供参考，不代表泛化性能)")

    # === 保存评估报告 ===
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'evaluation.json')

    evaluation_report = {
        'model': 'LSTM (单向)',
        'description': '单向LSTM分类器用于学生早期风险预测',
        'params': params,
        'folds': args.folds,
        'n_samples': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'cv_summary': {k: v for k, v in summary.items() if k != 'folds'},
        'cv_fold_details': fold_results,
        'full_data_self_eval': full_metrics,
        'note': '全量数据自评估仅供参考，请以交叉验证结果为准。',
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_report, f, indent=2, ensure_ascii=False)
    print(f"\n评估报告已保存至: {output_path}")

    # === 打印最终总结 ===
    print(f"\n{'=' * 60}")
    print(f"  LSTM模型评估总结")
    print(f"{'=' * 60}")
    print(f"  交叉验证 Accuracy: {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    print(f"  交叉验证 F1 Score: {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    print(f"  交叉验证 AUC:      {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
