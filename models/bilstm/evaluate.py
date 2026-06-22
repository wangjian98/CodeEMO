"""
双向LSTM模型评估脚本

在全量数据上重新训练BiLSTM模型，并使用5折交叉验证进行完整评估。
支持独立运行: python models/bilstm/evaluate.py

用法:
    python models/bilstm/evaluate.py
    python models/bilstm/evaluate.py --folds 10
    python models/bilstm/evaluate.py --output-dir outputs/bilstm
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

# ---- 路径设置: 使得可以独立运行 ----
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, get_device, set_seed
from common.feature_engineering import build_feature_matrix
from common.evaluator import evaluate, summarize_fold_results, print_results_table

from models.bilstm.model import create_model


def train_full_model(
    X_train,
    y_train,
    device,
    input_dim=46,
    hidden_dim=64,
    epochs=100,
    batch_size=32,
    patience=10,
    lr=1e-3,
):
    """在全量训练数据上训练BiLSTM模型，返回训练好的模型实例"""
    model = create_model(input_dim=input_dim, hidden_dim=hidden_dim)
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)

    n_samples = X_train_t.shape[0]
    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i : i + batch_size]
            batch_X = X_train_t[idx]
            batch_y = y_train_t[idx]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # 早停: 用训练损失监控（在全量数据训练时没有独立验证集）
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {
                k: v.clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"    Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  "
                    f"-> 早停 (patience={patience})"
                )
                break

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch}/{epochs}  loss={avg_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate_cross_validation(
    X, y, device, folds=5, epochs=100, batch_size=32, patience=10, lr=1e-3
):
    """执行5折交叉验证评估

    每一折:
        1. 在训练折上训练BiLSTM模型（全量训练数据）
        2. 在验证折上进行预测和评估

    Returns:
        fold_results: list of dict (每折的评估指标)
    """
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(X, y), start=1
    ):
        print(f"\n{'='*50}")
        print(f"  Fold {fold_idx}/{folds}")
        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
        print(f"{'='*50}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # 在训练折上全量训练
        model = train_full_model(
            X_train_scaled,
            y_train,
            device,
            input_dim=X.shape[1],
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            lr=lr,
        )

        # 在验证折上评估
        model.eval()
        X_val_t = torch.FloatTensor(X_val_scaled).to(device)
        with torch.no_grad():
            probs = model(X_val_t).cpu().numpy().flatten()

        y_pred = (probs >= 0.5).astype(int)
        metrics = evaluate(y_val, y_pred, probs)
        fold_results.append(metrics)

        print(
            f"  Fold {fold_idx} -> "
            f"Acc={metrics['accuracy']:.4f}  "
            f"F1={metrics['f1']:.4f}  "
            f"AUC={metrics['auc']:.4f}"
        )

    return fold_results


def main():
    parser = argparse.ArgumentParser(description="BiLSTM 模型评估")
    parser.add_argument(
        "--folds", type=int, default=5, help="交叉验证折数 (默认5)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/bilstm",
        help="结果输出目录 (默认outputs/bilstm)",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="最大训练轮数 (默认100)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="批大小 (默认32)"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="早停耐心值 (默认10)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="学习率 (默认1e-3)"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认42)")
    args = parser.parse_args()

    # ---- 设置随机种子 ----
    set_seed(args.seed)

    # ---- 获取设备 ----
    device = get_device()
    print(f"Device: {device}")

    # ---- 加载数据 ----
    ide_logs, passed = load_ide_logs()
    X, y, student_ids = build_feature_matrix(ide_logs, passed)
    print(f"Data: X={X.shape}, y={y.shape}")
    print(f"  Passed: {sum(y)}, Failed: {len(y) - sum(y)}")

    # ---- 全量训练 + 交叉验证评估 ----
    print(f"\n开始 {args.folds} 折交叉验证评估...")
    print("（每折在全量训练数据上重新训练BiLSTM模型）")

    fold_results = evaluate_cross_validation(
        X,
        y,
        device,
        folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        lr=args.lr,
    )

    # ---- 汇总并打印结果 ----
    summary = summarize_fold_results(fold_results)
    print_results_table("BiLSTM (Evaluation)", summary)

    # ---- 保存评估结果 ----
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "evaluation.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "BiLSTM",
                "task": "evaluation",
                "n_folds": args.folds,
                "description": "全量训练+5折交叉验证评估",
                "hyperparameters": {
                    "hidden_dim": 64,
                    "num_layers": 2,
                    "dropout": 0.3,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "patience": args.patience,
                    "lr": args.lr,
                },
                "summary": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n评估结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
