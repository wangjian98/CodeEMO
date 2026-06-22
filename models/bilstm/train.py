"""
双向LSTM模型训练脚本

使用5折分层交叉验证训练BiLSTM模型，对学生进行早期风险预测。
支持独立运行: python models/bilstm/train.py

用法:
    python models/bilstm/train.py
    python models/bilstm/train.py --folds 10
    python models/bilstm/train.py --output-dir outputs/bilstm
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
# 从 models/bilstm/train.py 向上回溯到项目根目录 CodeEMO/
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, get_device, set_seed
from common.feature_engineering import build_feature_matrix
from common.evaluator import evaluate, summarize_fold_results, print_results_table

from models.bilstm.model import create_model


def train_one_model(
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    input_dim=46,
    hidden_dim=64,
    epochs=100,
    batch_size=32,
    patience=10,
    lr=1e-3,
):
    """训练单个BiLSTM模型，返回验证集上的最佳概率预测和真实标签

    使用早停策略（early stopping），当验证集损失连续patience轮不下降时停止训练。
    """
    model = create_model(input_dim=input_dim, hidden_dim=hidden_dim)
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)

    n_samples = X_train_t.shape[0]
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # ---- 训练阶段 ----
        model.train()
        # 随机打乱训练数据
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

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ---- 验证阶段 ----
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()

        # ---- 早停判断 ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                k: v.clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"    Epoch {epoch}/{epochs}  "
                    f"train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}  "
                    f"-> 早停 (patience={patience})"
                )
                break

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"    Epoch {epoch}/{epochs}  "
                f"train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}"
            )

    # 加载最优权重进行预测
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_probs = model(X_val_t).cpu().numpy().flatten()

    val_pred = (val_probs >= 0.5).astype(int)
    return val_pred, val_probs


def main():
    parser = argparse.ArgumentParser(description="BiLSTM 模型训练 (5折交叉验证)")
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

    # ---- 5折分层交叉验证 ----
    skf = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(X, y), start=1
    ):
        print(f"\n{'='*50}")
        print(f"  Fold {fold_idx}/{args.folds}")
        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
        print(f"{'='*50}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 特征标准化 (在每折内独立拟合)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # 训练
        y_pred, y_prob = train_one_model(
            X_train_scaled,
            y_train,
            X_val_scaled,
            y_val,
            device,
            input_dim=X.shape[1],
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            lr=args.lr,
        )

        # 评估
        metrics = evaluate(y_val, y_pred, y_prob)
        fold_results.append(metrics)
        print(
            f"  Fold {fold_idx} -> "
            f"Acc={metrics['accuracy']:.4f}  "
            f"F1={metrics['f1']:.4f}  "
            f"AUC={metrics['auc']:.4f}"
        )

    # ---- 汇总结果 ----
    summary = summarize_fold_results(fold_results)
    print_results_table("BiLSTM", summary)

    # ---- 保存结果 ----
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "BiLSTM",
                "n_folds": args.folds,
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
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
