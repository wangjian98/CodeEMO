#!/usr/bin/env python3
"""重新生成 unified_compare/per_class_summary.json，使用统一的 failed=1 标签

对 outputs/unified_compare 下每个子目录：
- 加载 probs.npy
- 加载统一 ground truth (UNIFIED_Y)
- 计算 failed_F1 / failed_P / failed_R / passed_F1 / passed_P / passed_R / cm
- 写出新 per_class_summary.json
"""
import json
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix,
)

OUT = "/home/ubuntu/CodeEMO/outputs/unified_compare"
UNIFIED_Y = "/home/ubuntu/CodeEMO/outputs/bi_lstm_trans_v2/labels.npy"
UNIFIED_F = "/home/ubuntu/CodeEMO/outputs/bi_lstm_trans_v2/fold_idx.npy"

# 标签约定：LSTM/BiLSTM/Mamba 46d 的 probs 是 P(passed)，需要 1-p 翻转
PROBS_FLIP_NEEDED = {
    ("LSTM", "46d"),
    ("BiLSTM", "46d"),
    ("Mamba", "46d"),
}


def name_to_combo(dir_name):
    """把目录名转成 (model, features)"""
    name = dir_name
    if name.endswith("_46d"):
        return ("LSTM" if "lstm" in name else
                "BiLSTM" if "bilstm" in name else
                "Mamba" if "mamba" in name else
                "RF" if "rf" in name else
                "Transformer" if "transformer" in name else
                "PureMLP" if "pure_mlp" in name else
                None, "46d")
    if name.endswith("_7dim"):
        m = ("LSTM" if "lstm" in name else
             "BiLSTM" if "bilstm" in name else
             "Mamba" if "mamba" in name else
             "RF" if "rf" in name else
             "Transformer" if "transformer" in name else
             None)
        return (m, "7dim")
    # HDM-Net / MRE / Weighted 等：没有 features 后缀
    return (None, "—")


def per_class(probs, labels):
    """计算 failed/passed 双类的指标"""
    yh = (probs > 0.5).astype(int)
    acc = accuracy_score(labels, yh)
    cm = confusion_matrix(labels, yh)  # [[TN, FP], [FN, TP]] for failed=1
    
    failed_P = precision_score(labels, yh, pos_label=1, zero_division=0)
    failed_R = recall_score(labels, yh, pos_label=1, zero_division=0)
    failed_F1 = f1_score(labels, yh, pos_label=1, zero_division=0)
    passed_P = precision_score(labels, yh, pos_label=0, zero_division=0)
    passed_R = recall_score(labels, yh, pos_label=0, zero_division=0)
    passed_F1 = f1_score(labels, yh, pos_label=0, zero_division=0)
    auc = roc_auc_score(labels, probs)
    
    return {
        "failed_F1": float(failed_F1),
        "passed_F1": float(passed_F1),
        "failed_P": float(failed_P),
        "failed_R": float(failed_R),
        "passed_P": float(passed_P),
        "passed_R": float(passed_R),
        "acc": float(acc),
        "auc": float(auc),
        "raw_pos_rate": float(labels.mean()),
        "cm": [int(x) for x in cm.flatten().tolist()],
    }


def main():
    y = np.load(UNIFIED_Y)
    print(f"统一 ground truth: n={len(y)}, fail_rate={y.mean():.4f}")
    
    results = []
    for sub in sorted(os.listdir(OUT)):
        sub_path = os.path.join(OUT, sub)
        if not os.path.isdir(sub_path):
            continue
        probs_path = os.path.join(sub_path, "probs.npy")
        labels_path = os.path.join(sub_path, "labels.npy")
        if not (os.path.exists(probs_path) and os.path.exists(labels_path)):
            continue
        
        probs = np.load(probs_path)
        # 检查 labels.npy 是否与 UNIFIED_Y 一致
        labels_local = np.load(labels_path)
        labels_match = (labels_local == y).all()
        
        # 翻转 probs (LSTM/BiLSTM/Mamba 46d 的 probs 是 P(passed))
        combo = name_to_combo(sub)
        if combo[0] is not None and combo in PROBS_FLIP_NEEDED:
            probs_eval = 1.0 - probs
        else:
            probs_eval = probs
        
        m = per_class(probs_eval, y)
        m["name"] = sub
        m["labels_consistent"] = bool(labels_match)
        m["probs_flipped"] = combo in PROBS_FLIP_NEEDED if combo[0] is not None else False
        results.append(m)
    
    out_path = os.path.join(OUT, "per_class_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 报告
    print(f"\n生成 {len(results)} 条记录 → {out_path}")
    print()
    print(f"{'模型':<35} | failed_F1 | passed_F1 | AUC")
    print("-" * 80)
    for r in results:
        flag = "" if r["labels_consistent"] else " ⚠️labels不一致"
        flip = " ↻" if r["probs_flipped"] else ""
        print(f"{r['name']:<35} | {r['failed_F1']:.4f}   | {r['passed_F1']:.4f}   | {r['auc']:.4f}{flag}{flip}")


if __name__ == "__main__":
    main()