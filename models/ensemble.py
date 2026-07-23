"""
D + BiLSTM Ensemble v3 (FIXED):
  - 用 D 的 labels 作为标准 (pass=0, fail=1, risk 表示挂科)
  - BiLSTM 输出 P(通过) (因为 train 中 y=1 是 pass), 所以要 1 - probs_bi 转换为 P(挂科)
  - 但 fold_idx 必须严格对应同一 split, 优先用 D 的 fold_idx
"""
import os
import json
import sys
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "/home/ubuntu/CodeEMO")
from common.data_loader import set_seed
from models.mamba.steps.step1_preprocessing import preprocess as preprocess_seq
from common.data_loader import load_ide_logs

# 重新生成 fold_idx 用 D 的数据加载顺序
print("加载 D 的数据 (确保 fold_idx 与 D 训练时一致)...")
ide_logs, passed = load_ide_logs()
samples, student_ids, labels_d = preprocess_seq(ide_logs, passed, max_events=2000)
labels_d = np.array(labels_d)

set_seed(42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_idx = np.zeros(len(labels_d), dtype=int)
for f, (train_idx, test_idx) in enumerate(skf.split(samples, labels_d)):
    fold_idx[test_idx] = f
np.save("/home/ubuntu/CodeEMO/outputs/bi_lstm_trans_v2/fold_idx.npy", fold_idx)

# 加载 probs
probs_d = np.load("/home/ubuntu/CodeEMO/outputs/bi_lstm_trans_v2/probs.npy")  # P(挂科)
probs_bi_raw = np.load("/home/ubuntu/CodeEMO/outputs/bilstm_save_probs/probs.npy")  # P(通过) 因为 BiLSTM y=1 = pass

# BiLSTM 概率翻转: P(挂科) = 1 - P(通过)
probs_bi = 1 - probs_bi_raw

print(f"\n[验证] BiLSTM auc (原始定义 y=1=pass):")
labels_bi_raw = np.load("/home/ubuntu/CodeEMO/outputs/bilstm_save_probs/labels.npy")
print(f"  AUC(P pass | y=pass) = {roc_auc_score(labels_bi_raw, probs_bi_raw):.4f}")
print(f"[验证] BiLSTM auc (翻转后 P fail vs D labels):")
print(f"  AUC(P fail | D labels) = {roc_auc_score(labels_d, probs_bi):.4f}")
print(f"[验证] D auc:")
print(f"  AUC(P fail | D labels) = {roc_auc_score(labels_d, probs_d):.4f}\n")


def evaluate_per_fold(probs, labels, fold_idx, threshold=0.5):
    metrics = []
    for f in range(5):
        m = fold_idx == f
        y = labels[m]
        p = probs[m]
        pred = (p > threshold).astype(int)
        metrics.append({
            'accuracy': float(accuracy_score(y, pred)),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'f1': float(f1_score(y, pred, zero_division=0)),
            'auc': float(roc_auc_score(y, p)),
        })
    return metrics


def summarize(metrics):
    return {k: f"{np.mean([m[k] for m in metrics]):.4f} ± {np.std([m[k] for m in metrics]):.4f}"
            for k in ['accuracy', 'precision', 'recall', 'f1', 'auc']}


print("=" * 65)
print("单独的模型")
print("=" * 65)
print(f"\n方案 D v2 (30维手工特征 + SE):")
m_d = evaluate_per_fold(probs_d, labels_d, fold_idx)
print(f"  {summarize(m_d)}")
for i, m in enumerate(m_d, 1):
    print(f"  Fold {i}: Acc={m['accuracy']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f}")

print(f"\nBiLSTM 46维 (概率翻转 P(挂科)=1-P(通过)):")
m_b = evaluate_per_fold(probs_bi, labels_d, fold_idx)
print(f"  {summarize(m_b)}")
for i, m in enumerate(m_b, 1):
    print(f"  Fold {i}: Acc={m['accuracy']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f}")

print("\n" + "=" * 65)
print("网格搜索集成权重 α * D + (1-α) * BiLSTM")
print("=" * 65)
best_auc, best_alpha = 0, 0
for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    p = alpha * probs_d + (1 - alpha) * probs_bi
    auc = roc_auc_score(labels_d, p)
    mark = ""
    if auc > best_auc:
        best_auc = auc
        best_alpha = alpha
        mark = "  ← 当前最佳"
    print(f"  α={alpha:.1f}: AUC={auc:.4f}{mark}")

print(f"\n>>> 最佳权重: α={best_alpha:.2f}, 全局 AUC={best_auc:.4f}")

# 5 折
print("\n" + "=" * 65)
print(f"最终集成 (α={best_alpha:.2f} * D v2 + {1-best_alpha:.2f} * BiLSTM 46维)")
print("=" * 65)
final_probs = best_alpha * probs_d + (1 - best_alpha) * probs_bi
metrics_e = evaluate_per_fold(final_probs, labels_d, fold_idx)
print(f"  5折汇总: {summarize(metrics_e)}")
for i, m in enumerate(metrics_e, 1):
    print(f"  Fold {i}: Acc={m['accuracy']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f}")

out_dir = "/home/ubuntu/CodeEMO/outputs/ensemble_final"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'results.json'), 'w') as f:
    json.dump({
        'note': '统一 D labels (pass=0, fail=1), BiLSTM 概率翻转 P(挂科)=1-P(通过), fold_idx 重新对齐',
        'best_alpha': float(best_alpha),
        'best_global_auc': float(best_auc),
        'D_v2_summary': summarize(m_d),
        'BiLSTM_summary': summarize(m_b),
        'ensemble_summary': summarize(metrics_e),
        'D_v2_metrics': m_d,
        'BiLSTM_metrics': m_b,
        'ensemble_metrics': metrics_e,
    }, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存: {out_dir}/results.json")
