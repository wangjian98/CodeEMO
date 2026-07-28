"""
PR-DE-Net 综合分析报告

1. 加载所有 PR-DE-Net 实验
2. Gate 行为分析（按类别、按预测正确性）
3. 错误分析（哪类样本被分错）
4. 与现有最佳模型的对比
5. 输出 README.md
"""
import os, sys, json
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.evaluator import evaluate, summarize_fold_results


def load(name):
    p = f'outputs/pr_de_net/{name}/results.json'
    return json.load(open(p)) if os.path.exists(p) else None


# ============================================================
# 1. 全部实验对比
# ============================================================
print("=" * 80)
print("1. PR-DE-Net 全部实验结果")
print("=" * 80)
all_results = {
    'full (a=1,b=1,g=2)':        load('full'),
    'no_gate':                   load('no_gate'),
    'single_loss':               load('single_loss'),
    'mini_full (18.8K)':         load('mini_full'),
    'mini_alpha_high':           load('mini_alpha_high'),
    'mini_balanced':             load('mini_balanced'),
    'mini_gamma_low':            load('mini_gamma_low'),
    'v2_a (gamma=3)':            load('v2_a_gamma3'),
    'v2_b (alpha=1.5)':          load('v2_b_alpha15'),
    'v2_c (beta=1.5)':           load('v2_c_beta15'),
    'v2_d (alpha=0.5)':          load('v2_d_alpha05'),
    'v2_ensemble (4 v2 avg)':    load('v2_ensemble'),
}
print(f"{'config':30s} {'F1':>10s} {'P':>10s} {'R':>10s} {'AUC':>10s}")
for name, d in all_results.items():
    if d is None:
        continue
    cv = d['cv_results']
    print(f"{name:30s} {cv['f1_mean']:.4f}±{cv['f1_std']:.3f} "
          f"{cv['precision_mean']:.4f}±{cv['precision_std']:.3f} "
          f"{cv['recall_mean']:.4f}±{cv['recall_std']:.3f} "
          f"{cv['auc_mean']:.4f}±{cv['auc_std']:.3f}")


# ============================================================
# 2. Gate 行为分析
# ============================================================
print("\n" + "=" * 80)
print("2. Gate 行为分析 (full config)")
print("=" * 80)
gates = np.load('outputs/pr_de_net/full/gates.npy')
labels = np.load('outputs/pr_de_net/full/labels.npy')
probs = np.load('outputs/pr_de_net/full/probs.npy')
fold_idx = np.load('outputs/pr_de_net/full/fold_idx.npy')
y_pred = (probs > 0.5).astype(int)

print(f"Gate 全局统计: mean={gates.mean():.4f}, std={gates.std():.4f}")
print(f"  • Failed 样本 (y=1): gate={gates[labels==1].mean():.4f}±{gates[labels==1].std():.4f} → 偏向 Branch A (RNN)")
print(f"  • Passed 样本 (y=0): gate={gates[labels==0].mean():.4f}±{gates[labels==0].std():.4f} → 偏向 Branch B (Trans)")
print()
# 路由比例
gate_per_class = {
    'failed (y=1)':  gates[labels == 1],
    'passed (y=0)':  gates[labels == 0],
}
for name, g in gate_per_class.items():
    print(f"  {name}: 偏 RNN (g<0.5) 占比 {(g<0.5).mean()*100:.1f}%, 偏 Trans (g>0.5) 占比 {(g>0.5).mean()*100:.1f}%")

# 按预测正确性
print()
print("Gate 与预测正确性:")
correct = (y_pred == labels)
for name, mask in [('正确', correct), ('错误', ~correct)]:
    if mask.sum() == 0:
        continue
    g_cls = {}
    for cls_name, yv in [('failed', 1), ('passed', 0)]:
        sub = mask & (labels == yv)
        if sub.sum() > 0:
            g_cls[cls_name] = gates[sub].mean()
    print(f"  {name}: failed gate={g_cls.get('failed', 0):.3f}, passed gate={g_cls.get('passed', 0):.3f}, n={mask.sum()}")


# ============================================================
# 3. 错误分析
# ============================================================
print("\n" + "=" * 80)
print("3. 错误分析: A/B 分支各错什么")
print("=" * 80)
p_A = np.load('outputs/pr_de_net/full/probs_A.npy')
p_B = np.load('outputs/pr_de_net/full/probs_B.npy')
err = ~correct
print(f"错误总数: {err.sum()}/{len(labels)} ({err.mean()*100:.1f}%)")

# 在错误样本上 A 和 B 谁更准？
print("\n错误样本子集（failed/passed）:")
for cls, cls_name in [(1, 'failed'), (0, 'passed')]:
    sub = err & (labels == cls)
    if sub.sum() == 0:
        continue
    n = sub.sum()
    pA_acc = ((p_A[sub] > 0.5).astype(int) == cls).mean()
    pB_acc = ((p_B[sub] > 0.5).astype(int) == cls).mean()
    pF_acc = ((probs[sub] > 0.5).astype(int) == cls).mean()
    g_mean = gates[sub].mean()
    print(f"  {cls_name} (n={n}): "
          f"A准确率={pA_acc*100:.0f}%, B准确率={pB_acc*100:.0f}%, "
          f"Final准确率={pF_acc*100:.0f}%, gate均值={g_mean:.3f}")


# ============================================================
# 4. 融合对比
# ============================================================
print("\n" + "=" * 80)
print("4. 融合方案对比（统一标签口径）")
print("=" * 80)
print(f"{'方案':35s} {'F1':>10s} {'AUC':>10s}")
candidates = [
    ('LSTM-46d (单)',           'outputs/unified_compare/lstm_46d/probs.npy',     None),
    ('BiLSTM-46d (单)',          'outputs/unified_compare/bilstm_46d/probs.npy',    None),
    ('Transformer-7d (单)',       'outputs/unified_compare/transformer_7dim/probs.npy', None),
    ('RF-7dim (单)',             'outputs/unified_compare/rf_7dim/probs.npy',      None),
    ('HDM-Net v2 T3 (单)',       'outputs/unified_compare/hdm_net_v2/probs.npy',   None),
    ('Weighted 1/3/1 (无 PR)',   'outputs/unified_compare/weighted_1_3_1/probs.npy', None),
    ('Stack top-3 LR',           'outputs/unified_compare/stack_top3_LR_C0.1/probs.npy', None),
    ('Per-fold stack top-5',     'outputs/unified_compare/perfold_stack_top5/probs.npy', None),
    ('PR-DE-Net (单)',           'outputs/pr_de_net/full/probs.npy',                None),
]
for name, p, _ in candidates:
    if not os.path.exists(p):
        continue
    prob = np.load(p)
    if name in ['LSTM-46d (单)', 'BiLSTM-46d (单)', 'Mamba-46d (单)']:
        prob = 1 - prob   # flip P(passed) → P(failed)
    fr = []
    for f in range(5):
        fr.append(evaluate(labels[fold_idx == f],
                           (prob[fold_idx == f] > 0.5).astype(int),
                           prob[fold_idx == f]))
    summary = summarize_fold_results(fr)
    print(f"{name:35s} {summary['f1_mean']:.4f}±{summary['f1_std']:.3f} "
          f"{summary['auc_mean']:.4f}±{summary['auc_std']:.3f}")

# 3-way + 4-way
f3 = json.load(open('outputs/pr_de_net/fusion_3way.json'))['best_3way']
print(f"{'★ 3-way (RF+HDM+PR-DE)':35s} {f3['f1']:.4f}        {f3['auc']:.4f}")
print(f"  weights: {dict(f3['weights'])}")
if os.path.exists('outputs/pr_de_net/fusion_4way.json'):
    f4 = json.load(open('outputs/pr_de_net/fusion_4way.json'))['best_4way']
    print(f"{'★ 4-way (RF+HDM+LSTM+PR-DE)':35s} {f4['f1']:.4f}        {f4['auc']:.4f}")
    print(f"  weights: {dict(f4['weights'])}")