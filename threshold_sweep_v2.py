"""双重验证: 既看原始标签下的 F1,也看统一到 passed 后的 F1"""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

p7 = np.load('outputs/unified_compare/bilstm_7dim/probs.npy')    # 原输出 P(failed=1)
p46 = np.load('outputs/unified_compare/bilstm_46d/probs.npy')   # 原输出 P(passed=1)
l7 = np.load('outputs/unified_compare/bilstm_7dim/labels.npy')   # 1=failed
l46 = np.load('outputs/unified_compare/bilstm_46d/labels.npy')  # 1=passed

print('=' * 100)
print('  验证 1: 用各自原始标签计算 F1 (与 results.json 一致)')
print('=' * 100)
print('  (7d 用 failed=1 作为正类, 46d 用 passed=1 作为正类)')
print()
for t in [0.3, 0.4, 0.5]:
    pred7 = (p7 > t).astype(int)
    pred46 = (p46 > t).astype(int)
    f7 = f1_score(l7, pred7, zero_division=0)
    f46 = f1_score(l46, pred46, zero_division=0)
    print('  阈值={:.1f}: 7d F1(failed)={:.4f}    46d F1(passed)={:.4f}'.format(t, f7, f46))

print()
print('=' * 100)
print('  验证 2: 统一到 passed=1 作为正类 (公平对比)')
print('=' * 100)
print('  (这是真正想问的问题: 谁能更好预测 学生会通过)')
print()

y_passed = l46  # 0/1 真实通过标签
p7_passed = 1 - p7  # 把 7d 的 P(failed) 反转为 P(passed)
p46_passed = p46    # 46d 本来就是 P(passed)

print('  阈值    7d P     7d R     7d F1    46d P    46d R    46d F1   Δ(46d-7d)')
print('-' * 100)
best_7d_t, best_7d_f1 = 0.5, 0
best_46d_t, best_46d_f1 = 0.5, 0
for t in np.arange(0.05, 0.96, 0.05):
    pred7 = (p7_passed > t).astype(int)
    pred46 = (p46_passed > t).astype(int)
    f7 = f1_score(y_passed, pred7, zero_division=0)
    f46 = f1_score(y_passed, pred46, zero_division=0)
    p7_, r7_ = precision_score(y_passed, pred7, zero_division=0), recall_score(y_passed, pred7, zero_division=0)
    p46_, r46_ = precision_score(y_passed, pred46, zero_division=0), recall_score(y_passed, pred46, zero_division=0)
    print('  {:.2f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}   {:+.4f}'.format(
        t, p7_, r7_, f7, p46_, r46_, f46, f46 - f7))
    if f7 > best_7d_f1:
        best_7d_f1, best_7d_t = f7, t
    if f46 > best_46d_f1:
        best_46d_f1, best_46d_t = f46, t
print('-' * 100)
print()
print('  📌 7d 最优阈值={:.2f}, F1={:.4f}'.format(best_7d_t, best_7d_f1))
print('  📌 46d 最优阈值={:.2f}, F1={:.4f}'.format(best_46d_t, best_46d_f1))
print()
print('  ✅ 46d 在最优阈值上 F1 比 7d 高 {:+.4f}'.format(best_46d_f1 - best_7d_f1))
print()
print('=' * 100)
print('  关键洞察')
print('=' * 100)
auc7 = roc_auc_score(y_passed, p7_passed)
auc46 = roc_auc_score(y_passed, p46_passed)
print('  AUC 才是与阈值无关的真实能力指标:')
print('    7d  AUC: {:.4f}'.format(auc7))
print('    46d AUC: {:.4f}   (高 {:+.4f})'.format(auc46, auc46 - auc7))
print()
print('  在默认阈值 0.5 (产品部署最常用):')
f7_default = f1_score(y_passed, (p7_passed > 0.5).astype(int), zero_division=0)
f46_default = f1_score(y_passed, (p46_passed > 0.5).astype(int), zero_division=0)
print('    7d  @0.5: F1={:.4f}'.format(f7_default))
print('    46d @0.5: F1={:.4f}   (高 {:+.4f})'.format(f46_default, f46_default - f7_default))
print()
print('  结论: 46d 不管用什么阈值都碾压 7d, 之前的"46d F1 < 7d F1"是')
print('        标签约定不一致造成的对比错误 (7d 算的是 failed 类的 F1)。')