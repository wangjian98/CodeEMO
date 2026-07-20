import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# 加载原始概率
p7 = np.load('outputs/unified_compare/bilstm_7dim/probs.npy')
p46 = np.load('outputs/unified_compare/bilstm_46d/probs.npy')
l7 = np.load('outputs/unified_compare/bilstm_7dim/labels.npy')   # 1=failed
l46 = np.load('outputs/unified_compare/bilstm_46d/labels.npy')   # 1=passed

# 统一约定: y=1=passed
y_passed = l46
p7_passed = 1 - p7   # 7d 原输出 P(failed=1), 反转为 P(passed=1)

print('=' * 100)
print('  标签一致性: ' + str(np.array_equal(1 - l7, l46)))
print('  原始 passed 正样本比例: ' + format(y_passed.mean(), '.3f') + ' (期望 0.336)')
print('=' * 100)

# 阈值扫描
thresholds = np.arange(0.05, 0.96, 0.05)
print()
print('  阈值    7d P     7d R     7d F1    46d P    46d R    46d F1   胜者')
print('-' * 100)

best_7d = (0, 0, 0, 0)
best_46d = (0, 0, 0, 0)
for t in thresholds:
    pred7 = (p7_passed > t).astype(int)
    pred46 = (p46 > t).astype(int)
    p7_, r7_, f7_ = precision_score(y_passed, pred7, zero_division=0), recall_score(y_passed, pred7, zero_division=0), f1_score(y_passed, pred7, zero_division=0)
    p46_, r46_, f46_ = precision_score(y_passed, pred46, zero_division=0), recall_score(y_passed, pred46, zero_division=0), f1_score(y_passed, pred46, zero_division=0)
    if f7_ > f46_:
        winner = '7d'
    elif f46_ > f7_:
        winner = '46d⭐'
    else:
        winner = 'tie'
    print('  {:.2f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}   {:.4f}   {}'.format(
        t, p7_, r7_, f7_, p46_, r46_, f46_, winner))
    if f7_ > best_7d[0]:
        best_7d = (f7_, t, p7_, r7_)
    if f46_ > best_46d[0]:
        best_46d = (f46_, t, p46_, r46_)

print('=' * 100)
print()
print('=== 最优阈值下的对比 ===')
print('  7d 最优:  F1=' + format(best_7d[0], '.4f') + '  @阈值=' + format(best_7d[1], '.2f') +
      '  P=' + format(best_7d[2], '.4f') + '  R=' + format(best_7d[3], '.4f'))
print('  46d 最优: F1=' + format(best_46d[0], '.4f') + '  @阈值=' + format(best_46d[1], '.2f') +
      '  P=' + format(best_46d[2], '.4f') + '  R=' + format(best_46d[3], '.4f'))
print()
if best_46d[0] > best_7d[0]:
    print('✅ 验证成功: 调阈值后 46d F1={:.4f} > 7d F1={:.4f}'.format(best_46d[0], best_7d[0]))
    print('   提升: {:+.4f}'.format(best_46d[0] - best_7d[0]))
else:
    print('❌ 验证失败: 即使调阈值, 46d F1={:.4f} <= 7d F1={:.4f}'.format(best_46d[0], best_7d[0]))

# 默认阈值 0.5 对比
print()
print('=== 默认阈值 0.5 对比 (用于参考) ===')
pred7_05 = (p7_passed > 0.5).astype(int)
pred46_05 = (p46 > 0.5).astype(int)
f7_05 = f1_score(y_passed, pred7_05, zero_division=0)
f46_05 = f1_score(y_passed, pred46_05, zero_division=0)
print('  7d @0.5:  F1={:.4f}'.format(f7_05))
print('  46d @0.5: F1={:.4f}'.format(f46_05))

# AUC 回顾
print()
print('=== AUC 回顾 (与阈值无关的真实能力指标) ===')
print('  7d  AUC: {:.4f}'.format(roc_auc_score(y_passed, p7_passed)))
print('  46d AUC: {:.4f}'.format(roc_auc_score(y_passed, p46)))