"""4 模型 2x2 对比: (LSTM/BiLSTM) × (7d/46d), 统一到 passed=1"""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

# 加载 4 个模型的概率
configs = [
    ('LSTM 7d',  'outputs/unified_compare/lstm_7dim/probs.npy',   'outputs/unified_compare/lstm_7dim/labels.npy'),
    ('LSTM 46d', 'outputs/unified_compare/lstm_46d/probs.npy',   'outputs/unified_compare/lstm_46d/labels.npy'),
    ('BiLSTM 7d',  'outputs/unified_compare/bilstm_7dim/probs.npy',  'outputs/unified_compare/bilstm_7dim/labels.npy'),
    ('BiLSTM 46d', 'outputs/unified_compare/bilstm_46d/probs.npy',  'outputs/unified_compare/bilstm_46d/labels.npy'),
]

# 先确认所有模型的标签约定
print('=' * 100)
print('  Step 1: 检查每个模型的标签约定 (y=1 是 passed 还是 failed?)')
print('=' * 100)
y_ref = None
for name, p_path, l_path in configs:
    p = np.load(p_path)
    l = np.load(l_path)
    print('  {}: probs mean={:.3f}, labels pos_rate={:.3f}  -> y=1={}'.format(
        name, p.mean(), l.mean(), 'failed' if l.mean() > 0.5 else 'passed'))
    if y_ref is None:
        y_ref = l
    else:
        # 验证是否同标签约定
        if np.array_equal(l, y_ref):
            print('       ✓ 与基准一致')
        elif np.array_equal(l, 1 - y_ref):
            print('       ✓ 与基准相反 (但 student 顺序相同)')
        else:
            print('       ✗ 与基准无关 - 需进一步检查')

print()
print('=' * 100)
print('  Step 2: 统一到 passed=1 (公平对比)')
print('=' * 100)

y_passed = y_ref  # 用第一个作为基准

results = []
for name, p_path, l_path in configs:
    p_raw = np.load(p_path)
    l_raw = np.load(l_path)
    # 判断: 若 l_raw 的 pos_rate > 0.5 则 y=1=failed, 反转
    if l_raw.mean() > 0.5:
        # 模型训练目标 = failed, probs 是 P(failed)
        p_passed = 1 - p_raw
    else:
        p_passed = p_raw

    auc = roc_auc_score(y_passed, p_passed)

    # 默认阈值 0.5
    pred = (p_passed > 0.5).astype(int)
    acc = accuracy_score(y_passed, pred)
    p = precision_score(y_passed, pred, zero_division=0)
    r = recall_score(y_passed, pred, zero_division=0)
    f1 = f1_score(y_passed, pred, zero_division=0)

    # 最优阈值
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.96, 0.02):
        pred_t = (p_passed > t).astype(int)
        f1_t = f1_score(y_passed, pred_t, zero_division=0)
        if f1_t > best_f1:
            best_f1, best_t = f1_t, t

    pred_best = (p_passed > best_t).astype(int)
    p_best = precision_score(y_passed, pred_best, zero_division=0)
    r_best = recall_score(y_passed, pred_best, zero_division=0)
    acc_best = accuracy_score(y_passed, pred_best)

    results.append({
        'name': name, 'auc': auc,
        'acc_default': acc, 'p_default': p, 'r_default': r, 'f1_default': f1,
        'best_t': best_t, 'acc_best': acc_best, 'p_best': p_best, 'r_best': r_best, 'f1_best': best_f1
    })

print()
print('  ' + '-' * 95)
print('  统一比较表 (预测 passed=1):')
print('  ' + '-' * 95)
print('  {:<12} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6}'.format(
    '模型', 'AUC', 'Acc@.5', 'P@.5', 'R@.5', 'F1@.5', 'BestT', 'P@best', 'R@best', 'F1@best'))
print('  ' + '-' * 95)
for r in results:
    print('  {:<12} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.3f} | {:.4f} | {:.4f} | {:.4f}'.format(
        r['name'], r['auc'], r['acc_default'], r['p_default'], r['r_default'], r['f1_default'],
        r['best_t'], r['p_best'], r['r_best'], r['f1_best']))
print('  ' + '-' * 95)

# 排序输出
print()
print('=' * 100)
print('  Step 3: 综合排名 (按 AUC + F1_best 加权)')
print('=' * 100)
# 综合得分 = 0.5*AUC + 0.5*F1@best
ranked = sorted(results, key=lambda x: -(0.5 * x['auc'] + 0.5 * x['f1_best']))
print('  ' + '{:<4} {:<12} {:>8} {:>8} {:>10}'.format('排名', '模型', 'AUC', 'F1@best', '综合得分'))
for i, r in enumerate(ranked, 1):
    score = 0.5 * r['auc'] + 0.5 * r['f1_best']
    medal = ['🥇', '🥈', '🥉', '  '][i - 1]
    print('  {} {}{:<12} {:>8.4f} {:>8.4f} {:>10.4f}'.format(
        medal, ' ', r['name'], r['auc'], r['f1_best'], score))

print()
print('=' * 100)
print('  Step 4: 关键洞察')
print('=' * 100)
auc_lstm7 = next(r for r in results if r['name'] == 'LSTM 7d')['auc']
auc_lstm46 = next(r for r in results if r['name'] == 'LSTM 46d')['auc']
auc_bilstm7 = next(r for r in results if r['name'] == 'BiLSTM 7d')['auc']
auc_bilstm46 = next(r for r in results if r['name'] == 'BiLSTM 46d')['auc']

f1_lstm7 = next(r for r in results if r['name'] == 'LSTM 7d')['f1_best']
f1_lstm46 = next(r for r in results if r['name'] == 'LSTM 46d')['f1_best']
f1_bilstm7 = next(r for r in results if r['name'] == 'BiLSTM 7d')['f1_best']
f1_bilstm46 = next(r for r in results if r['name'] == 'BiLSTM 46d')['f1_best']

print('  📊 特征工程效果 (7d → 46d):')
print('    LSTM:  AUC {:.4f} → {:.4f}  (+{:.4f})'.format(auc_lstm7, auc_lstm46, auc_lstm46 - auc_lstm7))
print('    LSTM:  F1  {:.4f} → {:.4f}  (+{:.4f})'.format(f1_lstm7, f1_lstm46, f1_lstm46 - f1_lstm7))
print('    BiLSTM: AUC {:.4f} → {:.4f}  (+{:.4f})'.format(auc_bilstm7, auc_bilstm46, auc_bilstm46 - auc_bilstm7))
print('    BiLSTM: F1  {:.4f} → {:.4f}  (+{:.4f})'.format(f1_bilstm7, f1_bilstm46, f1_bilstm46 - f1_bilstm7))
print()
print('  📊 双向 vs 单向效果 (LSTM → BiLSTM):')
print('    7d:  AUC {:.4f} → {:.4f}  ({:+.4f})'.format(auc_lstm7, auc_bilstm7, auc_bilstm7 - auc_lstm7))
print('    7d:  F1  {:.4f} → {:.4f}  ({:+.4f})'.format(f1_lstm7, f1_bilstm7, f1_bilstm7 - f1_lstm7))
print('    46d: AUC {:.4f} → {:.4f}  ({:+.4f})'.format(auc_lstm46, auc_bilstm46, auc_bilstm46 - auc_lstm46))
print('    46d: F1  {:.4f} → {:.4f}  ({:+.4f})'.format(f1_lstm46, f1_bilstm46, f1_bilstm46 - f1_lstm46))

print()
print('=' * 100)
print('  Step 5: 综合最优选择')
print('=' * 100)
best_overall = ranked[0]
print('  🏆 综合最优: {}'.format(best_overall['name']))
print('     AUC = {:.4f},  F1@best = {:.4f}  (threshold={:.2f})'.format(
    best_overall['auc'], best_overall['f1_best'], best_overall['best_t']))
print('     Precision = {:.4f},  Recall = {:.4f}'.format(
    best_overall['p_best'], best_overall['r_best']))