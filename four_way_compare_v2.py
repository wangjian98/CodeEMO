"""4 模型 2x2 对比 v2: 修正 y_ref 选取"""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

configs = [
    ('LSTM 7d',   'outputs/unified_compare/lstm_7dim/probs.npy',   'outputs/unified_compare/lstm_7dim/labels.npy'),
    ('LSTM 46d',  'outputs/unified_compare/lstm_46d/probs.npy',    'outputs/unified_compare/lstm_46d/labels.npy'),
    ('BiLSTM 7d', 'outputs/unified_compare/bilstm_7dim/probs.npy',  'outputs/unified_compare/bilstm_7dim/labels.npy'),
    ('BiLSTM 46d','outputs/unified_compare/bilstm_46d/probs.npy', 'outputs/unified_compare/bilstm_46d/labels.npy'),
]

# Step 1: 寻找真正的 passed=1 标签 (pos_rate 应该是 ~0.336, 因为 159/473=0.336)
print('=' * 100)
print('  Step 1: 识别真实 passed 标签 (pos_rate 应该 ≈ 0.336)')
print('=' * 100)

y_passed = None
for name, p_path, l_path in configs:
    l = np.load(l_path)
    if abs(l.mean() - 0.336) < 0.05:  # 通过 = 1
        y_passed = l
        print('  ✓ 真实 passed 标签找到: {} (pos_rate={:.3f})'.format(name, l.mean()))
        break

if y_passed is None:
    print('  ⚠️ 没找到 pos_rate≈0.336 的标签, 使用全部 0/1 共识')
    # 投票: 大多数模型认为 1 的样本是真 1
    all_labels = [np.load(l_path) for _, _, l_path in configs]
    y_passed = (np.mean(all_labels, axis=0) > 0.5).astype(int)
    print('  共识 pos_rate: {:.3f}'.format(y_passed.mean()))

# Step 2: 为每个模型计算 "P(passed=1)" 概率
print()
print('=' * 100)
print('  Step 2: 统一到 passed=1 (公平对比)')
print('=' * 100)

results = []
for name, p_path, l_path in configs:
    p_raw = np.load(p_path)
    l_raw = np.load(l_path)
    # 判断此模型的 label 约定
    if abs(l_raw.mean() - 0.336) < 0.05:
        # 模型训练目标 = passed, probs 就是 P(passed)
        p_passed = p_raw
        conv = 'passed=1'
    else:
        # 模型训练目标 = failed, probs 是 P(failed)
        p_passed = 1 - p_raw
        conv = 'failed=1 → 反转'

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
        'name': name, 'auc': auc, 'conv': conv,
        'acc_default': acc, 'p_default': p, 'r_default': r, 'f1_default': f1,
        'best_t': best_t, 'acc_best': acc_best, 'p_best': p_best, 'r_best': r_best, 'f1_best': best_f1
    })
    print('  {:<12} {}: probs mean={:.3f}, P(passed) mean={:.3f}'.format(name, conv, p_raw.mean(), p_passed.mean()))

print()
print('  ' + '-' * 105)
print('  统一比较表 (预测 passed=1):')
print('  ' + '-' * 105)
print('  {:<12} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>5}'.format(
    '模型', 'AUC', 'Acc@.5', 'P@.5', 'R@.5', 'F1@.5', 'BestT', 'P@best', 'R@best', 'F1@best', 'AccB'))
print('  ' + '-' * 105)
for r in results:
    print('  {:<12} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.3f} | {:.4f} | {:.4f} | {:.4f} | {:.4f}'.format(
        r['name'], r['auc'], r['acc_default'], r['p_default'], r['r_default'], r['f1_default'],
        r['best_t'], r['p_best'], r['r_best'], r['f1_best'], r['acc_best']))
print('  ' + '-' * 105)

# Step 3: 综合排名
print()
print('=' * 100)
print('  Step 3: 综合排名 (0.5*AUC + 0.5*F1@best)')
print('=' * 100)
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

def get(name):
    return next(r for r in results if r['name'] == name)

lstm7, lstm46 = get('LSTM 7d'), get('LSTM 46d')
bilstm7, bilstm46 = get('BiLSTM 7d'), get('BiLSTM 46d')

print('  📊 特征工程效果 (7d → 46d):')
print('    LSTM:   AUC {:.4f} → {:.4f}  ({:+.4f})'.format(lstm7['auc'], lstm46['auc'], lstm46['auc'] - lstm7['auc']))
print('    LSTM:   F1  {:.4f} → {:.4f}  ({:+.4f})'.format(lstm7['f1_best'], lstm46['f1_best'], lstm46['f1_best'] - lstm7['f1_best']))
print('    BiLSTM: AUC {:.4f} → {:.4f}  ({:+.4f})'.format(bilstm7['auc'], bilstm46['auc'], bilstm46['auc'] - bilstm7['auc']))
print('    BiLSTM: F1  {:.4f} → {:.4f}  ({:+.4f})'.format(bilstm7['f1_best'], bilstm46['f1_best'], bilstm46['f1_best'] - bilstm7['f1_best']))

print()
print('  📊 双向 vs 单向 (LSTM → BiLSTM):')
print('    7d:   AUC {:.4f} → {:.4f}  ({:+.4f})'.format(lstm7['auc'], bilstm7['auc'], bilstm7['auc'] - lstm7['auc']))
print('    7d:   F1  {:.4f} → {:.4f}  ({:+.4f})'.format(lstm7['f1_best'], bilstm7['f1_best'], bilstm7['f1_best'] - lstm7['f1_best']))
print('    46d:  AUC {:.4f} → {:.4f}  ({:+.4f})'.format(lstm46['auc'], bilstm46['auc'], bilstm46['auc'] - lstm46['auc']))
print('    46d:  F1  {:.4f} → {:.4f}  ({:+.4f})'.format(lstm46['f1_best'], bilstm46['f1_best'], bilstm46['f1_best'] - lstm46['f1_best']))

print()
print('=' * 100)
print('  Step 5: 综合最优')
print('=' * 100)
best = ranked[0]
print('  🏆 综合最优: {}'.format(best['name']))
print('     AUC = {:.4f},  F1@best = {:.4f}  (threshold={:.2f})'.format(
    best['auc'], best['f1_best'], best['best_t']))
print('     Precision = {:.4f},  Recall = {:.4f},  Accuracy = {:.4f}'.format(
    best['p_best'], best['r_best'], best['acc_best']))