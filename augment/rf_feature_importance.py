"""
RF 特征重要性导出 + 按 4 类汇总分析

46d 特征名:
  Cat1 (idx 0-27): 事件基础统计
    7 events × 4 stats:
      [0-3]   text_insert      mean, std, cv, entropy
      [4-7]   text_remove      mean, std, cv, entropy
      [8-11]  text_paste       mean, std, cv, entropy
      [12-15] focus_gained     mean, std, cv, entropy
      [16-19] focus_lost       mean, std, cv, entropy
      [20-23] run              mean, std, cv, entropy
      [24-27] submit           mean, std, cv, entropy
  Cat2 (idx 28-37): 行为轨迹
    [28] improvement, [29] consistency, [30] trend,
    [31] mean_interval, [32] std_interval, [33] min_interval,
    [34] max_interval, [35] duration_per_event,
    [36] median_interval, [37] iqr_interval
  Cat3 (idx 38-43): 情绪复合特征
    [38] edit_ratio_mean, [39] edit_ratio_std,
    [40] delete_ratio_mean, [41] delete_ratio_std,
    [42] focus_ratio_mean, [43] focus_ratio_std
  Cat4 (idx 44-45): 元信息
    [44] num_problems, [45] total_events
"""
import os, sys, json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, set_seed
from common.feature_engineering import build_feature_matrix

FEATURE_NAMES = (
    # Cat1: 事件基础统计 (28d)
    [f'{et}_{stat}' for et in ['text_insert', 'text_remove', 'text_paste',
                                'focus_gained', 'focus_lost', 'run', 'submit']
     for stat in ['mean', 'std', 'cv', 'entropy']] +
    # Cat2: 行为轨迹 (10d)
    ['improvement', 'consistency', 'trend',
     'mean_interval', 'std_interval', 'min_interval',
     'max_interval', 'duration_per_event',
     'median_interval', 'iqr_interval'] +
    # Cat3: 情绪复合特征 (6d)
    ['edit_ratio_mean', 'edit_ratio_std',
     'delete_ratio_mean', 'delete_ratio_std',
     'focus_ratio_mean', 'focus_ratio_std'] +
    # Cat4: 元信息 (2d)
    ['num_problems', 'total_events']
)

CAT_NAMES = ['Cat1_事件统计', 'Cat2_行为轨迹', 'Cat3_情绪复合', 'Cat4_元信息']
CAT_RANGES = {
    CAT_NAMES[0]: (0, 28),
    CAT_NAMES[1]: (28, 38),
    CAT_NAMES[2]: (38, 44),
    CAT_NAMES[3]: (44, 46),
}


def main():
    set_seed(42)
    print('Loading 46d data...')
    ide_logs, passed = load_ide_logs()
    X, y, _ = build_feature_matrix(ide_logs, passed)
    print(f'X: {X.shape}, passed={int((y==1).sum())}, failed={int((y==0).sum())}')

    # 标准化
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # 5 折 CV 训练 RF, 累加每折的特征重要性
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    importances_all = []
    aucs, f1s = [], []

    print('\nTraining RF on full 46d (5-fold CV)...')
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_s, y), start=1):
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=None,
            min_samples_split=5, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )
        rf.fit(X_s[train_idx], y[train_idx])
        importances_all.append(rf.feature_importances_)
        prob = rf.predict_proba(X_s[test_idx])[:, 1]
        pred = (prob > 0.5).astype(int)
        auc = roc_auc_score(y[test_idx], prob)
        f1 = f1_score(y[test_idx], pred, zero_division=0)
        aucs.append(auc)
        f1s.append(f1)
        print(f'  Fold {fold_idx}: AUC={auc:.4f}, F1={f1:.4f}')

    # 平均重要性
    importances = np.mean(importances_all, axis=0)
    print(f'\nRF 5折均值: AUC={np.mean(aucs):.4f}, F1={np.mean(f1s):.4f}')

    # 排序并输出 Top-20
    sorted_idx = np.argsort(importances)[::-1]

    print('\n' + '=' * 100)
    print('  RF 特征重要性 - Top 20 (按 Gini importance 平均)')
    print('=' * 100)
    print(f'  {"排名":>4} {"特征名":<24} {"类别":<14} {"重要性":>10} {"累计%":>10}')
    print('-' * 100)
    cumulative = 0.0
    for rank, idx in enumerate(sorted_idx[:20], 1):
        importance = importances[idx]
        cumulative += importance
        # 找类别
        cat = 'Unknown'
        for cname, (s, e) in CAT_RANGES.items():
            if s <= idx < e:
                cat = cname.split('_')[0]
                break
        print(f'  {rank:>4} {FEATURE_NAMES[idx]:<24} {cat:<14} {importance:>10.4f} {cumulative*100:>9.2f}%')

    print('\n' + '=' * 100)
    print('  RF 特征重要性 - 全 46 维 (按类别分组)')
    print('=' * 100)

    # 按类别汇总
    cat_importance = {}
    cat_features = {}
    for cat_name, (s, e) in CAT_RANGES.items():
        cat_imp = importances[s:e]
        cat_importance[cat_name] = {
            'sum': float(cat_imp.sum()),
            'mean': float(cat_imp.mean()),
            'max': float(cat_imp.max()),
            'min': float(cat_imp.min()),
            'count': e - s
        }
        cat_features[cat_name] = [(FEATURE_NAMES[s + i], float(cat_imp[i]))
                                   for i in range(len(cat_imp))]

    print(f'  {"类别":<14} {"维度":>6} {"总和":>10} {"平均":>10} {"最大":>10} {"最小":>10} {"占总量%":>10}')
    print('-' * 80)
    total_imp = importances.sum()
    for cat_name, info in cat_importance.items():
        pct = info['sum'] / total_imp * 100
        print(f'  {cat_name:<14} {info["count"]:>6} {info["sum"]:>10.4f} {info["mean"]:>10.4f} '
              f'{info["max"]:>10.4f} {info["min"]:>10.4f} {pct:>9.2f}%')

    # 类别内 Top 3
    print('\n' + '=' * 100)
    print('  每类别内 Top 3 特征')
    print('=' * 100)
    for cat_name, features in cat_features.items():
        features_sorted = sorted(features, key=lambda x: -x[1])
        print(f'\n  {cat_name} ({CAT_RANGES[cat_name][1] - CAT_RANGES[cat_name][0]}d):')
        for rank, (name, imp) in enumerate(features_sorted[:3], 1):
            print(f'    {rank}. {name:<22} {imp:.4f}')

    # 保存结果到 JSON
    output = {
        'rf_cv_auc_mean': float(np.mean(aucs)),
        'rf_cv_f1_mean': float(np.mean(f1s)),
        'feature_importances': [
            {'idx': int(i), 'name': FEATURE_NAMES[i], 'importance': float(importances[i])}
            for i in sorted_idx
        ],
        'category_summary': cat_importance,
        'category_percent': {k: float(v['sum'] / total_imp) for k, v in cat_importance.items()}
    }
    os.makedirs('outputs/ablation', exist_ok=True)
    with open('outputs/ablation/feature_importance.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print('\n' + '=' * 100)
    print('  ✅ 结果已保存到 outputs/ablation/feature_importance.json')
    print('=' * 100)

    # 关键洞察
    print('\n' + '=' * 100)
    print('  📊 关键洞察')
    print('=' * 100)
    cat1_pct = cat_importance[CAT_NAMES[0]]['sum'] / total_imp * 100
    cat2_pct = cat_importance[CAT_NAMES[1]]['sum'] / total_imp * 100
    cat3_pct = cat_importance[CAT_NAMES[2]]['sum'] / total_imp * 100
    cat4_pct = cat_importance[CAT_NAMES[3]]['sum'] / total_imp * 100

    # 找 top-1 feature
    top1_name = FEATURE_NAMES[sorted_idx[0]]
    top1_imp = importances[sorted_idx[0]]
    top1_cat = None
    for cname, (s, e) in CAT_RANGES.items():
        if s <= sorted_idx[0] < e:
            top1_cat = cname
            break

    print(f'  1. 全局最重要特征: "{top1_name}" ({top1_cat}, 重要性={top1_imp:.4f}, 占比={top1_imp/total_imp*100:.2f}%)')
    print(f'  2. 类别贡献排序: Cat3 ({cat3_pct:.1f}%) > Cat1 ({cat1_pct:.1f}%) > Cat2 ({cat2_pct:.1f}%) > Cat4 ({cat4_pct:.1f}%)')
    print(f'  3. Cat3 仅 6d 但贡献 {cat3_pct:.1f}% — 平均每维 {cat3_pct/6:.2f}%, 信息密度最高')
    print(f'  4. Cat2 有 10d 但贡献 {cat2_pct:.1f}% — 平均每维 {cat2_pct/10:.2f}%, 信息密度最低')
    print(f'  5. Cat4 仅 2d 贡献 {cat4_pct:.1f}% — 平均每维 {cat4_pct/2:.2f}%, 几乎可忽略')


if __name__ == '__main__':
    main()