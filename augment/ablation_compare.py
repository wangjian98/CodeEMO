"""
RF vs LSTM 46d 对比 + 4 类特征消融实验

46d 特征分 4 类:
  Cat 1 (idx 0-27):  事件基础统计 7×4 = 28d
  Cat 2 (idx 28-37): 行为轨迹 10d
  Cat 3 (idx 38-43): 情绪复合特征 6d
  Cat 4 (idx 44-45): 元信息 2d

消融实验:
  A. Full 46d       (baseline)
  B. -Cat1 = 18d    (移除事件基础统计)
  C. -Cat2 = 36d    (移除行为轨迹)
  D. -Cat3 = 40d    (移除情绪复合)
  E. -Cat4 = 44d    (移除元信息)
  F. -Cat1234 = 0d  (无特征, 仅 RF baseline)
  G. Only 7d        (原始事件计数 7d, 对比基线)

模型: Random Forest, LSTM
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, set_seed
from common.feature_engineering import build_feature_matrix
from common.evaluator import evaluate
from models.lstm.model import LSTMClassifier

# 46d 特征的 4 类索引
CAT_RANGES = {
    'Cat1_事件统计': (0, 28),    # 28d: 7 events × 4 stats
    'Cat2_行为轨迹': (28, 38),   # 10d
    'Cat3_情绪复合': (38, 44),   # 6d
    'Cat4_元信息':   (44, 46),   # 2d
}
ALL_IDX = list(range(46))


def get_indices(ablation):
    """根据消融名称返回特征索引列表"""
    if ablation == 'A_Full46d':
        return ALL_IDX
    elif ablation == 'B_NoCat1':
        return [i for i in ALL_IDX if not (CAT_RANGES['Cat1_事件统计'][0] <= i < CAT_RANGES['Cat1_事件统计'][1])]
    elif ablation == 'C_NoCat2':
        return [i for i in ALL_IDX if not (CAT_RANGES['Cat2_行为轨迹'][0] <= i < CAT_RANGES['Cat2_行为轨迹'][1])]
    elif ablation == 'D_NoCat3':
        return [i for i in ALL_IDX if not (CAT_RANGES['Cat3_情绪复合'][0] <= i < CAT_RANGES['Cat3_情绪复合'][1])]
    elif ablation == 'E_NoCat4':
        return [i for i in ALL_IDX if not (CAT_RANGES['Cat4_元信息'][0] <= i < CAT_RANGES['Cat4_元信息'][1])]
    elif ablation == 'F_OnlyCat1':
        return list(range(CAT_RANGES['Cat1_事件统计'][0], CAT_RANGES['Cat1_事件统计'][1]))
    elif ablation == 'G_Only7d':
        return list(range(7))  # 原始 7 维事件计数
    else:
        raise ValueError(f'Unknown ablation: {ablation}')


ABLATION_NAMES = {
    'A_Full46d':  'A. Full 46d (基线)',
    'B_NoCat1':   'B. -Cat1 事件统计 (18d)',
    'C_NoCat2':   'C. -Cat2 行为轨迹 (36d)',
    'D_NoCat3':   'D. -Cat3 情绪复合 (40d)',
    'E_NoCat4':   'E. -Cat4 元信息 (44d)',
    'F_OnlyCat1': 'F. 仅 Cat1 (28d)',
    'G_Only7d':   'G. 仅 7d 原始',
}


def train_lstm_one_fold(X_train, y_train, X_val, y_val, device,
                        epochs=100, batch_size=32, patience=10, lr=1e-3):
    model = LSTMClassifier(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    Xt = torch.FloatTensor(X_train).to(device)
    yt = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    Xv = torch.FloatTensor(X_val).to(device)
    yv = torch.FloatTensor(y_val).unsqueeze(1).to(device)

    best_v = float('inf')
    best_state = None
    pc = 0
    n = Xt.shape[0]
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            loss = criterion(model(Xt[idx]), yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            v_loss = criterion(model(Xv), yv).item()
        if v_loss < best_v:
            best_v = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1
            if pc >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = model(Xv).squeeze(-1).cpu().numpy()
    preds = (probs > 0.5).astype(int)
    return preds, probs


def train_rf_one_fold(X_train, y_train, X_val, y_val, n_estimators=200, random_state=42):
    """训练 Random Forest"""
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    probs = rf.predict_proba(X_val)[:, 1]
    preds = (probs > 0.5).astype(int)
    return preds, probs


def run_cv(X, y, model_type, name, device, n_folds=5, seed=42):
    """5 折交叉验证, 计算 P/R/A/F1/AUC, 同时找最优阈值下的 F1"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics_default = []
    fold_metrics_best = []
    all_y_val = []
    all_probs = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X[train_idx], X[test_idx]
        y_train, y_val = y[train_idx], y[test_idx]

        # 标准化 (RF 其实不需要, 但统一一下)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        if model_type == 'lstm':
            preds, probs = train_lstm_one_fold(X_train_s, y_train, X_val_s, y_val, device)
        else:
            preds, probs = train_rf_one_fold(X_train_s, y_train, X_val_s, y_val)

        # 默认 0.5
        m_def = {
            'accuracy': accuracy_score(y_val, preds),
            'precision': precision_score(y_val, preds, zero_division=0),
            'recall': recall_score(y_val, preds, zero_division=0),
            'f1': f1_score(y_val, preds, zero_division=0),
            'auc': roc_auc_score(y_val, probs),
        }
        fold_metrics_default.append(m_def)

        all_y_val.append(y_val)
        all_probs.append(probs)

    # 汇总默认 0.5
    keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    summary = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics_default]
        summary[f'{k}_mean'] = float(np.mean(vals))
        summary[f'{k}_std'] = float(np.std(vals))

    # 全局找最优阈值
    y_all = np.concatenate(all_y_val)
    p_all = np.concatenate(all_probs)
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        f1_t = f1_score(y_all, (p_all > t).astype(int), zero_division=0)
        if f1_t > best_f1:
            best_f1, best_t = f1_t, t
    pred_best = (p_all > best_t).astype(int)
    summary['best_t'] = float(best_t)
    summary['f1_best'] = float(best_f1)
    summary['precision_best'] = float(precision_score(y_all, pred_best, zero_division=0))
    summary['recall_best'] = float(recall_score(y_all, pred_best, zero_division=0))
    summary['accuracy_best'] = float(accuracy_score(y_all, pred_best))

    print(f'  {name}: '
          f'AUC={summary["auc_mean"]:.4f} '
          f'Acc@.5={summary["accuracy_mean"]:.4f} '
          f'P@.5={summary["precision_mean"]:.4f} '
          f'R@.5={summary["recall_mean"]:.4f} '
          f'F1@.5={summary["f1_mean"]:.4f} | '
          f'F1@best={summary["f1_best"]:.4f}@t={summary["best_t"]:.2f}')

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='outputs/ablation')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}\n')

    # 加载 46d 数据
    print('Loading 46d data...')
    ide_logs, passed = load_ide_logs()
    X_full, y, student_ids = build_feature_matrix(ide_logs, passed)
    print(f'X: {X_full.shape}, passed={int((y==1).sum())}, failed={int((y==0).sum())}\n')

    os.makedirs(args.output_dir, exist_ok=True)

    results = {}

    # 7 个消融实验 × 2 模型 = 14 次 CV
    for ablation_key in ['A_Full46d', 'B_NoCat1', 'C_NoCat2', 'D_NoCat3',
                          'E_NoCat4', 'F_OnlyCat1', 'G_Only7d']:
        idx = get_indices(ablation_key)
        X = X_full[:, idx]
        name = ABLATION_NAMES[ablation_key]
        print(f'\n{"="*80}')
        print(f'消融: {name} | 维度: {X.shape[1]}')
        print(f'{"="*80}')

        for model_type in ['rf', 'lstm']:
            full_name = f'{ablation_key}_{model_type}'
            print(f'\n>>> 模型: {model_type.upper()}')
            summary = run_cv(X, y, model_type, name, device,
                             n_folds=args.folds, seed=args.seed)
            results[full_name] = summary

    # 保存 JSON
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 输出对比表
    print('\n\n')
    print('=' * 120)
    print(' ' * 40 + 'RF vs LSTM 46d 消融实验对比')
    print('=' * 120)
    print(f'{"消融":<22} {"模型":<6} {"AUC":>7} {"Acc":>7} {"P":>7} {"R":>7} {"F1@.5":>7} {"F1@best":>9} {"BestT":>6}')
    print('-' * 120)

    for ablation_key in ['A_Full46d', 'B_NoCat1', 'C_NoCat2', 'D_NoCat3',
                          'E_NoCat4', 'F_OnlyCat1', 'G_Only7d']:
        name = ABLATION_NAMES[ablation_key]
        for model_type in ['rf', 'lstm']:
            full_name = f'{ablation_key}_{model_type}'
            r = results[full_name]
            short_name = name.split(' ')[0] + (' ' + name.split(' ')[1] if len(name.split(' ')) > 1 else '')
            print(f'{short_name:<22} {model_type.upper():<6} '
                  f'{r["auc_mean"]:.4f}  '
                  f'{r["accuracy_mean"]:.4f}  '
                  f'{r["precision_mean"]:.4f}  '
                  f'{r["recall_mean"]:.4f}  '
                  f'{r["f1_mean"]:.4f}  '
                  f'{r["f1_best"]:.4f}   '
                  f'{r["best_t"]:.2f}')
    print('-' * 120)

    # 消融贡献分析
    print('\n')
    print('=' * 120)
    print('  消融贡献分析 (Δ vs A_Full46d baseline, 负值表示删除后下降)')
    print('=' * 120)
    print(f'{"消融":<22} {"模型":<6} {"ΔAUC":>8} {"ΔAcc":>8} {"ΔP":>8} {"ΔR":>8} {"ΔF1@.5":>9} {"ΔF1@best":>10}')
    print('-' * 120)
    for ablation_key in ['B_NoCat1', 'C_NoCat2', 'D_NoCat3', 'E_NoCat4', 'F_OnlyCat1', 'G_Only7d']:
        name = ABLATION_NAMES[ablation_key]
        for model_type in ['rf', 'lstm']:
            base = results[f'A_Full46d_{model_type}']
            r = results[f'{ablation_key}_{model_type}']
            d_auc = r['auc_mean'] - base['auc_mean']
            d_acc = r['accuracy_mean'] - base['accuracy_mean']
            d_p = r['precision_mean'] - base['precision_mean']
            d_r = r['recall_mean'] - base['recall_mean']
            d_f1 = r['f1_mean'] - base['f1_mean']
            d_f1b = r['f1_best'] - base['f1_best']
            short_name = name.split(' ')[0] + (' ' + name.split(' ')[1] if len(name.split(' ')) > 1 else '')
            print(f'{short_name:<22} {model_type.upper():<6} '
                  f'{d_auc:+.4f}  '
                  f'{d_acc:+.4f}  '
                  f'{d_p:+.4f}  '
                  f'{d_r:+.4f}  '
                  f'{d_f1:+.4f}   '
                  f'{d_f1b:+.4f}')
    print('-' * 120)


if __name__ == '__main__':
    main()