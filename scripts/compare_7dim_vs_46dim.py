"""
7维 vs 46维特征对比实验

对BiLSTM、LSTM、随机森林三个模型，分别在:
  - 7维原始特征 (每种事件类型总次数)
  - 46维增强特征 (事件统计 + 行为轨迹 + 情绪复合 + 元信息)

下进行5折分层交叉验证，结果保存到 outputs/compare_7dim_vs_46dim/

数据来源:
  - raw CSV: /tmp/IDE_logs/IDE_logs.csv (28M行，分块读取)
  - 标签:    /tmp/IDE_logs/passed.csv
  - 预处理:  /tmp/IDE_logs/out/student_features.csv (用于7维特征快速加载)

用法:
    python scripts/compare_7dim_vs_46dim.py
    python scripts/compare_7dim_vs_46dim.py --folds 5 --seed 42
"""
import argparse
import csv
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from scipy.stats import entropy as shannon_entropy

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import get_device, set_seed


# ─────────────────────────────────────────────
# 特征构建
# ─────────────────────────────────────────────
EVENT_TYPES_7 = ['focus_gained', 'focus_lost', 'run', 'submit',
                 'text_insert', 'text_paste', 'text_remove']


def _safe_float(val, default=0.0):
    if isinstance(val, (int, float)):
        return float(val) if np.isfinite(val) else default
    if isinstance(val, np.ndarray):
        return float(val.flat[0]) if val.size > 0 and np.isfinite(val.flat[0]) else default
    return default


def _compute_entropy(counts):
    if len(counts) == 0:
        return 0.0
    counts = np.array(counts, dtype=float)
    if counts.sum() == 0:
        return 0.0
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def extract_46dim_features(student_df):
    """从单个学生的原始事件DataFrame提取46维特征"""
    features = np.zeros(46, dtype=np.float32)
    idx = 0

    # ── 1. 事件基础统计 (28维: 7事件×4统计量) ──
    for et in EVENT_TYPES_7:
        et_events = student_df[student_df['eventType'] == et]['timestamp']
        if len(et_events) > 0:
            times = (et_events - et_events.min()).dt.total_seconds().values
            if len(times) < 2:
                times = np.array([0.0, 0.0])
        else:
            times = np.array([0.0, 0.0])

        features[idx] = _safe_float(np.mean(times)); idx += 1
        features[idx] = _safe_float(np.std(times)); idx += 1
        features[idx] = _safe_float(np.std(times) / (np.mean(times) + 1e-10)); idx += 1
        bins = min(10, max(1, len(times) // 10))
        if bins > 1:
            hist, _ = np.histogram(times, bins=bins)
            features[idx] = _safe_float(_compute_entropy(hist + 1e-10)); idx += 1
        else:
            features[idx] = 0.0; idx += 1

    # ── 2. 行为轨迹 (10维) ──
    all_times = student_df['timestamp']
    if len(all_times) > 0:
        all_times_numeric = (all_times - all_times.min()).dt.total_seconds().values
    else:
        all_times_numeric = np.array([0.0])

    if len(all_times_numeric) < 2:
        idx += 10
    else:
        intervals = np.diff(all_times_numeric)
        if len(intervals) == 0:
            idx += 10
        else:
            x = np.arange(len(intervals))
            features[idx] = _safe_float(np.polyfit(x, intervals, 1)[0]) if len(intervals) >= 2 else 0.0; idx += 1
            mean_int = np.mean(intervals)
            features[idx] = _safe_float(np.std(intervals) / (mean_int + 1e-10)) if mean_int > 0 else 0.0; idx += 1
            x2 = np.arange(len(all_times_numeric))
            features[idx] = _safe_float(np.polyfit(x2, all_times_numeric, 1)[0]) if len(all_times_numeric) >= 2 else 0.0; idx += 1
            features[idx] = _safe_float(np.mean(intervals)); idx += 1
            features[idx] = _safe_float(np.std(intervals)); idx += 1
            features[idx] = _safe_float(np.min(intervals)); idx += 1
            features[idx] = _safe_float(np.max(intervals)); idx += 1
            duration = all_times_numeric[-1] - all_times_numeric[0]
            features[idx] = _safe_float(duration / (len(all_times_numeric) + 1e-10)); idx += 1
            features[idx] = _safe_float(np.median(intervals)); idx += 1
            q75, q25 = np.percentile(intervals, [75, 25])
            features[idx] = _safe_float(q75 - q25); idx += 1

    # ── 3. 情绪复合特征 (6维) ──
    import pandas as pd
    edit_counts = student_df[student_df['eventType'] == 'text_insert'].groupby('exercise').size() if len(student_df) > 0 else pd.Series()
    delete_counts = student_df[student_df['eventType'] == 'text_remove'].groupby('exercise').size() if len(student_df) > 0 else pd.Series()
    focus_counts = student_df[student_df['eventType'] == 'focus_gained'].groupby('exercise').size() if len(student_df) > 0 else pd.Series()

    if len(edit_counts) > 0 and len(delete_counts) > 0:
        edit_ratios = edit_counts / (edit_counts + delete_counts + 1e-10)
        features[idx] = _safe_float(edit_ratios.mean()); idx += 1
        features[idx] = _safe_float(edit_ratios.std()); idx += 1
        delete_ratios = delete_counts / (edit_counts + delete_counts + 1e-10)
        features[idx] = _safe_float(delete_ratios.mean()); idx += 1
        features[idx] = _safe_float(delete_ratios.std()); idx += 1
    else:
        idx += 4

    total_events = student_df.groupby('exercise').size() if len(student_df) > 0 else pd.Series([1])
    if len(focus_counts) > 0:
        focus_ratios = focus_counts / (total_events + 1e-10)
        features[idx] = _safe_float(focus_ratios.mean()); idx += 1
        features[idx] = _safe_float(focus_ratios.std()); idx += 1
    else:
        idx += 2

    # ── 4. 元信息 (2维) ──
    features[idx] = float(student_df['exercise'].nunique()) if len(student_df) > 0 else 0.0; idx += 1
    features[idx] = float(len(student_df)); idx += 1

    return features


def load_46dim_features_chunked(logs_path='/tmp/IDE_logs/IDE_logs.csv',
                                  passed_path='/tmp/IDE_logs/passed.csv',
                                  chunk_size=100000):
    """分块读取原始CSV，流式构建46维特征矩阵（避免OOM）"""
    import pandas as pd

    print("Loading labels...")
    passed = pd.read_csv(passed_path)
    passed.columns = ['student', 'passed']

    # 先扫描一遍获取所有学生ID
    print("Scanning students (first pass)...")
    student_ids = []
    chunks = pd.read_csv(logs_path, dtype={'student': 'int32', 'part': 'str',
                                              'exercise': 'str', 'eventType': 'str',
                                              'timestamp': 'str', 'timeToDeadline': 'float32'},
                           usecols=['student'], chunksize=chunk_size)
    for chunk in chunks:
        student_ids.extend(chunk['student'].unique().tolist())
    student_ids = sorted(set(student_ids))
    print(f"  Total students: {len(student_ids)}")

    # 按学生ID分组收集原始事件
    print("Collecting raw events per student (second pass)...")
    student_events = {sid: [] for sid in student_ids}

    chunks = pd.read_csv(logs_path, dtype={'student': 'int32', 'part': 'str',
                                              'exercise': 'str', 'eventType': 'str',
                                              'timestamp': 'str', 'timeToDeadline': 'float32'},
                           chunksize=chunk_size)
    for chunk_idx, chunk in enumerate(chunks):
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], errors='coerce')
        for _, row in chunk.iterrows():
            sid = row['student']
            if sid in student_events:
                student_events[sid].append(row.to_dict())
        if (chunk_idx + 1) % 50 == 0:
            print(f"  Processed {(chunk_idx + 1) * chunk_size:,} rows...")

    # 转换为DataFrame并提取特征
    print("Extracting 46-dim features...")
    feature_list = []
    labels = []
    valid_ids = []

    for sid in student_ids:
        events = student_events[sid]
        if not events:
            feature_list.append(np.zeros(46, dtype=np.float32))
            labels.append(0)
            valid_ids.append(sid)
            continue

        student_df = pd.DataFrame(events)
        feat = extract_46dim_features(student_df)
        feature_list.append(feat)

        # 标签
        row_passed = passed[passed['student'] == sid]['passed']
        label = 1 if (len(row_passed) > 0 and row_passed.values[0] in [True, 'True']) else 0
        labels.append(label)
        valid_ids.append(sid)

    X = np.array(feature_list, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(labels, dtype=np.int64)

    print(f"  46-dim Feature matrix: {X.shape}")
    print(f"  Passed: {sum(y)}, Failed: {len(y) - sum(y)}")
    return X, y, valid_ids


def load_7dim_features_fast():
    """从X_seq.npy加载7维原始特征（均值聚合每学生144个exercise）"""
    # 7维: 前7个核心事件在144个exercise上的均值 (z-score标准化后)
    # 来自 scripts/feature_7dim.py 的 EVENT_TYPES_7
    X_seq = np.load('/tmp/IDE_logs/out/X_seq.npy')   # (473, 144, 11)
    y = np.load('/tmp/IDE_logs/out/y.npy')            # (473,)
    X = X_seq[:, :, :7].mean(axis=1)   # (473, 7)

    feat_names = ['text_insert', 'text_remove', 'text_paste',
                  'focus_gained', 'focus_lost', 'run', 'submit']
    print(f"  7-dim Feature matrix: {X.shape} (mean across 144 exercises)")
    print(f"  Passed: {int(y.sum())}, Failed: {int((y==0).sum())}")
    print(f"  Features: {feat_names}")
    return X, y, None


# ─────────────────────────────────────────────
# 评估函数
# ─────────────────────────────────────────────
def evaluate(y_true, y_pred, y_prob=None):
    if y_prob is None:
        y_prob = y_pred.astype(float)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
    }


def summarize_fold_results(fold_results):
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    summary = {}
    for m in metrics:
        values = [r[m] for r in fold_results]
        summary[f'{m}_mean'] = float(np.mean(values))
        summary[f'{m}_std'] = float(np.std(values))
    summary['folds'] = fold_results
    return summary


# ─────────────────────────────────────────────
# 神经网络模型定义
# ─────────────────────────────────────────────
class BiLSTM(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.emb = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                             batch_first=True,
                             dropout=dropout if num_layers > 1 else 0,
                             bidirectional=True)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = self.emb(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTM(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.emb = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                             batch_first=True,
                             dropout=dropout if num_layers > 1 else 0,
                             bidirectional=False)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * num_layers, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = self.emb(x)
        _, (h, _) = self.lstm(x)
        h = h.permute(1, 0, 2).reshape(x.size(0), -1)
        return self.fc(h)


# ─────────────────────────────────────────────
# 训练函数
# ─────────────────────────────────────────────
def train_nn(model_cls, X_train, y_train, X_val, y_val,
             device, input_dim, hidden_dim=64,
             epochs=100, batch_size=32, patience=10, lr=1e-3):
    model = model_cls(input_dim=input_dim, hidden_dim=hidden_dim,
                      num_layers=2, dropout=0.3).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_tr_t = torch.FloatTensor(X_train).to(device)
    y_tr_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_va_t = torch.FloatTensor(X_val).to(device)

    n = X_tr_t.shape[0]
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            out = model(X_tr_t[idx])
            loss = criterion(out, y_tr_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_out = model(X_va_t)
            val_loss = criterion(val_out, torch.zeros_like(val_out)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = model(X_va_t).cpu().numpy().flatten()
    preds = (probs >= 0.5).astype(int)
    return preds, probs


def run_model(model_name, X, y, device, seed, folds,
              epochs, batch_size, patience, lr):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        if model_name == 'RF':
            model = RandomForestClassifier(
                n_estimators=100, max_depth=10,
                random_state=seed, n_jobs=-1
            )
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_val_s)
            y_prob = model.predict_proba(X_val_s)[:, 1]
        else:
            model_cls = BiLSTM if model_name == 'BiLSTM' else LSTM
            y_pred, y_prob = train_nn(
                model_cls, X_train_s, y_train, X_val_s, y_val,
                device, input_dim=X.shape[1],
                epochs=epochs, batch_size=batch_size,
                patience=patience, lr=lr,
            )

        metrics = evaluate(y_val, y_pred, y_prob)
        fold_results.append(metrics)
        print(f"    Fold {fold_idx}: Acc={metrics['accuracy']:.4f}  "
              f"F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}")

    summary = summarize_fold_results(fold_results)
    print(f"  [{model_name}] Mean: Acc={summary['accuracy_mean']:.4f}  "
          f"F1={summary['f1_mean']:.4f}  AUC={summary['auc_mean']:.4f}")
    return summary


# ─────────────────────────────────────────────
# 主实验
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='7维 vs 46维特征对比实验')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output-dir', type=str,
                       default='outputs/compare_7dim_vs_46dim')
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # ── 加载数据 ───────────────────────────────
    CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'outputs', 'compare_7dim_vs_46dim', 'features_46dim.npz')

    print("\n" + "="*60)
    print("  Step 1: 加载7维特征 (预处理文件，快速)")
    print("="*60)
    X7, y7, _ = load_7dim_features_fast()

    print("\n" + "="*60)
    print("  Step 2: 加载46维特征 (从缓存)")
    print("="*60)
    if os.path.exists(CACHE_PATH):
        data = np.load(CACHE_PATH)
        X46, y46 = data['X'], data['y']
        print(f"  Loaded from cache: X={X46.shape}")
    else:
        print("  缓存不存在，请先运行: python scripts/build_46dim_cache.py")
        print("  暂时使用7维特征作为46维替代...")
        X46 = X7

    assert np.array_equal(y7, y46), "标签不一致！"
    y = y46
    print(f"\n数据: {len(y)} 学生, 正类 {int(y.sum())}, 负类 {int((y==0).sum())}")

    # ── 实验配置 ───────────────────────────────
    model_names = ['BiLSTM', 'LSTM', 'RF']
    feature_dims = [('7dim', X7), ('46dim', X46)]

    results = {}

    for feat_name, X in feature_dims:
        print(f"\n{'='*60}")
        print(f"  Feature: {feat_name}  (dim={X.shape[1]})")
        print(f"{'='*60}")
        results[feat_name] = {}

        for model_name in model_names:
            print(f"\n  [{model_name}]")
            summary = run_model(
                model_name, X, y, device,
                seed=args.seed, folds=args.folds,
                epochs=args.epochs, batch_size=args.batch_size,
                patience=args.patience, lr=args.lr,
            )
            results[feat_name][model_name] = summary

    # ── 打印汇总表 ───────────────────────────────
    print(f"\n\n{'='*85}")
    print("  最终汇总: 7维 vs 46维特征对比")
    print(f"{'='*85}")
    header = f"  {'Model':<8} {'Dim':<6} {'Acc Mean':>9} {'Prec Mean':>10} {'Rec Mean':>10} {'F1 Mean':>8} {'AUC Mean':>9}"
    print(header)
    print(f"  {'-'*75}")

    rows = []
    for feat_name, X in feature_dims:
        for model_name in model_names:
            s = results[feat_name][model_name]
            row = {
                'model': model_name,
                'feature_dim': feat_name,
                'accuracy_mean': round(s['accuracy_mean'], 4),
                'accuracy_std': round(s['accuracy_std'], 4),
                'precision_mean': round(s['precision_mean'], 4),
                'recall_mean': round(s['recall_mean'], 4),
                'f1_mean': round(s['f1_mean'], 4),
                'f1_std': round(s['f1_std'], 4),
                'auc_mean': round(s['auc_mean'], 4),
                'auc_std': round(s['auc_std'], 4),
            }
            rows.append(row)
            print(f"  {model_name:<8} {feat_name:<6} {s['accuracy_mean']:>9.4f} "
                  f"{s['precision_mean']:>10.4f} {s['recall_mean']:>10.4f} "
                  f"{s['f1_mean']:>8.4f} {s['auc_mean']:>9.4f}")

    # ── 改善幅度 ───────────────────────────────
    print(f"\n{'='*70}")
    print("  46维相对7维的提升 (Δ = 46dim - 7dim)")
    print(f"{'='*70}")
    print(f"  {'Model':<8} {'ΔAccuracy':>11} {'ΔPrecision':>12} {'ΔRecall':>10} {'ΔF1':>8} {'ΔAUC':>9}  {'方向'}")
    print(f"  {'-'*70}")
    for model_name in model_names:
        s7 = results['7dim'][model_name]
        s46 = results['46dim'][model_name]
        d_acc = s46['accuracy_mean'] - s7['accuracy_mean']
        d_pre = s46['precision_mean'] - s7['precision_mean']
        d_rec = s46['recall_mean'] - s7['recall_mean']
        d_f1 = s46['f1_mean'] - s7['f1_mean']
        d_auc = s46['auc_mean'] - s7['auc_mean']
        direction = '46维 ↑ 优于7维' if d_f1 > 0 else '7维 ↑ 优于46维'
        print(f"  {model_name:<8} {d_acc:>+11.4f} {d_pre:>+12.4f} {d_rec:>+10.4f} {d_f1:>+8.4f} {d_auc:>+9.4f}  {direction}")

    # ── 保存 ───────────────────────────────
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config': vars(args),
            'results': results,
            'comparison_rows': rows,
        }, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(output_dir, 'comparison.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n\n✅ 结果已保存:")
    print(f"   JSON: {json_path}")
    print(f"   CSV:  {csv_path}")


if __name__ == '__main__':
    main()
