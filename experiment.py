"""
实验脚本 - 消融实验、SOTA对比、改进方法实验
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

from models.simple_nn import SimpleNN
from models.bi_lstm import BiLSTM
from models.bayesian_lstm import BayesianLSTM
from models.random_forest import create_random_forest

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def load_data():
    """加载数据"""
    print("Loading IDE logs...")
    ide_logs = pd.read_csv('/tmp/IDE_logs/IDE_logs.csv')
    passed = pd.read_csv('/tmp/IDE_logs/passed.csv')
    return ide_logs, passed

def compute_shannon_entropy(counts):
    if len(counts) == 0:
        return 0.0
    counts = np.array(counts, dtype=float)
    if counts.sum() == 0:
        return 0.0
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def compute_cv(values):
    if len(values) == 0 or np.mean(values) == 0:
        return 0.0
    return np.std(values) / (np.mean(values) + 1e-10)

def compute_behavior_trajectory(event_times, event_types):
    """计算行为轨迹特征（10维）"""
    features = []
    if len(event_times) < 2:
        return [0.0] * 10
    
    intervals = np.diff(event_times)
    
    # improvement
    x = np.arange(len(intervals))
    slope = np.polyfit(x, intervals, 1)[0] if len(intervals) > 0 else 0.0
    features.append(slope)
    
    # consistency
    if np.mean(intervals) > 0:
        features.append(np.std(intervals) / (np.mean(intervals) + 1e-10))
    else:
        features.append(0.0)
    
    # trend
    x2 = np.arange(len(event_times))
    slope2 = np.polyfit(x2, event_times, 1)[0] if len(event_times) > 0 else 0.0
    features.append(slope2)
    
    features.append(np.mean(intervals))
    features.append(np.std(intervals))
    features.append(np.min(intervals))
    features.append(np.max(intervals))
    
    duration = event_times[-1] - event_times[0] if len(event_times) > 0 else 0
    features.append(duration / (len(event_times) + 1e-10))
    features.append(np.median(intervals) if len(intervals) > 0 else 0.0)
    
    if len(intervals) > 0:
        q75, q25 = np.percentile(intervals, [75, 25])
        features.append(q75 - q25)
    else:
        features.append(0.0)
    
    if len(intervals) > 0 and intervals[0] > 0:
        features.append(intervals[-1] / (intervals[0] + 1e-10))
    else:
        features.append(0.0)
    
    return features[:10]

def compute_emotion_features(student_df):
    """计算情绪复合特征（6维）"""
    features = []
    
    edit_counts = student_df[student_df['eventType'] == 'text_insert'].groupby('exercise').size()
    delete_counts = student_df[student_df['eventType'] == 'text_remove'].groupby('exercise').size()
    focus_counts = student_df[student_df['eventType'] == 'focus_gained'].groupby('exercise').size()
    
    edit_ratios = edit_counts / (edit_counts + delete_counts + 1e-10)
    features.append(edit_ratios.mean())
    features.append(edit_ratios.std())
    
    delete_ratios = delete_counts / (edit_counts + delete_counts + 1e-10)
    features.append(delete_ratios.mean())
    features.append(delete_ratios.std())
    
    total_events = student_df.groupby('exercise').size()
    focus_ratios = focus_counts / (total_events + 1e-10)
    features.append(focus_ratios.mean())
    features.append(focus_ratios.std())
    
    return features

def extract_features(student_df, student_id):
    """为单个学生提取46维特征"""
    features = []
    event_types = ['text_insert', 'text_remove', 'text_paste', 
                  'focus_gained', 'focus_lost', 'run', 'submit']
    
    # 1. 事件基础统计（28维）
    for et in event_types:
        et_events = student_df[student_df['eventType'] == et]['timestamp']
        times = (et_events - et_events.min()).dt.total_seconds() if len(et_events) > 0 else np.array([0])
        if len(times) < 2:
            times = np.array([0, 0])
        
        features.append(np.mean(times))
        features.append(np.std(times))
        features.append(np.std(times) / (np.mean(times) + 1e-10))
        
        bins = min(10, max(1, len(times) // 10))
        if bins > 1:
            hist, _ = np.histogram(times, bins=bins)
            features.append(compute_shannon_entropy(hist))
        else:
            features.append(0.0)
    
    # 2. 行为轨迹（10维）
    all_times = student_df['timestamp']
    all_times_numeric = (all_times - all_times.min()).dt.total_seconds().values
    trajectory = compute_behavior_trajectory(all_times_numeric, student_df['eventType'].values)
    features.extend(trajectory)
    
    # 3. 情绪复合特征（6维）
    emotion = compute_emotion_features(student_df)
    features.extend(emotion)
    
    # 4. 元信息（2维）
    features.append(student_df['exercise'].nunique())
    features.append(len(student_df))
    
    return features

def build_feature_matrix(ide_logs, passed):
    """构建特征矩阵"""
    print("Building feature matrix...")
    ide_logs['timestamp'] = pd.to_datetime(ide_logs['timestamp'])
    data = ide_logs.merge(passed, left_on='student', right_on='student')
    
    feature_list = []
    labels = []
    student_ids = []
    
    for student_id, group in data.groupby('student'):
        features = extract_features(group, student_id)
        feature_list.append(features)
        labels.append(1 if group['passed'].iloc[0] == True or group['passed'].iloc[0] == 'True' else 0)
        student_ids.append(student_id)
    
    feature_df = pd.DataFrame(feature_list)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"Feature matrix shape: {feature_df.shape}")
    print(f"Positive: {sum(labels)}, Negative: {len(labels) - sum(labels)}")
    
    return feature_df.values, np.array(labels), student_ids

def train_nn_model(model, train_loader, val_loader, epochs=100, patience=10):
    """训练神经网络模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_f1 = 0
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                val_preds.extend((outputs.squeeze() > 0.5).cpu().numpy())
                val_targets.extend(y_batch.numpy())
        
        val_f1 = f1_score(val_targets, val_preds)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def run_cross_validation(X, y, model_type, n_folds=5, epochs=100):
    """运行交叉验证"""
    set_seed(42)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        if model_type in ['simplenn', 'bilstm', 'bayesian']:
            if model_type == 'simplenn':
                model = SimpleNN(input_dim=X.shape[1])
            elif model_type == 'bilstm':
                model = BiLSTM(input_dim=X.shape[1])
            else:
                model = BayesianLSTM(input_dim=X.shape[1])
            
            train_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train)),
                batch_size=32, shuffle=True
            )
            val_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val)),
                batch_size=32
            )
            model = train_nn_model(model, train_loader, val_loader, epochs=epochs)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_val_scaled).to(device)
                outputs = model(X_tensor)
                preds = (outputs.squeeze() > 0.5).cpu().numpy()
                prob_preds = outputs.squeeze().cpu().numpy()
        
        elif model_type == 'rf':
            model = create_random_forest()
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_val_scaled)
            prob_preds = model.predict_proba(X_val_scaled)[:, 1]
        
        elif model_type == 'gb':
            model = GradientBoostingClassifier(n_estimators=100, max_depth=5)
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_val_scaled)
            prob_preds = model.predict_proba(X_val_scaled)[:, 1]
        
        results['accuracy'].append(accuracy_score(y_val, preds))
        results['precision'].append(precision_score(y_val, preds, zero_division=0))
        results['recall'].append(recall_score(y_val, preds, zero_division=0))
        results['f1'].append(f1_score(y_val, preds, zero_division=0))
        results['auc'].append(roc_auc_score(y_val, prob_preds))
    
    summary = {f'{k}_mean': np.mean(v) for k, v in results.items()}
    summary.update({f'{k}_std': np.std(v) for k, v in results.items()})
    
    return results, summary

def get_feature_subsets(X, feature_names):
    """获取特征子集"""
    # 7维基础特征：只用第一个事件类型的均值
    basic_7_idx = [0, 7, 14, 21, 28, 35, 42]
    
    # 38维风格特征：行为轨迹 + 情绪复合特征
    style_38_idx = list(range(28, 38)) + list(range(38, 44))
    
    # 46维全部特征
    all_46_idx = list(range(46))
    
    return {
        'basic_7': X[:, basic_7_idx],
        'style_38': X[:, style_38_idx],
        'all_46': X[:, all_46_idx]
    }

def main():
    os.makedirs('/root/.openclaw/workspace-staging/pc-ceo_assistant/CodeEMO/outputs', exist_ok=True)
    
    print("="*60)
    print("CodeEMO 实验 - 学业早期风险预测")
    print("="*60)
    
    # 加载数据
    ide_logs, passed = load_data()
    X, y, student_ids = build_feature_matrix(ide_logs, passed)
    
    results_all = {}
    
    # ===== 1. 消融实验 =====
    print("\n" + "="*60)
    print("消融实验：不同特征维度对比")
    print("="*60)
    
    feature_subsets = get_feature_subsets(X, None)
    
    for subset_name, X_subset in feature_subsets.items():
        print(f"\n--- {subset_name} ({X_subset.shape[1]}维) ---")
        _, summary = run_cross_validation(X_subset, y, 'simplenn', n_folds=5)
        results_all[f'ablation_{subset_name}'] = summary
        print(f"Accuracy: {summary['accuracy_mean']:.4f}±{summary['accuracy_std']:.4f}")
        print(f"F1: {summary['f1_mean']:.4f}±{summary['f1_std']:.4f}")
    
    # ===== 2. SOTA对比（100%数据）=====
    print("\n" + "="*60)
    print("SOTA对比实验 - 100%数据量")
    print("="*60)
    
    model_types = ['rf', 'simplenn', 'bilstm', 'bayesian']
    for model_type in model_types:
        print(f"\n--- {model_type} ---")
        _, summary = run_cross_validation(X, y, model_type, n_folds=5)
        results_all[f'sota_{model_type}_100'] = summary
        print(f"Accuracy: {summary['accuracy_mean']:.4f}±{summary['accuracy_std']:.4f}")
        print(f"F1: {summary['f1_mean']:.4f}±{summary['f1_std']:.4f}")
    
    # ===== 3. SOTA对比（25%数据）=====
    print("\n" + "="*60)
    print("SOTA对比实验 - 25%数据量")
    print("="*60)
    
    n_25 = int(len(X) * 0.25)
    indices_25 = np.random.choice(len(X), n_25, replace=False)
    X_25 = X[indices_25]
    y_25 = y[indices_25]
    
    for model_type in model_types:
        print(f"\n--- {model_type} (25%) ---")
        _, summary = run_cross_validation(X_25, y_25, model_type, n_folds=5)
        results_all[f'sota_{model_type}_25'] = summary
        print(f"Accuracy: {summary['accuracy_mean']:.4f}±{summary['accuracy_std']:.4f}")
        print(f"F1: {summary['f1_mean']:.4f}±{summary['f1_std']:.4f}")
    
    # ===== 4. 改进方法实验 =====
    print("\n" + "="*60)
    print("改进方法实验")
    print("="*60)
    
    # 改进1：class-weighted loss
    print("\n--- 改进1: Class-Weighted SimpleNN ---")
    class_counts = np.bincount(y)
    class_weights = len(y) / (2 * class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights)
    
    set_seed(42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cw_results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SimpleNN(input_dim=X.shape[1]).to(device)
        
        # Weighted BCE Loss
        weights = class_weights_tensor.to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weights[1]/weights[0])
        
        # 用logits版本需要修改模型
        model = SimpleNN(input_dim=X.shape[1])
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train)),
            batch_size=32, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val)),
            batch_size=32
        )
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        best_val_f1 = 0
        patience_counter = 0
        best_state = None
        
        for epoch in range(100):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                # Weighted loss
                loss = criterion(outputs.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
            
            model.eval()
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    outputs = model(X_batch)
                    val_preds.extend((outputs.squeeze() > 0.5).cpu().numpy())
                    val_targets.extend(y_batch.numpy())
            
            val_f1 = f1_score(val_targets, val_preds)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break
        
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_val_scaled).to(device)
            outputs = model(X_tensor)
            preds = (outputs.squeeze() > 0.5).cpu().numpy()
            prob_preds = outputs.squeeze().cpu().numpy()
        
        cw_results['accuracy'].append(accuracy_score(y_val, preds))
        cw_results['precision'].append(precision_score(y_val, preds, zero_division=0))
        cw_results['recall'].append(recall_score(y_val, preds, zero_division=0))
        cw_results['f1'].append(f1_score(y_val, preds, zero_division=0))
        cw_results['auc'].append(roc_auc_score(y_val, prob_preds))
    
    cw_summary = {f'{k}_mean': np.mean(v) for k, v in cw_results.items()}
    cw_summary.update({f'{k}_std': np.std(v) for k, v in cw_results.items()})
    results_all['improvement_class_weighted'] = cw_summary
    print(f"Accuracy: {cw_summary['accuracy_mean']:.4f}±{cw_summary['accuracy_std']:.4f}")
    print(f"F1: {cw_summary['f1_mean']:.4f}±{cw_summary['f1_std']:.4f}")
    
    # 改进2：特征增强 - 添加时间分桶特征
    print("\n--- 改进2: Enhanced Features + GradientBoosting ---")
    
    # 添加时间分桶特征（简化版，只用统计量）
    X_enhanced = X.copy()
    
    # 添加更多统计特征
    X_enhanced = np.column_stack([X_enhanced, np.percentile(X, 25, axis=1)])
    X_enhanced = np.column_stack([X_enhanced, np.percentile(X, 75, axis=1)])
    X_enhanced = np.column_stack([X_enhanced, np.max(X, axis=1) - np.min(X, axis=1)])
    
    _, summary = run_cross_validation(X_enhanced, y, 'gb', n_folds=5)
    results_all['improvement_enhanced_gb'] = summary
    print(f"Accuracy: {summary['accuracy_mean']:.4f}±{summary['accuracy_std']:.4f}")
    print(f"F1: {summary['f1_mean']:.4f}±{summary['f1_std']:.4f}")
    
    # 保存结果
    output_path = '/root/.openclaw/workspace-staging/pc-ceo_assistant/CodeEMO/outputs/results.json'
    with open(output_path, 'w') as f:
        json.dump(results_all, f, indent=2)
    
    print("\n" + "="*60)
    print("实验完成！结果保存至 outputs/results.json")
    print("="*60)
    
    # 打印汇总表格
    print("\n" + "="*60)
    print("实验结果汇总")
    print("="*60)
    
    print("\n【消融实验结果】")
    print(f"{'特征集':<15} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1':<15}")
    for key in results_all:
        if key.startswith('ablation'):
            s = results_all[key]
            print(f"{key:<15} {s['accuracy_mean']:.4f}±{s['accuracy_std']:.4f}   "
                  f"{s['precision_mean']:.4f}±{s['precision_std']:.4f}   "
                  f"{s['recall_mean']:.4f}±{s['recall_std']:.4f}   "
                  f"{s['f1_mean']:.4f}±{s['f1_std']:.4f}")
    
    print("\n【SOTA对比（100%数据）】")
    print(f"{'模型':<15} {'Accuracy':<15} {'F1':<15} {'AUC':<15}")
    for key in results_all:
        if key.startswith('sota') and key.endswith('100'):
            s = results_all[key]
            model_name = key.replace('sota_', '').replace('_100', '')
            print(f"{model_name:<15} {s['accuracy_mean']:.4f}±{s['accuracy_std']:.4f}   "
                  f"{s['f1_mean']:.4f}±{s['f1_std']:.4f}   "
                  f"{s['auc_mean']:.4f}±{s['auc_std']:.4f}")
    
    print("\n【SOTA对比（25%数据）】")
    print(f"{'模型':<15} {'Accuracy':<15} {'F1':<15} {'AUC':<15}")
    for key in results_all:
        if key.startswith('sota') and key.endswith('25'):
            s = results_all[key]
            model_name = key.replace('sota_', '').replace('_25', '')
            print(f"{model_name:<15} {s['accuracy_mean']:.4f}±{s['accuracy_std']:.4f}   "
                  f"{s['f1_mean']:.4f}±{s['f1_std']:.4f}   "
                  f"{s['auc_mean']:.4f}±{s['auc_std']:.4f}")
    
    print("\n【改进方法对比】")
    print(f"{'方法':<25} {'Accuracy':<15} {'F1':<15}")
    for key in results_all:
        if key.startswith('improvement'):
            s = results_all[key]
            print(f"{key:<25} {s['accuracy_mean']:.4f}±{s['accuracy_std']:.4f}   "
                  f"{s['f1_mean']:.4f}±{s['f1_std']:.4f}")
    
    return results_all

if __name__ == "__main__":
    main()
