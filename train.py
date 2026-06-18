"""
训练脚本 - 5折交叉验证
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
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

def train_nn_model(model, train_loader, val_loader, epochs=100, patience=10):
    """训练神经网络模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_f1 = 0
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证
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
        scheduler.step(val_f1)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model

def evaluate_model(model, X, y, model_type='nn', scaler=None):
    """评估模型"""
    if model_type == 'nn':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            outputs = model(X_tensor)
            preds = (outputs.squeeze() > 0.5).cpu().numpy()
    else:
        preds = model.predict(X)
    
    return {
        'accuracy': accuracy_score(y, preds),
        'precision': precision_score(y, preds),
        'recall': recall_score(y, preds),
        'f1': f1_score(y, preds)
    }

def cross_validate(X, y, model_type='simplenn', n_folds=5, epochs=100):
    """5折交叉验证"""
    set_seed(42)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n=== Fold {fold + 1}/{n_folds} ===")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        if model_type == 'simplenn':
            model = SimpleNN(input_dim=X.shape[1])
            train_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train)),
                batch_size=32, shuffle=True
            )
            val_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val)),
                batch_size=32
            )
            model = train_nn_model(model, train_loader, val_loader, epochs=epochs)
            metrics = evaluate_model(model, X_val_scaled, y_val, 'nn', scaler)
        
        elif model_type == 'bilstm':
            model = BiLSTM(input_dim=X.shape[1])
            train_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train)),
                batch_size=32, shuffle=True
            )
            val_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val)),
                batch_size=32
            )
            model = train_nn_model(model, train_loader, val_loader, epochs=epochs)
            metrics = evaluate_model(model, X_val_scaled, y_val, 'nn', scaler)
        
        elif model_type == 'bayesian':
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
            metrics = evaluate_model(model, X_val_scaled, y_val, 'nn', scaler)
        
        elif model_type == 'rf':
            model = create_random_forest()
            model.fit(X_train_scaled, y_train)
            metrics = evaluate_model(model, X_val_scaled, y_val, 'rf', scaler)
        
        for key in metrics:
            results[key].append(metrics[key])
        
        print(f"Fold {fold + 1} - Accuracy: {metrics['accuracy']:.4f}, "
              f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
              f"F1: {metrics['f1']:.4f}")
    
    # 计算平均值和标准差
    summary = {}
    for key in results:
        summary[f'{key}_mean'] = np.mean(results[key])
        summary[f'{key}_std'] = np.std(results[key])
    
    return results, summary

if __name__ == "__main__":
    print("Training script - use experiment.py for full experiments")
