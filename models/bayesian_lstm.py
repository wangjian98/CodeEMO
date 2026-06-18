"""
Bayesian LSTM模型 - 原论文方法
使用变分推理实现贝叶斯LSTM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianLSTM(nn.Module):
    """Bayesian LSTM使用变分推理"""
    
    def __init__(self, input_dim=46, hidden_dim=64, num_layers=2, dropout=0.3):
        super(BayesianLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 特征嵌入
        self.feature_embedding = nn.Linear(input_dim, hidden_dim)
        
        # 标准LSTM用于特征提取
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 变分输出层（预测均值和方差）
        self.fc_mean = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.fc_logvar = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, return_uncertainty=False):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        embedded = self.feature_embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_output = lstm_out[:, -1, :]
        
        mean = self.fc_mean(last_output)
        logvar = self.fc_logvar(last_output)
        
        # 使用sigmoid限制输出在[0,1]
        mean = torch.sigmoid(mean)
        
        if return_uncertainty:
            return mean, logvar
        return mean
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_bayesian_lstm(input_dim=46, hidden_dim=64):
    """创建Bayesian LSTM模型"""
    model = BayesianLSTM(input_dim=input_dim, hidden_dim=hidden_dim)
    print(f"Bayesian LSTM parameters: {model.count_parameters()}")
    return model

if __name__ == "__main__":
    model = create_bayesian_lstm(46)
    x = torch.randn(32, 10, 46)
    mean, logvar = model(x, return_uncertainty=True)
    print(f"Mean shape: {mean.shape}, Logvar shape: {logvar.shape}")
