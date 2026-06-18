"""
Bi-LSTM模型 - 序列方法
"""
import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    """Bi-LSTM模型用于序列建模"""
    
    def __init__(self, input_dim=46, hidden_dim=64, num_layers=2, dropout=0.3):
        super(BiLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 将特征映射到隐藏维度
        self.feature_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Bi-LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, features) 或 (batch, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # 添加序列维度
        
        # 特征嵌入
        embedded = self.feature_embedding(x)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # 取最后一个输出
        last_output = lstm_out[:, -1, :]
        
        # 全连接输出
        output = self.fc(last_output)
        return output
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_bi_lstm(input_dim=46, hidden_dim=64):
    """创建Bi-LSTM模型"""
    model = BiLSTM(input_dim=input_dim, hidden_dim=hidden_dim)
    print(f"Bi-LSTM parameters: {model.count_parameters()}")
    return model

if __name__ == "__main__":
    model = create_bi_lstm(46)
    # 测试输入
    x = torch.randn(32, 10, 46)  # (batch, seq_len, features)
    output = model(x)
    print(f"Output shape: {output.shape}")
