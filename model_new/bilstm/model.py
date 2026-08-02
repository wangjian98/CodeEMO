"""
双向LSTM模型 - 序列方法

将46维特征视为单步序列输入，通过双向LSTM提取前向和后向特征表示，
用于学生早期风险预测（通过/不通过二分类）。
"""
import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """双向LSTM分类器

    架构:
        Linear(46->64) -> unsqueeze(batch,1,64) -> BiLSTM(64,hidden=64,layers=2)
        -> last output (128-dim) -> Dropout(0.3) -> Linear(128->64) -> ReLU
        -> Dropout(0.3) -> Linear(64->1) -> Sigmoid

    Args:
        input_dim: 输入特征维度 (默认46)
        hidden_dim: LSTM隐藏层维度 (默认64)
        num_layers: LSTM层数 (默认2)
        dropout: Dropout比率 (默认0.3)
    """

    def __init__(self, input_dim=46, hidden_dim=64, num_layers=2, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 特征嵌入: 将46维映射到hidden_dim
        self.feature_embedding = nn.Linear(input_dim, hidden_dim)

        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # 输出层: 128(hidden_dim*2) -> 64 -> 1
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """前向传播

        Args:
            x: (batch, 46) 或 (batch, seq_len, 46)

        Returns:
            output: (batch, 1) 概率值
        """
        # 接受 (batch, 46) 或 (batch, seq_len, 46)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch, 1, 46)

        # 特征嵌入 -> (batch, seq_len, hidden_dim)
        embedded = self.feature_embedding(x)

        # BiLSTM -> output: (batch, seq_len, hidden_dim*2)
        lstm_out, _ = self.lstm(embedded)

        # 取最后一步输出 -> (batch, hidden_dim*2)
        last_output = lstm_out[:, -1, :]

        # 全连接输出 -> (batch, 1)
        output = self.fc(last_output)
        return output

    def count_parameters(self):
        """统计可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(input_dim=46, hidden_dim=64):
    """创建双向LSTM分类器

    Args:
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度

    Returns:
        BiLSTMClassifier 实例
    """
    model = BiLSTMClassifier(input_dim=input_dim, hidden_dim=hidden_dim)
    print(f"BiLSTM parameters: {model.count_parameters()}")
    return model


if __name__ == "__main__":
    model = create_model(46)
    # 测试输入 (batch=32, features=46)
    x = torch.randn(32, 46)
    output = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
