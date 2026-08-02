"""
LSTM单向模型 - 用于学生早期风险预测分类

架构:
  Linear(46->64) -> unsqueeze(batch,1,64) -> LSTM(64, hidden=64, layers=2)
  -> 取最后一层输出 -> Dropout -> Linear(128->64) -> ReLU -> Dropout -> Linear(64->1) -> Sigmoid
"""
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """单向LSTM分类器，处理46维聚合特征

    将46维特征通过线性层映射到64维，增加序列维度后输入2层LSTM，
    取最终时刻输出进行分类。

    参数:
        input_dim: 输入特征维度 (默认46)
        hidden_dim: LSTM隐藏层维度 (默认64)
        num_layers: LSTM层数 (默认2)
        dropout: Dropout比率 (默认0.3)
    """

    def __init__(self, input_dim=46, hidden_dim=64, num_layers=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 特征嵌入: 46维 -> 64维
        self.feature_embedding = nn.Linear(input_dim, hidden_dim)

        # 单向LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # 输出分类头: LSTM输出维度为 hidden_dim * num_layers = 128
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * num_layers, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """前向传播

        输入: (batch, 46) 或 (batch, seq_len, 46)
        输出: (batch, 1) - 概率值
        """
        # 输入shape统一处理
        if len(x.shape) == 2:
            # (batch, 46) -> (batch, 46)
            x = self.feature_embedding(x)       # (batch, 64)
            x = x.unsqueeze(1)                  # (batch, 1, 64)
        else:
            # (batch, seq_len, 46)
            batch_size, seq_len, feat_dim = x.shape
            x = x.view(batch_size * seq_len, feat_dim)
            x = self.feature_embedding(x)        # (batch*seq_len, 64)
            x = x.view(batch_size, seq_len, -1)  # (batch, seq_len, 64)

        # LSTM前向
        lstm_out, (hidden, _) = self.lstm(x)     # lstm_out: (batch, seq_len, 64)

        # 取最后一层的隐藏状态
        # hidden: (num_layers, batch, hidden_dim)
        # 将所有层的隐藏状态拼接: (batch, hidden_dim * num_layers)
        batch_size = hidden.shape[1]
        hidden_concat = hidden.permute(1, 0, 2).contiguous()  # (batch, num_layers, hidden_dim)
        hidden_concat = hidden_concat.view(batch_size, -1)     # (batch, num_layers * hidden_dim)

        # 分类输出
        output = self.classifier(hidden_concat)  # (batch, 1)
        return output

    def count_parameters(self):
        """计算可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(input_dim=46):
    """创建LSTM分类器模型并移至可用设备

    参数:
        input_dim: 输入特征维度 (默认46)

    返回:
        LSTMClassifier 实例 (已移至CPU/GPU)
    """
    import sys
    import os

    # 定位common模块
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from common.data_loader import get_device

    device = get_device()
    model = LSTMClassifier(input_dim=input_dim)
    model = model.to(device)
    print(f"LSTM Classifier parameters: {model.count_parameters()}")
    print(f"Device: {device}")
    return model


if __name__ == "__main__":
    model = create_model(46)
    print(model)

    # 测试前向传播
    x = torch.randn(32, 46)
    output = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
