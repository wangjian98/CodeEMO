"""
SimpleNN轻量神经网络模型
输入层(46维) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.3) → Dense(16, ReLU) → Dense(1, Sigmoid)
"""
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    """SimpleNN轻量神经网络 - 约1.5K参数"""
    
    def __init__(self, input_dim=46):
        super(SimpleNN, self).__init__()
        
        self.model = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
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
        return self.model(x)
    
    def count_parameters(self):
        """计算参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_simple_nn(input_dim=46):
    """创建SimpleNN模型"""
    model = SimpleNN(input_dim)
    print(f"SimpleNN parameters: {model.count_parameters()}")
    return model

if __name__ == "__main__":
    model = create_simple_nn(46)
    print(model)
