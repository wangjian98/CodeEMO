"""
Transformer分类器模型 - 将46维特征分组为伪序列进行建模

架构说明:
  1. 将46维特征向量切分为4个段(segment), 每段11维 (共44维, 末尾2维截断)
  2. 通过线性投影将每段映射到d_model维度, 形成伪序列
  3. 加入可学习的位置编码
  4. 经过多层TransformerEncoder进行特征交互
  5. 对序列输出做均值池化后送入分类头
"""
import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    """基于Transformer编码器的二分类模型

    将46维特征分割为n_segments个段, 每段投影到d_model维度,
    作为Transformer的输入序列, 最终输出通过sigmoid得到通过概率。

    Args:
        input_dim: 输入特征维度 (默认46)
        d_model: Transformer内部维度 (默认64)
        nhead: 多头注意力头数 (默认4)
        num_layers: Transformer编码器层数 (默认3)
        dropout: Dropout比率 (默认0.2)
    """

    def __init__(self, input_dim=46, d_model=64, nhead=4, num_layers=3, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # 将input_dim切分为n_segments个段
        self.n_segments = 4
        self.segment_size = input_dim // self.n_segments  # 46 // 4 = 11

        # 段投影: 将每个segment_size维的段投影到d_model维度
        self.segment_proj = nn.Linear(self.segment_size, d_model)

        # 可学习的位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_segments, d_model) * 0.02)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """前向传播

        Args:
            x: 输入特征 (batch_size, input_dim)

        Returns:
            输出概率 (batch_size, 1), 值域[0, 1], 1=通过
        """
        batch_size = x.shape[0]

        # 截取可用部分并重塑为序列: (batch, n_segments, segment_size)
        usable = self.n_segments * self.segment_size
        segments = x[:, :usable].view(batch_size, self.n_segments, self.segment_size)

        # 投影到d_model维度: (batch, n_segments, d_model)
        tokens = self.segment_proj(segments)

        # 加入位置编码
        tokens = tokens + self.pos_embed

        # Transformer编码: (batch, n_segments, d_model)
        out = self.transformer(tokens)

        # 均值池化: (batch, d_model)
        pooled = out.mean(dim=1)

        # 分类输出: (batch, 1)
        return self.classifier(pooled)


def create_model(input_dim=46, **kwargs):
    """创建Transformer分类器模型的工厂函数

    Args:
        input_dim: 输入特征维度 (默认46)
        **kwargs: 传递给TransformerClassifier的额外参数

    Returns:
        TransformerClassifier 实例
    """
    defaults = {
        'd_model': 64,
        'nhead': 4,
        'num_layers': 3,
        'dropout': 0.2,
    }
    defaults.update(kwargs)
    return TransformerClassifier(input_dim=input_dim, **defaults)


if __name__ == '__main__':
    # 简单的模型结构测试
    model = create_model(input_dim=46)
    print(model)
    print(f"\n总参数数: {sum(p.numel() for p in model.parameters()):,}")

    # 测试前向传播
    dummy_input = torch.randn(8, 46)
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
