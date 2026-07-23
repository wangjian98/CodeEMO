"""
BGM-Net: Behavior-Gated Mixture Network
面向CS1编程学生成绩的精准预测

核心创新:
  1. Entropy-Weighted Attention: 用Shannon熵作为事件类型级注意力权重
  2. Ratio Cross-Interaction: 比率特征的显式非线性交叉项
  3. Behavior Gate: 基于行为意图(比率特征)的动态路由融合

参数量: ~5,500 (LSTM的10%)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# 特征布局常量 (对应 feature_engineering.py 的46维布局)
# Cat1: [0:28] = 7事件类型 × 4统计量(mean, std, cv, entropy)
# Cat2: [28:38] = 10维轨迹特征 (BGM-Net不使用)
# Cat3: [38:44] = 6维比率特征(edit_ratio_mean/std, delete_ratio_mean/std, focus_ratio_mean/std)
# Cat4: [44:46] = 2维元信息(num_problems, total_events)

CAT1_DIM = 28          # 7 × 4
CAT3_DIM = 6           # 3 ratios × 2 stats
META_TOTAL_EVENTS_IDX = 45  # total_events 在46维中的位置
N_EVENT_TYPES = 7
STATS_PER_EVENT = 4    # mean, std, cv, entropy
ENTROPY_OFFSET = 3     # 每个事件类型的4个统计量中entropy是第4个(idx 3)

# Cat3内部布局: [edit_ratio_mean, edit_ratio_std, delete_ratio_mean, delete_ratio_std, focus_ratio_mean, focus_ratio_std]
EDIT_RATIO_MEAN_IDX = 0
DELETE_RATIO_MEAN_IDX = 2
FOCUS_RATIO_MEAN_IDX = 4


class EntropyWeightedAttention(nn.Module):
    """
    用每个事件类型的Shannon熵值作为注意力权重
    熵高 = 行为不确定性强 = 信息密度高 → 分配更多注意力
    """
    def __init__(self, n_events=N_EVENT_TYPES, n_stats=STATS_PER_EVENT):
        super().__init__()
        self.n_events = n_events
        self.n_stats = n_stats
        # 可学习温度参数
        self.temperature = nn.Parameter(torch.ones(1))
        # 额外的可学习缩放
        self.scale = nn.Parameter(torch.ones(n_events))

    def forward(self, cat1_features):
        """
        Args:
            cat1_features: (batch, 28) = (batch, 7_events × 4_stats)
        Returns:
            weighted_stats: (batch, 7) — 每个事件类型的熵加权统计摘要
            attention_weights: (batch, 7) — 注意力权重(用于可解释性)
        """
        batch_size = cat1_features.shape[0]
        # Reshape: (batch, 7, 4)
        reshaped = cat1_features.view(batch_size, self.n_events, self.n_stats)

        # 提取熵值: 每个事件类型的第4个统计量
        entropy_values = reshaped[:, :, ENTROPY_OFFSET]  # (batch, 7)

        # 计算注意力权重: softmax(entropy / temperature) * scale
        scaled_entropy = entropy_values * self.scale / (self.temperature.abs() + 1e-8)
        attention_weights = F.softmax(scaled_entropy, dim=-1)  # (batch, 7)

        # 用注意力权重加权所有4个统计量 → 每个事件类型得到一个加权汇总
        # attention_weights: (batch, 7) → (batch, 7, 1)
        weights_expanded = attention_weights.unsqueeze(-1)  # (batch, 7, 1)
        weighted = (reshaped * weights_expanded).sum(dim=1)  # (batch, 4)

        # 同时保留attention-weighted摘要和原始拼接
        # 拼接: [weighted_summary(4), top_entropy_stats(7)] → 11维
        # top_entropy_stats: 取每个事件类型的entropy值 (batch, 7)
        out = torch.cat([weighted, entropy_values], dim=-1)  # (batch, 11)
        return out, attention_weights


class IntentExpert(nn.Module):
    """
    意图专家分支: 处理Cat3比率特征 + 比率交叉项
    """
    def __init__(self, input_dim=CAT3_DIM, hidden_dim=32, output_dim=32, dropout=0.3):
        super().__init__()
        # 输入: 6原始比率 + 3交叉项 = 9维
        expanded_dim = input_dim + 3
        self.net = nn.Sequential(
            nn.Linear(expanded_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, ratio_features):
        """
        Args:
            ratio_features: (batch, 6) = [edit_m, edit_s, del_m, del_s, focus_m, focus_s]
        Returns:
            h_intent: (batch, 32)
        """
        edit_mean = ratio_features[:, EDIT_RATIO_MEAN_IDX]
        delete_mean = ratio_features[:, DELETE_RATIO_MEAN_IDX]
        focus_mean = ratio_features[:, FOCUS_RATIO_MEAN_IDX]

        # 比率交叉项 (3维)
        cross_edit_focus = edit_mean * focus_mean      # 编辑效率 × 专注度
        cross_edit_delete = edit_mean * delete_mean    # 编辑效率 × 探索性删除
        cross_focus_delete = focus_mean * delete_mean  # 专注度 × 删除比例

        cross = torch.stack([cross_edit_focus, cross_edit_delete, cross_focus_delete], dim=-1)
        expanded = torch.cat([ratio_features, cross], dim=-1)  # (batch, 9)
        return self.net(expanded)


class StatExpert(nn.Module):
    """
    统计专家分支: 处理Cat1事件统计特征 + 熵驱动注意力
    """
    def __init__(self, cat1_dim=CAT1_DIM, meta_dim=1, hidden_dim=64, output_dim=32, dropout=0.3):
        super().__init__()
        self.entropy_attention = EntropyWeightedAttention()
        # 输入: entropy_attention输出(11) + meta(1) = 12维 → 加上原始Cat1的某些关键统计量
        # 实际输入: attention_summary(11) + total_events(1) = 12
        # 再加上直接传入的cat1的关键统计量(选取mean和std, 14维) = 26维
        attention_out_dim = 11
        key_stats_dim = N_EVENT_TYPES * 2  # 每个事件类型的mean和std = 14
        input_dim = attention_out_dim + key_stats_dim + meta_dim  # 11 + 14 + 1 = 26

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, cat1_features, total_events):
        """
        Args:
            cat1_features: (batch, 28)
            total_events: (batch, 1)
        Returns:
            h_stat: (batch, 32)
            attention_weights: (batch, 7) — 用于可解释性
        """
        batch_size = cat1_features.shape[0]
        # 熵注意力
        attn_out, attn_weights = self.entropy_attention(cat1_features)  # (batch, 11), (batch, 7)

        # 提取关键统计量: 每个事件类型的mean和std
        reshaped = cat1_features.view(batch_size, N_EVENT_TYPES, STATS_PER_EVENT)
        key_stats = reshaped[:, :, :2].reshape(batch_size, -1)  # (batch, 14)

        # 拼接
        combined = torch.cat([attn_out, key_stats, total_events], dim=-1)  # (batch, 26)
        h_stat = self.net(combined)
        return h_stat, attn_weights


class BehaviorGate(nn.Module):
    """
    行为门控融合: 基于比率特征的动态路由
    g = sigmoid(W·[focus_ratio_mean, edit_ratio_mean, delete_ratio_mean])
    output = g ⊙ h_intent + (1-g) ⊙ h_stat
    """
    def __init__(self, output_dim=32):
        super().__init__()
        # 门控输入: 3个比率均值
        self.gate_net = nn.Sequential(
            nn.Linear(3, output_dim),
            nn.Sigmoid()
        )

    def forward(self, h_stat, h_intent, ratio_features):
        """
        Args:
            h_stat: (batch, 32) — 统计专家输出
            h_intent: (batch, 32) — 意图专家输出
            ratio_features: (batch, 6) — Cat3比率特征
        Returns:
            fused: (batch, 32)
            gate_values: (batch, 32) — 门控值(用于可解释性)
        """
        # 提取3个比率均值
        ratio_means = ratio_features[:, [EDIT_RATIO_MEAN_IDX, DELETE_RATIO_MEAN_IDX, FOCUS_RATIO_MEAN_IDX]]
        # (batch, 3)

        gate = self.gate_net(ratio_means)  # (batch, 32) ∈ (0,1)
        fused = gate * h_intent + (1 - gate) * h_stat
        return fused, gate


class BGMNet(nn.Module):
    """
    BGM-Net: Behavior-Gated Mixture Network

    输入: 46维特征 (与论文草稿完全一致)
    使用: Cat1(28) + Cat3(6) + total_events(1) = 35维有效输入
    输出: P(passed=1)

    可配置消融:
        use_gate=True/False (H1验证)
        use_entropy_attention=True/False (H2验证)
        use_cross_interaction=True/False (H3验证)
    """
    def __init__(self, use_gate=True, use_entropy_attention=True,
                 use_cross_interaction=True, dropout=0.3):
        super().__init__()
        self.use_gate = use_gate
        self.use_entropy_attention = use_entropy_attention
        self.use_cross_interaction = use_cross_interaction

        # Stat Expert
        if use_entropy_attention:
            self.stat_expert = StatExpert(dropout=dropout)
            stat_output_dim = 32
        else:
            # 退化: 直接用MLP处理Cat1+meta
            self.stat_expert = nn.Sequential(
                nn.Linear(CAT1_DIM + 1, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            stat_output_dim = 32

        # Intent Expert
        if use_cross_interaction:
            self.intent_expert = IntentExpert(dropout=dropout)
        else:
            self.intent_expert = nn.Sequential(
                nn.Linear(CAT3_DIM, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        intent_output_dim = 32

        # Behavior Gate
        if use_gate:
            self.gate = BehaviorGate(output_dim=32)

        # Classification Head
        fused_dim = stat_output_dim + intent_output_dim if not use_gate else 32
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 1),
            nn.Sigmoid()
        )

        # 保存attention权重
        self.last_attention = None
        self.last_gate = None

    def forward(self, x):
        """
        Args:
            x: (batch, 46) — 标准化后的46维特征
        Returns:
            prob: (batch, 1) — P(passed=1)
        """
        # 特征切片
        cat1 = x[:, 0:CAT1_DIM]                           # (batch, 28)
        cat3 = x[:, 38:44]                                # (batch, 6)
        total_events = x[:, META_TOTAL_EVENTS_IDX: META_TOTAL_EVENTS_IDX+1]  # (batch, 1)

        # Stat Expert
        if self.use_entropy_attention:
            h_stat, attn_weights = self.stat_expert(cat1, total_events)
            self.last_attention = attn_weights
        else:
            stat_input = torch.cat([cat1, total_events], dim=-1)
            h_stat = self.stat_expert(stat_input)

        # Intent Expert
        if self.use_cross_interaction:
            h_intent = self.intent_expert(cat3)
        else:
            h_intent = self.intent_expert(cat3)

        # Fusion
        if self.use_gate:
            fused, gate_values = self.gate(h_stat, h_intent, cat3)
            self.last_gate = gate_values
        else:
            fused = torch.cat([h_stat, h_intent], dim=-1)

        # Classify
        prob = self.classifier(fused)
        return prob

    def get_attention_weights(self):
        """获取最后一次forward的entropy attention权重"""
        return self.last_attention

    def get_gate_values(self):
        """获取最后一次forward的behavior gate值"""
        return self.last_gate


def count_parameters(model):
    """统计可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
