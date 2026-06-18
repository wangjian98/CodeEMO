"""
Mamba-SSM (Selective State Space Model) Implementation for Student Behavior Modeling
Based on: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Albert Gu & Tri Dao, 2023)

This module implements the core S6 (Selective SSM) mechanism and the Mamba block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat


class SiLU(nn.Module):
    """SiLU (Sigmoid Linear Unit) / Swish activation"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * output


class S6Block(nn.Module):
    """
    S6: Selective State Space Model core mechanism
    
    Key innovation of Mamba: parameters A, B, C are computed from input (selective),
    unlike classical SSM where they are fixed. This allows the model to dynamically
    decide what to remember and what to forget.
    
    State: h_k = A * h_{k-1} + B * x_k
    Output: y_k = C * h_k
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dtype=torch.float32):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.dtype = dtype
        
        # Input projection (dim -> d_inner)
        self.input_proj = nn.Linear(d_model, self.d_inner)
        
        # Conv branch for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            dtype=dtype
        )
        
        # SSM parameters (SELECTIVE - computed from input)
        # d_inner -> 1 for dt, d_inner -> d_state for B, d_inner -> d_state for C
        self.dt_rank = math.ceil(self.d_inner / 16) if self.d_inner > 16 else self.d_inner
        
        # Project x -> (dt, B, C) parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, dtype=dtype, bias=False)
        
        # dt initialization (log-space)
        self.dt_init = nn.Parameter(torch.empty(self.dt_rank, dtype=dtype))
        nn.init.uniform_(self.dt_init, -1.0, 0.0)
        
        # A matrix (state transition) - initialized to normal distribution
        A = repeat(
            torch.arange(1, d_state + 1, dtype=dtype), 
            'n -> d n', d=self.d_inner
        ).clone()
        A = A.log()  # Log-space for stability
        self.A = nn.Parameter(A)
        
        # D term (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner, dtype=dtype))
        
        # Output projection
        self.output_proj = nn.Linear(self.d_inner, d_model, dtype=dtype, bias=False)
        
        self.act = SiLU()
    
    def selective_scan(self, x, dt, A, B, C, D):
        """
        Parallel selective scan algorithm (hardware-aware)
        
        x: (batch, seq_len, d_inner)
        dt: (batch, seq_len, dt_rank)
        A: (d_inner, d_state)
        B: (batch, seq_len, d_state)
        C: (batch, seq_len, d_state)
        D: (d_inner,)
        
        Returns: (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        # Discretize dt (continuous -> discrete step size)
        dt = F.softplus(dt)  # Ensure positive
        
        # Discretize A and B: dA = exp(dt * A), dB = dt * B
        # dt: (batch, seq, d_inner), A: (d_inner, d_state)
        # dA: (batch, seq, d_inner, 1) after broadcasting
        dt_expanded = dt.unsqueeze(-1)  # (batch, seq, d_inner, 1)
        A_expanded = A.unsqueeze(0).unsqueeze(0)  # (1, 1, d_inner, d_state)
        dA = torch.exp(dt_expanded * A_expanded)  # (batch, seq, d_inner, d_state)
        
        # dB: (batch, seq, d_inner) * (batch, seq, d_state) -> (batch, seq, d_inner, d_state)
        B_expanded = B.unsqueeze(2)  # (batch, seq, 1, d_state)
        dt_expanded2 = dt.unsqueeze(-1)  # (batch, seq, d_inner, 1)
        dB = dt_expanded2 * B_expanded  # (batch, seq, d_inner, d_state)
        
        # Scan computation (sequential for now - parallel scan is more complex)
        h = torch.zeros(batch, d_inner, d_state, dtype=x.dtype, device=x.device)
        ys = []
        
        for t in range(seq_len):
            # dA[:, t]: (batch, d_inner, d_state), h: (batch, d_inner, d_state)
            # dB[:, t]: (batch, d_inner, d_state), x[:, t]: (batch, d_inner)
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            # h: (batch, d_inner, d_state), C[:, t]: (batch, d_state)
            # y: (batch, d_inner)
            y = torch.bmm(h, C[:, t].unsqueeze(-1)).squeeze(-1)
            ys.append(y)
        
        y = torch.stack(ys, dim=1)  # (batch, seq_len, d_inner)
        y = y + x * D
        
        return y
    
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        
        # Input projection
        x_inner = self.input_proj(x)  # (batch, seq, d_inner)
        
        # Local convolution
        x_conv = x_inner.transpose(1, 2)  # (batch, d_inner, seq)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Truncate padding
        x_conv = x_conv.transpose(1, 2)  # (batch, seq, d_inner)
        x_conv = self.act(x_conv)
        
        # Selective SSM parameters from x_conv
        x_flat = x_conv.reshape(-1, self.d_inner)
        x_params = self.x_proj(x_flat)  # (batch*seq, dt_rank + 2*d_state)
        
        dt, B_seq, C_seq = torch.split(
            x_params, 
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )
        
        dt = dt.reshape(batch, seq_len, self.dt_rank)
        B_seq = B_seq.reshape(batch, seq_len, self.d_state)
        C_seq = C_seq.reshape(batch, seq_len, self.d_state)
        
        # Project dt to d_inner for discretization
        dt = F.linear(dt, torch.eye(self.dt_rank, self.d_inner, device=dt.device)[:, :self.dt_rank])  # Simplified: just pad/truncate
        # Actually project properly: dt_rank -> d_inner
        if self.dt_rank < self.d_inner:
            # Pad dt to d_inner
            dt_padded = torch.zeros(batch, seq_len, self.d_inner, device=dt.device, dtype=dt.dtype)
            dt_padded[:, :, :self.dt_rank] = dt
            dt = dt_padded
        elif self.dt_rank > self.d_inner:
            dt = dt[:, :, :self.d_inner]
        
        # Compute skip connection before SSM
        y = self.selective_scan(x_conv, dt, self.A, B_seq, C_seq, self.D)
        
        # Output projection
        y = self.output_proj(y)
        
        return y


class MambaBlock(nn.Module):
    """Single Mamba block with residual connections"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dtype=torch.float32):
        super().__init__()
        self.mixer = S6Block(d_model, d_state, d_conv, expand, dtype)
        self.norm = RMSNorm(d_model)
    
    def forward(self, x):
        return self.mixer(self.norm(x)) + x


class MambaEncoder(nn.Module):
    """
    Mamba Encoder for Student Behavior Sequences
    
    Processes raw event sequences through multiple Mamba blocks to produce
    contextualized sequence representations.
    """
    def __init__(
        self, 
        d_model=64, 
        n_layers=6, 
        d_state=16, 
        d_conv=4, 
        expand=2,
        dtype=torch.float32
    ):
        super().__init__()
        self.d_model = d_model
        
        # Input embedding (will be defined based on feature dim in forward)
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dtype)
            for _ in range(n_layers)
        ])
        
        self.final_norm = RMSNorm(d_model)
    
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class MultiScaleFeatureExtractor(nn.Module):
    """
    Step 3: Multi-scale feature extraction
    
    Extracts features at three temporal scales:
    - Fine-grained: every 100 events (local rhythm, fluency)
    - Medium-grained: per exercise (problem-solving strategy, error patterns)  
    - Coarse-grained: per course part (learning trajectory evolution)
    
    Uses cross-scale attention for fusion.
    """
    def __init__(self, d_model, fine_window=100, n_exercises=30, n_parts=7):
        super().__init__()
        self.d_model = d_model
        self.fine_window = fine_window
        
        # Fine-grained pooling
        self.fine_pool = nn.AdaptiveAvgPool1d(fine_window)
        
        # Medium-grained: per exercise
        self.exercise_proj = nn.Linear(d_model, d_model)
        
        # Coarse-grained: per part
        self.part_proj = nn.Linear(d_model, d_model)
        
        # Cross-scale attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=4, 
            batch_first=True,
            dropout=0.1
        )
        
        # Output fusion
        self.fusion = nn.Linear(d_model * 3, d_model)
        self.norm = RMSNorm(d_model)
    
    def forward(self, x, exercise_ids, part_ids):
        """
        x: (batch, seq_len, d_model) - Mamba output
        exercise_ids: (batch, seq_len) - exercise index for each event
        part_ids: (batch, seq_len) - part index for each event
        Returns: (batch, d_model) - fused multi-scale representation
        """
        batch, seq_len, d_model = x.shape
        
        # === Fine-grained: window-based ===
        # Reshape to (batch * windows, window_size, d_model) for pooling
        n_windows = seq_len // self.fine_window
        if n_windows > 0:
            x_trunc = x[:, :n_windows * self.fine_window]
            x_windows = x_trunc.view(batch, n_windows, self.fine_window, d_model)
            # Pool each window: (batch, n_windows, d_model)
            fine_features = x_windows.mean(dim=2)
        else:
            fine_features = x.mean(dim=1, keepdim=True)
        
        # === Medium-grained: per exercise ===
        max_ex = max(exercise_ids.max().item(), 1)
        medium_features_list = []
        
        for ex in range(min(max_ex, 30)):  # Cap at 30 exercises
            mask = (exercise_ids == ex)
            if mask.any():
                ex_feat = (x * mask.unsqueeze(-1)).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
            else:
                ex_feat = torch.zeros(batch, d_model, device=x.device)
            medium_features_list.append(ex_feat)
        
        # Pad or truncate to fixed size
        n_exercises = 30
        medium_features = torch.zeros(batch, n_exercises, d_model, device=x.device)
        for i, feat in enumerate(medium_features_list[:n_exercises]):
            medium_features[:, i] = feat
        
        # === Coarse-grained: per part ===
        coarse_features_list = []
        for p in range(1, 8):  # Parts 1-7
            mask = (part_ids == p)
            if mask.any():
                p_feat = (x * mask.unsqueeze(-1)).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
            else:
                p_feat = torch.zeros(batch, d_model, device=x.device)
            coarse_features_list.append(p_feat)
        
        coarse_features = torch.stack(coarse_features_list, dim=1)  # (batch, 7, d_model)
        
        # === Cross-scale attention ===
        # Attend from fine to medium
        fine_enhanced, _ = self.cross_attn(fine_features, medium_features, medium_features)
        # Attend from coarse to medium
        coarse_enhanced, _ = self.cross_attn(coarse_features.mean(dim=1, keepdim=True), 
                                              medium_features, medium_features)
        
        # Concatenate and fuse
        fused = torch.cat([
            fine_features.mean(dim=1, keepdim=True),  # (batch, 1, d_model)
            fine_enhanced.mean(dim=1, keepdim=True),
            coarse_enhanced
        ], dim=1)  # (batch, 3, d_model)
        
        fused = self.fusion(fused.view(batch, -1))  # (batch, d_model * 3) -> (batch, d_model)
        
        return self.norm(fused)


class StudentPrototypeLayer(nn.Module):
    """
    Step 4: Student Prototype Discovery
    
    Uses K-Means clustering in latent space to discover student archetypes.
    Each student is assigned a prototype embedding based on their Mamba representation.
    
    This is implemented as a differentiable approximation using a learned prototype matrix.
    """
    def __init__(self, d_model, n_prototypes=4):
        super().__init__()
        self.n_prototypes = n_prototypes
        # Learnable prototype centers
        self.prototype_centers = nn.Parameter(torch.randn(n_prototypes, d_model) * 0.1)
    
    def forward(self, student_repr):
        """
        student_repr: (batch, d_model)
        Returns: (batch, d_model) - prototype-enhanced representation
        """
        # Compute distances to prototypes
        dists = torch.cdist(student_repr, self.prototype_centers)  # (batch, n_prototypes)
        # Soft assignment
        weights = F.softmax(-dists, dim=-1)  # (batch, n_prototypes)
        # Weighted prototype
        prototype_embed = torch.matmul(weights, self.prototype_centers)  # (batch, d_model)
        return prototype_embed


class MAMBAStudentModel(nn.Module):
    """
    Complete Mamba-based Student Performance Prediction Model
    
    Implements all 6 steps:
    1. Data preprocessing & event encoding (external, in mamba_features.py)
    2. Mamba backbone pretraining (this module)
    3. Multi-scale feature extraction
    4. Student prototype discovery
    5. Adaptive prediction heads
    6. Interpretability (attention-based)
    """
    def __init__(
        self,
        # Input encoding
        n_event_types=7,
        d_event_embed=16,
        d_time_interval=8,
        d_exercise_embed=16,
        d_deadline=8,
        d_model=64,
        # Mamba backbone
        mamba_layers=6,
        d_state=16,
        d_conv=4,
        expand=2,
        # Multi-scale
        n_exercises=30,
        n_parts=7,
        # Prediction heads
        n_prototypes=4,
        dropout=0.2
    ):
        super().__init__()
        
        # Step 1: Event encoding
        self.event_embed = nn.Embedding(n_event_types, d_event_embed)
        self.time_interval_embed = nn.Linear(1, d_time_interval)
        self.exercise_embed = nn.Embedding(n_exercises + 1, d_exercise_embed)
        self.deadline_embed = nn.Linear(1, d_deadline)
        
        # Combined input projection
        d_input = d_event_embed + d_time_interval + d_exercise_embed + d_deadline
        self.input_proj = nn.Linear(d_input, d_model)
        
        # Step 2: Mamba backbone
        self.mamba = MambaEncoder(
            d_model=d_model,
            n_layers=mamba_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # Step 3: Multi-scale feature extractor
        self.multi_scale = MultiScaleFeatureExtractor(
            d_model=d_model,
            n_exercises=n_exercises,
            n_parts=n_parts
        )
        
        # Step 4: Prototype layer
        self.prototype = StudentPrototypeLayer(d_model, n_prototypes)
        
        # Step 5: Adaptive prediction heads (classification only)
        self.dropout = nn.Dropout(dropout)
        
        # Risk prediction (classification: dropout/fail) - primary task
        self.risk_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)  # 2 classes: low risk, high risk
        )
    
    def forward(self, batch, return_representations=False):
        """
        batch contains:
            event_types: (batch, seq_len) - event type indices
            time_intervals: (batch, seq_len) - time since last event
            exercise_ids: (batch, seq_len) - exercise indices
            part_ids: (batch, seq_len) - part indices
            deadline_dists: (batch, seq_len) - distance to deadline
        """
        event_types = batch['event_types']
        time_intervals = batch['time_intervals']
        exercise_ids = batch['exercise_ids']
        part_ids = batch['part_ids']
        deadline_dists = batch['deadline_dists']
        
        batch_size, seq_len = event_types.shape
        
        # Encode events
        event_emb = self.event_embed(event_types)  # (batch, seq, d_event)
        time_emb = self.time_interval_embed(time_intervals.unsqueeze(-1))  # (batch, seq, d_time)
        ex_emb = self.exercise_embed(exercise_ids.clamp(0, 30))  # (batch, seq, d_ex)
        dl_emb = self.deadline_embed(deadline_dists.unsqueeze(-1))  # (batch, seq, d_dl)
        
        # Concatenate and project
        x = torch.cat([event_emb, time_emb, ex_emb, dl_emb], dim=-1)
        x = self.input_proj(x)  # (batch, seq, d_model)
        
        # Mamba encoding
        mamba_out = self.mamba(x)  # (batch, seq, d_model)
        
        # Multi-scale features
        multi_scale_repr = self.multi_scale(mamba_out, exercise_ids, part_ids)
        
        # Prototype enhancement
        proto_repr = self.prototype(multi_scale_repr)
        
        # Combine for prediction
        combined = torch.cat([multi_scale_repr, proto_repr], dim=-1)
        combined = self.dropout(combined)
        
        # Prediction: risk classification only
        risk_pred = self.risk_head(combined)  # (batch, 2)
        
        if return_representations:
            return {
                'risk': risk_pred,
                'repr': multi_scale_repr,
                'proto': proto_repr,
                'mamba_out': mamba_out
            }
        
        return {
            'risk': risk_pred
        }
    
    def get_interpretability(self, batch, mamba_out=None):
        """
        Step 6: Explainable predictions
        Returns per-prediction explanation:
        - Which time windows contributed most to the prediction
        - Which event types are most important
        - Which learning prototype this student belongs to
        """
        with torch.no_grad():
            event_types = batch['event_types']
            exercise_ids = batch['exercise_ids']
            part_ids = batch['part_ids']
            
            # 1. Event type importance (embedding norms as proxy)
            event_importance = self.event_embed.weight.norm(dim=-1)
            event_importance = F.softmax(event_importance, dim=0)
            
            # 2. Temporal importance: compute per-window contribution
            # Use gradient-based saliency on mamba output
            if mamba_out is None:
                # Fallback: uniform importance
                temporal_importance = torch.ones_like(event_types, dtype=torch.float32) / event_types.shape[1]
            else:
                # Use mean pooling of sequence as proxy for temporal attention
                seq_len = mamba_out.shape[1]
                # Compute per-segment importance (every 100 events)
                window_size = 100
                n_windows = seq_len // window_size
                if n_windows > 0:
                    mamba_reshaped = mamba_out[:, :n_windows*window_size].reshape(-1, n_windows, window_size, mamba_out.shape[-1])
                    window_scores = mamba_reshaped.mean(dim=2).norm(dim=-1)  # (batch, n_windows)
                    temporal_importance = F.softmax(window_scores, dim=-1)
                else:
                    temporal_importance = torch.ones(mamba_out.shape[0], 1, device=mamba_out.device) / mamba_out.shape[0]
            
            # 3. Prototype assignment (distance to each prototype center)
            if hasattr(self, 'prototype'):
                student_repr = self.multi_scale(
                    mamba_out, 
                    exercise_ids, 
                    part_ids
                ) if mamba_out is not None else None
                
                if student_repr is not None:
                    dists = torch.cdist(student_repr, self.prototype.prototype_centers)
                    proto_weights = F.softmax(-dists, dim=-1)  # Soft assignment
                    proto_id = proto_weights.argmax(dim=-1)
                else:
                    proto_weights = None
                    proto_id = None
            else:
                proto_weights = None
                proto_id = None
            
            return {
                # Which event types matter most
                'event_type_importance': event_importance.cpu().numpy(),
                'event_type_names': ['focus_gained', 'focus_lost', 'text_insert', 
                                     'text_remove', 'text_paste', 'run', 'submit'],
                # Which time windows contributed most
                'temporal_importance': temporal_importance.cpu().numpy() if temporal_importance is not None else None,
                'temporal_window_size': 100,
                # Which prototype this student belongs to
                'prototype_weights': proto_weights.cpu().numpy() if proto_weights is not None else None,
                'prototype_id': proto_id.cpu().numpy() if proto_id is not None else None,
                'n_prototypes': self.prototype.n_prototypes if hasattr(self, 'prototype') else 4,
            }


def create_mamba_student_model(config=None):
    """Factory function to create the model with default or custom config"""
    default_config = {
        'n_event_types': 7,
        'd_event_embed': 16,
        'd_time_interval': 8,
        'd_exercise_embed': 16,
        'd_deadline': 8,
        'd_model': 64,
        'mamba_layers': 6,
        'd_state': 16,
        'd_conv': 4,
        'expand': 2,
        'n_exercises': 30,
        'n_parts': 7,
        'n_prototypes': 4,
        'dropout': 0.2
    }
    
    if config:
        default_config.update(config)
    
    model = MAMBAStudentModel(**default_config)
    return model


if __name__ == "__main__":
    # Test the model
    model = create_mamba_student_model()
    
    # Dummy batch
    batch_size, seq_len = 4, 1000
    dummy_batch = {
        'event_types': torch.randint(0, 7, (batch_size, seq_len)),
        'time_intervals': torch.rand(batch_size, seq_len) * 10,
        'exercise_ids': torch.randint(0, 30, (batch_size, seq_len)),
        'part_ids': torch.randint(1, 8, (batch_size, seq_len)),
        'deadline_dists': torch.rand(batch_size, seq_len) * 1000,
    }
    
    outputs = model(dummy_batch)
    print(f"Grade prediction shape: {outputs['grade'].shape}")
    print(f"Risk prediction shape: {outputs['risk'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
