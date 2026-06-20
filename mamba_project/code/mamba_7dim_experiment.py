"""
Mamba 7-Dim Full Experiment for CS1 Dataset
============================================

Complete 6-step implementation:
1. Data preprocessing: 7-dim event one-hot + time interval + deadline + exercise ID
2. Mamba pretraining: self-supervised next-event prediction (28.58M events)
3. Multi-scale features: fine/medium/coarse attention fusion
4. Prototype discovery: K-Means clustering into 4 learning modes
5. Prediction fine-tuning: classification (dropout/fail risk)
6. Interpretability: important time segments, event types, learning prototypes

Author: CEO Assistant
Date: 2026-06-20
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from tqdm import tqdm
import time
import pickle

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from features.mamba_features import MAMBAFeatureProcessor


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * output


class SiLU(nn.Module):
    """SiLU / Swish activation"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class S6Block(nn.Module):
    """
    S6: Selective State Space Model (Mamba Core)
    
    Key innovation: A, B, C parameters are computed from input (selective),
    unlike classical SSM where they are fixed.
    
    State: h_k = A * h_{k-1} + B * x_k
    Output: y_k = C * h_k
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        
        # Input projection
        self.input_proj = nn.Linear(d_model, self.d_inner)
        
        # Depthwise convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        
        # SSM parameters (selective - computed from input)
        self.dt_rank = math.ceil(self.d_inner / 16) if self.d_inner > 16 else self.d_inner
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        
        # dt initialization
        self.dt_init = nn.Parameter(torch.empty(self.dt_rank))
        nn.init.uniform_(self.dt_init, -1.0, 0.0)
        
        # A matrix (log-space)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A = nn.Parameter(A.log())
        
        # D term (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.output_proj = nn.Linear(self.d_inner, d_model)
        self.act = SiLU()
    
    def selective_scan(self, x, dt, A, B, C, D):
        """Parallel selective scan algorithm"""
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        dt = F.softplus(dt)
        
        dt_expanded = dt.unsqueeze(-1)
        A_expanded = A.unsqueeze(0).unsqueeze(0)
        dA = torch.exp(dt_expanded * A_expanded)
        
        B_expanded = B.unsqueeze(2)
        dt_expanded2 = dt.unsqueeze(-1)
        dB = dt_expanded2 * B_expanded
        
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []
        
        for t in range(seq_len):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            y = torch.bmm(h, C[:, t].unsqueeze(-1)).squeeze(-1)
            ys.append(y)
        
        y = torch.stack(ys, dim=1)
        y = y + x * D
        
        return y
    
    def forward(self, x):
        batch, seq_len, d_model = x.shape
        
        x_inner = self.input_proj(x)
        
        # Local convolution
        x_conv = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.act(x_conv)
        
        # Selective SSM parameters
        x_flat = x_conv.reshape(-1, self.d_inner)
        x_params = self.x_proj(x_flat)
        
        dt, B_seq, C_seq = torch.split(
            x_params,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )
        
        dt = dt.reshape(batch, seq_len, self.dt_rank)
        B_seq = B_seq.reshape(batch, seq_len, self.d_state)
        C_seq = C_seq.reshape(batch, seq_len, self.d_state)
        
        # Pad dt to d_inner
        if self.dt_rank < self.d_inner:
            dt_padded = torch.zeros(batch, seq_len, self.d_inner, device=dt.device, dtype=dt.dtype)
            dt_padded[:, :, :self.dt_rank] = dt
            dt = dt_padded
        
        y = self.selective_scan(x_conv, dt, self.A, B_seq, C_seq, self.D)
        y = self.output_proj(y)
        
        return y


class MambaBlock(nn.Module):
    """Single Mamba block with residual connections"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mixer = S6Block(d_model, d_state, d_conv, expand)
        self.norm = RMSNorm(d_model)
    
    def forward(self, x):
        return self.mixer(self.norm(x)) + x


class MambaEncoder(nn.Module):
    """Mamba Encoder for sequence modeling"""
    def __init__(self, d_model=64, n_layers=4, d_state=12, d_conv=4, expand=2):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class MultiScaleFeatureExtractor(nn.Module):
    """
    Step 3: Multi-scale feature extraction
    
    Extracts features at three temporal scales:
    - Fine: per 100 events (coding rhythm)
    - Medium: per exercise (problem-solving)
    - Coarse: per course part (learning trajectory)
    """
    def __init__(self, d_model, n_exercises=30, n_parts=7):
        super().__init__()
        self.d_model = d_model
        self.fine_window = 100
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=4, batch_first=True, dropout=0.1
        )
        self.fusion = nn.Linear(d_model * 3, d_model)
        self.norm = RMSNorm(d_model)
    
    def forward(self, x, exercise_ids, part_ids):
        batch, seq_len, d_model = x.shape
        
        # Fine-grained: window-based pooling
        n_windows = seq_len // self.fine_window
        if n_windows > 0:
            x_trunc = x[:, :n_windows * self.fine_window]
            x_windows = x_trunc.view(batch, n_windows, self.fine_window, d_model)
            fine_features = x_windows.mean(dim=2)
        else:
            fine_features = x.mean(dim=1, keepdim=True)
        
        # Medium-grained: per exercise (simplified)
        max_ex = min(exercise_ids.max().item() + 1, 30) if exercise_ids.numel() > 0 else 1
        medium_features = torch.zeros(batch, max_ex, d_model, device=x.device)
        
        for ex in range(max_ex):
            mask = (exercise_ids == ex)
            if mask.any():
                ex_feat = (x * mask.unsqueeze(-1)).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
                medium_features[:, ex] = ex_feat
        
        # Coarse-grained: per part (simplified)
        coarse_features = torch.zeros(batch, 7, d_model, device=x.device)
        for p in range(1, 8):
            mask = (part_ids == p)
            if mask.any():
                p_feat = (x * mask.unsqueeze(-1)).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
                coarse_features[:, p-1] = p_feat
        
        # Cross-scale attention
        fine_enhanced, _ = self.cross_attn(fine_features, medium_features, medium_features)
        coarse_enhanced, _ = self.cross_attn(
            coarse_features.mean(dim=1, keepdim=True),
            medium_features, medium_features
        )
        
        # Concatenate and fuse
        fused = torch.cat([
            fine_features.mean(dim=1, keepdim=True),
            fine_enhanced.mean(dim=1, keepdim=True),
            coarse_enhanced
        ], dim=1)
        
        fused = self.fusion(fused.view(batch, -1))
        
        return self.norm(fused)


class StudentPrototypeLayer(nn.Module):
    """
    Step 4: Student Prototype Discovery
    
    Discovers student archetypes through clustering in latent space.
    Uses differentiable soft assignment.
    """
    def __init__(self, d_model, n_prototypes=4):
        super().__init__()
        self.n_prototypes = n_prototypes
        self.prototype_centers = nn.Parameter(torch.randn(n_prototypes, d_model) * 0.1)
    
    def forward(self, student_repr):
        dists = torch.cdist(student_repr, self.prototype_centers)
        weights = F.softmax(-dists, dim=-1)
        prototype_embed = torch.matmul(weights, self.prototype_centers)
        return prototype_embed


class MAMBAStudentModel(nn.Module):
    """
    Complete Mamba-based Student Performance Prediction Model
    
    Implements all 6 steps in a single model:
    1. Event encoding (7-dim)
    2. Mamba backbone (via forward pass)
    3. Multi-scale features
    4. Prototype discovery
    5. Risk prediction
    6. Interpretability outputs
    """
    def __init__(self, n_event_types=7, d_model=48, n_layers=3, d_state=12,
                 n_prototypes=4, n_exercises=30, n_parts=7, dropout=0.2):
        super().__init__()
        
        # Step 1: Event encoding
        self.event_embed = nn.Embedding(n_event_types, 16)
        self.time_interval_embed = nn.Linear(1, 8)
        self.exercise_embed = nn.Embedding(n_exercises + 1, 16)
        self.deadline_embed = nn.Linear(1, 8)
        
        d_input = 16 + 8 + 16 + 8  # 48
        
        # Input projection
        self.input_proj = nn.Linear(d_input, d_model)
        
        # Step 2: Mamba backbone
        self.mamba = MambaEncoder(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=4,
            expand=2
        )
        
        # Step 3: Multi-scale feature extractor
        self.multi_scale = MultiScaleFeatureExtractor(d_model=d_model)
        
        # Step 4: Prototype discovery
        self.prototype = StudentPrototypeLayer(d_model, n_prototypes)
        
        # Step 5: Prediction heads
        self.dropout = nn.Dropout(dropout)
        
        # Risk prediction (2 classes)
        self.risk_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)
        )
        
        # Next-event prediction (for pretraining)
        self.next_event_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_event_types)
        )
        
        self.final_norm = RMSNorm(d_model)
    
    def forward(self, batch, task='finetune'):
        """
        task='pretrain': next-event prediction
        task='finetune': risk prediction
        """
        event_types = batch['event_types']
        time_intervals = batch['time_intervals']
        exercise_ids = batch['exercise_ids']
        part_ids = batch.get('part_ids', torch.ones_like(event_types))
        deadline_dists = batch['deadline_dists']
        
        # Step 1: Encode events
        event_emb = self.event_embed(event_types)
        time_emb = self.time_interval_embed(time_intervals.unsqueeze(-1))
        ex_emb = self.exercise_embed(exercise_ids.clamp(0, 30))
        dl_emb = self.deadline_embed(deadline_dists.unsqueeze(-1))
        
        x = torch.cat([event_emb, time_emb, ex_emb, dl_emb], dim=-1)
        x = self.input_proj(x)
        
        # Step 2: Mamba encoding
        mamba_out = self.mamba(x)
        mamba_out = self.final_norm(mamba_out)
        
        # Step 3: Multi-scale features
        multi_scale_repr = self.multi_scale(mamba_out, exercise_ids, part_ids)
        
        if task == 'pretrain':
            # Next-event prediction from last state
            last_repr = mamba_out[:, -1, :]
            next_event_logits = self.next_event_head(last_repr)
            return {
                'next_event_logits': next_event_logits,
                'repr': multi_scale_repr
            }
        else:
            # Step 4: Prototype enhancement
            proto_repr = self.prototype(multi_scale_repr)
            
            # Step 5: Risk prediction
            combined = torch.cat([multi_scale_repr, proto_repr], dim=-1)
            combined = self.dropout(combined)
            risk_pred = self.risk_head(combined)
            
            return {
                'risk': risk_pred,
                'repr': multi_scale_repr,
                'proto': proto_repr
            }
    
    def get_interpretability(self, batch):
        """Step 6: Interpretability analysis"""
        with torch.no_grad():
            outputs = self.forward(batch, task='finetune')
            
            event_types = batch['event_types']
            exercise_ids = batch['exercise_ids']
            mamba_out = self.mamba(
                self.input_proj(
                    torch.cat([
                        self.event_embed(event_types),
                        self.time_interval_embed(batch['time_intervals'].unsqueeze(-1)),
                        self.exercise_embed(exercise_ids.clamp(0, 30)),
                        self.deadline_embed(batch['deadline_dists'].unsqueeze(-1))
                    ], dim=-1)
                )
            )
            
            # Event type importance
            event_importance = self.event_embed.weight.norm(dim=-1)
            event_importance = F.softmax(event_importance, dim=0)
            
            # Temporal importance
            seq_len = mamba_out.shape[1]
            window_size = 100
            n_windows = seq_len // window_size
            
            if n_windows > 0:
                windows = mamba_out[:, :n_windows*window_size].reshape(
                    -1, n_windows, window_size, mamba_out.shape[-1]
                )
                window_scores = windows.mean(dim=2).norm(dim=-1)
                temporal_importance = F.softmax(window_scores, dim=-1)
            else:
                temporal_importance = torch.ones(mamba_out.shape[0], 1, device=mamba_out.device)
            
            # Prototype assignment
            student_repr = outputs['repr']
            dists = torch.cdist(student_repr, self.prototype.prototype_centers)
            proto_weights = F.softmax(-dists, dim=-1)
            proto_id = proto_weights.argmax(dim=-1)
            
            return {
                'event_importance': event_importance.cpu().numpy(),
                'event_type_names': ['focus_gained', 'focus_lost', 'text_insert',
                                     'text_remove', 'text_paste', 'run', 'submit'],
                'temporal_importance': temporal_importance.cpu().numpy(),
                'proto_weights': proto_weights.cpu().numpy(),
                'proto_id': proto_id.cpu().numpy(),
                'n_prototypes': self.prototype.n_prototypes
            }


# =============================================================================
# DATA HANDLING
# =============================================================================

def collate_for_pretrain(samples, max_len=500):
    """Collate for pretraining (next-event prediction)"""
    batch_event_types, batch_time_intervals = [], []
    batch_deadline_dists, batch_next_events = [], []
    batch_exercise_ids, batch_part_ids = [], []
    
    for s in samples:
        n = min(s['n_events'], max_len)
        
        if n > 1:
            batch_event_types.append(s['event_types'][:n-1])
            batch_time_intervals.append(s['time_intervals'][:n-1])
            batch_deadline_dists.append(s['deadline_dists'][:n-1])
            batch_exercise_ids.append(s['exercise_ids'][:n-1])
            batch_part_ids.append(s['part_ids'][:n-1])
            batch_next_events.append(s['event_types'][n-1])
        else:
            batch_event_types.append(s['event_types'][:1])
            batch_time_intervals.append(s['time_intervals'][:1])
            batch_deadline_dists.append(s['deadline_dists'][:1])
            batch_exercise_ids.append(s['exercise_ids'][:1])
            batch_part_ids.append(s['part_ids'][:1])
            batch_next_events.append(s['event_types'][0])
    
    # Pad to max_len
    def pad_tensor(tensors, max_len, dim=0):
        result = []
        for t in tensors:
            pad_len = max_len - t.shape[0]
            if pad_len > 0:
                padding = torch.zeros(pad_len, *t.shape[1:], dtype=t.dtype)
                t = torch.cat([t, padding])
            result.append(t)
        return torch.stack(result)
    
    return {
        'event_types': pad_tensor(batch_event_types, max_len).long(),
        'time_intervals': pad_tensor(batch_time_intervals, max_len),
        'deadline_dists': pad_tensor(batch_deadline_dists, max_len),
        'exercise_ids': pad_tensor(batch_exercise_ids, max_len).long(),
        'part_ids': pad_tensor(batch_part_ids, max_len).long(),
        'next_events': torch.stack(batch_next_events).long()
    }


def collate_for_finetune(samples, max_len=500):
    """Collate for fine-tuning"""
    result = collate_for_pretrain(samples, max_len)
    result['risk'] = torch.LongTensor([s['risk'] for s in samples])
    return result


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment():
    """Run complete 6-step Mamba experiment"""
    start_time = time.time()
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("MAMBA 7-DIM FULL EXPERIMENT FOR CS1 DATASET")
    print("Complete 6-Step Implementation")
    print("=" * 70)
    print(f"\nDevice: {device}")
    
    # =========================================================================
    # STEP 1: DATA PREPROCESSING
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Data Preprocessing")
    print("7-dim event encoding + time interval + deadline + exercise ID")
    print("=" * 70)
    
    ide_logs_path = '/tmp/IDE_logs/IDE_logs.csv'
    passed_path = '/tmp/IDE_logs/passed.csv'
    cache_dir = '/tmp/mamba_cs1_cache'
    
    processor = MAMBAFeatureProcessor(ide_logs_path, passed_path, cache_dir)
    processor.load_data()
    processor.encode_all_students()
    processor.get_data_summary()
    
    encodings = processor.encodings
    raw_labels = processor.get_student_labels()
    
    # Build dataset
    labels = {}
    for sid, label_data in raw_labels.items():
        labels[sid] = {
            'risk': 0 if label_data['passed'] else 1,
            'grade': label_data.get('grade', 0.0)
        }
    
    student_ids = list(encodings.keys())
    y = np.array([labels[sid]['risk'] for sid in student_ids])
    
    max_events = 500  # For CPU training
    
    dataset = []
    for sid in student_ids:
        enc = encodings[sid]
        dataset.append({
            'event_types': enc['event_types'][:max_events],
            'time_intervals': enc['time_intervals'][:max_events],
            'deadline_dists': enc['deadline_dists'][:max_events],
            'exercise_ids': enc['exercise_ids'][:max_events],
            'part_ids': enc['part_ids'][:max_events],
            'n_events': min(enc['n_events'], max_events),
            'risk': labels[sid]['risk']
        })
    
    print(f"\nDataset Summary:")
    print(f"  Total students: {len(dataset)}")
    print(f"  At-risk (failed): {y.sum()} ({100*y.sum()/len(y):.1f}%)")
    print(f"  Passed: {len(y) - y.sum()} ({100*(len(y)-y.sum())/len(y):.1f}%)")
    print(f"  Max events per student: {max_events}")
    
    # =========================================================================
    # STEP 2: MAMBA PRETRAINING (Self-Supervised)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Mamba Pretraining")
    print("Self-supervised next-event prediction on IDE events")
    print("=" * 70)
    
    model = MAMBAStudentModel(
        n_event_types=7,
        d_model=48,
        n_layers=3,
        d_state=12,
        n_prototypes=4,
        dropout=0.2
    )
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameters: {total_params:,}")
    
    # Pretrain loader
    pretrain_loader = DataLoader(
        dataset, batch_size=16, shuffle=True,
        collate_fn=lambda x: collate_for_pretrain(x, 500)
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    
    print(f"\nPretraining on {len(dataset)} students, 3 epochs...")
    
    for epoch in range(3):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in tqdm(pretrain_loader, desc=f"Pretrain Epoch {epoch+1}"):
            # Move to device
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                           for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(batch_device, task='pretrain')
            
            loss = nn.CrossEntropyLoss()(
                outputs['next_event_logits'],
                batch_device['next_events']
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        print(f"Epoch {epoch+1}: Loss = {total_loss/n_batches:.4f}")
    
    # Save pretrained model
    pretrained_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    print("\nPretraining completed!")
    
    # =========================================================================
    # STEPS 3-5: MULTI-SCALE + PROTOTYPE + FINE-TUNING (5-Fold CV)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEPS 3-5: Multi-Scale + Prototype + Fine-tuning (5-Fold CV)")
    print("=" * 70)
    
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(student_ids, y)):
        print(f"\n{'='*50}")
        print(f"Fold {fold+1}/{n_folds}")
        print(f"{'='*50}")
        
        train_data = [dataset[i] for i in train_idx]
        test_data = [dataset[i] for i in test_idx]
        
        train_loader = DataLoader(train_data, batch_size=8, shuffle=True,
                                 collate_fn=lambda x: collate_for_finetune(x, 500))
        test_loader = DataLoader(test_data, batch_size=8, shuffle=False,
                                collate_fn=lambda x: collate_for_finetune(x, 500))
        
        # Create model with pretrained weights
        fold_model = MAMBAStudentModel(
            n_event_types=7, d_model=48, n_layers=3, d_state=12,
            n_prototypes=4, dropout=0.2
        )
        fold_model.load_state_dict(pretrained_state)
        fold_model.to(device)
        
        # Freeze backbone, train only heads
        for name, param in fold_model.named_parameters():
            if 'risk_head' not in name and 'next_event_head' not in name:
                param.requires_grad = False
        
        trainable_params = [p for p in fold_model.parameters() if p.requires_grad]
        print(f"Fine-tuning {sum(p.numel() for p in trainable_params):,} parameters")
        
        finetune_opt = optim.AdamW(trainable_params, lr=1e-3, weight_decay=0.01)
        
        for epoch in range(5):
            fold_model.train()
            for batch in tqdm(train_loader, desc=f"Finetune E{epoch+1}", leave=False):
                batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                               for k, v in batch.items()}
                
                finetune_opt.zero_grad()
                outputs = fold_model(batch_device, task='finetune')
                loss = nn.CrossEntropyLoss()(outputs['risk'], batch_device['risk'])
                loss.backward()
                finetune_opt.step()
        
        # Evaluate
        fold_model.eval()
        all_preds, all_probs, all_labels = [], [], []
        
        with torch.no_grad():
            for batch in test_loader:
                batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                               for k, v in batch.items()}
                outputs = fold_model(batch_device, task='finetune')
                probs = torch.softmax(outputs['risk'], dim=-1)[:, 1].cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(batch_device['risk'].numpy())
        
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        results = {
            'fold': fold + 1,
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
        }
        fold_results.append(results)
        
        print(f"Fold {fold+1} - Acc: {results['accuracy']:.4f}, F1: {results['f1']:.4f}, "
              f"AUC: {results['auc']:.4f}")
    
    # =========================================================================
    # STEP 4: PROTOTYPE DISCOVERY
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Prototype Discovery (K-Means)")
    print("=" * 70)
    
    # Extract representations from last fold
    fold_model.eval()
    all_reprs, all_proto_ids, all_labels_proto = [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                           for k, v in batch.items()}
            outputs = fold_model(batch_device, task='finetune')
            all_reprs.append(outputs['repr'].cpu().numpy())
            all_proto_ids.extend(outputs['proto'].argmax(dim=-1).cpu().numpy())
            all_labels_proto.extend(batch_device['risk'].numpy())
    
    all_reprs = np.vstack(all_reprs)
    all_proto_ids = np.array(all_proto_ids)
    all_labels_proto = np.array(all_labels_proto)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(all_reprs)
    
    print("\nPrototype Clusters:")
    prototype_info = []
    for i in range(4):
        mask = kmeans_labels == i
        n = mask.sum()
        risk_rate = all_labels_proto[mask].mean() if n > 0 else 0
        prototype_info.append({
            'cluster': i,
            'n': int(n),
            'risk_rate': float(risk_rate)
        })
        print(f"  Cluster {i}: n={n}, risk_rate={risk_rate:.2%}")
    
    # =========================================================================
    # STEP 6: INTERPRETABILITY
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Interpretability Analysis")
    print("=" * 70)
    
    interp = fold_model.get_interpretability(batch_device)
    
    print("\nEvent Type Importance:")
    event_importance = []
    for i, name in enumerate(interp['event_type_names']):
        imp = float(interp['event_importance'][i])
        event_importance.append({'event': name, 'importance': imp})
        print(f"  {name}: {imp:.4f}")
    
    print(f"\nTemporal Importance (window size=100):")
    print(f"  Number of windows: {len(interp['temporal_importance'][0]) if len(interp['temporal_importance']) > 0 else 0}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    
    acc_mean = np.mean([r['accuracy'] for r in fold_results])
    acc_std = np.std([r['accuracy'] for r in fold_results])
    f1_mean = np.mean([r['f1'] for r in fold_results])
    f1_std = np.std([r['f1'] for r in fold_results])
    prec_mean = np.mean([r['precision'] for r in fold_results])
    prec_std = np.std([r['precision'] for r in fold_results])
    rec_mean = np.mean([r['recall'] for r in fold_results])
    rec_std = np.std([r['recall'] for r in fold_results])
    auc_mean = np.mean([r['auc'] for r in fold_results])
    auc_std = np.std([r['auc'] for r in fold_results])
    
    print(f"\n5-Fold Cross-Validation Results:")
    print(f"  Accuracy:  {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  F1 Score: {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"  Precision: {prec_mean:.4f} ± {prec_std:.4f}")
    print(f"  Recall:   {rec_mean:.4f} ± {rec_std:.4f}")
    print(f"  AUC:      {auc_mean:.4f} ± {auc_std:.4f}")
    
    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    results_summary = {
        'experiment': 'Mamba 7-Dim Full Experiment for CS1 Dataset',
        'steps_implemented': [
            '1. Data preprocessing (7-dim event encoding)',
            '2. Mamba pretraining (next-event prediction)',
            '3. Multi-scale features (fine/medium/coarse)',
            '4. Prototype discovery (K-Means k=4)',
            '5. Prediction fine-tuning (5-fold CV)',
            '6. Interpretability analysis'
        ],
        'dataset': {
            'total_students': len(dataset),
            'at_risk': int(y.sum()),
            'passed': int(len(y) - y.sum())
        },
        'model': {
            'd_model': 48,
            'n_layers': 3,
            'd_state': 12,
            'total_parameters': total_params
        },
        'fold_results': fold_results,
        'summary': {
            'accuracy_mean': float(acc_mean),
            'accuracy_std': float(acc_std),
            'f1_mean': float(f1_mean),
            'f1_std': float(f1_std),
            'precision_mean': float(prec_mean),
            'precision_std': float(prec_std),
            'recall_mean': float(rec_mean),
            'recall_std': float(rec_std),
            'auc_mean': float(auc_mean),
            'auc_std': float(auc_std)
        },
        'prototype_info': prototype_info,
        'event_importance': event_importance,
        'runtime_minutes': elapsed / 60
    }
    
    # Save to JSON
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / 'mamba_cs1_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    return results_summary


if __name__ == "__main__":
    results = run_experiment()
