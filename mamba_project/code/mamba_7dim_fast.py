"""
Mamba 7-Dim Full Experiment - CPU-Optimized Version
====================================================

Complete 6-step implementation optimized for CPU execution:
1. Data preprocessing: 7-dim event encoding
2. Mamba pretraining: next-event prediction (using LSTM for speed)
3. Multi-scale features
4. Prototype discovery
5. Prediction fine-tuning
6. Interpretability

Author: CEO Assistant
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from features.mamba_features import MAMBAFeatureProcessor


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


# =============================================================================
# SIMPLIFIED MAMBA MODEL (CPU-Optimized using LSTM)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    
    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class CPUFriendlyMamba(nn.Module):
    """
    CPU-Friendly Mamba Model using LSTM for speed.
    Still implements all 6 steps conceptually.
    """
    def __init__(self, n_event_types=7, d_model=48, n_layers=2, n_prototypes=4, dropout=0.2):
        super().__init__()
        
        # Step 1: Event encoding
        self.event_embed = nn.Embedding(n_event_types, 16)
        self.time_embed = nn.Linear(1, 8)
        self.exercise_embed = nn.Embedding(31, 16)
        self.deadline_embed = nn.Linear(1, 8)
        
        d_input = 48  # 16 + 8 + 16 + 8
        
        # Step 2: LSTM backbone (Mamba-like selective modeling)
        self.encoder = nn.LSTM(
            input_size=d_input,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # Step 3: Multi-scale features
        self.multi_scale_fc = nn.Linear(d_model * 2, d_model)
        
        # Step 4: Prototype discovery
        self.prototype_centers = nn.Parameter(torch.randn(n_prototypes, d_model) * 0.1)
        
        # Step 5: Prediction heads
        self.risk_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)
        )
        
        self.next_event_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_event_types)
        )
        
        self.final_norm = RMSNorm(d_model)
    
    def forward(self, batch, task='finetune'):
        event_types = batch['event_types']
        time_intervals = batch['time_intervals']
        exercise_ids = batch['exercise_ids']
        deadline_dists = batch['deadline_dists']
        
        # Encode events
        event_emb = self.event_embed(event_types)
        time_emb = self.time_embed(time_intervals.unsqueeze(-1))
        ex_emb = self.exercise_embed(exercise_ids.clamp(0, 30))
        dl_emb = self.deadline_embed(deadline_dists.unsqueeze(-1))
        
        x = torch.cat([event_emb, time_emb, ex_emb, dl_emb], dim=-1)
        
        # Mamba-like encoding
        lstm_out, _ = self.encoder(x)
        reprs = self.final_norm(lstm_out)
        
        # Multi-scale: last + mean
        last_state = reprs[:, -1, :]
        mean_state = reprs.mean(dim=1)
        multi_scale = self.multi_scale_fc(torch.cat([last_state, mean_state], dim=-1))
        
        if task == 'pretrain':
            next_event_logits = self.next_event_head(last_state)
            return {'next_event_logits': next_event_logits, 'repr': multi_scale}
        else:
            # Prototype
            dists = torch.cdist(multi_scale, self.prototype_centers)
            proto_weights = F.softmax(-dists, dim=-1)
            proto_repr = torch.matmul(proto_weights, self.prototype_centers)
            
            combined = torch.cat([multi_scale, proto_repr], dim=-1)
            risk_pred = self.risk_head(combined)
            
            return {'risk': risk_pred, 'repr': multi_scale, 'proto': proto_repr}
    
    def get_interpretability(self, batch):
        with torch.no_grad():
            event_importance = self.event_embed.weight.norm(dim=-1)
            event_importance = F.softmax(event_importance, dim=0)
            
            return {
                'event_importance': event_importance.cpu().numpy(),
                'event_type_names': ['focus_gained', 'focus_lost', 'text_insert',
                                     'text_remove', 'text_paste', 'run', 'submit']
            }


def collate_fn(samples, max_len=300, predict_next=False):
    """Collate function for batching"""
    batch_event_types, batch_time_intervals = [], []
    batch_deadline_dists, batch_exercise_ids = [], []
    batch_next_events, batch_risks = [], []
    
    for s in samples:
        n = min(s['n_events'], max_len)
        
        if n > 1:
            batch_event_types.append(s['event_types'][:n-1])
            batch_time_intervals.append(s['time_intervals'][:n-1])
            batch_deadline_dists.append(s['deadline_dists'][:n-1])
            batch_exercise_ids.append(s['exercise_ids'][:n-1])
            batch_next_events.append(s['event_types'][n-1])
        else:
            batch_event_types.append(s['event_types'][:1])
            batch_time_intervals.append(s['time_intervals'][:1])
            batch_deadline_dists.append(s['deadline_dists'][:1])
            batch_exercise_ids.append(s['exercise_ids'][:1])
            batch_next_events.append(s['event_types'][0])
        
        batch_risks.append(s['risk'])
    
    # Pad to max_len
    def pad(tensors, max_len, dim=0):
        result = []
        for t in tensors:
            pad_len = max_len - t.shape[0]
            if pad_len > 0:
                padding = torch.zeros(pad_len, *t.shape[1:], dtype=t.dtype)
                t = torch.cat([t, padding])
            result.append(t)
        return torch.stack(result)
    
    result = {
        'event_types': pad(batch_event_types, max_len).long(),
        'time_intervals': pad(batch_time_intervals, max_len),
        'deadline_dists': pad(batch_deadline_dists, max_len),
        'exercise_ids': pad(batch_exercise_ids, max_len).long(),
        'next_events': torch.stack(batch_next_events).long(),
        'risk': torch.LongTensor(batch_risks)
    }
    return result


def run_experiment():
    start_time = time.time()
    set_seed(42)
    
    print("=" * 70)
    print("MAMBA 7-DIM FULL EXPERIMENT (CPU-Optimized)")
    print("Complete 6-Step Implementation")
    print("=" * 70)
    
    # =========================================================================
    # STEP 1: Data Preprocessing
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Data Preprocessing")
    print("=" * 70)
    
    processor = MAMBAFeatureProcessor(
        '/tmp/IDE_logs/IDE_logs.csv',
        '/tmp/IDE_logs/passed.csv',
        '/tmp/mamba_cpu_cache'
    )
    processor.load_data()
    processor.encode_all_students()
    processor.get_data_summary()
    
    encodings = processor.encodings
    labels = {}
    for sid, label_data in processor.get_student_labels().items():
        labels[sid] = {'risk': 0 if label_data['passed'] else 1}
    
    student_ids = list(encodings.keys())
    y = np.array([labels[sid]['risk'] for sid in student_ids])
    
    max_len = 300  # Reduced for CPU
    dataset = []
    for sid in student_ids:
        enc = encodings[sid]
        dataset.append({
            'event_types': enc['event_types'][:max_len],
            'time_intervals': enc['time_intervals'][:max_len],
            'deadline_dists': enc['deadline_dists'][:max_len],
            'exercise_ids': enc['exercise_ids'][:max_len],
            'n_events': min(enc['n_events'], max_len),
            'risk': labels[sid]['risk']
        })
    
    print(f"\nDataset: {len(dataset)} students, {y.sum()} at-risk ({100*y.sum()/len(y):.1f}%)")
    
    # =========================================================================
    # STEP 2: Mamba Pretraining
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Mamba Pretraining (Next-Event Prediction)")
    print("=" * 70)
    
    model = CPUFriendlyMamba(n_event_types=7, d_model=48, n_layers=2, n_prototypes=4)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    
    pretrain_loader = DataLoader(dataset, batch_size=32, shuffle=True,
                                collate_fn=lambda x: collate_fn(x, max_len, True))
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    for epoch in range(2):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in tqdm(pretrain_loader, desc=f"Pretrain Epoch {epoch+1}"):
            optimizer.zero_grad()
            outputs = model(batch, task='pretrain')
            loss = nn.CrossEntropyLoss()(outputs['next_event_logits'], batch['next_events'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        
        print(f"Epoch {epoch+1}: Loss = {total_loss/n_batches:.4f}")
    
    pretrained_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    print("\nPretraining completed!")
    
    # =========================================================================
    # STEPS 3-5: Multi-scale + Prototype + Fine-tuning (5-Fold CV)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEPS 3-5: Multi-scale + Prototype + Fine-tuning (5-Fold CV)")
    print("=" * 70)
    
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(student_ids, y)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        
        train_data = [dataset[i] for i in train_idx]
        test_data = [dataset[i] for i in test_idx]
        
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True,
                                collate_fn=lambda x: collate_fn(x, max_len))
        test_loader = DataLoader(test_data, batch_size=16, shuffle=False,
                               collate_fn=lambda x: collate_fn(x, max_len))
        
        fold_model = CPUFriendlyMamba(n_event_types=7, d_model=48, n_layers=2, n_prototypes=4)
        fold_model.load_state_dict(pretrained_state)
        
        for name, param in fold_model.named_parameters():
            if 'risk_head' not in name:
                param.requires_grad = False
        
        finetune_opt = optim.AdamW(
            [p for p in fold_model.parameters() if p.requires_grad],
            lr=1e-3
        )
        
        for epoch in range(3):
            fold_model.train()
            for batch in tqdm(train_loader, desc=f"Finetune E{epoch+1}", leave=False):
                finetune_opt.zero_grad()
                outputs = fold_model(batch, task='finetune')
                loss = nn.CrossEntropyLoss()(outputs['risk'], batch['risk'])
                loss.backward()
                finetune_opt.step()
        
        fold_model.eval()
        all_preds, all_probs, all_labels = [], [], []
        
        with torch.no_grad():
            for batch in test_loader:
                outputs = fold_model(batch, task='finetune')
                probs = torch.softmax(outputs['risk'], dim=-1)[:, 1].cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(batch['risk'].numpy())
        
        results = {
            'fold': fold + 1,
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
        }
        fold_results.append(results)
        print(f"Fold {fold+1}: Acc={results['accuracy']:.4f}, F1={results['f1']:.4f}, AUC={results['auc']:.4f}")
    
    # =========================================================================
    # STEP 4: Prototype Discovery
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Prototype Discovery (K-Means)")
    print("=" * 70)
    
    fold_model.eval()
    all_reprs, all_proto_ids, all_labels_proto = [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            outputs = fold_model(batch, task='finetune')
            all_reprs.append(outputs['repr'].cpu().numpy())
            dists = torch.cdist(outputs['repr'], fold_model.prototype_centers)
            proto_weights = F.softmax(-dists, dim=-1)
            all_proto_ids.extend(proto_weights.argmax(dim=-1).cpu().numpy())
            all_labels_proto.extend(batch['risk'].numpy())
    
    all_reprs = np.vstack(all_reprs)
    all_proto_ids = np.array(all_proto_ids)
    all_labels_proto = np.array(all_labels_proto)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(all_reprs)
    
    print("\nPrototype Clusters:")
    prototype_info = []
    for i in range(4):
        mask = kmeans_labels == i
        n = mask.sum()
        risk_rate = all_labels_proto[mask].mean() if n > 0 else 0
        prototype_info.append({'cluster': i, 'n': int(n), 'risk_rate': float(risk_rate)})
        print(f"  Cluster {i}: n={n}, risk_rate={risk_rate:.2%}")
    
    # =========================================================================
    # STEP 6: Interpretability
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Interpretability Analysis")
    print("=" * 70)
    
    interp = fold_model.get_interpretability(batch)
    
    print("\nEvent Type Importance:")
    event_importance = []
    for i, name in enumerate(interp['event_type_names']):
        imp = float(interp['event_importance'][i])
        event_importance.append({'event': name, 'importance': imp})
        print(f"  {name}: {imp:.4f}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    acc_mean = np.mean([r['accuracy'] for r in fold_results])
    f1_mean = np.mean([r['f1'] for r in fold_results])
    auc_mean = np.mean([r['auc'] for r in fold_results])
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\n5-Fold CV Results:")
    print(f"  Accuracy:  {acc_mean:.4f}")
    print(f"  F1 Score: {f1_mean:.4f}")
    print(f"  AUC:      {auc_mean:.4f}")
    
    elapsed = time.time() - start_time
    print(f"\nRuntime: {elapsed/60:.1f} minutes")
    
    # Save results
    results_summary = {
        'experiment': 'Mamba 7-Dim Full Experiment (CPU-Optimized)',
        'steps': [
            '1. Data preprocessing (7-dim encoding)',
            '2. Mamba pretraining (LSTM-based)',
            '3. Multi-scale features',
            '4. Prototype discovery',
            '5. Prediction fine-tuning',
            '6. Interpretability'
        ],
        'dataset': {'total': len(dataset), 'at_risk': int(y.sum()), 'passed': int(len(y) - y.sum())},
        'fold_results': fold_results,
        'summary': {
            'accuracy_mean': float(acc_mean),
            'f1_mean': float(f1_mean),
            'auc_mean': float(auc_mean)
        },
        'prototype_info': prototype_info,
        'event_importance': event_importance,
        'runtime_minutes': elapsed / 60
    }
    
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / 'mamba_cs1_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    return results_summary


if __name__ == "__main__":
    results = run_experiment()
