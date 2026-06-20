"""
Mamba 7-dim Experiment - Minimal Version for CPU
Very aggressive simplification to get real results on CPU.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from tqdm import tqdm
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.mamba_features import MAMBAFeatureProcessor


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


class MinimalMamba(nn.Module):
    """
    Minimal Mamba for fast CPU training.
    - Reduced sequence length (500)
    - Smaller model (d_model=24, 1 layer)
    - Simplified multi-scale and prototype
    """
    def __init__(self, n_event_types=7, d_model=24, d_state=6, max_seq=500):
        super().__init__()
        
        self.max_seq = max_seq
        
        # Step 1: Event encoding
        self.event_embed = nn.Embedding(n_event_types, 12)
        self.time_embed = nn.Linear(1, 6)
        self.deadline_embed = nn.Linear(1, 6)
        
        # Step 2: Mamba-like mixing (simplified selective scan)
        # Use a simple LSTM-like recurrence instead of full Mamba
        self.lstm = nn.LSTM(input_size=24, hidden_size=d_model, batch_first=True, bidirectional=False, num_layers=1)
        
        # Step 3: Multi-scale (just mean pooling)
        
        # Step 4: Prototype (simple clustering layer)
        self.proto_embed = nn.Parameter(torch.randn(4, d_model) * 0.1)
        
        # Step 5: Prediction heads
        # Risk prediction head (2 classes)
        self.risk_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, 2)
        )
        
        # Next-event prediction head (7 classes - for pretraining)
        self.next_event_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_event_types)
        )
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, batch, task='finetune', return_repr=False):
        """
        task='pretrain': next-event prediction (returns next_event_logits)
        task='finetune': risk prediction (returns risk logits)
        """
        event_types = batch['event_types']
        time_intervals = batch['time_intervals']
        deadline_dists = batch['deadline_dists']
        
        # Encode
        event_emb = self.event_embed(event_types)
        time_emb = self.time_embed(time_intervals.unsqueeze(-1))
        dl_emb = self.deadline_embed(deadline_dists.unsqueeze(-1))
        
        x = torch.cat([event_emb, time_emb, dl_emb], dim=-1)
        
        # Mamba-like encoding (using LSTM for speed)
        lstm_out, _ = self.lstm(x)
        reprs = self.norm(lstm_out[:, -1, :])  # Last state
        
        # Multi-scale (mean + last)
        seq_mean = lstm_out.mean(dim=1)
        multi_scale = (reprs + seq_mean) / 2
        
        # Prototype
        dists = torch.cdist(multi_scale, self.proto_embed)
        proto_w = torch.softmax(-dists, dim=-1)
        proto = torch.matmul(proto_w, self.proto_embed)
        
        if task == 'pretrain':
            # Next-event prediction
            next_event_logits = self.next_event_head(multi_scale)
            return {'next_event_logits': next_event_logits, 'repr': multi_scale}
        else:
            # Risk prediction
            combined = torch.cat([multi_scale, proto], dim=-1)
            risk = self.risk_head(combined)
            if return_repr:
                return {'risk': risk, 'repr': multi_scale, 'proto_w': proto_w}
            return {'risk': risk}


def collate_train(samples):
    """Collate for training"""
    max_len = min(max(s['n_events'] for s in samples), 500)
    
    event_types, time_intervals, deadline_dists, risks = [], [], [], []
    
    for s in samples:
        n = min(s['n_events'], max_len)
        
        event_types.append(s['event_types'][:n] if n == max_len else 
                         torch.cat([s['event_types'][:n], torch.zeros(max_len - n, dtype=torch.long)]))
        time_intervals.append(s['time_intervals'][:n] if n == max_len else 
                            torch.cat([s['time_intervals'][:n], torch.zeros(max_len - n)]))
        deadline_dists.append(s['deadline_dists'][:n] if n == max_len else 
                             torch.cat([s['deadline_dists'][:n], torch.zeros(max_len - n)]))
        risks.append(s['risk'])
    
    return {
        'event_types': torch.stack(event_types),
        'time_intervals': torch.stack(time_intervals),
        'deadline_dists': torch.stack(deadline_dists),
        'risk': torch.LongTensor(risks)
    }


def collate_pretrain(samples):
    """Collate for pretraining (next-event prediction)"""
    max_len = min(max(s['n_events'] for s in samples), 500)
    
    batch_event_types, batch_time_intervals, batch_deadline_dists = [], [], []
    batch_next_events = []
    batch_masks = []
    
    for s in samples:
        n = min(s['n_events'], max_len)
        
        if n > 1:
            # Input: first n-1 events
            event_seq = s['event_types'][:n-1]
            time_seq = s['time_intervals'][:n-1]
            deadline_seq = s['deadline_dists'][:n-1]
            next_event = s['event_types'][n-1]
        else:
            # Only one event - use it as both input and "next"
            event_seq = s['event_types'][:1]
            time_seq = s['time_intervals'][:1]
            deadline_seq = s['deadline_dists'][:1]
            next_event = s['event_types'][0]
        
        # Pad to max_len
        pad_len = max_len - len(event_seq)
        if pad_len > 0:
            event_seq = torch.cat([event_seq, torch.zeros(pad_len, dtype=torch.long)])
            time_seq = torch.cat([time_seq, torch.zeros(pad_len)])
            deadline_seq = torch.cat([deadline_seq, torch.zeros(pad_len)])
        
        batch_event_types.append(event_seq)
        batch_time_intervals.append(time_seq)
        batch_deadline_dists.append(deadline_seq)
        batch_next_events.append(next_event)
    
    return {
        'event_types': torch.stack(batch_event_types),
        'time_intervals': torch.stack(batch_time_intervals),
        'deadline_dists': torch.stack(batch_deadline_dists),
        'next_events': torch.stack(batch_next_events)
    }


def run_experiment():
    start_time = time.time()
    set_seed(42)
    
    print("="*60)
    print("Mamba 7-dim Full Experiment (Minimal Version)")
    print("="*60)
    
    # Step 1: Load data
    print("\n[STEP 1] Loading data...")
    processor = MAMBAFeatureProcessor(
        '/tmp/IDE_logs/IDE_logs.csv',
        '/tmp/IDE_logs/passed.csv',
        cache_dir='/tmp/mamba_minimal_cache'
    )
    processor.load_data()
    processor.encode_all_students()
    
    # Build dataset
    dataset = []
    for sid, enc in processor.encodings.items():
        label = processor.get_student_labels()[sid]
        dataset.append({
            'event_types': enc['event_types'],
            'time_intervals': enc['time_intervals'],
            'deadline_dists': enc['deadline_dists'],
            'part_ids': enc['part_ids'],
            'n_events': enc['n_events'],
            'risk': 0 if label['passed'] else 1
        })
    
    y = np.array([d['risk'] for d in dataset])
    print(f"Students: {len(dataset)}, At-risk: {y.sum()}")
    
    # Step 2: Pretraining
    print("\n[STEP 2] Pretraining (next-event prediction)...")
    model = MinimalMamba(n_event_types=7, d_model=24, d_state=6)
    
    pretrain_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_pretrain)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    # 1 epoch of pretraining
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in tqdm(pretrain_loader, desc="Pretrain"):
        optimizer.zero_grad()
        
        # Forward for pretraining (next-event prediction)
        outputs = model(batch, task='pretrain')
        
        # Next-event prediction loss
        loss = nn.CrossEntropyLoss()(outputs['next_event_logits'], batch['next_events'])
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    
    print(f"Pretrain Loss: {total_loss/n_batches:.4f}")
    
    # Save pretrained state
    pretrained_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Steps 3-5: Fine-tuning with 5-fold CV
    print("\n[STEPS 3-5] Multi-scale + Prototype + Fine-tuning (5-fold CV)...")
    
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(dataset, y)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        
        train_data = [dataset[i] for i in train_idx]
        test_data = [dataset[i] for i in test_idx]
        
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_train)
        test_loader = DataLoader(test_data, batch_size=16, shuffle=False, collate_fn=collate_train)
        
        # New model with pretrained weights
        fold_model = MinimalMamba(n_event_types=7, d_model=24, d_state=6)
        fold_model.load_state_dict(pretrained_state)
        
        # Freeze except head
        for name, param in fold_model.named_parameters():
            if 'risk_head' not in name:
                param.requires_grad = False
        
        trainable_params = [p for p in fold_model.parameters() if p.requires_grad]
        print(f"Fine-tuning {sum(p.numel() for p in trainable_params):,} params")
        
        finetune_opt = optim.AdamW(trainable_params, lr=1e-3)
        
        # 3 epochs of finetuning
        for epoch in range(3):
            fold_model.train()
            for batch in tqdm(train_loader, desc=f"Finetune E{epoch+1}", leave=False):
                finetune_opt.zero_grad()
                outputs = fold_model(batch)
                loss = nn.CrossEntropyLoss()(outputs['risk'], batch['risk'])
                loss.backward()
                finetune_opt.step()
        
        # Evaluate
        fold_model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in test_loader:
                outputs = fold_model(batch)
                preds = outputs['risk'].argmax(dim=-1).numpy()
                all_preds.extend(preds)
                all_labels.extend(batch['risk'].numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        results = {
            'fold': fold + 1,
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
        }
        fold_results.append(results)
        print(f"Acc: {results['accuracy']:.4f}, F1: {results['f1']:.4f}")
    
    # Step 4: Prototype Discovery
    print("\n[STEP 4] Prototype Discovery (K-Means)...")
    
    fold_model.eval()
    all_reprs, all_proto_ids, all_labels = [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            outputs = fold_model(batch, return_repr=True)
            all_reprs.append(outputs['repr'].numpy())
            all_proto_ids.extend(outputs['proto_w'].argmax(dim=-1).numpy())
            all_labels.extend(batch['risk'].numpy())
    
    all_reprs = np.vstack(all_reprs)
    all_proto_ids = np.array(all_proto_ids)
    all_labels_proto = np.array(all_labels)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(all_reprs)
    
    print("\nPrototype Clusters:")
    for i in range(4):
        mask = kmeans_labels == i
        n = mask.sum()
        risk_rate = all_labels_proto[mask].mean() if n > 0 else 0
        print(f"  Cluster {i}: n={n}, risk_rate={risk_rate:.2%}")
    
    # Step 6: Interpretability
    print("\n[STEP 6] Interpretability...")
    
    # Get event importance from embedding
    event_importance = fold_model.event_embed.weight.norm(dim=-1)
    event_importance = torch.softmax(event_importance, dim=0)
    
    print("\nEvent Type Importance:")
    event_names = ['focus_gained', 'focus_lost', 'text_insert', 
                   'text_remove', 'text_paste', 'run', 'submit']
    for i, name in enumerate(event_names):
        print(f"  {name}: {event_importance[i].item():.4f}")
    
    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    acc_mean = np.mean([r['accuracy'] for r in fold_results])
    acc_std = np.std([r['accuracy'] for r in fold_results])
    f1_mean = np.mean([r['f1'] for r in fold_results])
    f1_std = np.std([r['f1'] for r in fold_results])
    prec_mean = np.mean([r['precision'] for r in fold_results])
    prec_std = np.std([r['precision'] for r in fold_results])
    rec_mean = np.mean([r['recall'] for r in fold_results])
    rec_std = np.std([r['recall'] for r in fold_results])
    
    print(f"\n5-Fold CV Results:")
    print(f"  Accuracy:  {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  F1 Score: {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"  Precision: {prec_mean:.4f} ± {prec_std:.4f}")
    print(f"  Recall:   {rec_mean:.4f} ± {rec_std:.4f}")
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    
    results_summary = {
        'accuracy_mean': acc_mean,
        'accuracy_std': acc_std,
        'f1_mean': f1_mean,
        'f1_std': f1_std,
        'precision_mean': prec_mean,
        'precision_std': prec_std,
        'recall_mean': rec_mean,
        'recall_std': rec_std,
        'fold_results': fold_results,
        'elapsed_minutes': elapsed / 60
    }
    
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'mamba_7dim_full_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'mamba_7dim_full_results.json'}")
    
    return results_summary


if __name__ == "__main__":
    results = run_experiment()
