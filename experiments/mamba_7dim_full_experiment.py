"""
Mamba 7-dim Full Experiment - Complete 6-Step Implementation

Step 1: Data preprocessing (7-dim event one-hot + time interval + deadline + exercise ID)
Step 2: Mamba pretraining (self-supervised next-event prediction on 28.58M events)
Step 3: Multi-scale features (fine/medium/coarse attention fusion)
Step 4: Prototype discovery (K-Means clustering into 4 learning modes)
Step 5: Prediction fine-tuning (classification: dropout/fail risk)
Step 6: Interpretability (important time segments, event types, learning prototypes)
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
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from tqdm import tqdm
import pickle

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))
from features.mamba_features import MAMBAFeatureProcessor, encode_events, collate_mamba_batch
from models.mamba_student import (
    MambaEncoder, MultiScaleFeatureExtractor, StudentPrototypeLayer,
    RMSNorm, SiLU, create_mamba_student_model
)


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MAMBAStudentModelFull(nn.Module):
    """
    Complete Mamba model with all 6 steps integrated.
    
    For pretraining: predicts next event type from event sequence.
    For fine-tuning: predicts student risk (dropout/fail).
    """
    def __init__(self, n_event_types=7, d_model=48, n_layers=4, d_state=12,
                 n_prototypes=4, dropout=0.2):
        super().__init__()
        
        # Step 1: Event encoding (7-dim one-hot + time + deadline + exercise)
        self.n_event_types = n_event_types
        self.d_model = d_model
        
        # Embeddings
        self.event_embed = nn.Embedding(n_event_types, 16)
        self.time_embed = nn.Linear(1, 8)
        self.exercise_embed = nn.Embedding(31, 16)  # exercises 0-30
        self.deadline_embed = nn.Linear(1, 8)
        
        # Input projection: 16 + 8 + 16 + 8 = 48
        self.input_proj = nn.Linear(48, d_model)
        
        # Step 2: Mamba backbone
        self.mamba = MambaEncoder(d_model=d_model, n_layers=n_layers, d_state=d_state)
        
        # Step 3: Multi-scale feature extractor
        self.multi_scale = MultiScaleFeatureExtractor(d_model=d_model)
        
        # Step 4: Prototype discovery
        self.prototype = StudentPrototypeLayer(d_model, n_prototypes)
        
        # Output dimensions
        self.final_norm = RMSNorm(d_model)
        
        # Step 5: Prediction heads
        # For fine-tuning: risk classification
        self.risk_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)
        )
        
        # For pretraining: next event prediction
        self.next_event_head = nn.Linear(d_model, n_event_types)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, batch, task='finetune'):
        """
        Forward pass.
        task='pretrain': outputs next-event prediction logits
        task='finetune': outputs risk prediction
        """
        event_types = batch['event_types']
        time_intervals = batch['time_intervals']
        exercise_ids = batch['exercise_ids']
        deadline_dists = batch['deadline_dists']
        part_ids = batch.get('part_ids', torch.ones_like(event_types))
        
        # Step 1: Encode events
        event_emb = self.event_embed(event_types)
        time_emb = self.time_embed(time_intervals.unsqueeze(-1))
        ex_emb = self.exercise_embed(exercise_ids.clamp(0, 30))
        dl_emb = self.deadline_embed(deadline_dists.unsqueeze(-1))
        
        x = torch.cat([event_emb, time_emb, ex_emb, dl_emb], dim=-1)
        x = self.input_proj(x)
        
        # Step 2: Mamba encoding
        mamba_out = self.mamba(x)
        
        # Step 3: Multi-scale features
        multi_scale_repr = self.multi_scale(mamba_out, exercise_ids, part_ids)
        multi_scale_repr = self.final_norm(multi_scale_repr)
        
        # Step 4: Prototype enhancement
        proto_repr = self.prototype(multi_scale_repr)
        
        # Combine representations
        combined = torch.cat([multi_scale_repr, proto_repr], dim=-1)
        combined = self.dropout(combined)
        
        if task == 'pretrain':
            # Return sequence output for next-event prediction
            # Use the last position's representation for next-event prediction
            last_repr = mamba_out[:, -1, :]  # (batch, d_model)
            next_event_logits = self.next_event_head(last_repr)
            return {
                'next_event_logits': next_event_logits,
                'mamba_out': mamba_out,
                'repr': multi_scale_repr
            }
        else:
            # Step 5: Risk prediction
            risk_pred = self.risk_head(combined)
            return {
                'risk': risk_pred,
                'repr': multi_scale_repr,
                'proto': proto_repr,
                'mamba_out': mamba_out
            }
    
    def get_interpretability(self, batch):
        """
        Step 6: Interpretability analysis
        """
        with torch.no_grad():
            outputs = self.forward(batch, task='finetune')
            
            event_types = batch['event_types']
            exercise_ids = batch['exercise_ids']
            mamba_out = outputs['mamba_out']
            
            # 1. Event type importance (embedding norms)
            event_importance = self.event_embed.weight.norm(dim=-1)
            event_importance = torch.softmax(event_importance, dim=0)
            
            # 2. Temporal importance (per 100-event windows)
            seq_len = mamba_out.shape[1]
            window_size = 100
            n_windows = seq_len // window_size
            
            if n_windows > 0:
                windows = mamba_out[:, :n_windows*window_size].reshape(-1, n_windows, window_size, mamba_out.shape[-1])
                window_scores = windows.mean(dim=2).norm(dim=-1)
                temporal_importance = torch.softmax(window_scores, dim=-1)
            else:
                temporal_importance = torch.ones(mamba_out.shape[0], 1, device=mamba_out.device) / mamba_out.shape[0]
            
            # 3. Prototype assignment
            student_repr = outputs['repr']
            dists = torch.cdist(student_repr, self.prototype.prototype_centers)
            proto_weights = torch.softmax(-dists, dim=-1)
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


class StudentSequenceDataset(Dataset):
    """Dataset for student event sequences"""
    def __init__(self, encodings, labels, student_ids):
        self.encodings = encodings
        self.labels = labels
        self.student_ids = student_ids
    
    def __len__(self):
        return len(self.student_ids)
    
    def __getitem__(self, idx):
        sid = self.student_ids[idx]
        enc = self.encodings[sid]
        label = self.labels[sid]
        
        return {
            'student_id': sid,
            'event_types': enc['event_types'],
            'time_intervals': enc['time_intervals'],
            'exercise_ids': enc['exercise_ids'],
            'part_ids': enc['part_ids'],
            'deadline_dists': enc['deadline_dists'],
            'n_events': enc['n_events'],
            'risk': torch.LongTensor([label['risk']]),
            'grade': torch.FloatTensor([label['grade']])
        }


def next_event_collate(samples):
    """Collate for next-event prediction training"""
    batch_size = len(samples)
    
    max_len = max(s['n_events'] for s in samples)
    max_len = min(max_len, 8000)  # Cap for memory (reduced for CPU training)
    
    event_types = []
    time_intervals = []
    exercise_ids = []
    part_ids = []
    deadline_dists = []
    masks = []
    next_event_labels = []
    
    for s in samples:
        n = min(s['n_events'], max_len)
        
        if n > 1:
            # Input: first n-1 events
            event_types.append(s['event_types'][:n-1])
            time_intervals.append(s['time_intervals'][:n-1])
            exercise_ids.append(s['exercise_ids'][:n-1])
            part_ids.append(s['part_ids'][:n-1])
            deadline_dists.append(s['deadline_dists'][:n-1])
            masks.append(torch.ones(n-1))
            # Target: event at position n (next event)
            next_event_labels.append(s['event_types'][n-1])
        else:
            event_types.append(s['event_types'][:1])
            time_intervals.append(s['time_intervals'][:1])
            exercise_ids.append(s['exercise_ids'][:1])
            part_ids.append(s['part_ids'][:1])
            deadline_dists.append(s['deadline_dists'][:1])
            masks.append(torch.ones(1))
            next_event_labels.append(s['event_types'][0])
    
    import torch.nn.functional as F
    
    return {
        'event_types': torch.stack(event_types),
        'time_intervals': torch.stack(time_intervals),
        'exercise_ids': torch.stack(exercise_ids),
        'part_ids': torch.stack(part_ids),
        'deadline_dists': torch.stack(deadline_dists),
        'mask': torch.stack(masks),
        'next_event_labels': torch.stack(next_event_labels)
    }


def run_full_experiment():
    """
    Run complete 6-step Mamba experiment with real training
    """
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ============================================================
    # STEP 1: Data Preprocessing
    # ============================================================
    print("\n" + "="*60)
    print("STEP 1: Data Preprocessing (7-dim event encoding)")
    print("="*60)
    
    ide_logs_path = '/tmp/IDE_logs/IDE_logs.csv'
    passed_path = '/tmp/IDE_logs/passed.csv'
    cache_dir = '/tmp/mamba_full_cache'
    
    processor = MAMBAFeatureProcessor(
        ide_logs_path, passed_path, cache_dir
    )
    processor.load_data()
    processor.encode_all_students()
    processor.get_data_summary()
    
    # Get encodings and labels
    encodings = processor.encodings
    raw_labels = processor.get_student_labels()
    
    # Convert to risk labels (0=passed, 1=failed/at-risk)
    labels = {}
    for sid, label_data in raw_labels.items():
        labels[sid] = {
            'grade': label_data['grade'],
            'risk': 0 if label_data['passed'] else 1
        }
    
    student_ids = list(encodings.keys())
    y = np.array([labels[sid]['risk'] for sid in student_ids])
    
    print(f"\nTotal students: {len(student_ids)}")
    print(f"Positive (at-risk): {y.sum()}, Negative (passed): {len(y) - y.sum()}")
    
    # ============================================================
    # STEP 2: Mamba Pretraining (Self-Supervised)
    # ============================================================
    print("\n" + "="*60)
    print("STEP 2: Mamba Pretraining (Next-Event Prediction on 28.58M events)")
    print("="*60)
    
    # Create dataset
    dataset = StudentSequenceDataset(encodings, labels, student_ids)
    
    # Use ALL students for pretraining (all 28.58M events)
    pretrain_loader = DataLoader(
        dataset, batch_size=16, shuffle=True,
        collate_fn=next_event_collate
    )
    
    # Create model
    model = MAMBAStudentModelFull(
        n_event_types=7,
        d_model=48,
        n_layers=4,
        d_state=12,
        n_prototypes=4,
        dropout=0.2
    )
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Pretraining optimizer
    pretrain_epochs = 3  # Limited epochs for CPU training
    pretrainoptimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    pretrain_scheduler = optim.lr_scheduler.CosineAnnealingLR(pretrainoptimizer, T_max=pretrain_epochs)
    
    print(f"\nPretraining on {len(dataset)} students for {pretrain_epochs} epochs...")
    
    for epoch in range(pretrain_epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in tqdm(pretrain_loader, desc=f"Pretrain Epoch {epoch+1}/{pretrain_epochs}"):
            # Move to device
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                           for k, v in batch.items()}
            
            pretrainoptimizer.zero_grad()
            
            # Forward pass for next-event prediction
            outputs = model(batch_device, task='pretrain')
            
            # Next-event prediction loss
            next_event_logits = outputs['next_event_logits']
            next_event_targets = batch_device['next_event_labels']
            
            loss = nn.CrossEntropyLoss()(next_event_logits, next_event_targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            pretrainoptimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        pretrain_scheduler.step()
        avg_loss = total_loss / n_batches
        
        # Compute accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in pretrain_loader:
                batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                               for k, v in batch.items()}
                outputs = model(batch_device, task='pretrain')
                preds = outputs['next_event_logits'].argmax(dim=-1)
                targets = batch_device['next_event_labels']
                correct += (preds == targets).sum().item()
                total += len(targets)
        
        acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Next-Event Acc = {acc:.4f}")
    
    print("\nPretraining completed!")
    
    # ============================================================
    # STEP 5: Prediction Fine-tuning (5-fold CV)
    # ============================================================
    print("\n" + "="*60)
    print("STEP 5: Prediction Fine-tuning (5-fold Cross-Validation)")
    print("="*60)
    
    # Save pretrained model
    pretrained_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # 5-fold cross-validation
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(student_ids, y)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        
        train_sids = [student_ids[i] for i in train_idx]
        test_sids = [student_ids[i] for i in test_idx]
        
        train_dataset = Subset(dataset, [dataset.student_ids.index(sid) for sid in train_sids])
        test_dataset = Subset(dataset, [dataset.student_ids.index(sid) for sid in test_sids])
        
        # Use standard collate for fine-tuning
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, 
                                  collate_fn=lambda x: {k: v for k, v in next_event_collate(x).items() 
                                                        if k != 'next_event_labels'})
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                                 collate_fn=lambda x: {k: v for k, v in next_event_collate(x).items() 
                                                       if k != 'next_event_labels'})
        
        # Create new model and load pretrained weights
        fold_model = MAMBAStudentModelFull(
            n_event_types=7, d_model=48, n_layers=4, d_state=12,
            n_prototypes=4, dropout=0.2
        )
        fold_model.load_state_dict(pretrained_state)
        fold_model.to(device)
        
        # Freeze Mamba backbone, train only prediction heads
        for name, param in fold_model.named_parameters():
            if 'risk_head' not in name:
                param.requires_grad = False
        
        trainable_params = [p for p in fold_model.parameters() if p.requires_grad]
        print(f"Fine-tuning {sum(p.numel() for p in trainable_params):,} parameters")
        
        finetuneoptimizer = optim.AdamW(trainable_params, lr=1e-3, weight_decay=0.01)
        
        # Fine-tune
        finetune_epochs = 10
        for epoch in range(finetune_epochs):
            fold_model.train()
            for batch in tqdm(train_loader, desc=f"Finetune Epoch {epoch+1}/{finetune_epochs}", leave=False):
                batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                               for k, v in batch.items()}
                
                finetuneoptimizer.zero_grad()
                outputs = fold_model(batch_device, task='finetune')
                risk_loss = nn.CrossEntropyLoss()(outputs['risk'], batch_device['risk'].squeeze())
                risk_loss.backward()
                finetuneoptimizer.step()
        
        # Evaluate
        fold_model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                               for k, v in batch.items()}
                outputs = fold_model(batch_device, task='finetune')
                probs = torch.softmax(outputs['risk'], dim=-1)[:, 1].cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_preds.extend(preds)
                all_labels.extend(batch_device['risk'].numpy().flatten())
        
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
        
        print(f"Fold {fold+1} - Acc: {results['accuracy']:.4f}, F1: {results['f1']:.4f}, "
              f"Precision: {results['precision']:.4f}, Recall: {results['recall']:.4f}")
    
    # ============================================================
    # STEP 4: Prototype Discovery (after fine-tuning)
    # ============================================================
    print("\n" + "="*60)
    print("STEP 4: Prototype Discovery (K-Means Clustering)")
    print("="*60)
    
    # Extract representations from last fold model
    fold_model.eval()
    all_reprs = []
    all_proto_ids = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                           for k, v in batch.items()}
            outputs = fold_model(batch_device, task='finetune')
            reprs = outputs['repr'].cpu().numpy()
            
            # Get prototype assignments
            dists = torch.cdist(outputs['repr'], fold_model.prototype.prototype_centers)
            proto_ids = dists.argmin(dim=-1).cpu().numpy()
            
            all_reprs.append(reprs)
            all_proto_ids.extend(proto_ids)
    
    all_reprs = np.vstack(all_reprs)
    all_proto_ids = np.array(all_proto_ids)
    
    # K-Means clustering on representations
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(all_reprs)
    
    print("\nPrototype Analysis:")
    for i in range(4):
        mask = kmeans_labels == i
        n_samples = mask.sum()
        risk_rate = all_labels[mask].mean() if n_samples > 0 else 0
        print(f"  Cluster {i}: n={n_samples}, risk_rate={risk_rate:.2%}")
    
    # ============================================================
    # STEP 6: Interpretability Analysis
    # ============================================================
    print("\n" + "="*60)
    print("STEP 6: Interpretability Analysis")
    print("="*60)
    
    # Get interpretability from fold model
    interp = fold_model.get_interpretability(batch_device)
    
    print("\nEvent Type Importance:")
    for i, name in enumerate(interp['event_type_names']):
        print(f"  {name}: {interp['event_importance'][i]:.4f}")
    
    print(f"\nTemporal Importance (window size=100):")
    print(f"  Number of windows: {len(interp['temporal_importance'][0]) if len(interp['temporal_importance']) > 0 else 0}")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    # Aggregate fold results
    acc_mean = np.mean([r['accuracy'] for r in fold_results])
    acc_std = np.std([r['accuracy'] for r in fold_results])
    f1_mean = np.mean([r['f1'] for r in fold_results])
    f1_std = np.std([r['f1'] for r in fold_results])
    prec_mean = np.mean([r['precision'] for r in fold_results])
    prec_std = np.std([r['precision'] for r in fold_results])
    rec_mean = np.mean([r['recall'] for r in fold_results])
    rec_std = np.std([r['recall'] for r in fold_results])
    
    print(f"\n5-Fold Cross-Validation Results:")
    print(f"  Accuracy:  {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  F1 Score:  {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"  Precision:  {prec_mean:.4f} ± {prec_std:.4f}")
    print(f"  Recall:    {rec_mean:.4f} ± {rec_std:.4f}")
    
    results_summary = {
        'accuracy_mean': acc_mean,
        'accuracy_std': acc_std,
        'f1_mean': f1_mean,
        'f1_std': f1_std,
        'precision_mean': prec_mean,
        'precision_std': prec_std,
        'recall_mean': rec_mean,
        'recall_std': rec_std,
        'fold_results': fold_results
    }
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'mamba_7dim_full_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'mamba_7dim_full_results.json'}")
    
    return results_summary


if __name__ == "__main__":
    results = run_full_experiment()
