"""RF-LSTM-Attn v2 trainer - 5-fold CV, failed=1 label convention."""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, set_seed
from common.feature_engineering import build_feature_matrix
from common.evaluator import evaluate, summarize_fold_results, print_results_table

# v2 model lives in models/rf_lstm/ (sibling to v1) but a separate module
from models.rf_lstm.rf_lstm_v2_model import RFLSTMAttnV2, count_parameters

EVENT_TYPES = ['text_insert', 'text_remove', 'text_paste',
              'focus_gained', 'focus_lost', 'run', 'submit']

def build_7dim(ide_logs, students):
    counts = (ide_logs.groupby(['student', 'eventType']).size().unstack(fill_value=0))
    counts = counts.reindex(columns=EVENT_TYPES, fill_value=0).reindex(index=students, fill_value=0)
    return counts.values.astype(np.float32)

def make_rf_oof(X7, y, n_splits=5, seed=42):
    n = len(y)
    oof = np.zeros((n, 2), dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(X7, y):
        clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=seed, n_jobs=-1)
        clf.fit(X7[tr], y[tr])
        oof[va] = clf.predict_proba(X7[va])
    return oof

def metric(y_true, y_pred, y_prob):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_true, y_prob)),
    }

def summary(metrics):
    keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    out = {'n_folds_used': len(metrics)}
    for k in keys:
        vals = [m[k] for m in metrics]
        out[f'{k}_mean'] = float(np.mean(vals))
        out[f'{k}_std'] = float(np.std(vals))
    return out

def bce_loss(logits, target):
    return nn.functional.binary_cross_entropy_with_logits(logits, target)

def train_one_fold(model, optimizer, scheduler, device,
                    x_7d_tr, x_46d_tr, rf_tr, y_tr,
                    x_7d_va, x_46d_va, rf_va, y_va,
                    epochs=80, batch_size=32, patience=15):
    best_vloss = float('inf'); best_state = None; pat = 0
    x_7d_tr_t = torch.tensor(x_7d_tr, dtype=torch.float32, device=device)
    x_46d_tr_t = torch.tensor(x_46d_tr, dtype=torch.float32, device=device)
    rf_tr_t = torch.tensor(rf_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    x_7d_va_t = torch.tensor(x_7d_va, dtype=torch.float32, device=device)
    x_46d_va_t = torch.tensor(x_46d_va, dtype=torch.float32, device=device)
    rf_va_t = torch.tensor(rf_va, dtype=torch.float32, device=device)
    y_va_t = torch.tensor(y_va, dtype=torch.float32, device=device)
    n = len(y_tr)
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(n)
        for i in range(0, n, batch_size):
            b = idx[i:i+batch_size]
            logits = model(x_7d_tr_t[b], x_46d_tr_t[b], rf_tr_t[b])
            logits = torch.clamp(logits, min=-30, max=30)
            loss = bce_loss(logits, y_tr_t[b])
            if torch.isnan(loss): continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            v_logits = model(x_7d_va_t, x_46d_va_t, rf_va_t)
            v_loss = bce_loss(v_logits, y_va_t).item()
        if v_loss < best_vloss:
            best_vloss = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat >= patience: break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        v_logits = model(x_7d_va_t, x_46d_va_t, rf_va_t).cpu().numpy()
    return 1.0 / (1.0 + np.exp(-v_logits))

def main(output_dir=None, folds=5, epochs=80, batch_size=32, patience=15, seed=42):
    if output_dir is None:
        output_dir = os.path.join(_PROJECT_ROOT, 'outputs', 'unified_compare', 'rf_lstm_v2')
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 72)
    print("  RF-LSTM-Attn v2: Deep RF + LSTM architecture-level fusion")
    print(f"  output: {output_dir}")
    print("=" * 72)
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    ide_logs, passed_df = load_ide_logs()
    students = passed_df['student'].values
    y_passed = passed_df['passed'].values.astype(int)
    n = len(students)
    y_eval = 1 - y_passed

    X7 = build_7dim(ide_logs, students)
    X_full, _, _ = build_feature_matrix(ide_logs, passed_df)

    print("\n[Step 1] Computing 5-fold OOF RF probs (frozen feature)...")
    rf_probs = make_rf_oof(X7, y_eval, n_splits=folds, seed=seed)

    print(f"\n[Step 2] RF-LSTM-Attn v2 5-fold CV (epochs={epochs}, batch={batch_size}, patience={patience})")
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_idx_arr = np.zeros(n, dtype=int)
    all_probs = np.zeros(n, dtype=np.float64)
    fold_metrics = []

    for fi, (tr, va) in enumerate(skf.split(X7, y_eval), 1):
        scaler = StandardScaler()
        x_46d_tr = scaler.fit_transform(X_full[tr]).astype(np.float32)
        x_46d_va = scaler.transform(X_full[va]).astype(np.float32)
        x_7d_tr = X7[tr].astype(np.float32)
        x_7d_va = X7[va].astype(np.float32)
        rf_tr = rf_probs[tr].astype(np.float32)
        rf_va = rf_probs[va].astype(np.float32)

        torch.manual_seed(seed + fi); np.random.seed(seed + fi)
        model = RFLSTMAttnV2(hidden=64, lstm_layers=2, dropout=0.2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        probs_va = train_one_fold(
            model, optimizer, scheduler, device,
            x_7d_tr, x_46d_tr, rf_tr, y_eval[tr],
            x_7d_va, x_46d_va, rf_va, y_eval[va],
            epochs=epochs, batch_size=batch_size, patience=patience,
        )

        all_probs[va] = probs_va
        fold_idx_arr[va] = fi - 1
        yhat = (probs_va > 0.5).astype(int)
        m = metric(y_eval[va], yhat, probs_va)
        fold_metrics.append(m)
        print(f"  Fold {fi}/{folds}  Acc={m['accuracy']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  AUC={m['auc']:.4f}")

    cv = summary(fold_metrics)
    print_results_table('RF-LSTM-Attn v2', {
        **{k: cv[k] for k in cv if k.endswith('_mean') or k.endswith('_std')},
        'folds': fold_metrics
    })

    np.save(os.path.join(output_dir, 'probs.npy'), all_probs)
    np.save(os.path.join(output_dir, 'labels.npy'), y_eval.astype(np.int8))
    np.save(os.path.join(output_dir, 'fold_idx.npy'), fold_idx_arr)

    out = {
        'model': 'RF-LSTM-Attn v2',
        'label_convention': 'y=1=failed',
        'n_folds': folds,
        'architecture': {
            'n_params': int(count_parameters(RFLSTMAttnV2())),
            'streams': ['RF-Event Encoder (7-dim + RF probs)', 'BiLSTM Stream (46-dim + RF per-step)', 'Cross-View Attention (RF event attends to LSTM)'],
            'fusion': 'Concat(attended_RF_event + LSTM_last_hidden + raw_7dim + RF_probs) -> 2-layer MLP',
            'loss': 'Vanilla BCE + L2(1e-2)',
        },
        'cv_results': {k: cv[k] for k in cv if k != 'n_folds_used'},
        'fold_details': fold_metrics,
        'n_samples': int(n),
        'n_failed': int(y_eval.sum()),
        'n_passed': int(n - y_eval.sum()),
    }
    with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)

    print(f"\n[OK] Saved to: {output_dir}")
    print(f"\n=== Summary ===")
    for k in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        print(f"  {k.upper()} = {cv[k+'_mean']:.4f} ± {cv[k+'_std']:.4f}")
    print(f"  Parameters: {count_parameters(RFLSTMAttnV2()):,}")
    return cv

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='RF-LSTM-Attn v2 trainer')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--output-dir', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    main(output_dir=args.output_dir, folds=args.folds, epochs=args.epochs,
         batch_size=args.batch_size, patience=args.patience, seed=args.seed)