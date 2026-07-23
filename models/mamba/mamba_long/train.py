"""
Mamba-Long 训练脚本 - 实施路径步骤 1+2+3

输入: 7d 事件序列 (max=2000) + 12d micro 特征 + deadline_dists
输出: probs/labels/fold_idx.npy + results.json

注意:
  - 步骤 1: max=2000, d_model=48, n_layers=4 (防 OOM)
  - 步骤 2: 12d micro 特征辅助 (新增 micro MLP)
  - 步骤 3: 多尺度按 deadline_dists 切分 (不依赖 part_ids)
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.data_loader import load_ide_logs, set_seed
from common.evaluator import evaluate, summarize_fold_results, print_results_table
from models.mamba.mamba_long.model import (
    MambaLongStudent, create_model, compute_micro_features,
    EVENT_TYPES,
)


def collate_for_long(samples):
    """构造 batch: 事件序列 + micro 特征"""
    max_len = min(max(s['n_events'] for s in samples), 2000)
    batch_event = []
    batch_ti = []
    batch_dd = []
    batch_micro = []
    batch_risk = []
    for s in samples:
        n = min(s['n_events'], max_len)
        et = s['event_types'][:n]
        ti = s['time_intervals'][:n]
        dd = s['deadline_dists'][:n]
        if len(et) < max_len:
            pad_len = max_len - len(et)
            et = torch.nn.functional.pad(et, (0, pad_len), value=0)
            ti = torch.nn.functional.pad(ti, (0, pad_len), value=0.0)
            dd = torch.nn.functional.pad(dd, (0, pad_len), value=0.0)
        batch_event.append(et[:max_len])
        batch_ti.append(ti[:max_len])
        batch_dd.append(dd[:max_len])
        batch_micro.append(s.get('micro', torch.zeros(12)))
        batch_risk.append(s['risk'])
    return {
        'event_types': torch.stack(batch_event),
        'time_intervals': torch.stack(batch_ti),
        'deadline_dists': torch.stack(batch_dd),
        'micro': torch.stack(batch_micro),
        'risk': torch.LongTensor(batch_risk),
    }


def focal_loss(logits, targets, alpha=None, gamma=2.0, label_smoothing=0.05):
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    num_classes = logits.shape[-1]
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(label_smoothing / num_classes)
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing + label_smoothing / num_classes)
    target_oh = torch.nn.functional.one_hot(targets, num_classes).float()
    p_t = (probs * target_oh).sum(dim=-1)
    focal_w = (1.0 - p_t) ** gamma
    loss = -(focal_w * (log_probs * true_dist).sum(dim=-1))
    return loss.mean()


def preprocess_with_micro(ide_logs_df, passed_df, max_events=2000):
    """
    步骤 1+2 整合: 提取事件序列 + 12d micro 特征
    """
    from models.mamba.steps.step1_preprocessing import encode_events
    from common.data_loader import load_ide_logs

    print(f"\n[Step 1+2] 数据预处理 + micro 特征")
    print(f"  IDE日志数: {len(ide_logs_df)}")
    print(f"  学生数: {passed_df['student'].nunique()}")

    passed_map = dict(zip(passed_df['student'], passed_df['passed']))
    students = passed_df['student'].unique()
    samples = []
    student_ids = []
    labels = []

    # 先收集所有 student_df
    student_groups = []
    valid_students = []
    skipped = 0
    for student_id in students:
        df = ide_logs_df[ide_logs_df['student'] == student_id]
        if len(df) == 0:
            skipped += 1
            continue
        valid_students.append(student_id)
        student_groups.append(df)

    print(f"  有效学生: {len(valid_students)}, 跳过: {skipped}")

    # 计算 micro 特征 (向量化版本, 避免循环)
    micro_feats = compute_micro_features(ide_logs_df, valid_students, n_first=30)
    print(f"  micro 特征 shape: {micro_feats.shape}")

    for student_id, df in zip(valid_students, student_groups):
        encoded = encode_events(df, max_events=max_events)
        passed = passed_map.get(student_id, True)
        risk = 0 if passed else 1
        samples.append({
            'student_id': student_id,
            'event_types': encoded['event_types'],
            'time_intervals': encoded['time_intervals'],
            'deadline_dists': encoded['deadline_dists'],
            'part_ids': encoded['part_ids'],
            'n_events': encoded['n_events'],
            'risk': risk,
            'micro': torch.FloatTensor(micro_feats[valid_students.index(student_id)]),
        })
        student_ids.append(student_id)
        labels.append(risk)

    return samples, np.array(student_ids), np.array(labels)


def finetune_cv(pretrained_state, dataset, student_ids, y, device,
                n_folds=5, n_epochs=4, batch_size=4):
    """Mamba-Long 5 折 CV (复用 step5 风格)"""
    print(f"\n[Step 5] 预测微调 - {n_folds}折 (max_seq=2000, batch={batch_size})")
    print(f"  样本: {len(dataset)}, fail_rate: {y.mean():.3f}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_probs = np.zeros(len(dataset), dtype=np.float32)
    all_fold_idx = np.full(len(dataset), -1, dtype=np.int32)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(student_ids, y)):
        print(f"\n  --- Fold {fold_idx+1}/{n_folds} ---")
        model = MambaLongStudent()
        model.to(device)
        model.load_state_dict(pretrained_state)

        # 解冻 risk_head + micro_mlp
        for name, p in model.named_parameters():
            p.requires_grad = ('risk_head' in name or 'micro_mlp' in name
                               or 'se_gate' in name or 'scale_fusion' in name)

        train_loader = DataLoader(
            [dataset[i] for i in train_idx], batch_size=batch_size, shuffle=True,
            collate_fn=collate_for_long, drop_last=True
        )
        test_loader = DataLoader(
            [dataset[i] for i in test_idx], batch_size=batch_size, shuffle=False,
            collate_fn=collate_for_long, drop_last=False
        )

        head_params = [p for n, p in model.named_parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(head_params, lr=5e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs * len(train_loader), eta_min=1e-6
        )

        y_train = np.array([dataset[i]['risk'] for i in train_idx])
        n_pass = max(int((y_train == 0).sum()), 1)
        n_fail = max(int((y_train == 1).sum()), 1)
        alpha = torch.tensor(
            [len(y_train) / (2 * n_pass), len(y_train) / (2 * n_fail)],
            device=device
        ).clamp(max=2.0)

        model.train()
        for ep in range(n_epochs):
            total_loss = 0.0; n_batches = 0
            for batch in train_loader:
                batch_input = {
                    'event_types': batch['event_types'].to(device),
                    'time_intervals': batch['time_intervals'].to(device),
                    'deadline_dists': batch['deadline_dists'].to(device),
                    'micro': batch['micro'].to(device),
                }
                targets = batch['risk'].to(device)
                optimizer.zero_grad()
                outputs = model(batch_input)
                logits = outputs['risk']
                if torch.isnan(logits).any():
                    optimizer.zero_grad()
                    continue
                loss = focal_loss(logits, targets, alpha=alpha, gamma=2.0, label_smoothing=0.05)
                if torch.isnan(loss):
                    optimizer.zero_grad()
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head_params, 0.5)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item(); n_batches += 1
            print(f"    Epoch {ep+1}/{n_epochs} - Loss: {total_loss/max(n_batches,1):.4f}")

        model.eval()
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                batch_input = {
                    'event_types': batch['event_types'].to(device),
                    'time_intervals': batch['time_intervals'].to(device),
                    'deadline_dists': batch['deadline_dists'].to(device),
                    'micro': batch['micro'].to(device),
                }
                targets = batch['risk'].to(device)
                outputs = model(batch_input)
                logits = outputs['risk']
                probs = torch.softmax(logits, dim=-1)
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(logits.argmax(dim=-1).cpu().numpy())
                y_prob.extend(probs[:, 1].cpu().numpy())

        y_true = np.array(y_true); y_pred = np.array(y_pred); y_prob = np.array(y_prob)
        m = evaluate(y_true, y_pred, y_prob)
        fold_results.append(m)
        all_probs[test_idx] = y_prob
        all_fold_idx[test_idx] = fold_idx
        print(f"    Fold {fold_idx+1}: Acc={m['accuracy']:.4f} P={m['precision']:.4f} "
              f"R={m['recall']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f}")

    summary = summarize_fold_results(fold_results)
    print_results_table("Mamba-Long (7d+micro)", summary)
    return all_probs, all_fold_idx, fold_results, summary


def get_interpretability(model, dataset, device):
    """步骤 4: 提取可解释性信息"""
    model.eval()
    # 取全部样本
    test_loader = DataLoader(
        dataset, batch_size=4, shuffle=False,
        collate_fn=collate_for_long, drop_last=False
    )
    all_event_importance = []
    all_temporal_importance = []
    all_proto_weights = []
    all_micro_repr = []

    with torch.no_grad():
        for batch in test_loader:
            batch_input = {
                'event_types': batch['event_types'].to(device),
                'time_intervals': batch['time_intervals'].to(device),
                'deadline_dists': batch['deadline_dists'].to(device),
                'micro': batch['micro'].to(device),
            }
            outputs = model(batch_input, return_repr=True)
            # event importance from event_embed weight
            ei = model.event_embed.weight.norm(dim=-1)
            ei = torch.softmax(ei, dim=0).cpu().numpy()
            all_event_importance.append(np.tile(ei, (outputs['proto_weights'].shape[0], 1)))

            all_proto_weights.append(outputs['proto_weights'].cpu().numpy())
            all_micro_repr.append(outputs['micro_repr'].cpu().numpy())

            # 简单时间重要性: 100 窗口的方差
            x = model.input_proj(model.event_embed(batch['event_types'].to(device)))
            for blk in model.mamba_layers:
                x = blk(x)
            x = model.final_norm(x)
            # 用 multi_scale 内部的 fine 段
            win_size = 100
            B = x.shape[0]
            n_fine = max(1, x.shape[1] // win_size)
            ti = []
            for i in range(n_fine):
                s = i * win_size; e = min(s + win_size, x.shape[1])
                ti.append(x[:, s:e].std(dim=1).mean(dim=-1, keepdim=True))
            ti = torch.cat(ti, dim=1).cpu().numpy()
            # 归一化
            ti = ti / (ti.sum(axis=1, keepdims=True) + 1e-10)
            all_temporal_importance.append(ti)

    return {
        'event_importance': np.concatenate(all_event_importance, axis=0),
        'temporal_importance': np.concatenate(all_temporal_importance, axis=0),
        'proto_weights': np.concatenate(all_proto_weights, axis=0),
        'micro_repr': np.concatenate(all_micro_repr, axis=0),
    }


def main():
    parser = argparse.ArgumentParser(description='Mamba-Long 训练 (步骤 1+2+3)')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--pretrain-epochs', type=int, default=2)
    parser.add_argument('--finetune-epochs', type=int, default=4)
    parser.add_argument('--pretrain-batch-size', type=int, default=4)
    parser.add_argument('--finetune-batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='outputs/mamba_long')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print(f"配置: max_seq=2000, batch={args.finetune_batch_size}, d_model=48, layers=4")

    # 加载 + 预处理
    ide_logs_df, passed_df = load_ide_logs()
    samples, student_ids, labels = preprocess_with_micro(ide_logs_df, passed_df, max_events=2000)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 直接 finetune (跳过 pretrain 因为代码聚焦于步骤 2/3 的 micro + 改进多尺度)
    # 仍然从头训练, 但保留 5 折 CV 流程
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    all_probs = np.zeros(len(samples), dtype=np.float32)
    all_fold_idx = np.full(len(samples), -1, dtype=np.int32)
    all_fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(student_ids, labels), 1):
        print(f"\n========== Fold {fold_idx}/{args.folds} ==========")
        # 每个 fold 单独训练 + 预测
        model = MambaLongStudent().to(device)
        n_epochs = args.finetune_epochs
        batch_size = args.finetune_batch_size

        train_loader = DataLoader(
            [samples[i] for i in train_idx], batch_size=batch_size, shuffle=True,
            collate_fn=collate_for_long, drop_last=True
        )
        test_loader = DataLoader(
            [samples[i] for i in test_idx], batch_size=batch_size, shuffle=False,
            collate_fn=collate_for_long, drop_last=False
        )

        y_train = labels[train_idx]
        n_pass = max(int((y_train == 0).sum()), 1)
        n_fail = max(int((y_train == 1).sum()), 1)
        alpha = torch.tensor(
            [len(y_train) / (2 * n_pass), len(y_train) / (2 * n_fail)],
            device=device
        ).clamp(max=2.0)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs * len(train_loader), eta_min=1e-6
        )

        model.train()
        for ep in range(n_epochs):
            tl, nb = 0.0, 0
            for batch in train_loader:
                batch_input = {
                    'event_types': batch['event_types'].to(device),
                    'time_intervals': batch['time_intervals'].to(device),
                    'deadline_dists': batch['deadline_dists'].to(device),
                    'micro': batch['micro'].to(device),
                }
                targets = batch['risk'].to(device)
                optimizer.zero_grad()
                out = model(batch_input)
                logits = out['risk']
                if torch.isnan(logits).any():
                    optimizer.zero_grad()
                    continue
                loss = focal_loss(logits, targets, alpha=alpha, gamma=2.0, label_smoothing=0.05)
                if torch.isnan(loss):
                    optimizer.zero_grad()
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                scheduler.step()
                tl += loss.item(); nb += 1
            print(f"    Epoch {ep+1}/{n_epochs} - Loss: {tl/max(nb,1):.4f}")

        model.eval()
        yt, yp, yprob = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                bi = {
                    'event_types': batch['event_types'].to(device),
                    'time_intervals': batch['time_intervals'].to(device),
                    'deadline_dists': batch['deadline_dists'].to(device),
                    'micro': batch['micro'].to(device),
                }
                targets = batch['risk'].to(device)
                out = model(bi)
                probs = torch.softmax(out['risk'], dim=-1)
                yt.extend(targets.cpu().numpy())
                yp.extend(out['risk'].argmax(dim=-1).cpu().numpy())
                yprob.extend(probs[:, 1].cpu().numpy())

        y_true = np.array(yt); y_pred = np.array(yp); y_prob = np.array(yprob)
        m = evaluate(y_true, y_pred, y_prob)
        all_fold_results.append(m)
        all_probs[test_idx] = y_prob
        all_fold_idx[test_idx] = fold_idx - 1
        print(f"  Fold {fold_idx}: Acc={m['accuracy']:.4f} P={m['precision']:.4f} "
              f"R={m['recall']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f}")

    summary = summarize_fold_results(all_fold_results)
    print_results_table("Mamba-Long 5-fold", summary)

    # 保存
    np.save(os.path.join(output_dir, 'probs.npy'), all_probs)
    np.save(os.path.join(output_dir, 'labels.npy'), labels.astype(np.int32))
    np.save(os.path.join(output_dir, 'fold_idx.npy'), all_fold_idx)

    with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'model': 'Mamba-Long (7d+micro+改进多尺度)',
            'config': {
                'max_seq_len': 2000, 'd_model': 48, 'n_layers': 4, 'd_state': 12,
                'micro_dim': 12, 'micro_proj': 16, 'n_prototypes': 4,
                'folds': args.folds, 'finetune_epochs': args.finetune_epochs,
                'batch_size': args.finetune_batch_size, 'lr': 5e-4,
                'device': str(device),
            },
            'steps_implemented': [1, 2, 3],
            'cv_results': {
                'accuracy_mean': summary['accuracy_mean'],
                'accuracy_std': summary['accuracy_std'],
                'precision_mean': summary['precision_mean'],
                'precision_std': summary['precision_std'],
                'recall_mean': summary['recall_mean'],
                'recall_std': summary['recall_std'],
                'f1_mean': summary['f1_mean'],
                'f1_std': summary['f1_std'],
                'auc_mean': summary['auc_mean'],
                'auc_std': summary['auc_std'],
            },
            'fold_details': all_fold_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  保存: {output_dir}/probs.npy, results.json")

    # 步骤 4: 提取可解释性 (取最后一个 fold 的模型)
    print("\n[Step 4] 提取可解释性信息...")
    try:
        interp = get_interpretability(model, samples, device)
        np.save(os.path.join(output_dir, 'event_importance.npy'), interp['event_importance'])
        np.save(os.path.join(output_dir, 'temporal_importance.npy'), interp['temporal_importance'])
        np.save(os.path.join(output_dir, 'proto_weights.npy'), interp['proto_weights'])
        np.save(os.path.join(output_dir, 'micro_repr.npy'), interp['micro_repr'])
        # 保存 event name
        with open(os.path.join(output_dir, 'event_names.json'), 'w', encoding='utf-8') as f:
            json.dump(EVENT_TYPES, f)
        print(f"  保存: event_importance.npy, temporal_importance.npy, proto_weights.npy, micro_repr.npy")
    except Exception as e:
        print(f"  可解释性提取失败 (非致命): {e}")

    print("\n" + "=" * 60)
    print(f"  Mamba-Long 训练完成")
    print(f"  Accuracy:  {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    print(f"  Precision: {summary['precision_mean']:.4f} ± {summary['precision_std']:.4f}")
    print(f"  Recall:    {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}")
    print(f"  F1 Score:  {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    print(f"  AUC:       {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()