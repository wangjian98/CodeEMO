#!/usr/bin/env python3
"""
CodeEMO 统一运行入口 v2

支持 model × features 组合:
  --model    {lstm, bilstm, mamba, all}
  --features {7dim, 46dim, all}
  --max-seq-len  (仅 7dim 有效, 默认 500)

用法:
  python main.py --model bilstm --features 46dim
  python main.py --model all --features all
  python main.py --model lstm --features 7dim --max-seq-len 2000
"""
import os
import sys
import json
import argparse
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()

# 模型 × 特征 → 训练脚本映射表
MODEL_FEATURE_MAP = {
    ('lstm', '7dim'):    ('models/lstm/train.py',         ['--folds', '{folds}', '--output-dir', '{outdir}', '--max-seq-len', '{max_seq}']),
    ('lstm', '46dim'):   ('models/lstm/train_46d.py',     ['--folds', '{folds}', '--output-dir', '{outdir}']),
    ('bilstm', '7dim'):  ('models/bilstm/train.py',       ['--folds', '{folds}', '--output-dir', '{outdir}', '--max-seq-len', '{max_seq}']),
    ('bilstm', '46dim'): ('models/bilstm_save_probs.py',  ['--folds', '{folds}', '--output-dir', '{outdir}']),
    ('mamba', '7dim'):   ('models/mamba/train_ms.py',     ['--folds', '{folds}', '--output-dir', '{outdir}', '--max-seq-len', '{max_seq}',
                                                           '--pretrain-epochs', '2', '--finetune-epochs', '4']),
    ('mamba', '46dim'):  ('models/mamba/train_46d.py',    ['--folds', '{folds}', '--output-dir', '{outdir}']),
    # RF / Transformer 用统一训练入口，输出到 outputs/unified_compare/
    ('rf', '7dim'):      ('models/rf/train_unified.py',   ['--features', '7d', '--folds', '{folds}', '--output-dir', '{outdir}']),
    ('rf', '46dim'):     ('models/rf/train_unified.py',   ['--features', '46d', '--folds', '{folds}', '--output-dir', '{outdir}']),
    ('transformer', '7dim'):  ('models/transformer/train_unified.py', ['--features', '7d', '--folds', '{folds}', '--output-dir', '{outdir}', '--device', 'cpu']),
    ('transformer', '46dim'): ('models/transformer/train_unified.py', ['--features', '46d', '--folds', '{folds}', '--output-dir', '{outdir}', '--device', 'cpu']),
}


def run_model(model, features, folds, max_seq, output_base):
    """运行单个模型×特征组合"""
    key = (model, features)
    if key not in MODEL_FEATURE_MAP:
        print(f"  ⚠ 不支持的组合: {model} × {features}")
        return None

    script_rel, args_template = MODEL_FEATURE_MAP[key]
    script = PROJECT_ROOT / script_rel
    outdir = PROJECT_ROOT / output_base / f"{model}_{features}"

    args = [str(arg).format(folds=folds, max_seq=max_seq, outdir=str(outdir))
            for arg in args_template]
    cmd = [sys.executable, str(script)] + args

    print(f"\n{'#'*60}")
    print(f"# 模型: {model} × 特征: {features}")
    print(f"# 命令: {' '.join(cmd)}")
    print(f"{'#'*60}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - start

    rf = outdir / 'results.json'
    if rf.exists():
        with open(rf) as f:
            data = json.load(f)
        data['_elapsed_seconds'] = elapsed
        return data
    else:
        print(f"  ⚠ 结果文件未找到: {rf}")
        return None


def main():
    parser = argparse.ArgumentParser(description='CodeEMO 统一运行 v2')
    parser.add_argument('--model', type=str, default='all',
                        choices=['lstm', 'bilstm', 'mamba', 'all'])
    parser.add_argument('--features', type=str, default='all',
                        choices=['7dim', '46dim', 'all'])
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--max-seq-len', type=int, default=500,
                        help='7dim 特征的最大序列长度 (默认 500)')
    parser.add_argument('--output-base', type=str, default='outputs/unified_compare',
                        help='输出根目录')
    args = parser.parse_args()

    model_list = ['lstm', 'bilstm', 'mamba', 'rf', 'transformer'] if args.model == 'all' else [args.model]
    feat_list = ['7dim', '46dim'] if args.features == 'all' else [args.features]

    print("=" * 60)
    print(f"  CodeEMO 统一运行 v2 - model × features 网格")
    print("=" * 60)
    print(f"  模型: {model_list}")
    print(f"  特征: {feat_list}")
    print(f"  折数: {args.folds}")
    print(f"  max-seq-len (7dim): {args.max_seq_len}")
    print(f"  输出: {args.output_base}")

    os.makedirs(args.output_base, exist_ok=True)

    all_results = {}
    combos = [(m, f) for m in model_list for f in feat_list]
    total_start = time.time()

    for i, (m, f) in enumerate(combos, 1):
        print(f"\n>>> [{i}/{len(combos)}] {m} × {f}")
        result = run_model(m, f, args.folds, args.max_seq_len, args.output_base)
        all_results[f"{m}_{f}"] = result

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  全部完成, 总耗时 {total_elapsed/60:.1f} 分钟")
    print(f"{'='*60}")

    # 汇总
    summary_path = PROJECT_ROOT / args.output_base / 'unified_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n汇总: {summary_path}")

    # 简短对比表
    print(f"\n{'组合':<22} {'F1':<14} {'AUC':<14} {'Accuracy':<14}")
    print("-" * 70)
    for name, r in all_results.items():
        if r is None:
            continue
        cv = r.get('cv_results', r.get('cv_metrics', {}))
        if 'f1_mean' in cv:
            f1 = cv['f1_mean']
            auc = cv['auc_mean']
            acc = cv['accuracy_mean']
        else:
            f1 = cv.get('f1', {}).get('mean', 0)
            auc = cv.get('auc', {}).get('mean', 0)
            acc = cv.get('accuracy', {}).get('mean', 0)
        print(f"{name:<22} {f1:.4f}        {auc:.4f}        {acc:.4f}")


if __name__ == '__main__':
    main()