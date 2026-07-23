#!/bin/bash
# 重跑 mamba_46d + 统一对比

set -e
cd /home/ubuntu/CodeEMO
source .venv/bin/activate

LOG_DIR=/home/ubuntu/CodeEMO/outputs/unified_logs
mkdir -p $LOG_DIR

echo "=== 重跑 Mamba-46d (16:50 起): $(date) ==="
python models/mamba/train_46d.py --output-dir outputs/unified_compare/mamba_46d 2>&1 | tee $LOG_DIR/mamba_46d_v2.log

echo ""
echo "=== Mamba-46d 完成, 开始统一对比 + 可视化: $(date) ==="
python compare_all_unified.py 2>&1 | tee $LOG_DIR/compare.log