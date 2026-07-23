#!/bin/bash
# Mamba-Long 5 步骤实施路径

set -e
cd /home/ubuntu/CodeEMO
source .venv/bin/activate

LOG_DIR=/home/ubuntu/CodeEMO/outputs/mamba_long/logs
mkdir -p $LOG_DIR

START=$(date +%s)
echo "=== Mamba-Long 实施路径开始: $(date) ==="

# 步骤 1+2+3: 训练 (max=2000, micro, 改进多尺度)
echo ""
echo "===== 步骤 1+2+3: Mamba-Long 训练 ====="
python models/mamba/mamba_long/train.py 2>&1 | tee $LOG_DIR/train.log

# 步骤 4: 可解释性可视化
echo ""
echo "===== 步骤 4: 可解释性可视化 ====="
python models/mamba/mamba_long/interpret.py 2>&1 | tee $LOG_DIR/interpret.log

# 步骤 5: Late Fusion
echo ""
echo "===== 步骤 5: Late Fusion (加入 7 模型) ====="
python models/mamba/mamba_long/fusion.py 2>&1 | tee $LOG_DIR/fusion.log

END=$(date +%s)
ELAPSED=$((END - START))
echo ""
echo "=== Mamba-Long 5 步骤全部完成: $(date), 耗时 ${ELAPSED}s ==="