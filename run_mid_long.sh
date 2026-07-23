#!/bin/bash
# 重跑 MID + LONG, 然后 fusion (SHT 已有数据)

set -e
cd /home/ubuntu/CodeEMO
source .venv/bin/activate

LOG_DIR=/home/ubuntu/CodeEMO/outputs/multi_scale_logs
mkdir -p $LOG_DIR

START_TIME=$(date +%s)
echo "=== 重跑 MID + LONG (08:46 起): $(date) ==="

# MID (max=500)
echo ""
echo "===== 1/2: Mamba_MID (max_seq=500) - batch=16, drop_last=True ====="
python models/mamba/train_ms.py \
    --max-seq-len 500 \
    --output-dir outputs/mamba_mid \
    --pretrain-epochs 2 \
    --finetune-epochs 6 \
    --pretrain-batch-size 16 \
    --finetune-batch-size 16 \
    2>&1 | tee $LOG_DIR/mamba_mid.log

# LONG (max=2000)
echo ""
echo "===== 2/2: Mamba_LONG (max_seq=2000) - batch=16, drop_last=True ====="
python models/mamba/train_ms.py \
    --max-seq-len 2000 \
    --output-dir outputs/mamba_long \
    --pretrain-epochs 3 \
    --finetune-epochs 8 \
    --pretrain-batch-size 16 \
    --finetune-batch-size 16 \
    2>&1 | tee $LOG_DIR/mamba_long.log

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=== MID + LONG 训练完成: $(date), 耗时 ${ELAPSED}s ==="

# Fusion
echo ""
echo "===== Late Fusion + 对比分析 ====="
python multi_scale_mamba_fusion.py 2>&1 | tee $LOG_DIR/fusion.log