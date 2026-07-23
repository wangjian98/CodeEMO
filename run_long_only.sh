#!/bin/bash
# 仅跑 LONG (max=2000) - 降低 batch_size + 缩小模型, 防 OOM

set -e
cd /home/ubuntu/CodeEMO
source .venv/bin/activate

LOG_DIR=/home/ubuntu/CodeEMO/outputs/multi_scale_logs
mkdir -p $LOG_DIR

START_TIME=$(date +%s)
echo "=== 重跑 LONG (09:55 起): $(date) ==="
echo "=== 配置: max=2000, batch=4, d_model=48 n_layers=4 d_state=12 (降规模) ==="

python models/mamba/train_ms.py \
    --max-seq-len 2000 \
    --output-dir outputs/mamba_long \
    --pretrain-epochs 2 \
    --finetune-epochs 6 \
    --pretrain-batch-size 4 \
    --finetune-batch-size 4 \
    --d-model 48 \
    --n-layers 4 \
    --d-state 12 \
    2>&1 | tee $LOG_DIR/mamba_long.log

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=== LONG 训练完成: $(date), 耗时 ${ELAPSED}s ==="

# Fusion
echo ""
echo "===== Late Fusion + 对比分析 ====="
python multi_scale_mamba_fusion.py 2>&1 | tee $LOG_DIR/fusion.log