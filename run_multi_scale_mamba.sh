#!/bin/bash
# 多尺度 Mamba 后台训练脚本
# 依次跑 SHT(50) → MID(500) → LONG(2000), 完成后自动跑 fusion

set -e
cd /home/ubuntu/CodeEMO
source .venv/bin/activate

LOG_DIR=/home/ubuntu/CodeEMO/outputs/multi_scale_logs
mkdir -p $LOG_DIR

START_TIME=$(date +%s)
echo "=== 多尺度 Mamba 训练开始: $(date) ==="

# 训练 SHT (max=50) - 短序列, 微调 epochs 减半
echo ""
echo "===== 1/3: Mamba_SHT (max_seq=50) ====="
python models/mamba/train_ms.py \
    --max-seq-len 50 \
    --output-dir outputs/mamba_sht \
    --pretrain-epochs 2 \
    --finetune-epochs 6 \
    --pretrain-batch-size 16 \
    --finetune-batch-size 16 \
    2>&1 | tee $LOG_DIR/mamba_sht.log

# 训练 MID (max=500)
echo ""
echo "===== 2/3: Mamba_MID (max_seq=500) ====="
python models/mamba/train_ms.py \
    --max-seq-len 500 \
    --output-dir outputs/mamba_mid \
    --pretrain-epochs 2 \
    --finetune-epochs 6 \
    --pretrain-batch-size 8 \
    --finetune-batch-size 8 \
    2>&1 | tee $LOG_DIR/mamba_mid.log

# 训练 LONG (max=2000) - 长序列
echo ""
echo "===== 3/3: Mamba_LONG (max_seq=2000) ====="
python models/mamba/train_ms.py \
    --max-seq-len 2000 \
    --output-dir outputs/mamba_long \
    --pretrain-epochs 3 \
    --finetune-epochs 8 \
    --pretrain-batch-size 8 \
    --finetune-batch-size 8 \
    2>&1 | tee $LOG_DIR/mamba_long.log

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=== 训练完成: $(date), 耗时 ${ELAPSED}s ==="

# 跑 fusion 分析
echo ""
echo "===== Late Fusion + 对比分析 ====="
python multi_scale_mamba_fusion.py 2>&1 | tee $LOG_DIR/fusion.log