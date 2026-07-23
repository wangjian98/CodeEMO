#!/bin/bash
# 跑全部 6 个模型×特征组合 (后台)

set -e
cd /home/ubuntu/CodeEMO
source .venv/bin/activate

LOG_DIR=/home/ubuntu/CodeEMO/outputs/unified_logs
mkdir -p $LOG_DIR

START_TIME=$(date +%s)
echo "=== 统一对比训练开始: $(date) ==="

# 6 组合顺序: 先跑快的 (46d), 再跑慢的 (7d/Mamba)
echo ""
echo "===== 1/6: LSTM × 46d (快, ~3分钟) ====="
python models/lstm/train_46d.py --output-dir outputs/unified_compare/lstm_46d 2>&1 | tee $LOG_DIR/lstm_46d.log

echo ""
echo "===== 2/6: BiLSTM × 46d (B46, ~3分钟) ====="
python models/bilstm_save_probs.py --output-dir outputs/unified_compare/bilstm_46d 2>&1 | tee $LOG_DIR/bilstm_46d.log

echo ""
echo "===== 3/6: Mamba × 46d (新, ~5分钟) ====="
python models/mamba/train_46d.py --output-dir outputs/unified_compare/mamba_46d 2>&1 | tee $LOG_DIR/mamba_46d.log

echo ""
echo "===== 4/6: LSTM × 7d (max=500, ~5分钟) ====="
python models/lstm_7dim_gpu.py --output-dir outputs/unified_compare/lstm_7dim --max-seq-len 500 2>&1 | tee $LOG_DIR/lstm_7dim.log

echo ""
echo "===== 5/6: BiLSTM × 7d (max=500, ~5分钟) ====="
python models/bilstm_7dim_gpu.py --output-dir outputs/unified_compare/bilstm_7dim --max-seq-len 500 2>&1 | tee $LOG_DIR/bilstm_7dim.log

echo ""
echo "===== 6/6: Mamba × 7d (max=500, Mamba_MID, ~10分钟) ====="
python models/mamba/train_ms.py \
    --max-seq-len 500 \
    --output-dir outputs/unified_compare/mamba_7dim \
    --pretrain-epochs 2 \
    --finetune-epochs 4 \
    --pretrain-batch-size 16 \
    --finetune-batch-size 16 \
    2>&1 | tee $LOG_DIR/mamba_7dim.log

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=== 全部训练完成: $(date), 耗时 ${ELAPSED}s ==="

# 统一对比 + 可视化
echo ""
echo "===== 统一对比 + 可视化 ====="
python compare_all_unified.py 2>&1 | tee $LOG_DIR/compare.log