#!/usr/bin/env bash
# model_new/run_all.sh
# Sequentially trains all 5 canonical models in model_new/ on the CS1 dataset.
# Each model's outputs are written to model_new/outputs/<model>/.
#
# Usage:
#   bash model_new/run_all.sh           # 5-fold CV with defaults
#   bash model_new/run_all.sh --folds 10
#
set -e

# Resolve repo root (parent of this script's parent)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODEL_NEW_ROOT="$SCRIPT_DIR"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Optional flags forwarded to each train.py
EXTRA_ARGS=("$@")

MODELS=(rf lstm bilstm transformer hdm_net)

echo "================================================================"
echo " model_new — run_all.sh"
echo " Repo root   : $REPO_ROOT"
echo " model_new/  : $MODEL_NEW_ROOT"
echo " Models      : ${MODELS[*]}"
echo " Extra args  : ${EXTRA_ARGS[*]:-<none>}"
echo "================================================================"

for m in "${MODELS[@]}"; do
    echo
    echo "================================================================"
    echo " >> $m"
    echo "================================================================"
    cd "$MODEL_NEW_ROOT/$m"
    python3 train.py "${EXTRA_ARGS[@]}"
done

echo
echo "================================================================"
echo " All models finished."
echo " Outputs : $MODEL_NEW_ROOT/outputs/<model>/results.json"
echo "================================================================"
