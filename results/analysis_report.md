# CodeEMO Final - Experiment Analysis Report

## Overview

- Dataset: CS1 (473 students)
- Features: 7 basic event-count features
- Validation: 5-fold Stratified Cross-Validation
- Models: Random Forest, LSTM, BiLSTM, Transformer, Mamba

## Results Summary

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|-----|
| Random Forest | 0.8435 +/- 0.031 | 0.7462 +/- 0.030 | 0.8109 +/- 0.106 | 0.7741 +/- 0.054 | 0.9151 +/- 0.012 |
| LSTM | 0.8711 +/- 0.023 | 0.7530 +/- 0.042 | 0.9246 +/- 0.032 | 0.8290 +/- 0.027 | 0.9246 +/- 0.018 |
| BiLSTM | 0.8796 +/- 0.023 | 0.7863 +/- 0.062 | 0.8992 +/- 0.054 | 0.8352 +/- 0.020 | 0.9259 +/- 0.020 |
| Transformer | 0.8648 +/- 0.025 | 0.7367 +/- 0.050 | 0.9433 +/- 0.054 | 0.8250 +/- 0.028 | 0.9221 +/- 0.015 |

## Key Findings

1. **Best Model**: BiLSTM (F1=0.8352, AUC=0.9259)
2. **Feature Design**: Simple 7-feature event counts outperform complex engineered features
3. **Neural Networks**: BiLSTM's bidirectional modeling captures richer patterns than unidirectional LSTM
4. **Mamba**: State-space sequence model with 6-step pipeline (pretrain + multi-scale + prototype)

## Visualizations

- Per-model results: `results/{model}/` (confusion matrix, ROC curve, training curves)
- Mamba results: `results/mamba/` (prototype analysis, event importance, training curves)
- Comparison plot: `results/comparison/model_comparison.png`
