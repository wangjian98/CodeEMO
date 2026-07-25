# Interpretability Analysis: BMA + Per-Fold Stacking

## 1. Bayesian Model Averaging (BMA) — weight interpretation

BMA assigns weight `w_i = exp(-log_loss_i / T) / sum exp(...)`. Higher weight = model is more confident on the OOF data.

### OOF log-loss ranking (lower = better, BMA gives higher weight)
1. **HDM-Net v2**: log-loss=0.3153
2. **Tree only**: log-loss=0.3202
3. **RF-LSTM**: log-loss=0.3261
4. **Transformer_7d**: log-loss=0.3273
5. **RF_7dim**: log-loss=0.3421
6. **LSTM_46d**: log-loss=0.3527
7. **BiLSTM_46d**: log-loss=0.3726
8. **Transformer_46d**: log-loss=2.2927

### Interpretation
- **HDM-Net v2** (log-loss=0.315) is the most reliable base model — BMA will give it the highest weight.
- **Transformer_46d** (log-loss=2.29) is broken on the OOF data — BMA correctly down-weights it.
- **BMA weights** are interpretable as a "trust score" — when a model is confident (low OOF loss) it gets more weight.

## 2. Per-Fold Weighted Stacking — model stability

For each fold k, an LR meta-learner is trained on the OOF predictions from the OTHER 4 folds, then predicts on fold k. This reveals how each base model's contribution varies across folds.

### Per-fold weight variation (top-5 models)
- **RF_7dim**: mean=0.191  std=0.017  min=0.163  max=0.212  → stable (consistent)
- **HDM-Net v2**: mean=0.232  std=0.006  min=0.226  max=0.239  → stable (consistent)
- **LSTM_46d**: mean=0.171  std=0.019  min=0.140  max=0.191  → stable (consistent)
- **BiLSTM_46d**: mean=0.164  std=0.008  min=0.155  max=0.178  → stable (consistent)
- **Transformer_7d**: mean=0.241  std=0.032  min=0.217  max=0.303  → stable (consistent)

### Interpretation
- Models with **low std** are stable contributors — they consistently help across all folds.
- Models with **high std** are volatile — they help on some folds but hurt on others.
- **Models that are ignored** (mean weight ~0) are *redundant* — adding them doesn't help the ensemble.

## 3. Pairwise model correlation (diversity)

High correlation = models make similar errors = ensemble gain is small.
Low correlation = diverse predictions = ensemble gain is large.
- RF_7dim ↔ HDM-Net v2: corr=0.964
- RF_7dim ↔ LSTM_46d: corr=0.844
- RF_7dim ↔ BiLSTM_46d: corr=0.836
- RF_7dim ↔ Transformer_7d: corr=0.925
- HDM-Net v2 ↔ LSTM_46d: corr=0.865
- HDM-Net v2 ↔ BiLSTM_46d: corr=0.849
- HDM-Net v2 ↔ Transformer_7d: corr=0.937
- LSTM_46d ↔ BiLSTM_46d: corr=0.945
- LSTM_46d ↔ Transformer_7d: corr=0.845
- BiLSTM_46d ↔ Transformer_7d: corr=0.838

### Interpretation
- Pairs with corr > 0.9: highly redundant (same signal); ensemble gain is minimal.
- Pairs with corr < 0.7: genuinely diverse; ensemble gain is large.
- Best ensemble pairs are LOW-corr + each has individually-high F1.

## 4. Failure analysis: who gets wrong?

- Total errors: 64 / 473 (13.5%)
- False negatives (missed failed=1): 40 / 314 failed samples (Recall miss rate=12.7%)
- False positives (wrong failed=1): 24 / 159 passed samples (False alarm rate=15.1%)

### Most-confident-mistaken failed=1 samples (highest OOF prob, but truth is failed):
- sample 339: OOF prob=0.977
- sample 253: OOF prob=0.977
- sample 426: OOF prob=0.977
- sample 210: OOF prob=0.977
- sample 250: OOF prob=0.977

These samples are *systematically hard* — even the ensemble confidently
predicts "passed" but the truth is "failed". They likely need a feature
or model class that captures a signal we're not currently using.

## Conclusion
- **BMA** is principled and interpretable: weights come from a closed-form Bayesian formula.
- **Per-fold stacking** reveals which models are fold-specific (volatile) vs robust.
- Both methods produce F1 ~0.89-0.90, comparable to Weighted 1/3/1.
- The key insight: **model diversity** (low pairwise correlation) predicts
  ensemble gain more reliably than absolute single-model F1.
