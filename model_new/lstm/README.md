# LSTM — Standalone Runner

LSTM for the CS1 MOOC student-outcome prediction task.

## Run

```bash
cd model_new/lstm
python train.py                       # 5-fold CV with default hyperparameters
python train.py --folds 10            # 10-fold CV
python train.py --output-dir /tmp/run
```

## What it does

- Loads IDE logs from `/tmp/IDE_logs/` (or `--data-dir` if your dataset lives elsewhere).
- Builds per-student feature tensors for this model.
- Trains under 5-fold `StratifiedKFold` (random_state = 42).
- Evaluates Accuracy / Precision / Recall / F1 / AUC per fold.
- Writes `<output-dir>/results.json` and prints a summary table.

## Architecture

See `model.py` for the model definition. The exact class is imported as:

```python
from model import <model_class>
```

## Files

- `model.py`     — model class definition.
- `train.py`     — standalone entry; only depends on bundled `common/`.
- `README.md`    — (this file).

## Outputs

```
model_new/lstm/outputs/lstm/results.json
```

## Self-contained guarantees

`train.py` adds only `model_new/` to `sys.path`. The bundled `common/` provides `load_ide_logs`, `set_seed`, `get_device`, `build_feature_matrix`, `evaluate`, `summarize_fold_results`, `print_results_table`. No code from the legacy `models/` or `CodeEMO/common/` is referenced.
