# model_new — Self-Contained Model Directory

`model_new/` is a clean, self-contained re-organisation of the five canonical models from `models/`. Every model directory is independent — **each model can be trained and evaluated by `cd`-ing into its folder and running `python train.py`**, with no dependency on `models/` or the rest of the codebase beyond the bundled `model_new/common/` utility layer.

## Why `model_new/`?

The original `models/` directory mixes the five core models with ten-plus ablation/experimental models (`bgm_net`, `cream`, `cw_hdm_net`, `mamba`, `mre`, `csem_net`, ...). `model_new/` extracts the five canonical models used in the paper (RF, LSTM, BiLSTM, Transformer, HDM-Net v2) into a single self-contained tree, simplifying both research reproduction and external deployment.

## Layout

```
model_new/
├── README.md                       # (this file)
├── run_all.sh                      # run all 5 models sequentially
│
├── common/                         # bundled shared utilities
│   ├── __init__.py
│   ├── data_loader.py              # load_ide_logs(), set_seed(), get_device()
│   ├── evaluator.py                # evaluate(), summarize_fold_results(), print_results_table()
│   └── feature_engineering.py      # build_feature_matrix()
│
├── rf/                             # 1. Random Forest (sklearn)
│   ├── __init__.py
│   ├── README.md
│   ├── model.py                    # create_model()
│   └── train.py                    # standalone entry: python train.py
│
├── lstm/                           # 2. LSTM
│   ├── __init__.py
│   ├── README.md
│   ├── model.py                    # LSTMClassifier, create_model()
│   └── train.py                    # standalone entry: python train.py
│
├── bilstm/                         # 3. BiLSTM
│   ├── __init__.py
│   ├── README.md
│   ├── model.py                    # BiLSTMClassifier, create_model()
│   └── train.py                    # standalone entry: python train.py
│
├── transformer/                    # 4. Transformer
│   ├── __init__.py
│   ├── README.md
│   ├── model.py                    # TransformerClassifier, create_model()
│   └── train.py                    # standalone entry: python train.py
│
└── hdm_net/                        # 5. HDM-Net v2 (4-branch XCA + PIG)
    ├── __init__.py
    ├── README.md
    ├── model.py                    # TreeHead, SeqBranch, AttnBranch, XCA, PIG, HDMNet, count_parameters()
    └── train.py                    # standalone entry: python train.py
```

## Run any single model

```bash
# from /home/ubuntu/CodeEMO
cd model_new/rf
python train.py                  # 5-fold CV with default hyperparameters
python train.py --folds 10       # 10-fold CV
python train.py --output-dir /tmp/my_rf_run
```

The same pattern works for `lstm/`, `bilstm/`, `transformer/`, and `hdm_net/`.

Each `train.py` automatically:
- sets `sys.path[0]` to its own parent (`model_new/`),
- imports `common.data_loader`, `common.feature_engineering`, `common.evaluator` from the bundled local copy,
- imports its own model from the sibling `model.py`,
- writes results to `<output-dir>/results.json` and prints a 5-fold summary table.

## Run all five models

```bash
cd /home/ubuntu/CodeEMO
bash model_new/run_all.sh
```

`run_all.sh` iterates over the five model directories, executes `python train.py`, and writes each model's results under `model_new/outputs/<model>/results.json`.

## Data assumption

Each `train.py` loads IDE logs from the **default CS1 dataset path**:

```
/tmp/IDE_logs/IDE_logs.csv
/tmp/IDE_logs/passed.csv
```

Override via the train script's `--data-dir` argument (where supported) or via environment variables if your dataset lives elsewhere.

## Mapping to `models/`

| `model_new/` | `models/` | Notes |
|---|---|---|
| `rf/` | `models/rf/` | Drop-in equivalent |
| `lstm/` | `models/lstm/` | Uses `models/lstm/model.py` only (excludes `model_attn.py` and `train_46d.py`) |
| `bilstm/` | `models/bilstm/` | Drop-in equivalent |
| `transformer/` | `models/transformer/` | Drop-in equivalent |
| `hdm_net/` | `models/hdm_net/` | 4-branch XCA + PIG architecture (33,220 params) |

The original `models/` directory is **left untouched** for backward compatibility. To run the legacy `models/<name>/train.py`, `cd models/<name> && python train.py` still works.

## Reproducibility

- Random seed: 42 (PyTorch + NumPy + Python `random`).
- Cross-validation: 5-fold `StratifiedKFold`, `random_state = 42`.
- Hardware: any CUDA-capable GPU (RTX 4090 recommended); CPU fallback automatic.

## What changed relative to `models/`

The only mechanical changes between `models/<name>/train.py` and `model_new/<name>/train.py` are:

1. **`_PROJECT_ROOT`** is reset from `../../` (CodeEMO root) to `../` (`model_new/`). This is the only directory added to `sys.path`.
2. **`from models.<name>.model import …`** becomes **`from model import …`** (sibling-file import).
3. The bundled `common/` utilities are loaded via the same `from common.X import …` syntax — they are resolved against `model_new/`, **not** against the original `CodeEMO/common/`. This makes each model runnable without any external dependency on the parent repository.

No other code (model architecture, training loop, evaluation logic) was modified.

## License

Internal research code. See repository root for license details.
