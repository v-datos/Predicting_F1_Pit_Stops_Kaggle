<p align="center">
  <img src="path/to/your_image.png" alt="Predicting F1 Pit Stops Header" width="800">
</p>

# Predicting F1 Pit Stops

Public Kaggle project for [Playground Series - Season 6 Episode 5](https://www.kaggle.com/competitions/playground-series-s6e5). The task is to predict the probability that a Formula 1 driver pits on the next lap.

## Competition

- Target: `PitNextLap`
- Submission columns: `id`, `PitNextLap`
- Metric: ROC AUC
- Best public score from this repository: `0.94542`
- Best model: `lightgbm_f1_features_no_pitstop_tuned_leaves_v1`

## Setup

```bash
git clone https://github.com/v-datos/Predicting_F1_Pit_Stops_Kaggle.git
cd Predicting_F1_Pit_Stops_Kaggle
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Data

Download the official Kaggle files into `data/raw/`:

```bash
kaggle competitions download -c playground-series-s6e5 -p data/raw
unzip data/raw/playground-series-s6e5.zip -d data/raw
```

Expected files:

- `data/raw/train.csv`
- `data/raw/test.csv`
- `data/raw/sample_submission.csv`

The data files are not included in this repository. Generated reports and submissions are also ignored by git.

## Train And Validate

Run all local validation experiments and generate a submission from the best local validation model:

```bash
python -m predicting_f1_pit_stops.train
```

This writes:

- `reports/tables/baseline_metrics.csv`
- `reports/tables/year_shift_audit.csv`
- `submissions/submission.csv`

The validation split is deterministic and keeps each `Year`/`Race` group entirely in train or validation.

## Kaggle Kernel Runner

The script in `kaggle/catboost_gpu_kernel/` runs CatBoost and LightGBM variants on Kaggle compute. Before pushing the kernel, edit `kaggle/catboost_gpu_kernel/kernel-metadata.json` and replace the placeholder kernel id with your Kaggle username.

```bash
kaggle kernels push -p kaggle/catboost_gpu_kernel
```

## Results

| Model | Aligned Validation ROC AUC | Public Kaggle ROC AUC |
|---|---:|---:|
| `lightgbm_f1_features_no_pitstop_tuned_leaves_v1` | 0.941948 | 0.94542 |
| `lightgbm_f1_features_no_pitstop_tuned_regularized_v1` | 0.940689 | 0.94398 |
| `lightgbm_f1_features_no_pitstop_tuned_slow_v1` | 0.941017 | 0.94392 |
| `lightgbm_f1_features_with_pitstop_v1` | 0.940624 | not submitted |
| `catboost_f1_features_with_pitstop_v1` | 0.935226 | not submitted |

See [docs/modeling.md](docs/modeling.md) for validation and feature details.

## Checks

```bash
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/predicting_f1_pit_stops/ --ignore-missing-imports
python -m pytest tests/ -q
make check
```
