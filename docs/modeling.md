# Modeling Notes

## Validation

The main validation split is `deterministic_year_race_row_balanced_v1`.

It keeps each `Year`/`Race` group entirely in train or validation and balances validation rows across five deterministic folds. The selected fold is the one closest to the overall target rate and 20% validation size.

Aligned split diagnostics:

- Train rows: 351,113
- Validation rows: 88,027
- Train `Year`/`Race` groups: 85
- Validation `Year`/`Race` groups: 19
- Train target rate: 0.201770
- Validation target rate: 0.187863

## Feature Policy

The strongest submitted models use baseline race/lap fields plus F1-specific engineered features:

- Tire-life and race-progress bins
- Compound/stint interactions
- Tire-life, stint, position, and degradation interactions
- Absolute lap-time delta

`PitStop` is treated cautiously. It is present in train and test, but current-lap pit status can become a shortcut for next-lap pit prediction. The best public submission excludes `PitStop`.

## Model Results

| Experiment | Model Family | Features | Aligned Validation ROC AUC | Public Kaggle ROC AUC |
|---|---|---|---:|---:|
| `lightgbm_f1_features_no_pitstop_tuned_leaves_v1` | LightGBM | F1 engineered, no `PitStop` | 0.941948 | 0.94542 |
| `lightgbm_f1_features_no_pitstop_tuned_regularized_v1` | LightGBM | F1 engineered, no `PitStop` | 0.940689 | 0.94398 |
| `lightgbm_f1_features_no_pitstop_tuned_slow_v1` | LightGBM | F1 engineered, no `PitStop` | 0.941017 | 0.94392 |
| `lightgbm_f1_features_with_pitstop_v1` | LightGBM | F1 engineered with `PitStop` | 0.940624 | not submitted |
| `lightgbm_f1_features_no_pitstop_v1` | LightGBM | F1 engineered, no `PitStop` | 0.939899 | not submitted |
| `catboost_f1_features_with_pitstop_v1` | CatBoost | F1 engineered with `PitStop` | 0.935226 | not submitted |
| `catboost_f1_features_no_pitstop_v1` | CatBoost | F1 engineered, no `PitStop` | 0.933633 | not submitted |

Current selected model: `lightgbm_f1_features_no_pitstop_tuned_leaves_v1`.

## Reproducibility

Run local validation and regenerate `submissions/submission.csv`:

```bash
python -m predicting_f1_pit_stops.train
```

Run the Kaggle boosted-tree script kernel:

```bash
kaggle kernels push -p kaggle/catboost_gpu_kernel
```
