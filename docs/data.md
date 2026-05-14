# Data Notes

This project uses the official Kaggle files for Playground Series - Season 6 Episode 5.

## Files

| File | Rows | Columns | Purpose |
|---|---:|---:|---|
| `train.csv` | 439,140 | 16 | Labeled training data |
| `test.csv` | 188,165 | 15 | Unlabeled test data |
| `sample_submission.csv` | 188,165 | 2 | Required submission shape |

## Schema

Training columns:

```text
id, Driver, Compound, LapNumber, Stint, TyreLife, Position, LapTime (s),
LapTime_Delta, Race, Year, PitStop, Cumulative_Degradation, RaceProgress,
Position_Change, PitNextLap
```

Test columns are the same except `PitNextLap` is absent. Submission columns are:

```text
id, PitNextLap
```

## Target Balance

- Overall `PitNextLap` target rate: 19.90%.
- The 2023 rows have an unusually low target rate of 0.9607%.
- The 2022, 2024, and 2025 target rates are much higher, around 26-30%.

The 2023 anomaly is important for validation. The training code therefore reports both a grouped mixed validation split and a leave-one-year-out audit.

## Artifact Policy

Raw Kaggle files, generated tables, model files, and submission CSVs are generated locally and ignored by git.
