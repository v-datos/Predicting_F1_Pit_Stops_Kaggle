"""Kaggle runner for boosted-tree pit-stop experiments.

This script is self-contained so it can be pushed as a Kaggle script kernel.
It reads the official competition files from Kaggle input, runs CatBoost and
LightGBM experiments, and writes:

- /kaggle/working/baseline_metrics.csv
- /kaggle/working/year_shift_audit.csv
- /kaggle/working/submission.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from catboost import CatBoostClassifier, CatBoostError
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET_COLUMN = "PitNextLap"
RANDOM_STATE = 42
N_SPLITS = 5
SPLIT_ID = "deterministic_year_race_row_balanced_v1"

CATEGORICAL_FEATURES = ["Driver", "Compound", "Race"]
NUMERIC_FEATURES = [
    "Year",
    "LapNumber",
    "Stint",
    "TyreLife",
    "Position",
    "LapTime (s)",
    "LapTime_Delta",
    "Cumulative_Degradation",
    "RaceProgress",
    "Position_Change",
]
PITSTOP_FEATURES = ["PitStop"]
ENGINEERED_CATEGORICAL_FEATURES = [
    "TyreLifeBin",
    "RaceProgressBin",
    "CompoundStint",
]
ENGINEERED_NUMERIC_FEATURES = [
    "TyreLifePerStint",
    "TyreLifeRaceProgress",
    "StintRaceProgress",
    "PositionRaceProgress",
    "DegradationPerTyreLife",
    "DegradationRaceProgress",
    "LapTimeDeltaAbs",
]


@dataclass(frozen=True)
class FeatureSet:
    name: str
    categorical_features: list[str]
    numeric_features: list[str]

    @property
    def feature_columns(self) -> list[str]:
        return self.categorical_features + self.numeric_features


@dataclass(frozen=True)
class Experiment:
    name: str
    feature_set: FeatureSet
    model_family: str
    model_params: dict[str, int | float] | None = None


@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    validation: pd.DataFrame
    validation_fold: int


CATBOOST_F1_FEATURE_SET = FeatureSet(
    name="catboost_f1_features_no_pitstop_v1",
    categorical_features=CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES,
    numeric_features=NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES,
)
CATBOOST_F1_PITSTOP_FEATURE_SET = FeatureSet(
    name="catboost_f1_features_with_pitstop_v1",
    categorical_features=CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES,
    numeric_features=NUMERIC_FEATURES + PITSTOP_FEATURES + ENGINEERED_NUMERIC_FEATURES,
)
CATBOOST_EXPERIMENTS = [
    Experiment(CATBOOST_F1_FEATURE_SET.name, CATBOOST_F1_FEATURE_SET, "catboost"),
    Experiment(CATBOOST_F1_PITSTOP_FEATURE_SET.name, CATBOOST_F1_PITSTOP_FEATURE_SET, "catboost"),
]
LIGHTGBM_F1_FEATURE_SET = FeatureSet(
    name="lightgbm_f1_features_no_pitstop_v1",
    categorical_features=CATBOOST_F1_FEATURE_SET.categorical_features,
    numeric_features=CATBOOST_F1_FEATURE_SET.numeric_features,
)
LIGHTGBM_F1_PITSTOP_FEATURE_SET = FeatureSet(
    name="lightgbm_f1_features_with_pitstop_v1",
    categorical_features=CATBOOST_F1_PITSTOP_FEATURE_SET.categorical_features,
    numeric_features=CATBOOST_F1_PITSTOP_FEATURE_SET.numeric_features,
)
LIGHTGBM_EXPERIMENTS = [
    Experiment(LIGHTGBM_F1_FEATURE_SET.name, LIGHTGBM_F1_FEATURE_SET, "lightgbm"),
    Experiment(
        LIGHTGBM_F1_PITSTOP_FEATURE_SET.name,
        LIGHTGBM_F1_PITSTOP_FEATURE_SET,
        "lightgbm",
    ),
    Experiment(
        "lightgbm_f1_features_no_pitstop_tuned_leaves_v1",
        LIGHTGBM_F1_FEATURE_SET,
        "lightgbm",
        {
            "num_leaves": 63,
            "min_child_samples": 40,
            "learning_rate": 0.04,
            "n_estimators": 450,
            "colsample_bytree": 0.80,
            "subsample": 0.80,
            "reg_alpha": 0.0,
            "reg_lambda": 0.5,
        },
    ),
    Experiment(
        "lightgbm_f1_features_no_pitstop_tuned_regularized_v1",
        LIGHTGBM_F1_FEATURE_SET,
        "lightgbm",
        {
            "num_leaves": 31,
            "min_child_samples": 80,
            "learning_rate": 0.05,
            "n_estimators": 400,
            "colsample_bytree": 0.75,
            "subsample": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        },
    ),
    Experiment(
        "lightgbm_f1_features_no_pitstop_tuned_slow_v1",
        LIGHTGBM_F1_FEATURE_SET,
        "lightgbm",
        {
            "num_leaves": 31,
            "min_child_samples": 40,
            "learning_rate": 0.03,
            "n_estimators": 700,
            "colsample_bytree": 0.80,
            "subsample": 0.80,
            "reg_alpha": 0.05,
            "reg_lambda": 0.5,
        },
    ),
]
EXPERIMENTS = CATBOOST_EXPERIMENTS + LIGHTGBM_EXPERIMENTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/kaggle/input/playground-series-s6e5"),
        help="Directory containing train.csv, test.csv, and sample_submission.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working"),
        help="Directory where metrics, audit, and submission files are written.",
    )
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument(
        "--task-type",
        choices=["GPU", "CPU", "auto"],
        default="auto",
        help="Use GPU, CPU, or try GPU then fall back to CPU.",
    )
    return parser.parse_args()


def resolve_input_dir(input_dir: Path) -> Path:
    """Find the Kaggle input directory containing the competition CSV files."""
    required = {"train.csv", "test.csv", "sample_submission.csv"}
    if all((input_dir / name).exists() for name in required):
        return input_dir

    input_root = Path("/kaggle/input")
    candidates = []
    if input_root.exists():
        for train_path in input_root.rglob("train.csv"):
            candidate = train_path.parent
            if all((candidate / name).exists() for name in required):
                candidates.append(candidate)

    if candidates:
        resolved = sorted(candidates, key=lambda path: str(path))[0]
        print(f"Resolved Kaggle input directory: {resolved}")
        return resolved

    available = []
    if input_root.exists():
        available = [str(path) for path in sorted(input_root.glob("*"))]
    msg = (
        f"Could not find required files {sorted(required)} under {input_dir} "
        f"or {input_root}. Available /kaggle/input entries: {available}"
    )
    raise FileNotFoundError(msg)


def add_f1_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    tyre_life = features["TyreLife"].clip(lower=1)
    race_progress = features["RaceProgress"].clip(lower=0)

    features["TyreLifeBin"] = pd.cut(
        features["TyreLife"],
        bins=[0, 5, 12, 20, 30, float("inf")],
        labels=["fresh", "early", "middle", "old", "very_old"],
        include_lowest=True,
    ).astype("object")
    features["RaceProgressBin"] = pd.cut(
        features["RaceProgress"],
        bins=[-0.001, 0.25, 0.5, 0.75, 1.0],
        labels=["race_q1", "race_q2", "race_q3", "race_q4"],
        include_lowest=True,
    ).astype("object")
    features["CompoundStint"] = (
        features["Compound"].astype(str) + "_S" + features["Stint"].astype(str)
    )
    features["TyreLifePerStint"] = tyre_life / features["Stint"].clip(lower=1)
    features["TyreLifeRaceProgress"] = tyre_life * race_progress
    features["StintRaceProgress"] = features["Stint"] * race_progress
    features["PositionRaceProgress"] = features["Position"] * race_progress
    features["DegradationPerTyreLife"] = features["Cumulative_Degradation"] / tyre_life
    features["DegradationRaceProgress"] = features["Cumulative_Degradation"] * race_progress
    features["LapTimeDeltaAbs"] = features["LapTime_Delta"].abs()
    return features


def year_race_groups(df: pd.DataFrame) -> pd.Series:
    return df["Year"].astype(str) + "::" + df["Race"].astype(str)


def stratified_year_race_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = select_aligned_year_race_split(df)
    return split.train, split.validation


def select_aligned_year_race_split(df: pd.DataFrame) -> SplitResult:
    folds = aligned_year_race_folds(df)
    y = df[TARGET_COLUMN].astype(int)
    best_train_idx = None
    best_val_idx = None
    best_fold_id = None
    best_score = float("inf")
    overall_rate = float(y.mean())
    target_val_fraction = 1.0 / N_SPLITS

    for fold_id, train_idx, val_idx in folds:
        val_y = y.iloc[val_idx]
        val_fraction = len(val_idx) / len(df)
        score = abs(float(val_y.mean()) - overall_rate) + abs(val_fraction - target_val_fraction)
        if score < best_score:
            best_score = score
            best_train_idx = train_idx
            best_val_idx = val_idx
            best_fold_id = fold_id

    if best_train_idx is None or best_val_idx is None or best_fold_id is None:
        raise RuntimeError("Unable to create an aligned Year/Race split")

    return SplitResult(
        df.iloc[best_train_idx].copy(),
        df.iloc[best_val_idx].copy(),
        best_fold_id,
    )


def aligned_year_race_folds(df: pd.DataFrame) -> list[tuple[int, list[int], list[int]]]:
    group_key = year_race_groups(df)
    group_stats = (
        pd.DataFrame({"group": group_key, "target": df[TARGET_COLUMN].astype(int)})
        .groupby("group", sort=True)
        .agg(rows=("target", "size"))
        .reset_index()
        .sort_values(["rows", "group"], ascending=[False, True], kind="mergesort")
    )
    fold_groups: list[list[str]] = [[] for _ in range(N_SPLITS)]
    fold_rows = [0 for _ in range(N_SPLITS)]

    for row in group_stats.itertuples(index=False):
        fold_id = min(range(N_SPLITS), key=lambda item: (fold_rows[item], item))
        fold_groups[fold_id].append(str(row.group))
        fold_rows[fold_id] += int(row.rows)

    folds = []
    positions = pd.Series(range(len(df)), index=df.index)
    for fold_id, groups_for_fold in enumerate(fold_groups):
        validation_mask = group_key.isin(groups_for_fold)
        validation_idx = positions.loc[validation_mask].tolist()
        train_idx = positions.loc[~validation_mask].tolist()
        folds.append((fold_id, train_idx, validation_idx))
    return folds


def build_catboost_model(
    task_type: Literal["GPU", "CPU"],
    iterations: int,
    learning_rate: float,
    depth: int,
) -> CatBoostClassifier:
    return CatBoostClassifier(
        auto_class_weights="Balanced",
        allow_writing_files=False,
        eval_metric="AUC",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        random_seed=RANDOM_STATE,
        task_type=task_type,
        verbose=False,
    )


def lightgbm_params(model_params: dict[str, int | float] | None = None) -> dict[str, object]:
    params: dict[str, object] = {
        "class_weight": "balanced",
        "colsample_bytree": 0.8,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "num_leaves": 31,
        "random_state": RANDOM_STATE,
        "subsample": 0.8,
        "verbosity": -1,
    }
    if model_params:
        params.update(model_params)
    return params


def lightgbm_model_name(model_params: dict[str, int | float] | None = None) -> str:
    params = lightgbm_params(model_params)
    return (
        "LGBMClassifier(class_weight='balanced', "
        f"n_estimators={params['n_estimators']}, "
        f"learning_rate={params['learning_rate']}, "
        f"num_leaves={params['num_leaves']}, "
        f"min_child_samples={params.get('min_child_samples', 'default')}, "
        f"feature_fraction={params['colsample_bytree']}, "
        f"bagging_fraction={params['subsample']}, "
        f"lambda_l1={params.get('reg_alpha', 0.0)}, "
        f"lambda_l2={params.get('reg_lambda', 0.0)})"
    )


def build_lightgbm_model(experiment: Experiment) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                experiment.feature_set.categorical_features,
            ),
            ("numeric", "passthrough", experiment.feature_set.numeric_features),
        ],
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LGBMClassifier(**lightgbm_params(experiment.model_params))),
        ],
    )


def fit_predict_proba(
    experiment: Experiment,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    task_type: Literal["GPU", "CPU"],
    iterations: int,
    learning_rate: float,
    depth: int,
) -> pd.Series:
    train_features = add_f1_features(train)
    validation_features = add_f1_features(validation)
    columns = experiment.feature_set.feature_columns

    if experiment.model_family == "lightgbm":
        model = build_lightgbm_model(experiment)
        model.fit(train_features[columns], train[TARGET_COLUMN])
        probabilities = model.predict_proba(validation_features[columns])[:, 1]
        return pd.Series(probabilities, index=validation.index)

    cat_features = [columns.index(column) for column in experiment.feature_set.categorical_features]
    model = build_catboost_model(task_type, iterations, learning_rate, depth)
    model.fit(train_features[columns], train[TARGET_COLUMN], cat_features=cat_features)
    probabilities = model.predict_proba(validation_features[columns])[:, 1]
    return pd.Series(probabilities, index=validation.index)


def evaluate_experiment(
    experiment: Experiment,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    task_type: Literal["GPU", "CPU"],
    iterations: int,
    learning_rate: float,
    depth: int,
) -> float:
    probabilities = fit_predict_proba(
        experiment,
        train,
        validation,
        task_type,
        iterations,
        learning_rate,
        depth,
    )
    return float(roc_auc_score(validation[TARGET_COLUMN], probabilities))


def evaluate_with_fallback(
    experiment: Experiment,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    requested_task_type: str,
    iterations: int,
    learning_rate: float,
    depth: int,
) -> tuple[float, Literal["GPU", "CPU"]]:
    if experiment.model_family == "lightgbm":
        auc = evaluate_experiment(
            experiment,
            train,
            validation,
            "CPU",
            iterations,
            learning_rate,
            depth,
        )
        return auc, "CPU"

    task_types: list[Literal["GPU", "CPU"]]
    if requested_task_type == "auto":
        task_types = ["GPU", "CPU"]
    elif requested_task_type == "GPU":
        task_types = ["GPU"]
    else:
        task_types = ["CPU"]

    last_error: Exception | None = None
    for task_type in task_types:
        try:
            auc = evaluate_experiment(
                experiment,
                train,
                validation,
                task_type,
                iterations,
                learning_rate,
                depth,
            )
            return auc, task_type
        except CatBoostError as exc:
            last_error = exc
            if task_type == "GPU" and requested_task_type == "auto":
                print(f"GPU CatBoost unavailable, falling back to CPU: {exc}")
                continue
            raise
    if last_error:
        raise last_error
    raise RuntimeError("No CatBoost task type was attempted")


def metric_row(
    experiment: Experiment,
    auc: float,
    task_type: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    validation_fold: int,
) -> dict[str, int | float | str]:
    if experiment.model_family == "lightgbm":
        model_name = lightgbm_model_name(experiment.model_params)
    else:
        model_name = (
            "CatBoostClassifier(auto_class_weights='Balanced', iterations=150, "
            f"learning_rate=0.08, depth=6, task_type='{task_type}')"
        )
    return {
        "experiment": experiment.name,
        "model": model_name,
        "model_family": experiment.model_family,
        "split": "deterministic Year/Race row-balanced fold, best of 5 folds",
        "split_id": SPLIT_ID,
        "validation_fold": validation_fold,
        "features": ",".join(experiment.feature_set.feature_columns),
        "metric": "roc_auc",
        "validation_roc_auc": auc,
        "task_type_used": task_type,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_target_rate": float(train[TARGET_COLUMN].mean()),
        "validation_target_rate": float(validation[TARGET_COLUMN].mean()),
        "train_year_race_groups": train.groupby(["Year", "Race"]).ngroups,
        "validation_year_race_groups": validation.groupby(["Year", "Race"]).ngroups,
    }


def year_shift_audit_rows(
    experiment: Experiment,
    train: pd.DataFrame,
    task_type: Literal["GPU", "CPU"],
    iterations: int,
    learning_rate: float,
    depth: int,
) -> list[dict[str, int | float | str]]:
    rows = []
    for holdout_year in sorted(train["Year"].unique()):
        split_train = train.loc[train["Year"] != holdout_year].copy()
        validation = train.loc[train["Year"] == holdout_year].copy()
        auc = evaluate_experiment(
            experiment,
            split_train,
            validation,
            task_type,
            iterations,
            learning_rate,
            depth,
        )
        rows.append(
            {
                "experiment": experiment.name,
                "holdout_year": int(holdout_year),
                "validation_roc_auc": auc,
                "target_rate": float(validation[TARGET_COLUMN].mean()),
                "train_rows": len(split_train),
                "validation_rows": len(validation),
                "task_type_used": task_type,
                "notes": (
                    "2023 has unusually low target rate; interpret this holdout separately."
                    if int(holdout_year) == 2023
                    else ""
                ),
            },
        )
    return rows


def build_submission(
    experiment: Experiment,
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    task_type: Literal["GPU", "CPU"],
    iterations: int,
    learning_rate: float,
    depth: int,
) -> pd.DataFrame:
    train_features = add_f1_features(train)
    test_features = add_f1_features(test)
    columns = experiment.feature_set.feature_columns

    if experiment.model_family == "lightgbm":
        model = build_lightgbm_model(experiment)
        model.fit(train_features[columns], train[TARGET_COLUMN])
        submission = sample_submission.copy()
        submission[TARGET_COLUMN] = model.predict_proba(test_features[columns])[:, 1]
        return submission

    cat_features = [columns.index(column) for column in experiment.feature_set.categorical_features]
    model = build_catboost_model(task_type, iterations, learning_rate, depth)
    model.fit(train_features[columns], train[TARGET_COLUMN], cat_features=cat_features)
    submission = sample_submission.copy()
    submission[TARGET_COLUMN] = model.predict_proba(test_features[columns])[:, 1]
    return submission


def main() -> None:
    args = parse_args()
    input_dir = resolve_input_dir(args.input_dir)
    train = pd.read_csv(input_dir / "train.csv")
    test = pd.read_csv(input_dir / "test.csv")
    sample_submission = pd.read_csv(input_dir / "sample_submission.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split = select_aligned_year_race_split(train)
    split_train = split.train
    validation = split.validation
    rows = []
    task_type_by_experiment: dict[str, Literal["GPU", "CPU"]] = {}
    for experiment in EXPERIMENTS:
        auc, task_type = evaluate_with_fallback(
            experiment,
            split_train,
            validation,
            args.task_type,
            args.iterations,
            args.learning_rate,
            args.depth,
        )
        task_type_by_experiment[experiment.name] = task_type
        rows.append(
            metric_row(experiment, auc, task_type, split_train, validation, split.validation_fold)
        )
        print(f"{experiment.name}: validation ROC AUC {auc:.6f} ({task_type})")

    metrics = pd.DataFrame(rows).sort_values("validation_roc_auc", ascending=False)
    metrics.to_csv(args.output_dir / "baseline_metrics.csv", index=False)
    best = metrics.iloc[0]
    best_experiment = next(
        experiment for experiment in EXPERIMENTS if experiment.name == best["experiment"]
    )
    best_catboost = metrics.loc[
        metrics["experiment"].isin([item.name for item in CATBOOST_EXPERIMENTS])
    ].iloc[0]
    best_catboost_experiment = next(
        experiment
        for experiment in CATBOOST_EXPERIMENTS
        if experiment.name == best_catboost["experiment"]
    )

    audit_rows = []
    for audit_experiment in [best_catboost_experiment, *LIGHTGBM_EXPERIMENTS]:
        audit_rows.extend(
            year_shift_audit_rows(
                audit_experiment,
                train,
                task_type_by_experiment[audit_experiment.name],
                args.iterations,
                args.learning_rate,
                args.depth,
            ),
        )
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(args.output_dir / "year_shift_audit.csv", index=False)

    submission = build_submission(
        best_experiment,
        train,
        test,
        sample_submission,
        task_type_by_experiment[best_experiment.name],
        args.iterations,
        args.learning_rate,
        args.depth,
    )
    submission.to_csv(args.output_dir / "submission.csv", index=False)

    print(f"Best experiment: {best['experiment']} ({best['validation_roc_auc']:.6f})")
    print(f"Validation split: {SPLIT_ID}, fold {split.validation_fold}")
    print(
        f"Best CatBoost experiment: {best_catboost['experiment']} "
        f"({best_catboost['validation_roc_auc']:.6f})"
    )
    print(f"Wrote {args.output_dir / 'baseline_metrics.csv'}")
    print(f"Wrote {args.output_dir / 'year_shift_audit.csv'}")
    print(f"Wrote {args.output_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()
