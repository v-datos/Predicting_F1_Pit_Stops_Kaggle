from datetime import UTC, datetime
from typing import NamedTuple

import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from predicting_f1_pit_stops.data import (
    load_sample_submission,
    load_test,
    load_train,
    validate_submission,
)
from predicting_f1_pit_stops.paths import DEFAULT_SUBMISSION_PATH, PROJECT_ROOT
from predicting_f1_pit_stops.schema import TARGET_COLUMN

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
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
EXCLUDED_FEATURES = ["PitStop"]
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
STRATEGY_CATEGORICAL_FEATURES = [
    "CompoundTyreLifeBin",
    "CompoundRaceProgressBin",
    "DegradationSlopeBin",
    "CompoundDegradationBin",
    "PitWindowBand",
    "CompoundPitWindowBand",
    "LapTimeDeltaSign",
    "CompoundLapTimeDeltaSign",
    "RaceYear",
    "RaceCompound",
]
STRATEGY_NUMERIC_FEATURES = [
    "TyreLifeLapRatio",
    "PositionChangeAbs",
    "PositionTyreLife",
]
PITSTOP_STRATEGY_CATEGORICAL_FEATURES = [
    "PitStopRaceProgressBin",
    "PitStopStint",
]
PITSTOP_STRATEGY_NUMERIC_FEATURES = [
    "PitStopRaceProgress",
]

RANDOM_STATE = 42
N_SPLITS = 5
SPLIT_ID = "deterministic_year_race_row_balanced_v1"
METRICS_PATH = PROJECT_ROOT / "reports" / "tables" / "baseline_metrics.csv"
YEAR_SHIFT_AUDIT_PATH = PROJECT_ROOT / "reports" / "tables" / "year_shift_audit.csv"


class FeatureSet(NamedTuple):
    name: str
    categorical_features: list[str]
    numeric_features: list[str]
    excluded_features: list[str]
    add_engineered_features: bool = False
    add_strategy_features: bool = False

    @property
    def feature_columns(self) -> list[str]:
        return self.categorical_features + self.numeric_features


class Experiment(NamedTuple):
    name: str
    feature_set: FeatureSet
    model_family: str
    model_params: dict[str, int | float] | None = None


class SplitResult(NamedTuple):
    train: pd.DataFrame
    validation: pd.DataFrame
    validation_fold: int


BASELINE_FEATURE_SET = FeatureSet(
    name="logistic_regression_no_pitstop_v1",
    categorical_features=CATEGORICAL_FEATURES,
    numeric_features=NUMERIC_FEATURES,
    excluded_features=["id", "PitStop", TARGET_COLUMN],
)
PITSTOP_FEATURE_SET = FeatureSet(
    name="logistic_regression_with_pitstop_v1",
    categorical_features=CATEGORICAL_FEATURES,
    numeric_features=NUMERIC_FEATURES + PITSTOP_FEATURES,
    excluded_features=["id", TARGET_COLUMN],
)
F1_FEATURE_SET = FeatureSet(
    name="logistic_regression_f1_features_no_pitstop_v1",
    categorical_features=CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES,
    numeric_features=NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES,
    excluded_features=["id", "PitStop", TARGET_COLUMN],
    add_engineered_features=True,
)
F1_PITSTOP_FEATURE_SET = FeatureSet(
    name="logistic_regression_f1_features_with_pitstop_v1",
    categorical_features=CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES,
    numeric_features=NUMERIC_FEATURES + PITSTOP_FEATURES + ENGINEERED_NUMERIC_FEATURES,
    excluded_features=["id", TARGET_COLUMN],
    add_engineered_features=True,
)
STRATEGY_FEATURE_SET = FeatureSet(
    name="logistic_regression_strategy_features_no_pitstop_v1",
    categorical_features=(
        CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES + STRATEGY_CATEGORICAL_FEATURES
    ),
    numeric_features=NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES + STRATEGY_NUMERIC_FEATURES,
    excluded_features=["id", "PitStop", TARGET_COLUMN],
    add_engineered_features=True,
    add_strategy_features=True,
)
STRATEGY_PITSTOP_FEATURE_SET = FeatureSet(
    name="logistic_regression_strategy_features_with_pitstop_v1",
    categorical_features=(
        CATEGORICAL_FEATURES
        + ENGINEERED_CATEGORICAL_FEATURES
        + STRATEGY_CATEGORICAL_FEATURES
        + PITSTOP_STRATEGY_CATEGORICAL_FEATURES
    ),
    numeric_features=(
        NUMERIC_FEATURES
        + PITSTOP_FEATURES
        + ENGINEERED_NUMERIC_FEATURES
        + STRATEGY_NUMERIC_FEATURES
        + PITSTOP_STRATEGY_NUMERIC_FEATURES
    ),
    excluded_features=["id", TARGET_COLUMN],
    add_engineered_features=True,
    add_strategy_features=True,
)
FEATURE_SETS = [
    BASELINE_FEATURE_SET,
    PITSTOP_FEATURE_SET,
    F1_FEATURE_SET,
    F1_PITSTOP_FEATURE_SET,
    STRATEGY_FEATURE_SET,
    STRATEGY_PITSTOP_FEATURE_SET,
]
CATBOOST_F1_FEATURE_SET = FeatureSet(
    name="catboost_f1_features_no_pitstop_v1",
    categorical_features=F1_FEATURE_SET.categorical_features,
    numeric_features=F1_FEATURE_SET.numeric_features,
    excluded_features=F1_FEATURE_SET.excluded_features,
    add_engineered_features=True,
)
CATBOOST_F1_PITSTOP_FEATURE_SET = FeatureSet(
    name="catboost_f1_features_with_pitstop_v1",
    categorical_features=F1_PITSTOP_FEATURE_SET.categorical_features,
    numeric_features=F1_PITSTOP_FEATURE_SET.numeric_features,
    excluded_features=F1_PITSTOP_FEATURE_SET.excluded_features,
    add_engineered_features=True,
)
LOGISTIC_EXPERIMENTS = [
    Experiment(feature_set.name, feature_set, "logistic_regression") for feature_set in FEATURE_SETS
]
CATBOOST_EXPERIMENTS = [
    Experiment(CATBOOST_F1_FEATURE_SET.name, CATBOOST_F1_FEATURE_SET, "catboost"),
    Experiment(CATBOOST_F1_PITSTOP_FEATURE_SET.name, CATBOOST_F1_PITSTOP_FEATURE_SET, "catboost"),
]
LIGHTGBM_F1_FEATURE_SET = FeatureSet(
    name="lightgbm_f1_features_no_pitstop_v1",
    categorical_features=F1_FEATURE_SET.categorical_features,
    numeric_features=F1_FEATURE_SET.numeric_features,
    excluded_features=F1_FEATURE_SET.excluded_features,
    add_engineered_features=True,
)
LIGHTGBM_F1_PITSTOP_FEATURE_SET = FeatureSet(
    name="lightgbm_f1_features_with_pitstop_v1",
    categorical_features=F1_PITSTOP_FEATURE_SET.categorical_features,
    numeric_features=F1_PITSTOP_FEATURE_SET.numeric_features,
    excluded_features=F1_PITSTOP_FEATURE_SET.excluded_features,
    add_engineered_features=True,
)
LIGHTGBM_TUNED_EXPERIMENTS = [
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
LIGHTGBM_EXPERIMENTS = [
    Experiment(LIGHTGBM_F1_FEATURE_SET.name, LIGHTGBM_F1_FEATURE_SET, "lightgbm"),
    Experiment(
        LIGHTGBM_F1_PITSTOP_FEATURE_SET.name,
        LIGHTGBM_F1_PITSTOP_FEATURE_SET,
        "lightgbm",
    ),
    *LIGHTGBM_TUNED_EXPERIMENTS,
]
ENSEMBLE_EXPERIMENT_NAME = "oof_rank_ensemble_v1"
ENSEMBLE_COMPONENT_NAMES = [
    "catboost_f1_features_with_pitstop_v1",
    "lightgbm_f1_features_no_pitstop_v1",
]
EXPERIMENTS = LOGISTIC_EXPERIMENTS + CATBOOST_EXPERIMENTS + LIGHTGBM_EXPERIMENTS


def year_race_groups(df: pd.DataFrame) -> pd.Series:
    """Return the grouping key used to keep each Year/Race together."""
    return df["Year"].astype(str) + "::" + df["Race"].astype(str)


def stratified_year_race_split(
    df: pd.DataFrame,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows deterministically while keeping Year/Race groups intact."""
    result = select_aligned_year_race_split(
        df,
        n_splits=n_splits,
        random_state=random_state,
    )
    return result.train, result.validation


def select_aligned_year_race_split(
    df: pd.DataFrame,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> SplitResult:
    """Select the same deterministic grouped validation fold across environments."""
    folds = aligned_year_race_folds(df, n_splits=n_splits, random_state=random_state)
    y = df[TARGET_COLUMN].astype(int)
    best_train_idx = None
    best_val_idx = None
    best_fold_id = None
    best_score = float("inf")
    overall_rate = float(y.mean())
    target_val_fraction = 1.0 / n_splits

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
        msg = "Unable to create an aligned Year/Race split"
        raise RuntimeError(msg)

    return SplitResult(
        df.iloc[best_train_idx].copy(),
        df.iloc[best_val_idx].copy(),
        best_fold_id,
    )


def stratified_year_race_folds(
    df: pd.DataFrame,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> list[tuple[list[int], list[int]]]:
    """Return grouped folds for out-of-fold validation."""
    return [
        (train_idx, validation_idx)
        for _, train_idx, validation_idx in aligned_year_race_folds(
            df,
            n_splits=n_splits,
            random_state=random_state,
        )
    ]


def aligned_year_race_folds(
    df: pd.DataFrame,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> list[tuple[int, list[int], list[int]]]:
    """Return deterministic row-balanced Year/Race folds.

    The scikit-learn StratifiedGroupKFold assignment differed between local and
    Kaggle environments. This implementation only depends on pandas sorting and
    input rows, so local and Kaggle runners select the same grouped holdout.
    """
    if random_state != RANDOM_STATE:
        msg = f"{SPLIT_ID} does not use alternate random_state values"
        raise ValueError(msg)

    group_key = year_race_groups(df)
    group_stats = (
        pd.DataFrame({"group": group_key, "target": df[TARGET_COLUMN].astype(int)})
        .groupby("group", sort=True)
        .agg(rows=("target", "size"))
        .reset_index()
        .sort_values(["rows", "group"], ascending=[False, True], kind="mergesort")
    )
    fold_groups: list[list[str]] = [[] for _ in range(n_splits)]
    fold_rows = [0 for _ in range(n_splits)]

    for row in group_stats.itertuples(index=False):
        fold_id = min(range(n_splits), key=lambda item: (fold_rows[item], item))
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


def add_f1_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add current-lap F1 strategy features without using target or future rows."""
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


def add_strategy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add strategy-inspired current-lap crosses from the reference project review."""
    features = df.copy()
    if not set(ENGINEERED_CATEGORICAL_FEATURES + ENGINEERED_NUMERIC_FEATURES).issubset(
        features.columns
    ):
        features = add_f1_features(features)

    lap_number = features["LapNumber"].clip(lower=1)
    degradation_slope = features["DegradationPerTyreLife"]

    features["CompoundTyreLifeBin"] = (
        features["Compound"].astype(str) + "_" + features["TyreLifeBin"].astype(str)
    )
    features["CompoundRaceProgressBin"] = (
        features["Compound"].astype(str) + "_" + features["RaceProgressBin"].astype(str)
    )
    features["DegradationSlopeBin"] = pd.cut(
        degradation_slope,
        bins=[float("-inf"), -8, -4, -1, 0, float("inf")],
        labels=["severe_loss", "high_loss", "moderate_loss", "low_loss", "gain"],
        include_lowest=True,
    ).astype("object")
    features["CompoundDegradationBin"] = (
        features["Compound"].astype(str) + "_" + features["DegradationSlopeBin"].astype(str)
    )
    features["PitWindowBand"] = pd.cut(
        features["RaceProgress"],
        bins=[-0.001, 0.2, 0.45, 0.7, 0.9, 1.0],
        labels=["early", "first_window", "middle", "late", "endgame"],
        include_lowest=True,
    ).astype("object")
    features["CompoundPitWindowBand"] = (
        features["Compound"].astype(str) + "_" + features["PitWindowBand"].astype(str)
    )
    features["LapTimeDeltaSign"] = pd.cut(
        features["LapTime_Delta"],
        bins=[float("-inf"), -0.5, 0.5, float("inf")],
        labels=["faster", "stable", "slower"],
        include_lowest=True,
    ).astype("object")
    features["CompoundLapTimeDeltaSign"] = (
        features["Compound"].astype(str) + "_" + features["LapTimeDeltaSign"].astype(str)
    )
    features["RaceYear"] = features["Race"].astype(str) + "_" + features["Year"].astype(str)
    features["RaceCompound"] = features["Race"].astype(str) + "_" + features["Compound"].astype(str)
    features["TyreLifeLapRatio"] = features["TyreLife"] / lap_number
    features["PositionChangeAbs"] = features["Position_Change"].abs()
    features["PositionTyreLife"] = features["Position"] * features["TyreLife"]
    features["PitStopRaceProgress"] = features["PitStop"] * features["RaceProgress"]
    features["PitStopRaceProgressBin"] = (
        features["PitStop"].astype(str) + "_" + features["RaceProgressBin"].astype(str)
    )
    features["PitStopStint"] = (
        features["PitStop"].astype(str) + "_S" + features["Stint"].astype(str)
    )
    return features


def prepare_features(df: pd.DataFrame, feature_set: FeatureSet) -> pd.DataFrame:
    """Return a DataFrame containing the columns required by a feature set."""
    features = df
    if feature_set.add_engineered_features:
        features = add_f1_features(features)
    if feature_set.add_strategy_features:
        features = add_strategy_features(features)
    return features


def build_model(feature_set: FeatureSet = BASELINE_FEATURE_SET) -> Pipeline:
    """Build the first leakage-aware baseline model."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                feature_set.categorical_features,
            ),
            ("numeric", StandardScaler(), feature_set.numeric_features),
        ],
    )
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ],
    )


def lightgbm_params(model_params: dict[str, int | float] | None = None) -> dict[str, object]:
    """Return LightGBM defaults plus experiment-specific overrides."""
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
    """Return a compact, reproducible LightGBM model description."""
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


def build_lightgbm_model(
    feature_set: FeatureSet = LIGHTGBM_F1_FEATURE_SET,
    model_params: dict[str, int | float] | None = None,
) -> Pipeline:
    """Build the LightGBM model used as a second boosted-tree family."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                feature_set.categorical_features,
            ),
            ("numeric", "passthrough", feature_set.numeric_features),
        ],
    )
    classifier = LGBMClassifier(**lightgbm_params(model_params))
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ],
    )


def build_catboost_model(task_type: str | None = None) -> CatBoostClassifier:
    """Build the CatBoost classifier used for F1 feature ablations."""
    params: dict[str, int | float | str | bool] = {
        "auto_class_weights": "Balanced",
        "allow_writing_files": False,
        "eval_metric": "AUC",
        "iterations": 150,
        "learning_rate": 0.08,
        "depth": 6,
        "random_seed": RANDOM_STATE,
        "verbose": False,
    }
    if task_type:
        params["task_type"] = task_type
    return CatBoostClassifier(**params)


def evaluate_model(
    model: Pipeline,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_set: FeatureSet = BASELINE_FEATURE_SET,
) -> float:
    """Fit on train and return validation ROC AUC."""
    train_features = prepare_features(train, feature_set)
    validation_features = prepare_features(validation, feature_set)
    model.fit(train_features[feature_set.feature_columns], train[TARGET_COLUMN])
    probabilities = model.predict_proba(validation_features[feature_set.feature_columns])[:, 1]
    return float(roc_auc_score(validation[TARGET_COLUMN], probabilities))


def sklearn_probabilities(
    model: Pipeline,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_set: FeatureSet,
) -> pd.Series:
    """Fit an sklearn-compatible pipeline and return validation probabilities."""
    train_features = prepare_features(train, feature_set)
    validation_features = prepare_features(validation, feature_set)
    model.fit(train_features[feature_set.feature_columns], train[TARGET_COLUMN])
    probabilities = model.predict_proba(validation_features[feature_set.feature_columns])[:, 1]
    return pd.Series(probabilities, index=validation.index)


def evaluate_catboost_model(
    model: CatBoostClassifier,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_set: FeatureSet,
) -> float:
    """Fit CatBoost with native categorical handling and return validation ROC AUC."""
    train_features = prepare_features(train, feature_set)
    validation_features = prepare_features(validation, feature_set)
    columns = feature_set.feature_columns
    cat_features = [columns.index(column) for column in feature_set.categorical_features]
    model.fit(
        train_features[columns],
        train[TARGET_COLUMN],
        cat_features=cat_features,
    )
    probabilities = model.predict_proba(validation_features[columns])[:, 1]
    return float(roc_auc_score(validation[TARGET_COLUMN], probabilities))


def catboost_probabilities(
    model: CatBoostClassifier,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_set: FeatureSet,
) -> pd.Series:
    """Fit CatBoost and return validation probabilities."""
    train_features = prepare_features(train, feature_set)
    validation_features = prepare_features(validation, feature_set)
    columns = feature_set.feature_columns
    cat_features = [columns.index(column) for column in feature_set.categorical_features]
    model.fit(
        train_features[columns],
        train[TARGET_COLUMN],
        cat_features=cat_features,
    )
    probabilities = model.predict_proba(validation_features[columns])[:, 1]
    return pd.Series(probabilities, index=validation.index)


def catboost_submission(
    model: CatBoostClassifier,
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    feature_set: FeatureSet,
) -> pd.DataFrame:
    """Train CatBoost on all labeled rows and return a valid test submission."""
    train_features = prepare_features(train, feature_set)
    test_features = prepare_features(test, feature_set)
    columns = feature_set.feature_columns
    cat_features = [columns.index(column) for column in feature_set.categorical_features]
    model.fit(
        train_features[columns],
        train[TARGET_COLUMN],
        cat_features=cat_features,
    )
    submission = sample_submission.copy()
    submission[TARGET_COLUMN] = model.predict_proba(test_features[columns])[:, 1]
    validate_submission(submission, sample_submission)
    return submission


def evaluate_experiment(
    experiment: Experiment,
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> float:
    """Fit one experiment and return validation ROC AUC."""
    probabilities = experiment_probabilities(experiment, train, validation)
    return float(roc_auc_score(validation[TARGET_COLUMN], probabilities))


def experiment_probabilities(
    experiment: Experiment,
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.Series:
    """Fit one experiment and return validation probabilities."""
    if experiment.model_family == "catboost":
        return catboost_probabilities(
            build_catboost_model(),
            train,
            validation,
            experiment.feature_set,
        )
    if experiment.model_family == "lightgbm":
        return sklearn_probabilities(
            build_lightgbm_model(experiment.feature_set, experiment.model_params),
            train,
            validation,
            experiment.feature_set,
        )
    return sklearn_probabilities(
        build_model(experiment.feature_set),
        train,
        validation,
        experiment.feature_set,
    )


def model_submission(
    model: Pipeline,
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    feature_set: FeatureSet = BASELINE_FEATURE_SET,
) -> pd.DataFrame:
    """Train on all labeled rows and return a valid test submission."""
    train_features = prepare_features(train, feature_set)
    test_features = prepare_features(test, feature_set)
    model.fit(train_features[feature_set.feature_columns], train[TARGET_COLUMN])
    submission = sample_submission.copy()
    submission[TARGET_COLUMN] = model.predict_proba(test_features[feature_set.feature_columns])[
        :, 1
    ]
    validate_submission(submission, sample_submission)
    return submission


def lightgbm_submission(
    model: Pipeline,
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    feature_set: FeatureSet,
) -> pd.DataFrame:
    """Train LightGBM on all labeled rows and return a valid test submission."""
    return model_submission(model, train, test, sample_submission, feature_set)


def experiment_submission(
    experiment: Experiment,
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> pd.DataFrame:
    """Train one experiment on all labeled rows and return a valid submission."""
    if experiment.model_family == "catboost":
        return catboost_submission(
            build_catboost_model(),
            train,
            test,
            sample_submission,
            experiment.feature_set,
        )
    if experiment.model_family == "lightgbm":
        return lightgbm_submission(
            build_lightgbm_model(experiment.feature_set, experiment.model_params),
            train,
            test,
            sample_submission,
            experiment.feature_set,
        )
    return model_submission(
        build_model(experiment.feature_set),
        train,
        test,
        sample_submission,
        experiment.feature_set,
    )


def split_summary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    validation_fold: int | None = None,
) -> dict[str, int | float | str]:
    """Return compact diagnostics for the selected validation split."""
    return {
        "split_id": SPLIT_ID,
        "validation_fold": "" if validation_fold is None else validation_fold,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_target_rate": float(train[TARGET_COLUMN].mean()),
        "validation_target_rate": float(validation[TARGET_COLUMN].mean()),
        "train_year_race_groups": train.groupby(["Year", "Race"]).ngroups,
        "validation_year_race_groups": validation.groupby(["Year", "Race"]).ngroups,
    }


def metric_row(
    auc: float,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    experiment: Experiment,
    validation_fold: int | None = None,
) -> dict[str, int | float | str]:
    """Return a metrics row for one validation experiment."""
    summary = split_summary(train, validation, validation_fold)
    feature_set = experiment.feature_set
    if experiment.model_family == "catboost":
        model_name = (
            "CatBoostClassifier(auto_class_weights='Balanced', iterations=150, "
            "learning_rate=0.08, depth=6)"
        )
    elif experiment.model_family == "lightgbm":
        model_name = lightgbm_model_name(experiment.model_params)
    elif experiment.model_family == "ensemble":
        model_name = "Equal-weight OOF rank average"
    else:
        model_name = "LogisticRegression(class_weight='balanced')"
    return {
        "experiment": experiment.name,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_files": "data/raw/train.csv;data/raw/test.csv;data/raw/sample_submission.csv",
        "command": "python -m predicting_f1_pit_stops.train",
        "model": model_name,
        "split": "deterministic Year/Race row-balanced fold, best of 5 folds",
        "features": ",".join(feature_set.feature_columns),
        "excluded_features": ",".join(feature_set.excluded_features),
        "metric": "roc_auc",
        "validation_roc_auc": auc,
        **summary,
    }


def rank_ensemble_probabilities(predictions: list[pd.Series]) -> pd.Series:
    """Average percentile ranks so diverse model scores share a comparable scale."""
    if not predictions:
        msg = "At least one prediction series is required for a rank ensemble"
        raise ValueError(msg)
    ranked = [prediction.rank(method="average", pct=True) for prediction in predictions]
    return sum(ranked) / len(ranked)


def ensemble_component_experiments() -> list[Experiment]:
    """Return ensemble components in declared order and fail if one is missing."""
    experiments_by_name = {experiment.name: experiment for experiment in EXPERIMENTS}
    missing = [name for name in ENSEMBLE_COMPONENT_NAMES if name not in experiments_by_name]
    if missing:
        msg = f"Missing ensemble component experiments: {missing}"
        raise RuntimeError(msg)
    return [experiments_by_name[name] for name in ENSEMBLE_COMPONENT_NAMES]


def oof_experiment_probabilities(
    experiment: Experiment,
    train: pd.DataFrame,
) -> pd.Series:
    """Return out-of-fold probabilities for one experiment on grouped folds."""
    predictions = pd.Series(index=train.index, dtype=float)
    for fold_train_idx, fold_validation_idx in stratified_year_race_folds(train):
        fold_train = train.iloc[fold_train_idx].copy()
        fold_validation = train.iloc[fold_validation_idx].copy()
        fold_probabilities = experiment_probabilities(experiment, fold_train, fold_validation)
        predictions.loc[fold_validation.index] = fold_probabilities
    if predictions.isna().any():
        msg = f"Missing OOF predictions for {experiment.name}"
        raise RuntimeError(msg)
    return predictions


def ensemble_metric_row(
    auc: float,
    train: pd.DataFrame,
    component_names: list[str],
) -> dict[str, int | float | str]:
    """Return the metric row for the equal-weight out-of-fold rank ensemble."""
    group_count = train.groupby(["Year", "Race"]).ngroups
    return {
        "experiment": ENSEMBLE_EXPERIMENT_NAME,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_files": "data/raw/train.csv;data/raw/test.csv;data/raw/sample_submission.csv",
        "command": "python -m predicting_f1_pit_stops.train",
        "model": "Equal-weight OOF rank average",
        "split": "deterministic Year/Race row-balanced 5-fold OOF",
        "split_id": SPLIT_ID,
        "validation_fold": "oof",
        "features": "rank_average:" + ",".join(component_names),
        "excluded_features": "component-specific",
        "metric": "roc_auc",
        "validation_roc_auc": auc,
        "train_rows": len(train),
        "validation_rows": len(train),
        "train_target_rate": float(train[TARGET_COLUMN].mean()),
        "validation_target_rate": float(train[TARGET_COLUMN].mean()),
        "train_year_race_groups": group_count,
        "validation_year_race_groups": group_count,
    }


def experiment_test_probabilities(
    experiment: Experiment,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.Series:
    """Train one experiment on all labeled rows and return test probabilities."""
    if experiment.model_family == "catboost":
        train_features = prepare_features(train, experiment.feature_set)
        test_features = prepare_features(test, experiment.feature_set)
        columns = experiment.feature_set.feature_columns
        cat_features = [
            columns.index(column) for column in experiment.feature_set.categorical_features
        ]
        model = build_catboost_model()
        model.fit(
            train_features[columns],
            train[TARGET_COLUMN],
            cat_features=cat_features,
        )
        probabilities = model.predict_proba(test_features[columns])[:, 1]
        return pd.Series(probabilities, index=test.index)

    if experiment.model_family == "lightgbm":
        model = build_lightgbm_model(experiment.feature_set, experiment.model_params)
    else:
        model = build_model(experiment.feature_set)
    train_features = prepare_features(train, experiment.feature_set)
    test_features = prepare_features(test, experiment.feature_set)
    model.fit(train_features[experiment.feature_set.feature_columns], train[TARGET_COLUMN])
    probabilities = model.predict_proba(test_features[experiment.feature_set.feature_columns])[:, 1]
    return pd.Series(probabilities, index=test.index)


def rank_ensemble_submission(
    component_experiments: list[Experiment],
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> pd.DataFrame:
    """Train ensemble components on all labeled rows and return rank-averaged submission."""
    predictions = [
        experiment_test_probabilities(experiment, train, test)
        for experiment in component_experiments
    ]
    submission = sample_submission.copy()
    submission[TARGET_COLUMN] = rank_ensemble_probabilities(predictions).to_numpy()
    validate_submission(submission, sample_submission)
    return submission


def write_metrics(rows: list[dict[str, int | float | str]], path=METRICS_PATH) -> pd.DataFrame:
    """Write metrics for all validation experiments."""
    metrics = pd.DataFrame(rows).sort_values("validation_roc_auc", ascending=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(path, index=False)
    return metrics


def year_shift_audit_row(
    experiment: Experiment,
    train: pd.DataFrame,
    holdout_year: int,
) -> dict[str, int | float | str]:
    """Return a leave-one-year-out audit row for one experiment."""
    split_train = train.loc[train["Year"] != holdout_year].copy()
    validation = train.loc[train["Year"] == holdout_year].copy()
    auc = evaluate_experiment(experiment, split_train, validation)
    note = (
        "2023 has unusually low target rate; interpret this holdout separately."
        if holdout_year == 2023
        else ""
    )
    return {
        "experiment": experiment.name,
        "model_family": experiment.model_family,
        "holdout_year": holdout_year,
        "validation_roc_auc": auc,
        "target_rate": float(validation[TARGET_COLUMN].mean()),
        "train_rows": len(split_train),
        "validation_rows": len(validation),
        "notes": note,
    }


def write_year_shift_audit(
    rows: list[dict[str, int | float | str]],
    path=YEAR_SHIFT_AUDIT_PATH,
) -> pd.DataFrame:
    """Write leave-one-year-out year-shift audit results."""
    audit = pd.DataFrame(rows).sort_values(["experiment", "holdout_year"])
    path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(path, index=False)
    return audit


def main() -> None:
    """Run baseline validation and write a model-based Kaggle submission."""
    train = load_train()
    test = load_test()
    sample_submission = load_sample_submission()

    split = select_aligned_year_race_split(train)
    split_train = split.train
    validation = split.validation
    rows = []
    for experiment in EXPERIMENTS:
        probabilities = experiment_probabilities(experiment, split_train, validation)
        auc = float(roc_auc_score(validation[TARGET_COLUMN], probabilities))
        rows.append(metric_row(auc, split_train, validation, experiment, split.validation_fold))

    component_experiments = ensemble_component_experiments()
    component_oof_predictions = [
        oof_experiment_probabilities(experiment, train) for experiment in component_experiments
    ]
    ensemble_probabilities = rank_ensemble_probabilities(
        component_oof_predictions,
    )
    ensemble_auc = float(roc_auc_score(train[TARGET_COLUMN], ensemble_probabilities))
    rows.append(
        ensemble_metric_row(
            ensemble_auc,
            train,
            [experiment.name for experiment in component_experiments],
        ),
    )

    metrics = write_metrics(rows)
    best = metrics.iloc[0]
    best_experiment = next(
        (experiment for experiment in EXPERIMENTS if experiment.name == best["experiment"]),
        None,
    )
    best_catboost = metrics.loc[
        metrics["experiment"].isin([item.name for item in CATBOOST_EXPERIMENTS])
    ].iloc[0]
    best_catboost_experiment = next(
        experiment
        for experiment in CATBOOST_EXPERIMENTS
        if experiment.name == best_catboost["experiment"]
    )
    audit_experiments = [
        Experiment(
            F1_PITSTOP_FEATURE_SET.name,
            F1_PITSTOP_FEATURE_SET,
            "logistic_regression",
        ),
        best_catboost_experiment,
        *LIGHTGBM_EXPERIMENTS,
    ]
    audit_rows = [
        year_shift_audit_row(experiment, train, int(holdout_year))
        for experiment in audit_experiments
        for holdout_year in sorted(train["Year"].unique())
    ]
    audit = write_year_shift_audit(audit_rows)

    if best["experiment"] == ENSEMBLE_EXPERIMENT_NAME:
        submission = rank_ensemble_submission(
            component_experiments,
            train,
            test,
            sample_submission,
        )
    elif best_experiment is not None:
        submission = experiment_submission(best_experiment, train, test, sample_submission)
    else:
        msg = f"Unknown best experiment: {best['experiment']}"
        raise RuntimeError(msg)
    DEFAULT_SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(DEFAULT_SUBMISSION_PATH, index=False)

    for row in metrics.itertuples(index=False):
        print(f"{row.experiment}: validation ROC AUC {row.validation_roc_auc:.6f}")
    print(f"Best experiment: {best['experiment']} ({best['validation_roc_auc']:.6f})")
    print(f"Validation split: {SPLIT_ID}, fold {split.validation_fold}")
    print(
        f"Best CatBoost experiment: {best_catboost['experiment']} "
        f"({best_catboost['validation_roc_auc']:.6f})"
    )
    print(f"Wrote metrics to {METRICS_PATH}")
    print(f"Wrote year-shift audit to {YEAR_SHIFT_AUDIT_PATH} ({len(audit)} rows)")
    print(f"Wrote submission to {DEFAULT_SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
