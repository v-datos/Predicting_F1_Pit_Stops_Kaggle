import pandas as pd
import pytest

from predicting_f1_pit_stops.schema import TARGET_COLUMN
from predicting_f1_pit_stops.train import (
    CATBOOST_EXPERIMENTS,
    ENSEMBLE_COMPONENT_NAMES,
    ENSEMBLE_EXPERIMENT_NAME,
    F1_FEATURE_SET,
    FEATURE_COLUMNS,
    LIGHTGBM_EXPERIMENTS,
    PITSTOP_FEATURE_SET,
    STRATEGY_FEATURE_SET,
    STRATEGY_PITSTOP_FEATURE_SET,
    add_f1_features,
    add_strategy_features,
    aligned_year_race_folds,
    rank_ensemble_probabilities,
    select_aligned_year_race_split,
    stratified_year_race_split,
    year_race_groups,
    year_shift_audit_row,
)


def test_year_race_groups_combines_year_and_race() -> None:
    df = pd.DataFrame({"Year": [2025, 2025], "Race": ["A", "B"]})

    assert year_race_groups(df).tolist() == ["2025::A", "2025::B"]


def test_stratified_year_race_split_keeps_groups_disjoint() -> None:
    rows = []
    for group_id in range(10):
        for lap in range(4):
            rows.append(
                {
                    "Year": 2020 + (group_id % 2),
                    "Race": f"Race {group_id}",
                    "Driver": "D001",
                    "Compound": "MEDIUM",
                    "LapNumber": lap + 1,
                    "Stint": 1,
                    "TyreLife": float(lap + 1),
                    "Position": 1,
                    "LapTime (s)": 90.0,
                    "LapTime_Delta": 0.0,
                    "Cumulative_Degradation": -1.0,
                    "RaceProgress": 0.1,
                    "Position_Change": 0,
                    TARGET_COLUMN: int(group_id % 2 == 0),
                },
            )
    df = pd.DataFrame(rows)

    train, validation = stratified_year_race_split(df, n_splits=5, random_state=42)

    train_groups = set(year_race_groups(train))
    validation_groups = set(year_race_groups(validation))
    assert train_groups.isdisjoint(validation_groups)
    assert len(train) + len(validation) == len(df)


def test_aligned_year_race_folds_are_deterministic_and_row_balanced() -> None:
    rows = []
    for group_id in range(10):
        for _lap in range(group_id + 1):
            rows.append(
                {
                    "Year": 2020 + (group_id % 2),
                    "Race": f"Race {group_id}",
                    TARGET_COLUMN: int(group_id % 3 == 0),
                },
            )
    df = pd.DataFrame(rows)

    first = aligned_year_race_folds(df, n_splits=5)
    second = aligned_year_race_folds(df, n_splits=5)
    split = select_aligned_year_race_split(df, n_splits=5)

    assert first == second
    assert split.validation_fold in {0, 1, 2, 3, 4}
    assert sorted(index for _, _, validation in first for index in validation) == list(
        range(len(df)),
    )
    validation_lengths = [len(validation) for _, _, validation in first]
    assert max(validation_lengths) - min(validation_lengths) <= 4


def test_feature_columns_exclude_pitstop_and_target() -> None:
    assert "PitStop" not in FEATURE_COLUMNS
    assert TARGET_COLUMN not in FEATURE_COLUMNS


def test_pitstop_feature_set_includes_pitstop() -> None:
    assert "PitStop" in PITSTOP_FEATURE_SET.feature_columns
    assert TARGET_COLUMN not in PITSTOP_FEATURE_SET.feature_columns


def test_catboost_experiments_cover_pitstop_ablation() -> None:
    experiments = {experiment.name: experiment for experiment in CATBOOST_EXPERIMENTS}

    assert set(experiments) == {
        "catboost_f1_features_no_pitstop_v1",
        "catboost_f1_features_with_pitstop_v1",
    }
    assert (
        "PitStop"
        not in experiments["catboost_f1_features_no_pitstop_v1"].feature_set.feature_columns
    )
    assert (
        "PitStop" in experiments["catboost_f1_features_with_pitstop_v1"].feature_set.feature_columns
    )


def test_lightgbm_experiments_cover_pitstop_and_tuning() -> None:
    experiments = {experiment.name: experiment for experiment in LIGHTGBM_EXPERIMENTS}

    assert set(experiments) == {
        "lightgbm_f1_features_no_pitstop_v1",
        "lightgbm_f1_features_with_pitstop_v1",
        "lightgbm_f1_features_no_pitstop_tuned_leaves_v1",
        "lightgbm_f1_features_no_pitstop_tuned_regularized_v1",
        "lightgbm_f1_features_no_pitstop_tuned_slow_v1",
    }
    assert (
        "PitStop"
        not in experiments["lightgbm_f1_features_no_pitstop_v1"].feature_set.feature_columns
    )
    assert (
        "PitStop" in experiments["lightgbm_f1_features_with_pitstop_v1"].feature_set.feature_columns
    )
    tuned_params = experiments["lightgbm_f1_features_no_pitstop_tuned_regularized_v1"].model_params
    assert tuned_params is not None
    assert tuned_params["min_child_samples"] == 80
    assert tuned_params["reg_lambda"] == 1.0


def test_rank_ensemble_uses_expected_components() -> None:
    assert ENSEMBLE_EXPERIMENT_NAME == "oof_rank_ensemble_v1"
    assert ENSEMBLE_COMPONENT_NAMES == [
        "catboost_f1_features_with_pitstop_v1",
        "lightgbm_f1_features_no_pitstop_v1",
    ]


def test_rank_ensemble_probabilities_average_percentile_ranks() -> None:
    first = pd.Series([0.1, 0.4, 0.2])
    second = pd.Series([0.7, 0.3, 0.2])

    ensemble = rank_ensemble_probabilities([first, second])

    expected = (
        first.rank(method="average", pct=True) + second.rank(method="average", pct=True)
    ) / 2
    assert ensemble.tolist() == expected.tolist()


def test_f1_feature_set_adds_engineered_columns() -> None:
    df = pd.DataFrame(
        {
            "Compound": ["MEDIUM"],
            "Stint": [2],
            "TyreLife": [10.0],
            "RaceProgress": [0.5],
            "Position": [4],
            "Cumulative_Degradation": [-20.0],
            "LapTime_Delta": [-1.5],
        },
    )

    features = add_f1_features(df)

    assert F1_FEATURE_SET.add_engineered_features
    assert features["CompoundStint"].tolist() == ["MEDIUM_S2"]
    assert features["TyreLifePerStint"].tolist() == [5.0]
    assert features["TyreLifeRaceProgress"].tolist() == [5.0]
    assert features["StintRaceProgress"].tolist() == [1.0]
    assert features["PositionRaceProgress"].tolist() == [2.0]
    assert features["DegradationPerTyreLife"].tolist() == [-2.0]
    assert features["DegradationRaceProgress"].tolist() == [-10.0]
    assert features["LapTimeDeltaAbs"].tolist() == [1.5]


def test_strategy_feature_set_adds_reference_inspired_columns() -> None:
    df = pd.DataFrame(
        {
            "Race": ["Canadian Grand Prix"],
            "Year": [2025],
            "Compound": ["MEDIUM"],
            "PitStop": [1],
            "LapNumber": [20],
            "Stint": [2],
            "TyreLife": [10.0],
            "RaceProgress": [0.5],
            "Position": [4],
            "Position_Change": [-2],
            "Cumulative_Degradation": [-20.0],
            "LapTime_Delta": [-1.5],
        },
    )

    features = add_strategy_features(df)

    assert STRATEGY_FEATURE_SET.add_strategy_features
    assert STRATEGY_PITSTOP_FEATURE_SET.add_strategy_features
    assert features["CompoundTyreLifeBin"].tolist() == ["MEDIUM_early"]
    assert features["CompoundRaceProgressBin"].tolist() == ["MEDIUM_race_q2"]
    assert features["DegradationSlopeBin"].tolist() == ["moderate_loss"]
    assert features["CompoundDegradationBin"].tolist() == ["MEDIUM_moderate_loss"]
    assert features["PitWindowBand"].tolist() == ["middle"]
    assert features["CompoundPitWindowBand"].tolist() == ["MEDIUM_middle"]
    assert features["LapTimeDeltaSign"].tolist() == ["faster"]
    assert features["CompoundLapTimeDeltaSign"].tolist() == ["MEDIUM_faster"]
    assert features["RaceYear"].tolist() == ["Canadian Grand Prix_2025"]
    assert features["RaceCompound"].tolist() == ["Canadian Grand Prix_MEDIUM"]
    assert features["TyreLifeLapRatio"].tolist() == [0.5]
    assert features["PositionChangeAbs"].tolist() == [2]
    assert features["PositionTyreLife"].tolist() == [40.0]
    assert features["PitStopRaceProgress"].tolist() == [0.5]
    assert features["PitStopRaceProgressBin"].tolist() == ["1_race_q2"]
    assert features["PitStopStint"].tolist() == ["1_S2"]


def test_year_shift_audit_row_reports_holdout_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "Year": [2022, 2022, 2023, 2023],
            "Race": ["A", "A", "B", "B"],
            TARGET_COLUMN: [0, 1, 0, 0],
        },
    )

    def fake_evaluate_experiment(*_args: object) -> float:
        return 0.75

    monkeypatch.setattr(
        "predicting_f1_pit_stops.train.evaluate_experiment",
        fake_evaluate_experiment,
    )

    row = year_shift_audit_row(CATBOOST_EXPERIMENTS[0], df, 2023)

    assert row["experiment"] == "catboost_f1_features_no_pitstop_v1"
    assert row["holdout_year"] == 2023
    assert row["validation_roc_auc"] == 0.75
    assert row["target_rate"] == 0.0
    assert row["train_rows"] == 2
    assert row["validation_rows"] == 2
    assert "2023" in str(row["notes"])
