from pathlib import Path

import pandas as pd

from predicting_f1_pit_stops.paths import (
    SAMPLE_SUBMISSION_PATH,
    TEST_PATH,
    TRAIN_PATH,
)
from predicting_f1_pit_stops.schema import (
    ID_COLUMN,
    MINIMUM_TEST_COLUMNS,
    MINIMUM_TRAIN_COLUMNS,
    SUBMISSION_COLUMNS,
    TARGET_COLUMN,
)


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file and raise a clear error when it is missing."""
    if not path.exists():
        msg = f"Missing CSV file: {path}"
        raise FileNotFoundError(msg)
    return pd.read_csv(path)


def load_train(path: Path = TRAIN_PATH) -> pd.DataFrame:
    """Load official Kaggle train data and verify minimum columns."""
    train = read_csv(path)
    validate_required_columns(train, MINIMUM_TRAIN_COLUMNS, "train")
    return train


def load_test(path: Path = TEST_PATH) -> pd.DataFrame:
    """Load official Kaggle test data and verify minimum columns."""
    test = read_csv(path)
    validate_required_columns(test, MINIMUM_TEST_COLUMNS, "test")
    return test


def load_sample_submission(path: Path = SAMPLE_SUBMISSION_PATH) -> pd.DataFrame:
    """Load official Kaggle sample submission and verify required columns."""
    sample_submission = read_csv(path)
    validate_required_columns(sample_submission, SUBMISSION_COLUMNS, "sample_submission")
    return sample_submission


def validate_required_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    """Raise ValueError if required columns are missing from a DataFrame."""
    missing = [column for column in required if column not in df.columns]
    if missing:
        msg = f"{name} is missing required columns: {missing}"
        raise ValueError(msg)


def validate_submission(submission: pd.DataFrame, sample_submission: pd.DataFrame) -> None:
    """Verify submission shape and columns against the sample submission."""
    validate_required_columns(submission, SUBMISSION_COLUMNS, "submission")
    if list(submission.columns) != SUBMISSION_COLUMNS:
        msg = f"submission columns must be exactly {SUBMISSION_COLUMNS}"
        raise ValueError(msg)
    if len(submission) != len(sample_submission):
        msg = "submission row count must match sample_submission"
        raise ValueError(msg)
    if not submission[ID_COLUMN].equals(sample_submission[ID_COLUMN]):
        msg = "submission ids must match sample_submission ids in order"
        raise ValueError(msg)
    if not submission[TARGET_COLUMN].between(0, 1).all():
        msg = f"{TARGET_COLUMN} predictions must be probabilities between 0 and 1"
        raise ValueError(msg)


def target_prior(train: pd.DataFrame) -> float:
    """Return the mean target rate for a constant-probability baseline."""
    validate_required_columns(train, [TARGET_COLUMN], "train")
    return float(train[TARGET_COLUMN].mean())
