from pathlib import Path

import pandas as pd
import pytest

from predicting_f1_pit_stops.data import (
    load_sample_submission,
    target_prior,
    validate_submission,
)
from predicting_f1_pit_stops.schema import TARGET_COLUMN


def test_target_prior_uses_train_target_mean() -> None:
    train = pd.DataFrame({"id": [1, 2, 3, 4], TARGET_COLUMN: [0, 0, 1, 1]})

    assert target_prior(train) == 0.5


def test_validate_submission_accepts_matching_probability_file() -> None:
    sample = pd.DataFrame({"id": [10, 11], TARGET_COLUMN: [0.0, 0.0]})
    submission = pd.DataFrame({"id": [10, 11], TARGET_COLUMN: [0.2, 0.8]})

    validate_submission(submission, sample)


def test_validate_submission_rejects_wrong_id_order() -> None:
    sample = pd.DataFrame({"id": [10, 11], TARGET_COLUMN: [0.0, 0.0]})
    submission = pd.DataFrame({"id": [11, 10], TARGET_COLUMN: [0.2, 0.8]})

    with pytest.raises(ValueError, match="ids must match"):
        validate_submission(submission, sample)


def test_load_sample_submission_requires_target_column(tmp_path: Path) -> None:
    path = tmp_path / "sample_submission.csv"
    pd.DataFrame({"id": [1, 2]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_sample_submission(path)
