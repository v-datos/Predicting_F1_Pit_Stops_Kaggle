import pandas as pd

from predicting_f1_pit_stops.data import (
    load_sample_submission,
    load_train,
    target_prior,
    validate_submission,
)
from predicting_f1_pit_stops.paths import DEFAULT_SUBMISSION_PATH
from predicting_f1_pit_stops.schema import TARGET_COLUMN


def constant_probability_submission(
    sample_submission: pd.DataFrame,
    probability: float,
) -> pd.DataFrame:
    """Create a valid submission using one probability for every row."""
    submission = sample_submission.copy()
    submission[TARGET_COLUMN] = probability
    return submission


def main() -> None:
    """Generate the first constant-prior baseline submission."""
    train = load_train()
    sample_submission = load_sample_submission()
    probability = target_prior(train)
    submission = constant_probability_submission(sample_submission, probability)
    validate_submission(submission, sample_submission)

    DEFAULT_SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(DEFAULT_SUBMISSION_PATH, index=False)
    print(f"Wrote {DEFAULT_SUBMISSION_PATH} with constant probability {probability:.6f}")


if __name__ == "__main__":
    main()
