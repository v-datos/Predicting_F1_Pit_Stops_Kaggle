import pandas as pd

from predicting_f1_pit_stops.baseline import constant_probability_submission
from predicting_f1_pit_stops.schema import TARGET_COLUMN


def test_constant_probability_submission_preserves_ids() -> None:
    sample = pd.DataFrame({"id": [439140, 439141], TARGET_COLUMN: [0.0, 0.0]})

    submission = constant_probability_submission(sample, 0.25)

    assert submission["id"].tolist() == [439140, 439141]
    assert submission[TARGET_COLUMN].tolist() == [0.25, 0.25]
