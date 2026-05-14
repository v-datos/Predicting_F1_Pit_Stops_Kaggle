from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

TRAIN_PATH = RAW_DIR / "train.csv"
TEST_PATH = RAW_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = RAW_DIR / "sample_submission.csv"
DEFAULT_SUBMISSION_PATH = SUBMISSIONS_DIR / "submission.csv"
