TARGET_COLUMN = "PitNextLap"
ID_COLUMN = "id"

SUBMISSION_COLUMNS = [ID_COLUMN, TARGET_COLUMN]

MINIMUM_TRAIN_COLUMNS = [
    ID_COLUMN,
    "Driver",
    "Compound",
    "Race",
    "Year",
    "PitStop",
    "LapNumber",
    "Stint",
    "TyreLife",
    "Position",
    "LapTime (s)",
    "LapTime_Delta",
    "Cumulative_Degradation",
    "RaceProgress",
    "Position_Change",
    TARGET_COLUMN,
]

MINIMUM_TEST_COLUMNS = [
    ID_COLUMN,
    "Driver",
    "Compound",
    "Race",
    "Year",
    "PitStop",
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
