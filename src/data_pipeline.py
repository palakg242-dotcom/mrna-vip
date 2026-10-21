"""Week 1 - data loading, validation, cleaning, encoding, splitting and scaling.

Satisfies FR-1.1 to FR-1.4 of the PRD.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REQUIRED_COLUMNS = [
    "Patient_ID", "Age", "Biomarker_Level", "Pre_Existing_Conditions",
    "Vaccine_Dose_mcg", "Prior_Reaction_History", "Reaction_Score",
]
NUMERIC_COLS = ["Age", "Biomarker_Level", "Vaccine_Dose_mcg"]
CATEGORICAL_COL = "Prior_Reaction_History"
COUNT_COL = "Pre_Existing_Conditions"
TARGET_COL = "Reaction_Score"
FEATURE_COLS = NUMERIC_COLS + [CATEGORICAL_COL, COUNT_COL]

# Prior reaction history is ordinal (None < Mild < Moderate), so we encode it
# in that order instead of alphabetically (LabelEncoder would give Mild=0).
REACTION_ORDER = {"None": 0, "Mild": 1, "Moderate": 2}
RANDOM_SEED = 42


class MissingColumnsError(ValueError):
    """Raised when the input CSV lacks one or more required columns (FR-1.1)."""


def load_raw(path: str) -> pd.DataFrame:
    """Load the CSV and validate that all 7 required columns exist (FR-1.1).

    IMPORTANT: keep_default_na=False. In this dataset the text "None" in
    Prior_Reaction_History means "no prior reaction" - a real category. By
    default pandas turns the string "None" into NaN, which would make ~58% of
    the column look missing, and mode-imputation would then wrongly relabel
    those patients as "Mild". Only truly empty cells count as missing.
    """
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise MissingColumnsError(
            f"Required column(s) missing from {path}: {', '.join(missing)}"
        )
    return df


def profile(df: pd.DataFrame) -> None:
    """Print shape, dtypes, missing counts and summary statistics."""
    print("Shape:", df.shape)
    print("\nDtypes:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nSummary statistics:\n", df.describe(include="all").T)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values (FR-1.2).

    Imputation (not row removal) is used because only ~4% of rows are affected
    and with 500 rows we cannot afford to throw data away.
    - Numeric columns -> median (robust to the skew in Biomarker_Level).
    - Prior_Reaction_History -> mode (most frequent category).
    """
    df = df.copy()
    for col in NUMERIC_COLS:
        df[col] = df[col].fillna(df[col].median())
    df[CATEGORICAL_COL] = df[CATEGORICAL_COL].fillna(df[CATEGORICAL_COL].mode()[0])

    unknown = set(df[CATEGORICAL_COL].unique()) - set(REACTION_ORDER)
    if unknown:
        raise ValueError(f"Unexpected {CATEGORICAL_COL} categories: {sorted(unknown)}")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal-encode Prior_Reaction_History (None=0, Mild=1, Moderate=2)."""
    df = df.copy()
    df[CATEGORICAL_COL] = df[CATEGORICAL_COL].map(REACTION_ORDER)
    return df


def split_data(df: pd.DataFrame, seed: int = RANDOM_SEED):
    """80/20 train/test split, stratified on Reaction_Score (FR-1.4)."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit the scaler on the TRAINING set only, to avoid test-set leakage."""
    return StandardScaler().fit(X_train[NUMERIC_COLS])


def apply_scaler(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Standardize Age, Biomarker_Level, Vaccine_Dose_mcg (FR-1.3)."""
    X = X.copy()
    X[NUMERIC_COLS] = scaler.transform(X[NUMERIC_COLS])
    return X
