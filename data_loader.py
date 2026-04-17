"""
data_loader.py
==============
Ingests the xAPI-Educational Mining Dataset (StudentPerformance.csv),
extracts academic and behavioural features relevant to Outcome-Based
Education (OBE), and produces a filtered working set of 480–650 records
suitable for fuzzy-logic rule validation.

Dataset source:
    Amrieh, E. A., Hamtini, T., & Aljarah, I. (2016).
    "Mining Educational Data to Predict Student's Academic Performance
    using Ensemble Methods."
    International Journal of Database Theory and Application, 9(8), 119–136.
"""

import sys
import pathlib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR  = pathlib.Path(__file__).parent / "data"
DATA_FILE = DATA_DIR / "StudentPerformance.csv"

# ---------------------------------------------------------------------------
# Column name map
# The xAPI dataset ships with different capitalisation variants depending on
# the source (Kaggle vs UCI).  We normalise everything to lowercase_snake.
# ---------------------------------------------------------------------------
COLUMN_ALIASES: dict[str, str] = {
    # Academic features
    "raisedhands"         : "raised_hands",
    "raised_hands"        : "raised_hands",
    "visittedresources"   : "visited_resources",
    "visitedresources"    : "visited_resources",
    "visited_resources"   : "visited_resources",
    "announcementsview"   : "announcements_view",
    "announcements_view"  : "announcements_view",
    "discussion"          : "discussion",
    # Internal assessment / mid-term equivalents
    # The dataset encodes academic outcome as a three-class label:
    #   H = High (≥ 85), M = Medium (65–84), L = Low (< 65)
    # We also retain the semester, topic, and stage for stratification.
    "class"               : "performance_class",
    "semester"            : "semester",
    "topic"               : "topic",
    "stagelevel"          : "stage_level",
    "stage_level"         : "stage_level",
    "grade"               : "grade",
    "gender"              : "gender",
    "nationality"         : "nationality",
    "placeofstudy"        : "place_of_study",
    "place_of_study"      : "place_of_study",
    "studentabsencedays"  : "absence_days",
    "student_absence_days": "absence_days",
    "absence_days"        : "absence_days",
    "relation"            : "relation",
    "parentansweringsurvey"   : "parent_survey",
    "parent_answering_survey" : "parent_survey",
    "parentsschoolsatisfaction": "parent_satisfaction",
    "parent_school_satisfaction": "parent_satisfaction",
}

# ---------------------------------------------------------------------------
# Feature groups used throughout the OBE assessment pipeline
# ---------------------------------------------------------------------------
ACADEMIC_FEATURES: list[str] = [
    "raised_hands",        # proxy for classroom engagement / internal assessment
    "announcements_view",  # information-seeking behaviour, mirrors mid-term prep
    "discussion",          # collaborative academic participation
    "performance_class",   # final outcome label  (H / M / L)
]

BEHAVIOURAL_FEATURES: list[str] = [
    "visited_resources",   # LMS resource access — direct behavioural indicator
    "absence_days",        # Under-7 vs above-7 absences (binary in source data)
    "parent_survey",       # parental engagement
    "parent_satisfaction", # institutional satisfaction signal
]

# Combined working feature set (excludes pure demographic columns)
WORKING_FEATURES: list[str] = ACADEMIC_FEATURES + BEHAVIOURAL_FEATURES

# ---------------------------------------------------------------------------
# Target working-set window
# ---------------------------------------------------------------------------
FILTER_MIN = 480
FILTER_MAX = 650


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case all column names and apply the alias map."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df.rename(columns=COLUMN_ALIASES, inplace=True)
    return df


def _encode_absence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the categorical absence column to a numeric sentinel so it can
    participate in downstream fuzzy membership calculations.
        'Under-7'  → 0   (low absence)
        'Above-7'  → 1   (high absence)
    """
    if "absence_days" in df.columns:
        mapping = {"under-7": 0, "above-7": 1}
        df["absence_days"] = (
            df["absence_days"]
            .str.strip()
            .str.lower()
            .map(mapping)
        )
    return df


def _encode_binary_text(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Map Yes/No text columns to 1/0."""
    if col in df.columns:
        df[col] = (
            df[col]
            .str.strip()
            .str.lower()
            .map({"yes": 1, "no": 0})
        )
    return df


def _validate_numeric_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    The xAPI dataset specifies that raised_hands, visited_resources,
    announcements_view, and discussion are each in [0, 100].
    Drop rows that fall outside this range — they represent data-entry
    artefacts, not genuine edge cases.
    """
    bounded = ["raised_hands", "visited_resources", "announcements_view", "discussion"]
    for col in bounded:
        if col in df.columns:
            df = df[(df[col] >= 0) & (df[col] <= 100)]
    return df


def _filter_to_working_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a working set of FILTER_MIN–FILTER_MAX records while preserving
    the original class distribution (stratified sampling).

    Strategy
    --------
    1. Drop rows with any missing value in WORKING_FEATURES.
    2. If the clean frame already falls inside [480, 650], return it as-is.
    3. If it is larger, draw a stratified sample of FILTER_MAX records.
    4. If it is smaller than FILTER_MIN, raise a descriptive error rather
       than silently proceeding with an under-powered set.
    """
    available = [c for c in WORKING_FEATURES if c in df.columns]
    df_clean  = df.dropna(subset=available).reset_index(drop=True)

    n = len(df_clean)

    if n < FILTER_MIN:
        raise ValueError(
            f"After cleaning, only {n} records remain — below the minimum of "
            f"{FILTER_MIN} needed for statistically valid fuzzy rule coverage. "
            f"Check that your CSV is complete and correctly formatted."
        )

    if FILTER_MIN <= n <= FILTER_MAX:
        print(f"[filter] Dataset has {n} clean records — within target window "
              f"[{FILTER_MIN}, {FILTER_MAX}]. No sampling required.")
        return df_clean

    # Stratified downsample: preserve class proportions
    target   = FILTER_MAX
    df_sampled = (
        df_clean
        .groupby("performance_class", group_keys=False)
        .apply(
            lambda grp: grp.sample(
                n=max(1, round(target * len(grp) / n)),
                random_state=42,
            )
        )
        .reset_index(drop=True)
    )

    # Minor rounding artefacts — trim or pad to hit exactly FILTER_MAX
    if len(df_sampled) > target:
        df_sampled = df_sampled.iloc[:target]

    print(f"[filter] Stratified sample: {n} → {len(df_sampled)} records "
          f"(target ≤ {FILTER_MAX}).")
    return df_sampled


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(path: pathlib.Path = DATA_FILE) -> pd.DataFrame:
    """
    Load, normalise, clean, and filter the StudentPerformance CSV.

    Returns
    -------
    pd.DataFrame
        Working set with WORKING_FEATURES columns plus any demographic
        columns that survived the alias mapping.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            f"Download xAPI-Edu-Data.csv, rename it to StudentPerformance.csv, "
            f"and place it in the '{DATA_DIR}' directory.\n"
            f"Kaggle: https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data"
        )

    print(f"[load]   Reading CSV from: {path}")
    df = pd.read_csv(path)
    print(f"[load]   Raw shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # --- Normalise ---
    df = _normalise_columns(df)

    # --- Encode categorical columns that will feed fuzzy inputs ---
    df = _encode_absence(df)
    df = _encode_binary_text(df, "parent_survey")
    df = _encode_binary_text(df, "parent_satisfaction")

    # --- Range validation ---
    df = _validate_numeric_ranges(df)
    print(f"[clean]  After range validation: {len(df)} rows")

    # --- Filter to working set ---
    df_working = _filter_to_working_set(df)

    return df_working


def summarise(df: pd.DataFrame) -> None:
    """Print a concise statistical summary of the working set."""
    separator = "─" * 60

    print(f"\n{separator}")
    print(f"  Working Set Summary  ({len(df)} records)")
    print(separator)

    print("\n  [Academic Features]")
    academic_cols = [c for c in ACADEMIC_FEATURES if c in df.columns]
    print(df[academic_cols].describe().round(2).to_string())

    print("\n  [Behavioural Features]")
    behav_cols = [c for c in BEHAVIOURAL_FEATURES if c in df.columns]
    print(df[behav_cols].describe().round(2).to_string())

    print(f"\n  [Class Distribution]")
    if "performance_class" in df.columns:
        dist = df["performance_class"].value_counts()
        for label, count in dist.items():
            pct = 100 * count / len(df)
            print(f"    {label:>2} : {count:>4} records  ({pct:.1f}%)")

    print(f"\n  [Missing Values]")
    working_cols = [c for c in WORKING_FEATURES if c in df.columns]
    nulls = df[working_cols].isnull().sum()
    if nulls.sum() == 0:
        print("    No missing values in working feature set.")
    else:
        print(nulls[nulls > 0].to_string())

    print(f"\n{separator}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        working_df = load_dataset()
        summarise(working_df)

        # Persist the filtered working set for downstream pipeline steps
        out_path = DATA_DIR / "working_set.csv"
        working_df.to_csv(out_path, index=False)
        print(f"[save]   Working set saved to: {out_path}")

    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
