"""
preprocessing.py
----------------
Date parsing, feature engineering, null imputation, and categorical encoding
for the Alberta wildfire dataset.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATE_COLS = [
    "FIRE_START_DATE",
    "DISCOVERED_DATE",
    "REPORTED_DATE",
    "DISPATCH_DATE",
    "FIRST_UC_DATE",
    "FIRST_EX_DATE",
]

NUMERIC_COLS = [
    "TEMPERATURE",
    "RELATIVE_HUMIDITY",
    "WIND_SPEED",
    "DETECTION_LAG_HRS",
    "DISPATCH_LAG_HRS",
    "FIRE_SPREAD_RATE",
    "ASSESSMENT_HECTARES",
]

CAT_COLS = ["FUEL_TYPE", "FIRE_TYPE", "GENERAL_CAUSE", "FOREST_AREA"]


def parse_dates(df: pd.DataFrame, date_cols: list = None) -> pd.DataFrame:
    """Convert date columns to datetime.

    Parameters
    ----------
    df : pd.DataFrame
    date_cols : list, optional
        Columns to parse. Defaults to DATE_COLS.

    Returns
    -------
    pd.DataFrame
    """
    if date_cols is None:
        date_cols = DATE_COLS
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived columns from raw data.

    Derived columns
    ---------------
    FIRE_MONTH              : month of ignition
    DETECTION_LAG_HRS       : hours from start to discovery (clipped 0–72)
    DISPATCH_LAG_HRS        : hours from discovery to dispatch (clipped 0–48)
    SUPPRESSION_DURATION_HRS: hours from start to under control (clipped 0–2000)
    FOREST_AREA             : first letter of FIRE_NUMBER (forest area code)
    LARGE_FIRE              : 1 if SIZE_CLASS in {D, E}, else 0
    CAUSE_BINARY            : 1 if GENERAL_CAUSE == 'Lightning', else 0

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    if "FIRE_START_DATE" in df.columns:
        df["FIRE_MONTH"] = df["FIRE_START_DATE"].dt.month

    if "DISCOVERED_DATE" in df.columns and "FIRE_START_DATE" in df.columns:
        df["DETECTION_LAG_HRS"] = (
            (df["DISCOVERED_DATE"] - df["FIRE_START_DATE"]).dt.total_seconds() / 3600
        ).clip(0, 72)

    if "DISPATCH_DATE" in df.columns and "DISCOVERED_DATE" in df.columns:
        df["DISPATCH_LAG_HRS"] = (
            (df["DISPATCH_DATE"] - df["DISCOVERED_DATE"]).dt.total_seconds() / 3600
        ).clip(0, 48)

    if "FIRST_UC_DATE" in df.columns and "FIRE_START_DATE" in df.columns:
        df["SUPPRESSION_DURATION_HRS"] = (
            (df["FIRST_UC_DATE"] - df["FIRE_START_DATE"]).dt.total_seconds() / 3600
        ).clip(0, 2000)

    if "FIRE_NUMBER" in df.columns:
        df["FOREST_AREA"] = df["FIRE_NUMBER"].str[0].str.upper()

    if "SIZE_CLASS" in df.columns:
        df["LARGE_FIRE"] = df["SIZE_CLASS"].isin(["D", "E"]).astype(int)
        large = df["LARGE_FIRE"].sum()
        print(f"Large fire breakdown:")
        print(f"  Total fires:            {len(df):,}")
        print(f"  Large fires (D/E):      {large:,} ({large / len(df) * 100:.1f}%)")
        print(f"  Small fires (A/B/C):    {len(df) - large:,} ({(len(df) - large) / len(df) * 100:.1f}%)")

    if "GENERAL_CAUSE" in df.columns:
        df["CAUSE_BINARY"] = (df["GENERAL_CAUSE"] == "Lightning").astype(int)

    return df


def impute_missing(
    df: pd.DataFrame,
    numeric_cols: list = None,
    cat_cols: list = None,
) -> pd.DataFrame:
    """Fill missing values: mean for numeric, mode for categorical.

    Parameters
    ----------
    df : pd.DataFrame
    numeric_cols : list, optional
    cat_cols : list, optional

    Returns
    -------
    pd.DataFrame
    """
    if numeric_cols is None:
        numeric_cols = NUMERIC_COLS
    if cat_cols is None:
        cat_cols = CAT_COLS

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    print("Null handling complete (mean for numeric, mode for categorical).")
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label-encode FUEL_TYPE, FOREST_AREA, and FIRE_TYPE into new *_ENC columns.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    df : pd.DataFrame
        DataFrame with added *_ENC columns.
    encoders : dict
        Mapping of column name → fitted LabelEncoder.
    """
    encoders: dict = {}
    for col in ["FUEL_TYPE", "FOREST_AREA", "FIRE_TYPE"]:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_ENC"] = le.fit_transform(df[col].fillna("Unknown"))
            encoders[col] = le
    return df, encoders


def full_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run all preprocessing steps in order.

    Steps: parse_dates → engineer_features → impute_missing → encode_categoricals

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    df : pd.DataFrame
    encoders : dict
    """
    df = parse_dates(df)
    df = engineer_features(df)
    df = impute_missing(df)
    df, encoders = encode_categoricals(df)
    print("Preprocessing complete.")
    return df, encoders
