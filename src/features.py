"""
features.py
-----------
Feature definitions and utilities for building the model-ready dataset.
"""

import pandas as pd
import numpy as np

# Mapping of column name → human-readable description used throughout the project
FEATURE_CANDIDATES: dict[str, str] = {
    "TEMPERATURE":         "Temperature at assessment (°C)",
    "RELATIVE_HUMIDITY":   "Relative humidity (%)",
    "WIND_SPEED":          "Wind speed (km/h)",
    "FIRE_MONTH":          "Month of ignition",
    "DETECTION_LAG_HRS":   "Detection lag (hours)",
    "DISPATCH_LAG_HRS":    "Dispatch lag (hours)",
    "FIRE_SPREAD_RATE":    "Fire spread rate (m/min)",
    "ASSESSMENT_HECTARES": "Size at assessment (ha)",
    "CAUSE_BINARY":        "Lightning ignition (1=yes)",
    "FUEL_TYPE_ENC":       "Fuel type (encoded)",
    "FOREST_AREA_ENC":     "Forest area (encoded)",
    "FIRE_TYPE_ENC":       "Fire type (encoded)",
}


def get_feature_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return only the feature columns that are present in *df*.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    feature_cols : list[str]
        Column names present in the DataFrame.
    feature_names : list[str]
        Corresponding human-readable descriptions.
    """
    feature_cols = [c for c in FEATURE_CANDIDATES if c in df.columns]
    feature_names = [FEATURE_CANDIDATES[c] for c in feature_cols]
    return feature_cols, feature_names


def build_model_dataset(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Drop rows with any NaN in features or target and return arrays.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame (output of preprocessing.full_pipeline).

    Returns
    -------
    X : np.ndarray  shape (n_samples, n_features)
    y : np.ndarray  shape (n_samples,)
    feature_cols : list[str]
    feature_names : list[str]
    """
    feature_cols, feature_names = get_feature_cols(df)

    if "LARGE_FIRE" not in df.columns:
        raise ValueError(
            "'LARGE_FIRE' column not found. Run preprocessing.engineer_features first."
        )

    model_df = df[feature_cols + ["LARGE_FIRE"]].dropna()
    X = model_df[feature_cols].values
    y = model_df["LARGE_FIRE"].values

    print(f"Modelling dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
    print(f"Large fires: {y.sum():,} ({y.mean() * 100:.1f}%)")

    return X, y, feature_cols, feature_names


def print_descriptive_stats(df: pd.DataFrame) -> None:
    """Print a formatted descriptive statistics table for all model features.

    Parameters
    ----------
    df : pd.DataFrame
    """
    feature_cols, feature_names = get_feature_cols(df)

    print("\n" + "=" * 90)
    print(
        f"{'Variable':<35} {'Min':>8} {'Median':>10} {'Mean':>10} {'IQR':>10} {'Max':>10}"
    )
    print("-" * 90)
    for col, name in zip(feature_cols, feature_names):
        vals = df[col].dropna()
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        print(
            f"{name:<35} {vals.min():>8.2f} {vals.median():>10.2f} "
            f"{vals.mean():>10.2f} {(q3 - q1):>10.2f} {vals.max():>10.2f}"
        )
    print("=" * 90)
