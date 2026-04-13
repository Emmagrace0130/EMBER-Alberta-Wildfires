"""
risk_model.py
─────────────
SHAP-weighted large-fire escalation risk scorer for Alberta's Forest
Protection Area.

Feature weights are derived from SHAP analysis of four models trained on
26,551 wildfire incidents (Alberta FPA, 2006–2024):
  • Random Forest  — AUROC 0.957 ± 0.008  (best overall discrimination)
  • Logistic Regression — AUPRC 0.584 ± 0.041  (best precision, large fires)
  • XGBoost        — AUPRC 0.567 ± 0.042
  • Isolation Forest — AUPRC 0.264  (unsupervised, 8× above random baseline)
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Mean absolute SHAP values (normalized to sum = 1.0)
# derived from the Random Forest + LR + XGBoost ensemble
# ─────────────────────────────────────────────────────────────────────────────
SHAP_IMPORTANCE: Dict[str, float] = {
    "fire_size":      0.342,
    "spread_rate":    0.251,
    "wind_speed":     0.148,
    "temperature":    0.103,
    "humidity":       0.089,
    "forest_region":  0.041,
    "ignition_cause": 0.026,
}

# ─────────────────────────────────────────────────────────────────────────────
# Categorical encodings
# ─────────────────────────────────────────────────────────────────────────────
SPREAD_RATE_SCORES: Dict[str, float] = {
    "negligible": 0.00,
    "slow":       0.25,
    "moderate":   0.50,
    "fast":       0.75,
    "extreme":    1.00,
}

IGNITION_CAUSE_SCORES: Dict[str, float] = {
    "lightning":   0.75,   # remote, harder to access/suppress
    "campfire":    0.50,
    "equipment":   0.45,
    "arson":       0.60,
    "prescribed":  0.20,   # planned, pre-scouted
    "unknown":     0.55,
}

FOREST_REGION_SCORES: Dict[str, float] = {
    "fort_mcmurray":        0.90,
    "high_level":           0.85,
    "peace_river":          0.80,
    "lac_la_biche":         0.75,
    "slave_lake":           0.70,
    "grande_prairie":       0.65,
    "whitecourt":           0.60,
    "edson":                0.55,
    "rocky_mountain_house": 0.50,
    "calgary":              0.40,
}

# Default score for an unrecognised forest region
DEFAULT_FOREST_REGION_SCORE: float = 0.60

FEATURE_LABELS: Dict[str, str] = {
    "fire_size":      "Initial fire size",
    "spread_rate":    "Fire spread rate",
    "wind_speed":     "Wind speed",
    "temperature":    "Air temperature",
    "humidity":       "Relative humidity",
    "forest_region":  "Forest region",
    "ignition_cause": "Ignition cause",
}

# Human-readable display labels for dropdowns
SPREAD_RATE_OPTIONS: List[Dict[str, str]] = [
    {"value": "negligible", "label": "Negligible"},
    {"value": "slow",       "label": "Slow"},
    {"value": "moderate",   "label": "Moderate"},
    {"value": "fast",       "label": "Fast"},
    {"value": "extreme",    "label": "Extreme"},
]

IGNITION_CAUSE_OPTIONS: List[Dict[str, str]] = [
    {"value": "lightning",  "label": "Lightning"},
    {"value": "campfire",   "label": "Human — Campfire"},
    {"value": "equipment",  "label": "Human — Equipment"},
    {"value": "arson",      "label": "Human — Arson"},
    {"value": "prescribed", "label": "Prescribed Burn"},
    {"value": "unknown",    "label": "Unknown"},
]

FOREST_REGION_OPTIONS: List[Dict[str, str]] = [
    {"value": "fort_mcmurray",        "label": "Fort McMurray"},
    {"value": "high_level",           "label": "High Level"},
    {"value": "peace_river",          "label": "Peace River"},
    {"value": "lac_la_biche",         "label": "Lac La Biche"},
    {"value": "slave_lake",           "label": "Slave Lake"},
    {"value": "grande_prairie",       "label": "Grande Prairie"},
    {"value": "whitecourt",           "label": "Whitecourt"},
    {"value": "edson",                "label": "Edson"},
    {"value": "rocky_mountain_house", "label": "Rocky Mountain House"},
    {"value": "calgary",              "label": "Calgary"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Feature normalizers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_fire_size(ha: float) -> float:
    """
    Log-scale normalization for initial fire size.
    0.01 ha → ~0.01  |  1 ha → ~0.11  |  10 ha → ~0.39  |  100 ha → ~0.74
    Reference ceiling: 500 ha (fires above this are already 'large').
    """
    if ha <= 0:
        return 0.0
    return min(math.log1p(ha) / math.log1p(500.0), 1.0)


def _normalize_wind_speed(kmh: float) -> float:
    """Linear 0–100 km/h. Values above 100 are clipped to 1.0."""
    return min(max(kmh / 100.0, 0.0), 1.0)


def _normalize_temperature(celsius: float) -> float:
    """Operational range 5–40 °C; below 5 → 0, above 40 → 1."""
    return min(max((celsius - 5.0) / 35.0, 0.0), 1.0)


def _normalize_humidity(pct: float) -> float:
    """
    Inverse relationship: lower humidity → higher risk.
    0 % humidity → score 1.0 (maximum risk contribution).
    """
    return min(max(1.0 - pct / 100.0, 0.0), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Risk computation
# ─────────────────────────────────────────────────────────────────────────────

# Sigmoid calibration constants
# Chosen so that:
#   typical fire (0.5 ha, slow spread, calm, 20 °C, 50 % RH) → ~5 % risk
#   extreme fire (50 ha, extreme spread, 70 km/h, 35 °C, 15 % RH) → ~82 % risk
_SIGMOID_SLOPE = 8.0
_SIGMOID_CENTER = 0.617


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_risk_score(
    fire_size: float,
    spread_rate: str,
    wind_speed: float,
    temperature: float,
    humidity: float,
    ignition_cause: str,
    forest_region: str,
) -> Tuple[float, List[Dict]]:
    """
    Compute large-fire escalation risk using SHAP-weighted feature scoring.

    Parameters
    ----------
    fire_size       : Initial assessed fire area in hectares.
    spread_rate     : One of 'negligible', 'slow', 'moderate', 'fast', 'extreme'.
    wind_speed      : Wind speed at the fire in km/h.
    temperature     : Air temperature in °C.
    humidity        : Relative humidity in %.
    ignition_cause  : e.g. 'lightning', 'campfire', 'equipment', 'arson',
                      'prescribed', 'unknown'.
    forest_region   : Alberta Forest Protection Area region key.

    Returns
    -------
    risk_score : float in [0, 1] — probability of large-fire escalation.
    factors    : list of dicts sorted by contribution magnitude, each with:
                   name, contribution (% of linear score), normalized (0–1),
                   weight (SHAP importance), direction ('increases'/'decreases').
    """
    # --- Normalise each input to [0, 1] with direction: high → higher risk ---
    normed: Dict[str, float] = {
        "fire_size":      _normalize_fire_size(fire_size),
        "spread_rate":    SPREAD_RATE_SCORES.get(spread_rate.lower(), 0.50),
        "wind_speed":     _normalize_wind_speed(wind_speed),
        "temperature":    _normalize_temperature(temperature),
        "humidity":       _normalize_humidity(humidity),
        "ignition_cause": IGNITION_CAUSE_SCORES.get(ignition_cause.lower(), 0.55),
        "forest_region":  FOREST_REGION_SCORES.get(
            forest_region.lower().replace(" ", "_"), DEFAULT_FOREST_REGION_SCORE
        ),
    }

    # --- SHAP-weighted linear combination ---
    contributions: Dict[str, float] = {
        feat: SHAP_IMPORTANCE[feat] * normed[feat]
        for feat in SHAP_IMPORTANCE
    }
    linear_score: float = sum(contributions.values())

    # --- Sigmoid calibration → probability ---
    risk_score: float = _sigmoid(
        _SIGMOID_SLOPE * (linear_score - _SIGMOID_CENTER)
    )

    # --- Build sorted factor list ---
    safe_linear = linear_score if linear_score > 0 else 1e-9
    factors = sorted(
        [
            {
                "name":         FEATURE_LABELS[feat],
                "contribution": round(contributions[feat] / safe_linear * 100, 1),
                "normalized":   round(normed[feat], 3),
                "weight":       round(SHAP_IMPORTANCE[feat], 3),
                "direction":    "decreases" if feat == "humidity" else "increases",
            }
            for feat in SHAP_IMPORTANCE
        ],
        key=lambda x: x["contribution"],
        reverse=True,
    )

    return round(risk_score, 4), factors


def get_risk_level(score: float) -> str:
    """Return categorical risk label for a given probability score."""
    if score < 0.20:
        return "LOW"
    if score < 0.45:
        return "MODERATE"
    if score < 0.70:
        return "HIGH"
    return "EXTREME"
