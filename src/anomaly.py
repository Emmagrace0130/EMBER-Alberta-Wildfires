"""
anomaly.py
----------
Isolation Forest anomaly detection for identifying large-fire events
as unsupervised outliers.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)


def run_isolation_forest(
    X_scaled: np.ndarray,
    y: np.ndarray,
    contamination: float = None,
    n_estimators: int = 200,
    random_state: int = 42,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit an Isolation Forest and evaluate it against the large-fire labels.

    Large fires are treated as anomalies.  The contamination parameter is set
    to the observed large-fire rate by default so the model's anomaly budget
    matches the true prevalence.

    Parameters
    ----------
    X_scaled      : np.ndarray  Scaled feature matrix (full dataset).
    y             : np.ndarray  Binary target (1 = large fire).
    contamination : float, optional  Defaults to y.mean().
    n_estimators  : int
    random_state  : int

    Returns
    -------
    iso_ap    : float        AUPRC score.
    iso_pred  : np.ndarray   Binary predictions (1 = predicted large fire).
    iso_scores: np.ndarray   Anomaly scores (higher = more anomalous).
    """
    if contamination is None:
        contamination = float(y.mean())

    print("Running Isolation Forest anomaly detection…")
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(X_scaled)

    # Flip sign: higher score → more anomalous → more likely large fire
    iso_scores = -iso.score_samples(X_scaled)
    iso_pred = (iso.predict(X_scaled) == -1).astype(int)

    iso_ap = average_precision_score(y, iso_scores)
    iso_prec = precision_score(y, iso_pred, zero_division=0)
    iso_rec = recall_score(y, iso_pred, zero_division=0)
    iso_f1 = f1_score(y, iso_pred, zero_division=0)

    print(f"\nIsolation Forest Results (unsupervised anomaly detection):")
    print(f"  AUPRC:     {iso_ap:.3f}")
    print(f"  Precision: {iso_prec:.3f}")
    print(f"  Recall:    {iso_rec:.3f}")
    print(f"  F1 Score:  {iso_f1:.3f}")
    print(
        f"\nNote: Isolation Forest is unsupervised — it learns what is 'unusual'\n"
        f"without being told which fires are large. AUPRC of {iso_ap:.3f} vs\n"
        f"random baseline of {contamination:.3f}."
    )

    return iso_ap, iso_pred, iso_scores
