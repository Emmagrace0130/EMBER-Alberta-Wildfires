"""
shap_analysis.py
----------------
SHAP value computation and feature importance extraction
for the final Random Forest model.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


def compute_shap(
    rf_final: RandomForestClassifier,
    X_scaled: np.ndarray,
    feature_names: list,
    sample_size: int = 500,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute SHAP values for the positive class (large fire).

    Parameters
    ----------
    rf_final      : fitted RandomForestClassifier
    X_scaled      : np.ndarray  Full scaled feature matrix.
    feature_names : list[str]
    sample_size   : int   Number of samples to use for SHAP (for speed).
    random_state  : int

    Returns
    -------
    sv         : np.ndarray  shape (sample_size, n_features)
                 SHAP values for the positive class.
    mean_shap  : np.ndarray  shape (n_features,)  Mean absolute SHAP.
    sorted_idx : np.ndarray  Indices that sort mean_shap ascending.
    """
    if not _SHAP_AVAILABLE:
        raise ImportError("shap is not installed. Run: pip install shap")

    n = min(sample_size, X_scaled.shape[0])
    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(X_scaled.shape[0], n, replace=False)
    X_sample = X_scaled[sample_idx]

    print(f"Computing SHAP values on {n} samples…")
    explainer = _shap.TreeExplainer(rf_final)
    shap_values = explainer.shap_values(X_sample)

    # Handle list (one array per class) or 3-D array formats
    if isinstance(shap_values, list):
        sv = shap_values[1]          # positive class
    else:
        sv = shap_values

    if sv.ndim == 3 and sv.shape[2] > 1:
        sv = sv[:, :, 1]             # positive class from 3-D array

    mean_shap = np.abs(sv).mean(axis=0)
    sorted_idx = np.argsort(mean_shap)

    print("SHAP computation complete.")
    return sv, mean_shap, sorted_idx
