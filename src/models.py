"""
models.py
---------
Model definitions, cross-validation runner, and final model trainer
for the Alberta wildfire large-fire prediction task.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False


def get_models(scale_pos_weight: float = 1.0) -> dict:
    """Return a fresh dict of untrained classifiers.

    Parameters
    ----------
    scale_pos_weight : float
        Ratio of negatives to positives, used by XGBoost.
        Compute as ``(y == 0).sum() / (y == 1).sum()``.

    Returns
    -------
    dict[str, estimator]
    """
    m = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
    }
    if _XGBOOST_AVAILABLE:
        m["XGBoost"] = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
            n_jobs=-1,
        )
    return m


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    models: dict = None,
    n_splits: int = 10,
    random_state: int = 42,
) -> tuple[dict, dict, dict, np.ndarray, np.ndarray]:
    """Run stratified k-fold cross-validation with SMOTE oversampling.

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    models : dict, optional
        Dict of {name: estimator}. Defaults to get_models().
    n_splits : int
        Number of CV folds.
    random_state : int

    Returns
    -------
    results  : dict  {model_name: {'auprc': [], 'auroc': [], 'acc': []}}
    roc_data : dict  {model_name: [interpolated TPR arrays]}
    prc_data : dict  {model_name: [interpolated precision arrays]}
    mean_fpr : np.ndarray  shape (100,)
    mean_rec : np.ndarray  shape (100,)
    """
    if models is None:
        scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
        models = get_models(scale_pos_weight)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scaler = StandardScaler()

    mean_fpr = np.linspace(0, 1, 100)
    mean_rec = np.linspace(0, 1, 100)

    results = {name: {"auprc": [], "auroc": [], "acc": []} for name in models}
    roc_data = {name: [] for name in models}
    prc_data = {name: [] for name in models}

    print(f"Running {n_splits}-fold cross-validation…")

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)

        smote = SMOTE(
            random_state=random_state,
            k_neighbors=min(5, max(1, int(y_train.sum()) - 1)),
        )
        try:
            X_res, y_res = smote.fit_resample(X_train_sc, y_train)
        except Exception:
            X_res, y_res = X_train_sc, y_train

        for name, clf in models.items():
            clf.fit(X_res, y_res)
            proba = clf.predict_proba(X_val_sc)[:, 1]
            pred = clf.predict(X_val_sc)

            # AUPRC
            prec, rec, _ = precision_recall_curve(y_val, proba)
            prc_data[name].append(np.interp(mean_rec, rec[::-1], prec[::-1]))
            results[name]["auprc"].append(average_precision_score(y_val, proba))

            # AUROC
            fpr, tpr, _ = roc_curve(y_val, proba)
            roc_data[name].append(np.interp(mean_fpr, fpr, tpr))
            results[name]["auroc"].append(auc(fpr, tpr))

            # Accuracy
            results[name]["acc"].append((pred == y_val).mean())

        print(f"  Fold {fold}/{n_splits} complete.")

    print("Cross-validation finished.")
    return results, roc_data, prc_data, mean_fpr, mean_rec


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, StandardScaler]:
    """Train the final Random Forest on the full dataset (with SMOTE).

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    random_state : int

    Returns
    -------
    rf_final : RandomForestClassifier  (fitted)
    scaler   : StandardScaler          (fitted on full X)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(
        random_state=random_state,
        k_neighbors=min(5, max(1, int(y.sum()) - 1)),
    )
    try:
        X_res, y_res = smote.fit_resample(X_scaled, y)
    except Exception:
        X_res, y_res = X_scaled, y

    rf_final = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    rf_final.fit(X_res, y_res)
    print("Final Random Forest model trained.")
    return rf_final, scaler
