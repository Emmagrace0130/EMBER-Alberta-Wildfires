"""
evaluation.py
-------------
Formatted printing of cross-validation results and final model summaries.
"""

import numpy as np


def print_cv_results(results: dict) -> None:
    """Print a formatted 10-fold CV summary table.

    Parameters
    ----------
    results : dict
        Output from models.run_cross_validation — structure:
        {model_name: {'auprc': [...], 'auroc': [...], 'acc': [...]}}
    """
    print("\n" + "=" * 70)
    print("10-FOLD CROSS-VALIDATION RESULTS (mean ± std)")
    print("PRIMARY METRIC: AUC-Precision-Recall (AUPRC)")
    print("=" * 70)
    print(f"{'Model':<22} {'AUPRC':^20} {'AUROC':^20} {'Accuracy':^15}")
    print("-" * 70)
    for name, res in results.items():
        ap_m = np.mean(res["auprc"]); ap_s = np.std(res["auprc"])
        ar_m = np.mean(res["auroc"]); ar_s = np.std(res["auroc"])
        ac_m = np.mean(res["acc"]);   ac_s = np.std(res["acc"])
        print(
            f"{name:<22} {ap_m:.3f} ± {ap_s:.3f}      "
            f"{ar_m:.3f} ± {ar_s:.3f}    "
            f"{ac_m:.3f} ± {ac_s:.3f}"
        )
    print("=" * 70)


def print_final_summary(
    results: dict,
    iso_ap: float,
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    """Print the final results summary including Isolation Forest.

    Parameters
    ----------
    results : dict
        Cross-validation results dict.
    iso_ap : float
        AUPRC from Isolation Forest (anomaly.run_isolation_forest).
    X : np.ndarray
    y : np.ndarray
    """
    baseline = y.mean()

    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY — LARGE FIRE PREDICTION")
    print("=" * 70)
    print(
        f"Dataset: {X.shape[0]:,} fires | "
        f"Large fires: {y.sum():,} ({y.mean() * 100:.1f}%)"
    )
    print(
        f"Features: {X.shape[1]} | "
        f"CV: 10-fold stratified | "
        f"Oversampling: SMOTE"
    )
    print()
    print(f"{'Model':<22} {'AUPRC (mean±SD)':^22} {'AUROC (mean±SD)':^22}")
    print("-" * 70)
    for name, res in results.items():
        ap = f"{np.mean(res['auprc']):.3f} ± {np.std(res['auprc']):.3f}"
        ar = f"{np.mean(res['auroc']):.3f} ± {np.std(res['auroc']):.3f}"
        print(f"{name:<22} {ap:^22} {ar:^22}")
    print(
        f"{'Isolation Forest':<22} "
        f"{iso_ap:.3f} (unsupervised)".ljust(46)
    )
    print(f"\nRandom baseline AUPRC: {baseline:.3f}")
    print("=" * 70)
