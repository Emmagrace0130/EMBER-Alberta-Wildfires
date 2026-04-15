"""
run_pipeline.py
---------------
Standalone script that replicates the full best_try.ipynb pipeline.

Usage
-----
    python run_pipeline.py --data fp-historical-wildfire-data-2006-2024.xlsx

Optional flags
--------------
    --output-dir  Directory to save figures and model (default: current dir)
    --no-shap     Skip SHAP computation (faster)
    --folds       Number of CV folds (default: 10)
"""

import argparse
import os
import warnings
import joblib

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Alberta Wildfire large-fire prediction pipeline.")
    parser.add_argument("--data", required=True, help="Path to the Excel data file.")
    parser.add_argument("--output-dir", default=".", help="Directory to save figures and model.")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP computation.")
    parser.add_argument("--folds", type=int, default=10, help="Number of CV folds.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    def out(filename):
        """Build an output path inside --output-dir."""
        return os.path.join(args.output_dir, filename)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1 — Loading data")
    print("=" * 60)
    from src.data_loader import load_data
    df = load_data(args.data)

    # ------------------------------------------------------------------
    # 2. Preprocessing
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2 — Preprocessing")
    print("=" * 60)
    from src.preprocessing import full_pipeline
    df, encoders = full_pipeline(df)

    # ------------------------------------------------------------------
    # 3. Descriptive statistics & feature selection
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3 — Feature statistics")
    print("=" * 60)
    from src.features import print_descriptive_stats, build_model_dataset
    print_descriptive_stats(df)

    # ------------------------------------------------------------------
    # 4. Build model dataset
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4 — Building model dataset")
    print("=" * 60)
    X, y, feature_cols, feature_names = build_model_dataset(df)

    # ------------------------------------------------------------------
    # 5. Cross-validation (RF, LR, XGBoost)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"STEP 5 — {args.folds}-fold cross-validation")
    print("=" * 60)
    from src.models import get_models, run_cross_validation
    import numpy as np

    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    models = get_models(scale_pos_weight)

    results, roc_data, prc_data, mean_fpr, mean_rec = run_cross_validation(
        X, y, models=models, n_splits=args.folds
    )

    from src.evaluation import print_cv_results
    print_cv_results(results)

    # ------------------------------------------------------------------
    # 6. Isolation Forest (unsupervised)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6 — Isolation Forest anomaly detection")
    print("=" * 60)
    from src.anomaly import run_isolation_forest
    from sklearn.preprocessing import StandardScaler

    scaler_iso = StandardScaler()
    X_scaled_full = scaler_iso.fit_transform(X)
    iso_ap, iso_pred, iso_scores = run_isolation_forest(X_scaled_full, y)

    # ------------------------------------------------------------------
    # 7. Final summary
    # ------------------------------------------------------------------
    from src.evaluation import print_final_summary
    print_final_summary(results, iso_ap=iso_ap, X=X, y=y)

    # ------------------------------------------------------------------
    # 8. Figures
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7 — Generating figures")
    print("=" * 60)

    import matplotlib
    matplotlib.use("Agg")

    from src.visualization import (
        plot_prc_curves,
        plot_roc_curves,
        plot_model_comparison,
        plot_size_distribution,
        plot_annual_trends,
    )

    baseline = float(y.mean())

    plot_prc_curves(
        prc_data, results, mean_rec, baseline=baseline,
        save=True, filename=out("fig_prc_curves.png"),
    )
    plot_roc_curves(
        roc_data, results, mean_fpr,
        save=True, filename=out("fig_roc_curves.png"),
    )
    plot_model_comparison(
        results, iso_ap=iso_ap, baseline=baseline,
        save=True, filename=out("fig_model_comparison.png"),
    )
    plot_size_distribution(df, save=True, filename=out("fig_size_distribution.png"))
    plot_annual_trends(df, save=True, filename=out("fig_annual_trends.png"))

    # ------------------------------------------------------------------
    # 9. Train final model
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 8 — Training final Random Forest")
    print("=" * 60)
    from src.models import train_final_model
    rf_final, scaler_final = train_final_model(X, y)

    # ------------------------------------------------------------------
    # 10. SHAP analysis
    # ------------------------------------------------------------------
    if not args.no_shap:
        print("\n" + "=" * 60)
        print("STEP 9 — SHAP feature importance")
        print("=" * 60)
        from src.shap_analysis import compute_shap
        from src.visualization import plot_shap_bar

        X_scaled = scaler_final.transform(X)
        sv, mean_shap, sorted_idx = compute_shap(rf_final, X_scaled, feature_names)
        plot_shap_bar(mean_shap, feature_names, save=True, filename=out("fig_shap.png"))
    else:
        print("\nSHAP skipped (--no-shap).")

    # ------------------------------------------------------------------
    # 11. Save models + scaler
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 10 — Saving models and scaler")
    print("=" * 60)

    # Random Forest (final)
    model_path = out("wildfire_model.pkl")
    joblib.dump(rf_final, model_path)
    print(f"Random Forest saved : {model_path}")

    # Scaler (MUST be saved alongside model for inference)
    scaler_path = out("scaler.pkl")
    joblib.dump(scaler_final, scaler_path)
    print(f"Scaler saved        : {scaler_path}")

    # Logistic Regression (best AUPRC in CV — train on same scaled + SMOTE data)
    from sklearn.linear_model import LogisticRegression
    from imblearn.over_sampling import SMOTE as _SMOTE

    X_scaled_lr = scaler_final.transform(X)
    smote_lr = _SMOTE(random_state=42, k_neighbors=min(5, max(1, int(y.sum()) - 1)))
    try:
        X_res_lr, y_res_lr = smote_lr.fit_resample(X_scaled_lr, y)
    except Exception:
        X_res_lr, y_res_lr = X_scaled_lr, y
    lr_final = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr_final.fit(X_res_lr, y_res_lr)
    lr_path = out("lr_model.pkl")
    joblib.dump(lr_final, lr_path)
    print(f"Logistic Regression : {lr_path}")

    print("\n✓ Pipeline complete. Outputs written to:", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
