"""
tests/test_pipeline.py
----------------------
Tests that mirror the structure of best_try.ipynb, section by section.

All tests use a small synthetic DataFrame so no real Excel file is needed.
Run with:  pytest tests/test_pipeline.py -v
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Synthetic dataset fixture
# ---------------------------------------------------------------------------

N = 200  # number of synthetic fire records


def make_synthetic_df(n: int = N, seed: int = 42) -> pd.DataFrame:
    """Build a small synthetic wildfire DataFrame that mirrors the real schema."""
    rng = np.random.default_rng(seed)

    start_dates = [datetime(2010, 1, 1) + timedelta(days=int(d))
                   for d in rng.integers(0, 365 * 14, n)]

    df = pd.DataFrame({
        "FIRE_NUMBER":        [f"{chr(65 + i % 5)}{i:04d}" for i in range(n)],
        "FIRE_START_DATE":    start_dates,
        "DISCOVERED_DATE":    [d + timedelta(hours=float(h))
                               for d, h in zip(start_dates, rng.uniform(0, 24, n))],
        "DISPATCH_DATE":      [d + timedelta(hours=float(h))
                               for d, h in zip(start_dates, rng.uniform(1, 30, n))],
        "FIRST_UC_DATE":      [d + timedelta(hours=float(h))
                               for d, h in zip(start_dates, rng.uniform(10, 500, n))],
        "FIRST_EX_DATE":      [d + timedelta(hours=float(h))
                               for d, h in zip(start_dates, rng.uniform(20, 1000, n))],
        "TEMPERATURE":        rng.uniform(5, 40, n),
        "RELATIVE_HUMIDITY":  rng.uniform(10, 90, n),
        "WIND_SPEED":         rng.uniform(0, 60, n),
        "FIRE_SPREAD_RATE":   rng.uniform(0, 200, n),
        "ASSESSMENT_HECTARES":rng.uniform(0.01, 500, n),
        "SIZE_CLASS":         rng.choice(["A", "B", "C", "D", "E"],
                                         n, p=[0.40, 0.35, 0.15, 0.06, 0.04]),
        "GENERAL_CAUSE":      rng.choice(["Lightning", "Human", "Unknown"], n,
                                         p=[0.5, 0.4, 0.1]),
        "FUEL_TYPE":          rng.choice(["C-3", "C-2", "M-1", "D-1", "S-1"], n),
        "FIRE_TYPE":          rng.choice(["Surface", "Crown", "Ground"], n),
        "YEAR":               [d.year for d in start_dates],
        "CURRENT_SIZE":       rng.uniform(0.01, 1000, n),
    })

    # Sprinkle a few NaNs to test imputation
    for col in ["TEMPERATURE", "RELATIVE_HUMIDITY", "WIND_SPEED",
                "FIRE_SPREAD_RATE", "ASSESSMENT_HECTARES"]:
        idx = rng.choice(n, size=10, replace=False)
        df.loc[idx, col] = np.nan
    for col in ["FUEL_TYPE", "FIRE_TYPE", "GENERAL_CAUSE"]:
        idx = rng.choice(n, size=5, replace=False)
        df.loc[idx, col] = np.nan

    return df


@pytest.fixture(scope="module")
def raw_df():
    return make_synthetic_df()


# ===========================================================================
# Section 1 — Imports (cells 1-6 in the notebook)
# Just verify all src modules are importable.
# ===========================================================================

class TestImports:
    def test_src_package_importable(self):
        import src  # noqa: F401

    def test_data_loader_importable(self):
        from src.data_loader import load_data  # noqa: F401

    def test_preprocessing_importable(self):
        from src.preprocessing import (  # noqa: F401
            parse_dates, engineer_features,
            impute_missing, encode_categoricals, full_pipeline,
        )

    def test_features_importable(self):
        from src.features import (  # noqa: F401
            FEATURE_CANDIDATES, get_feature_cols,
            build_model_dataset, print_descriptive_stats,
        )

    def test_models_importable(self):
        from src.models import (  # noqa: F401
            get_models, run_cross_validation, train_final_model,
        )

    def test_evaluation_importable(self):
        from src.evaluation import print_cv_results, print_final_summary  # noqa: F401

    def test_visualization_importable(self):
        from src.visualization import (  # noqa: F401
            plot_roc_curves, plot_prc_curves, plot_model_comparison,
            plot_shap_bar, plot_size_distribution, plot_annual_trends,
        )

    def test_shap_analysis_importable(self):
        from src.shap_analysis import compute_shap  # noqa: F401

    def test_anomaly_importable(self):
        from src.anomaly import run_isolation_forest  # noqa: F401


# ===========================================================================
# Section 2 — Data loading (cells 7-9)
# ===========================================================================

class TestDataLoading:
    def test_synthetic_df_has_expected_shape(self, raw_df):
        assert raw_df.shape[0] == N
        assert raw_df.shape[1] > 0

    def test_required_columns_present(self, raw_df):
        required = [
            "FIRE_NUMBER", "FIRE_START_DATE", "DISCOVERED_DATE",
            "SIZE_CLASS", "GENERAL_CAUSE", "FUEL_TYPE",
        ]
        for col in required:
            assert col in raw_df.columns, f"Missing column: {col}"


# ===========================================================================
# Section 3 — Date parsing (cells 10-11)
# ===========================================================================

class TestDateParsing:
    def test_parse_dates_converts_columns(self, raw_df):
        from src.preprocessing import parse_dates
        df = raw_df.copy()
        # Convert to strings first to simulate loading from Excel
        for col in ["FIRE_START_DATE", "DISCOVERED_DATE"]:
            df[col] = df[col].astype(str)
        df = parse_dates(df)
        assert pd.api.types.is_datetime64_any_dtype(df["FIRE_START_DATE"])
        assert pd.api.types.is_datetime64_any_dtype(df["DISCOVERED_DATE"])

    def test_parse_dates_handles_invalid_dates(self):
        from src.preprocessing import parse_dates
        df = pd.DataFrame({"FIRE_START_DATE": ["not-a-date", "2020-01-01"]})
        df = parse_dates(df, date_cols=["FIRE_START_DATE"])
        assert pd.isna(df.loc[0, "FIRE_START_DATE"])
        assert pd.notna(df.loc[1, "FIRE_START_DATE"])


# ===========================================================================
# Section 4 — Feature engineering (cells 12-18)
# ===========================================================================

class TestFeatureEngineering:
    @pytest.fixture(scope="class")
    def engineered_df(self, raw_df):
        from src.preprocessing import engineer_features
        return engineer_features(raw_df.copy())

    def test_fire_month_created(self, engineered_df):
        assert "FIRE_MONTH" in engineered_df.columns
        assert engineered_df["FIRE_MONTH"].between(1, 12).all()

    def test_detection_lag_created_and_clipped(self, engineered_df):
        assert "DETECTION_LAG_HRS" in engineered_df.columns
        assert (engineered_df["DETECTION_LAG_HRS"] >= 0).all()
        assert (engineered_df["DETECTION_LAG_HRS"] <= 72).all()

    def test_dispatch_lag_created_and_clipped(self, engineered_df):
        assert "DISPATCH_LAG_HRS" in engineered_df.columns
        assert (engineered_df["DISPATCH_LAG_HRS"] >= 0).all()
        assert (engineered_df["DISPATCH_LAG_HRS"] <= 48).all()

    def test_forest_area_extracted(self, engineered_df):
        assert "FOREST_AREA" in engineered_df.columns
        assert engineered_df["FOREST_AREA"].str.len().max() == 1

    def test_large_fire_binary_target(self, engineered_df):
        assert "LARGE_FIRE" in engineered_df.columns
        assert set(engineered_df["LARGE_FIRE"].unique()).issubset({0, 1})
        large = engineered_df.loc[engineered_df["SIZE_CLASS"].isin(["D", "E"]), "LARGE_FIRE"]
        assert (large == 1).all()

    def test_cause_binary(self, engineered_df):
        assert "CAUSE_BINARY" in engineered_df.columns
        lightning = engineered_df.loc[engineered_df["GENERAL_CAUSE"] == "Lightning", "CAUSE_BINARY"]
        assert (lightning == 1).all()


# ===========================================================================
# Section 5 — Null imputation (cells 19-24)
# ===========================================================================

class TestImputation:
    def test_no_nulls_in_numeric_cols_after_impute(self, raw_df):
        from src.preprocessing import engineer_features, impute_missing
        df = engineer_features(raw_df.copy())
        df = impute_missing(df)
        for col in ["TEMPERATURE", "RELATIVE_HUMIDITY", "WIND_SPEED"]:
            if col in df.columns:
                assert df[col].isnull().sum() == 0, f"{col} still has nulls"

    def test_no_nulls_in_cat_cols_after_impute(self, raw_df):
        from src.preprocessing import engineer_features, impute_missing
        df = engineer_features(raw_df.copy())
        df = impute_missing(df)
        for col in ["FUEL_TYPE", "FIRE_TYPE"]:
            if col in df.columns:
                assert df[col].isnull().sum() == 0, f"{col} still has nulls"


# ===========================================================================
# Section 6 — Feature selection & encoding (cells 25-30)
# ===========================================================================

class TestFeaturesAndEncoding:
    @pytest.fixture(scope="class")
    def processed_df(self, raw_df):
        from src.preprocessing import full_pipeline
        df, _ = full_pipeline(raw_df.copy())
        return df

    def test_encoded_columns_created(self, processed_df):
        for col in ["FUEL_TYPE_ENC", "FOREST_AREA_ENC", "FIRE_TYPE_ENC"]:
            assert col in processed_df.columns

    def test_encoded_columns_are_integer(self, processed_df):
        for col in ["FUEL_TYPE_ENC", "FOREST_AREA_ENC", "FIRE_TYPE_ENC"]:
            assert pd.api.types.is_integer_dtype(processed_df[col])

    def test_get_feature_cols_returns_subset(self, processed_df):
        from src.features import get_feature_cols, FEATURE_CANDIDATES
        feature_cols, feature_names = get_feature_cols(processed_df)
        assert len(feature_cols) > 0
        assert len(feature_cols) == len(feature_names)
        for col in feature_cols:
            assert col in FEATURE_CANDIDATES


# ===========================================================================
# Section 7 — Build model dataset (cells 31-32)
# ===========================================================================

class TestBuildModelDataset:
    @pytest.fixture(scope="class")
    def model_data(self, raw_df):
        from src.preprocessing import full_pipeline
        from src.features import build_model_dataset
        df, _ = full_pipeline(raw_df.copy())
        return build_model_dataset(df)

    def test_X_is_2d_array(self, model_data):
        X, y, _, _ = model_data
        assert X.ndim == 2

    def test_y_is_binary(self, model_data):
        X, y, _, _ = model_data
        assert set(np.unique(y)).issubset({0, 1})

    def test_no_nans_in_X(self, model_data):
        X, y, _, _ = model_data
        assert not np.isnan(X).any()

    def test_X_y_same_length(self, model_data):
        X, y, _, _ = model_data
        assert X.shape[0] == y.shape[0]


# ===========================================================================
# Section 8 — Cross-validation (cells 33-45)
# ===========================================================================

class TestCrossValidation:
    @pytest.fixture(scope="class")
    def cv_outputs(self, raw_df):
        from src.preprocessing import full_pipeline
        from src.features import build_model_dataset
        from src.models import get_models, run_cross_validation

        df, _ = full_pipeline(raw_df.copy())
        X, y, _, _ = build_model_dataset(df)
        scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
        models = get_models(scale_pos_weight)
        # Use only RF + LR and 3 folds to keep tests fast
        models = {k: v for k, v in models.items()
                  if k in ("Random Forest", "Logistic Regression")}
        return run_cross_validation(X, y, models=models, n_splits=3)

    def test_results_have_all_models(self, cv_outputs):
        results, *_ = cv_outputs
        assert "Random Forest" in results
        assert "Logistic Regression" in results

    def test_auprc_lists_have_correct_length(self, cv_outputs):
        results, *_ = cv_outputs
        for name, res in results.items():
            assert len(res["auprc"]) == 3, f"{name}: expected 3 folds"

    def test_auprc_scores_in_valid_range(self, cv_outputs):
        results, *_ = cv_outputs
        for name, res in results.items():
            for score in res["auprc"]:
                assert 0.0 <= score <= 1.0

    def test_roc_data_arrays_shape(self, cv_outputs):
        _, roc_data, _, mean_fpr, _ = cv_outputs
        for name, arrays in roc_data.items():
            for arr in arrays:
                assert arr.shape == mean_fpr.shape


# ===========================================================================
# Section 9 — Final model training (cells 60, 76)
# ===========================================================================

class TestFinalModel:
    @pytest.fixture(scope="class")
    def final_model(self, raw_df):
        from src.preprocessing import full_pipeline
        from src.features import build_model_dataset
        from src.models import train_final_model
        df, _ = full_pipeline(raw_df.copy())
        X, y, _, _ = build_model_dataset(df)
        return train_final_model(X, y), X, y

    def test_model_is_fitted(self, final_model):
        (rf, scaler), X, y = final_model
        # predict should not raise NotFittedError
        X_scaled = scaler.transform(X[:5])
        preds = rf.predict(X_scaled)
        assert preds.shape == (5,)

    def test_predict_proba_shape(self, final_model):
        (rf, scaler), X, y = final_model
        X_scaled = scaler.transform(X[:10])
        proba = rf.predict_proba(X_scaled)
        assert proba.shape == (10, 2)

    def test_proba_sums_to_one(self, final_model):
        (rf, scaler), X, y = final_model
        X_scaled = scaler.transform(X[:10])
        proba = rf.predict_proba(X_scaled)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ===========================================================================
# Section 10 — Isolation Forest (cell 72)
# ===========================================================================

class TestIsolationForest:
    @pytest.fixture(scope="class")
    def iso_outputs(self, raw_df):
        from src.preprocessing import full_pipeline
        from src.features import build_model_dataset
        from src.models import train_final_model
        from src.anomaly import run_isolation_forest
        from sklearn.preprocessing import StandardScaler

        df, _ = full_pipeline(raw_df.copy())
        X, y, _, _ = build_model_dataset(df)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return run_isolation_forest(X_scaled, y), y

    def test_iso_ap_in_range(self, iso_outputs):
        (iso_ap, iso_pred, iso_scores), y = iso_outputs
        assert 0.0 <= iso_ap <= 1.0

    def test_iso_pred_is_binary(self, iso_outputs):
        (iso_ap, iso_pred, iso_scores), y = iso_outputs
        assert set(np.unique(iso_pred)).issubset({0, 1})

    def test_iso_scores_same_length_as_y(self, iso_outputs):
        (iso_ap, iso_pred, iso_scores), y = iso_outputs
        assert len(iso_scores) == len(y)


# ===========================================================================
# Section 11 — Evaluation output (cells 50-52, 79)
# ===========================================================================

class TestEvaluation:
    def test_print_cv_results_runs(self, capsys, raw_df):
        from src.preprocessing import full_pipeline
        from src.features import build_model_dataset
        from src.models import get_models, run_cross_validation
        from src.evaluation import print_cv_results

        df, _ = full_pipeline(raw_df.copy())
        X, y, _, _ = build_model_dataset(df)
        models = {k: v for k, v in get_models().items()
                  if k == "Logistic Regression"}
        results, *_ = run_cross_validation(X, y, models=models, n_splits=3)
        print_cv_results(results)

        captured = capsys.readouterr()
        assert "AUPRC" in captured.out
        assert "Logistic Regression" in captured.out

    def test_print_final_summary_runs(self, capsys, raw_df):
        from src.preprocessing import full_pipeline
        from src.features import build_model_dataset
        from src.models import get_models, run_cross_validation
        from src.evaluation import print_final_summary

        df, _ = full_pipeline(raw_df.copy())
        X, y, _, _ = build_model_dataset(df)
        models = {k: v for k, v in get_models().items()
                  if k == "Logistic Regression"}
        results, *_ = run_cross_validation(X, y, models=models, n_splits=3)
        print_final_summary(results, iso_ap=0.15, X=X, y=y)

        captured = capsys.readouterr()
        assert "FINAL RESULTS" in captured.out


# ===========================================================================
# Section 12 — Visualizations (cells 46, 53, 59, 63, 64)
# Non-interactive: just verify the functions complete without exceptions.
# ===========================================================================

class TestVisualization:
    @pytest.fixture(scope="class")
    def viz_data(self, raw_df):
        """Pre-compute everything needed for plot tests."""
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend

        from src.preprocessing import full_pipeline
        from src.features import build_model_dataset
        from src.models import get_models, run_cross_validation

        df, _ = full_pipeline(raw_df.copy())
        X, y, feature_cols, feature_names = build_model_dataset(df)
        models = {k: v for k, v in get_models().items()
                  if k in ("Random Forest", "Logistic Regression")}
        results, roc_data, prc_data, mean_fpr, mean_rec = run_cross_validation(
            X, y, models=models, n_splits=3
        )
        mean_shap = np.abs(np.random.default_rng(0).standard_normal(
            (50, len(feature_names))
        )).mean(axis=0)
        return {
            "df": df,
            "results": results,
            "roc_data": roc_data,
            "prc_data": prc_data,
            "mean_fpr": mean_fpr,
            "mean_rec": mean_rec,
            "mean_shap": mean_shap,
            "feature_names": feature_names,
            "baseline": float(y.mean()),
        }

    def test_plot_roc_curves(self, viz_data, tmp_path):
        from src.visualization import plot_roc_curves
        plot_roc_curves(
            viz_data["roc_data"], viz_data["results"], viz_data["mean_fpr"],
            save=True, filename=str(tmp_path / "roc.png"),
        )
        assert (tmp_path / "roc.png").exists()

    def test_plot_prc_curves(self, viz_data, tmp_path):
        from src.visualization import plot_prc_curves
        plot_prc_curves(
            viz_data["prc_data"], viz_data["results"], viz_data["mean_rec"],
            baseline=viz_data["baseline"],
            save=True, filename=str(tmp_path / "prc.png"),
        )
        assert (tmp_path / "prc.png").exists()

    def test_plot_model_comparison(self, viz_data, tmp_path):
        from src.visualization import plot_model_comparison
        plot_model_comparison(
            viz_data["results"], iso_ap=0.15,
            baseline=viz_data["baseline"],
            save=True, filename=str(tmp_path / "compare.png"),
        )
        assert (tmp_path / "compare.png").exists()

    def test_plot_shap_bar(self, viz_data, tmp_path):
        from src.visualization import plot_shap_bar
        plot_shap_bar(
            viz_data["mean_shap"], viz_data["feature_names"],
            save=True, filename=str(tmp_path / "shap.png"),
        )
        assert (tmp_path / "shap.png").exists()

    def test_plot_size_distribution(self, viz_data, tmp_path):
        from src.visualization import plot_size_distribution
        plot_size_distribution(
            viz_data["df"],
            save=True, filename=str(tmp_path / "size_dist.png"),
        )
        assert (tmp_path / "size_dist.png").exists()

    def test_plot_annual_trends(self, viz_data, tmp_path):
        from src.visualization import plot_annual_trends
        plot_annual_trends(
            viz_data["df"],
            save=True, filename=str(tmp_path / "annual.png"),
        )
        assert (tmp_path / "annual.png").exists()
