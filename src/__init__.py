"""
EMBER Alberta Wildfires — source modules.

Modules
-------
data_loader     : load raw Excel data into a DataFrame
preprocessing   : date parsing, feature engineering, imputation, encoding
features        : feature definitions and model-dataset builder
models          : model definitions and cross-validation runner
evaluation      : CV results printing and final summary
visualization   : all plot functions (ROC, PRC, SHAP, distributions)
shap_analysis   : SHAP value computation and feature importance
anomaly         : Isolation Forest anomaly detection
"""

from .data_loader import load_data
from .preprocessing import parse_dates, engineer_features, impute_missing, encode_categoricals
from .features import FEATURE_CANDIDATES, get_feature_cols, build_model_dataset
from .models import get_models, run_cross_validation, train_final_model
from .evaluation import print_cv_results, print_final_summary
from .visualization import (
    plot_roc_curves,
    plot_prc_curves,
    plot_model_comparison,
    plot_shap_bar,
    plot_size_distribution,
    plot_annual_trends,
)
from .shap_analysis import compute_shap
from .anomaly import run_isolation_forest
