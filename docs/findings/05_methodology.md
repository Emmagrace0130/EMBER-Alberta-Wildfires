# Methodology — Alberta Wildfire Prediction Pipeline

**Document type:** Methodology  
**Pipeline file:** run_pipeline.py  
**Source modules:** src/ (data_loader, preprocessing, features, models, evaluation, visualization, shap_analysis, anomaly)  
**Tags:** methodology, pipeline, preprocessing, SMOTE, cross-validation, random-forest, logistic-regression, XGBoost, isolation-forest, SHAP, label-encoding, standard-scaler

---

## Pipeline Overview

The pipeline follows 10 sequential steps, from raw Excel data to saved model and figures.

```
Raw Excel → Load → Preprocess → Feature Engineering → 
Build Dataset → Cross-Validation → Isolation Forest → 
Final Model Training → SHAP Analysis → Save Outputs
```

---

## Step 1: Data Loading

- Input: Excel file (`fp-historical-wildfire-data-2006-2024.xlsx`)
- Library: `pandas.read_excel`
- Output: Raw DataFrame (26,551 rows × 50 columns)

---

## Step 2: Date Parsing

Six date columns were parsed to `datetime64` using `pd.to_datetime(..., errors='coerce')`:
- `FIRE_START_DATE`, `DISCOVERED_DATE`, `REPORTED_DATE`, `DISPATCH_DATE`, `FIRST_UC_DATE`, `FIRST_EX_DATE`

Invalid or missing dates are coerced to `NaT`.

---

## Step 3: Feature Engineering

Derived features created from raw columns:

| Feature | Derivation | Notes |
|---|---|---|
| `FIRE_MONTH` | `FIRE_START_DATE.month` | Integer 1–12 |
| `DETECTION_LAG_HRS` | `(DISCOVERED − START).seconds / 3600` | Clipped 0–72 hrs |
| `DISPATCH_LAG_HRS` | `(DISPATCH − DISCOVERED).seconds / 3600` | Clipped 0–48 hrs |
| `SUPPRESSION_DURATION_HRS` | `(FIRST_UC − START).seconds / 3600` | Clipped 0–2000 hrs |
| `FOREST_AREA` | `FIRE_NUMBER[0].upper()` | Regional zone code |
| `LARGE_FIRE` | `SIZE_CLASS ∈ {D, E}` | Binary target (0/1) |
| `CAUSE_BINARY` | `GENERAL_CAUSE == 'Lightning'` | Binary cause flag |

---

## Step 4: Null Imputation

| Strategy | Applied to |
|---|---|
| **Mean imputation** | TEMPERATURE, RELATIVE_HUMIDITY, WIND_SPEED, DETECTION_LAG_HRS, DISPATCH_LAG_HRS, FIRE_SPREAD_RATE, ASSESSMENT_HECTARES |
| **Mode imputation** | FUEL_TYPE, FIRE_TYPE, GENERAL_CAUSE, FOREST_AREA |

Imputation is fit on the full dataset (for preprocessing) but inside each CV fold (for model training).

---

## Step 5: Categorical Encoding

`LabelEncoder` was used to convert three categorical columns to integer codes:

| Original | Encoded |
|---|---|
| `FUEL_TYPE` | `FUEL_TYPE_ENC` |
| `FOREST_AREA` | `FOREST_AREA_ENC` |
| `FIRE_TYPE` | `FIRE_TYPE_ENC` |

Unknown or null values were filled with the string `"Unknown"` before encoding.

---

## Step 6: Model Dataset Construction

- Features: 12 columns from `FEATURE_CANDIDATES` dictionary that are present in the DataFrame
- Target: `LARGE_FIRE` (binary)
- Rows with any remaining NaN in features or target are dropped
- Final shape: **25,862 samples × 12 features**

---

## Step 7: Cross-Validation Design

```
StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
```

Within each fold:
1. Split into train/validation (stratified by LARGE_FIRE)
2. Fit `StandardScaler` on training fold, transform both
3. Apply `SMOTE` to training fold only (k_neighbors = min(5, positive_count − 1))
4. Fit each model on resampled training data
5. Evaluate on original (non-oversampled) validation fold

Metrics collected per fold:
- **AUPRC** (primary): `average_precision_score`
- **AUROC** (secondary): `auc(fpr, tpr)`
- **Accuracy**: `(pred == y_val).mean()`
- ROC curve data (interpolated to 100 points on [0,1])
- PRC curve data (interpolated to 100 points on [0,1])

---

## Step 8: Models

Three supervised classifiers were evaluated:

### Random Forest
- 100 trees, max depth 10, `class_weight='balanced'`
- Feature sampling: `sqrt(n_features)` per split

### Logistic Regression
- L2 regularisation, `class_weight='balanced'`
- `max_iter=1000`, solver defaults

### XGBoost
- 100 trees, max depth 6, learning rate 0.1
- `scale_pos_weight = n_negative / n_positive` (~29)
- Handles imbalance natively via pos weight

---

## Step 9: Isolation Forest (Unsupervised)

```
IsolationForest(
    n_estimators=200,
    contamination=y.mean(),   # 0.033
    random_state=42,
    n_jobs=-1
)
```

- Fit on full scaled dataset without labels
- Anomaly score = negated `score_samples()` (higher = more anomalous)
- Predictions: `predict() == -1` → large fire
- Evaluated with `average_precision_score(y, anomaly_scores)`

---

## Step 10: Final Model Training

The final Random Forest was retrained on the **full dataset** (all 25,862 records) after cross-validation, using SMOTE on the full training set. This model is saved as `wildfire_model.pkl` for deployment.

```
RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced')
```

---

## Step 11: SHAP Analysis

`shap.TreeExplainer` was applied to the final Random Forest on a random sample of 500 scaled records. SHAP values for the positive class (class 1, large fire) were extracted. Mean absolute SHAP values were computed across all samples to rank features.

---

## Reproducibility

All random operations use `random_state=42`. The pipeline is fully deterministic given the same input data and library versions.

```bash
python run_pipeline.py --data fp-historical-wildfire-data-2006-2024.xlsx --output-dir results/
```

---

## Dependencies

| Library | Purpose |
|---|---|
| pandas | Data loading and manipulation |
| numpy | Numerical computation |
| scikit-learn | Models, preprocessing, metrics, CV |
| imbalanced-learn | SMOTE oversampling |
| xgboost | XGBoost classifier |
| shap | SHAP explainability |
| matplotlib | Figure generation |
| joblib | Model serialisation |
