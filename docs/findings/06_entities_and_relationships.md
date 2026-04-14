# Entities and Relationships Index — Alberta Wildfire Prediction

**Document type:** Knowledge Graph Seed / Entity Index  
**Purpose:** Structured entity and relationship definitions for GraphRAG ingestion  
**Tags:** entities, relationships, knowledge-graph, graphrag, RAG, wildfire, Alberta, machine-learning

---

## Entities

### Project / Study
- **Name:** Alberta Wildfire Large-Fire Escalation Prediction
- **Type:** Research Study
- **Domain:** Wildfire Risk Management, Machine Learning
- **Geographic scope:** Alberta Forest Protection Area (FPA)
- **Temporal scope:** 2006–2024
- **Output:** Trained model (wildfire_model.pkl), 6 figures, findings documents

---

### Dataset
- **Name:** fp-historical-wildfire-data-2006-2024
- **Type:** Tabular Dataset
- **Source:** Alberta Government, Forest Protection Division
- **Rows:** 26,551
- **Columns:** 50
- **Positive class (large fire):** 865 records (3.34%)
- **Negative class (small fire):** 25,682 records (96.66%)

---

### Models

| Entity | Type | Key Parameters | AUPRC (CV) | AUROC (CV) |
|---|---|---|---|---|
| Logistic Regression | Supervised Classifier | class_weight=balanced, max_iter=1000 | 0.584 ± 0.041 | 0.950 ± 0.009 |
| Random Forest | Supervised Classifier | n_estimators=100, max_depth=10, class_weight=balanced | 0.513 ± 0.044 | 0.959 ± 0.007 |
| Random Forest (final) | Supervised Classifier | n_estimators=200, max_depth=10, class_weight=balanced | N/A (full train) | N/A |
| Isolation Forest | Unsupervised Anomaly Detector | n_estimators=200, contamination=0.033 | ~0.05–0.10 | N/A |

---

### Features

| Entity | Type | Importance (RF Gini) | Direction |
|---|---|---|---|
| Assessment hectares | Numeric, direct measurement | 0.4232 | Positive |
| Fire spread rate (m/min) | Numeric, direct measurement | 0.2353 | Positive |
| Wind speed (km/h) | Numeric, weather | 0.0629 | Positive |
| Lightning ignition | Binary, derived | 0.0562 | Positive |
| Temperature (°C) | Numeric, weather | 0.0493 | Positive |
| Forest area (encoded) | Categorical, derived | 0.0388 | Mixed |
| Fire type (encoded) | Categorical, encoded | 0.0365 | Mixed |
| Dispatch lag (hours) | Numeric, derived | 0.0248 | Positive |
| Fuel type (encoded) | Categorical, encoded | 0.0235 | Mixed |
| Relative humidity (%) | Numeric, weather | 0.0227 | Negative |
| Month of ignition | Numeric, derived | 0.0146 | Mixed |
| Detection lag (hours) | Numeric, derived | 0.0122 | Positive |

---

### Metrics

| Entity | Type | Definition | Used As |
|---|---|---|---|
| AUPRC | Evaluation Metric | Area under Precision-Recall Curve | Primary metric |
| AUROC | Evaluation Metric | Area under ROC Curve | Secondary metric |
| Accuracy | Evaluation Metric | (TP + TN) / Total | Reported, not primary |
| Precision | Evaluation Metric | TP / (TP + FP) | Classification report |
| Recall | Evaluation Metric | TP / (TP + FN) | Classification report |
| F1 Score | Evaluation Metric | Harmonic mean of precision & recall | Classification report |

---

### Techniques

| Entity | Type | Purpose |
|---|---|---|
| SMOTE | Data Augmentation | Oversample minority class (large fires) in training folds |
| StandardScaler | Preprocessing | Normalise features to zero mean, unit variance |
| LabelEncoder | Preprocessing | Encode categorical columns to integers |
| StratifiedKFold | Evaluation | 10-fold CV preserving class ratio per fold |
| SHAP TreeExplainer | Explainability | Compute Shapley values for Random Forest |
| Mean imputation | Preprocessing | Fill numeric nulls with column mean |
| Mode imputation | Preprocessing | Fill categorical nulls with most frequent value |

---

## Relationships

```
Alberta Wildfire Dataset  --[source for]-->         Prediction Study
Alberta Wildfire Dataset  --[contains]-->           Large fires (3.34%)
Alberta Wildfire Dataset  --[contains]-->           Small fires (96.66%)

Prediction Study          --[predicts]-->           Large fire escalation
Prediction Study          --[uses]-->               Random Forest
Prediction Study          --[uses]-->               Logistic Regression
Prediction Study          --[uses]-->               Isolation Forest

Logistic Regression       --[outperforms on AUPRC]-->   Random Forest
Random Forest             --[outperforms on AUROC]-->   Logistic Regression
Isolation Forest          --[provides unsupervised signal for]-->  Large fire detection

SMOTE                     --[applied inside]-->          CV training folds
SMOTE                     --[addresses]-->               Class imbalance
StandardScaler            --[normalises]-->              Feature matrix
StratifiedKFold           --[evaluates]-->               Model generalisation

Assessment hectares       --[most important predictor of]-->  Large fire
Fire spread rate          --[second most important predictor of]-->  Large fire
Wind speed                --[amplifies probability of]-->  Large fire
Lightning ignition        --[increases risk of]-->  Large fire
Relative humidity         --[reduces probability of]-->  Large fire

AUPRC                     --[preferred metric for]-->  Imbalanced classification
AUROC                     --[misleading when]-->  Severe class imbalance
Random Forest (final)     --[saved as]-->  wildfire_model.pkl
```

---

## Cross-Document References

| Topic | Document |
|---|---|
| Study overview and key findings | 01_executive_summary.md |
| Dataset schema and statistics | 02_dataset_characteristics.md |
| CV results and model configurations | 03_model_performance.md |
| Feature rankings and SHAP | 04_feature_importance.md |
| Step-by-step pipeline | 05_methodology.md |
