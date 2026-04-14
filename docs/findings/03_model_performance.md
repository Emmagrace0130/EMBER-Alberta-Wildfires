# Model Performance Report — Large-Fire Escalation Prediction

**Document type:** Model Evaluation  
**Evaluation method:** 10-fold stratified cross-validation  
**Primary metric:** AUPRC (Area Under Precision-Recall Curve)  
**Secondary metric:** AUROC (Area Under ROC Curve)  
**Tags:** model-performance, cross-validation, AUPRC, AUROC, random-forest, logistic-regression, isolation-forest, SMOTE, class-imbalance

---

## Why AUPRC is the Primary Metric

With only 3.3% positive examples, AUROC can be misleadingly high even for poor models. **AUPRC is more informative** for imbalanced problems because it focuses on the positive (large fire) class.

- **Random baseline AUPRC:** 0.033 (= positive class prevalence)
- **All models substantially exceed this baseline**

---

## 10-Fold Cross-Validation Results

All models were evaluated using stratified 10-fold CV with SMOTE oversampling applied only within each training fold (no data leakage). A `StandardScaler` was fit on each training fold and applied to the validation fold.

### Supervised Models

| Model | AUPRC (mean ± SD) | AUROC (mean ± SD) | Accuracy (mean ± SD) |
|---|---|---|---|
| **Logistic Regression** | **0.584 ± 0.041** | 0.950 ± 0.009 | 0.906 ± 0.005 |
| Random Forest | 0.513 ± 0.044 | 0.959 ± 0.007 | 0.943 ± 0.005 |

> **Best model by AUPRC: Logistic Regression** (0.584 ± 0.041)  
> **Best model by AUROC: Random Forest** (0.959 ± 0.007)

### Unsupervised Baseline — Isolation Forest

The Isolation Forest was trained without labels, treating large fires as statistical anomalies.

| Model | AUPRC | Notes |
|---|---|---|
| Isolation Forest | ~0.05–0.10 | Unsupervised; no CV (fit on full dataset) |
| Random baseline | 0.033 | = positive class prevalence |

The Isolation Forest provides an independent signal that large fires are structurally unusual events, even without label supervision.

---

## Model Configurations

### Logistic Regression
```
LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
```

### Random Forest
```
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

### Final Model (Random Forest, full dataset)
```
RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```
Saved as: `results/wildfire_model.pkl`

---

## In-Sample Performance (Final RF on Full Dataset)

> ⚠️ These numbers are optimistic — the model has seen this data during training.

| Metric | Score |
|---|---|
| AUROC | 0.984 |
| AUPRC | 0.711 |
| Large fire recall | 91% |
| Large fire precision | 42% |
| Overall accuracy | 95% |

The high recall is intentional: `class_weight='balanced'` and SMOTE both push the model to prioritise catching large fires, accepting some false alarms — the correct operational tradeoff for wildfire risk.

---

## Precision-Recall Tradeoff Interpretation

| Recall | Meaning |
|---|---|
| 91% | 9 out of every 10 large fires are flagged |
| 42% precision | For every 10 flagged fires, ~4 are truly large |

In the context of wildfire management, **high recall is preferred**: missing a large fire (false negative) is far costlier than a false alarm (false positive) that triggers unnecessary resource pre-positioning.

---

## SMOTE Configuration

```
SMOTE(
    random_state=42,
    k_neighbors=min(5, max(1, positive_count_in_fold - 1))
)
```

SMOTE was applied **inside each CV fold** to prevent data leakage. The adaptive `k_neighbors` handles folds with very few positive examples.

---

## Relationships Between Entities

- `Random Forest` **trained on** `Alberta Wildfire Dataset`
- `Logistic Regression` **evaluated against** `Random Forest`
- `Logistic Regression` **outperforms** `Random Forest` on `AUPRC`
- `Random Forest` **outperforms** `Logistic Regression` on `AUROC`
- `SMOTE` **applied to** training folds to address `class imbalance`
- `Isolation Forest` **detects** large fires as `anomalies`
- `AUPRC` **preferred over** `AUROC` for `imbalanced classification`
