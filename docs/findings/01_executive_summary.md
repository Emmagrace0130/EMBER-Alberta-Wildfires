# Executive Summary — Alberta Wildfire Large-Fire Escalation Prediction

**Document type:** Executive Summary  
**Domain:** Wildfire risk prediction, Alberta Forest Protection Area  
**Dataset:** fp-historical-wildfire-data-2006-2024 (Alberta Government)  
**Date:** April 2026  
**Tags:** wildfire, machine-learning, classification, large-fire, Alberta, random-forest, logistic-regression, SMOTE, SHAP

---

## Overview

This study applies supervised machine learning and unsupervised anomaly detection to 18 years of Alberta wildfire records (2006–2024) to predict which fires will escalate into **large fires** (Size Class D or E, ≥ 40 ha). Early identification of large-fire potential is operationally critical — large fires represent only **3.3% of all fires** but account for the majority of total area burned and suppression cost.

---

## Key Question

> Can we predict, at or near the time of discovery/assessment, whether a wildfire will become a large fire (≥ 40 ha)?

---

## Dataset at a Glance

| Property | Value |
|---|---|
| Total fire records | 26,551 |
| Records used for modelling | 25,862 |
| Features used | 12 |
| Large fires (Class D/E) | 865 (3.34%) |
| Small fires (Class A/B/C) | 24,997 (96.66%) |
| Time period | 2006 – 2024 |
| Geographic scope | Alberta Forest Protection Area (FPA) |

---

## Key Findings

### 1. Large-fire prediction is feasible with moderate-to-strong performance
Both supervised models achieved **AUROC > 0.95** in held-out cross-validation, confirming strong discriminative ability. The primary metric (AUPRC) ranged from 0.51–0.58, which is **15–18× higher than the random baseline** of 0.033.

### 2. Logistic Regression outperformed Random Forest on the primary metric
Despite being simpler, Logistic Regression achieved a higher mean AUPRC (0.584) than Random Forest (0.513) in 10-fold cross-validation, suggesting that linear decision boundaries may be sufficient and that the RF model may be slightly overfit to the training distribution.

### 3. Fire size at assessment is the strongest single predictor
Assessment hectares alone accounts for ~42% of Random Forest feature importance, followed by fire spread rate (~24%). Together these two operational measurements explain two-thirds of predictive power.

### 4. Isolation Forest provides useful unsupervised signal
Even without labels, the Isolation Forest anomaly detector identifies large fires as statistically unusual events, providing an independent cross-check on supervised model outputs.

### 5. Class imbalance is a significant challenge
At 3.34% positive rate, a naive classifier achieves 96.7% accuracy. SMOTE oversampling and class-weight balancing were required to achieve meaningful recall on the minority class.

---

## Recommended Model for Deployment

**Logistic Regression** (class-weight balanced, max_iter=1000) is recommended for production use:
- Highest AUPRC in cross-validation
- Interpretable coefficients
- Fast inference
- Robust to new data distributions

The **Random Forest** model (`wildfire_model.pkl`) is retained as a secondary model for ensemble use and SHAP-based feature attribution.

---

## Figures Generated

| File | Description |
|---|---|
| `fig_prc_curves.png` | Precision-Recall curves (10-fold CV, mean ± 1 SD) |
| `fig_roc_curves.png` | ROC curves (10-fold CV, mean ± 1 SD) |
| `fig_model_comparison.png` | AUPRC bar chart across all models |
| `fig_shap.png` | SHAP mean absolute feature importance (Random Forest) |
| `fig_size_distribution.png` | Fire size class distribution |
| `fig_annual_trends.png` | Annual fire count vs. area burned 2006–2024 |
