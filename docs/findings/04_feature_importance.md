# Feature Importance & SHAP Analysis — Large-Fire Prediction

**Document type:** Feature Analysis  
**Model:** Random Forest (final, n_estimators=200)  
**Method:** Gini impurity-based importance + SHAP TreeExplainer  
**Tags:** feature-importance, SHAP, random-forest, assessment-hectares, fire-spread-rate, wind-speed, lightning, fuel-type, detection-lag

---

## Feature Set

Twelve features were used for modelling, derived from the raw dataset through preprocessing and feature engineering. All features are available at or near the time of initial fire assessment.

| Feature Column | Description | Source |
|---|---|---|
| `ASSESSMENT_HECTARES` | Fire size at assessment (ha) | Direct measurement |
| `FIRE_SPREAD_RATE` | Rate of spread at assessment (m/min) | Direct measurement |
| `WIND_SPEED` | Wind speed at assessment (km/h) | Weather observation |
| `CAUSE_BINARY` | Lightning ignition (1=yes, 0=no) | Derived from GENERAL_CAUSE |
| `TEMPERATURE` | Temperature at assessment (°C) | Weather observation |
| `FOREST_AREA_ENC` | Alberta forest management zone | Derived from FIRE_NUMBER |
| `FIRE_TYPE_ENC` | Fire type (surface/crown/ground) | Label-encoded |
| `DISPATCH_LAG_HRS` | Hours from discovery to dispatch | Derived from dates |
| `FUEL_TYPE_ENC` | Forest fuel type classification | Label-encoded |
| `RELATIVE_HUMIDITY` | Relative humidity at assessment (%) | Weather observation |
| `FIRE_MONTH` | Month of ignition (1–12) | Derived from FIRE_START_DATE |
| `DETECTION_LAG_HRS` | Hours from start to discovery (0–72) | Derived from dates |

---

## Random Forest Feature Importances (Gini)

Ranked by mean decrease in Gini impurity across all 200 trees:

| Rank | Feature | Importance |
|---|---|---|
| 1 | Size at assessment (ha) | **0.4232** |
| 2 | Fire spread rate (m/min) | **0.2353** |
| 3 | Wind speed (km/h) | 0.0629 |
| 4 | Lightning ignition | 0.0562 |
| 5 | Temperature at assessment (°C) | 0.0493 |
| 6 | Forest area (encoded) | 0.0388 |
| 7 | Fire type (encoded) | 0.0365 |
| 8 | Dispatch lag (hours) | 0.0248 |
| 9 | Fuel type (encoded) | 0.0235 |
| 10 | Relative humidity (%) | 0.0227 |
| 11 | Month of ignition | 0.0146 |
| 12 | Detection lag (hours) | 0.0122 |

**Top 2 features (assessment hectares + spread rate) account for 65.9% of total importance.**

---

## SHAP Analysis

SHAP (SHapley Additive exPlanations) values were computed using `shap.TreeExplainer` on the final Random Forest, applied to a random sample of 500 records from the full scaled dataset.

### Interpretation of SHAP Values

- A **positive SHAP value** for a feature pushes the prediction toward large fire (class 1)
- A **negative SHAP value** pushes toward small fire (class 0)
- **Mean |SHAP|** measures average magnitude of impact regardless of direction

### Key SHAP Findings

1. **Assessment hectares** has the highest mean |SHAP| value — fires that are already large at assessment are strongly predicted to escalate further.

2. **Fire spread rate** is the second most influential feature. High spread rates at the time of assessment are strongly associated with large-fire outcomes.

3. **Wind speed** and **temperature** act as amplifying conditions — high wind and high temperature increase the probability of large-fire escalation, consistent with fire weather science.

4. **Lightning ignition** (CAUSE_BINARY) contributes positively to large-fire probability. Lightning-caused fires tend to start in remote locations with delayed detection, giving them more time to grow before suppression begins.

5. **Detection lag** has the lowest SHAP importance — once a fire is reported, the response speed (dispatch lag) matters more than how long it went undetected.

---

## Operational Implications

| Feature | Operational Action |
|---|---|
| Assessment hectares | Fires > threshold ha at assessment → escalate priority immediately |
| Fire spread rate | High spread rate → pre-position aerial resources |
| Wind speed | Forecast high winds → elevated watch on all active fires |
| Lightning ignition | Remote lightning-caused fires → increased aerial reconnaissance frequency |
| Forest area | Certain management zones have higher large-fire rates → zone-specific trigger thresholds |

---

## Feature Relationships (for Graph Extraction)

- `Assessment hectares` **most important predictor of** `large fire escalation`
- `Fire spread rate` **second most important predictor of** `large fire escalation`
- `Wind speed` **positively associated with** `large fire probability`
- `Temperature` **positively associated with** `large fire probability`
- `Relative humidity` **negatively associated with** `large fire probability`
- `Lightning ignition` **positively associated with** `large fire escalation`
- `Detection lag` **least predictive feature** in this dataset
- `Assessment hectares` + `Fire spread rate` **explain 65.9% of** `Random Forest importance`
