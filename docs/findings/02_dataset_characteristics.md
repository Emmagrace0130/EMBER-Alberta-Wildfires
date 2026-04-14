# Dataset Characteristics — Alberta Historical Wildfire Data 2006–2024

**Document type:** Data Description  
**Source:** Alberta Government — Forest Protection Division  
**File:** fp-historical-wildfire-data-2006-2024.xlsx  
**Tags:** dataset, wildfire, Alberta, FPA, size-class, fire-cause, temporal-trends, class-imbalance

---

## Source and Scope

The dataset contains historical wildfire records for the **Alberta Forest Protection Area (FPA)** spanning 2006 to 2024, published by the Alberta Government Forest Protection Division. Each row represents a single wildfire event with attributes recorded at discovery, assessment, and suppression stages.

- **Total records:** 26,551 fires
- **Columns:** 50 raw attributes
- **Time span:** 2006–2024 (18 years)
- **Geography:** Alberta Forest Protection Area

---

## Target Variable — Large Fire

The binary prediction target is defined as:

> **LARGE_FIRE = 1** if `SIZE_CLASS` ∈ {D, E}  (fire ≥ 40 ha)  
> **LARGE_FIRE = 0** if `SIZE_CLASS` ∈ {A, B, C} (fire < 40 ha)

### Size Class Distribution

| Size Class | Hectares | Count | % of Total |
|---|---|---|---|
| A | 0 – 0.1 ha | 17,503 | 65.9% |
| B | 0.1 – 4 ha | 6,898 | 26.0% |
| C | 4 – 40 ha | 1,281 | 4.8% |
| D | 40 – 200 ha | 384 | 1.4% |
| E | ≥ 200 ha | 485 | 1.8% |
| **Large (D+E)** | **≥ 40 ha** | **869** | **3.27%** |

This severe **class imbalance** (3.3% positive rate) is the central modelling challenge. A naive majority-class classifier achieves 96.7% accuracy while detecting zero large fires.

---

## Ignition Cause Distribution

| Cause | Count |
|---|---|
| Lightning | 9,292 |
| Recreation | 5,730 |
| Resident | 4,442 |
| Incendiary | 2,310 |
| Power Line / Industry | 1,429 |

Lightning is the most common cause (~35% of fires) and is encoded as a binary feature (`CAUSE_BINARY = 1`) due to its known association with fire spread under dry conditions.

---

## Temporal Coverage

Records span **2006 to 2024**, covering 18 wildfire seasons. The dataset includes years with extreme fire activity (notably 2016, the Fort McMurray wildfire year) as well as low-activity years, providing good variation for model training.

---

## Date Columns and Derived Lag Features

The following date fields were present and used to engineer lag features:

| Column | Description |
|---|---|
| `FIRE_START_DATE` | Estimated time of ignition |
| `DISCOVERED_DATE` | Time fire was discovered/reported |
| `REPORTED_DATE` | Official report time |
| `DISPATCH_DATE` | Time initial resources were dispatched |
| `FIRST_UC_DATE` | Time fire was first brought under control |
| `FIRST_EX_DATE` | Time fire was declared extinguished |

### Derived Features from Dates

| Feature | Formula | Clip Range |
|---|---|---|
| `DETECTION_LAG_HRS` | DISCOVERED_DATE − FIRE_START_DATE | 0–72 hours |
| `DISPATCH_LAG_HRS` | DISPATCH_DATE − DISCOVERED_DATE | 0–48 hours |
| `SUPPRESSION_DURATION_HRS` | FIRST_UC_DATE − FIRE_START_DATE | 0–2000 hours |
| `FIRE_MONTH` | Month of FIRE_START_DATE | 1–12 |

---

## Null Handling

Missing values were imputed using standard strategies:

- **Numeric columns** (TEMPERATURE, RELATIVE_HUMIDITY, WIND_SPEED, DETECTION_LAG_HRS, DISPATCH_LAG_HRS, FIRE_SPREAD_RATE, ASSESSMENT_HECTARES): filled with **column mean**
- **Categorical columns** (FUEL_TYPE, FIRE_TYPE, GENERAL_CAUSE, FOREST_AREA): filled with **column mode**

---

## Forest Area Codes

`FOREST_AREA` was extracted from the first character of `FIRE_NUMBER` (e.g., "A0001" → Area "A"), providing a regional indicator across Alberta's forest management zones.

---

## Known Limitations

1. **Assessment-time features:** TEMPERATURE, WIND_SPEED, and ASSESSMENT_HECTARES are recorded at assessment, which may occur after the fire has already grown — this creates some data leakage risk for real-time deployment.
2. **Missing weather data:** Many records have null weather observations; mean imputation was used but may mask seasonal or geographic variation.
3. **Historical bias:** Fire detection and reporting practices have changed over 18 years, potentially introducing temporal confounding.
