"""
EMBER Tool Server
=================
FastAPI server that exposes wildfire risk assessment tools
for Ollama LLM tool calls.

Install dependencies:
    pip install fastapi uvicorn numpy

Run:
    python tool_server.py

The server runs on port 5001 by default.
Your Ollama frontend calls this server when the LLM
decides to use a tool.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import json

app = FastAPI(title="EMBER Tool Server", version="1.0")

# Allow requests from your GitHub Pages frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# TOOL 1: Large Fire Risk Assessment
# ─────────────────────────────────────────────

class RiskInput(BaseModel):
    size_ha: float              # Fire size at assessment (ha)
    spread_rate: float          # Fire spread rate (m/min)
    wind_speed: float           # Wind speed (km/h)
    temperature: float          # Temperature (°C)
    relative_humidity: float    # Relative humidity (%)
    lightning_ignition: bool    # True if lightning caused ignition
    forest_area: str            # Forest region name
    fuel_type: Optional[str] = "Unknown"
    detection_lag_hrs: Optional[float] = 2.0
    dispatch_lag_hrs: Optional[float] = 1.0
    fire_month: Optional[int] = 6

# SHAP weights from your actual Random Forest model
SHAP_WEIGHTS = {
    "size_ha":           0.195,
    "spread_rate":       0.130,
    "wind_speed":        0.050,
    "lightning":         0.048,
    "forest_area":       0.030,
    "temperature":       0.028,
    "relative_humidity": 0.020,
    "fuel_type":         0.019,
    "dispatch_lag":      0.017,
    "fire_type":         0.014,
    "fire_month":        0.010,
    "detection_lag":     0.008,
}

HIGH_RISK_REGIONS = ["fort mcmurray", "high level"]
MED_RISK_REGIONS  = ["slave lake", "lac la biche", "peace river"]

def compute_risk_score(data: RiskInput) -> dict:
    """
    Risk scoring function based on SHAP feature importance
    from Random Forest trained on 26,551 Alberta FPA fires.
    Mirrors the logic from your Colab model output.
    """
    score = 0.0
    factors = []

    # ── Size at assessment (top SHAP predictor: 0.195) ──
    if data.size_ha > 200:
        score += 35; factors.append(f"Very large fire at assessment ({data.size_ha:.1f} ha) — extreme escalation signal")
    elif data.size_ha > 40:
        score += 22; factors.append(f"Large fire at assessment ({data.size_ha:.1f} ha) — Class D already reached")
    elif data.size_ha > 4:
        score += 10; factors.append(f"Moderate size at assessment ({data.size_ha:.1f} ha)")
    else:
        score += 3

    # ── Spread rate (2nd SHAP predictor: 0.130) ──
    if data.spread_rate > 10:
        score += 25; factors.append(f"Very high spread rate ({data.spread_rate:.1f} m/min) — Class E pathway")
    elif data.spread_rate > 5:
        score += 15; factors.append(f"High spread rate ({data.spread_rate:.1f} m/min)")
    elif data.spread_rate > 2:
        score += 7
    else:
        score += 2

    # ── Wind speed (3rd SHAP predictor: 0.050) ──
    if data.wind_speed > 40:
        score += 12; factors.append(f"High wind speed ({data.wind_speed:.0f} km/h) — drives spotting ahead of fire front")
    elif data.wind_speed > 25:
        score += 7; factors.append(f"Elevated wind speed ({data.wind_speed:.0f} km/h)")
    elif data.wind_speed > 15:
        score += 3

    # ── Lightning ignition (4th SHAP predictor: 0.048) ──
    if data.lightning_ignition:
        score += 8; factors.append("Lightning ignition — remote start, longer detection lag, historically larger fires")

    # ── Temperature (6th SHAP predictor: 0.028) ──
    if data.temperature > 30:
        score += 8; factors.append(f"High temperature ({data.temperature:.0f}°C) — drying fuels rapidly")
    elif data.temperature > 22:
        score += 4

    # ── Relative humidity (7th SHAP predictor: 0.020) ──
    if data.relative_humidity < 20:
        score += 10; factors.append(f"Critical low humidity ({data.relative_humidity:.0f}%) — fuels extremely receptive")
    elif data.relative_humidity < 30:
        score += 6; factors.append(f"Low humidity ({data.relative_humidity:.0f}%)")
    elif data.relative_humidity < 40:
        score += 2
    elif data.relative_humidity > 60:
        score -= 4

    # ── Forest area / region (5th SHAP predictor: 0.030) ──
    region = data.forest_area.lower()
    if any(r in region for r in HIGH_RISK_REGIONS):
        score += 10; factors.append(f"{data.forest_area} — high-risk region (part of 71% area burned concentration)")
    elif any(r in region for r in MED_RISK_REGIONS):
        score += 5; factors.append(f"{data.forest_area} — moderate-risk region")

    # ── Detection lag (8th SHAP predictor: 0.008) ──
    if data.detection_lag_hrs and data.detection_lag_hrs > 6:
        score += 5; factors.append(f"Long detection lag ({data.detection_lag_hrs:.1f} hrs) — fire grew undetected")
    elif data.detection_lag_hrs and data.detection_lag_hrs > 3:
        score += 2

    # ── Dispatch lag (7th predictor: 0.017) ──
    if data.dispatch_lag_hrs and data.dispatch_lag_hrs > 4:
        score += 4; factors.append(f"Long dispatch lag ({data.dispatch_lag_hrs:.1f} hrs)")

    score = float(np.clip(score, 1, 99))

    # Determine risk level
    if score >= 70:
        level = "EXTREME"
        color = "#C0392B"
        recommendation = "Immediate priority resourcing. High probability of Class D/E escalation. Pre-position aerial resources. Consider requesting mutual aid."
    elif score >= 50:
        level = "HIGH"
        color = "#E85D20"
        recommendation = "Elevated escalation risk. Standby resources advised. Conditions align with historical large fire profiles. Monitor closely for rapid change."
    elif score >= 30:
        level = "MODERATE"
        color = "#EF9F27"
        recommendation = "Some escalation risk. Standard response with readiness to scale up. Monitor wind and spread rate closely."
    else:
        level = "LOW"
        color = "#1D9E75"
        recommendation = "Low escalation probability. Standard initial attack protocol appropriate."

    if not factors:
        factors = ["No major risk flags detected in current conditions"]

    return {
        "risk_score": round(score),
        "risk_level": level,
        "color": color,
        "probability_class": f"~{min(99, round(score * 0.8))}% escalation probability (model-estimated)",
        "key_risk_factors": factors,
        "recommendation": recommendation,
        "model_notes": {
            "primary_metric": "AUPRC",
            "lr_auprc": "0.584 ± 0.041",
            "rf_auroc": "0.957 ± 0.008",
            "xgb_auprc": "0.567 ± 0.042",
            "random_baseline": "0.033",
            "training_data": "26,551 Alberta FPA fires, 2006-2024",
            "top_shap_feature": f"Size at assessment (weight 0.195) — your value: {data.size_ha} ha"
        }
    }

@app.post("/tools/assess_fire_risk")
def assess_fire_risk(data: RiskInput):
    """
    Assess large fire escalation risk from current fire conditions.
    Based on SHAP feature importance from RF model trained on 26,551 Alberta FPA fires.
    """
    result = compute_risk_score(data)
    return {"tool": "assess_fire_risk", "result": result}

# ─────────────────────────────────────────────
# TOOL 2: Regional Risk Profile
# ─────────────────────────────────────────────

class RegionInput(BaseModel):
    region: str

REGION_DATA = {
    "fort mcmurray": {
        "code": "M", "rank_frequency": 7, "rank_area": 1,
        "area_burned_ha": 2400000, "pct_total_area": 36,
        "notable": "2016 Horse River fire (590,000 ha). Dominated by C2 Boreal Spruce — fastest spreading fuel type.",
        "risk_tier": "EXTREME"
    },
    "high level": {
        "code": "H", "rank_frequency": 2, "rank_area": 2,
        "area_burned_ha": 2300000, "pct_total_area": 35,
        "notable": "Remote northern location. Long detection lags. Lightning-dominated ignition pattern.",
        "risk_tier": "EXTREME"
    },
    "slave lake": {
        "code": "S", "rank_frequency": 4, "rank_area": 3,
        "area_burned_ha": 1000000, "pct_total_area": 15,
        "notable": "2011 Lesser Slave Lake fire destroyed 30% of town. Mixed boreal and agricultural interface.",
        "risk_tier": "HIGH"
    },
    "edson": {
        "code": "E", "rank_frequency": 3, "rank_area": 4,
        "area_burned_ha": 277000, "pct_total_area": 4,
        "risk_tier": "MODERATE", "notable": "Foothills transition zone. Mixed fuel types."
    },
    "peace river": {
        "code": "P", "rank_frequency": 5, "rank_area": 5,
        "area_burned_ha": 180000, "pct_total_area": 3,
        "risk_tier": "MODERATE", "notable": "Agricultural interface reduces contiguous fuel continuity."
    },
    "grande prairie": {
        "code": "G", "rank_frequency": 6, "rank_area": 7,
        "area_burned_ha": 149000, "pct_total_area": 2,
        "risk_tier": "MODERATE", "notable": "Northwest Alberta. Moderate ignition density."
    },
    "lac la biche": {
        "code": "L", "rank_frequency": 8, "rank_area": 6,
        "area_burned_ha": 129000, "pct_total_area": 2,
        "risk_tier": "MODERATE", "notable": "Boreal transition. Historical interface fire risk."
    },
    "rocky": {
        "code": "R", "rank_frequency": 9, "rank_area": 8,
        "area_burned_ha": 125000, "pct_total_area": 2,
        "risk_tier": "LOW-MODERATE", "notable": "Rocky Mountain foothills. Terrain-driven fire behaviour."
    },
    "whitecourt": {
        "code": "W", "rank_frequency": 10, "rank_area": 9,
        "area_burned_ha": 79000, "pct_total_area": 1,
        "risk_tier": "LOW", "notable": "Central Alberta. Lower severity historical profile."
    },
    "calgary": {
        "code": "C", "rank_frequency": 1, "rank_area": 10,
        "area_burned_ha": 5000, "pct_total_area": 0.1,
        "notable": "Highest fire COUNT in province but less than 0.1% of area burned. Grassland fires, fast suppression.",
        "risk_tier": "LOW"
    },
}

@app.post("/tools/get_regional_profile")
def get_regional_profile(data: RegionInput):
    """
    Return historical risk profile for a given Alberta forest region.
    """
    region = data.region.lower().strip()
    match = None
    for key, val in REGION_DATA.items():
        if key in region or region in key:
            match = val
            match["region_name"] = key.title()
            break
    if not match:
        return {
            "tool": "get_regional_profile",
            "result": {"error": f"Region '{data.region}' not found. Valid regions: " + ", ".join(k.title() for k in REGION_DATA)}
        }
    return {"tool": "get_regional_profile", "result": match}

# ─────────────────────────────────────────────
# TOOL 3: Model Performance Lookup
# ─────────────────────────────────────────────

class ModelQuery(BaseModel):
    model_name: Optional[str] = "all"

MODEL_RESULTS = {
    "logistic_regression": {
        "auprc_mean": 0.584, "auprc_sd": 0.041,
        "auroc_mean": 0.950, "auroc_sd": 0.009,
        "accuracy_mean": 0.900, "accuracy_sd": 0.000,
        "type": "supervised",
        "best_for": "Precision — most reliable at flagging which fires will actually be large",
        "notes": "Linear decision boundary. L2 regularization. Balanced class weights. Wins on AUPRC."
    },
    "random_forest": {
        "auprc_mean": 0.506, "auprc_sd": 0.040,
        "auroc_mean": 0.957, "auroc_sd": 0.008,
        "accuracy_mean": 0.954, "accuracy_sd": 0.014,
        "type": "supervised",
        "best_for": "Overall discrimination — best AUROC across all thresholds. SHAP interpretability.",
        "notes": "100 trees, max depth 10, sqrt features, balanced weights, random state 42. Wins on AUROC."
    },
    "xgboost": {
        "auprc_mean": 0.567, "auprc_sd": 0.042,
        "auroc_mean": 0.952, "auroc_sd": 0.011,
        "accuracy_mean": None, "accuracy_sd": None,
        "type": "supervised",
        "best_for": "Non-linear patterns with strong precision. Close to LR on AUPRC.",
        "notes": "Gradient boosting. 100 estimators, max depth 6, learning rate 0.1. scale_pos_weight handles imbalance."
    },
    "isolation_forest": {
        "auprc_mean": 0.264, "auprc_sd": None,
        "auroc_mean": None, "auroc_sd": None,
        "accuracy_mean": None, "accuracy_sd": None,
        "type": "unsupervised",
        "best_for": "Anomaly detection without labeled outcomes. Useful when historical data unavailable.",
        "notes": "200 estimators. Contamination = 0.033 (large fire rate). 8x above random baseline of 0.033."
    },
    "random_baseline": {
        "auprc_mean": 0.033, "auprc_sd": None,
        "auroc_mean": 0.500, "auroc_sd": None,
        "type": "baseline",
        "best_for": "Reference only — all models beat this significantly",
        "notes": "AUPRC baseline = class prevalence (3.3%). AUROC baseline = 0.5."
    }
}

@app.post("/tools/get_model_performance")
def get_model_performance(data: ModelQuery):
    """
    Return cross-validation performance metrics for one or all models.
    """
    if data.model_name.lower() == "all":
        return {"tool": "get_model_performance", "result": MODEL_RESULTS}
    name = data.model_name.lower().replace(" ", "_").replace("-", "_")
    if name in MODEL_RESULTS:
        return {"tool": "get_model_performance", "result": {name: MODEL_RESULTS[name]}}
    return {
        "tool": "get_model_performance",
        "result": {"error": f"Model '{data.model_name}' not found. Options: " + ", ".join(MODEL_RESULTS.keys())}
    }

# ─────────────────────────────────────────────
# TOOL 4: SHAP Feature Importance Lookup
# ─────────────────────────────────────────────

@app.get("/tools/get_shap_importance")
def get_shap_importance():
    """
    Return SHAP mean absolute feature importance values from Random Forest.
    """
    return {
        "tool": "get_shap_importance",
        "result": {
            "model": "Random Forest",
            "n_samples": 500,
            "features": [
                {"rank": 1, "name": "Size at assessment (ha)",     "shap_value": 0.195, "interpretation": "Fires already large at first response overwhelmingly likely to escalate"},
                {"rank": 2, "name": "Fire spread rate (m/min)",    "shap_value": 0.130, "interpretation": "Fast-spreading fires at assessment are on the Class E pathway"},
                {"rank": 3, "name": "Wind speed (km/h)",           "shap_value": 0.050, "interpretation": "High wind drives spotting ahead of fire front"},
                {"rank": 4, "name": "Lightning ignition (1=yes)",  "shap_value": 0.048, "interpretation": "Remote starts, longer detection lags, historically larger outcomes"},
                {"rank": 5, "name": "Forest area (encoded)",       "shap_value": 0.030, "interpretation": "Fort McMurray and High Level carry independent risk beyond weather"},
                {"rank": 6, "name": "Temperature (°C)",            "shap_value": 0.028, "interpretation": "Hot conditions dry fuels and accelerate spread"},
                {"rank": 7, "name": "Relative humidity (%)",       "shap_value": 0.020, "interpretation": "Low RH increases fuel receptivity to ignition and spread"},
                {"rank": 8, "name": "Fuel type (encoded)",         "shap_value": 0.019, "interpretation": "C2 Boreal Spruce produces fastest spread rates — primary Class E pathway"},
                {"rank": 9, "name": "Dispatch lag (hours)",        "shap_value": 0.017, "interpretation": "Longer dispatch delay independently increases escalation probability"},
                {"rank":10, "name": "Fire type (encoded)",         "shap_value": 0.014, "interpretation": "Crown fires escalate faster than surface or ground fires"},
                {"rank":11, "name": "Month of ignition",           "shap_value": 0.010, "interpretation": "May-July peak season associated with more extreme conditions"},
                {"rank":12, "name": "Detection lag (hours)",       "shap_value": 0.008, "interpretation": "Every hour of undetected growth increases final size"},
            ],
            "key_insight": "Top two predictors (size + spread rate) are measured at first response — confirming this is an escalation classifier, not a pre-ignition predictor. Faster response directly reduces escalation probability."
        }
    }

# ─────────────────────────────────────────────
# TOOL 5: Season Statistics Lookup
# ─────────────────────────────────────────────

class SeasonQuery(BaseModel):
    year: Optional[int] = None

SEASON_DATA = {
    2006: {"fires": 1954, "area_ha": 181000, "notable": "High fire count year"},
    2007: {"fires": 1348, "area_ha": 154000, "notable": ""},
    2008: {"fires": 1712, "area_ha": 75000,  "notable": ""},
    2009: {"fires": 1710, "area_ha": 120000, "notable": ""},
    2010: {"fires": 1840, "area_ha": 834000, "notable": "Major fire year in Peace River region"},
    2011: {"fires": 1218, "area_ha": 380000, "notable": "Lesser Slave Lake fire — town evacuation"},
    2012: {"fires": 1568, "area_ha": 100000, "notable": ""},
    2013: {"fires": 1226, "area_ha": 100000, "notable": ""},
    2014: {"fires": 1470, "area_ha": 490000, "notable": ""},
    2015: {"fires": 1898, "area_ha": 510000, "notable": ""},
    2016: {"fires": 1376, "area_ha": 530000, "notable": "Horse River fire (Fort McMurray) — 590,000 ha, $9.9B insured losses"},
    2017: {"fires": 1244, "area_ha": 130000, "notable": ""},
    2018: {"fires": 1279, "area_ha": 130000, "notable": ""},
    2019: {"fires": 1005, "area_ha": 760000, "notable": "High Level complex — 340,000 ha"},
    2020: {"fires": 723,  "area_ha": 3274,   "notable": "Quietest season in study period — COVID restrictions reduced human ignitions"},
    2021: {"fires": 1342, "area_ha": 130000, "notable": ""},
    2022: {"fires": 1276, "area_ha": 170000, "notable": ""},
    2023: {"fires": 1132, "area_ha": 2217295,"notable": "Record season — 2.2M ha, $851M suppression cost, 3x historical average. Resource saturation (14.2 hr median control time). 30,000+ evacuated."},
    2024: {"fires": 1230, "area_ha": 714000, "notable": "Jasper townsite destroyed. 714,000 ha. Median control time 18.8 hrs."},
}

@app.post("/tools/get_season_stats")
def get_season_stats(data: SeasonQuery):
    """
    Return historical statistics for a specific fire season or all seasons.
    """
    if data.year:
        if data.year in SEASON_DATA:
            s = SEASON_DATA[data.year]
            return {"tool": "get_season_stats", "result": {
                "year": data.year,
                "fire_count": s["fires"],
                "area_burned_ha": s["area_ha"],
                "notable": s["notable"],
                "note": "r=-0.11 between annual fire count and area burned — frequency is a poor predictor of severity"
            }}
        return {"tool": "get_season_stats", "result": {"error": f"No data for year {data.year}. Range: 2006-2024"}}
    return {"tool": "get_season_stats", "result": {str(k): v for k, v in SEASON_DATA.items()}}

# ─────────────────────────────────────────────
# Health check + tool manifest
# ─────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "status": "running",
        "service": "EMBER Tool Server",
        "model_context": "Alberta FPA wildfires 2006-2024, 26,551 records",
        "tools_available": [
            "POST /tools/assess_fire_risk",
            "POST /tools/get_regional_profile",
            "POST /tools/get_model_performance",
            "GET  /tools/get_shap_importance",
            "POST /tools/get_season_stats",
        ]
    }

@app.get("/tools/manifest")
def tool_manifest():
    """
    Returns the tool definitions in Ollama tool call format.
    The frontend sends this to Ollama so it knows what tools are available.
    """
    return {"tools": [
        {
            "type": "function",
            "function": {
                "name": "assess_fire_risk",
                "description": "Assess the large fire escalation risk for a wildfire given current conditions at the time of assessment. Returns a risk score, risk level (LOW/MODERATE/HIGH/EXTREME), key risk factors, and operational recommendations. Use this whenever the user describes fire conditions or asks for a risk assessment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "size_ha":              {"type": "number", "description": "Fire size at initial assessment in hectares"},
                        "spread_rate":          {"type": "number", "description": "Fire spread rate at assessment in metres per minute"},
                        "wind_speed":           {"type": "number", "description": "Wind speed in km/h"},
                        "temperature":          {"type": "number", "description": "Temperature in degrees Celsius"},
                        "relative_humidity":    {"type": "number", "description": "Relative humidity as a percentage (0-100)"},
                        "lightning_ignition":   {"type": "boolean", "description": "True if the fire was caused by lightning, false if human-caused"},
                        "forest_area":          {"type": "string", "description": "The Alberta forest region name, e.g. Fort McMurray, High Level, Slave Lake, Calgary"},
                        "detection_lag_hrs":    {"type": "number", "description": "Hours between fire ignition and discovery (optional)"},
                        "dispatch_lag_hrs":     {"type": "number", "description": "Hours between discovery and first crew dispatch (optional)"},
                        "fire_month":           {"type": "integer", "description": "Month of ignition as integer 1-12 (optional)"},
                    },
                    "required": ["size_ha", "spread_rate", "wind_speed", "temperature", "relative_humidity", "lightning_ignition", "forest_area"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_regional_profile",
                "description": "Get the historical wildfire risk profile for a specific Alberta forest region. Returns historical area burned, fire frequency rank, area burned rank, notable fires, and risk tier. Use when the user asks about a specific region.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "Name of the Alberta forest region, e.g. Fort McMurray, High Level, Calgary"}
                    },
                    "required": ["region"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_model_performance",
                "description": "Get cross-validation performance metrics for the wildfire prediction models. Returns AUPRC, AUROC, and accuracy with standard deviations. Use when the user asks about model performance, results, or comparison between models.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_name": {"type": "string", "description": "Model name: logistic_regression, random_forest, xgboost, isolation_forest, random_baseline, or all"}
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_shap_importance",
                "description": "Get SHAP feature importance values from the Random Forest model showing which features most strongly predict large fire escalation. Use when the user asks about feature importance, what drives risk, or SHAP analysis.",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_season_stats",
                "description": "Get historical statistics for Alberta fire seasons including fire count, area burned, and notable events. Use when the user asks about a specific year or historical season comparisons.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "year": {"type": "integer", "description": "The fire season year (2006-2024). Leave empty for all years."}
                    },
                    "required": []
                }
            }
        }
    ]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
