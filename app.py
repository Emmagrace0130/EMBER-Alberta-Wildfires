"""
app.py — EMBER Alberta Wildfire Risk Assessor
Flask backend serving the chat interface and /api/assess endpoint.
"""

from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from risk_model import (
    FOREST_REGION_OPTIONS,
    IGNITION_CAUSE_OPTIONS,
    SPREAD_RATE_OPTIONS,
    compute_risk_score,
    get_risk_level,
)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template(
        "index.html",
        spread_rate_options=SPREAD_RATE_OPTIONS,
        ignition_cause_options=IGNITION_CAUSE_OPTIONS,
        forest_region_options=FOREST_REGION_OPTIONS,
    )


@app.route("/api/assess", methods=["POST"])
def assess():
    """
    POST /api/assess
    Body (JSON):
      fire_size       float  — hectares
      spread_rate     str
      wind_speed      float  — km/h
      temperature     float  — °C
      humidity        float  — %
      ignition_cause  str
      forest_region   str

    Response (JSON):
      risk_score      float  — 0–100 (%)
      risk_level      str    — LOW / MODERATE / HIGH / EXTREME
      factors         list   — top contributing features
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    required = [
        "fire_size",
        "spread_rate",
        "wind_speed",
        "temperature",
        "humidity",
        "ignition_cause",
        "forest_region",
    ]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        fire_size     = float(data["fire_size"])
        wind_speed    = float(data["wind_speed"])
        temperature   = float(data["temperature"])
        humidity      = float(data["humidity"])
    except (TypeError, ValueError):
        return jsonify({"error": "One or more numeric fields contain invalid values."}), 400

    spread_rate    = str(data["spread_rate"])
    ignition_cause = str(data["ignition_cause"])
    forest_region  = str(data["forest_region"])

    score, factors = compute_risk_score(
        fire_size,
        spread_rate,
        wind_speed,
        temperature,
        humidity,
        ignition_cause,
        forest_region,
    )

    return jsonify(
        {
            "risk_score": round(score * 100, 1),
            "risk_level": get_risk_level(score),
            "factors": factors,
        }
    )


if __name__ == "__main__":
    import os
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug)
