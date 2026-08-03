import json
import time
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException

from ..core.config import DATA_DIR

router = APIRouter(prefix="/api/map", tags=["map"])

# ---------------------------------------------------------------------------
# Static fire data
# ---------------------------------------------------------------------------

@router.get("/fires")
def get_fire_map():
    with open(DATA_DIR / "fire_map.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Live weather — Open-Meteo, no API key required
# ---------------------------------------------------------------------------

ALBERTA_STATIONS = [
    ("Edmonton",        53.546, -113.493),
    ("Calgary",         51.045, -114.072),
    ("Fort McMurray",   56.726, -111.380),
    ("Lethbridge",      49.694, -112.832),
    ("Red Deer",        52.268, -113.811),
    ("Medicine Hat",    50.040, -110.677),
    ("Grande Prairie",  55.170, -118.797),
    ("Lloydminster",    53.278, -110.005),
    ("Camrose",         53.017, -112.827),
    ("Brooks",          50.564, -111.898),
    ("Banff",           51.178, -115.571),
    ("Jasper",          52.873, -118.082),
    ("Lac La Biche",    54.767, -111.967),
    ("Slave Lake",      55.430, -114.773),
    ("High Level",      58.516, -117.136),
    ("Peace River",     56.237, -117.286),
    ("Wetaskiwin",      52.968, -113.371),
    ("Edson",           53.582, -116.439),
    ("Hinton",          53.407, -117.563),
    ("Drayton Valley",  53.217, -114.977),
]

_weather_cache: dict[str, Any] = {"data": None, "ts": 0.0}
_CACHE_TTL = 900  # 15 minutes


def _fire_risk(temp: float, rh: float, wind: float) -> float:
    """Simple 0-1 fire weather risk index based on temperature, RH, and wind."""
    t = min(1.0, max(0.0, (temp - 10) / 30))   # 10°C → 0, 40°C → 1
    h = min(1.0, max(0.0, (60 - rh) / 55))      # 60% RH → 0,  5% RH → 1
    w = min(1.0, max(0.0, wind / 60))            # 0 km/h → 0, 60+ → 1
    return round(t * 0.4 + h * 0.4 + w * 0.2, 3)


@router.get("/weather")
async def get_weather():
    now = time.time()
    if _weather_cache["data"] and now - _weather_cache["ts"] < _CACHE_TTL:
        return _weather_cache["data"]

    lats = ",".join(str(s[1]) for s in ALBERTA_STATIONS)
    lons = ",".join(str(s[2]) for s in ALBERTA_STATIONS)
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lats}&longitude={lons}"
        "&current=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation"
        "&timezone=America%2FDenver&forecast_days=1"
    )

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            raw = resp.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Weather API error: {exc}")

    # Open-Meteo returns a list when multiple locations are requested
    if not isinstance(raw, list):
        raw = [raw]

    stations = []
    for i, item in enumerate(raw):
        name, lat, lon = ALBERTA_STATIONS[i]
        cur = item.get("current", {})
        temp = cur.get("temperature_2m", 0)
        rh   = cur.get("relative_humidity_2m", 50)
        wind = cur.get("wind_speed_10m", 0)
        wdir = cur.get("wind_direction_10m", 0)
        precip = cur.get("precipitation", 0)
        stations.append({
            "name":    name,
            "lat":     round(item["latitude"], 4),
            "lon":     round(item["longitude"], 4),
            "temp":    temp,
            "rh":      rh,
            "wind":    wind,
            "wdir":    wdir,
            "precip":  precip,
            "risk":    _fire_risk(temp, rh, wind),
        })

    result = {"updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stations": stations}
    _weather_cache["data"] = result
    _weather_cache["ts"]   = now
    return result
