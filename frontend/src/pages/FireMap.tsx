import { useEffect, useRef, useState } from "react";
import { MapContainer, TileLayer, useMap, CircleMarker, Tooltip } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "leaflet.heat";
import { fetchFireMapData } from "../lib/api";
import type { FireMapData } from "../data/types";

// ── Types ─────────────────────────────────────────────────────────────────────

interface WeatherStation {
  name: string;
  lat: number;
  lon: number;
  temp: number;
  rh: number;
  wind: number;
  wdir: number;
  precip: number;
  risk: number;
}

interface WeatherData {
  updated: string;
  stations: WeatherStation[];
}

type Layer = "fires" | "temperature" | "humidity" | "wind" | "risk";

// ── Leaflet.heat augmentation ─────────────────────────────────────────────────

type HeatLayer = { addTo(map: L.Map): HeatLayer; remove(): void };
declare module "leaflet" {
  function heatLayer(pts: [number, number, number][], opts?: object): HeatLayer;
}

// ── Color helpers ─────────────────────────────────────────────────────────────

function lerp(a: string, b: string, t: number): string {
  const h = (hex: string) => [
    parseInt(hex.slice(1, 3), 16),
    parseInt(hex.slice(3, 5), 16),
    parseInt(hex.slice(5, 7), 16),
  ];
  const [ar, ag, ab] = h(a);
  const [br, bg, bb] = h(b);
  const r = Math.round(ar + (br - ar) * t);
  const g = Math.round(ag + (bg - ag) * t);
  const bb2 = Math.round(ab + (bb - ab) * t);
  return `rgb(${r},${g},${bb2})`;
}

function tempColor(v: number) {
  // -10°C (blue) → 15°C (teal) → 28°C (amber) → 40°C (ember)
  const t = Math.min(1, Math.max(0, (v + 10) / 50));
  if (t < 0.5) return lerp("#1e90ff", "#34d399", t * 2);
  return lerp("#34d399", "#ff6a3d", (t - 0.5) * 2);
}

function rhColor(v: number) {
  // 5% (ember) → 35% (amber) → 60% (green) → 100% (blue)
  const t = Math.min(1, Math.max(0, (v - 5) / 95));
  if (t < 0.45) return lerp("#ff6a3d", "#34d399", t / 0.45);
  return lerp("#34d399", "#1e90ff", (t - 0.45) / 0.55);
}

function windColor(v: number) {
  // 0 km/h (blue) → 20 (amber) → 50+ (ember)
  const t = Math.min(1, Math.max(0, v / 55));
  if (t < 0.5) return lerp("#1e90ff", "#ffb347", t * 2);
  return lerp("#ffb347", "#ff6a3d", (t - 0.5) * 2);
}

function riskColor(v: number) {
  // 0 (green) → 0.4 (amber) → 0.7 (orange) → 1 (red)
  if (v < 0.4) return lerp("#34d399", "#ffb347", v / 0.4);
  if (v < 0.7) return lerp("#ffb347", "#ff6a3d", (v - 0.4) / 0.3);
  return lerp("#ff6a3d", "#cc1100", (v - 0.7) / 0.3);
}

function stationColor(station: WeatherStation, layer: Layer): string {
  switch (layer) {
    case "temperature": return tempColor(station.temp);
    case "humidity":    return rhColor(station.rh);
    case "wind":        return windColor(station.wind);
    case "risk":        return riskColor(station.risk);
    default:            return "#ff6a3d";
  }
}

function windArrow(deg: number) {
  return `rotate(${deg}deg)`;
}

// ── Subcomponents ─────────────────────────────────────────────────────────────

function HeatmapLayer({ points }: { points: [number, number, number][] }) {
  const map = useMap();
  const layerRef = useRef<HeatLayer | null>(null);
  useEffect(() => {
    if (!points.length) return;
    layerRef.current = L.heatLayer(points, {
      radius: 18, blur: 22, maxZoom: 10, max: 1.0,
      gradient: {
        0.0: "#1a0a00", 0.2: "#7b1a00", 0.45: "#c9491f",
        0.65: "#ff6a3d", 0.85: "#ffb347", 1.0: "#fff176",
      },
    }).addTo(map);
    return () => { layerRef.current?.remove(); };
  }, [map, points]);
  return null;
}

function WeatherLayer({ stations, layer }: { stations: WeatherStation[]; layer: Layer }) {
  return (
    <>
      {stations.map((s) => (
        <CircleMarker
          key={s.name}
          center={[s.lat, s.lon]}
          radius={18}
          pathOptions={{
            fillColor: stationColor(s, layer),
            fillOpacity: 0.85,
            color: "rgba(255,255,255,0.25)",
            weight: 1.5,
          }}
        >
          <Tooltip direction="top" offset={[0, -10]} opacity={0.95}>
            <div style={{ minWidth: 160, fontFamily: "Inter, sans-serif", fontSize: 12 }}>
              <div style={{ fontWeight: 700, marginBottom: 4, color: "#ff6a3d" }}>{s.name}</div>
              <div>🌡 Temp: {s.temp.toFixed(1)}°C</div>
              <div>💧 Humidity: {s.rh}%</div>
              <div>
                💨 Wind: {s.wind.toFixed(1)} km/h{" "}
                <span style={{ display: "inline-block", transform: windArrow(s.wdir) }}>↑</span>
              </div>
              <div>🌧 Precip: {s.precip.toFixed(1)} mm</div>
              <div style={{ marginTop: 4, fontWeight: 600, color: riskColor(s.risk) }}>
                Fire Risk: {(s.risk * 100).toFixed(0)}%
              </div>
            </div>
          </Tooltip>
        </CircleMarker>
      ))}
    </>
  );
}

// ── Layer config ──────────────────────────────────────────────────────────────

const LAYERS: { id: Layer; label: string; gradient: string; from: string; to: string }[] = [
  { id: "fires",       label: "🔥 Fire Density",    gradient: "linear-gradient(to right,#7b1a00,#c9491f,#ff6a3d,#ffb347,#fff176)", from: "Low", to: "High" },
  { id: "risk",        label: "⚠ Fire Risk",         gradient: "linear-gradient(to right,#34d399,#ffb347,#ff6a3d,#cc1100)",          from: "Low", to: "Extreme" },
  { id: "temperature", label: "🌡 Temperature",       gradient: "linear-gradient(to right,#1e90ff,#34d399,#ff6a3d)",                  from: "Cold", to: "Hot" },
  { id: "humidity",    label: "💧 Humidity",          gradient: "linear-gradient(to right,#ff6a3d,#34d399,#1e90ff)",                  from: "Dry", to: "Humid" },
  { id: "wind",        label: "💨 Wind Speed",        gradient: "linear-gradient(to right,#1e90ff,#ffb347,#ff6a3d)",                  from: "Calm", to: "Strong" },
];

// ── Page ──────────────────────────────────────────────────────────────────────

export default function FireMap() {
  const [fireData,    setFireData]    = useState<FireMapData | null>(null);
  const [weatherData, setWeatherData] = useState<WeatherData | null>(null);
  const [activeLayer, setActiveLayer] = useState<Layer>("fires");
  const [fireError,   setFireError]   = useState(false);
  const [wxError,     setWxError]     = useState(false);

  useEffect(() => {
    fetchFireMapData().then(setFireData).catch(() => setFireError(true));
    fetchWeather().then(setWeatherData).catch(() => setWxError(true));
  }, []);

  const currentLayerDef = LAYERS.find((l) => l.id === activeLayer)!;

  return (
    <div className="flex flex-col" style={{ minHeight: "calc(100vh - 65px)" }}>
      {/* Header */}
      <div className="border-b border-white/10 bg-charcoal-light px-6 py-4 sm:px-8">
        <div className="mx-auto max-w-6xl">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <p className="text-xs font-semibold uppercase tracking-widest text-ember mb-1">
                Fire Map
              </p>
              <h1 className="text-2xl font-semibold sm:text-3xl">Alberta Wildfire & Weather Map</h1>
              <p className="mt-1 text-sm text-offwhite/60">
                {activeLayer === "fires"
                  ? fireData ? `${fireData.count.toLocaleString()} fires 2006–2024` : "Loading…"
                  : weatherData
                  ? `Live conditions at ${weatherData.stations.length} stations · updated ${weatherData.updated.slice(11, 16)} UTC`
                  : "Loading weather…"}
              </p>
            </div>
            {/* Legend */}
            <div className="flex items-center gap-3 rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-xs self-start">
              <span className="text-offwhite/50">{currentLayerDef.from}</span>
              <div className="h-3 w-28 rounded-full" style={{ background: currentLayerDef.gradient }} />
              <span className="text-offwhite/50">{currentLayerDef.to}</span>
            </div>
          </div>

          {/* Layer toggles */}
          <div className="mt-3 flex flex-wrap gap-2">
            {LAYERS.map((l) => (
              <button
                key={l.id}
                onClick={() => setActiveLayer(l.id)}
                className={`rounded-full px-3 py-1.5 text-xs font-medium transition-colors border ${
                  activeLayer === l.id
                    ? "border-ember/60 bg-ember/15 text-ember"
                    : "border-white/10 bg-white/5 text-offwhite/60 hover:text-offwhite"
                }`}
              >
                {l.label}
              </button>
            ))}
            {wxError && (
              <span className="text-xs text-red-400/70 self-center pl-2">
                Weather unavailable
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Map */}
      <div className="flex-1 relative" style={{ minHeight: 500 }}>
        {fireError && activeLayer === "fires" && (
          <div className="absolute inset-0 flex items-center justify-center text-offwhite/50 text-sm z-10">
            Failed to load fire data.
          </div>
        )}
        <MapContainer
          center={[55.0, -115.0]}
          zoom={5}
          style={{ height: "100%", width: "100%", minHeight: 500 }}
          scrollWheelZoom
        >
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
            maxZoom={19}
          />
          {activeLayer === "fires" && fireData && (
            <HeatmapLayer points={fireData.points} />
          )}
          {activeLayer !== "fires" && weatherData && (
            <WeatherLayer stations={weatherData.stations} layer={activeLayer} />
          )}
        </MapContainer>

        {/* Weather attribution */}
        {activeLayer !== "fires" && (
          <div className="absolute bottom-6 left-3 z-[1000] rounded-lg border border-white/10 bg-charcoal/80 px-2.5 py-1 text-[10px] text-offwhite/40 backdrop-blur-sm pointer-events-none">
            Weather: Open-Meteo.com · CC BY 4.0
          </div>
        )}
      </div>
    </div>
  );
}

// ── API helper (local to this file) ──────────────────────────────────────────

async function fetchWeather(): Promise<WeatherData> {
  const res = await fetch("/api/map/weather");
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}
