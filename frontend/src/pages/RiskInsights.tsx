import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import Hero from "../components/Hero";
import SectionHeader from "../components/SectionHeader";
import StatCard from "../components/StatCard";
import Card from "../components/Card";
import { fetchRiskInsights } from "../lib/api";
import type { RiskInsights as RiskInsightsType } from "../data/types";

const chartTitles: Record<string, string> = {
  "fig_annual_trends.png": "Annual fire count vs. area burned (2006-2024)",
  "fig_model_comparison.png": "Model comparison by AUPRC",
  "fig_prc_curves.png": "Precision-recall curves (10-fold CV)",
  "fig_roc_curves.png": "ROC curves (10-fold CV)",
  "fig_shap.png": "SHAP feature importance",
  "fig_size_distribution.png": "Fire size class distribution",
};

const tooltipStyle = {
  background: "#15171b",
  border: "1px solid rgba(255,255,255,0.1)",
  borderRadius: 12,
  color: "#f5f1ea",
  fontSize: 12,
};

export default function RiskInsights() {
  const [risk, setRisk] = useState<RiskInsightsType | null>(null);

  useEffect(() => {
    fetchRiskInsights().then(setRisk).catch(() => undefined);
  }, []);

  if (!risk) {
    return <div className="section text-offwhite/50">Loading risk insights...</div>;
  }

  return (
    <div>
      <Hero
        eyebrow="Risk Insights"
        title="18 years of Alberta wildfire data, made readable"
        description="Seasonal trends, cause patterns, and the conditions that most strongly predict a fire will escalate — drawn directly from 26,551 Forest Protection Area wildfire records."
      />

      <section className="section pt-8">
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <StatCard value={risk.totalFires.toLocaleString()} label={`Fires recorded, ${risk.yearsSpanned}`} />
          <StatCard value={`${risk.largeFireRate}%`} label={`Escalate to large fires (${risk.largeFireCount.toLocaleString()} fires)`} delay={0.05} />
          <StatCard value={`${Math.round(risk.recall * 100)}%`} label="Large-fire recall" delay={0.1} />
          <StatCard value={risk.modelMetrics[0]?.auprc.toFixed(3) ?? "—"} label={`Best AUPRC (${risk.modelMetrics[0]?.model})`} delay={0.15} />
        </div>
      </section>

      <section className="section">
        <SectionHeader
          eyebrow="Seasonal trend"
          title="Fire activity by year"
          description="Fire counts vary considerably year to year, shaped by drought, lightning activity, and weather. Lower counts in recent years don't necessarily mean lower risk per fire."
        />
        <Card className="h-80 p-4">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={risk.annualTrend} margin={{ top: 10, right: 16, left: -16, bottom: 0 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
              <XAxis dataKey="year" stroke="#8b8d92" fontSize={12} />
              <YAxis stroke="#8b8d92" fontSize={12} />
              <Tooltip contentStyle={tooltipStyle} />
              <Line type="monotone" dataKey="fireCount" stroke="#ff6a3d" strokeWidth={2.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </Card>
      </section>

      <section className="section">
        <SectionHeader
          eyebrow="Cause patterns"
          title="What starts these fires"
          description="Lightning causes more fires than any single human activity — but recreation and resident-caused fires together account for more than a third of all ignitions."
        />
        <Card className="h-96 p-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={risk.causeBreakdown}
              layout="vertical"
              margin={{ top: 10, right: 24, left: 16, bottom: 0 }}
            >
              <CartesianGrid stroke="rgba(255,255,255,0.06)" horizontal={false} />
              <XAxis type="number" stroke="#8b8d92" fontSize={12} unit="%" />
              <YAxis
                type="category"
                dataKey="cause"
                stroke="#8b8d92"
                fontSize={11}
                width={150}
              />
              <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => `${v}%`} />
              <Bar dataKey="percent" fill="#e8b923" radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </section>

      <section className="section">
        <SectionHeader
          eyebrow="Escalation drivers"
          title="What predicts a fire will become a large fire"
          description="Ranked by importance in EMBER's trained Random Forest model. Assessment hectares and fire spread rate alone explain about two-thirds of predictive power."
        />
        <Card className="h-96 p-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={risk.featureImportance}
              layout="vertical"
              margin={{ top: 10, right: 24, left: 16, bottom: 0 }}
            >
              <CartesianGrid stroke="rgba(255,255,255,0.06)" horizontal={false} />
              <XAxis type="number" stroke="#8b8d92" fontSize={12} />
              <YAxis
                type="category"
                dataKey="feature"
                stroke="#8b8d92"
                fontSize={11}
                width={150}
              />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="importance" fill="#ff6a3d" radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
        <div className="mt-6 grid gap-4 sm:grid-cols-2">
          {risk.featureImportance.slice(0, 4).map((f) => (
            <Card key={f.feature} className="text-sm">
              <span className="font-semibold text-offwhite">{f.feature}</span>
              <p className="mt-1 text-offwhite/60">{f.direction}</p>
            </Card>
          ))}
        </div>
      </section>

      <section className="section">
        <SectionHeader
          eyebrow="Model performance"
          title="Generated analysis figures"
          description="These charts come directly from EMBER's model training pipeline."
        />
        <div className="grid gap-6 sm:grid-cols-2">
          {risk.charts.map((chart, i) => (
            <Card key={chart} delay={i * 0.05}>
              <img
                src={`/assets/charts/${chart}`}
                alt={chartTitles[chart] ?? chart}
                className="w-full rounded-lg border border-white/10"
                loading="lazy"
              />
              <p className="mt-3 text-sm text-offwhite/60">{chartTitles[chart] ?? chart}</p>
            </Card>
          ))}
        </div>
      </section>
    </div>
  );
}
