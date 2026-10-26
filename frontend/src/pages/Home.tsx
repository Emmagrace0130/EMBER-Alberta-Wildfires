import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { Flame, MapPinned, ShieldCheck, Sparkles } from "lucide-react";
import Hero from "../components/Hero";
import SectionHeader from "../components/SectionHeader";
import StatCard from "../components/StatCard";
import Card from "../components/Card";
import { fetchCapabilities, fetchPlans, fetchRiskInsights } from "../lib/api";
import type { Capability, MitigationPlan, RiskInsights } from "../data/types";

export default function Home() {
  const [risk, setRisk] = useState<RiskInsights | null>(null);
  const [plans, setPlans] = useState<MitigationPlan[]>([]);
  const [capabilities, setCapabilities] = useState<Capability[]>([]);

  useEffect(() => {
    fetchRiskInsights().then(setRisk).catch(() => undefined);
    fetchPlans().then((p) => setPlans(p.slice(0, 3))).catch(() => undefined);
    fetchCapabilities().then((c) => setCapabilities(c.slice(0, 4))).catch(() => undefined);
  }, []);

  return (
    <div>
      <Hero
        eyebrow="Wildfire mitigation & decision support"
        title="EMBER helps communities understand wildfire risk and act before disaster happens."
        description="EMBER is a wildfire mitigation and decision-support platform designed to help communities explore risk, view mitigation strategies, and understand actions that can reduce wildfire impacts before a fire begins."
        primaryCta={{ label: "Explore Mitigation Plans", to: "/mitigation-plans" }}
        secondaryCta={{ label: "View Risk Insights", to: "/risk-insights" }}
      >
        {risk && (
          <div className="mt-16 grid grid-cols-2 gap-4 sm:grid-cols-4">
            <StatCard value={risk.totalFires.toLocaleString()} label="Fires analyzed (2006-2024)" delay={0.05} />
            <StatCard value={`${risk.largeFireRate}%`} label="Become large fires" delay={0.1} />
            <StatCard value={`${Math.round(risk.recall * 100)}%`} label="Large-fire escalations flagged" delay={0.15} />
            <StatCard value={risk.modelMetrics[0]?.auprc.toFixed(3) ?? "—"} label="Best model AUPRC" delay={0.2} />
          </div>
        )}
      </Hero>

      <section className="section">
        <SectionHeader
          eyebrow="What EMBER does"
          title="From raw wildfire data to clear, actionable mitigation guidance"
          description="EMBER organizes Alberta wildfire history, mitigation documents, and risk patterns into one platform — built for communities, planners, and researchers."
        />
        <div className="grid gap-6 sm:grid-cols-3">
          <Card>
            <Flame className="mb-4 h-6 w-6 text-ember" />
            <h3 className="text-lg font-semibold">Risk, explained plainly</h3>
            <p className="mt-2 text-sm text-offwhite/65">
              26,551 real Alberta wildfire records translated into clear,
              human-readable risk patterns — no statistics background required.
            </p>
          </Card>
          <Card delay={0.1}>
            <ShieldCheck className="mb-4 h-6 w-6 text-ember" />
            <h3 className="text-lg font-semibold">Mitigation, organized</h3>
            <p className="mt-2 text-sm text-offwhite/65">
              FireSmart guidance, prevention templates, and research reports
              indexed by category, hazard level, and region.
            </p>
          </Card>
          <Card delay={0.2}>
            <Sparkles className="mb-4 h-6 w-6 text-ember" />
            <h3 className="text-lg font-semibold">Built to grow</h3>
            <p className="mt-2 text-sm text-offwhite/65">
              The same data and document library will power AI-assisted
              prediction and mitigation guidance as EMBER expands.
            </p>
          </Card>
        </div>
      </section>

      <section className="section">
        <SectionHeader
          eyebrow="Why mitigation matters"
          title="A small share of fires cause most of the damage"
          description={
            risk
              ? `Only ${risk.largeFireRate}% of the ${risk.totalFires.toLocaleString()} fires recorded since 2006 grew beyond 40 hectares — but those large fires drive the overwhelming majority of suppression cost and burned area. Acting on the right risk signals before a fire starts is where mitigation has the most leverage.`
              : "A small fraction of wildfires escalate to large, costly fires. Mitigation planning focused on the right risk signals has outsized impact."
          }
        />
        {capabilities.length > 0 && (
          <div className="grid gap-6 sm:grid-cols-2">
            {capabilities.map((cap, i) => (
              <Card key={cap.id} delay={i * 0.08} className="flex flex-col gap-2">
                <div className="flex items-center justify-between">
                  <h3 className="text-base font-semibold">{cap.title}</h3>
                  <span className="pill">{cap.status}</span>
                </div>
                <p className="text-sm text-offwhite/65">{cap.description}</p>
              </Card>
            ))}
          </div>
        )}
        <div className="mt-8">
          <Link to="/what-ember-can-do" className="btn-secondary">
            See everything EMBER can do
          </Link>
        </div>
      </section>

      <section className="section">
        <SectionHeader
          eyebrow="Featured mitigation plans"
          title="Real guidance, drawn from authoritative sources"
          description="Each plan in the library comes from an actual wildfire protection document — FireSmart guidebooks, prevention templates, and research reports."
        />
        <div className="grid gap-6 sm:grid-cols-3">
          {plans.map((plan, i) => (
            <Card key={plan.id} delay={i * 0.08}>
              <div className="mb-3 flex flex-wrap gap-2">
                {plan.categories.slice(0, 2).map((c) => (
                  <span key={c} className="pill">{c}</span>
                ))}
              </div>
              <h3 className="text-lg font-semibold">{plan.title}</h3>
              <p className="mt-2 text-sm text-offwhite/65">{plan.summary}</p>
              <Link
                to={`/mitigation-plans/${plan.id}`}
                className="mt-4 inline-block text-sm font-medium text-ember hover:text-ember-light"
              >
                View plan →
              </Link>
            </Card>
          ))}
        </div>
      </section>

      <section className="section">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="glass-panel flex flex-col items-start gap-6 p-10 sm:flex-row sm:items-center sm:justify-between"
        >
          <div>
            <h2 className="text-2xl font-semibold sm:text-3xl">
              Ready to explore wildfire risk in your area?
            </h2>
            <p className="mt-2 max-w-lg text-sm text-offwhite/65">
              Browse mitigation plans, dig into risk insights, or reach out to
              talk about a community partnership.
            </p>
          </div>
          <div className="flex gap-3">
            <Link to="/how-it-works" className="btn-secondary">
              <MapPinned className="h-4 w-4" /> How it works
            </Link>
            <Link to="/contact" className="btn-primary">
              Contact us
            </Link>
          </div>
        </motion.div>
      </section>
    </div>
  );
}
