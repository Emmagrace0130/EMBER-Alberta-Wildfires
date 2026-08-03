import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Download, Search } from "lucide-react";
import SectionHeader from "../components/SectionHeader";
import Card from "../components/Card";
import { fetchPlans } from "../lib/api";
import type { MitigationPlan } from "../data/types";

const JURISDICTION_TABS = [
  { id: "all",           label: "All Regions",    flag: "🌎" },
  { id: "Alberta",       label: "Alberta",         flag: "🍁" },
  { id: "Canada",        label: "Canada",          flag: "🇨🇦" },
  { id: "United States", label: "United States",   flag: "🇺🇸" },
];

export default function MitigationPlans() {
  const [plans, setPlans]           = useState<MitigationPlan[]>([]);
  const [query, setQuery]           = useState("");
  const [activeTag, setActiveTag]   = useState<string | null>(null);
  const [jurisdiction, setJuris]    = useState("all");
  const [loading, setLoading]       = useState(true);

  useEffect(() => {
    fetchPlans()
      .then(setPlans)
      .finally(() => setLoading(false));
  }, []);

  const allTags = useMemo(() => {
    const tags = new Set<string>();
    plans.forEach((p) => p.categories.forEach((c) => tags.add(c)));
    return Array.from(tags).sort();
  }, [plans]);

  const filtered = useMemo(() => {
    return plans.filter((p) => {
      const matchesJuris = jurisdiction === "all" || p.jurisdiction === jurisdiction;
      const matchesQuery =
        !query ||
        `${p.title} ${p.summary} ${p.region}`.toLowerCase().includes(query.toLowerCase());
      const matchesTag = !activeTag || p.categories.includes(activeTag);
      return matchesJuris && matchesQuery && matchesTag;
    });
  }, [plans, query, activeTag, jurisdiction]);

  // Count per jurisdiction for badge numbers
  const counts = useMemo(() => {
    const c: Record<string, number> = { all: plans.length };
    plans.forEach((p) => {
      c[p.jurisdiction] = (c[p.jurisdiction] ?? 0) + 1;
    });
    return c;
  }, [plans]);

  return (
    <div>
      <section className="section pb-0">
        <SectionHeader
          eyebrow="Mitigation Plans"
          title="A library of mitigation guidance, drawn from real sources"
          description="Community guidebooks, prevention templates, and research reports — organized by jurisdiction, category, and hazard level."
        />

        {/* Jurisdiction tabs */}
        <div className="mb-6 flex flex-wrap gap-2 border-b border-white/10 pb-6">
          {JURISDICTION_TABS.map((tab) => {
              const active = jurisdiction === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => { setJuris(tab.id); setActiveTag(null); }}
                className={`flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition-all border ${
                  active
                    ? "border-ember/60 bg-ember/15 text-ember shadow-sm"
                    : "border-white/10 bg-white/5 text-offwhite/60 hover:text-offwhite hover:border-white/20"
                }`}
              >
                <span>{tab.flag}</span>
                <span>{tab.label}</span>
                <span
                  className={`rounded-full px-1.5 py-0.5 text-xs font-semibold ${
                    active ? "bg-ember/30 text-ember" : "bg-white/10 text-offwhite/40"
                  }`}
                >
                  {tab.id === "all" ? plans.length : (counts[tab.id] ?? 0)}
                </span>
              </button>
            );
          })}
        </div>

        {/* Search + category filter */}
        <div className="mb-10 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="relative w-full max-w-sm">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-offwhite/40" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search mitigation plans…"
              className="w-full rounded-full border border-white/10 bg-white/5 py-2.5 pl-9 pr-4 text-sm text-offwhite placeholder:text-offwhite/40 focus:border-ember/50 focus:outline-none"
            />
          </div>

          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => setActiveTag(null)}
              className={`pill transition-colors ${!activeTag ? "border-ember/50 text-ember" : ""}`}
            >
              All categories
            </button>
            {allTags.map((tag) => (
              <button
                key={tag}
                onClick={() => setActiveTag(tag)}
                className={`pill transition-colors ${
                  activeTag === tag ? "border-ember/50 text-ember" : ""
                }`}
              >
                {tag}
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className="section pt-0">
        {loading && <p className="text-sm text-offwhite/50">Loading mitigation plans…</p>}
        {!loading && filtered.length === 0 && (
          <p className="text-sm text-offwhite/50">No plans match that filter.</p>
        )}
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {filtered.map((plan, i) => (
            <Card key={plan.id} delay={i * 0.05} className="flex flex-col">
              <div className="mb-3 flex flex-wrap gap-2">
                <span className="pill text-amber">{plan.hazardLevel}</span>
                {/* Jurisdiction badge */}
                <span className="pill text-xs text-offwhite/50">
                  {plan.jurisdiction === "Alberta"       ? "🍁 Alberta" :
                   plan.jurisdiction === "Canada"        ? "🇨🇦 Canada" :
                   plan.jurisdiction === "United States" ? "🇺🇸 US" : plan.jurisdiction}
                </span>
              </div>
              <h3 className="text-lg font-semibold">{plan.title}</h3>
              <p className="mt-1 text-xs uppercase tracking-wide text-offwhite/40">{plan.region}</p>
              <p className="mt-3 flex-1 text-sm text-offwhite/65">{plan.summary}</p>
              <div className="mt-3 flex flex-wrap gap-1.5">
                {plan.categories.slice(0, 2).map((c) => (
                  <span key={c} className="pill text-xs">{c}</span>
                ))}
              </div>
              <div className="mt-5 flex items-center justify-between">
                <Link
                  to={`/mitigation-plans/${plan.id}`}
                  className="text-sm font-medium text-ember hover:text-ember-light"
                >
                  View details →
                </Link>
                {plan.downloadUrl && (
                  <a
                    href={plan.downloadUrl}
                    download
                    className="rounded-full border border-white/10 p-2 text-offwhite/60 hover:border-ember/50 hover:text-ember"
                    title="Download source PDF"
                  >
                    <Download className="h-4 w-4" />
                  </a>
                )}
              </div>
            </Card>
          ))}
        </div>
      </section>
    </div>
  );
}
