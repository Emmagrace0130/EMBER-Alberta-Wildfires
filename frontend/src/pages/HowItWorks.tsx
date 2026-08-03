import { ArrowRight, Database, FileSearch, Map, MessageSquareText } from "lucide-react";
import Hero from "../components/Hero";
import SectionHeader from "../components/SectionHeader";
import Card from "../components/Card";

const steps = [
  {
    icon: Database,
    title: "Data sources",
    description:
      "18 years of Alberta Forest Protection Area wildfire records (2006-2024) — fire size, cause, weather conditions, response timing — plus six authoritative mitigation and prevention documents.",
  },
  {
    icon: FileSearch,
    title: "Risk modeling",
    description:
      "Statistical models trained on the dataset rank which conditions — fire spread rate, wind, dispatch lag — most strongly predict a fire will escalate past 40 hectares.",
  },
  {
    icon: Map,
    title: "Mitigation knowledge",
    description:
      "Source PDFs are broken into searchable sections and tagged by category — defensible space, fuel management, community planning — so guidance is easy to find.",
  },
  {
    icon: MessageSquareText,
    title: "What's next: conversational guidance",
    description:
      "The same risk models and document library are designed to power a future assistant that can answer specific mitigation questions, grounded in real sources.",
  },
];

export default function HowItWorks() {
  return (
    <div>
      <Hero
        eyebrow="How It Works"
        title="From historical data to mitigation guidance"
        description="EMBER's architecture connects a real wildfire dataset, a trained risk model, and an organized mitigation document library — built so future predictive features can plug into the same foundation."
      />

      <section className="section">
        <SectionHeader title="The pipeline, step by step" />
        <div className="flex flex-col gap-6">
          {steps.map((step, i) => {
            const Icon = step.icon;
            return (
              <div key={step.title} className="flex items-start gap-4">
                <Card delay={i * 0.08} className="flex-1">
                  <div className="flex items-center gap-3">
                    <div className="rounded-xl bg-ember/10 p-3">
                      <Icon className="h-5 w-5 text-ember" />
                    </div>
                    <h3 className="text-base font-semibold">{step.title}</h3>
                  </div>
                  <p className="mt-3 text-sm text-offwhite/65">{step.description}</p>
                </Card>
                {i < steps.length - 1 && (
                  <ArrowRight className="mt-8 hidden h-5 w-5 flex-none text-offwhite/20 sm:block" />
                )}
              </div>
            );
          })}
        </div>
      </section>

      <section className="section">
        <SectionHeader
          eyebrow="Methodology"
          title="Why we trust these numbers"
          description="Models are evaluated with 10-fold stratified cross-validation and SMOTE oversampling applied only inside each training fold, to avoid leaking information about rare large-fire cases into validation. AUPRC — not accuracy — is treated as the primary metric, because large fires make up only 3.27% of records."
        />
        <div className="grid gap-6 sm:grid-cols-2">
          <Card>
            <h3 className="text-sm font-semibold">Visualization tools</h3>
            <p className="mt-2 text-sm text-offwhite/65">
              Risk Insights renders live charts (annual trend, cause breakdown,
              feature importance) directly from the underlying dataset, plus
              the original model-evaluation figures generated during training.
            </p>
          </Card>
          <Card delay={0.1}>
            <h3 className="text-sm font-semibold">Document retrieval</h3>
            <p className="mt-2 text-sm text-offwhite/65">
              Mitigation plans and resources are indexed by category, hazard
              level, and region so they're easy to filter and search — the
              same indexing that will later support semantic, AI-assisted
              search.
            </p>
          </Card>
        </div>
      </section>
    </div>
  );
}
