import { Compass, Database, Target, Telescope } from "lucide-react";
import Hero from "../components/Hero";
import SectionHeader from "../components/SectionHeader";
import Card from "../components/Card";

export default function About() {
  return (
    <div>
      <Hero
        eyebrow="About EMBER"
        title="Built from real wildfire data, for the people who plan around it"
        description="EMBER grew out of an analysis of 18 years of Alberta Forest Protection Area wildfire records — and the realization that the gap between research and the people who need it is mostly an organization problem, not a data problem."
      />

      <section className="section">
        <div className="grid gap-6 sm:grid-cols-2">
          <Card>
            <Target className="mb-4 h-6 w-6 text-ember" />
            <h3 className="text-lg font-semibold">Mission</h3>
            <p className="mt-2 text-sm text-offwhite/65">
              Make wildfire risk and mitigation guidance understandable and
              actionable for communities, planners, and researchers — not just
              specialists who already know where to look.
            </p>
          </Card>
          <Card delay={0.1}>
            <Telescope className="mb-4 h-6 w-6 text-ember" />
            <h3 className="text-lg font-semibold">Vision</h3>
            <p className="mt-2 text-sm text-offwhite/65">
              A platform that grows from a static knowledge base into an
              interactive assistant — one that can eventually answer
              "what should we do, here, given current conditions?"
            </p>
          </Card>
        </div>
      </section>

      <section className="section">
        <SectionHeader
          eyebrow="Why wildfire mitigation matters"
          title="The leverage is before the fire starts"
          description="Suppression is reactive and expensive. Mitigation — fuel management, defensible space, faster dispatch — changes the odds long before a fire is reported. EMBER's own analysis of Alberta wildfire history found that fire spread rate, wind, and dispatch lag all measurably shift escalation risk, which is exactly where prevention work has leverage."
        />
      </section>

      <section className="section">
        <SectionHeader
          eyebrow="How EMBER is different"
          title="A systems-thinking approach to wildfire planning"
        />
        <div className="grid gap-6 sm:grid-cols-2">
          <Card>
            <Compass className="mb-4 h-6 w-6 text-ember" />
            <h3 className="text-lg font-semibold">Connected, not siloed</h3>
            <p className="mt-2 text-sm text-offwhite/65">
              Mitigation documents, risk statistics, and (eventually) live
              predictions live in one platform instead of scattered PDFs and
              spreadsheets.
            </p>
          </Card>
          <Card delay={0.1}>
            <Database className="mb-4 h-6 w-6 text-ember" />
            <h3 className="text-lg font-semibold">Data-informed planning philosophy</h3>
            <p className="mt-2 text-sm text-offwhite/65">
              Every statistic on EMBER traces back to a real source — the
              26,551-record Alberta wildfire dataset (2006-2024) or an
              authoritative mitigation document — not a generic claim.
            </p>
          </Card>
        </div>
      </section>
    </div>
  );
}
