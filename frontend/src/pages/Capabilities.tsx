import { useEffect, useState } from "react";
import Hero from "../components/Hero";
import SectionHeader from "../components/SectionHeader";
import Card from "../components/Card";
import { fetchCapabilities } from "../lib/api";
import type { Capability } from "../data/types";

const statusColor: Record<string, string> = {
  Available: "text-ember",
  "Coming soon": "text-amber",
  Planned: "text-offwhite/50",
};

export default function Capabilities() {
  const [capabilities, setCapabilities] = useState<Capability[]>([]);

  useEffect(() => {
    fetchCapabilities().then(setCapabilities).catch(() => undefined);
  }, []);

  const categories = Array.from(new Set(capabilities.map((c) => c.category)));

  return (
    <div>
      <Hero
        eyebrow="What EMBER Can Do"
        title="A platform, not just a static website"
        description="EMBER organizes mitigation knowledge today, and is built to grow into AI-assisted prediction and guidance — without starting from scratch."
      />

      <section className="section">
        {categories.map((category) => (
          <div key={category} className="mb-12">
            <h3 className="mb-6 text-sm font-semibold uppercase tracking-wide text-offwhite/40">
              {category}
            </h3>
            <div className="grid gap-6 sm:grid-cols-2">
              {capabilities
                .filter((c) => c.category === category)
                .map((cap, i) => (
                  <Card key={cap.id} delay={i * 0.06} className="flex flex-col gap-2">
                    <div className="flex items-center justify-between">
                      <h4 className="text-base font-semibold">{cap.title}</h4>
                      <span className={`pill ${statusColor[cap.status] ?? ""}`}>{cap.status}</span>
                    </div>
                    <p className="text-sm text-offwhite/65">{cap.description}</p>
                  </Card>
                ))}
            </div>
          </div>
        ))}
      </section>

      <SectionHeader
        align="center"
        eyebrow="Roadmap"
        title="What's next"
        description="EMBER's data and document library are already built — capabilities marked 'Coming soon' are the next layer, connecting this same content to live predictions and conversational guidance."
      />
    </div>
  );
}
