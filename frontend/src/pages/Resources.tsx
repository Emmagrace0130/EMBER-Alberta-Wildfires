import { useEffect, useState } from "react";
import { Download, FileText, Search } from "lucide-react";
import Hero from "../components/Hero";
import Card from "../components/Card";
import { fetchResources, search as searchApi } from "../lib/api";
import type { Resource, SearchResultItem } from "../data/types";

export default function Resources() {
  const [resources, setResources] = useState<Resource[]>([]);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResultItem[] | null>(null);

  useEffect(() => {
    fetchResources().then(setResources).catch(() => undefined);
  }, []);

  useEffect(() => {
    const handle = setTimeout(() => {
      if (!query.trim()) {
        setResults(null);
        return;
      }
      searchApi(query)
        .then((r) => setResults(r.results))
        .catch(() => setResults([]));
    }, 250);
    return () => clearTimeout(handle);
  }, [query]);

  return (
    <div>
      <Hero
        eyebrow="Resources / Library"
        title="A searchable knowledge center for wildfire mitigation"
        description="Guides, templates, research reports, and EMBER's own findings — all in one place."
      >
        <div className="relative mt-10 max-w-lg">
          <Search className="absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-offwhite/40" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search resources and mitigation plans..."
            className="w-full rounded-full border border-white/10 bg-white/5 py-3 pl-11 pr-4 text-sm text-offwhite placeholder:text-offwhite/40 focus:border-ember/50 focus:outline-none"
          />
        </div>
      </Hero>

      <section className="section">
        {results !== null ? (
          <>
            <h3 className="mb-6 text-sm font-semibold uppercase tracking-wide text-offwhite/40">
              {results.length} result{results.length === 1 ? "" : "s"} for "{query}"
            </h3>
            <div className="grid gap-6 sm:grid-cols-2">
              {results.map((r) => (
                <Card key={`${r.type}-${r.id}`}>
                  <span className="pill mb-3 inline-block">{r.type}</span>
                  <h4 className="text-base font-semibold">{r.title}</h4>
                  <p className="mt-2 text-sm text-offwhite/65">{r.snippet}</p>
                </Card>
              ))}
            </div>
          </>
        ) : (
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {resources.map((resource, i) => (
              <Card key={resource.id} delay={i * 0.05} className="flex flex-col">
                <div className="flex items-center gap-3">
                  <FileText className="h-5 w-5 flex-none text-ember" />
                  <span className="pill">{resource.type}</span>
                </div>
                <h4 className="mt-3 text-base font-semibold">{resource.title}</h4>
                <p className="mt-2 flex-1 text-sm text-offwhite/65">{resource.description}</p>
                {resource.downloadUrl ? (
                  <a
                    href={resource.downloadUrl}
                    className="mt-4 inline-flex items-center gap-2 text-sm font-medium text-ember hover:text-ember-light"
                  >
                    <Download className="h-4 w-4" /> Download
                  </a>
                ) : (
                  resource.externalNote && (
                    <p className="mt-4 text-xs text-offwhite/40">{resource.externalNote}</p>
                  )
                )}
              </Card>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
