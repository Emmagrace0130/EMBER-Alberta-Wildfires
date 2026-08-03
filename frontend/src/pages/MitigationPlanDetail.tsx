import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft, Download, FileText } from "lucide-react";
import { fetchPlan } from "../lib/api";
import type { MitigationPlan } from "../data/types";
import Card from "../components/Card";

export default function MitigationPlanDetail() {
  const { id } = useParams<{ id: string }>();
  const [plan, setPlan] = useState<MitigationPlan | null>(null);
  const [notFound, setNotFound] = useState(false);

  useEffect(() => {
    if (!id) return;
    fetchPlan(id)
      .then(setPlan)
      .catch(() => setNotFound(true));
  }, [id]);

  if (notFound) {
    return (
      <div className="section">
        <p className="text-offwhite/70">That mitigation plan couldn't be found.</p>
        <Link to="/mitigation-plans" className="mt-4 inline-block text-ember">
          ← Back to Mitigation Plans
        </Link>
      </div>
    );
  }

  if (!plan) {
    return <div className="section text-offwhite/50">Loading...</div>;
  }

  return (
    <div className="section">
      <Link
        to="/mitigation-plans"
        className="mb-8 inline-flex items-center gap-2 text-sm text-offwhite/60 hover:text-ember"
      >
        <ArrowLeft className="h-4 w-4" /> Back to Mitigation Plans
      </Link>

      <div className="mb-6 flex flex-wrap gap-2">
        <span className="pill text-amber">{plan.hazardLevel} hazard</span>
        {plan.categories.map((c) => (
          <span key={c} className="pill">{c}</span>
        ))}
      </div>

      <h1 className="text-3xl font-semibold sm:text-4xl">{plan.title}</h1>
      <p className="mt-2 text-sm text-offwhite/50">
        {plan.source} · {plan.region}
      </p>

      <Card className="mt-8">
        <p className="text-base leading-relaxed text-offwhite/75">{plan.summary}</p>
      </Card>

      <div className="mt-8 grid gap-6 sm:grid-cols-2">
        <Card>
          <FileText className="mb-3 h-5 w-5 text-ember" />
          <h3 className="text-sm font-semibold">Source document</h3>
          <p className="mt-1 text-sm text-offwhite/65">
            {plan.fileName ?? "Internal research notes"}
            {plan.fileSizeMb ? ` · ${plan.fileSizeMb} MB` : ""}
          </p>
          <p className="mt-1 text-sm text-offwhite/50">
            {plan.chunkCount} indexed text segments
          </p>
        </Card>

        {plan.downloadUrl && (
          <Card className="flex flex-col items-start justify-between">
            <div>
              <h3 className="text-sm font-semibold">Download the full document</h3>
              <p className="mt-1 text-sm text-offwhite/65">
                Get the original PDF this plan summary is based on.
              </p>
            </div>
            <a href={plan.downloadUrl} className="btn-primary mt-4">
              <Download className="h-4 w-4" /> Download PDF
            </a>
          </Card>
        )}
      </div>
    </div>
  );
}
