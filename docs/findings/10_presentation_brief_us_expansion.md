# EMBER — Presentation Brief for Visiting Researchers
**Prepared:** 2026-08-03
**Purpose:** Source material for slides introducing EMBER and its expansion toward a US-relevant platform.

---

## 1. What EMBER Is (One-Slide Pitch)

EMBER is a wildfire risk and mitigation decision-support platform. It has three connected layers:

1. **Prediction (ML):** Given fire and weather conditions, predict the probability a wildfire escalates into a *large fire* (≥ 40 hectares).
2. **Guidance (RAG/GRAG):** Retrieve grounded mitigation advice from authoritative wildfire-protection documents, using a vector store (ChromaDB) plus a Neo4j knowledge graph.
3. **Interface (Agent + Web App):** A ReAct LLM agent (local, via Ollama — no external API calls) ties prediction and retrieval together and answers natural-language questions; a React/FastAPI web app exposes this to end users (mitigation plan library, risk insights, chat, fire map).

**Core message:** EMBER doesn't just predict risk — it connects that prediction to *what to do about it*, sourced from real planning documents rather than generic LLM knowledge.

---

## 2. The Data & Model Foundation (Currently Canada/Alberta-Based)

| Property | Value |
|---|---|
| Source | Alberta Forest Protection Area historical wildfire records |
| Records | 26,551 total (25,862 used for modeling) |
| Time span | 2006–2024 (18 years) |
| Target | "Large fire" = Size Class D/E, ≥ 40 hectares |
| Positive rate | 3.34% (865 large fires vs. 24,997 small) |
| Features | 12, engineered from ~50 raw columns (assessment hectares, fire spread rate, wind speed, cause, temperature, humidity, dispatch/detection lag, fuel type, etc.) |

**Models:** Logistic Regression (AUPRC 0.584, chosen for deployment — best primary metric, interpretable, fast) and Random Forest (AUPRC 0.513, AUROC 0.959, kept for SHAP-based feature attribution). Both exceed AUROC 0.95. Class imbalance handled via SMOTE + class weighting.

**Top predictors:** Assessment hectares (~42% importance) and fire spread rate (~24%) — together explain roughly two-thirds of predictive power.

This model, and the underlying dataset, is entirely Alberta/Canada-specific. It is the part of EMBER that is *not yet* US-based — the US expansion so far lives in the guidance/document layer, not the predictive model.

---

## 3. How EMBER Is Becoming US-Relevant (The "US-Based" Angle)

This is the accurate, current state of US work in the repo — useful for framing honestly to researchers rather than overclaiming.

### What's already in place
The **mitigation guidance library** (the RAG corpus and the "Mitigation Plans" section of the web app) is deliberately **multi-jurisdictional**, not Alberta-only. It already blends:

- **Alberta sources** — FireSmart Community Protection Guidebook, Farm & Acreage Wildfire Risk Reduction Guide, Alberta WUI Fire Analysis
- **Canada (national) sources** — CWFIS (Canadian Wildland Fire Information System) data services guide
- **United States sources** —
  - *Wildfire Prevention, Education & Mitigation Planning Guide* (US Bureau of Land Management, 2009) — community risk assessment frameworks, public education campaign design, prevention program metrics
  - *Wildfire Risk Assessment Methodology* (US Forest Service, Rocky Mountain Research Station, RMRS-GTR-292) — the technical/methodological foundation EMBER's own risk-scoring and feature-importance framing draws on

The frontend's Mitigation Plans page has jurisdiction filtering built in (🍁 Alberta / 🇨🇦 Canada / 🇺🇸 United States tabs), so the platform is architected to present region-specific guidance side by side rather than assuming one country.

### What this means in practice
- EMBER's **prediction model** answers "how risky is this fire, based on Alberta's 18-year history?"
- EMBER's **guidance layer** already answers "what should be done about it?" using US federal methodology and planning templates alongside Canadian ones — because wildfire mitigation practice (defensible space, fuel treatment, WUI planning) generalizes across the US-Canada border more readily than fire-behavior statistics do.
- The USFS RMRS-GTR-292 report specifically underpins how EMBER frames risk categories and feature importance — i.e., the US research literature already shapes EMBER's methodology, even before US fire data is added.

### What's not yet done (be upfront about this with researchers)
- No US wildfire incident dataset (e.g., a state agency's historical fire records, or a national source like NIFC/WFIGS) has been incorporated into the **predictive model** yet. The ML pipeline (`src/`, `best_try.ipynb`) is still trained exclusively on Alberta FPA data.
- There is no US-specific model, retraining, or transfer-learning step yet — the natural next phase, if the goal is a genuinely US-calibrated risk model, is sourcing a comparable US wildfire dataset with matching fields (fire size class, cause, dispatch timing, weather at time of fire) and either retraining or building a second model alongside the Alberta one.
- Phase 2 (FastAPI backend) and Phase 3 (React frontend) are functional for the document/plans/insights experience; the live prediction (`/predict`) and RAG chat (`/ask`) endpoints are marked "Coming soon" in the app itself — i.e., agent + prediction integration is validated in notebooks but not yet wired into the production web app.

**Suggested framing for slides:** "EMBER started as an Alberta-specific wildfire escalation model. We're broadening it into a US-relevant platform in two stages: (1) mitigation guidance — already multi-jurisdictional, blending Canadian and US federal sources — and (2) predictive modeling — the next step, which requires a comparable US wildfire dataset to train a US-calibrated risk model alongside the existing Alberta one." This is honest about current state and gives researchers a clear opening to suggest US data sources or collaboration.

---

## 4. Architecture Snapshot (For a Diagram Slide)

```
User Question / Fire Conditions
         │
         ▼
   ReAct Agent (llama3.1:8b via Ollama, local — no external API)
         │
   ┌─────┼──────────────┐
   ▼     ▼              ▼
ML Models   ChromaDB (RAG)   Neo4j Knowledge Graph (GRAG)
(RF + LR)   542 chunks,      Documents → Chunks → Concepts
            embeddinggemma   + analytical layer (Feature,
                              RiskLevel, MitigationAction,
                              DataInsight nodes)
         │
   FINAL ANSWER: risk level + mitigation advice, grounded in data & documents
```

**Web app:** React + Vite frontend, FastAPI backend, containerized via Docker Compose. Pages: Home, About, How It Works, Mitigation Plans (jurisdiction-filterable), Risk Insights, Fire Map, Capabilities, Chat, Resources, Contact.

---

## 5. Suggested Slide Outline

1. **Title** — EMBER: Wildfire Risk & Mitigation Decision Support
2. **The problem** — Large fires are rare (3.3%) but dominate cost/damage; mitigation has more leverage than suppression
3. **What EMBER does** — prediction + grounded guidance + conversational agent (one slide, the pitch from §1)
4. **The data & model** — Alberta dataset stats, model performance (§2)
5. **Architecture** — the diagram (§4)
6. **Expanding to a US context** — what's already multi-jurisdictional (guidance library, §3) vs. what's next (US predictive data, §3)
7. **Where we'd value input from visiting researchers** — e.g., candidate US wildfire datasets, cross-border methodology validity, model transfer considerations
8. **Roadmap** — Phase checklist (agent/prediction wiring into production endpoints, Docker Compose finalization, evaluation harness)

---

*Source materials for this brief: `README.md`, `PLAN.md`, `.github/copilot-instructions.md`, `docs/findings/01_executive_summary.md`, `docs/findings/08_session_log_2026-04-15.md`, `backend/app/data/plans.json`, `backend/app/data/resources.json`, `backend/app/data/capabilities.json`, `frontend/src/pages/MitigationPlans.tsx`, `frontend/src/pages/About.tsx`.*
