# EMBER — Project Plan & Progress Tracker

**Alberta Wildfire Large-Fire Escalation Risk Assessor**  
*Predictive ML + RAG-based mitigation guidance*

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│  React / Vite frontend  ─────────────────────  Port 5173        │
│   • Prediction form (12 input fields)                           │
│   • Chat UI for mitigation guidance                             │
│   • Risk probability gauge / explanation panel                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP REST
┌──────────────────────────▼──────────────────────────────────────┐
│                      FASTAPI BACKEND                            │
│  Python 3.12   ──────────────────────────────  Port 8000        │
│   POST /predict   → ML inference pipeline                       │
│   POST /ask       → RAG/agent query endpoint                    │
│   GET  /health    → health check                                │
└────────┬───────────────────────────────────────────────────────┘
         │
         ├──────────────────────────────────────────────────────┐
         │  ML MODELS (joblib)                                  │
         │  results/wildfire_model.pkl   (Random Forest 200T)  │
         │  results/lr_model.pkl         (Logistic Regression) │
         │  results/scaler.pkl           (StandardScaler)      │
         │                                                      │
         └──────────────────────────────────────────────────────┘
         │
         ├──────────────────────────────────────────────────────┐
         │  RAG PIPELINE                                        │
         │  ChromaDB (vector_store/)   542 chunks               │
         │  Embedding: embeddinggemma:latest (MRR=0.931)        │
         │  LLM: llama3.1:8b via Ollama (port 11434)           │
         │  Agent: ReAct loop (Reason + Act)                   │
         └──────────────────────────────────────────────────────┘
         │
         └──────────────────────────────────────────────────────┐
            Neo4j (optional GRAG enhancement)                   │
            bolt://localhost:7687                               │
            Chunk → Concept → Document graph                   │
            └────────────────────────────────────────────────── ┘
```

---

## Phase Status

### ✅ Phase 0 — ML Training (`best_try.ipynb`)
**Status: COMPLETE**

- [x] EDA on 25,862 Alberta FPA wildfire records (2006–2024)
- [x] Feature engineering: 12 predictors from raw data
- [x] 10-fold stratified CV with SMOTE
- [x] Random Forest: AUROC 0.958, AUPRC 0.513
- [x] Logistic Regression: AUROC 0.950, AUPRC 0.584 (**best by primary metric**)
- [x] SHAP feature importance analysis
- [x] Final Random Forest trained on full dataset → `results/wildfire_model.pkl`
- [x] Figures saved to `results/`

**Known gaps:**
- [ ] `results/scaler.pkl` not saved — **blocks inference**
- [ ] `results/lr_model.pkl` not saved — only RF persisted

---

### ✅ Phase 1a — RAG Corpus Build (`rag_grag_pipeline.ipynb`)
**Status: COMPLETE**

- [x] 6 PDFs chunked (350 words, 70-word overlap) → 542 chunks
- [x] Chunks saved: `results/all_chunks.json`
- [x] Embedding grid search run (2/7 models succeeded):
  - ✅ `embeddinggemma:latest` — MRR **0.931**, R@10 **1.000** ← **SELECTED**
  - ✅ `snowflake-arctic-embed2:latest` — MRR 0.870, R@10 0.987
  - ❌ 5 other models returned 400 (Ollama API format mismatch)
- [x] Full corpus embedded → `results/best_embeddings.npy`
- [x] ChromaDB vector store built → `vector_store/` (542 docs)
- [x] Neo4j knowledge graph: Documents → Chunks → Concepts
- [x] ReAct agent framework coded (Section 6)

**Known gaps:**
- [ ] Grid search 400 errors need fixing for complete comparison
- [ ] `predict_fire_risk` agent tool not yet wired to saved models
- [ ] End-to-end agent test not yet run

---

### 🔲 Phase 1b — Model Integration & Agent Testing
**Status: IN PROGRESS**

**Goals:**
- [ ] Fix `scaler.pkl` / `lr_model.pkl` persistence (re-run final training block)
- [ ] Fix 400 errors in embedding grid search (fallback to `/api/embeddings` endpoint)
- [ ] Wire `predict_fire_risk(inputs_json)` tool into the ReAct agent
- [ ] Integrate prediction results with RAG mitigation guidance
- [ ] End-to-end agent test: "given these conditions, what's the risk and what should I do?"

**The `predict_fire_risk` tool signature:**
```python
def predict_fire_risk(inputs: dict) -> str:
    """
    inputs keys: ASSESSMENT_HECTARES, FIRE_SPREAD_RATE, WIND_SPEED,
                 CAUSE_BINARY, TEMPERATURE, FOREST_AREA_ENC, FIRE_TYPE_ENC,
                 DISPATCH_LAG_HRS, FUEL_TYPE_ENC, RELATIVE_HUMIDITY,
                 FIRE_MONTH, DETECTION_LAG_HRS
    Returns: formatted string with RF prob, LR prob, ensemble estimate
    """
```

---

### 🔲 Phase 2 — FastAPI Backend (`backend/`)
**Status: NOT STARTED**

**Directory structure to build:**
```
backend/
├── main.py              # FastAPI app entry point
├── routers/
│   ├── predict.py       # POST /predict
│   └── ask.py           # POST /ask
├── services/
│   ├── ml_service.py    # Load models, run inference
│   └── rag_service.py   # RAG pipeline wrapper
├── schemas/
│   └── fire_input.py    # Pydantic input/output models
├── requirements.txt
└── Dockerfile
```

**API endpoints:**
```
POST /predict
  Body: { ASSESSMENT_HECTARES: float, FIRE_SPREAD_RATE: float, ... }
  Returns: { rf_probability: float, lr_probability: float,
             ensemble: float, risk_level: str, confidence: str }

POST /ask
  Body: { question: str, context: dict|null }
  Returns: { answer: str, sources: list[str] }

GET /health
  Returns: { status: "ok", models_loaded: bool, vector_store: bool }
```

**Middleware/Security:**
- CORS restricted to `localhost:5173` in dev, configurable for prod
- Input validation via Pydantic (all numeric fields bounded)
- No raw user input logged

---

### 🔲 Phase 3 — React/Vite Frontend (`frontend/`)
**Status: NOT STARTED**

**Directory structure to build:**
```
frontend/
├── src/
│   ├── components/
│   │   ├── PredictionForm.jsx   # 12-field input form
│   │   ├── RiskGauge.jsx        # Probability display
│   │   ├── ChatPanel.jsx        # Mitigation chat UI
│   │   └── SourceCitations.jsx  # RAG source display
│   ├── pages/
│   │   ├── Predict.jsx
│   │   └── Chat.jsx
│   ├── api/
│   │   └── emberApi.js          # Axios client for backend
│   └── App.jsx
├── package.json
├── vite.config.js
└── Dockerfile
```

**Key UX flows:**
1. User fills in fire conditions → hits Predict → sees probability gauge + plain-language risk description
2. User clicks "What should I do?" → opens chat panel pre-populated with conditions
3. Chat panel supports follow-up questions answered by RAG agent

---

### 🔲 Phase 4 — Docker Compose Containerization
**Status: NOT STARTED**

```yaml
# docker-compose.yml (target)
services:
  ollama:      # Ollama sidecar — exposes port 11434
  neo4j:       # Neo4j (optional GRAG) — ports 7474, 7687
  backend:     # FastAPI — port 8000
  frontend:    # Vite/React (nginx in prod) — port 5173/80
```

**Notes:**
- Ollama model weights need volume mount or pull-on-start script
- ChromaDB runs embedded inside the backend container (no separate service)
- ML model artifacts (`results/*.pkl`) baked into backend image

---

### 🔲 Phase 5 — Evaluation & Monitoring
**Status: NOT STARTED**

- [ ] RAG evaluation harness: automated MRR/AUPRC checks on held-out question set
- [ ] Model drift detection: compare input distributions against training distribution
- [ ] Logging schema for agent interactions (anonymized)
- [ ] README badges: model AUPRC, RAG MRR, test coverage

---

## Known Issues & Decisions Log

| Date | Issue | Decision |
|---|---|---|
| Apr 2026 | 5/7 embedding models returned HTTP 400 on `/api/embed` | Root cause: these models require `/api/embeddings` endpoint with `"prompt"` key. Fix in Phase 1b. |
| Apr 2026 | `scaler.pkl` not persisted in `run_pipeline.py` | Must re-run `train_final_model` block and save scaler alongside RF model. |
| Apr 2026 | LR model not saved to disk | Add `joblib.dump(lr_final, "results/lr_model.pkl")` in notebook. |
| Apr 2026 | Grid search ran only on 150-chunk subsample | Intentional — faster. Full corpus embedded separately in Section 4. |
| Apr 2026 | AUPRC is primary metric (not AUROC) | Class imbalance (3.34% positive rate) makes AUROC misleading. AUPRC is 15–18× baseline. |
| Apr 2026 | LR outperforms RF on AUPRC (0.584 vs 0.513) | LR is preferred for production; RF kept for SHAP interpretability. |

---

## Feature Importance Reference

| Rank | Feature | RF Gini Importance | Direction |
|---|---|---|---|
| 1 | Assessment hectares (ha) | 0.4232 | ↑ larger = riskier |
| 2 | Fire spread rate (m/min) | 0.2353 | ↑ faster = riskier |
| 3 | Wind speed (km/h) | 0.0629 | ↑ windier = riskier |
| 4 | Lightning ignition (0/1) | 0.0562 | lightning = riskier |
| 5 | Temperature (°C) | 0.0493 | ↑ hotter = riskier |
| 6 | Forest management zone | 0.0388 | varies |
| 7 | Fire type (encoded) | 0.0365 | crown > surface |
| 8 | Dispatch lag (hrs) | 0.0248 | ↑ longer = riskier |
| 9 | Fuel type (encoded) | 0.0235 | varies |
| 10 | Relative humidity (%) | 0.0227 | ↓ drier = riskier |
| 11 | Month of ignition | 0.0146 | Jul–Aug riskiest |
| 12 | Detection lag (hrs) | 0.0122 | ↑ longer = riskier |

---

## Quick Start

```bash
# 1. Activate venv
firemitvenv\Scripts\activate

# 2. Ensure Ollama is running
# Verify: curl http://localhost:11434/api/tags

# 3. Open notebook
jupyter notebook rag_grag_pipeline.ipynb

# 4. Run sections in order: 1 → 2 → 3 → 4 → 6
# (Section 5 = Neo4j, optional)
```
