# EMBER — Alberta Wildfire Risk Assessor
## GitHub Copilot Instructions

### Project Purpose
EMBER predicts large-fire escalation risk for Alberta wildfires and provides actionable mitigation guidance via an agentic LLM assistant. The system:
1. Runs ML models (Random Forest, Logistic Regression) to predict probability of a fire becoming a large fire (≥ 40 ha)
2. Uses RAG over wildfire mitigation documentation to explain what landowners/communities can do to reduce risk
3. Exposes everything through a FastAPI backend and React/Vite frontend, all containerized

---

### Repository Layout
```
src/                  # Python ML pipeline modules (preprocessing, features, models, evaluation, etc.)
docs/findings/        # Markdown reports on model performance, feature importance, methodology
docs/data_docs/       # Source PDFs chunked for RAG (FireSmart, CWFIS, farm/acreage guide, etc.)
results/              # Saved artifacts: wildfire_model.pkl, scaler.pkl, lr_model.pkl,
                      #   best_embeddings.npy, all_chunks.json, rag_config.json,
                      #   embedding_grid_search.csv, figures
vector_store/         # ChromaDB persistent collection: "wildfire_docs"
rag_grag_pipeline.ipynb  # Main RAG/GRAG development notebook
best_try.ipynb        # Original ML pipeline notebook
run_pipeline.py       # Standalone CLI pipeline script
backend/              # FastAPI application (to be built — Phase 2)
frontend/             # React + Vite application (to be built — Phase 3)
```

---

### ML Models
- **Input features (12):** `ASSESSMENT_HECTARES`, `FIRE_SPREAD_RATE`, `WIND_SPEED`, `CAUSE_BINARY` (lightning=1), `TEMPERATURE`, `FOREST_AREA_ENC`, `FIRE_TYPE_ENC`, `DISPATCH_LAG_HRS`, `FUEL_TYPE_ENC`, `RELATIVE_HUMIDITY`, `FIRE_MONTH`, `DETECTION_LAG_HRS`
- **Random Forest final** (`results/wildfire_model.pkl`): n_estimators=200, max_depth=10, class_weight=balanced. Trained on full dataset (25,862 records).
- **Logistic Regression** (`results/lr_model.pkl`): class_weight=balanced, max_iter=1000. Best cross-validation AUPRC (0.584 ± 0.041).
- **Scaler** (`results/scaler.pkl`): `StandardScaler` fitted on full training set — **must be applied before inference with either model**.
- **Class imbalance:** 3.34% positive rate. AUROC > 0.95 but AUPRC ~0.51–0.58 (primary metric). Models over-predict 0s; use `predict_proba` and threshold tuning.
- **Positive class:** large fire = Size Class D (40–200 ha) or E (200+ ha)

---

### RAG / GRAG Stack
- **Embedding model:** `embeddinggemma:latest` (dim=768, MRR=0.931 on eval set) via Ollama local API (`http://localhost:11434`)
- **Vector store:** ChromaDB persistent at `vector_store/`, collection `wildfire_docs`, 542 chunks
- **LLM:** `llama3.1:8b` via Ollama (local, no external API calls)
- **Corpus:** 6 PDFs — FireSmart Community Protection Guide, CWFIS Data Services, Farm & Acreage Wildfire Risk Reduction, Alberta WUI Fires, Prevention Plan Template, USFS Rocky Mountain GTR-292
- **Chunking:** sliding window, 350 words, 70-word overlap

---

### Agent Tools (Section 6 of notebook)
| Tool | Purpose |
|---|---|
| `predict_fire_risk(inputs)` | Run RF + LR ensemble; return probability + confidence band |
| `rag_retrieve(query)` | ChromaDB cosine search → top-k mitigation chunks |
| `grag_vector_search(query)` | Neo4j vector index search with graph context |
| `grag_concept_neighbors(concept)` | Knowledge graph traversal |
| `grag_concept_chunks(concept)` | Graph → chunks via concept nodes |
| `grag_graph_summary()` | High-level graph stats |

---

### Ports (do not rebind)
| Port | Service |
|---|---|
| 11434 | Ollama (local LLM/embedding inference) |
| 7687 | Neo4j Bolt |
| 7474 | Neo4j HTTP browser |
| 8000 | FastAPI backend (Phase 2) |
| 5173 | Vite dev server / React frontend (Phase 3) |

---

### Key Conventions
- **Python 3.12+**, venv at `firemitvenv/`. Activate: `firemitvenv\Scripts\activate`
- **No external LLM API keys** — everything runs locally through Ollama
- **Inference pipeline:** raw inputs → `preprocessing.preprocess_inference_row()` → `StandardScaler.transform()` → `model.predict_proba()[:,1]`
- **Model serialization:** `joblib.dump` / `joblib.load` only (not pickle directly)
- Always save `scaler.pkl` alongside any trained model or the model won't be usable for inference
- Embedding API: use Ollama `/api/embed` with `{"model": ..., "input": [...]}` — if a model returns 400, fall back to `/api/embeddings` with `{"model": ..., "prompt": "..."}` (single text)
- **Security:** never log or expose raw user inputs in production; validate all numeric inputs at API boundary

---

### Phase Checklist
- [x] Phase 0 — Data analysis & ML model training (`best_try.ipynb`)
- [x] Phase 1a — RAG corpus chunking & vector store build (`rag_grag_pipeline.ipynb`)
- [ ] Phase 1b — Fix scaler persistence; save LR model; add `predict_fire_risk` tool; end-to-end agent test
- [ ] Phase 2 — FastAPI backend (`backend/`) with `/predict`, `/ask`, `/health` routes
- [ ] Phase 3 — React/Vite frontend (`frontend/`) with prediction form + chat UI
- [ ] Phase 4 — Docker Compose containerization (Ollama sidecar, FastAPI, React, Neo4j optional)
- [ ] Phase 5 — Evaluation harness: automated RAG quality checks, model drift monitoring
