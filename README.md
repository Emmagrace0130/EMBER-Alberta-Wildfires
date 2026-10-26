# EMBER — Alberta Wildfire Risk Assessor

Predicts large-fire escalation risk for Alberta wildfires and provides actionable mitigation guidance via an agentic LLM assistant.

**ML models** trained on 26,551 Alberta Forest Protection Area wildfire records (2006–2024) predict whether a fire will escalate to ≥ 40 hectares. A **RAG/GRAG pipeline** retrieves mitigation guidance from wildfire protection documents. A **ReAct agent** ties both systems together with structured reasoning.

---

## Architecture

```
User Question / Fire Conditions
         │
         ▼
┌──────────────────┐
│   ReAct Agent    │  llama3.1:8b via Ollama (local)
│   (max 8 steps)  │
└────┬─────┬───┬───┘
     │     │   │
     ▼     ▼   ▼
  ML Models  ChromaDB   Neo4j Knowledge Graph
  (predict)  (RAG)      (GRAG)
     │     │        │
     └─────┴────────┘
            │
     FINAL ANSWER
  (risk level + mitigation advice
   grounded in data & documents)
```

---

## Key Results

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| AUPRC (primary) | **0.584 ± 0.041** | 0.513 ± 0.044 |
| AUROC | 0.950 ± 0.009 | 0.959 ± 0.007 |
| Positive class rate | 3.34% | 3.34% |

Top predictors: **Assessment Hectares** (42.3% importance) and **Fire Spread Rate** (23.5%).

---

## Notebooks

### `best_try.ipynb` — ML Training Pipeline
The original data analysis and model training notebook. Runs the full ML pipeline from raw data to saved models:
- Exploratory data analysis on 25,862 Alberta wildfire records
- Feature engineering (12 features from 50 raw columns)
- 10-fold stratified cross-validation with SMOTE oversampling
- Random Forest and Logistic Regression training and evaluation
- SHAP feature importance analysis
- Isolation Forest unsupervised anomaly detection
- Saves `wildfire_model.pkl`, `lr_model.pkl`, `scaler.pkl` to `results/`
- Generates all figures (`fig_*.png`)

### `rag_grag_pipeline.ipynb` — RAG/GRAG Development Pipeline
The full RAG and Graph-RAG development notebook. Builds the retrieval and knowledge graph systems:
- **Section 1–2:** PDF extraction from 6 wildfire mitigation documents using pypdf
- **Section 3:** Sliding-window chunking (350 words, 70-word overlap) → 510 chunks
- **Section 4:** Embedding grid search across Ollama models; winner: `nomic-embed-text:latest` (dim=768, MRR=0.984)
- **Section 4 (cont):** ChromaDB vector store build + full corpus embedding
- **Section 5:** Neo4j knowledge graph creation (Documents → Chunks → Concepts)
- **Section 6:** ReAct agent framework with 6 tools

### `ember_agent_test.ipynb` — Agent Test Notebook
A streamlined notebook that **skips training and grid search**, loading all pre-computed artifacts for fast iteration on the agent and graph:

| Section | Purpose |
|---|---|
| 1 — Setup & Configuration | Paths, Ollama/Neo4j config |
| 2a — Chunks & Embeddings | Load `all_chunks.json` + `best_embeddings.npy` |
| 2b — RAG Configuration | Load `rag_config.json` + ChromaDB collection |
| 2c — ML Models | Load RF, LR, scaler via joblib |
| 3 — Embedding Helper | Compact `embed_texts()` with error handling + warm-up |
| 4a — Neo4j Container | Docker-managed `ember-neo4j` start + connect |
| 4b — Document & Chunk nodes | Core graph layer (6 docs, 510 chunks) |
| 4b+ — Analytical Knowledge Layer | Feature, RiskLevel, MitigationAction, DataInsight nodes |
| 4c — Embeddings & Vector Index | Attach 768-dim vectors + cosine index |
| 4d — Concept Nodes | 26 domain concepts + MENTIONS + RELATED_TO |
| 4e — Verify Graph | Count all node types and relationships |
| 5a — Tool Definitions | 6 agent tools (predict, RAG retrieve, GRAG search/traversal) |
| 5b — ReAct Agent | System prompt + reasoning loop |
| 6a — Prediction + Mitigation | End-to-end fire risk scenario |
| 6b — Mitigation Questions | RAG-focused retrieval tests |
| 6c — Interactive | General questions |
| 6d — Analytical Layer Tests | Tests targeting Feature/RiskLevel/MitigationAction/DataInsight nodes |
| 6e — Direct Graph Queries | Verify analytical layer wiring without the agent |

---

## Knowledge Graph

The Neo4j graph has two layers:

### Core Layer
- **6 Document** nodes (one per source PDF)
- **510 Chunk** nodes with 768-dim embedding vectors
- **26 Concept** nodes (domain vocabulary)
- Relationships: `CONTAINS`, `MENTIONS`, `RELATED_TO`
- Vector index: `ember_chunk_vectors` (cosine, dim=768)

### Analytical Knowledge Layer
All analytical nodes carry a dual `:Concept` label so existing GRAG tools discover them automatically.

- **12 Feature** nodes — importance rank, Gini score, SHAP direction
- **4 RiskLevel** nodes — probability thresholds (LOW / MODERATE / HIGH / VERY HIGH)
- **10 MitigationAction** nodes — 5 operational + 5 preventive
- **6 DataInsight** nodes — key findings from the ML analysis
- Relationships: `PREDICTS`, `TRIGGERS_ACTION`, `ASSESSES`, `RECOMMENDS`

---

## Agent Tools

| Tool | System | Description |
|---|---|---|
| `predict_fire_risk(json)` | ML | RF + LR ensemble → probability + risk level |
| `rag_retrieve(query)` | RAG | ChromaDB cosine search → top-5 document chunks |
| `grag_vector_search(query)` | GRAG | Neo4j vector index → chunks + graph concepts |
| `grag_concept_neighbors(concept)` | GRAG | Graph traversal: find related concepts |
| `grag_concept_chunks(concept)` | GRAG | Graph → chunks that mention a concept |
| `grag_graph_summary()` | GRAG | List all documents and chunk counts |

---

## Repository Layout

```
├── best_try.ipynb              # ML training pipeline (Phase 0)
├── rag_grag_pipeline.ipynb     # RAG/GRAG development pipeline (Phase 1a)
├── ember_agent_test.ipynb      # Agent test notebook (Phase 1b)
├── run_pipeline.py             # Standalone CLI pipeline script
├── tool_server.py              # Tool server utilities
├── src/                        # Python ML pipeline modules
│   ├── data_loader.py          #   Data loading from Excel
│   ├── preprocessing.py        #   Null handling, encoding, target creation
│   ├── features.py             #   Feature engineering (lags, month, etc.)
│   ├── models.py               #   Model training (RF, LR, IF)
│   ├── evaluation.py           #   Cross-validation, metrics
│   ├── visualization.py        #   Figure generation
│   ├── shap_analysis.py        #   SHAP feature importance
│   └── anomaly.py              #   Isolation Forest
├── results/                    # Saved artifacts
│   ├── wildfire_model.pkl      #   Random Forest (200 trees, balanced)
│   ├── lr_model.pkl            #   Logistic Regression (balanced)
│   ├── scaler.pkl              #   StandardScaler
│   ├── all_chunks.json         #   510 text chunks from 6 PDFs
│   ├── best_embeddings.npy     #   (510, 768) embedding matrix
│   ├── rag_config.json         #   Best model config
│   └── embedding_grid_search.csv
├── vector_store/               # ChromaDB persistent store
├── docs/
│   ├── data_docs/              #   Source PDFs for RAG corpus
│   └── findings/               #   Analysis reports & documentation
│       ├── 01_executive_summary.md
│       ├── 02_dataset_characteristics.md
│       ├── 03_model_performance.md
│       ├── 04_feature_importance.md
│       ├── 05_methodology.md
│       ├── 06_entities_and_relationships.md
│       ├── 07_notebook_run_order.md
│       ├── 08_session_log_2026-04-15.md
│       └── 09_rag_grag_operations.md
├── tests/                      # Test suite
├── frontend/                   # React + Vite (Phase 3 — planned)
├── backend/                    # FastAPI (Phase 2 — planned)
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Activate the virtual environment
firemitvenv\Scripts\activate          # Windows
source firemitvenv/Scripts/activate   # Git Bash

# 2. Ensure Ollama is running with required models
ollama list   # should show llama3.1:8b and nomic-embed-text:latest
# If missing:
ollama pull llama3.1:8b
ollama pull nomic-embed-text:latest

# 3. Ensure Docker is running (for Neo4j)
docker info

# 4. Open the test notebook
jupyter notebook ember_agent_test.ipynb
# Run cells top-to-bottom (Sections 1 → 6)
```

### Ports

| Port | Service |
|---|---|
| 11434 | Ollama (local LLM + embedding inference) |
| 7687 | Neo4j Bolt |
| 7474 | Neo4j HTTP browser |
| 8000 | FastAPI backend (Phase 2) |
| 5173 | Vite dev server (Phase 3) |

---

## Phase Checklist

- [x] **Phase 0** — Data analysis & ML model training (`best_try.ipynb`)
- [x] **Phase 1a** — RAG corpus chunking & vector store build (`rag_grag_pipeline.ipynb`)
- [x] **Phase 1b** — Agent tools, knowledge graph, analytical layer, test notebook (`ember_agent_test.ipynb`)
- [ ] **Phase 2** — FastAPI backend (`/predict`, `/ask`, `/health`)
- [ ] **Phase 3** — React/Vite frontend (prediction form + chat UI)
- [ ] **Phase 4** — Docker Compose containerization
- [ ] **Phase 5** — Evaluation harness (RAG quality, model drift)

---

## Documentation

| Document | Description |
|---|---|
| [01_executive_summary.md](docs/findings/01_executive_summary.md) | Project overview and key findings |
| [02_dataset_characteristics.md](docs/findings/02_dataset_characteristics.md) | Dataset description, class distribution, features |
| [03_model_performance.md](docs/findings/03_model_performance.md) | Cross-validation results, model comparison |
| [04_feature_importance.md](docs/findings/04_feature_importance.md) | Gini importance + SHAP analysis |
| [05_methodology.md](docs/findings/05_methodology.md) | Pipeline steps, preprocessing, CV design |
| [06_entities_and_relationships.md](docs/findings/06_entities_and_relationships.md) | Knowledge graph seed entities |
| [07_notebook_run_order.md](docs/findings/07_notebook_run_order.md) | Cell run-order reference for notebooks |
| [08_session_log_2026-04-15.md](docs/findings/08_session_log_2026-04-15.md) | Development session log |
| [09_rag_grag_operations.md](docs/findings/09_rag_grag_operations.md) | RAG/GRAG architecture and graph schema guide |

---

## Environment

- **Python:** 3.14+ (venv at `firemitvenv/`)
- **LLM:** llama3.1:8b (local via Ollama)
- **Embedding:** nomic-embed-text:latest (dim=768)
- **Vector Store:** ChromaDB (persistent)
- **Graph DB:** Neo4j 5.x (Docker container `ember-neo4j`)
- **Docker:** v29.2.1+
- **No external API keys required** — all inference runs locally

---

## License

See [LICENSE](LICENSE).
