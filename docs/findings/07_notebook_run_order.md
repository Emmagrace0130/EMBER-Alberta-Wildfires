# EMBER — Notebook Run Order Reference

## Full Pipeline (`rag_grag_pipeline.ipynb`)

Run all cells top-to-bottom on first use. Subsequent runs can skip the grid search (Section 3).

| Cell # | Section | Purpose | Time |
|--------|---------|---------|------|
| 2 | Setup | `pip install` dependencies (incl. `docker` SDK) | ~10s |
| 4 | Section 1 | Imports, config, paths, constants | <1s |
| 5 | Section 1 | Verify Ollama models | ~2s |
| 8 | Section 1b | Load ML models (RF, LR, scaler) | <1s |
| 11 | Section 2 | PDF extraction + chunking → `all_chunks`, `pdf_files` | ~5s |
| 12 | Section 2 | Define `save_chunks` / `load_chunks` helpers | <1s |
| 17 | Section 3 | Embedding helper (`embed_texts`, warm-up, eval functions) | <1s |
| 19 | Section 3 | Build eval corpus + query set (150 samples) | <1s |
| 21 | Section 3 | **Grid search** (all embedding models) | 5–30min |
| 22 | Section 3 | Analyse results, pick best model, save CSV + plot | ~2s |
| 24 | Section 4 | Re-embed full 510-chunk corpus with best model | 1–3min |
| 25 | Section 4 | Build persistent ChromaDB vector store | ~5s |
| 26 | Section 4 | Save `rag_config.json` | <1s |
| 29 | Section 5 | **Start Neo4j Docker container + connect** | 10–60s (first run pulls image) |
| 31 | Section 5 | Schema constraints + Document/Chunk nodes | ~30s |
| 32 | Section 5 | Attach embeddings + create vector index | ~60s |
| 33 | Section 5 | Concept nodes + MENTIONS/RELATED_TO relationships | ~30s |
| 36 | Section 6 | Load RAG config + ChromaDB collection | <1s |
| 37 | Section 6 | Register tool definitions | <1s |
| 38 | Section 6 | LLM helper + ReAct agent | <1s |
| 39 | Section 6 | Run test questions | 1–3min per question |

## Quick Test Run (artifacts already on disk)

Use the **`ember_agent_test.ipynb`** notebook instead. It loads all pre-computed artifacts and jumps straight to graph setup + agent testing.

### Minimum cells from the main notebook (if not using test notebook)

Skip Sections 3–4 entirely. Run only:

| Cell # | What |
|--------|------|
| 2 | pip install |
| 4 | Imports + config |
| 5 | Verify Ollama |
| 8 | Load ML models |
| 11 | PDF extract + chunk (needed for `pdf_files` + `all_chunks`) |
| 12 | Helpers |
| 17 | Embedding helper |
| 29 | Neo4j Docker + connect |
| 31–33 | Build graph |
| 36–38 | Load RAG + register tools + agent |
| 39+ | Test questions |

## Saved Artifacts

| File | Contents |
|------|----------|
| `results/all_chunks.json` | 510 chunks from 6 PDFs |
| `results/best_embeddings.npy` | (510, 768) float32 embedding matrix |
| `results/rag_config.json` | Best model name, dim, paths, grid search summary |
| `results/embedding_grid_search.csv` | Full grid search results |
| `results/wildfire_model.pkl` | Trained Random Forest classifier |
| `results/lr_model.pkl` | Trained Logistic Regression classifier |
| `results/scaler.pkl` | Fitted StandardScaler |
| `vector_store/` | ChromaDB persistent collection (`wildfire_docs`, 510 chunks) |

## Ports

| Port | Service |
|------|---------|
| 11434 | Ollama (local LLM + embedding inference) |
| 7687 | Neo4j Bolt (Docker container `ember-neo4j`) |
| 7474 | Neo4j HTTP browser |
