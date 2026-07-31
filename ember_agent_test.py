#!/usr/bin/env python
# coding: utf-8

# # EMBER — Agent Test Notebook
# 
# **Purpose:** Fast-reload notebook that skips training/grid-search and loads all pre-computed artifacts.
# 
# **Prerequisites:** Run `rag_grag_pipeline.ipynb` at least once (Sections 1–4) to generate:
# - `results/all_chunks.json`
# - `results/best_embeddings.npy`
# - `results/rag_config.json`
# - `results/wildfire_model.pkl`, `lr_model.pkl`, `scaler.pkl`
# - `vector_store/` (ChromaDB)
# 
# **Sections:**
# 1. Setup & Configuration
# 2. Load Saved Artifacts (chunks, embeddings, ML models, RAG config)
# 3. Embedding Helper
# 4. Neo4j Knowledge Graph (Docker container)
# 5. Agent Setup (tools + ReAct loop)
# 6. Test & Interactive Use

# In[ ]:


# get_ipython().run_line_magic('pip', 'install -q chromadb neo4j pypdf python-dotenv tqdm docker 2>&1 | tail -6')


# ---
# ## 1 — Setup & Configuration

# In[2]:


import os, json, re, time, textwrap
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import requests
from tqdm import tqdm
from dotenv import load_dotenv
import chromadb
from neo4j import GraphDatabase

load_dotenv(override=True)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DOCS_DIR    = Path("docs/data_docs")
VECTOR_STORE_DIR = Path("vector_store")
RESULTS_DIR      = Path("results")

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_LOCAL_URL = os.getenv("OLLAMA_LOCAL_URL", "http://localhost:11434")
AGENT_LLM        = "llama3.1:8b"

# ── Neo4j ─────────────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

print(f"Ollama : {OLLAMA_LOCAL_URL}")
print(f"Neo4j  : {NEO4J_URI}")


# ---
# ## 2 — Load Saved Artifacts

# ### 2a — Chunks & Embeddings

# In[3]:


# ── Load chunks from disk ─────────────────────────────────────────────────────
with open(RESULTS_DIR / "all_chunks.json") as f:
    all_chunks = json.load(f)

pdf_files = sorted(DATA_DOCS_DIR.glob("*.pdf"))

print(f"Chunks loaded  : {len(all_chunks)}")
print(f"PDF files      : {len(pdf_files)}")


# In[4]:


# ── Load pre-computed embeddings ──────────────────────────────────────────────
best_vecs = np.load(RESULTS_DIR / "best_embeddings.npy")
print(f"Embeddings loaded : {best_vecs.shape}")


# ### 2b — RAG Configuration

# In[5]:


# ── Load RAG config + ChromaDB collection ─────────────────────────────────────
with open(RESULTS_DIR / "rag_config.json") as f:
    rag_config = json.load(f)

BEST_MODEL = rag_config["best_embedding_model"]
BEST_DIM   = rag_config["embedding_dim"]

_chroma    = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
collection = _chroma.get_collection("wildfire_docs")

print(f"Best model     : {BEST_MODEL}  (dim={BEST_DIM})")
print(f"ChromaDB count : {collection.count()} chunks")
print(f"Agent LLM      : {AGENT_LLM}")


# ### 2c — ML Models

# In[6]:


import joblib

FEATURE_ORDER = [
    "ASSESSMENT_HECTARES", "FIRE_SPREAD_RATE", "WIND_SPEED", "CAUSE_BINARY",
    "TEMPERATURE", "FOREST_AREA_ENC", "FIRE_TYPE_ENC", "DISPATCH_LAG_HRS",
    "FUEL_TYPE_ENC", "RELATIVE_HUMIDITY", "FIRE_MONTH", "DETECTION_LAG_HRS",
]

_rf_model  = None
_lr_model  = None
_ml_scaler = None

def load_ml_models() -> dict:
    global _rf_model, _lr_model, _ml_scaler
    status = {}
    for attr, fname, label in [
        ("_rf_model",  "wildfire_model.pkl", "Random Forest"),
        ("_lr_model",  "lr_model.pkl",       "Logistic Regression"),
        ("_ml_scaler", "scaler.pkl",         "Scaler"),
    ]:
        p = RESULTS_DIR / fname
        if p.exists():
            globals()[attr] = joblib.load(p)
            status[label] = f"\u2713 loaded  ({fname})"
        else:
            status[label] = f"\u2717 missing ({fname})"
    for label, msg in status.items():
        print(f"  {label:<22}: {msg}")
    return status

load_ml_models()


# ---
# ## 3 — Embedding Helper
# 
# Provides `embed_texts()` for the agent tools (RAG retrieve, GRAG vector search).

# In[7]:


# ── Embedding helper (compact version — no grid search functions) ─────────────
_warmed_models: set = set()
_model_api_mode: dict = {}
_model_dim: dict = {}
EMBED_BATCH_SIZE = 16
MAX_EMBED_CHARS  = 2000


def _try_embed_endpoint(base_url: str, model: str, text: str, endpoint: str) -> list | None:
    try:
        if endpoint == "embed":
            resp = requests.post(f"{base_url}/api/embed",
                                json={"model": model, "input": [text]}, timeout=180)
            resp.raise_for_status()
            data = resp.json()
            if "embeddings" in data and data["embeddings"]:
                return data["embeddings"][0]
        else:
            resp = requests.post(f"{base_url}/api/embeddings",
                                json={"model": model, "prompt": text}, timeout=180)
            resp.raise_for_status()
            data = resp.json()
            if "embedding" in data and data["embedding"]:
                return data["embedding"]
    except Exception:
        pass
    return None


def warm_up_model(model: str, base_url: str = None) -> bool:
    base_url = base_url or OLLAMA_LOCAL_URL
    label = model.split(":")[0]
    for endpoint in ("embed", "embeddings"):
        emb = _try_embed_endpoint(base_url, model, "warmup", endpoint)
        if emb is not None:
            _warmed_models.add(model)
            _model_api_mode[model] = endpoint
            _model_dim[model] = len(emb)
            print(f"  \u2713 {label:<30} dim={len(emb)}  [/api/{endpoint}]")
            return True
    print(f"  \u2717 {label:<30} FAILED")
    return False


def _safe_truncate(text: str) -> str:
    return text[:MAX_EMBED_CHARS] if len(text) > MAX_EMBED_CHARS else text


def _embed_single_safe(text: str, model: str, base_url: str) -> Optional[List[float]]:
    text = _safe_truncate(text)
    mode = _model_api_mode.get(model, "embed")
    if mode == "embed":
        try:
            resp = requests.post(f"{base_url}/api/embed",
                                json={"model": model, "input": [text]}, timeout=180)
            resp.raise_for_status()
            return resp.json()["embeddings"][0]
        except Exception:
            pass
    try:
        resp = requests.post(f"{base_url}/api/embeddings",
                            json={"model": model, "prompt": text}, timeout=180)
        resp.raise_for_status()
        return resp.json()["embedding"]
    except Exception:
        pass
    return None


def embed_texts(
    texts: List[str],
    model: str,
    base_url: str = None,
    batch_size: int = EMBED_BATCH_SIZE,
    show_progress: bool = False,
) -> np.ndarray:
    base_url = base_url or OLLAMA_LOCAL_URL
    label = model.split(":")[0][:24]
    if model not in _warmed_models:
        print(f"    [{label}] loading model ...", flush=True)
        warm_up_model(model, base_url)
    mode = _model_api_mode.get(model, "embed")
    dim  = _model_dim.get(model, 768)
    all_embs: List[List[float]] = []
    n_fallback = n_zero = 0

    if mode == "embed":
        batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
        pbar = tqdm(batches, desc=label, unit="batch", disable=not show_progress, leave=False)
        for batch in pbar:
            truncated = [_safe_truncate(t) for t in batch]
            try:
                resp = requests.post(f"{base_url}/api/embed",
                                    json={"model": model, "input": truncated}, timeout=120)
                resp.raise_for_status()
                all_embs.extend(resp.json()["embeddings"])
            except requests.exceptions.HTTPError:
                n_fallback += 1
                for text in batch:
                    emb = _embed_single_safe(text, model, base_url)
                    all_embs.append(emb if emb else [0.0] * dim)
                    if emb is None: n_zero += 1
    else:
        pbar = tqdm(texts, desc=label, unit="text", disable=not show_progress, leave=False)
        for text in pbar:
            emb = _embed_single_safe(text, model, base_url)
            all_embs.append(emb if emb else [0.0] * dim)
            if emb is None: n_zero += 1

    if n_fallback: print(f"  \u26a0 {n_fallback} batch(es) fell back to single-text mode")
    if n_zero:     print(f"  \u26a0 {n_zero} text(s) could not be embedded")
    return np.array(all_embs, dtype=np.float32)


# Warm up the best model immediately
warm_up_model(BEST_MODEL)
print(f"\nembed_texts ready  (model: {BEST_MODEL})")


# ---
# ## 4 — Neo4j Knowledge Graph

# ### 4a — Start Container & Connect

# In[8]:


import docker as _docker

NEO4J_CONTAINER_NAME = "ember-neo4j"
NEO4J_IMAGE          = "neo4j:5"
NEO4J_BOLT_PORT      = 7687
NEO4J_HTTP_PORT      = 7474

neo4j_driver = None


def _ensure_neo4j_container() -> None:
    client = _docker.from_env()
    try:
        container = client.containers.get(NEO4J_CONTAINER_NAME)
        if container.status != "running":
            print(f"Starting existing container '{NEO4J_CONTAINER_NAME}' ...")
            container.start()
        else:
            print(f"Container '{NEO4J_CONTAINER_NAME}' already running.")
        return
    except _docker.errors.NotFound:
        pass
    print(f"Creating container '{NEO4J_CONTAINER_NAME}' (image: {NEO4J_IMAGE}) ...")
    client.containers.run(
        NEO4J_IMAGE, name=NEO4J_CONTAINER_NAME, detach=True,
        ports={f"{NEO4J_BOLT_PORT}/tcp": NEO4J_BOLT_PORT,
               f"{NEO4J_HTTP_PORT}/tcp": NEO4J_HTTP_PORT},
        environment={"NEO4J_AUTH": f"neo4j/{NEO4J_PASSWORD}",
                     "NEO4J_PLUGINS": '["graph-data-science"]'},
        restart_policy={"Name": "unless-stopped"},
    )


def _wait_for_neo4j(uri, user, password, retries=30, delay=2.0):
    print("Waiting for Neo4j to become ready ", end="", flush=True)
    for _ in range(retries):
        try:
            drv = GraphDatabase.driver(uri, auth=(user, password))
            drv.verify_connectivity()
            print(" ready!")
            return drv
        except Exception:
            print(".", end="", flush=True)
            time.sleep(delay)
    raise RuntimeError(f"Neo4j not ready after {retries * delay:.0f}s")


try:
    _ensure_neo4j_container()
    neo4j_driver = _wait_for_neo4j(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    print(f"\u2713 Neo4j connected: {NEO4J_URI}")
except Exception as e:
    print(f"\u2717 Neo4j setup failed: {e}")
    print("  \u2192 GRAG tools will return fallback messages; RAG still works.")


def run_cypher(query: str, params: dict = {}) -> list:
    if neo4j_driver is None:
        raise RuntimeError("Neo4j not connected.")
    with neo4j_driver.session() as session:
        return [dict(r) for r in session.run(query, params)]


# ### 4b — Build Graph (Document + Chunk nodes)

# In[9]:


if neo4j_driver:
    # Schema constraints
    for stmt in [
        "CREATE CONSTRAINT ember_doc_title IF NOT EXISTS FOR (d:Document) REQUIRE d.title IS UNIQUE",
        "CREATE CONSTRAINT ember_chunk_id  IF NOT EXISTS FOR (c:Chunk)    REQUIRE c.chunk_id IS UNIQUE",
        "CREATE CONSTRAINT ember_concept   IF NOT EXISTS FOR (co:Concept) REQUIRE co.name IS UNIQUE",
    ]:
        try: run_cypher(stmt)
        except Exception as e: print(f"  Schema note: {e}")

    # Clean previous EMBER data
    run_cypher("MATCH (n) WHERE n.project = 'EMBER' DETACH DELETE n")
    print("Cleared previous EMBER nodes.")

    # Document nodes
    for pdf_path in pdf_files:
        run_cypher(
            "MERGE (d:Document {title: $title}) SET d.path = $path, d.project = 'EMBER'",
            {"title": pdf_path.stem[:80], "path": str(pdf_path)},
        )
    print(f"Created {len(pdf_files)} Document nodes.")

    # Chunk nodes + CONTAINS
    print(f"Creating {len(all_chunks)} Chunk nodes ...")
    BATCH = 200
    for i in range(0, len(all_chunks), BATCH):
        batch = all_chunks[i:i+BATCH]
        run_cypher(
            """UNWIND $chunks AS c
            MERGE (ch:Chunk {chunk_id: c.id})
            SET ch.text = c.text, ch.doc_title = c.doc_title, ch.project = 'EMBER'
            WITH ch, c
            MATCH (d:Document {title: c.doc_title})
            MERGE (d)-[:CONTAINS]->(ch)""",
            {"chunks": batch},
        )
        print(f"  {min(i+BATCH, len(all_chunks)):>5} / {len(all_chunks)}")
    print("Chunk nodes created.")
else:
    print("Skipped — Neo4j not connected.")


# ### 4b+ — Analytical Knowledge Layer
# 
# Adds structured analytical nodes derived from the ML findings (`docs/findings/`):
# 
# | Node Label | Count | Purpose |
# |---|---|---|
# | **Feature** | 12 | Feature importance, SHAP direction, descriptions — lets the agent explain *what drives risk* |
# | **RiskLevel** | 4 | Probability thresholds (LOW / MODERATE / HIGH / VERY HIGH) — lets the agent explain *what a score means* |
# | **MitigationAction** | 10 | Operational and preventive actions — lets the agent recommend *what to do* |
# | **DataInsight** | 6 | Key analytical findings — lets the agent cite *evidence* for its reasoning |
# 
# All new nodes carry a secondary `:Concept` label and are connected via `:RELATED_TO`
# to existing domain concepts, so **existing GRAG tools automatically discover them**
# through concept traversal — no tool changes needed.
# 
# **New relationship types:**
# - `(Feature)-[:PREDICTS]->(Concept:Large Fire)` — feature → target
# - `(Feature)-[:TRIGGERS_ACTION]->(MitigationAction)` — risk factor → response
# - `(RiskLevel)-[:ASSESSES]->(Concept:Large Fire)` — threshold → target
# - `(RiskLevel)-[:RECOMMENDS]->(MitigationAction)` — risk level → guidance
# - `(DataInsight)-[:RELATED_TO]->(Concept)` — insight → domain concept
# - `(MitigationAction)-[:RELATED_TO]->(Concept)` — action → program/procedure

# In[10]:


if neo4j_driver:
    print("Building analytical knowledge layer ...")

    # ── 1. Feature nodes (dual-label: Feature + Concept) ─────────────────
    #   Encodes ML analysis results directly into the graph so the agent
    #   can answer "what drives large fire risk?" via graph traversal.
    FEATURES_ANALYTICAL = [
        {"name": "Assessment Hectares", "column": "ASSESSMENT_HECTARES",
         "rank": 1, "gini": 0.4232, "direction": "positive",
         "desc": "Fire size at initial assessment (ha). Strongest single predictor — accounts for 42.3% of RF importance."},
        {"name": "Fire Spread Rate", "column": "FIRE_SPREAD_RATE",
         "rank": 2, "gini": 0.2353, "direction": "positive",
         "desc": "Rate of fire spread at assessment (m/min). Second strongest predictor at 23.5% importance."},
        {"name": "Wind Speed", "column": "WIND_SPEED",
         "rank": 3, "gini": 0.0629, "direction": "positive",
         "desc": "Wind speed at assessment (km/h). Amplifies fire spread and escalation probability."},
        {"name": "Lightning Ignition", "column": "CAUSE_BINARY",
         "rank": 4, "gini": 0.0562, "direction": "positive",
         "desc": "Whether fire was lightning-caused (binary). Lightning fires tend to start in remote areas with delayed detection."},
        {"name": "Temperature", "column": "TEMPERATURE",
         "rank": 5, "gini": 0.0493, "direction": "positive",
         "desc": "Temperature at assessment (°C). Higher temperatures increase fire behaviour intensity."},
        {"name": "Forest Area Code", "column": "FOREST_AREA_ENC",
         "rank": 6, "gini": 0.0388, "direction": "mixed",
         "desc": "Alberta forest management zone. Some zones have structurally higher large-fire rates."},
        {"name": "Fire Type", "column": "FIRE_TYPE_ENC",
         "rank": 7, "gini": 0.0365, "direction": "mixed",
         "desc": "Fire type classification (surface/crown/ground). Crown fires are more likely to escalate."},
        {"name": "Dispatch Lag", "column": "DISPATCH_LAG_HRS",
         "rank": 8, "gini": 0.0248, "direction": "positive",
         "desc": "Hours from discovery to resource dispatch. Longer delays allow more fire growth."},
        {"name": "Fuel Type", "column": "FUEL_TYPE_ENC",
         "rank": 9, "gini": 0.0235, "direction": "mixed",
         "desc": "Forest fuel type classification. Coniferous fuel types support faster fire spread."},
        {"name": "Relative Humidity", "column": "RELATIVE_HUMIDITY",
         "rank": 10, "gini": 0.0227, "direction": "negative",
         "desc": "Relative humidity at assessment (%). Higher humidity reduces fire intensity — only feature with consistently negative SHAP direction."},
        {"name": "Fire Month", "column": "FIRE_MONTH",
         "rank": 11, "gini": 0.0146, "direction": "mixed",
         "desc": "Month of ignition (1-12). Peak fire season is May-August in Alberta."},
        {"name": "Detection Lag", "column": "DETECTION_LAG_HRS",
         "rank": 12, "gini": 0.0122, "direction": "positive",
         "desc": "Hours from fire start to discovery (0-72h). Least predictive feature once fire is reported."},
    ]

    for f in FEATURES_ANALYTICAL:
        run_cypher("""
            MERGE (feat:Concept:Feature {name: $name})
            SET feat.type         = 'Feature',
                feat.column_name  = $column,
                feat.importance_rank  = $rank,
                feat.gini_importance  = $gini,
                feat.shap_direction   = $direction,
                feat.description      = $desc,
                feat.project          = 'EMBER'
        """, f)

    # Link every feature to the Large Fire target concept
    run_cypher("""
        MATCH (f:Feature {project: 'EMBER'}), (lf:Concept {name: 'Large Fire'})
        MERGE (f)-[:PREDICTS]->(lf)
    """)
    print(f"  Created {len(FEATURES_ANALYTICAL)} Feature nodes → linked to Large Fire.")

    # ── 2. RiskLevel nodes ────────────────────────────────────────────────
    #   Encodes the probability thresholds from predict_fire_risk so the
    #   agent can explain what each risk level means via graph traversal.
    RISK_LEVELS = [
        {"name": "LOW Risk",       "level": "LOW",       "min_prob": 0.0,  "max_prob": 0.10,
         "desc": "Conditions not strongly indicative of a large fire. Monitor normally."},
        {"name": "MODERATE Risk",  "level": "MODERATE",  "min_prob": 0.10, "max_prob": 0.25,
         "desc": "Some risk factors present. Monitor closely and review resource availability."},
        {"name": "HIGH Risk",      "level": "HIGH",      "min_prob": 0.25, "max_prob": 0.45,
         "desc": "Multiple risk factors elevated. Consider pre-emptive resource positioning."},
        {"name": "VERY HIGH Risk", "level": "VERY_HIGH", "min_prob": 0.45, "max_prob": 1.0,
         "desc": "Conditions strongly associated with large-fire escalation. Immediate action recommended."},
    ]

    for r in RISK_LEVELS:
        run_cypher("""
            MERGE (rl:Concept:RiskLevel {name: $name})
            SET rl.type             = 'RiskLevel',
                rl.level            = $level,
                rl.min_probability  = $min_prob,
                rl.max_probability  = $max_prob,
                rl.description      = $desc,
                rl.project          = 'EMBER'
        """, r)

    run_cypher("""
        MATCH (rl:RiskLevel {project: 'EMBER'}), (lf:Concept {name: 'Large Fire'})
        MERGE (rl)-[:ASSESSES]->(lf)
    """)
    print(f"  Created {len(RISK_LEVELS)} RiskLevel nodes.")

    # ── 3. MitigationAction nodes ────────────────────────────────────────
    #   Operational response + preventive community actions drawn from
    #   the feature-importance analysis (doc 04) and the RAG corpus.
    MITIGATION_ACTIONS = [
        # Operational response actions
        {"name": "Escalate suppression priority", "category": "Operational",
         "trigger": "Assessment hectares above threshold",
         "desc": "When fire size at assessment exceeds historical thresholds, immediately escalate suppression priority and resource allocation."},
        {"name": "Pre-position aerial resources", "category": "Operational",
         "trigger": "High fire spread rate",
         "desc": "When spread rate at assessment is elevated, pre-position water bombers and helicopters for rapid initial attack."},
        {"name": "Issue elevated fire watch", "category": "Operational",
         "trigger": "High wind speed forecast",
         "desc": "When wind speeds exceed 30 km/h, issue elevated watch on all active fires in affected zones."},
        {"name": "Increase aerial reconnaissance", "category": "Operational",
         "trigger": "Remote lightning ignition",
         "desc": "For lightning-caused fires in remote areas, increase aerial reconnaissance frequency to detect growth early."},
        {"name": "Reduce dispatch lag", "category": "Operational",
         "trigger": "Any new fire detection",
         "desc": "Minimize time from detection to resource dispatch. Dispatch lag directly correlates with large-fire probability."},
        # Preventive / community mitigation actions
        {"name": "Create defensible space", "category": "Preventive",
         "trigger": "WUI proximity",
         "desc": "Establish fire-resistant zones around structures by removing combustible vegetation and debris within 10-30 metres."},
        {"name": "Implement FireSmart landscaping", "category": "Preventive",
         "trigger": "Community planning",
         "desc": "Use non-combustible materials, maintain green lawns, space trees, and remove ladder fuels near structures."},
        {"name": "Develop community protection plan", "category": "Preventive",
         "trigger": "Community planning",
         "desc": "Create formal wildfire protection plans including fuel management, evacuation routes, and communication protocols."},
        {"name": "Conduct fuel management treatments", "category": "Preventive",
         "trigger": "Forest management",
         "desc": "Thin forests, remove deadfall, and create fuel breaks to reduce fire intensity and spread potential."},
        {"name": "Prepare evacuation plan", "category": "Preventive",
         "trigger": "Community preparedness",
         "desc": "Develop and practise evacuation plans with multiple routes, meeting points, and communication procedures."},
    ]

    for a in MITIGATION_ACTIONS:
        run_cypher("""
            MERGE (ma:Concept:MitigationAction {name: $name})
            SET ma.type              = 'Action',
                ma.category          = $category,
                ma.trigger_condition = $trigger,
                ma.description       = $desc,
                ma.project           = 'EMBER'
        """, a)
    print(f"  Created {len(MITIGATION_ACTIONS)} MitigationAction nodes.")

    # ── 4. DataInsight nodes ─────────────────────────────────────────────
    #   Key analytical findings from the model analysis. Lets the agent
    #   cite evidence when explaining predictions and recommendations.
    DATA_INSIGHTS = [
        {"name": "Top 2 features explain 65.9% of RF importance",
         "category": "Feature Importance", "source": "04_feature_importance.md",
         "desc": "Assessment hectares (42.3%) and fire spread rate (23.5%) together account for nearly two-thirds of the Random Forest model's predictive power."},
        {"name": "Class imbalance: 3.34% positive rate",
         "category": "Dataset", "source": "02_dataset_characteristics.md",
         "desc": "Only 865 of 25,862 fires (3.34%) are large fires (>=40 ha). Probabilities above ~10% should be taken seriously relative to the base rate."},
        {"name": "LR outperforms RF on primary metric",
         "category": "Model Comparison", "source": "03_model_performance.md",
         "desc": "Logistic Regression achieved AUPRC 0.584 vs Random Forest 0.513 in 10-fold cross-validation."},
        {"name": "Lightning fires have delayed detection",
         "category": "Cause Analysis", "source": "02_dataset_characteristics.md",
         "desc": "Lightning-caused fires (35% of all fires) tend to start in remote locations, leading to longer detection lags and more time for growth before suppression."},
        {"name": "High recall preferred over precision",
         "category": "Operational Policy", "source": "03_model_performance.md",
         "desc": "Models achieve 91% recall at 42% precision — missing a large fire is far costlier than a false alarm in wildfire management."},
        {"name": "Relative humidity is protective",
         "category": "Weather", "source": "04_feature_importance.md",
         "desc": "Higher relative humidity reduces fire intensity and spread probability. Only feature with a consistently negative SHAP direction."},
    ]

    for ins in DATA_INSIGHTS:
        run_cypher("""
            MERGE (di:Concept:DataInsight {name: $name})
            SET di.type            = 'Insight',
                di.category        = $category,
                di.source_document = $source,
                di.description     = $desc,
                di.project         = 'EMBER'
        """, ins)
    print(f"  Created {len(DATA_INSIGHTS)} DataInsight nodes.")

    # ── 5. Cross-layer relationships ─────────────────────────────────────

    # Feature → MitigationAction (operational triggers from doc 04)
    FEATURE_ACTION_LINKS = [
        ("Assessment Hectares", "Escalate suppression priority"),
        ("Fire Spread Rate",    "Pre-position aerial resources"),
        ("Wind Speed",          "Issue elevated fire watch"),
        ("Lightning Ignition",  "Increase aerial reconnaissance"),
        ("Dispatch Lag",        "Reduce dispatch lag"),
    ]
    for feat_name, action_name in FEATURE_ACTION_LINKS:
        run_cypher("""
            MATCH (f:Feature {name: $feat}), (a:MitigationAction {name: $action})
            MERGE (f)-[:TRIGGERS_ACTION]->(a)
        """, {"feat": feat_name, "action": action_name})

    # RiskLevel → MitigationAction (recommended actions per risk level)
    RISK_ACTION_LINKS = [
        ("LOW Risk",       ["Implement FireSmart landscaping"]),
        ("MODERATE Risk",  ["Implement FireSmart landscaping",
                            "Develop community protection plan",
                            "Create defensible space"]),
        ("HIGH Risk",      ["Pre-position aerial resources",
                            "Issue elevated fire watch",
                            "Create defensible space",
                            "Prepare evacuation plan"]),
        ("VERY HIGH Risk", ["Escalate suppression priority",
                            "Pre-position aerial resources",
                            "Issue elevated fire watch",
                            "Increase aerial reconnaissance",
                            "Prepare evacuation plan"]),
    ]
    for risk_name, actions in RISK_ACTION_LINKS:
        for action_name in actions:
            run_cypher("""
                MATCH (rl:RiskLevel {name: $risk}), (a:MitigationAction {name: $action})
                MERGE (rl)-[:RECOMMENDS]->(a)
            """, {"risk": risk_name, "action": action_name})

    # MitigationAction → existing domain Concepts
    ACTION_CONCEPT_LINKS = [
        ("Create defensible space",          "FireSmart"),
        ("Create defensible space",          "Wildland-Urban Interface"),
        ("Implement FireSmart landscaping",  "FireSmart"),
        ("Conduct fuel management treatments", "fuel management"),
        ("Prepare evacuation plan",          "evacuation"),
        ("Escalate suppression priority",    "fire suppression"),
        ("Develop community protection plan", "FireSmart"),
        ("Develop community protection plan", "Wildland-Urban Interface"),
    ]
    for action_name, concept_name in ACTION_CONCEPT_LINKS:
        run_cypher("""
            MATCH (a:MitigationAction {name: $action}), (c:Concept {name: $concept})
            MERGE (a)-[:RELATED_TO]->(c)
        """, {"action": action_name, "concept": concept_name})

    # DataInsight → relevant domain Concepts
    INSIGHT_CONCEPT_LINKS = [
        ("Top 2 features explain 65.9% of RF importance", "Random Forest"),
        ("Class imbalance: 3.34% positive rate",          "Large Fire"),
        ("LR outperforms RF on primary metric",           "Logistic Regression"),
        ("LR outperforms RF on primary metric",           "AUPRC"),
        ("Lightning fires have delayed detection",        "lightning"),
        ("Lightning fires have delayed detection",        "detection lag"),
        ("High recall preferred over precision",          "Large Fire"),
        ("Relative humidity is protective",               "Large Fire"),
    ]
    for insight_name, concept_name in INSIGHT_CONCEPT_LINKS:
        run_cypher("""
            MATCH (di:DataInsight {name: $insight}), (c:Concept {name: $concept})
            MERGE (di)-[:RELATED_TO]->(c)
        """, {"insight": insight_name, "concept": concept_name})

    print("\n  Cross-layer relationships created.")
    print("  \u2713 Analytical knowledge layer complete.")
else:
    print("Skipped — Neo4j not connected.")


# ### 4c — Attach Embeddings & Vector Index

# In[11]:


if neo4j_driver:
    print(f"Attaching {len(all_chunks)} embeddings (dim={BEST_DIM}) ...")
    BATCH = 100
    for i in range(0, len(all_chunks), BATCH):
        batch_data = [
            {"id": all_chunks[j]["id"], "emb": best_vecs[j].tolist()}
            for j in range(i, min(i+BATCH, len(all_chunks)))
        ]
        run_cypher(
            "UNWIND $rows AS r MATCH (c:Chunk {chunk_id: r.id}) SET c.embedding = r.emb",
            {"rows": batch_data},
        )
        if (i // BATCH) % 5 == 0:
            print(f"  {min(i+BATCH, len(all_chunks)):>5} / {len(all_chunks)}")

    try:
        run_cypher(f"""
            CREATE VECTOR INDEX ember_chunk_vectors IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {BEST_DIM},
                `vector.similarity_function`: 'cosine'
            }}}}
        """)
        print(f"Vector index created (dim={BEST_DIM}).")
    except Exception as e:
        print(f"Vector index note: {e}")
else:
    print("Skipped — Neo4j not connected.")


# ### 4d — Concept Nodes & Relationships

# In[12]:


if neo4j_driver:
    WILDFIRE_CONCEPTS = [
        ("Large Fire", "Classification"), ("AUPRC", "Metric"), ("AUROC", "Metric"),
        ("Random Forest", "Model"), ("Logistic Regression", "Model"),
        ("Isolation Forest", "Model"), ("SMOTE", "Technique"), ("SHAP", "Technique"),
        ("assessment hectares", "Feature"), ("spread rate", "Feature"),
        ("wind speed", "Feature"), ("lightning", "Cause"),
        ("prescribed fire", "Technique"), ("FireSmart", "Program"),
        ("Wildland-Urban Interface", "Concept"), ("fire weather index", "Index"),
        ("CWFIS", "Dataset"), ("fire behaviour", "Concept"),
        ("fuel management", "Technique"), ("evacuation", "Procedure"),
        ("Alberta Forest Protection", "Organization"), ("fire suppression", "Procedure"),
        ("detection lag", "Feature"), ("dispatch lag", "Feature"),
        ("Size Class D", "Classification"), ("Size Class E", "Classification"),
    ]

    for name, ctype in WILDFIRE_CONCEPTS:
        run_cypher(
            "MERGE (c:Concept {name: $name}) SET c.type = $type, c.project = 'EMBER'",
            {"name": name, "type": ctype},
        )
    print(f"Created {len(WILDFIRE_CONCEPTS)} Concept nodes.")

    print("Linking chunks to concepts ...")
    for name, _ in tqdm(WILDFIRE_CONCEPTS):
        run_cypher(
            """MATCH (ch:Chunk {project: 'EMBER'}), (co:Concept {name: $name})
            WHERE toLower(ch.text) CONTAINS toLower($name)
            MERGE (ch)-[:MENTIONS]->(co)""",
            {"name": name},
        )

    run_cypher("""
        MATCH (c1:Concept)<-[:MENTIONS]-(ch:Chunk)-[:MENTIONS]->(c2:Concept)
        WHERE id(c1) < id(c2)
        MERGE (c1)-[:RELATED_TO]->(c2)
    """)
    print("Concept relationships created.")
else:
    print("Skipped — Neo4j not connected.")


# ### 4e — Verify Graph

# In[13]:


if neo4j_driver:
    print("\u2500\u2500 Graph Summary \u2500\u2500")
    # Core nodes
    for label in ["Document", "Chunk", "Concept"]:
        n = run_cypher(f"MATCH (n:{label} {{project:'EMBER'}}) RETURN count(n) AS c")[0]["c"]
        print(f"  {label:<20}: {n}")
    # Analytical layer nodes (subset of Concept via dual labels)
    print()
    for label in ["Feature", "RiskLevel", "MitigationAction", "DataInsight"]:
        n = run_cypher(f"MATCH (n:{label} {{project:'EMBER'}}) RETURN count(n) AS c")[0]["c"]
        print(f"  {label:<20}: {n}")
    # Relationships
    print()
    for rel in ["CONTAINS", "MENTIONS", "RELATED_TO", "PREDICTS",
                "TRIGGERS_ACTION", "ASSESSES", "RECOMMENDS"]:
        n = run_cypher(
            f"MATCH (a {{project:'EMBER'}})-[r:{rel}]->(b) RETURN count(r) AS c"
        )[0]["c"]
        print(f"  :{rel:<20}: {n}")
else:
    print("Neo4j not connected \u2014 graph not available.")


# ---
# ## 5 — Agent Setup (Tools + ReAct Loop)

# ### 5a — Tool Definitions

# In[14]:


# ── 1. predict_fire_risk ──────────────────────────────────────────────────────
def predict_fire_risk(inputs_json: str) -> str:
    import json as _json
    try:
        inputs = _json.loads(inputs_json) if isinstance(inputs_json, str) else inputs_json
    except Exception as e:
        return f"Error parsing inputs JSON: {e}"
    load_ml_models()
    if _ml_scaler is None:
        return "MISSING: scaler.pkl not found in results/."
    try:
        row  = np.array([[float(inputs.get(f, 0.0)) for f in FEATURE_ORDER]])
        X_sc = _ml_scaler.transform(row)
    except Exception as e:
        return f"Error building feature vector: {e}"
    probs: Dict[str, float] = {}
    if _rf_model is not None:
        probs["Random Forest"] = float(_rf_model.predict_proba(X_sc)[0, 1])
    if _lr_model is not None:
        probs["Logistic Regression"] = float(_lr_model.predict_proba(X_sc)[0, 1])
    if not probs:
        return "No ML models loaded."
    ensemble = sum(probs.values()) / len(probs)
    if   ensemble < 0.10: risk_level, risk_desc = "LOW",       "Conditions not strongly indicative of a large fire."
    elif ensemble < 0.25: risk_level, risk_desc = "MODERATE",  "Some risk factors present; monitor closely."
    elif ensemble < 0.45: risk_level, risk_desc = "HIGH",      "Multiple risk factors elevated; consider pre-emptive action."
    else:                 risk_level, risk_desc = "VERY HIGH", "Conditions strongly associated with large-fire escalation."
    lines = [
        "=== EMBER FIRE RISK PREDICTION ===",
        f"Ensemble probability : {ensemble:.1%}  \u2192  {risk_level} RISK",
        f"Interpretation       : {risk_desc}", "",
    ]
    for name, prob in probs.items():
        lines.append(f"  {name:<24}: {prob:.1%}")
    lines += ["", "Key inputs provided:"]
    for k, v in inputs.items():
        lines.append(f"  {k}: {v}")
    lines += ["", "Note: class imbalance (3.3% positive rate) means probabilities are calibrated",
              "relative to the historical base rate. Values above ~10% should be treated seriously."]
    return "\n".join(lines)


# ── 2. rag_retrieve ───────────────────────────────────────────────────────────
def rag_retrieve(query: str, k: int = 5) -> str:
    q_vec = embed_texts([query], model=BEST_MODEL)[0].tolist()
    res = collection.query(query_embeddings=[q_vec], n_results=k,
                           include=["documents", "metadatas", "distances"])
    parts = []
    for i, (doc, meta, dist) in enumerate(
        zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ):
        parts.append(f"[Source {i+1} | {meta['doc_title']} | score={1-dist:.3f}]\n" + doc[:500])
    return "\n\n".join(parts)


# ── 3. grag_vector_search ─────────────────────────────────────────────────────
def grag_vector_search(query: str, k: int = 5) -> str:
    if neo4j_driver is None:
        return "Neo4j not connected \u2014 use rag_retrieve instead."
    q_vec = embed_texts([query], model=BEST_MODEL)[0].tolist()
    try:
        rows = run_cypher("""
            CALL db.index.vector.queryNodes('ember_chunk_vectors', $k, $vec)
            YIELD node AS ch, score
            MATCH (d:Document)-[:CONTAINS]->(ch)
            OPTIONAL MATCH (ch)-[:MENTIONS]->(co:Concept)
            RETURN d.title AS doc, ch.text AS text, score, collect(co.name) AS concepts
        """, {"k": k, "vec": q_vec})
    except Exception as e:
        return f"Graph vector search unavailable ({e}). Use rag_retrieve."
    if not rows:
        return "No results from graph vector search."
    parts = []
    for i, r in enumerate(rows):
        concepts = ", ".join(r["concepts"]) if r["concepts"] else "none"
        parts.append(f"[Source {i+1} | {r['doc']} | score={r['score']:.3f}]\nConcepts: {concepts}\n" + r["text"][:500])
    return "\n\n".join(parts)


# ── 4. grag_concept_neighbors ─────────────────────────────────────────────────
def grag_concept_neighbors(concept_name: str) -> str:
    if neo4j_driver is None:
        return "Neo4j not connected."
    rows = run_cypher("""
        MATCH (c:Concept)-[:RELATED_TO]-(c2:Concept)
        WHERE toLower(c.name) CONTAINS toLower($name)
        RETURN c.name AS source, c2.name AS neighbor, c2.type AS type
        ORDER BY c2.name LIMIT 25
    """, {"name": concept_name})
    if not rows:
        return f"No related concepts found for '{concept_name}'."
    return "\n".join(f"  {r['source']} \u2194 {r['neighbor']} ({r['type']})" for r in rows)


# ── 5. grag_concept_chunks ────────────────────────────────────────────────────
def grag_concept_chunks(concept_name: str) -> str:
    if neo4j_driver is None:
        return "Neo4j not connected."
    rows = run_cypher("""
        MATCH (co:Concept)<-[:MENTIONS]-(ch:Chunk)<-[:CONTAINS]-(d:Document)
        WHERE toLower(co.name) CONTAINS toLower($name)
        RETURN d.title AS doc, ch.text AS text LIMIT 4
    """, {"name": concept_name})
    if not rows:
        return f"No chunks found mentioning '{concept_name}'."
    return "\n\n".join(f"[{r['doc']}]\n{r['text'][:500]}" for r in rows)


# ── 6. grag_graph_summary ─────────────────────────────────────────────────────
def grag_graph_summary() -> str:
    if neo4j_driver is None:
        return "Neo4j not connected."
    docs = run_cypher(
        "MATCH (d:Document {project:'EMBER'}) "
        "RETURN d.title AS title, size((d)-[:CONTAINS]->()) AS chunks"
    )
    return "Documents in graph:\n" + "\n".join(f"  - {r['title']} ({r['chunks']} chunks)" for r in docs)


TOOLS: Dict[str, callable] = {
    "predict_fire_risk":      predict_fire_risk,
    "rag_retrieve":           rag_retrieve,
    "grag_vector_search":     grag_vector_search,
    "grag_concept_neighbors": grag_concept_neighbors,
    "grag_concept_chunks":    grag_concept_chunks,
    "grag_graph_summary":     grag_graph_summary,
}

print("Tools registered:", list(TOOLS.keys()))


# ### 5b — ReAct Agent

# In[15]:


def llm_call(prompt: str, model: str = AGENT_LLM) -> str:
    resp = requests.post(
        f"{OLLAMA_LOCAL_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


SYSTEM_PROMPT = """You are EMBER \u2014 an expert wildfire risk analyst assistant for Alberta, Canada.

You have access to predictive ML models and a document retrieval system.

Available tools:
  predict_fire_risk('<json>')           \u2014 run ML models to predict large-fire probability
                                          JSON keys: ASSESSMENT_HECTARES, FIRE_SPREAD_RATE,
                                          WIND_SPEED, CAUSE_BINARY (0/1), TEMPERATURE,
                                          FOREST_AREA_ENC, FIRE_TYPE_ENC, DISPATCH_LAG_HRS,
                                          FUEL_TYPE_ENC, RELATIVE_HUMIDITY, FIRE_MONTH,
                                          DETECTION_LAG_HRS
  rag_retrieve(\"<query>\")               \u2014 semantic search over mitigation document chunks
  grag_vector_search(\"<query>\")         \u2014 semantic search via Neo4j graph index
  grag_concept_neighbors(\"<concept>\")   \u2014 related concepts in knowledge graph
  grag_concept_chunks(\"<concept>\")      \u2014 document chunks mentioning a concept
  grag_graph_summary()                  \u2014 list all documents in graph

Rules:
- Think step by step.
- When the user asks for a fire risk prediction AND provides numerical inputs, call predict_fire_risk FIRST.
- After getting a prediction, always follow up with rag_retrieve to provide mitigation guidance.
- For general mitigation questions, use rag_retrieve or grag_vector_search.
- For conceptual/relational questions, combine grag_concept_neighbors + grag_concept_chunks.
- To call a tool write EXACTLY:  ACTION: tool_name(\"argument\")
- For JSON arguments:            ACTION: predict_fire_risk('{\"ASSESSMENT_HECTARES\": 5.0, ...}')
- For no-argument tools write:   ACTION: grag_graph_summary()
- When you have enough information write:  FINAL ANSWER: <your answer>

Always end your response with actionable, specific guidance the user can act on.
"""


def _parse_action(text: str) -> Tuple[Optional[str], Optional[str]]:
    m = re.search(r"ACTION:\s*(\w+)\('(.+?)'\)", text, re.DOTALL)
    if m: return m.group(1), m.group(2)
    m = re.search(r'ACTION:\s*(\w+)\("(.+?)"\)', text, re.DOTALL)
    if m: return m.group(1), m.group(2)
    m = re.search(r'ACTION:\s*(\w+)\(\)', text)
    if m: return m.group(1), ""
    return None, None


def react_agent(question: str, max_steps: int = 8, verbose: bool = True) -> str:
    sep = "\u2550" * 72
    if verbose:
        print(f"\n{sep}")
        print(f"QUESTION: {question}")
        print(sep)
    history = [f"QUESTION: {question}"]
    for step in range(max_steps):
        prompt = SYSTEM_PROMPT + "\n\n" + "\n".join(history) + "\n\nTHOUGHT:"
        response = llm_call(prompt)
        if verbose:
            print(f"\n[Step {step + 1}]")
            print(textwrap.fill(response[:400], width=90, subsequent_indent="  "))
        history.append(f"THOUGHT: {response}")
        if "FINAL ANSWER:" in response:
            answer = response.split("FINAL ANSWER:", 1)[-1].strip()
            if verbose:
                print(f"\n{sep}\nFINAL ANSWER:")
                print(textwrap.fill(answer, width=90))
            return answer
        tool_name, arg = _parse_action(response)
        if tool_name and tool_name in TOOLS:
            try:
                obs = TOOLS[tool_name](arg) if arg else TOOLS[tool_name]()
                obs = obs[:1500]
            except Exception as e:
                obs = f"Tool error: {e}"
            if verbose:
                print(f"\n  \u2192 OBSERVATION from {tool_name}():")
                print(textwrap.indent(obs[:600], "    "))
            history.append(f'ACTION: {tool_name}(\"{arg}\")')
            history.append(f"OBSERVATION: {obs}")
        else:
            history.append("THOUGHT: I should now provide a final answer.")
    prompt = SYSTEM_PROMPT + "\n\n" + "\n".join(history) + "\n\nFINAL ANSWER:"
    return llm_call(prompt)


def ask(question: str, verbose: bool = True) -> str:
    """Ask the RAG/GRAG agent a question."""
    return react_agent(question, verbose=verbose)


print("ReAct agent ready.")
print(f"Agent LLM : {AGENT_LLM}")
print(f"Tools     : {list(TOOLS.keys())}")


# ---
# ## 6 — Test & Interactive Use

# ### 6a — Prediction + Mitigation (end-to-end)

# In[16]:


answer = ask(
    "I have a fire that is currently 8 hectares, spreading at 45 m/min, with 60 km/h winds, "
    "lightning-caused, temperature 32\u00b0C, forest area code 1, fire type 0, dispatch lag 2 hours, "
    "fuel type 3, relative humidity 18%, ignition in July, detection lag 1 hour. "
    "What is the probability this becomes a large fire and what should I do?"
)


# ### 6b — Mitigation Questions

# In[17]:


ask("What are the main causes of large wildfires in Alberta and how can communities prepare?")


# In[18]:


ask("What strategies does FireSmart recommend for reducing wildfire risk at Wildland-Urban Interface?")


# In[19]:


ask("How does wind speed affect wildfire behaviour and escalation risk?")


# In[21]:


ask("What data layers and services does CWFIS provide for wildfire monitoring?")


# ### 6c — Interactive

# In[22]:


ask("What are best practices for farm owners to reduce wildfire ignition risk near buildings?")


# ### 6d — Analytical Knowledge Layer Tests
# 
# These questions specifically exercise the new graph structures (Feature, RiskLevel, MitigationAction, DataInsight nodes) and the new relationship types (PREDICTS, TRIGGERS_ACTION, ASSESSES, RECOMMENDS).
# 

# In[23]:


# Feature importance — should traverse Feature nodes and PREDICTS relationships
ask("What are the top predictors of large fire escalation and how important is each one?")


# In[ ]:


# Risk level explanation — should discover RiskLevel nodes and ASSESSES relationship
ask("If a fire has a 30% predicted probability of becoming large, what does that mean and what actions should be taken?")


# In[24]:


# Feature → Action chain — should traverse TRIGGERS_ACTION relationships
ask("Wind speeds are forecasted at 50 km/h today. What specific operational actions should fire managers take and why?")


# In[25]:


# DataInsight discovery — should find insights linked to concepts via RELATED_TO
ask("What does the data tell us about lightning-caused fires and why are they particularly dangerous?")


# In[26]:


# RiskLevel → MitigationAction via RECOMMENDS — full prediction + graph-guided advice
ask(
    "A fire was just reported: 12 hectares, spreading at 60 m/min, wind 55 km/h, "
    "lightning-caused, temperature 35°C, forest area code 2, fire type 1, dispatch lag 3 hours, "
    "fuel type 4, relative humidity 12%, ignition in August, detection lag 2 hours. "
    "Predict the risk level and tell me exactly what the recommended actions are for that risk level."
)


# In[27]:


# Preventive actions — should find MitigationAction nodes with category=Preventive + FireSmart concept links
ask("What preventive measures should a community at the Wildland-Urban Interface implement before fire season?")


# In[28]:


# Model comparison insight — should discover DataInsight about LR vs RF + AUPRC concept
ask("Which model performs better for predicting large fires — Random Forest or Logistic Regression — and why?")


# ### 6e — Direct Graph Queries (Analytical Layer Verification)
# 
# These cells query the graph directly (bypassing the agent) to verify the analytical knowledge layer is correctly wired.
# 

# In[29]:


if neo4j_driver:
    # ── Feature importance ranking from graph ─────────────────────────────
    print("── Features ranked by importance ──")
    rows = run_cypher("""
        MATCH (f:Feature {project: 'EMBER'})-[:PREDICTS]->(lf:Concept {name: 'Large Fire'})
        RETURN f.name AS feature, f.importance_rank AS rank,
               f.gini_importance AS gini, f.shap_direction AS direction
        ORDER BY f.importance_rank
    """)
    for r in rows:
        print(f"  #{r['rank']:>2}  {r['feature']:<24} gini={r['gini']:.4f}  dir={r['direction']}")

    # ── Risk level → recommended actions ──────────────────────────────────
    print("\n── Risk levels and their recommended actions ──")
    rows = run_cypher("""
        MATCH (rl:RiskLevel {project: 'EMBER'})-[:RECOMMENDS]->(a:MitigationAction)
        RETURN rl.level AS risk, rl.min_probability AS min_p, rl.max_probability AS max_p,
               collect(a.name) AS actions
        ORDER BY rl.min_probability
    """)
    for r in rows:
        print(f"\n  {r['risk']} ({r['min_p']:.0%}–{r['max_p']:.0%}):")
        for a in r["actions"]:
            print(f"    → {a}")

    # ── Feature → Action trigger chains ───────────────────────────────────
    print("\n── Feature → triggered action chains ──")
    rows = run_cypher("""
        MATCH (f:Feature)-[:TRIGGERS_ACTION]->(a:MitigationAction)
        RETURN f.name AS feature, f.gini_importance AS gini, a.name AS action, a.category AS cat
        ORDER BY f.importance_rank
    """)
    for r in rows:
        print(f"  {r['feature']:<24} → {r['action']} ({r['cat']})")

    # ── Data insights and their linked concepts ───────────────────────────
    print("\n── Data insights → linked concepts ──")
    rows = run_cypher("""
        MATCH (di:DataInsight {project: 'EMBER'})-[:RELATED_TO]->(c:Concept)
        RETURN di.name AS insight, di.category AS cat, collect(c.name) AS concepts
    """)
    for r in rows:
        print(f"  [{r['cat']}] {r['insight']}")
        print(f"    → {', '.join(r['concepts'])}")
else:
    print("Neo4j not connected — skipping graph verification.")

