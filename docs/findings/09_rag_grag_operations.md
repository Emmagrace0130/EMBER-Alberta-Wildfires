# EMBER — RAG & GRAG Operations Guide

**Project:** EMBER — Alberta Wildfire Risk Assessor  
**Date:** April 15, 2026  
**Purpose:** Technical reference describing the Retrieval-Augmented Generation (RAG) and Graph-RAG (GRAG) systems, their operations, and the knowledge graph structure.

---

## Table of Contents

1. [What is RAG?](#1--what-is-rag)
2. [What is GRAG?](#2--what-is-grag-graph-rag)
3. [How EMBER Uses RAG and GRAG Together](#3--how-ember-uses-rag-and-grag-together)
4. [The Embedding Pipeline](#4--the-embedding-pipeline)
5. [The ChromaDB Vector Store (RAG)](#5--the-chromadb-vector-store-rag)
6. [The Neo4j Knowledge Graph (GRAG)](#6--the-neo4j-knowledge-graph-grag)
7. [Agent Tools and the ReAct Loop](#7--agent-tools-and-the-react-loop)
8. [Data Flow Diagrams](#8--data-flow-diagrams)

---

## 1 — What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that enhances a language model's responses by first retrieving relevant documents from an external knowledge base, then passing those documents as context to the LLM when generating an answer.

### The Problem RAG Solves

Large Language Models are trained on general data and have no knowledge of your specific documents, data, or domain expertise. They can also "hallucinate" — confidently stating incorrect information. RAG solves both problems by:

1. **Grounding** the LLM's response in actual source documents
2. **Extending** the LLM's knowledge to include your private/domain-specific data
3. **Providing citations** so users can verify the information

### How RAG Works (Step by Step)

```
User Question
    │
    ▼
┌─────────────────────┐
│  1. EMBED the query  │  Convert the question into a vector (list of numbers)
│     using the same   │  that captures its semantic meaning.
│     embedding model  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  2. SEARCH the       │  Compare the query vector against all document chunk
│     vector store     │  vectors using cosine similarity. Return the top-k
│     (ChromaDB)       │  most similar chunks.
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  3. BUILD a prompt   │  Combine the user's question with the retrieved
│     with context     │  chunks as context for the LLM.
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  4. GENERATE answer  │  The LLM reads the context and generates a response
│     via LLM          │  grounded in the retrieved documents.
└─────────────────────┘
```

### Key Concepts

- **Embedding:** A numerical vector representation of text. Texts with similar meaning have vectors that are close together in vector space.
- **Vector Store:** A database optimised for storing and searching embeddings by similarity (cosine, dot product, etc.).
- **Chunk:** A segment of a larger document. Documents are split into chunks because embedding models have limited input sizes and smaller chunks produce more focused retrieval.
- **Top-k Retrieval:** Returning the k most similar chunks to the query. EMBER uses k=5 by default.

---

## 2 — What is GRAG (Graph-RAG)?

**Graph-RAG (GRAG)** extends standard RAG by adding a **knowledge graph** layer on top of the vector store. Instead of only retrieving chunks by vector similarity, GRAG can also traverse structured relationships between concepts to find contextually relevant information.

### Why Add a Graph?

Standard RAG has limitations:
- It only finds chunks that are **semantically similar** to the query
- It cannot follow **chains of reasoning** (e.g., "Feature X → increases risk → triggers Action Y")
- It treats all chunks as independent — it doesn't know which chunks are **related to each other**

A knowledge graph adds:
- **Structured relationships** between entities (concepts, features, actions)
- **Multi-hop traversal** — follow chains of relationships to find indirectly related information
- **Concept-level retrieval** — search by concept name rather than text similarity
- **Contextual enrichment** — augment retrieved chunks with their graph neighbours

### How GRAG Extends RAG

```
Standard RAG:          Query → Vector Search → Chunks → LLM

GRAG adds:             Query → Vector Search → Chunks ──┐
                                                         ├──→ LLM
                       Query → Graph Traversal → Related ┘
                                Concepts, Actions,
                                Insights, Features
```

---

## 3 — How EMBER Uses RAG and GRAG Together

EMBER's agent can use both systems depending on the question type:

| Question Type | Primary System | Example |
|---|---|---|
| Mitigation guidance | RAG | "How do I protect my farm from wildfire?" |
| Fire risk prediction | ML Models | "What's the risk for this fire?" |
| Feature explanation | GRAG | "What factors drive large fire risk?" |
| Concept relationships | GRAG | "What's related to FireSmart?" |
| Document overview | GRAG | "What documents are in the system?" |
| Combined prediction + advice | ML + RAG + GRAG | "Predict risk and tell me what to do" |

The agent decides which tools to use at each step through a **ReAct reasoning loop** (see Section 7).

---

## 4 — The Embedding Pipeline

### Embedding Model

- **Model:** `nomic-embed-text:latest` (selected via grid search, MRR = 0.984)
- **Dimensions:** 768
- **Server:** Ollama running locally on port 11434
- **API endpoint:** `/api/embed` (batch mode) with `/api/embeddings` fallback (single text)

### How Embeddings Are Created

1. **Text input** → truncated to 2000 characters max (`MAX_EMBED_CHARS`)
2. **Batch request** → sent to Ollama `/api/embed` with up to 16 texts per batch
3. **Error handling:**
   - If batch fails (HTTP 400/500), fall back to single-text mode
   - If single text fails, insert a zero-vector (768 zeros) as placeholder
   - Warm-up call on first use to detect which API endpoint the model supports

### Chunking Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Chunk size | 350 words | Balances context coverage with retrieval precision |
| Overlap | 70 words | Ensures concepts near chunk boundaries aren't lost |
| Minimum | 25 words | Filters out fragments too short to be useful |

### Source Corpus

6 PDF documents in `docs/data_docs/`:

| Document | Description |
|---|---|
| FireSmart Community Protection Guide | Community-level wildfire mitigation strategies |
| CWFIS Data Services | Canadian Wildland Fire Information System overview |
| Farm & Acreage Wildfire Risk Reduction | Agricultural wildfire mitigation guide |
| Alberta WUI Fires | Wildland-Urban Interface fire analysis |
| Prevention Plan Template | Wildfire prevention planning framework |
| USFS Rocky Mountain GTR-292 | US Forest Service technical report on fire risk |

---

## 5 — The ChromaDB Vector Store (RAG)

### Overview

ChromaDB is an open-source vector database that stores embeddings and supports fast similarity search.

- **Location:** `vector_store/` (persistent on disk)
- **Collection:** `wildfire_docs`
- **Contents:** 510 chunks, each with:
  - `document` — the chunk text
  - `embedding` — 768-dimensional vector
  - `metadata` — `{doc_title, chunk_index}` for source attribution

### How RAG Retrieval Works in EMBER

```python
# 1. Embed the user's query
query_vector = embed_texts(["user question"], model="nomic-embed-text:latest")

# 2. Search ChromaDB for the 5 most similar chunks
results = collection.query(
    query_embeddings=[query_vector],
    n_results=5,
    include=["documents", "metadatas", "distances"]
)

# 3. Return formatted results with source attribution and similarity scores
```

The similarity metric is **cosine distance** — lower distance = higher similarity. Scores are converted to similarity as `1 - distance`.

---

## 6 — The Neo4j Knowledge Graph (GRAG)

### Overview

Neo4j is a graph database that stores entities (nodes) and their relationships (edges). EMBER's graph encodes both the document corpus and structured analytical knowledge.

### Infrastructure

- **Database:** Neo4j 5.x running in Docker container `ember-neo4j`
- **Ports:** 7687 (Bolt protocol), 7474 (HTTP browser)
- **Auth:** `neo4j/password`
- **Management:** Python `docker` SDK for container lifecycle

---

### Node Types

The graph has two layers: a **core document layer** and an **analytical knowledge layer**.

#### Core Layer

| Label | Count | Properties | Description |
|---|---|---|---|
| **Document** | 6 | `title`, `path`, `project` | One per source PDF |
| **Chunk** | 510 | `chunk_id`, `text`, `doc_title`, `embedding`, `project` | Text segments with 768-dim vectors |
| **Concept** | 26 | `name`, `type`, `project` | Domain vocabulary terms |

#### Analytical Knowledge Layer

All analytical nodes carry a dual `:Concept` label so existing GRAG tools discover them through concept traversal.

| Label | Count | Key Properties | Description |
|---|---|---|---|
| **Feature** | 12 | `column_name`, `importance_rank`, `gini_importance`, `shap_direction`, `description` | ML feature metadata from SHAP/Gini analysis |
| **RiskLevel** | 4 | `level`, `min_probability`, `max_probability`, `description` | Prediction probability thresholds |
| **MitigationAction** | 10 | `category` (Operational/Preventive), `trigger_condition`, `description` | Actionable mitigation guidance |
| **DataInsight** | 6 | `category`, `source_document`, `description` | Key findings from the ML analysis |

---

### Concept Node Details (26 Domain Concepts)

| Name | Type | Description |
|---|---|---|
| Large Fire | Classification | Target class: fires ≥ 40 ha (Size Class D or E) |
| Size Class D | Classification | Fires 40–200 ha |
| Size Class E | Classification | Fires ≥ 200 ha |
| AUPRC | Metric | Primary evaluation metric (Area Under PRC) |
| AUROC | Metric | Secondary evaluation metric (Area Under ROC) |
| Random Forest | Model | Ensemble tree classifier (n_estimators=200) |
| Logistic Regression | Model | Linear classifier (class_weight=balanced) |
| Isolation Forest | Model | Unsupervised anomaly detector |
| SMOTE | Technique | Minority class oversampling |
| SHAP | Technique | Feature importance via Shapley values |
| assessment hectares | Feature | Fire size at assessment |
| spread rate | Feature | Fire spread rate at assessment |
| wind speed | Feature | Wind speed at assessment |
| lightning | Cause | Lightning-caused ignition |
| prescribed fire | Technique | Controlled burns for fuel management |
| FireSmart | Program | Community wildfire protection program |
| Wildland-Urban Interface | Concept | Zone where structures meet wildland vegetation |
| fire weather index | Index | Fire weather severity measure |
| CWFIS | Dataset | Canadian Wildland Fire Information System |
| fire behaviour | Concept | How fire spreads and intensifies |
| fuel management | Technique | Reducing available fuel to limit fire spread |
| evacuation | Procedure | Emergency population relocation |
| Alberta Forest Protection | Organization | Provincial fire management authority |
| fire suppression | Procedure | Active firefighting operations |
| detection lag | Feature | Hours from fire start to discovery |
| dispatch lag | Feature | Hours from discovery to resource dispatch |

---

### Relationship Types

#### Core Layer Relationships

| Relationship | Pattern | Count | Description |
|---|---|---|---|
| `CONTAINS` | `(Document)-[:CONTAINS]->(Chunk)` | ~510 | Document contains chunk |
| `MENTIONS` | `(Chunk)-[:MENTIONS]->(Concept)` | Variable | Text match: chunk text contains concept name |
| `RELATED_TO` | `(Concept)-[:RELATED_TO]->(Concept)` | Variable | Co-occurrence: two concepts mentioned in the same chunk |

#### Analytical Layer Relationships

| Relationship | Pattern | Count | Description |
|---|---|---|---|
| `PREDICTS` | `(Feature)-[:PREDICTS]->(Concept:Large Fire)` | 12 | Feature is a predictor of large fire |
| `TRIGGERS_ACTION` | `(Feature)-[:TRIGGERS_ACTION]->(MitigationAction)` | 5 | Risk factor triggers recommended response |
| `ASSESSES` | `(RiskLevel)-[:ASSESSES]->(Concept:Large Fire)` | 4 | Risk level applies to large fire assessment |
| `RECOMMENDS` | `(RiskLevel)-[:RECOMMENDS]->(MitigationAction)` | ~14 | Risk level recommends specific actions |
| `RELATED_TO` | `(MitigationAction)-[:RELATED_TO]->(Concept)` | 8 | Action relates to domain concept |
| `RELATED_TO` | `(DataInsight)-[:RELATED_TO]->(Concept)` | 8 | Insight references domain concept |

---

### Vector Index

```
Name:       ember_chunk_vectors
Node label: Chunk
Property:   embedding
Dimensions: 768
Similarity: cosine
```

This index enables **semantic search directly within Neo4j**, combining vector similarity with graph traversal in a single query. For example, finding similar chunks AND their related concepts in one Cypher query.

---

### Graph Traversal Examples

**Find features that predict large fires:**
```cypher
MATCH (f:Feature)-[:PREDICTS]->(lf:Concept {name: 'Large Fire'})
RETURN f.name, f.gini_importance, f.shap_direction
ORDER BY f.importance_rank
```

**Find recommended actions for a high-risk prediction:**
```cypher
MATCH (rl:RiskLevel {level: 'VERY_HIGH'})-[:RECOMMENDS]->(a:MitigationAction)
RETURN a.name, a.category, a.description
```

**Trace from a feature to its recommended action:**
```cypher
MATCH (f:Feature {name: 'Wind Speed'})-[:TRIGGERS_ACTION]->(a:MitigationAction)
RETURN f.name, f.gini_importance, a.name, a.description
```

**Find all concepts related to FireSmart:**
```cypher
MATCH (c:Concept {name: 'FireSmart'})-[:RELATED_TO]-(c2:Concept)
RETURN c2.name, c2.type
```

**Vector search with concept enrichment:**
```cypher
CALL db.index.vector.queryNodes('ember_chunk_vectors', 5, $queryVector)
YIELD node AS ch, score
MATCH (d:Document)-[:CONTAINS]->(ch)
OPTIONAL MATCH (ch)-[:MENTIONS]->(co:Concept)
RETURN d.title, ch.text, score, collect(co.name) AS concepts
```

---

## 7 — Agent Tools and the ReAct Loop

### Tools

EMBER's agent has 6 tools that span the ML, RAG, and GRAG systems:

| Tool | System | Description |
|---|---|---|
| `predict_fire_risk(json)` | ML | Run RF + LR ensemble; return probability + risk level |
| `rag_retrieve(query)` | RAG | ChromaDB cosine search → top-5 mitigation chunks |
| `grag_vector_search(query)` | GRAG | Neo4j vector index → chunks + graph concepts |
| `grag_concept_neighbors(concept)` | GRAG | Graph traversal: find related concepts |
| `grag_concept_chunks(concept)` | GRAG | Graph → chunks that mention a concept |
| `grag_graph_summary()` | GRAG | List all documents and chunk counts |

### How the Agent Decides Which Tool to Use

The agent follows a **ReAct (Reason + Act) loop**:

```
1. THINK  — Reason about what to do next
2. ACT    — Call a tool (ACTION: tool_name("argument"))
3. OBSERVE — Read the tool's output
4. Repeat until enough information is gathered
5. FINAL ANSWER — Synthesise everything into a response
```

### ReAct Loop Example

```
QUESTION: "I have a fire spreading at 45 m/min with 60 km/h winds.
           What's the risk and what should I do?"

THOUGHT: The user wants a risk prediction. I should call predict_fire_risk
         with the provided values, then follow up with mitigation advice.

ACTION: predict_fire_risk('{"FIRE_SPREAD_RATE": 45, "WIND_SPEED": 60, ...}')

OBSERVATION: Ensemble probability: 72.3% → VERY HIGH RISK
             Random Forest: 68.1%, Logistic Regression: 76.5%

THOUGHT: This is very high risk. I should retrieve mitigation guidance
         relevant to high wind and rapid spread conditions.

ACTION: rag_retrieve("wildfire mitigation high wind conditions rapid spread")

OBSERVATION: [Source 1 | FireSmart Guide | score=0.847]
             Create defensible space of at least 10 metres ...

THOUGHT: I have enough information to give a complete answer.

FINAL ANSWER: Your fire shows a **VERY HIGH** escalation risk (72.3%).
              Both models strongly agree. Recommended immediate actions:
              1. Escalate suppression priority ...
              2. Pre-position aerial resources ...
              [continues with specific guidance from retrieved documents]
```

### System Prompt

The agent operates under a system prompt that instructs it to:
1. Always call `predict_fire_risk` first when numerical inputs are provided
2. Follow up predictions with `rag_retrieve` for mitigation guidance
3. Use GRAG tools for conceptual and relational questions
4. End every response with actionable, specific guidance
5. Limit to 8 reasoning steps maximum

### LLM

- **Model:** `llama3.1:8b` running locally via Ollama
- **No external API calls** — all inference is local
- **Endpoint:** `POST /api/generate` with `stream: false`

---

## 8 — Data Flow Diagrams

### End-to-End Pipeline (Build Time)

```
PDFs (6 documents)
    │
    ▼
┌─────────────────┐
│  pypdf extract   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌───────────────────────┐
│  Chunk (350w,   │────▶│  results/              │
│  70w overlap)   │     │  all_chunks.json       │
└────────┬────────┘     └───────────────────────┘
         │
         ▼
┌─────────────────┐     ┌───────────────────────┐
│  Embed via      │────▶│  results/              │
│  nomic-embed    │     │  best_embeddings.npy   │
└────────┬────────┘     └───────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
ChromaDB    Neo4j
(RAG)       (GRAG)
```

### Query Time (RAG Path)

```
User Question ──▶ embed_texts() ──▶ ChromaDB.query()
                                         │
                                         ▼
                                    Top-5 Chunks
                                         │
                                         ▼
                              LLM generates answer
                              grounded in chunks
```

### Query Time (GRAG Path)

```
User Question ──▶ embed_texts() ──▶ Neo4j vector search
                                         │
                                    ┌────┴────┐
                                    ▼         ▼
                               Chunks    Concept neighbours
                                    │         │
                                    └────┬────┘
                                         ▼
                              LLM generates answer
                              with chunk context +
                              graph relationships
```

### Full Agent Flow

```
User Question
    │
    ▼
┌──────────────┐
│  ReAct Loop  │◄──────────────────────────────┐
│  (max 8      │                               │
│   steps)     │                               │
└──────┬───────┘                               │
       │                                       │
       ▼                                       │
┌──────────────┐     ┌───────────────────┐     │
│  THINK       │────▶│  Choose next tool  │────┘
└──────────────┘     └────────┬──────────┘
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
              ML Models    ChromaDB   Neo4j
              (predict)    (RAG)     (GRAG)
                    │         │         │
                    └─────────┼─────────┘
                              ▼
                       FINAL ANSWER
                    (grounded in data,
                     models, & documents)
```
