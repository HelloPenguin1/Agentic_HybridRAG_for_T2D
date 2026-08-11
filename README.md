# Agentic GraphRAG for Type 2 Diabetes Nursing Management

A hybrid Retrieval-Augmented Generation (RAG) system that combines a structured pharmaceutical knowledge graph with a dense clinical document vector store, orchestrated by an agentic LangGraph pipeline, to support Type 2 Diabetes (T2D) clinical decision-making for nursing workflows.

---

## Problem Statement

Type 2 Diabetes management in nursing settings demands accurate, multi-faceted information — from drug-drug interactions and ATC classifications, to clinical guidelines on HbA1c targets, dosing schedules, and dietary restrictions. Static RAG pipelines fail here because:

- **Structured relational queries** (e.g. *"which drugs interact with Metformin?"*) require graph traversal, not vector similarity.
- **Prose clinical questions** (e.g. *"what is the recommended HbA1c target?"*) are better served by dense semantic retrieval over guideline documents.
- **Real-time or temporal queries** require live web search, not stale local indices.
- Standard RAG provides no hallucination audit trail or citation traceability.

This system addresses all four failure modes with a single unified pipeline.

---

## Architecture

The final workflow is a **Unified Threshold-Gated Hybrid Routing Pipeline** built with LangGraph:

```
START → Router ─┬─ graph     → Graph Retriever ─┬─ (has data)           → Evidence Gate
                │                               └─ (empty + graph-only) → Vector Retriever
                ├─ vector    → Vector Retriever ──────────────────────→ Evidence Gate
                ├─ both      → Graph + Vector Retriever (parallel) ───→ Evidence Gate
                └─ real_time → Web Search ────────────────────────────→ Synthesizer

Evidence Gate ──┬─ (sufficient local evidence) → Synthesizer
                └─ (local gap detected)         → Web Search → Synthesizer

Synthesizer → Hallucination Grader ─┬─ faithful     → Citation Agent → END
                                    └─ hallucinated → Refiner → Citation Agent → END
```

### Key Pipeline Nodes

| Node | Role |
|---|---|
| **Router** | LLM-based query classifier. Routes to `graph`, `vector`, `both`, or `real_time`. |
| **Graph Retriever** | Translates natural language to Cypher via `GraphCypherQAChain`; queries Neo4j AuraDB. Falls back to vector if empty. |
| **Vector Retriever** | Hybrid ensemble retriever: Qdrant dense search (score threshold 0.8) + BM25 (70/30 weights), reranked by FlashRank. |
| **Evidence Gate** | Pure-Python quality gate (no LLM). Triggers web search if both retrievers return empty, or query is flagged `real_time`. |
| **Web Search** | Live web fallback when local evidence is insufficient. |
| **Synthesizer** | Merges all evidence sources into a coherent clinical answer. |
| **Hallucination Grader** | Checks every claim in the answer against the full evidence (vector + graph + web). Routes to Refiner if hallucinated. |
| **Refiner** | Corrects hallucinated outputs; feeds back into Citation Agent. |
| **Citation Agent** | Annotates the final answer with numbered superscript references. Builds a traceable `References` section linking each claim to its source (guideline chapter, graph node, or web result). |

---

## Data Ingestion

### Vector Store — Clinical Guidelines (Qdrant Cloud)

**Source documents:** Clinical T2D nursing guidelines and ADA standards in PDF format.

**Ingestion pipeline (`1_vectordb_ingestion/`):**
1. **Parsing** — PDFs are parsed with **LlamaParse** (layout agent mode) to preserve tables, multi-column text, and section hierarchy, outputting structured Markdown.
2. **Two-stage splitting** — `MarkdownHeaderTextSplitter` first splits by document structure (`#`, `##`, `###`), then `RecursiveCharacterTextSplitter` produces 1000-character chunks (100-char overlap). Each chunk inherits hierarchical metadata (chapter, section, page number, source file).
3. **Embedding & storage** — Seven embedding models were benchmarked across separate **Qdrant Cloud** collections to determine the best retrieval performance for medical queries.

**Embedding models evaluated:**

| Model | Type | Dimension |
|---|---|---|
| `pritamdeka/MedEmbed-base-v0.1` | Medical | 768 |
| `NeuML/pubmedbert-base-embeddings` | Medical | 768 |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | Biomedical | 768 |
| `emilyalsentzer/Bio_ClinicalBERT` | Clinical | 768 |
| `BAAI/bge-m3` | General | 1024 |
| `intfloat/e5-large-v2` | General | 1024 |
| `intfloat/e5-base-v2` | General | 768 |

---

### Knowledge Graph — DrugBank Pharmaceutical Data (Neo4j AuraDB)

**Source:** DrugBank full XML export, filtered to the **ATC A10** class (antidiabetics).

**Ingestion pipeline (`3_drugbank_graphdb_ingestion/`):**
1. **Parsing** — `DrugBankDiabetesParser` walks the DrugBank XML using namespace-aware XPath, filtering drugs strictly by ATC code prefix `A10`.
2. **Graph construction** — Extracted data is serialised to `graph_data.json` (~11 MB) and batch-loaded into Neo4j AuraDB via `aura_ingestion2.py`.

**Graph schema:**

| Node | Key Properties |
|---|---|
| `(:Drug)` | `drugbank_id`, `name`, `description`, `indication`, `mechanism_of_action`, `toxicity`, `half_life`, `clearance`, `available_dosages` |
| `(:Target)` | `name` (biological/molecular target) |
| `(:Category)` | `name` (e.g. Sulfonylureas, Biguanides) |
| `(:ATC)` | `code` (WHO ATC code, e.g. `A10BA02`) |
| `(:Product)` | `brand_name`, `labeller`, `country` |
| `(:FoodInteraction)` | `description` (dietary rules) |

| Relationship | Edge Property |
|---|---|
| `(:Drug)-[:ACTS_ON]→(:Target)` | `action` (e.g. agonist, inhibitor) |
| `(:Drug)-[:BELONGS_TO]→(:Category)` | — |
| `(:Drug)-[:HAS_ATC_CODE]→(:ATC)` | — |
| `(:Drug)-[:MARKETED_AS]→(:Product)` | — |
| `(:Drug)-[:HAS_DIETARY_RULE]→(:FoodInteraction)` | — |
| `(:Drug)-[:INTERACTS_WITH]→(:Drug)` | `description` (clinical warning) |

---

## Databases

| Database | Role | Hosted |
|---|---|---|
| **Qdrant Cloud** | Dense vector store for clinical guideline chunks | Cloud (managed) |
| **Neo4j AuraDB** | Property graph for pharmaceutical knowledge | Cloud (managed) |

---

## Evaluation

Evaluation is tracked via **LangSmith** (`4_Evaluation/`), using an LLM-as-Judge pattern with GPT-4o-mini scoring on a 0–5 scale (normalised to 0–1).

### Metrics

| Metric | Description |
|---|---|
| **Answer Correctness** | Factual accuracy of the system answer vs. ground truth |
| **Faithfulness** | Degree to which every claim is grounded in retrieved evidence (hallucination proxy) |
| **Completeness** | Coverage of all key clinical facts from the ground truth |
| **Router Accuracy** | Exact-match check on whether the router chose the expected retrieval strategy |
| **Cypher Semantic Correctness** | LLM judge comparing generated Cypher to expected Cypher by intent, not syntax |
| **Context Recall** | Whether graph-retrieved facts contain the ground truth clinical information |
| **E2E Quality** | End-to-end answer quality scored 1–5 against golden answers (normalised) |

Latency is tracked automatically by LangSmith. Evaluations were run across four configurations: `vector_only`, `graph_only`, `fixed_hybrid`, and `adaptive_router`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (StateGraph) |
| LLM Providers | OpenAI, Google Gemini, Groq |
| Vector Retrieval | Qdrant Cloud + BM25 (EnsembleRetriever) + FlashRank reranking |
| Graph Retrieval | Neo4j AuraDB + LangChain `GraphCypherQAChain` |
| PDF Parsing | LlamaParse (layout agent) |
| Evaluation | LangSmith + custom LLM-as-Judge evaluators |
| API Layer | FastAPI + Uvicorn |
| Graph Data Source | DrugBank XML (ATC A10 class) |
