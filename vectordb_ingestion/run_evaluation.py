"""
run_evaluation.py  ←  Colab-optimized
══════════════════════════════════════════════════════════
Experiment 1: MedEmbed only (no reranker)                       ← hardcoded baseline
Experiment 2: MedEmbed + bge-reranker-base                      ← hardcoded baseline
Experiment 3: MedEmbed + BM25 EnsembleRetriever + bge-reranker-base   ← runs live
Experiment 4: MedEmbed + BM25 RRF + bge-reranker-base (new)     ← runs live

FILES TO UPLOAD TO COLAB (Files panel → Upload):
  • eval_questions.json       → /content/eval_questions.json
  • processed_chunks.json    → /content/processed_chunks.json

Copy each CELL block into a separate Colab cell and run top to bottom.
"""


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 1 — Install dependencies                              ║
# ╚══════════════════════════════════════════════════════════════╝
"""
!pip install -q langchain-huggingface langchain-qdrant langchain-community sentence-transformers rank_bm25 nltk numpy pandas matplotlib
!python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
"""


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 2 — Config & imports                                  ║
# ╚══════════════════════════════════════════════════════════════╝

# Core imports and pipeline configuration
import json, time
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from nltk.tokenize import word_tokenize

QDRANT_URL     = "https://a8673c43-e709-40d5-b254-4900c45d634f.us-east-1-1.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.TJKHmQcBA44EA14l01N5OWkOhaBKdSCw7Xpa_mnH7ls"

EMBED_MODEL     = "abhinand/MedEmbed-base-v0.1"
COLLECTION_NAME = "ada_model_medembed_base_v0.1"
RERANKER_MODEL  = "BAAI/bge-reranker-base"

QDRANT_TOP = 10   # semantic candidates from Qdrant
BM25_TOP   = 10   # keyword candidates from BM25
MERGED_CAP = 20   # reranker sees same load as Exp 2
RERANK_TOP = 5    # final docs after reranking

EVAL_PATH = Path("/content/eval_queries_multi_chunk.json")
CHUNKS_PATH = Path("/content/processed_chunks")

with open(EVAL_PATH) as f:
    eval_queries = json.load(f)
with open(CHUNKS_PATH) as f:
    processed_chunks = json.load(f)

print(f"✓ Loaded {len(eval_queries)} queries")
print(f"✓ Loaded {len(processed_chunks)} chunks for BM25 index")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 3 — Metric functions                                  ║
# ╚══════════════════════════════════════════════════════════════╝

# Standard IR metrics used across all three experiments
def precision_at_k(retrieved: list, relevant: list, k: int = 5) -> float:
    return len(set(retrieved[:k]) & set(relevant)) / k

def recall_at_k(retrieved: list, relevant: list, k: int = 10) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)

def mrr(retrieved: list, relevant: list) -> float:
    rel_set = set(relevant)
    for i, cid in enumerate(retrieved, 1):
        if cid in rel_set:
            return 1.0 / i
    return 0.0

def ndcg_at_k(retrieved: list, scores: dict, k: int = 5) -> float:
    dcg  = sum(scores.get(cid, 0) / np.log2(i + 1)
               for i, cid in enumerate(retrieved[:k], 1))
    idcg = sum(s / np.log2(i + 1)
               for i, s in enumerate(sorted(scores.values(), reverse=True)[:k], 1))
    return dcg / idcg if idcg > 0 else 0.0

def hit_at_k(retrieved: list, relevant: list, k: int = 5) -> float:
    return 1.0 if set(retrieved[:k]) & set(relevant) else 0.0


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 4 — Build Qdrant retriever and BM25 index             ║
# ╚══════════════════════════════════════════════════════════════╝

# Load MedEmbed + connect to existing Qdrant collection (no re-embedding)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vector_store = QdrantVectorStore.from_existing_collection(
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)
qdrant_retriever = vector_store.as_retriever(search_kwargs={"k": QDRANT_TOP})

# Build BM25 index from processed_chunks.json using word_tokenize
bm25_docs = [
    Document(
        page_content=chunk.get("page_content", ""),
        metadata={"chunk_id": chunk["metadata"]["chunk_id"]}
    )
    for i, chunk in enumerate(processed_chunks)
]
bm25_retriever = BM25Retriever.from_documents(
    bm25_docs,
    k=BM25_TOP,
    preprocess_func=word_tokenize,
)

# RRF ensemble: equal weights give each retriever 0.5 vote in rank fusion
rrf_retriever = EnsembleRetriever(
    retrievers=[qdrant_retriever, bm25_retriever],
    weights=[0.5, 0.5],
)


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 5 — Load bge-reranker-base                            ║
# ╚══════════════════════════════════════════════════════════════╝

# Single shared CrossEncoder for Experiments 3 and 4
cross_encoder = CrossEncoder(RERANKER_MODEL)
print(f"✓ Reranker loaded: {RERANKER_MODEL}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 6 — EnsembleRetriever eval loop (Experiment 3)        ║
# ╚══════════════════════════════════════════════════════════════╝

def evaluate_hybrid() -> dict:
    """Run MedEmbed + BM25 via EnsembleRetriever → bge-reranker-base → top-5."""
    p5_list, r10_list, mrr_list, ndcg5_list, hit5_list, lat_list = [], [], [], [], [], []

    for i, item in enumerate(eval_queries, 1):
        query        = item["query"]
        relevant_ids = item["ground_truth"]["relevant_chunk_ids"]
        rel_scores   = item["ground_truth"]["relevance_scores"]

        t0 = time.time()

        # EnsembleRetriever fuses Qdrant + BM25 via RRF and deduplicates internally
        merged = rrf_retriever.invoke(query)[:MERGED_CAP]

        # Rerank candidates with bge-reranker-base → top-5
        passages  = [doc.page_content for doc in merged]
        ce_inputs = [[query, p] for p in passages]
        ce_scores = cross_encoder.predict(ce_inputs)

        ranked_docs  = sorted(zip(ce_scores, merged), key=lambda x: x[0], reverse=True)
        top_docs     = [doc for _, doc in ranked_docs[:RERANK_TOP]]
        lat          = time.time() - t0

        # Extract chunk IDs for metric computation
        retrieved_ids = [doc.metadata.get("chunk_id") for doc in top_docs]
        all_reranked  = [doc.metadata.get("chunk_id") for _, doc in ranked_docs]

        p5    = precision_at_k(retrieved_ids, relevant_ids, k=5)
        r10   = recall_at_k(all_reranked,     relevant_ids, k=10)
        mrr_s = mrr(retrieved_ids,             relevant_ids)
        ndcg5 = ndcg_at_k(retrieved_ids,       rel_scores,  k=5)
        hit5  = hit_at_k(retrieved_ids,         relevant_ids, k=5)

        p5_list.append(p5);  r10_list.append(r10);  mrr_list.append(mrr_s)
        ndcg5_list.append(ndcg5);  hit5_list.append(hit5);  lat_list.append(lat)

        print(f"  [{i:02d}] P@5={p5:.2f} NDCG@5={ndcg5:.2f} Hit@5={int(hit5)} "
              f"candidates={len(merged):02d} | {lat*1000:.0f}ms | {query[:50]}…")

    return {
        "Pipeline":    "MedEmbed + BM25 EnsembleRetriever + bge-reranker-base",
        "Precision@5": round(float(np.mean(p5_list)),   3),
        "Recall@10":   round(float(np.mean(r10_list)),  3),
        "MRR":         round(float(np.mean(mrr_list)),  3),
        "NDCG@5":      round(float(np.mean(ndcg5_list)),3),
        "Hit@5":       round(float(np.mean(hit5_list)), 3),
        "Latency_ms":  round(float(np.mean(lat_list)) * 1000, 1),
    }

exp3_result = evaluate_hybrid()
print(f"\n  Precision@5 : {exp3_result['Precision@5']}")
print(f"  Recall@10   : {exp3_result['Recall@10']}")
print(f"  MRR         : {exp3_result['MRR']}")
print(f"  NDCG@5      : {exp3_result['NDCG@5']}")
print(f"  Hit@5       : {exp3_result['Hit@5']}")
print(f"  Latency     : {exp3_result['Latency_ms']} ms")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 7 — RRF fusion eval loop (Experiment 4)               ║
# ╚══════════════════════════════════════════════════════════════╝

def evaluate_rrf() -> dict:
    """Run MedEmbed + BM25 via RRF → bge-reranker-base → top-5."""
    p5_list, r10_list, mrr_list, ndcg5_list, hit5_list, lat_list = [], [], [], [], [], []

    for i, item in enumerate(eval_queries, 1):
        query        = item["query"]
        relevant_ids = item["ground_truth"]["relevant_chunk_ids"]
        rel_scores   = item["ground_truth"]["relevance_scores"]

        t0 = time.time()

        # EnsembleRetriever applies RRF internally and returns up to MERGED_CAP docs
        rrf_docs = rrf_retriever.invoke(query)[:MERGED_CAP]

        # Rerank RRF candidates with bge-reranker-base → top-5
        passages  = [doc.page_content for doc in rrf_docs]
        ce_inputs = [[query, p] for p in passages]
        ce_scores = cross_encoder.predict(ce_inputs)

        ranked_docs  = sorted(zip(ce_scores, rrf_docs), key=lambda x: x[0], reverse=True)
        top_docs     = [doc for _, doc in ranked_docs[:RERANK_TOP]]
        lat          = time.time() - t0

        # Extract chunk IDs for metric computation
        retrieved_ids = [doc.metadata.get("chunk_id") for doc in top_docs]
        all_reranked  = [doc.metadata.get("chunk_id") for _, doc in ranked_docs]

        p5    = precision_at_k(retrieved_ids, relevant_ids, k=5)
        r10   = recall_at_k(all_reranked,     relevant_ids, k=10)
        mrr_s = mrr(retrieved_ids,             relevant_ids)
        ndcg5 = ndcg_at_k(retrieved_ids,       rel_scores,  k=5)
        hit5  = hit_at_k(retrieved_ids,         relevant_ids, k=5)

        p5_list.append(p5);  r10_list.append(r10);  mrr_list.append(mrr_s)
        ndcg5_list.append(ndcg5);  hit5_list.append(hit5);  lat_list.append(lat)

        print(f"  [{i:02d}] P@5={p5:.2f} NDCG@5={ndcg5:.2f} Hit@5={int(hit5)} "
              f"candidates={len(rrf_docs):02d} | {lat*1000:.0f}ms | {query[:50]}…")

    return {
        "Pipeline":    "MedEmbed + BM25 RRF + bge-reranker-base",
        "Precision@5": round(float(np.mean(p5_list)),   3),
        "Recall@10":   round(float(np.mean(r10_list)),  3),
        "MRR":         round(float(np.mean(mrr_list)),  3),
        "NDCG@5":      round(float(np.mean(ndcg5_list)),3),
        "Hit@5":       round(float(np.mean(hit5_list)), 3),
        "Latency_ms":  round(float(np.mean(lat_list)) * 1000, 1),
    }

exp4_result = evaluate_rrf()
print(f"\n  Precision@5 : {exp4_result['Precision@5']}")
print(f"  Recall@10   : {exp4_result['Recall@10']}")
print(f"  MRR         : {exp4_result['MRR']}")
print(f"  NDCG@5      : {exp4_result['NDCG@5']}")
print(f"  Hit@5       : {exp4_result['Hit@5']}")
print(f"  Latency     : {exp4_result['Latency_ms']} ms")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 8 — Combined results table (all 4 experiments)        ║
# ╚══════════════════════════════════════════════════════════════╝

# Hardcoded Exp1/Exp2 baselines — update from your saved CSVs if values differ
exp1_baseline = {
    "Pipeline":    "MedEmbed only (no reranker)",
    "Precision@5": 0.393,
    "Recall@10":   0.562,
    "MRR":         0.797,
    "NDCG@5":      0.511,
    "Hit@5":       0.867,
    "Latency_ms":  249.8,
}
exp2_baseline = {
    "Pipeline":    "MedEmbed + bge-reranker-base",
    "Precision@5": 0.0,   # ← fill in from reranker_evaluation_results.csv
    "Recall@10":   0.0,
    "MRR":         0.0,
    "NDCG@5":      0.0,
    "Hit@5":       0.0,
    "Latency_ms":  0.0,
}

df = pd.DataFrame([exp1_baseline, exp2_baseline, exp3_result, exp4_result])

print("\n" + "="*80)
print("RETRIEVAL EVALUATION — Experiments 1 / 2 / 3 / 4")
print(f"Qdrant top-{QDRANT_TOP} + BM25 top-{BM25_TOP} → bge-reranker-base → top-{RERANK_TOP}")
print("="*80)
print(df.to_string(index=False))

csv_path = "/content/hybrid_evaluation_results.csv"
df.to_csv(csv_path, index=False)
print(f"\n✅ Saved to {csv_path}  (download from Files panel)")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 9 — Bar chart visualization                           ║
# ╚══════════════════════════════════════════════════════════════╝

# Visualize all metrics across all four pipeline configurations
import matplotlib.pyplot as plt

metrics = ["Precision@5", "Recall@10", "MRR", "NDCG@5", "Hit@5"]
colors  = ["steelblue", "seagreen", "darkorange", "mediumpurple", "tomato"]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for ax, metric, color in zip(axes, metrics, colors):
    ax.barh(df["Pipeline"], df[metric], color=color)
    ax.set_xlabel(metric)
    ax.set_title(metric)
    ax.invert_yaxis()
    for j, v in enumerate(df[metric]):
        ax.text(v + 0.005, j, f"{v:.3f}", va="center", fontsize=8)

axes[-1].axis("off")
plt.suptitle(
    f"Retrieval Evaluation Exp 1–4  (Qdrant {QDRANT_TOP} + BM25 {BM25_TOP} → rerank → top-{RERANK_TOP})",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig("/content/hybrid_evaluation_charts.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Chart saved — download from Files panel")