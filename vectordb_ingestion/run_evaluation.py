"""
run_evaluation.py  ←  Colab-optimized
═══════════════════════════════════════
Copy each CELL block into a separate Colab cell and run top to bottom.

FILES TO UPLOAD TO COLAB (Files panel → Upload):
  • eval_queries_multi_chunk.json   → /content/eval_queries_multi_chunk.json

PACKAGES (run Cell 1 first):
  !pip install langchain-huggingface langchain-qdrant sentence-transformers numpy pandas
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 1 — Install dependencies                              ║
# ╚══════════════════════════════════════════════════════════════╝
"""
!pip install -q langchain-huggingface langchain-qdrant sentence-transformers numpy pandas
"""


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 2 — Config & imports                                  ║
# ╚══════════════════════════════════════════════════════════════╝

import json, os, time
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# ── Credentials ───────────────────────────────────────────────
QDRANT_URL     = "https://a8673c43-e709-40d5-b254-4900c45d634f.us-east-1-1.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.TJKHmQcBA44EA14l01N5OWkOhaBKdSCw7Xpa_mnH7ls"

# ── Eval dataset (upload this file to Colab) ──────────────────
EVAL_PATH = Path("/content/eval_queries_multi_chunk.json")
K = 10   # retrieve top-K docs per query

# ── Models: name → (HuggingFace model path, Qdrant collection) ─
MODELS = {
    "bge-m3":           ("BAAI/bge-m3",                             "ada_model_bge_m3"),
    "e5-large-v2":      ("intfloat/e5-large-v2",                    "ada_model_e5_large_v2"),
    "MedEmbed":         ("abhinand/MedEmbed-base-v0.1",             "ada_model_medembed_base_v0.1"),
    "e5-base-v2":       ("intfloat/e5-base-v2",                     "ada_model_e5_base_v2"),
    "pubmedbert":       ("NeuML/pubmedbert-base-embeddings",        "ada_model_pubmedbert_base_embeddings"),
    "all-mini":         ("sentence-transformers/all-MiniLM-L6-v2",  "ada_model_all_mini"),
    "Bio_ClinicalBERT": ("emilyalsentzer/Bio_ClinicalBERT",         "ada_model_bio_clinicalbert"),
}

# ── Load eval queries ─────────────────────────────────────────
with open(EVAL_PATH) as f:
    eval_queries = json.load(f)

print(f"✓ Loaded {len(eval_queries)} queries from {EVAL_PATH.name}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 3 — Metric functions                                  ║
# ╚══════════════════════════════════════════════════════════════╝

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

print("✓ Metric functions defined")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 4 — Run evaluation across all models                  ║
# ╚══════════════════════════════════════════════════════════════╝

all_results = []

for model_name, (model_path, collection_name) in MODELS.items():
    print(f"\n{'='*60}")
    print(f"Model      : {model_name}")
    print(f"Collection : {collection_name}")
    print(f"{'='*60}")

    # Load embedding model (downloads on first run, cached after)
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Connect to Qdrant
    vector_store = QdrantVectorStore.from_existing_collection(
        collection_name=collection_name,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    p5_list, r10_list, mrr_list, ndcg5_list, hit5_list, lat_list = [], [], [], [], [], []

    for i, item in enumerate(eval_queries, 1):
        query          = item["query"]
        relevant_ids   = item["ground_truth"]["relevant_chunk_ids"]
        rel_scores     = item["ground_truth"]["relevance_scores"]

        # Retrieve
        t0   = time.time()
        docs = vector_store.similarity_search(query, k=K)
        lat  = time.time() - t0

        retrieved_ids = [doc.metadata.get("chunk_id") for doc in docs]

        p5    = precision_at_k(retrieved_ids, relevant_ids, k=5)
        r10   = recall_at_k(retrieved_ids, relevant_ids, k=10)
        mrr_s = mrr(retrieved_ids, relevant_ids)
        ndcg5 = ndcg_at_k(retrieved_ids, rel_scores, k=5)
        hit5  = hit_at_k(retrieved_ids, relevant_ids, k=5)

        p5_list.append(p5);  r10_list.append(r10);  mrr_list.append(mrr_s)
        ndcg5_list.append(ndcg5);  hit5_list.append(hit5);  lat_list.append(lat)

        print(f"  [{i:02d}] P@5={p5:.2f} NDCG@5={ndcg5:.2f} Hit@5={int(hit5)} | {lat*1000:.0f}ms | {query[:55]}...")

    result = {
        "Model":        model_name,
        "Precision@5":  round(float(np.mean(p5_list)),   3),
        "Recall@10":    round(float(np.mean(r10_list)),  3),
        "MRR":          round(float(np.mean(mrr_list)),  3),
        "NDCG@5":       round(float(np.mean(ndcg5_list)),3),
        "Hit@5":        round(float(np.mean(hit5_list)), 3),
        "Latency_ms":   round(float(np.mean(lat_list)) * 1000, 1),
    }
    all_results.append(result)

    print(f"\n  Precision@5 : {result['Precision@5']}")
    print(f"  Recall@10   : {result['Recall@10']}")
    print(f"  MRR         : {result['MRR']}")
    print(f"  NDCG@5      : {result['NDCG@5']}")
    print(f"  Hit@5       : {result['Hit@5']}")
    print(f"  Latency     : {result['Latency_ms']} ms")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 5 — Summary table + save CSV                         ║
# ╚══════════════════════════════════════════════════════════════╝

df = (pd.DataFrame(all_results)
        .sort_values("NDCG@5", ascending=False)
        .reset_index(drop=True))

print("\n" + "="*70)
print("FINAL RESULTS — sorted by NDCG@5")
print("="*70)
print(df.to_string(index=False))

csv_path = "/content/embedding_evaluation_results.csv"
df.to_csv(csv_path, index=False)
print(f"\n✅ Saved to {csv_path}  (download from Files panel)")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 6 — Bar chart visualization (optional)               ║
# ╚══════════════════════════════════════════════════════════════╝

import matplotlib.pyplot as plt

metrics = ["Precision@5", "Recall@10", "MRR", "NDCG@5", "Hit@5"]
colors  = ["steelblue", "seagreen", "darkorange", "mediumpurple", "tomato"]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for ax, metric, color in zip(axes, metrics, colors):
    ax.barh(df["Model"], df[metric], color=color)
    ax.set_xlabel(metric)
    ax.set_title(metric)
    ax.invert_yaxis()
    for i, v in enumerate(df[metric]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)

axes[-1].axis("off")   # hide unused 6th panel
plt.suptitle("Embedding Model Evaluation", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("/content/embedding_evaluation_charts.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Chart saved — download from Files panel")