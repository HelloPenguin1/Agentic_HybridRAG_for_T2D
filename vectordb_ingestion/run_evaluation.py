"""
run_evaluation.py
─────────────────
Evaluates all 7 Qdrant embedding collections against eval_queries.json.

Retrieval : LangChain QdrantVectorStore.similarity_search()
Matching  : exact chunk_id  (doc.metadata["chunk_id"] == ground_truth_chunk_id)
Metrics   : Precision@5 | MRR | Recall@10 | Avg Latency (ms)

Run from vectordb_ingestion/:
    python run_evaluation.py
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")

QDRANT_URL     = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
EVAL_PATH      = Path(__file__).parent / "eval_queries.json"
K              = 10   # retrieve top-K docs per query

MODELS = {
    "MedEmbed-base-v0.1":        ("abhinand/MedEmbed-base-v0.1",                "ada_model_medembed_base_v0.1"),
    "pubmedbert-base-embeddings": ("NeuML/pubmedbert-base-embeddings",           "ada_model_pubmedbert_base_embeddings"),
    "bge-m3":                    ("BAAI/bge-m3",                                "ada_model_bge_m3"),
    "Bio_ClinicalBERT":          ("emilyalsentzer/Bio_ClinicalBERT",            "ada_model_bio_clinicalbert"),
    "e5-large-v2":               ("intfloat/e5-large-v2",                       "ada_model_e5_large_v2"),
    "e5-base-v2":                ("intfloat/e5-base-v2",                        "ada_model_e5_base_v2"),
    "all-mini":                  ("sentence-transformers/all-MiniLM-L6-v2",     "ada_model_all_mini"),
}

# ── Load queries ──────────────────────────────────────────────────────────────
with open(EVAL_PATH, "r", encoding="utf-8") as f:
    eval_queries = json.load(f)

print(f"Loaded {len(eval_queries)} queries from {EVAL_PATH.name}\n")

# ── Evaluate ──────────────────────────────────────────────────────────────────
all_results = []

for model_name, (model_path, collection_name) in MODELS.items():
    print(f"Model      : {model_name}")
    print(f"Collection : {collection_name}")

    # Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Connect to Qdrant collection
    vector_store = QdrantVectorStore.from_existing_collection(
        collection_name=collection_name,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    p5_list, mrr_list, r10_list, lat_list = [], [], [], []

    for i, item in enumerate(eval_queries, 1):
        query   = item["query"]
        gt_id   = item["ground_truth_chunk_id"]

        # Retrieve top-K docs
        t0   = time.time()
        docs = vector_store.similarity_search(query, k=K)
        lat  = time.time() - t0
        lat_list.append(lat)

        # Find rank of ground-truth chunk
        rank = None
        for pos, doc in enumerate(docs, start=1):
            if doc.metadata.get("chunk_id") == gt_id:
                rank = pos
                break

        p5_list.append(1 if rank and rank <= 5  else 0)
        mrr_list.append(1 / rank if rank else 0)
        r10_list.append(1 if rank and rank <= 10 else 0)

        status = f"✓ rank {rank}" if rank else "✗ not found"
        print(f"  [{i:02d}] {status:12s} | {lat*1000:6.1f} ms | {query[:55]}...")

    n = len(eval_queries)
    result = {
        "Model":           model_name,
        "Precision@5":     sum(p5_list)  / n,
        "MRR":             sum(mrr_list) / n,
        "Recall@10":       sum(r10_list) / n,
        "Avg_Latency_ms":  sum(lat_list) / n * 1000,
    }
    all_results.append(result)

    print(f"\n  Precision@5 : {result['Precision@5']:.3f}")
    print(f"  MRR         : {result['MRR']:.3f}")
    print(f"  Recall@10   : {result['Recall@10']:.3f}")
    print(f"  Avg Latency : {result['Avg_Latency_ms']:.1f} ms")

# ── Summary table ─────────────────────────────────────────────────────────────
df = pd.DataFrame(all_results).sort_values("MRR", ascending=False).reset_index(drop=True)

print("FINAL RESULTS — sorted by MRR")
print(df.to_string(index=False))

df.to_csv(Path(__file__).parent / "embedding_evaluation_results.csv", index=False)
print("\nResults saved to embedding_evaluation_results.csv")
