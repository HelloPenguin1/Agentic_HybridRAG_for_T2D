# Evaluation Code for Embedding Models
# Add this to your evaluating_embeddings.ipynb

import time
import pandas as pd
from typing import List, Dict
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# Your Qdrant connection details
QDRANT_URL = "https://a8673c43-e709-40d5-b254-4900c45d634f.us-east-1-1.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = ""  # Add your API key

# Collection names (matching your ingestion pattern)
COLLECTION_NAMES = {
    "MedEmbed-base-v0.1": "ada_model_medembed_base_v0.1",
    "pubmedbert-base-embeddings": "ada_model_pubmedbert_base_embeddings",
    "bge-m3": "ada_model_bge_m3",
    "Bio_ClinicalBERT": "ada_model_bio_clinicalbert",
    "e5-large-v2": "ada_model_e5_large_v2",
    "e5-base-v2": "ada_model_e5_base_v2",
    "all-mini": "ada_model_all_mini"
}

# Embedding model paths (same as ingestion)
EMBEDDING_MODELS = {
    "MedEmbed-base-v0.1": "abhinand/MedEmbed-base-v0.1",
    "pubmedbert-base-embeddings": "NeuML/pubmedbert-base-embeddings",
    "bge-m3": "BAAI/bge-m3",
    "Bio_ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
    "e5-large-v2": "intfloat/e5-large-v2",
    "e5-base-v2": "intfloat/e5-base-v2",
    "all-mini": "sentence-transformers/all-MiniLM-L6-v2"
}


def text_similarity(text1: str, text2: str, threshold: float = 0.8) -> bool:
    """
    Check if two texts are similar (simple substring matching).
    For more robust matching, consider using fuzzy matching libraries.
    """
    text1_clean = text1.strip().lower()
    text2_clean = text2.strip().lower()
    
    # Check if either text contains the other (with some tolerance)
    if text1_clean in text2_clean or text2_clean in text1_clean:
        return True
    
    # Check for significant overlap
    words1 = set(text1_clean.split())
    words2 = set(text2_clean.split())
    if len(words1) > 0 and len(words2) > 0:
        overlap = len(words1.intersection(words2)) / min(len(words1), len(words2))
        return overlap >= threshold
    
    return False


def evaluate_single_model(
    model_name: str,
    collection_name: str,
    eval_dataset: List[Dict],
    k: int = 10
) -> Dict:
    """
    Evaluate a single embedding model using the test dataset.
    
    Args:
        model_name: Name of the embedding model
        collection_name: Qdrant collection name
        eval_dataset: List of questions with reference_contexts
        k: Number of chunks to retrieve (default 10 for Recall@10)
    
    Returns:
        Dictionary with metrics: Precision@5, MRR, Recall@10, Latency
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Collection: {collection_name}")
    print(f"{'='*60}")
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODELS[model_name],
        model_kwargs={'device': 'cpu'},  # Change to 'cuda' if GPU available
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Connect to Qdrant collection
    vector_store = QdrantVectorStore.from_existing_collection(
        collection_name=collection_name,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
    
    # Metrics storage
    precision_at_5 = []
    mrr_scores = []
    recall_at_10 = []
    latencies = []
    
    # Evaluate each question
    for idx, item in enumerate(eval_dataset, 1):
        question = item['question']
        reference_contexts = item['reference_contexts']
        
        print(f"  [{idx}/{len(eval_dataset)}] Evaluating: {question[:60]}...")
        
        # Retrieve top-k chunks and measure latency
        start_time = time.time()
        retrieved_docs = vector_store.similarity_search(question, k=k)
        latency = time.time() - start_time
        latencies.append(latency)
        
        # Find position of first matching reference context
        found_at_position = None
        for pos, doc in enumerate(retrieved_docs, start=1):
            retrieved_text = doc.page_content
            
            # Check if any reference context matches this retrieved chunk
            for ref_context in reference_contexts:
                if text_similarity(ref_context, retrieved_text):
                    found_at_position = pos
                    break
            
            if found_at_position:
                break
        
        # Calculate metrics for this question
        # Precision@5: Is reference in top 5?
        p5 = 1 if found_at_position and found_at_position <= 5 else 0
        precision_at_5.append(p5)
        
        # MRR: Reciprocal rank of first match
        mrr = 1 / found_at_position if found_at_position else 0
        mrr_scores.append(mrr)
        
        # Recall@10: Is reference in top 10?
        r10 = 1 if found_at_position and found_at_position <= 10 else 0
        recall_at_10.append(r10)
        
        # Log result
        status = f"✓ Found at position {found_at_position}" if found_at_position else "✗ Not found"
        print(f"    {status} | Latency: {latency*1000:.2f}ms")
    
    # Aggregate metrics
    results = {
        'Model': model_name,
        'Precision@5': sum(precision_at_5) / len(precision_at_5),
        'MRR': sum(mrr_scores) / len(mrr_scores),
        'Recall@10': sum(recall_at_10) / len(recall_at_10),
        'Avg_Latency_ms': (sum(latencies) / len(latencies)) * 1000,
        'Total_Questions': len(eval_dataset)
    }
    
    print(f"\n📊 Results for {model_name}:")
    print(f"  Precision@5:  {results['Precision@5']:.3f}")
    print(f"  MRR:          {results['MRR']:.3f}")
    print(f"  Recall@10:    {results['Recall@10']:.3f}")
    print(f"  Avg Latency:  {results['Avg_Latency_ms']:.2f}ms")
    
    return results


def evaluate_all_models(eval_dataset: List[Dict]) -> pd.DataFrame:
    """
    Evaluate all embedding models and return results as DataFrame.
    
    Args:
        eval_dataset: List of questions with reference_contexts
    
    Returns:
        DataFrame with evaluation results for all models
    """
    all_results = []
    
    for model_name, collection_name in COLLECTION_NAMES.items():
        try:
            results = evaluate_single_model(
                model_name=model_name,
                collection_name=collection_name,
                eval_dataset=eval_dataset,
                k=10
            )
            all_results.append(results)
        except Exception as e:
            print(f"\n❌ Error evaluating {model_name}: {e}")
            continue
    
    # Create DataFrame and sort by MRR (or any metric you prefer)
    df = pd.DataFrame(all_results)
    df = df.sort_values('MRR', ascending=False)
    
    return df


# Example usage in your notebook:
"""
# Run evaluation
results_df = evaluate_all_models(eval_dataset)

# Display results
print("\n" + "="*80)
print("FINAL RESULTS - All Embedding Models")
print("="*80)
print(results_df.to_string(index=False))

# Save to CSV
results_df.to_csv('embedding_evaluation_results.csv', index=False)
print("\n✅ Results saved to embedding_evaluation_results.csv")

# Visualize (optional)
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Precision@5
axes[0, 0].barh(results_df['Model'], results_df['Precision@5'])
axes[0, 0].set_xlabel('Precision@5')
axes[0, 0].set_title('Precision@5 by Model')

# MRR
axes[0, 1].barh(results_df['Model'], results_df['MRR'])
axes[0, 1].set_xlabel('MRR')
axes[0, 1].set_title('Mean Reciprocal Rank by Model')

# Recall@10
axes[1, 0].barh(results_df['Model'], results_df['Recall@10'])
axes[1, 0].set_xlabel('Recall@10')
axes[1, 0].set_title('Recall@10 by Model')

# Latency
axes[1, 1].barh(results_df['Model'], results_df['Avg_Latency_ms'])
axes[1, 1].set_xlabel('Latency (ms)')
axes[1, 1].set_title('Average Latency by Model')

plt.tight_layout()
plt.savefig('embedding_evaluation_charts.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Charts saved to embedding_evaluation_charts.png")
"""
