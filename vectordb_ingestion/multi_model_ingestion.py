"""
Multi-Model Vector Database Ingestion Script
Loads all processed chunks and ingests them into Qdrant using multiple embedding models
for retrieval performance comparison.
"""

import json
import os
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Configuration
PROCESSED_CHUNKS_DIR = Path(__file__).parent / "processed_chunks"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # Required for Qdrant Cloud
DEVICE = "cpu"  # Change to "cuda" if you have a GPU

# Embedding models to test
EMBEDDING_MODELS = {
    "MedEmbed-base-v0.1": "pritamdeka/MedEmbed-base-v0.1",
    "pubmedbert-base-embeddings": "NeuML/pubmedbert-base-embeddings",
    "bge-m3": "BAAI/bge-m3",
    "Bio_ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
    "BiomedNLP-PubMedBERT": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "e5-large-v2": "intfloat/e5-large-v2",
    "e5-base-v2": "intfloat/e5-base-v2",
}


def load_all_chunks(chunks_dir: Path) -> List[Document]:
    """
    Load all JSON chunk files from the processed_chunks directory
    and convert them to LangChain Document objects.
    
    Args:
        chunks_dir: Path to the directory containing JSON chunk files
        
    Returns:
        List of LangChain Document objects
    """
    all_documents = []
    json_files = list(chunks_dir.glob("*.json"))
    
    print(f"\n📂 Found {len(json_files)} JSON files to process")
    
    for json_file in tqdm(json_files, desc="Loading chunks"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Convert dictionaries to LangChain Document objects
            documents = [
                Document(
                    page_content=chunk["page_content"],
                    metadata=chunk["metadata"]
                )
                for chunk in data
            ]
            
            all_documents.extend(documents)
            print(f"  ✓ Loaded {len(documents)} chunks from {json_file.name}")
            
        except Exception as e:
            print(f"  ✗ Error loading {json_file.name}: {e}")
            continue
    
    print(f"\n✅ Total documents loaded: {len(all_documents)}")
    return all_documents


def create_embeddings(model_name: str, device: str = "cpu") -> HuggingFaceEmbeddings:
    """
    Create HuggingFace embeddings for a given model.
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to run the model on ('cpu' or 'cuda')
        
    Returns:
        HuggingFaceEmbeddings instance
    """
    print(f"  🔧 Initializing embeddings: {model_name}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity search
    )


def ingest_to_qdrant(
    documents: List[Document],
    embeddings: HuggingFaceEmbeddings,
    collection_name: str,
    qdrant_url: str = QDRANT_URL,
    api_key: str = None,
    force_recreate: bool = False
) -> QdrantVectorStore:
    """
    Ingest documents into Qdrant with the specified embeddings.
    
    Args:
        documents: List of LangChain Document objects
        embeddings: HuggingFaceEmbeddings instance
        collection_name: Name of the Qdrant collection
        qdrant_url: URL of the Qdrant instance
        api_key: API key for Qdrant Cloud (optional for local)
        force_recreate: Whether to recreate the collection if it exists
        
    Returns:
        QdrantVectorStore instance
    """
    print(f"Ingesting {len(documents)} documents into collection: {collection_name}")
    
    try:
        # Prepare connection parameters
        connection_params = {
            "documents": documents,
            "embedding": embeddings,
            "url": qdrant_url,
            "collection_name": collection_name,
            "force_recreate": force_recreate
        }
        
        # Add API key if provided (for Qdrant Cloud)
        if api_key:
            connection_params["api_key"] = api_key
        
        vector_store = QdrantVectorStore.from_documents(**connection_params)
        print(f"  ✅ Successfully created collection: {collection_name}")
        return vector_store
        
    except Exception as e:
        print(f"  ✗ Error ingesting to {collection_name}: {e}")
        raise


def main():
    """
    Main function to orchestrate the multi-model ingestion process.
    """
    print("🚀 Multi-Model Vector Database Ingestion")
    
    # Step 1: Load all documents
    print("\n📖 Step 1: Loading all processed chunks...")
    all_documents = load_all_chunks(PROCESSED_CHUNKS_DIR)
    
    if not all_documents:
        print("No documents found. Exiting.")
        return
    
    # Step 2: Process each embedding model
    print(f"\n🔄 Step 2: Processing {len(EMBEDDING_MODELS)} embedding models...")
    
    results = {}
    
    for model_key, model_path in EMBEDDING_MODELS.items():
        print(f"\n{'=' * 80}")
        print(f"Processing Model: {model_key}")
        print(f"{'=' * 80}")
        
        try:
            # Create embeddings
            embeddings = create_embeddings(model_path, device=DEVICE)
            
            # Create collection name (sanitize for Qdrant)
            collection_name = f"ada_model_{model_key.lower().replace('-', '_')}"
            
            # Ingest to Qdrant
            vector_store = ingest_to_qdrant(
                documents=all_documents,
                embeddings=embeddings,
                collection_name=collection_name,
                api_key=QDRANT_API_KEY,
                force_recreate=False  # Set to True to overwrite existing collections
            )
            
            results[model_key] = {
                "collection_name": collection_name,
                "status": "success",
                "document_count": len(all_documents)
            }
            
        except Exception as e:
            print(f"Failed to process {model_key}: {e}")
            results[model_key] = {
                "collection_name": f"ada_clinical_{model_key.lower().replace('-', '_')}",
                "status": "failed",
                "error": str(e)
            }
            continue
    
    # Step 3: Summary
    print("\n" + "=" * 80)
    print("📋 INGESTION SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for r in results.values() if r["status"] == "success")
    failed = sum(1 for r in results.values() if r["status"] == "failed")
    
    print(f"\n✅ Successful: {successful}/{len(EMBEDDING_MODELS)}")
    print(f"❌ Failed: {failed}/{len(EMBEDDING_MODELS)}")
    
    print("\n📊 Collection Details:")
    for model_key, result in results.items():
        status_icon = "✅" if result["status"] == "success" else "❌"
        print(f"  {status_icon} {model_key:30s} → {result['collection_name']}")
        if result["status"] == "failed":
            print(f"      Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 80)
    print("🎉 Ingestion process completed!")
    print("=" * 80)
    
    # Save results to JSON
    results_file = Path(__file__).parent / "ingestion_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {results_file}")


if __name__ == "__main__":
    main()
