# Multi-Model Vector Database Ingestion

This directory contains scripts for ingesting processed medical document chunks into Qdrant Cloud using multiple embedding models for retrieval performance comparison.

## Overview

The ingestion pipeline loads all processed JSON chunks and creates separate Qdrant collections for each embedding model, enabling systematic comparison of retrieval metrics across different medical-domain and general-purpose embeddings.

## Setup

### 1. Install Dependencies

```powershell
cd c:\dev\Diabetes_AgenticGraphRAG\vectordb_ingestion
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Ensure your `.env` file in the project root contains:

```bash
QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your-api-key-here
```

### 3. Verify Qdrant Cloud Connection

Test your connection:

```powershell
python -c "from dotenv import load_dotenv; import os; from qdrant_client import QdrantClient; load_dotenv(); client = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY')); print('✅ Connected:', client.get_collections())"
```

## Usage

### Run Full Ingestion

Ingest all chunks into 7 separate collections:

```powershell
python multi_model_ingestion.py
```

This will:
- Load all 16 JSON files from `processed_chunks/`
- Create 7 Qdrant collections (one per embedding model)
- Display progress with detailed logging
- Save results to `ingestion_results.json`

**Expected Runtime**: 15-30 minutes depending on CPU/GPU

### Test Retrieval

Verify collections and test sample queries:

```powershell
python test_retrieval.py
```

This will:
- List all collections in Qdrant Cloud
- Run a test query against each collection
- Display top 3 results per model

## Embedding Models

The following 7 embedding models are used:

| Model Key | HuggingFace Path | Type | Dimension |
|-----------|------------------|------|-----------|
| `MedEmbed-base-v0.1` | `pritamdeka/MedEmbed-base-v0.1` | Medical | 768 |
| `pubmedbert-base-embeddings` | `NeuML/pubmedbert-base-embeddings` | Medical | 768 |
| `bge-m3` | `BAAI/bge-m3` | General | 1024 |
| `Bio_ClinicalBERT` | `emilyalsentzer/Bio_ClinicalBERT` | Clinical | 768 |
| `BiomedNLP-PubMedBERT` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | Biomedical | 768 |
| `e5-large-v2` | `intfloat/e5-large-v2` | General | 1024 |
| `e5-base-v2` | `intfloat/e5-base-v2` | General | 768 |

## Collection Names

Collections are created with the naming convention:
```
ada_model_{model_key}
```

For example:
- `ada_clinical_medembed_base_v0_1`
- `ada_clinical_bge_m3`
- `ada_clinical_e5_large_v2`

## Configuration Options

### GPU Acceleration

To use GPU for faster embedding generation, edit `multi_model_ingestion.py`:

```python
DEVICE = "cuda"  # Change from "cpu" to "cuda"
```

### Force Recreate Collections

To overwrite existing collections, edit `multi_model_ingestion.py`:

```python
force_recreate = True  # Change from False to True
```

## Output Files

- **`ingestion_results.json`**: Summary of ingestion status for each model
- **Progress logs**: Displayed in console during execution

## Troubleshooting

### Connection Issues

If you get connection errors:
1. Verify your Qdrant Cloud cluster is running
2. Check that `QDRANT_URL` and `QDRANT_API_KEY` are correct in `.env`
3. Ensure your IP is whitelisted in Qdrant Cloud settings

### Memory Issues

If you run out of memory:
1. Process models one at a time by commenting out others in `EMBEDDING_MODELS` dict
2. Use smaller batch sizes (modify `QdrantVectorStore.from_documents()` parameters)

### Model Download Issues

If model downloads fail:
1. Check your internet connection
2. Verify HuggingFace access (some models may require authentication)
3. Set `HF_TOKEN` in your `.env` file if needed

## Next Steps

After successful ingestion, you can:
1. Run retrieval evaluation metrics (RAGAS, etc.)
2. Compare retrieval performance across models
3. Analyze which embedding model works best for your medical queries
4. Use the best-performing collection for your production RAG system
