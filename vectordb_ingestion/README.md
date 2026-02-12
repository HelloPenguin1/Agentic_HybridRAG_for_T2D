# Medical-Grade PDF Extraction Pipeline

## Overview
Production-ready extraction pipeline for Type 2 Diabetes medical documents implementing a **3-phase approach** for maximum traceability and context preservation.

## The 3-Phase Approach

### **Phase 1: Advanced Extraction (LlamaParse)**
Uses LlamaParse with layout agent to handle:
- ✅ Multi-column layouts (no word salad)
- ✅ Complex tables (preserved as markdown)
- ✅ Hierarchical structure (headers, sections)

### **Phase 2: Structural Split (Two-Stage)**
1. **Stage 1**: Split by markdown headers (chapter, section, subsection)
2. **Stage 2**: Split into LLM-friendly chunks (1000 chars, 100 overlap)

### **Phase 3: Metadata Enrichment**
Each chunk contains:
- `chapter_name` - Main section (e.g., "Clinical Pharmacology")
- `section_heading` - Subsection (e.g., "Adverse Reactions")
- `subsection_heading` - Sub-subsection (if applicable)
- `page_number` - Exact page in PDF (1-indexed)
- `source` - Filename
- `source_path` - Full path

## Why This Approach is "Medical Grade"

| Feature | Benefit |
|---------|---------|
| **Context Preservation** | Chunks from "Contraindications" never mix with "Benefits" |
| **No Column Scrambling** | Layout agent reads Column A fully before Column B |
| **Explainability** | Display: "Source: file.pdf, Page 12, Section: Renal Impairment" |
| **Auditability** | Doctors can verify AI claims by checking exact page |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from vectordb_ingestion.data_loader import MedicalDataLoader

# Initialize
loader = MedicalDataLoader(
    pdf_dir="t2d_extraction_pipeline/data/raw_pdfs",
    output_dir="vectordb_ingestion/processed_chunks",
    chunk_size=1000,
    chunk_overlap=100
)

# Process all PDFs
loader.process_all_pdfs()
```

### Run Example Script

```bash
python vectordb_ingestion/example_usage.py
```

### Inspect Chunks

```python
# Load chunks
chunks = loader.load_chunks_from_disk("processed_chunks/ada_cardio_disease_manag_chunks.json")

# Get statistics
stats = loader.get_chunk_stats(chunks)
print(f"Total chunks: {stats['total_chunks']}")
print(f"Unique pages: {stats['unique_pages']}")
```

## Output Format

Each chunk is a LangChain Document:

```python
Document(
    page_content="## 4.2 Adverse Reactions\n\nCommon adverse reactions include...",
    metadata={
        "chapter_name": "Chapter 4: Clinical Pharmacology",
        "section_heading": "4.2 Adverse Reactions",
        "page_number": 42,
        "source": "drug_label_v2.pdf",
        "source_path": "C:\\path\\to\\drug_label_v2.pdf"
    }
)
```

Saved as JSON:

```json
[
  {
    "page_content": "## 4.2 Adverse Reactions\n\nCommon adverse reactions include...",
    "metadata": {
      "chapter_name": "Chapter 4: Clinical Pharmacology",
      "section_heading": "4.2 Adverse Reactions",
      "page_number": 42,
      "source": "drug_label_v2.pdf",
      "source_path": "C:\\path\\to\\drug_label_v2.pdf"
    }
  }
]
```

## Configuration

### Chunk Size

Adjust based on your embedding model's context window:

```python
loader = MedicalDataLoader(
    pdf_dir="...",
    output_dir="...",
    chunk_size=1500,  # Larger chunks
    chunk_overlap=150
)
```

### Parsing Instructions

Customize for your document type in `data_loader.py`:

```python
parsing_instruction="""
Your custom instructions here...
"""
```

## Next Steps: Vector DB Ingestion

Once you have chunks:

```python
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

# Load chunks
chunks = loader.load_chunks_from_disk("processed_chunks/ada_cardio_disease_manag_chunks.json")

# Create embeddings and store
embeddings = OpenAIEmbeddings()
vectorstore = QdrantVectorStore.from_documents(
    chunks,
    embeddings,
    collection_name="t2d_medical_docs",
    url="http://localhost:6333"
)
```

## Traceability in Action

When your RAG system retrieves a chunk, you can display:

```
📄 Source: Hypertension_2024.pdf
📖 Page: 12
📑 Section: Renal Impairment > Dosage Adjustments
```

This allows medical professionals to verify AI-generated claims instantly.

## Architecture

```
MedicalDataLoader
├── __init__()              # Initialize parsers and splitters
├── process_pdf()           # 3-phase pipeline for single PDF
├── save_chunks()           # Save to JSON
├── process_all_pdfs()      # Batch process all PDFs
├── load_chunks_from_disk() # Load saved chunks
└── get_chunk_stats()       # Quality assessment
```

## Quality Metrics

The `get_chunk_stats()` method provides:
- Total chunks created
- Average/min/max chunk sizes
- Unique pages covered
- Unique chapters identified

Use these to validate extraction quality before vector DB ingestion.
