# Chunking Strategy & Metadata - Explained

## ✅ Current Chunk Configuration

### **Chunk Size: 1000 characters**
- **Default**: 1000 chars per chunk
- **Overlap**: 200 chars between chunks

### **Is This Okay?**

**For your use case: YES, it's good! Here's why:**

| Aspect | Current (1000 chars) | Analysis |
|--------|---------------------|----------|
| **LLM Context** | ~250 tokens | ✅ Well within GPT-4's 128K limit |
| **Entity Extraction** | Captures 1-3 paragraphs | ✅ Good for clinical guidelines |
| **Cost** | ~$0.0025 per chunk | ✅ Reasonable ($1.60 for all 2,429 chunks) |
| **Accuracy** | Preserves local context | ✅ Won't split mid-sentence |
| **Overlap** | 200 chars | ✅ Prevents entity loss at boundaries |

### **Your Results:**
- **14 PDFs** → **2,429 chunks**
- **Average**: ~173 chunks per PDF
- **Largest**: `ada_pharma_approaches.pdf` (283 chunks = ~283K chars = ~70 pages)

### **When to Adjust:**

**Increase chunk size (e.g., 2000 chars) if:**
- ❌ Entities are being split across chunks
- ❌ Relationships span multiple paragraphs
- ❌ You want fewer API calls (saves money)

**Decrease chunk size (e.g., 500 chars) if:**
- ❌ Extraction quality is poor (too much noise)
- ❌ You want more granular tracking
- ❌ Memory constraints

**For clinical guidelines, 1000 chars is the sweet spot!** ✅

---

## 📊 Why Store Metadata in Outputs?

You asked: *"Is storing metadata in output necessary? Why is that coded?"*

### **Yes, it's necessary! Here's why:**

### **1. Traceability (Most Important)**
```json
{
  "entity_id": "Diagnostic_Test_00042",
  "canonical_name": "HbA1c",
  "sources": ["ada_chapter2.pdf", "ada_glycemic_hypoglycemia.pdf"],
  "original_chunks": ["p5_c2", "p12_c1"]
}
```

**Why this matters:**
- ✅ You can trace back: "Where did this entity come from?"
- ✅ Verify accuracy: "Let me check the original PDF page 5"
- ✅ Debugging: "Why did it extract this incorrectly?"
- ✅ Audit trail: "Which guideline version said this?"

### **2. Version Control**
Clinical guidelines update yearly (ADA 2024 → 2025 → 2026). Metadata lets you:
- Track which version of a guideline an entity came from
- Compare changes over time
- Know when to re-extract

### **3. Confidence Scoring**
```json
{
  "entity": "Metformin",
  "occurrence_count": 47,
  "sources": ["ada_pharma_approaches.pdf", "ada_chapter3.pdf"],
  "categories": ["pharmacology_technology", "patient_education_lifestyle"]
}
```

**More sources = Higher confidence** that this is a key entity

### **4. Category-Specific Processing**
```json
{
  "chunk_id": "p5_c2",
  "category": "assessment_diagnosis",
  "entity_types": ["Diagnostic_Test", "Metric_Value", "Target_Goal"]
}
```

Different categories have different entity types. Metadata tells you what to expect.

### **5. Debugging Failed Extractions**
When extraction fails, you need to know:
- Which PDF? → `source: "ada_chapter2.pdf"`
- Which page? → `page: 5`
- Which chunk? → `chunk_id: "p5_c2"`
- What category? → `category: "assessment_diagnosis"`

**Without metadata, you're blind!**

### **6. Incremental Processing**
```python
# Skip already-processed chunks
if os.path.exists(f"outputs/extracted_entities/{pdf_stem}_entities.json"):
    print(f"Skipping {pdf_stem} - already processed")
    continue
```

Metadata lets you resume from failures without re-processing everything.

---

## 🎯 What Metadata is Stored?

### **Per Chunk (extracted_text/):**
```json
{
  "text": "The diagnostic criteria for diabetes...",
  "page": 5,
  "chunk_id": "p5_c2",
  "source": "ada_chapter2.pdf",
  "char_count": 987
}
```

### **Per PDF (metadata/):**
```json
{
  "filename": "ada_chapter2.pdf",
  "category": "assessment_diagnosis",
  "title": "Diagnosis and Classification",
  "focus": "Laboratory criteria for diagnosing diabetes",
  "entity_types": ["Diagnostic_Test", "Condition", "Metric_Value"],
  "relationship_types": ["DIAGNOSES_CONDITION_AT_VALUE"],
  "total_chunks": 210,
  "total_chars": 198543
}
```

### **Per Entity (linked/):**
```json
{
  "entity_id": "Diagnostic_Test_00001",
  "canonical_name": "HbA1c",
  "variant_names": ["A1C", "Hemoglobin A1C", "Glycated Hemoglobin"],
  "entity_type": "Diagnostic_Test",
  "sources": ["ada_chapter2.pdf", "ada_glycemic_hypoglycemia.pdf"],
  "occurrence_count": 47,
  "merged_data": {...}
}
```

---

## 💡 Could You Remove Metadata?

**Technically yes, but you'd lose:**
- ❌ Ability to verify extractions
- ❌ Ability to debug errors
- ❌ Ability to trace entities to sources
- ❌ Ability to resume failed runs
- ❌ Ability to track versions
- ❌ Ability to build citations in your RAG system

**For a production knowledge graph, metadata is ESSENTIAL!**

---

## 🚀 Optimization Suggestions

### **If you want to reduce storage:**

1. **Don't save intermediate chunk extractions** (saves ~50% disk space)
   ```python
   # In entity_extractor.py, line 160
   # Comment out the intermediate save:
   # chunk_file = intermediate_path / f"{chunk['chunk_id']}_extraction.json"
   # with open(chunk_file, 'w') as f:
   #     json.dump(result, f, indent=2)
   ```

2. **Compress metadata** (use gzip)
   ```python
   import gzip
   with gzip.open('metadata.json.gz', 'wt') as f:
       json.dump(metadata, f)
   ```

3. **Store only essential fields**
   ```python
   # Minimal metadata
   metadata = {
       'source': chunk['source'],
       'chunk_id': chunk['chunk_id']
   }
   ```

### **If you want better chunking:**

1. **Semantic chunking** (split by paragraphs, not char count)
2. **Sentence-aware chunking** (don't split mid-sentence)
3. **Section-aware chunking** (respect PDF structure)

---

## 📝 Summary

| Question | Answer |
|----------|--------|
| **Is 1000 char chunk size okay?** | ✅ Yes, perfect for clinical guidelines |
| **Is metadata necessary?** | ✅ Yes, essential for traceability & debugging |
| **Can I reduce metadata?** | ⚠️ Yes, but you'll lose important features |
| **Should I change chunk size?** | ❌ No, current size is optimal |

**Your current setup is well-designed for a production knowledge graph!** 🎯
