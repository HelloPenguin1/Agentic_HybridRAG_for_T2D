# Question Generation Plan - Quick Reference

## Goal
Generate 25-30 nursing-focused questions with ground truth context to evaluate 7 embedding models in Qdrant.

## Question Distribution
- **Factual (30%)**: 7-9 questions - "What is the dose of metformin?"
- **Procedural (30%)**: 7-9 questions - "How to teach insulin injection technique?"
- **Clinical Decision (40%)**: 10-12 questions - "Patient on metformin has eGFR of 28, what to do?"

## Two-Stage Pipeline

### Stage 1: Question Generation
**Script**: `question_generator.py`

1. Sample 50 diverse chunks from 2,632 processed chunks
2. Use LLM (Gemini 2.0 Flash) to generate questions from each chunk
3. Classify questions as factual/procedural/clinical
4. Ensure 30/30/40 distribution
5. Output: `generated_questions.json`

**LLM Prompt Template**:
```
Given this medical text about diabetes nursing care:
{chunk_text}

Generate 2-3 {question_type} questions that:
- Are specific to nursing practice
- Can be answered from this text
- Are clinically relevant

Question Type: {factual|procedural|clinical_decision}
```

### Stage 2: Ground Truth Annotation
**Script**: `ground_truth_annotator.py`

1. For each generated question, scan all 2,632 chunks
2. Use LLM to identify relevant chunks (top 5-10)
3. Extract answer snippets from relevant chunks
4. Assign relevance scores (0-3 scale)
5. Output: `eval_ground_truth.json`

**LLM Prompt Template**:
```
Question: {question}

Review these text chunks and identify which contain relevant information:
Chunk 1: {text}
Chunk 2: {text}
...

For each relevant chunk, provide:
- Relevance score (3=directly answers, 2=partial, 1=tangential, 0=irrelevant)
- Answer snippet (exact text that answers the question)
```

## Output JSON Schema

```json
{
  "metadata": {
    "total_questions": 30,
    "distribution": {"factual": 9, "procedural": 9, "clinical": 12}
  },
  "questions": [
    {
      "question_id": "Q001",
      "question": "What is the starting dose of metformin?",
      "question_type": "factual",
      "ground_truth": {
        "answer": "500-850 mg once or twice daily",
        "relevant_chunks": [
          {
            "chunk_id": "ada_pharma_approaches_chunks_042",
            "relevance_score": 3,
            "answer_snippet": "Metformin should be initiated at 500 mg...",
            "metadata": {
              "source": "ada_pharma_approaches.pdf",
              "page": 8
            }
          }
        ]
      }
    }
  ]
}
```

## Implementation Steps

1. **Setup** (30 min)
   - Create Python scripts
   - Configure Gemini API

2. **Generate Questions** (2-3 hours)
   - Run `question_generator.py`
   - Manual review and filtering

3. **Annotate Ground Truth** (3-4 hours)
   - Run `ground_truth_annotator.py`
   - Validate completeness

4. **Export Dataset** (30 min)
   - Combine into final JSON
   - Generate summary stats

5. **Validate** (1 hour)
   - Manual review of 10 samples
   - Verify accuracy

## Key Benefits

✅ **Scalable**: LLM automates most work  
✅ **Reproducible**: Can regenerate with different parameters  
✅ **Comprehensive**: Ground truth includes metadata for detailed analysis  
✅ **Ready for Experiments**: Direct input for Precision@5, MRR, Recall@10 metrics  

## Cost & Time
- **LLM API Costs**: ~$1-2 (Gemini 2.0 Flash)
- **Total Time**: 5-8 hours (including validation)

## Next Steps
After dataset creation:
1. Run Experiment 1 (compare 7 embedding models)
2. Calculate metrics (P@5, MRR, Recall@10)
3. Select best model
4. Proceed to Experiment 2 (add reranking)
