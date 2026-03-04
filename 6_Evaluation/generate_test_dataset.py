"""
generate_test_dataset.py
Generates 25-30 eval questions for the Agentic GraphRAG T2D project.
Randomly samples chunks across ADA guideline files, then asks the LLM to
produce a realistic nurse/patient question, a ground-truth answer, the
correct retrieval route, and question type — all in one call.

Output: 6_Evaluation/test_dataset.json
"""

import json
import os
import random
import time
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"

CHUNKS_DIR  = Path(__file__).parent.parent / "3_vectordb_ingestion" / "processed_chunks"
OUTPUT_PATH = Path(__file__).parent / "test_dataset.json"
TOTAL_QUESTIONS = 28
MIN_CHUNK_CHARS = 400

random.seed(None)

# Load all chunks from all files into one pool
all_chunks = []
for f in CHUNKS_DIR.glob("*.json"):
    with open(f, encoding="utf-8") as fh:
        chunks = json.load(fh)
    for c in chunks:
        text = c["page_content"].strip()
        if (
            len(text) >= MIN_CHUNK_CHARS
            and not text.startswith("Diabetes Care Volume")
            and "doi.org" not in text[:80]
            and not text.startswith("|")
            and text.count(". ") >= 3
        ):
            all_chunks.append(c)

sampled = random.sample(all_chunks, TOTAL_QUESTIONS)
print(f"Sampled {len(sampled)} chunks from {CHUNKS_DIR.name}/\n")


def generate_qa(chunk_text: str) -> dict | None:
    prompt = f"""You are helping build an evaluation dataset for a Type 2 Diabetes clinical assistant used by nurses.

Given the excerpt below, produce ONE realistic question a nurse or patient would ask about T2D management, a detailed ground-truth answer, the correct retrieval route, and the question type.

Retrieval route rules (pick EXACTLY one):
- "graph"  → the answer is a specific fact, dose, threshold, list of medications/complications, or a named relationship between entities (e.g. "What is the max dose of metformin?", "Which SGLT2 inhibitors are approved for CKD?")
- "vector" → the answer requires guideline text, a procedure, protocol, or clinical how-to (e.g. "How should a nurse monitor blood glucose in hospital?", "What is the protocol for hypoglycemia management?")
- "both"   → the question asks what something IS (definition/overview) OR needs both a structured fact AND contextual explanation (e.g. "What is SGLT2 inhibitor therapy and which patients benefit?", "What is CKD and how does it affect diabetes management?")

Question type rules (pick EXACTLY one):
- "factual"     → clear bounded answer (a number, threshold, drug name, list)
- "procedural"  → asks about steps, a process, or how to do something
- "clinical"    → nurse interprets a guideline, lab value, or med to make a care decision

Rules for the question:
- Specific — must name a drug, condition, lab value, or clinical entity
- NOT a hypothetical scenario ("A patient named John...") 
- NOT about study authors, publication years, or citations
- Single sentence ending with a question mark

EXCERPT:
{chunk_text[:2000]}

Reply in JSON only:
{{
  "question": "...",
  "ground_truth_answer": "...",
  "expected_route": "graph" | "vector" | "both",
  "question_type": "factual" | "procedural" | "clinical"
}}"""

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        if not data.get("question") or not data.get("ground_truth_answer"):
            return None
        if data["expected_route"] not in ("graph", "vector", "both"):
            return None
        if data["question_type"] not in ("factual", "procedural", "clinical"):
            return None
        return data
    except Exception as e:
        print(f"  error: {e}")
        return None


dataset = []
for i, chunk in enumerate(sampled, 1):
    chunk_id = chunk["metadata"].get("chunk_id", "?")
    print(f"[{i}/{len(sampled)}] chunk={chunk_id}")
    print(f"  {chunk['page_content'][:90].strip()}...")

    result = generate_qa(chunk["page_content"])
    if not result:
        print("  skipped\n")
        continue

    print(f"  Q: {result['question']}")
    print(f"  route={result['expected_route']}  type={result['question_type']}\n")

    dataset.append({
        "question_id":         f"Q{len(dataset)+1:03d}",
        "question":            result["question"],
        "expected_route":      result["expected_route"],
        "ground_truth_answer": result["ground_truth_answer"],
        "question_type":       result["question_type"],
    })

    if i < len(sampled):
        time.sleep(2)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

route_counts = {r: sum(1 for q in dataset if q["expected_route"] == r) for r in ["graph", "vector", "both"]}
type_counts  = {t: sum(1 for q in dataset if q["question_type"] == t) for t in ["factual", "procedural", "clinical"]}
print(f"Saved {len(dataset)} questions → {OUTPUT_PATH.name}")
print(f"Routes : {route_counts}")
print(f"Types  : {type_counts}")
