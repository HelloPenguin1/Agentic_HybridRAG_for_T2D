"""
generate_eval_queries.py
────────────────────────
Samples 1 substantive chunk from each of the 10 source files in processed_chunks/,
uses Groq (llama-3.3-70b) to generate one nursing-focused query per chunk,
and saves eval_queries.json.

Output format:
[
  {
    "query": "...",
    "ground_truth_chunk_id": "4_12",
    "source": "ada_chapter2_chunks.json",
    "chunk_text": "..."   ← for human review only
  },
  ...
]

Run from the vectordb_ingestion/ directory:
    python generate_eval_queries.py
"""

import json
import os
import random
import time
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"

CHUNKS_DIR  = Path(__file__).parent / "processed_chunks"
OUTPUT_PATH = Path(__file__).parent / "eval_queries.json"

# Sample exactly 1 chunk from each of these 10 files (one per clinical topic)
TARGET_FILES = [
    "ada_chapter2_chunks.json",           # Diagnosis & classification
    "ada_chapter3_chunks.json",           # Prevention / prediabetes
    "ada_chapter7_chunks.json",           # Technology (CGM, AID)
    "ada_chapter13_chunks.json",          # Older adults
    "ada_cardio_disease_manag_chunks.json",  # Cardiovascular
    "ada_chronic_kidney_manag_chunks.json",  # CKD
    "ada_glycemic_hypoglycemia_chunks.json", # Glycemic / hypoglycemia
    "ada_obesity_and_weight_chunks.json",    # Obesity & weight
    "ada_pharma_approaches_chunks.json",     # Pharmacology
    "ada_retino_neuro_footcare_chunks.json", # Complications
]

MIN_CHUNK_CHARS = 200   # skip very short/header-only chunks
random.seed(42)         # reproducible sampling


# ── Helper: pick one good chunk from a file ───────────────────────────────────
def sample_chunk(filepath: Path) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter out short/junk chunks
    candidates = [
        c for c in data
        if len(c["page_content"].strip()) >= MIN_CHUNK_CHARS
        # Skip pure header/citation chunks
        and not c["page_content"].strip().startswith("Diabetes Care Volume")
        and "doi.org" not in c["page_content"][:80]
    ]

    if not candidates:
        raise ValueError(f"No suitable chunks found in {filepath.name}")

    return random.choice(candidates)


# ── Helper: generate a nursing query for a chunk ──────────────────────────────
def generate_query(chunk_text: str) -> str:
    prompt = f"""You are an expert at creating clinical evaluation questions for nursing staff.

Given the following excerpt from ADA diabetes clinical guidelines, write ONE realistic question that:
- A nurse or nursing student would actually about T2 Diabetes that cover factual infomation or information about clinical maintainence
- Can be answered DIRECTLY and COMPLETELY from the excerpt below
- Is specific (not vague like "what is diabetes?")
- Is a single sentence ending with a question mark

Return ONLY the question — no explanation, no numbering, no quotes.

EXCERPT:
{chunk_text[:1500]}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=128,
    )
    return response.choices[0].message.content.strip().strip('"').strip("'")


# ── Main ──────────────────────────────────────────────────────────────────────
eval_queries = []

print("=" * 60)
print(f"Generating eval queries using Groq ({MODEL})")
print("=" * 60)

for i, filename in enumerate(TARGET_FILES, 1):
    fpath = CHUNKS_DIR / filename
    if not fpath.exists():
        print(f"\n[{i}/10] ⚠️  File not found: {filename}, skipping")
        continue

    print(f"\n[{i}/10] {filename}")

    # Sample a chunk
    chunk = sample_chunk(fpath)
    chunk_id   = chunk["metadata"]["chunk_id"]
    chunk_text = chunk["page_content"]
    source     = chunk["metadata"].get("source", filename)

    print(f"  chunk_id : {chunk_id}")
    print(f"  text     : {chunk_text[:120].strip()}...")

    # Generate query
    query = generate_query(chunk_text)
    print(f"  query    : {query}")

    eval_queries.append({
        "query": query,
        "ground_truth_chunk_id": chunk_id,
        "source": source,
        "chunk_text": chunk_text,   # kept for human review
    })

    # Respect Groq rate limits (30 req/min free tier)
    if i < len(TARGET_FILES):
        time.sleep(2)

# ── Save ──────────────────────────────────────────────────────────────────────
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(eval_queries, f, indent=2, ensure_ascii=False)

size_kb = OUTPUT_PATH.stat().st_size / 1024
print(f"\n{'='*60}")
print(f"✅ Saved → {OUTPUT_PATH.name}  ({size_kb:.1f} KB)")
print(f"   {len(eval_queries)} eval queries ready for retrieval evaluation")
