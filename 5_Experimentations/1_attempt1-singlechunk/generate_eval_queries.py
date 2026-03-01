"""
generate_eval_queries.py
────────────────────────
Samples chunks from each source file in processed_chunks/, generates one
nursing-focused query per chunk using Groq, and APPENDS results to eval_queries.json.

How many per file: set SAMPLES_PER_FILE (default 2 → 20 total across 10 files).
Already-sampled chunk_ids are skipped so re-running never produces duplicates.

Run from vectordb_ingestion/:
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

CHUNKS_DIR      = Path(__file__).parent / "processed_chunks"
OUTPUT_PATH     = Path(__file__).parent / "eval_queries.json"
SAMPLES_PER_FILE = 2       # change to 1 for 10 total, 2 for 20, etc.
MIN_CHUNK_CHARS  = 300     # skip short / header / citation chunks
random.seed(None)          # random seed so each run picks different chunks

TARGET_FILES = [
    "ada_chapter2_chunks.json",
    "ada_chapter3_chunks.json",
    "ada_chapter7_chunks.json",
    "ada_chapter13_chunks.json",
    "ada_cardio_disease_manag_chunks.json",
    "ada_chronic_kidney_manag_chunks.json",
    "ada_glycemic_hypoglycemia_chunks.json",
    "ada_obesity_and_weight_chunks.json",
    "ada_pharma_approaches_chunks.json",
    "ada_retino_neuro_footcare_chunks.json",
]

# ── Load existing queries (to avoid re-sampling same chunk_ids) ───────────────
if OUTPUT_PATH.exists():
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        existing = json.load(f)
else:
    existing = []

used_chunk_ids = {item["ground_truth_chunk_id"] for item in existing}
print(f"Existing queries : {len(existing)}")
print(f"Used chunk_ids   : {len(used_chunk_ids)}\n")


# ── Helper: pick N unseen substantive chunks from a file ─────────────────────
def sample_chunks(filepath: Path, n: int) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = [
        c for c in data
        if len(c["page_content"].strip()) >= MIN_CHUNK_CHARS
        and c["metadata"].get("chunk_id") not in used_chunk_ids
        # Skip reference lists and page headers
        and not c["page_content"].strip().startswith("Diabetes Care Volume")
        and "doi.org" not in c["page_content"][:80]
        and not c["page_content"].strip().startswith("|")   # pure table rows
    ]

    if not candidates:
        return []

    return random.sample(candidates, min(n, len(candidates)))


# ── Helper: generate a nursing query ─────────────────────────────────────────
def generate_query(chunk_text: str) -> str:
    prompt = f"""You are an expert at creating clinical evaluation questions for nursing staff.

Given the following excerpt from ADA diabetes clinical guidelines, write ONE realistic question that:
- A nurse or nursing student would actually ask during patient care
- Can be answered DIRECTLY and COMPLETELY from the excerpt below
- Is specific and clinical (not vague like "what is diabetes?")
- Is a single sentence ending with a question mark

Return ONLY the question — no explanation, no numbering, no quotes.

EXCERPT:
{chunk_text[:1500]}
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=128,
    )
    return response.choices[0].message.content.strip().strip('"').strip("'")


# ── Main ──────────────────────────────────────────────────────────────────────
new_queries = []

print("=" * 60)
print(f"Generating {SAMPLES_PER_FILE} queries × {len(TARGET_FILES)} files = "
      f"{SAMPLES_PER_FILE * len(TARGET_FILES)} new queries")
print("=" * 60)

for i, filename in enumerate(TARGET_FILES, 1):
    fpath = CHUNKS_DIR / filename
    if not fpath.exists():
        print(f"\n[{i}/{len(TARGET_FILES)}] ⚠️  Not found: {filename}")
        continue

    print(f"\n[{i}/{len(TARGET_FILES)}] {filename}")
    chunks = sample_chunks(fpath, SAMPLES_PER_FILE)

    if not chunks:
        print("  ⚠️  No unused candidates found — skipping")
        continue

    for j, chunk in enumerate(chunks, 1):
        chunk_id   = chunk["metadata"]["chunk_id"]
        chunk_text = chunk["page_content"]
        source     = chunk["metadata"].get("source", filename)

        print(f"  sample {j}: chunk_id={chunk_id}")
        print(f"  text   : {chunk_text[:100].strip()}...")

        query = generate_query(chunk_text)
        print(f"  query  : {query}")

        new_queries.append({
            "query":                  query,
            "ground_truth_chunk_id":  chunk_id,
            "source":                 source,
            "chunk_text":             chunk_text,
        })

        # Mark as used so sibling samples in same run don't collide
        used_chunk_ids.add(chunk_id)

        if not (i == len(TARGET_FILES) and j == len(chunks)):
            time.sleep(2)   # respect Groq rate limit


# ── Append & save ─────────────────────────────────────────────────────────────
combined = existing + new_queries

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(combined, f, indent=2, ensure_ascii=False)

size_kb = OUTPUT_PATH.stat().st_size / 1024
print(f"\n{'='*60}")
print(f" {OUTPUT_PATH.name} updated")
print(f"   Before : {len(existing)} queries")
print(f"   Added  : {len(new_queries)} queries")
print(f"   Total  : {len(combined)} queries  ({size_kb:.1f} KB)")
