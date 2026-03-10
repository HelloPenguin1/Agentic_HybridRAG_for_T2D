"""
generate_test_dataset.py
Generates 25-30 eval questions for the Agentic GraphRAG T2D project.
Randomly samples chunks across ADA guideline files, then asks the LLM to
produce a realistic nurse/patient question, a ground-truth answer, the
correct retrieval route, and question type — all in one call.

Output: 6_Evaluation/test_dataset.json
"""

import json
import re
import os
import random
import time
from pathlib import Path
from langchain_openai import ChatOpenAI

from groq import Groq
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-120b"

CHUNKS_DIR  = Path(__file__).parent.parent / "3_vectordb_ingestion" / "processed_chunks"
OUTPUT_PATH = Path(__file__).parent / "test_dataset.json"
TOTAL_QUESTIONS = 28
MIN_CHUNK_CHARS = 500

random.seed(None)

# ── Junk-chunk filter ──────────────────────────────────────────────────
REF_PATTERNS = re.compile(
    r"(^\s*#{1,3}\s*(References|Additional References|Bibliography))"
    r"|(^\s*\d{1,3}\.\s+[A-Z][a-z]+\s+[A-Z]{1,2}[,.])",   # numbered author citation
    re.MULTILINE,
)

def is_clinical_chunk(text: str) -> bool:
    """Return True only when the chunk contains usable clinical prose."""
    # Too short
    if len(text) < MIN_CHUNK_CHARS:
        return False
    # Starts with journal header
    if text.startswith("Diabetes Care Volume"):
        return False
    # DOI / link-heavy header
    if "doi.org" in text[:120]:
        return False
    # Pure markdown table
    if text.startswith("|") or text.count("|") > len(text) // 40:
        return False
    # Too few sentences (likely a table or list of names)
    if text.count(". ") < 4:
        return False
    # Reference / citation sections  (>40 % of lines look like citations)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    citation_lines = sum(1 for l in lines if re.match(r'^\d{1,3}\.\s+[A-Z]', l))
    if citation_lines / max(len(lines), 1) > 0.35:
        return False
    # Section header is "References" / "Additional References"
    if REF_PATTERNS.search(text[:300]):
        return False
    # Mostly numbers / symbols (tables disguised as text)
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.50:
        return False
    return True


# ── Load all chunks from all files into one pool ──────────────────────
all_chunks = []
for f in CHUNKS_DIR.glob("*.json"):
    with open(f, encoding="utf-8") as fh:
        chunks = json.load(fh)
    for c in chunks:
        text = c["page_content"].strip()
        if is_clinical_chunk(text):
            all_chunks.append(c)

print(f"Eligible clinical chunks: {len(all_chunks)}")
sampled = random.sample(all_chunks, min(TOTAL_QUESTIONS, len(all_chunks)))
print(f"Sampled {len(sampled)} chunks from {CHUNKS_DIR.name}/\n")


def generate_qa(chunk_text: str) -> dict | None:
    prompt = f"""You are generating evaluation questions for an Agentic GraphRAG clinical assistant used by nurses for Type 2 Diabetes (T2D) care.

Given the EXCERPT below, produce ONE precise, entity-specific clinical question and its answer.

IF the excerpt is mostly citations, reference lists, author names, journal info,
table-of-contents material, or otherwise lacks actionable clinical content,
return EXACTLY: {{"skip": true}}

═══════════════════════════════════════════════
STEP 1 — Find concrete clinical entities in the excerpt:
  • drug names or drug classes (e.g. empagliflozin, SGLT2 inhibitors)
  • lab values or thresholds  (e.g. A1C < 7%, eGFR < 30 mL/min)
  • complications or conditions (e.g. DKA, CKD stage 4, retinopathy)
  • doses or frequencies       (e.g. 500 mg twice daily)
  • screening / monitoring recommendations
  • named procedures or therapies

If you cannot find at least ONE such entity, return {{"skip": true}}.

STEP 2 — Write a question that names the entity and asks about a
clinically useful relationship (dose, indication, threshold, screening
interval, mechanism, complication link, etc.).

═══════════════════════════════════════════════
QUESTION RULES

MUST:
  ✔ Name a specific drug, lab value, condition, therapy, or guideline concept
  ✔ Be answerable from the excerpt alone
  ✔ Use concrete clinical language
  ✔ Be one sentence ending with ?

MUST NOT:
  ✘ Be vague ("How is diabetes treated?")
  ✘ Use generic phrases like "treatment options" or "diabetes management"
  ✘ Mention authors, study names, journal titles, or publication years
  ✘ Be a patient-scenario / hypothetical story
  ✘ Ask about information not present in the excerpt

═══════════════════════════════════════════════
ROUTE (pick one)
  graph  — answer is a specific fact, dose, list, threshold, or entity relationship
  vector — answer requires guideline prose, protocol steps, or procedural explanation
  both   — needs structured fact + contextual explanation, or a definition

TYPE (pick one)
  factual    — number, drug name, threshold, list
  procedural — steps, process, monitoring workflow
  clinical   — interpreting a guideline recommendation for patient care

═══════════════════════════════════════════════
EXCERPT:
{chunk_text[:2000]}

═══════════════════════════════════════════════
Respond with JSON only — no markdown fences, no extra text.

Either:
{{"skip": true}}

Or:
{{
  "question": "...",
  "ground_truth_answer": "...",
  "expected_route": "graph | vector | both",
  "question_type": "factual | procedural | clinical"
}}
"""

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        # LLM decided the excerpt is unusable
        if data.get("skip"):
            return None
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


# Load existing questions so we can append
existing = []
if OUTPUT_PATH.exists():
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        try:
            existing = json.load(f)
        except json.JSONDecodeError:
            existing = []
start_id = len(existing)
print(f"Existing questions: {start_id}  (will append new ones)\n")

new_questions = []
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

    new_questions.append({
        "question_id":         f"Q{start_id + len(new_questions) + 1:03d}",
        "question":            result["question"],
        "expected_route":      result["expected_route"],
        "ground_truth_answer": result["ground_truth_answer"],
        "question_type":       result["question_type"],
    })

    if i < len(sampled):
        time.sleep(2)

dataset = existing + new_questions
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

route_counts = {r: sum(1 for q in dataset if q["expected_route"] == r) for r in ["graph", "vector", "both"]}
type_counts  = {t: sum(1 for q in dataset if q["question_type"] == t) for t in ["factual", "procedural", "clinical"]}
print(f"Saved {len(dataset)} questions → {OUTPUT_PATH.name}")
print(f"Routes : {route_counts}")
print(f"Types  : {type_counts}")
