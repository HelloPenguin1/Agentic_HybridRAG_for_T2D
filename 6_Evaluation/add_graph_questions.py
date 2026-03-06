"""
add_graph_questions.py
Samples random ADA guideline chunks and asks the LLM to generate
graph-oriented factual questions + ground truth answers FROM the chunk.

Trims existing test_dataset.json to half, then appends graph questions.

Usage: python 6_Evaluation/add_graph_questions.py
"""

import json, os, random, time
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"

CHUNKS_DIR   = Path(__file__).parent.parent / "3_vectordb_ingestion" / "processed_chunks"
DATASET_PATH = Path(__file__).parent / "test_dataset.json"
NUM_GRAPH_QS = 20
MIN_CHUNK_CHARS = 400

random.seed(None)


GRAPH_SCHEMA_HINT = """
AVAILABLE NODE LABELS: Medication, Disease, Riskfactor, Adverseeffect, Diagnostictest,
Biomarker, Intervention, Symptom, Targetgoal, Patientprofile, Socialdeterminant, Medicaldevice, Anatomy

AVAILABLE RELATIONSHIPS: TREATS, CAUSES, INCREASES_RISK_OF, DETECTS, MEASURES,
HAS_TARGET_GOAL, REQUIRES_MONITORING, BARRIER_TO, ADMINISTERS, AFFECTS, INTERACTS_WITH

COMMON PATTERNS:
- (Medication)-[:TREATS]->(Disease)
- (Medication)-[:CAUSES]->(Adverseeffect)
- (Riskfactor)-[:INCREASES_RISK_OF]->(Disease)
- (Diagnostictest)-[:DETECTS]->(Disease)
- (Diagnostictest)-[:MEASURES]->(Biomarker)
- (Medication)-[:REQUIRES_MONITORING]->(Diagnostictest)
- (Socialdeterminant)-[:BARRIER_TO]->(Intervention)
"""


def generate_graph_qa(chunk_text: str) -> dict | None:
    """Given a chunk, ask LLM to produce a graph-oriented question + answer FROM the chunk."""
    prompt = f"""You are building an evaluation dataset for a Type 2 Diabetes knowledge graph.

Given the clinical excerpt below, produce ONE simple factual question that a
knowledge graph could answer using entity relationships.

KNOWLEDGE GRAPH SCHEMA:
{GRAPH_SCHEMA_HINT}

QUESTION STYLE — pick one of these patterns:
- "What medications treat [disease from excerpt]?"
- "What are the side effects of [medication from excerpt]?"
- "What increases the risk of [disease from excerpt]?"
- "What diagnostic tests detect [condition from excerpt]?"
- "What is linked to [entity from excerpt]?"
- "List all [entity type] for [entity from excerpt]."
- "What does [test from excerpt] measure?"
- "What monitoring is required for [medication from excerpt]?"

RULES:
1. The question MUST be answerable from the excerpt below — do NOT invent facts
2. The ground_truth_answer MUST come directly from the excerpt text
3. Write the answer as 1-2 coherent sentences (not a bullet list)
4. The question should be simple and direct — one relationship lookup
5. Name specific drugs, diseases, or tests from the excerpt

EXCERPT:
{chunk_text[:2000]}

Reply in JSON only:
{{
  "question": "...",
  "ground_truth_answer": "..."
}}"""

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        if not data.get("question") or not data.get("ground_truth_answer"):
            return None
        return data
    except Exception as e:
        print(f"  error: {e}")
        return None


def main():
    # 1. Load + filter chunks
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

    sampled = random.sample(all_chunks, min(NUM_GRAPH_QS + 5, len(all_chunks)))
    print(f"Sampled {len(sampled)} chunks\n")

    # 2. Load existing dataset, keep first half
    with open(DATASET_PATH, encoding="utf-8") as f:
        existing = json.load(f)
    keep_count = len(existing) // 2
    kept = existing[:keep_count]
    print(f"Existing: {len(existing)} → trimmed to {keep_count}\n")

    # 3. Generate graph questions from chunks
    new_qs = []
    for i, chunk in enumerate(sampled, 1):
        if len(new_qs) >= NUM_GRAPH_QS:
            break

        chunk_id = chunk["metadata"].get("chunk_id", "?")
        print(f"[{i}/{len(sampled)}] chunk={chunk_id}")

        result = generate_graph_qa(chunk["page_content"])
        if not result:
            print("  skipped\n")
            continue

        print(f"  Q: {result['question']}")
        print(f"  A: {result['ground_truth_answer'][:90]}...\n")

        new_qs.append({
            "question_id":         f"Q{keep_count + len(new_qs) + 1:03d}",
            "question":            result["question"],
            "expected_route":      "graph",
            "ground_truth_answer": result["ground_truth_answer"],
            "question_type":       "factual",
        })

        time.sleep(2)

    # 4. Save combined dataset
    combined = kept + new_qs
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    graph_ct  = sum(1 for q in combined if q.get("expected_route") == "graph")
    vector_ct = sum(1 for q in combined if q.get("expected_route") == "vector")
    both_ct   = sum(1 for q in combined if q.get("expected_route") == "both")
    print(f"{'='*50}")
    print(f"Final: {len(combined)} questions  (graph={graph_ct} vector={vector_ct} both={both_ct})")
    print(f"Saved → {DATASET_PATH.name}")


if __name__ == "__main__":
    main()
