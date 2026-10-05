import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

load_dotenv()

DATASET = Path(__file__).resolve().with_name("graph_eval_v2.json")
TOP_K = 10  # GraphCypherQAChain default top_k -> keeps context identical to graph_raw_cntx

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE", "neo4j"),
    refresh_schema=False,  # schema not needed to run raw Cypher
)

data = json.loads(DATASET.read_text(encoding="utf-8"))

for i, item in enumerate(data, 1):
    try:
        rows = graph.query(item["expected_cypher"])[:TOP_K]  # list[dict], same as chain's raw result
    except Exception as e:
        print(f"[{i}/{len(data)}] ERROR: {e}")
        rows = []
    if not rows:
        print(f"[{i}/{len(data)}] WARNING: empty result -> {item['question']}")
    item["ground_truth_context"] = rows
    item["ground_truth_answer"] = ""

DATASET.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print(f"Done. Updated {len(data)} items in {DATASET}")