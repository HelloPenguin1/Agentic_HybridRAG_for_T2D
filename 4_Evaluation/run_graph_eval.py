import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # project root
sys.path.insert(0, str(Path(__file__).parent))  # 6_Evaluation/
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from langsmith import evaluate
from eval_runners import run_graph_only
from evaluators import (
    cypher_query_correctness,
    context_recall_evaluator,
    e2e_quality_evaluator,
)

DATASET = "T2D_Graph_Dataset"
EVALUATORS = [cypher_query_correctness, context_recall_evaluator, e2e_quality_evaluator]

configs = [
    # ("adaptive-router", run_adaptive_router),
    # ("fixed-hybrid",    run_fixed_hybrid),
    # ("vector-only",     run_vector_only),
    ("graph-only", run_graph_only),
]

for i, (name, runner) in enumerate(configs):
    print(f" Running: {name}  ({i + 1}/{len(configs)})")
    evaluate(
        runner,
        data=DATASET,
        evaluators=EVALUATORS,
        experiment_prefix="graph_eval_v1",
        max_concurrency=1,
    )
    print(f" Done: {name}")
    if i < len(configs) - 1:
        print(" Waiting 60s before next config (Groq rate limit)...")
        time.sleep(60)
