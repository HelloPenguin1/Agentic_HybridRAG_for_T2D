"""
run_evaluation.py
Runs LangSmith evaluation for all 4 workflow configurations sequentially.

Run from project root:
    python 6_Evaluation/run_evaluation.py

Note: max_concurrency=1 and a 60s sleep between configs keeps Groq TPM
usage well within free-tier limits (~14,400 tokens/min).
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # project root
sys.path.insert(0, str(Path(__file__).parent))          # 6_Evaluation/
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from langsmith import evaluate
from eval_runners import run_adaptive_router, run_fixed_hybrid, run_vector_only, run_graph_only
from evaluators import answer_correctness, faithfulness, completeness, router_accuracy

DATASET   = "diabetes-nursing-qa-v3"
EVALUATORS = [answer_correctness, faithfulness, completeness, router_accuracy]

configs = [
    #("adaptive-router", run_adaptive_router),
    #("fixed-hybrid",    run_fixed_hybrid),
    ("vector-only",     run_vector_only),
    #("graph-only",      run_graph_only),
]

for i, (name, runner) in enumerate(configs):
    print(f" Running: {name}  ({i+1}/{len(configs)})")
    evaluate(
        runner,
        data=DATASET,
        evaluators=EVALUATORS,
        experiment_prefix=name,
        max_concurrency=1,   # one question at a time — avoids Groq TPM spikes
    )
    print(f" Done: {name}")
    if i < len(configs) - 1:
        print(" Waiting 60s before next config (Groq rate limit)...")
        time.sleep(60)
