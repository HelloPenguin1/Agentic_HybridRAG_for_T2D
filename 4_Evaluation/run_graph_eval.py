"""Run graph-only workflow (graph_retriever -> synthesizer) on the LangSmith graph dataset.

Run from repo root:  python 4_Evaluation/run_graph_eval.py
UI-defined judge evaluators attached to the dataset score the outputs automatically.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from langsmith import evaluate
from workflows.graph_only import workflow

DATASET = "Graph_Eval_Dataset"  


def run_graph_only(inputs: dict) -> dict:
    state = workflow.invoke({"question": inputs["question"]})
    return {
        "generated_cypher": state.get("generated_cypher"),
        "graph_docs": state.get("graph_docs", []),
        "retrieved_context": state.get("graph_result", ""),
        "final_answer": state.get("final_answer", ""),
    }


def cypher_execution_success(outputs: dict) -> dict:
    return {"results": [
        {"key": "cypher_executed", "score": float(outputs.get("generated_cypher") is not None)},
        {"key": "non_empty_result", "score": float(bool(outputs.get("graph_docs")))},
    ]}


if __name__ == "__main__":
    evaluate(
        run_graph_only,
        data=DATASET,
        evaluators=[cypher_execution_success],
        experiment_prefix="graph_eval_v1",
        max_concurrency=1,
    )