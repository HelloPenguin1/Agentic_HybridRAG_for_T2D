"""
eval_runners.py
Runner functions for LangSmith evaluation of 4 workflow configurations.

Each function takes {"question": str} and returns a dict with:
  final_answer   — the system's response (compared against ground truth)
  retrieved_context — combined graph + vector context (used for faithfulness eval)
  router_choice  — which route was taken (used for router accuracy eval)

Pass these directly to langsmith.evaluate() as the `target` argument.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # project root

from workflows.adaptive import workflow as adaptive_workflow
from workflows.naive_hybrid    import workflow as naive_hybrid_workflow
from workflows.vector_only     import workflow as vector_only_workflow
from workflows.graph_only      import workflow as graph_only_workflow


def _context(result: dict) -> str:
    """Combine graph and vector results into a single context string."""
    parts = []
    if result.get("graph_result"):
        parts.append(f"[Graph] {result['graph_result']}")
    if result.get("vector_result"):
        parts.append(f"[Vector] {result['vector_result']}")
    return "\n\n".join(parts) if parts else ""


def run_adaptive_router(inputs: dict) -> dict:
    """Adaptive routing — router agent decides graph / vector / both per question."""
    result = adaptive_workflow.invoke({"question": inputs["question"]})
    return {
        "final_answer":       result["final_answer"],
        "retrieved_context":  _context(result),
        "router_choice":      result.get("router_choice", ""),
    }


def run_fixed_hybrid(inputs: dict) -> dict:
    """Fixed hybrid — always runs graph AND vector retrievers in parallel, no routing."""
    result = naive_hybrid_workflow.invoke({"question": inputs["question"]})
    return {
        "final_answer":       result["final_answer"],
        "retrieved_context":  _context(result),
        "router_choice":      "both",   # fixed, always both
    }


def run_vector_only(inputs: dict) -> dict:
    """Vector-only — semantic search over ADA guideline chunks, no graph."""
    result = vector_only_workflow.invoke({"question": inputs["question"]})
    return {
        "final_answer":       result["final_answer"],
        "retrieved_context":  _context(result),
        "router_choice":      "vector",  # fixed
    }


def run_graph_only(inputs: dict) -> dict:
    """Graph-only — Neo4j Cypher retrieval only, no vector search."""
    result = graph_only_workflow.invoke({"question": inputs["question"]})
    return {
        "final_answer":       result["final_answer"],
        "retrieved_context":  _context(result),
        "router_choice":      "graph",   # fixed
    }
