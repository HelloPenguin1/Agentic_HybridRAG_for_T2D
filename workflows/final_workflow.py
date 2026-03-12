"""
Final Workflow — Threshold-Gated Hybrid Routing Pipeline.

Architecture:
  START → router ─┬─ graph  → graph_retriever ─┬─ (has data)          → evidence_gate
                   │                            ├─ (empty + graph-only) → vector_retriever
                   │                            └─ (empty + both)      → evidence_gate
                   ├─ vector → vector_retriever ──────────────────────→ evidence_gate
                   ├─ both   → graph_retriever & vector_retriever ───→ evidence_gate
                   └─ real_time → web_search ────────────────────────→ web_synthesizer

  evidence_gate ──┬─ (local evidence sufficient) → synthesizer → hallucination_grader ─┬─ faithful     → citation_agent
                  │                                                                    └─ hallucinated → refiner → citation_agent
                  └─ (local gap / real_time)      → web_search → web_synthesizer ──────→ citation_agent

  citation_agent → END
"""

from langgraph.graph import StateGraph, START, END
from core.state import GraphState

# ── Node imports ─────────────────────────────────────────────────────────
from nodes.graph_retriever_chain import GraphRetrieverChain
from nodes.vector_retriever import VectorRetriever
from nodes.synthesizer import synthesizer
from nodes.router import Router, route_decision
from nodes.fallback_to_vector import fallback_to_vector
from nodes.hallucinaton_checker import hallucination_grader, hallucination_decision
from nodes.refiner_node import refiner_node
from nodes.citation_agent import citation_agent
from nodes.evidence_gate import evidence_gate, evidence_gate_decision
from nodes.web_search import web_search_node
from nodes.web_synthesizer import web_synthesizer


# ── Instantiate stateful retrievers ─────────────────────────────────────
graph_retriever_chain = GraphRetrieverChain()
vector_retriever = VectorRetriever()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BUILD THE GRAPH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
graph = StateGraph(GraphState)

# ── Register nodes ──────────────────────────────────────────────────────
graph.add_node("router",               Router)
graph.add_node("graph_retriever",      graph_retriever_chain.graph_retriever_node)
graph.add_node("vector_retriever",     vector_retriever.vector_retriever_node)
graph.add_node("evidence_gate",        evidence_gate)
graph.add_node("synthesizer",          synthesizer)
graph.add_node("hallucination_grader", hallucination_grader)
graph.add_node("refiner",             refiner_node)
graph.add_node("citation_agent",       citation_agent)
graph.add_node("web_search",           web_search_node)
graph.add_node("web_synthesizer",      web_synthesizer)


# ── 1. START → Router ───────────────────────────────────────────────────
graph.add_edge(START, "router")


# ── 2. Router conditional edges ─────────────────────────────────────────
#    graph | vector | both | real_time
graph.add_conditional_edges(
    "router",
    route_decision,
    ["graph_retriever", "vector_retriever", "web_search"],
)


# ── 3. Graph retriever → fallback logic (existing) ──────────────────────
#    If graph returned data → evidence_gate
#    If graph empty + router was "graph" → vector_retriever (fallback)
#    If graph empty + router was "both"  → evidence_gate (vector already ran)
graph.add_conditional_edges(
    "graph_retriever",
    fallback_to_vector,
    ["vector_retriever", "evidence_gate"],
)


# ── 4. Vector retriever → Evidence Gate ─────────────────────────────────
graph.add_edge("vector_retriever", "evidence_gate")


# ── 5. Evidence Gate (Dual-Trigger Reasoning Gate) ──────────────────────
#    Sufficient local evidence → synthesizer (local path)
#    Local gap / real-time     → web_search  (web path)
graph.add_conditional_edges(
    "evidence_gate",
    evidence_gate_decision,
    ["synthesizer", "web_search"],
)


# ── 6. LOCAL PATH: Synthesizer → Hallucination Check → Refiner loop ────
graph.add_edge("synthesizer", "hallucination_grader")
graph.add_conditional_edges(
    "hallucination_grader",
    hallucination_decision,
    {
        "end":     "citation_agent",
        "refiner": "refiner",
    },
)
graph.add_edge("refiner", "citation_agent")


# ── 7. WEB PATH: Web Search → Web Synthesizer → Citation Agent ─────────
#    Bypasses hallucination/refiner (external evidence can't be locally audited)
graph.add_edge("web_search", "web_synthesizer")
graph.add_edge("web_synthesizer", "citation_agent")


# ── 8. Terminal ─────────────────────────────────────────────────────────
graph.add_edge("citation_agent", END)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPILE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
workflow = graph.compile()


if __name__ == "__main__":
    import os

    # ── Graph visualisation ──────────────────────────────────────────────
    png_path = "workflow_graph.png"
    with open(png_path, "wb") as f:
        f.write(workflow.get_graph(xray=True).draw_mermaid_png())

    print(f"Graph saved → {os.path.abspath(png_path)}")
    os.startfile(os.path.abspath(png_path))

    # ── Continuous Run Loop ──────────────────────────────────────────────
    print("\nAgent ready. Press Ctrl+C to exit.\n")

    try:
        while True:
            question = input("Question: ")

            result = workflow.invoke({"question": question})

            print(f"\nRouter choice    : {result['router_choice'].upper()}")
            print(f"Router reasoning : {result['router_reasoning']}")
            print(f"Web search used  : {result.get('web_search_used', False)}")
            print(f"Hallucination    : {result.get('hallucination_score', 'N/A (web path)')}")
            print(f"Citations        : {len(result.get('citations') or [])}")
            print(f"\n{'─'*60}")
            print(f"Final Answer:\n{result['final_answer']}")
            print(f"{'='*60}\n")

    except KeyboardInterrupt:
        print("\nExiting...")
