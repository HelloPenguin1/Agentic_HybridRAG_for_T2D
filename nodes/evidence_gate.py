"""
Evidence Gate — Threshold-Gated Reasoning Gate for the Agentic GraphRAG pipeline.

Dual-Trigger Logic:
  Trigger 1 (Local Gap)  : Both vector_docs AND graph_docs are empty.
  Trigger 2 (Ambiguity)  : Router classified the query as 'real_time'.

If either trigger fires → route to web_search.
Otherwise             → route to synthesizer (local evidence is sufficient).
"""


def evidence_gate(state) -> dict:
    """LangGraph node: inspect retrieved evidence and flag a web-search need.

    This node does NOT call an LLM — it is a pure-Python quality gate.
    It writes `web_search_used` to state so downstream nodes know the source.
    """
    vector_docs   = state.get("vector_docs") or []
    graph_docs    = state.get("graph_docs")  or []
    router_choice = state.get("router_choice", "")

    needs_web = False

    # Trigger 1: Local Gap — both retrievers came back empty
    if not vector_docs and not graph_docs:
        print("[EvidenceGate] Trigger 1 fired: both retrievers empty.")
        needs_web = True

    # Trigger 2: Ambiguity — router flagged a real-time / temporal query
    if router_choice == "real_time":
        print("[EvidenceGate] Trigger 2 fired: real-time query detected.")
        needs_web = True

    return {"web_search_used": needs_web}


# ── Conditional edge function ────────────────────────────────────────────
def evidence_gate_decision(state) -> str:
    """Return the next node name based on the evidence gate result."""
    if state.get("web_search_used"):
        return "web_search"
    return "synthesizer"
