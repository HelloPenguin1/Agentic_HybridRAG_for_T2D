def fallback_to_vector(state):
    """
    Conditional edge after graph_retriever.
    Routes to evidence_gate if graph returned data, or triggers fallback if not.

    - graph_raw_cntx has records     → evidence_gate
    - graph_raw_cntx is empty and router was 'graph' only
                                     → vector_retriever (fallback)
    - graph_raw_cntx is empty and router was 'both'
                                     → evidence_gate (vector already ran in parallel)
    """
    graph_raw_cntx = state.get("graph_raw_cntx") or []
    router_choice = state.get("router_choice", "graph")

    if graph_raw_cntx:
        return "evidence_gate"

    # Graph returned no records.
    if router_choice == "graph":
        print("[fallback] Graph empty — falling back to vector retriever.")
        return "vector_retriever"
    else:
        print("[fallback] Graph empty — relying on vector retriever results.")
        return "evidence_gate"
