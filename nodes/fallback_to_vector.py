

from config.output_validation import GRAPH_EMPTY


def fallback_to_vector(state):
    """
    Conditional edge after graph_retriever.
    Routes to synthesizer if graph returned data, or triggers fallback if not.

    - graph_result has real data     → synthesizer
    - graph_result is GRAPH_EMPTY sentinel AND router was 'graph' only
                                     → vector_retriever (fallback)
    - graph_result is GRAPH_EMPTY sentinel AND router was 'both'
                                     → synthesizer (vector already ran in parallel)
    """
    graph_result  = state.get("graph_result", GRAPH_EMPTY)
    router_choice = state.get("router_choice", "graph")

    if graph_result and graph_result != GRAPH_EMPTY:
        return "synthesizer"

    # graph is empty / sentinel
    if router_choice == "graph":
        print("[fallback] Graph empty — falling back to vector retriever.")
        return "vector_retriever"
    else:
        print("[fallback] Graph empty — relying on vector retriever results.")
        return "synthesizer"
    
