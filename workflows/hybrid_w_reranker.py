from langgraph.graph import StateGraph, START, END
from core.state import GraphState
from nodes.graph_retriever_chain import GraphRetrieverChain
from nodes.vector_retriever import VectorRetriever
from nodes.synthesizer import synthesizer
from nodes.router import Router, route_decision
from nodes.fallback_to_vector import fallback_to_vector
from nodes.reranker import reranker_node

graph_retriever_chain = GraphRetrieverChain()
vector_retriever = VectorRetriever()

# Initialize the graph
graph = StateGraph(GraphState)

# Add nodes
graph.add_node("router",           Router)
graph.add_node("graph_retriever",  graph_retriever_chain.graph_retriever_node)
graph.add_node("vector_retriever", vector_retriever.vector_retriever_node)
graph.add_node("reranker",         reranker_node)
graph.add_node("synthesizer",      synthesizer)

# ── Edges ─────────────────────────────────────────────────────────────────────

graph.add_edge(START, "router")

# Router conditionally fans out to one or both retrievers
graph.add_conditional_edges(
    "router",
    route_decision,
    ["graph_retriever", "vector_retriever"],
)

# After graph_retriever: fall back to vector if graph empty, otherwise reranker
def _graph_after_retrieve(state):
    """
    If graph returned real data  → reranker
    If graph is empty + router was 'graph' only → vector_retriever (fallback)
    If graph is empty + router was 'both'       → reranker (vector already ran)
    """
    from config.output_validation import GRAPH_EMPTY
    graph_result  = state.get("graph_result", GRAPH_EMPTY)
    router_choice = state.get("router_choice", "graph")

    if graph_result and graph_result != GRAPH_EMPTY:
        return "reranker"

    if router_choice == "graph":
        print("[fallback] Graph empty — falling back to vector retriever.")
        return "vector_retriever"
    else:
        print("[fallback] Graph empty — relying on vector results; going to reranker.")
        return "reranker"

graph.add_conditional_edges(
    "graph_retriever",
    _graph_after_retrieve,
    ["vector_retriever", "reranker"],
)

# Vector retriever always goes to reranker
graph.add_edge("vector_retriever", "reranker")
graph.add_edge("reranker",         "synthesizer")
graph.add_edge("synthesizer",      END)

# Compile
workflow = graph.compile()


if __name__ == "__main__":
    import os

    # ── Graph visualisation ───────────────────────────────────────────────────
    png_path = "workflow_reranker_graph.png"
    with open(png_path, "wb") as f:
        f.write(workflow.get_graph(xray=True).draw_mermaid_png())
    print(f"Graph saved → {os.path.abspath(png_path)}")
    os.startfile(os.path.abspath(png_path))

    # ── Run ───────────────────────────────────────────────────────────────────
    question = "What is CKD?"
    print(f"\n{'='*60}")
    print(f"Question : {question}")
    print(f"{'='*60}")

    result = workflow.invoke({"question": question})

    print(f"\nRouter choice    : {result['router_choice'].upper()}")
    print(f"Router reasoning : {result['router_reasoning']}")
    ranked = result.get("ranked_docs", [])
    print(f"\nRanked docs kept : {len(ranked)}")
    for i, d in enumerate(ranked):
        print(f"  [{i+1}] source={d['source']} | {d['text'][:120]}...")
    print(f"\n{'─'*60}")
    print(f"Final Answer:\n{result['final_answer']}")
    print(f"{'='*60}")
