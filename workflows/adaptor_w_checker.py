from langgraph.graph import StateGraph, START, END
from core.state import GraphState
from nodes.graph_retriever_chain import GraphRetrieverChain
from nodes.vector_retriever import VectorRetriever
from nodes.synthesizer import synthesizer
from nodes.router import Router, route_decision
from nodes.fallback_to_vector import fallback_to_vector
from nodes.hallucinaton_checker import hallucination_grader, hallucination_decision

graph_retriever_chain = GraphRetrieverChain()
vector_retriever = VectorRetriever()

# Initialize the graph
graph = StateGraph(GraphState)

# Add nodes
graph.add_node("router", Router)
graph.add_node("graph_retriever", graph_retriever_chain.graph_retriever_node)
graph.add_node("vector_retriever", vector_retriever.vector_retriever_node)
graph.add_node("synthesizer", synthesizer)
graph.add_node("hallucination_grader", hallucination_grader)

# Edges
graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router",
    route_decision,
    ["graph_retriever", "vector_retriever"],
)

# After graph_retriever: conditionally fall back to vector if graph returned empty
graph.add_conditional_edges(
    "graph_retriever",
    fallback_to_vector,
    ["vector_retriever", "synthesizer"],
)
graph.add_edge("vector_retriever", "synthesizer")

# Synthesizer → hallucination check → END or loop back
graph.add_edge("synthesizer", "hallucination_grader")
graph.add_conditional_edges(
    "hallucination_grader",
    hallucination_decision,
    {
        "end": END,
        "synthesizer": "synthesizer",
    },
)

# Compile
workflow = graph.compile()


if __name__ == "__main__":
    import os

    # ── Graph visualisation ──────────────────────────────────────────────
    png_path = "workflow_graph.png"
    with open(png_path, "wb") as f:
        f.write(workflow.get_graph(xray=True).draw_mermaid_png())

    print(f"Graph saved → {os.path.abspath(png_path)}")
    os.startfile(
        os.path.abspath(png_path)
    )  # opens with default image viewer on Windows

    # ── Continuous Run Loop ──────────────────────────────────────────────
    print("\nAgent ready. Press Ctrl+C to exit.\n")

    try:
        while True:
            question = input("Question: ")

            result = workflow.invoke({"question": question})

            print(f"\nRouter choice    : {result['router_choice'].upper()}")
            print(f"Router reasoning : {result['router_reasoning']}")
            print(f"Hallucination    : {result['hallucination_score']}")
            print(f"\n{'─' * 60}")
            print(f"Final Answer:\n{result['final_answer']}")
            print(f"{'=' * 60}\n")

    except KeyboardInterrupt:
        print("\nExiting...")
