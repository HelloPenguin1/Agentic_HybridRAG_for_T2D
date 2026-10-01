from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from core.state import GraphState
from nodes.vector_retriever import VectorRetriever
from nodes.synthesizer import synthesizer

vector_retriever = VectorRetriever()
# initialize the graph
graph = StateGraph(GraphState)

# add nodes to the graph
graph.add_node("vector_retriever", vector_retriever.vector_retriever_node)
graph.add_node("synthesizer", synthesizer)

# add edges to the graph
graph.add_edge(START, "vector_retriever")
graph.add_edge("vector_retriever", "synthesizer")
graph.add_edge("synthesizer", END)

# compile the graph
workflow = graph.compile()


if __name__ == "__main__":
    print("Testing workflow...")
    result = workflow.invoke(
        {
            "question": "What is the recommended first-line pharmacologic treatment for hyperglycemia caused by mTOR kinase inhibitors such as everolimus?"
        }
    )
    print(result["final_answer"])
