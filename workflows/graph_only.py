from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from core.state import GraphState
from nodes.graph_retriever_chain import GraphRetrieverChain
from nodes.synthesizer import synthesizer

graph_retriever = GraphRetrieverChain()

# initialize the graph
graph = StateGraph(GraphState)

# add nodes to the graph
graph.add_node("graph_retriever", graph_retriever.graph_retriever_node)
graph.add_node("synthesizer", synthesizer)

# add edges to the graph
graph.add_edge(START, "graph_retriever")
graph.add_edge("graph_retriever", "synthesizer")
graph.add_edge("synthesizer", END)

# compile the graph
workflow = graph.compile()

