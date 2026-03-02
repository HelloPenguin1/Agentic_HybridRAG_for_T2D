#Define the data structure that flows through the Langgraph workflow
from typing_extensions import TypedDict
from typing import Optional, Literal, List, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    question: str

    router_choice: Literal["graph", "vector", "both"] 
    router_reasoning: str
    
    graph_result: Optional[str]
    vector_result: Optional[str]

    # for citation, context, and reranking
    vector_docs: Optional[List[Any]]   # list of LangChain Document objects
    graph_docs: Optional[List[Any]]    # list of raw Neo4j result dicts


    final_answer: str

