#Define the data structure that flows through the Langgraph workflow
from typing_extensions import TypedDict
from typing import Optional, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    question: str

    router_choice: Literal["graph", "vector", "both"] 
    router_reasoning: str
    
    graph_result: Optional[str]
    vector_result: Optional[str]


    final_answer: str

