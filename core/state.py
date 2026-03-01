#Define the data structure that flows through the Langgraph workflow
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    question: str

    graph_result: str
    vector_result: str
    
    final_answer: str

