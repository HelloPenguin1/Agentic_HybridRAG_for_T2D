# Define the data structure that flows through the Langgraph workflow
from typing_extensions import TypedDict
from typing import Optional, Literal, List, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    question: str

    router_choice: Literal["graph", "vector", "both", "real_time"]
    router_reasoning: str

    web_search_used: Optional[bool]  # True when evidence gate triggered web search

    generated_cypher: Optional[str]

    graph_result: Optional[str]
    vector_result: Optional[str]

    # for citation, context, and reranking
    vector_docs: Optional[List[Any]]  # list of LangChain Document objects
    graph_docs: Optional[List[Any]]  # list of raw Neo4j result dicts
    web_docs: Optional[List[dict]]  # [{url, title, content}] from Tavily
    web_result: Optional[str]  # formatted web evidence string for synthesizer

    final_answer: str

    hallucination_score: Optional[Literal["faithful", "hallucinated"]]
    audit_feedback: Optional[str]  # Stores diagnostic feedback for the Refiner

    # Traceability: structured citation map from Citation Agent
    citations: Optional[
        List[dict]
    ]  # [{"id": "V1", "source_type": "vector", "label": "..."}]
