from pydantic import BaseModel, Field
from typing import Literal

# Sentinel returned by graph_retriever when the graph context is empty
GRAPH_EMPTY = "##NO_GRAPH_RESULT##"

# Structured Output Schema for Router in Workflow
class RouterOutput(BaseModel):
    router_choice: Literal["graph", "vector", "both"] = Field(description="Router LLM's decision on which retriever to use based on the question")
    router_reasoning: str = Field(description="Router LLM's reasoning for its retriever choice")


