from pydantic import BaseModel, Field
from typing import Literal, List

# Sentinel returned by graph_retriever when the graph context is empty
GRAPH_EMPTY = "##NO_GRAPH_RESULT##"

# Structured Output Schema for Router in Workflow
class RouterOutput(BaseModel):
    router_choice: Literal["graph", "vector", "both", "real_time"] = Field(description="Router LLM's decision on which retriever to use based on the question")
    router_reasoning: str = Field(description="Router LLM's reasoning for its retriever choice")


class GradeHallucination(BaseModel):
    """Binary score for hallucination check."""
    binary_score: Literal["yes", "no"] = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )
    explanation: str = Field(
        description="Brief explanation of why the answer is or isn't grounded"
    )