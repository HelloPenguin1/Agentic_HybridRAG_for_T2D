from pydantic import BaseModel, Field
from typing import Literal

# Structured Output Schema for Router in Workflow
class RouterOutput(BaseModel):
    router_choice: Literal["graph", "vector", "both"] = Field(descrption="Router LLM's decision on which retriever to use based on the question")
    router_reasoning: str = Field(description="Router LLM's reasoning for its retriever choice")

