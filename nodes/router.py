from config.settings import (
    router_llm,
)  # already configured with structured output in settings
from config.prompts2 import router_prompt


def Router(state):
    """Route the input question to the appropriate retriever node"""
    chain = router_prompt | router_llm
    decision = chain.invoke({"question": state["question"]})
    return {
        "router_choice": decision.router_choice,
        "router_reasoning": decision.router_reasoning,
    }


# Conditional edge logic
def route_decision(state):
    """Return the next node name(s) for LangGraph conditional routing."""
    choice = state["router_choice"]
    if choice == "graph":
        return "graph_retriever"
    elif choice == "vector":
        return "vector_retriever"
    elif choice == "both":
        return ["graph_retriever", "vector_retriever"]
    elif choice == "real_time":
        return "web_search"
