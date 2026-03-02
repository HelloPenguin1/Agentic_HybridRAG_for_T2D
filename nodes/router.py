
from config.settings import router_llm  #already configured with structured output in settings
from config.prompts import router_prompt

def Router(state):
    """Route the input question to the appropriate retriever node"""
    chain = router_prompt | router_llm
    decision = chain.invoke({"question": state["question"]})
    return {"router_choice": decision["router_choice"], "router_reasoning": decision["router_reasoning"]}



 # Conditional edge logic 
def route_decision(state):
    # Return the node name you want to visit next
    if state["router_choice"] == "graph":
        return "graph_retriever"
    elif state["router_choice"] == "vector":
        return "vector_retriever"
    elif state["decision"] == "both":   #if both retriever, send question to both retrievers parallely
        return "both_retriever"