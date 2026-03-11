from config.prompts2 import refiner_prompt
from core.state import GraphState
from config.settings import response_llm
from langchain_core.output_parsers import StrOutputParser

def refiner_node(state: GraphState):
    print("---REFINING ANSWER BASED ON AUDIT---")
    
    # The prompt here tells the LLM: "Here is a draft and its errors. Fix it."
    refine_chain = refiner_prompt | response_llm | StrOutputParser()

    refined_answer = refine_chain.invoke({
        "original_answer": state["final_answer"],
        "audit_feedback": state["audit_feedback"],
        "context": (state.get("vector_result") or "") + "\n" + (state.get("graph_result") or "")
    })
    
    return {"final_answer": refined_answer}