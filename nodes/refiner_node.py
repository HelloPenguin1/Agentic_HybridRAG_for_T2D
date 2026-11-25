from config.prompts2 import refiner_prompt
from config.settings import response_llm
from core.state import GraphState
from langchain_core.output_parsers import StrOutputParser
from nodes.evidence import format_evidence


def refiner_node(state: GraphState):
    print("---REFINING ANSWER BASED ON AUDIT---")
    chain = refiner_prompt | response_llm | StrOutputParser()
    refined = chain.invoke({
        "original_answer": state["final_answer"],
        "audit_feedback": state["audit_feedback"],
        "context": format_evidence(state.get("evidence") or []),  # same IDs the draft cited
    })
    return {"final_answer": refined}
