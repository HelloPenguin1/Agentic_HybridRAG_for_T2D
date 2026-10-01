from config.prompts2 import refiner_prompt
from core.state import GraphState
from config.settings import response_llm
from langchain_core.output_parsers import StrOutputParser


def refiner_node(state: GraphState):
    print("---REFINING ANSWER BASED ON AUDIT---")

    # The prompt here tells the LLM: "Here is a draft and its errors. Fix it."
    refine_chain = refiner_prompt | response_llm | StrOutputParser()

    # Combine all evidence sources as context
    context_parts = []
    if state.get("vector_result"):
        context_parts.append(state["vector_result"])
    if state.get("graph_result"):
        context_parts.append(state["graph_result"])
    if state.get("web_result"):
        context_parts.append(state["web_result"])

    refined_answer = refine_chain.invoke(
        {
            "original_answer": state["final_answer"],
            "audit_feedback": state["audit_feedback"],
            "context": "\n".join(context_parts),
        }
    )

    return {"final_answer": refined_answer}
