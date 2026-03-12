"""
Web Synthesizer — Dedicated synthesis node for web-search evidence.

When the Evidence Gate triggers a web search, the answer is built here
and routed directly to citation_agent (bypassing hallucination/refiner,
since external web evidence cannot be audited against local docs).
"""

from langchain_core.output_parsers import StrOutputParser
from config.settings import response_llm
from config.prompts2 import web_synthesizer_prompt
from core.state import GraphState


def web_synthesizer(state: GraphState) -> dict:
    """Synthesise a clinical answer from web-search results."""
    print("---WEB SYNTHESIZER: BUILDING ANSWER FROM WEB EVIDENCE---")

    question      = state["question"]
    web_evidence  = state.get("vector_result", "")

    chain = web_synthesizer_prompt | response_llm | StrOutputParser()

    final_answer = chain.invoke({
        "question": question,
        "web_evidence": web_evidence,
    })

    return {"final_answer": final_answer}
