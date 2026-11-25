from langchain_core.output_parsers import StrOutputParser
from config.settings import response_llm
from config.prompts2 import base_synthesizer_prompt
from nodes.evidence import build_evidence, format_evidence, docs_to_text  # noqa: F401  (docs_to_text re-exported)


def synthesizer(state):
    """Single-pass answer + citations: the LLM sees ID-tagged evidence and cites it inline as [V1]/[G2]/[W1]."""
    evidence = build_evidence(state)
    chain = base_synthesizer_prompt | response_llm | StrOutputParser()
    final_answer = chain.invoke({"question": state["question"], "evidence": format_evidence(evidence)})
    if not final_answer or not final_answer.strip():
        raise RuntimeError("Synthesizer returned an empty answer.")
    return {"final_answer": final_answer, "evidence": evidence}