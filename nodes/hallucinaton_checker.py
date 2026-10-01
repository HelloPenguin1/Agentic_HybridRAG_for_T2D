from config.settings import hallucination_llm
from config.prompts2 import hallucination_grader_prompt


def hallucination_grader(state):
    """Grade whether the final_answer is grounded in the retrieved evidence."""
    # Gather raw evidence from all sources
    v_docs = state.get("vector_docs") or []
    g_docs = state.get("graph_docs") or []
    w_docs = state.get("web_docs") or []
    generation = state.get("final_answer", "")

    # Format evidence into a single string
    v_context = "\n".join([doc.page_content for doc in v_docs])
    g_context = "\n".join([str(record) for record in g_docs])
    w_context = "\n".join([doc["content"] for doc in w_docs])

    full_evidence = (
        f"--- VECTOR EVIDENCE ---\n{v_context}\n\n"
        f"--- GRAPH EVIDENCE ---\n{g_context}\n\n"
        f"--- WEB EVIDENCE ---\n{w_context}"
    )

    # Run the grader chain
    chain = hallucination_grader_prompt | hallucination_llm
    res = chain.invoke({"evidence": full_evidence, "answer": generation})

    return {
        "hallucination_score": "faithful"
        if res.binary_score == "yes"
        else "hallucinated",
        "audit_feedback": res.explanation,
    }


# Conditional edge logic
def hallucination_decision(state):
    """Return the next node name based on the hallucination grade."""
    if state["hallucination_score"] == "faithful":
        return "end"
    else:
        return "refiner"
