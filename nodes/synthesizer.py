from langchain_core.output_parsers import StrOutputParser
from config.settings import response_llm
from config.prompts2 import base_synthesizer_prompt


def docs_to_text(docs):
    return "\n".join("; ".join(f"{k}: {v}" for k, v in row.items()) for row in docs)


def synthesizer(state):
    """Unified synthesizer — handles graph, vector, and web evidence."""
    question = state["question"]

    vector_result = state.get("vector_result", "")

    graph_ctx_to_text = docs_to_text(state.get("graph_raw_cntx") or [])
    web_result = state.get("web_result", "")

    chain = base_synthesizer_prompt | response_llm | StrOutputParser()

    final_answer = chain.invoke(
        {
            "question": question,
            "vector_result": vector_result
            if vector_result
            else "No results were returned from the vector database.",
            "graph_raw_cntx": graph_ctx_to_text
            if graph_ctx_to_text
            else "No results were returned from the knowledge graph.",
            "web_result": web_result if web_result else "No web search was performed.",
        }
    )

    return {"final_answer": final_answer}
