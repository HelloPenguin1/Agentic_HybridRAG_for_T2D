from langchain_core.output_parsers import StrOutputParser
from config.settings import response_llm
from config.prompts2 import base_synthesizer_prompt 


def synthesizer(state):
    question = state["question"]
    vector_result = state.get("vector_result", "")
    graph_result = state.get("graph_result", "")

    chain = base_synthesizer_prompt | response_llm | StrOutputParser()

    final_answer = chain.invoke({
        "question": question,
        "vector_result": vector_result if vector_result else "No results were returned from the vector",
        "graph_result": graph_result if graph_result else "No results were returned from the graph"
    })

    return {"final_answer": final_answer}


def hybrid_synthesizer(state):
    question = state["question"]
    vector_result = state.get("vector_result", "")
    graph_result = state.get("graph_result", "")

    chain = fixed_hybrid_synthesizer_prompt | response_llm | StrOutputParser()

    final_answer = chain.invoke({
        "question": question,
        "vector_result": vector_result if vector_result else "No results were returned from the vector",
        "graph_result": graph_result if graph_result else "No results were returned from the graph"
    })

    return {"final_answer": final_answer}