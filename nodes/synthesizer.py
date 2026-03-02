from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import response_llm
from config.prompts import synthesizer_prompt


def synthesizer(state):
    question = state["question"]
    vector_result = state.get("vector_result", "")
    graph_result = state.get("graph_result", "")

    chain = synthesizer_prompt | response_llm | StrOutputParser()

    final_answer = chain.invoke({
        "question": question,
        "vector_result": vector_result if vector_result else "No results were returned from the vector",
        "graph_result": graph_result if graph_result else "No results were returned from the graph"
    })

    return {"final_answer": final_answer}


Adjust this so that it provides a more complete answer based on the context provided by the vector and graph retrievers
for exmaple rohjt now, hybrid workflpw ios not working well with the question: what is ckd ?