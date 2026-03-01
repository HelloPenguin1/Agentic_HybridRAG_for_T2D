from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import response_llm
from config.prompts import synthesizer_prompt


def synthesizer(state):
    question = state["question"]
    graph_result = state["graph_result"]

    chain = synthesizer_prompt | response_llm | StrOutputParser()

    final_answer = chain.invoke({
        "question": question,
        "graph_result": graph_result if graph_result else "No results were returned from the graph."
    })

    return {"final_answer": final_answer}
