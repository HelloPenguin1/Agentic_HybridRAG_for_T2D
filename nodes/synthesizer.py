from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import response_llm
from config.prompts import synthesizer_prompt


def synthesizer(state):
    question = state["question"]
    vector_result = state["vector_result"]

    chain = synthesizer_prompt | response_llm | StrOutputParser()

    final_answer = chain.invoke({
        "question": question,
        "vector_result": vector_result if vector_result else "No results were returned from the vector"
    })

    return {"final_answer": final_answer}
