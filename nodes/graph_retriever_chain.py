import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from config.settings import translator_llm
from config.prompts import cypher_chain_generation_prompt, cypher_chain_qa_prompt

load_dotenv()


class GraphRetrieverChain:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD"),
            database="neo4j",
        )

        self.chain = GraphCypherQAChain.from_llm(
            graph=self.graph,
            llm=translator_llm,
            cypher_prompt=cypher_chain_generation_prompt,
            qa_prompt=cypher_chain_qa_prompt,
            verbose=True,
            allow_dangerous_requests=True,
            return_intermediate_steps=True,
        )

    def graph_retriever_node(self, state):
        """LangGraph node: runs the full QA chain and stores the answer as graph_result."""
        question = state["question"]
        try:
            result = self.chain.invoke({"query": question})
            graph_result = result.get("result", "No result returned from graph.")
        except Exception as e:
            print(f"GraphCypherQAChain error: {e}")
            graph_result = ""
        return {"graph_result": graph_result}


if __name__ == "__main__":
    print("Testing GraphRetrieverChain...")
    try:
        retriever = GraphRetrieverChain()
        question = "What are some complications that are a result of Diabetes"
        print(f"\nQuestion: {question}")
        result = retriever.chain.invoke({"query": question})
        print(f"\nAnswer: {result['result']}")
    except Exception as e:
        print(f"Error: {e}")
