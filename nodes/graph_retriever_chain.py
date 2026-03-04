import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from config.settings import translator_llm, qa_llm
from config.prompts import cypher_chain_generation_prompt, cypher_chain_qa_prompt
from config.output_validation import GRAPH_EMPTY

load_dotenv()


class GraphRetrieverChain:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
            refresh_schema=True
        )


        self.chain = GraphCypherQAChain.from_llm(
            graph=self.graph,
            cypher_llm=translator_llm,   
            qa_llm=qa_llm,               
            cypher_prompt=cypher_chain_generation_prompt,
            qa_prompt=cypher_chain_qa_prompt,
            verbose=True,
            allow_dangerous_requests=True,
            return_intermediate_steps=True,
        )

    def graph_retriever_node(self, state):
        """LangGraph node. Returns GRAPH_EMPTY sentinel if the graph context was empty."""
        question = state["question"]
        try:
            result = self.chain.invoke({"query": question})

            # intermediate_steps: [{'query': cypher}, {'context': [...rows...]}]
            steps = result.get("intermediate_steps", [])
            context = steps[1].get("context", []) if len(steps) > 1 else []

            if not context:
                print("[GraphRetriever] Empty context — returning GRAPH_EMPTY sentinel.")
                return {"graph_result": GRAPH_EMPTY, "graph_docs": []}

            return {"graph_result": result.get("result", ""), "graph_docs": context}

        except Exception as e:
            print(f"GraphCypherQAChain error: {e}")
            return {"graph_result": GRAPH_EMPTY, "graph_docs": []}


if __name__ == "__main__":
    print("Testing GraphRetrieverChain...")
    try:
        retriever = GraphRetrieverChain()
        question = input("Enter your question: ")
        result = retriever.chain.invoke({"query": question})
        print(f"\nAnswer: {result['result']}")
    except Exception as e:
        print(f"Error: {e}")
