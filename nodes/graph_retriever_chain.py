import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from config.settings import translator_llm, qa_llm
from config.prompts2 import cypher_chain_generation_prompt, cypher_chain_qa_prompt

load_dotenv()


class GraphRetrieverChain:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
            refresh_schema=True,
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
            return_direct=True,
        )
    def graph_retriever_node(self, state):
        """LangGraph node that returns raw Neo4j records and the generated Cypher."""
        question = state["question"]
        try:
            result = self.chain.invoke({"query": question})

            # Direct mode returns raw rows as "result"; intermediate steps retain Cypher.
            steps = result.get("intermediate_steps", [])

            generated_cypher_query = (
                steps[0].get("query", "No query generated")
                if steps
                else "No query generated"
            )
            context = result.get("result", [])

            if not context:
                print("[GraphRetriever] Empty context.")
                return {
                    "graph_raw_cntx": [],
                    "generated_cypher": generated_cypher_query,
                }

            return {
                "generated_cypher": generated_cypher_query,
                "graph_raw_cntx": context,
            }

        except Exception as e:
            print(f"GraphCypherQAChain error: {e}")
            return {
                "graph_raw_cntx": [],
                "generated_cypher": None,
            }


if __name__ == "__main__":
    print("Testing GraphRetrieverChain... (Press Ctrl+C to exit)")

    try:
        retriever = GraphRetrieverChain()

        while True:
            question = input("\nEnter your question: ")

            result = retriever.chain.invoke({"query": question})

            print(f"Generated_cypher: {result['intermediate_steps'][0]['query']}")
            print("=" * 100)
            print(f"\nAnswer: {result['result']}")

    except KeyboardInterrupt:
        print("\nExiting...")

    except Exception as e:
        print(f"Error: {e}")
