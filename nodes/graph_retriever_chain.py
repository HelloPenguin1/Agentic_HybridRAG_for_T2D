
import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

from config.settings import translator_llm, qa_llm
from config.prompts2 import (
    cypher_chain_generation_prompt,
    cypher_chain_qa_prompt,
)

load_dotenv()


class GraphRetrieverChain:

    def __init__(self):
        # -------------------------------------------------
        # 1. Neo4j connection configuration
        # -------------------------------------------------
        uri = os.getenv("NEO4J_URI")

        if not uri:
            raise ValueError("NEO4J_URI is missing from environment variables.")

        # Accept self-signed certificates while retaining TLS encryption.
        # neo4j+ssc and bolt+ssc disable certificate verification.
        if uri.startswith("neo4j+s://"):
            uri = uri.replace("neo4j+s://", "neo4j+ssc://", 1)

        elif uri.startswith("bolt+s://"):
            uri = uri.replace("bolt+s://", "bolt+ssc://", 1)

        username = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        database = os.getenv("NEO4J_DATABASE", "neo4j")

        print(f"[Neo4j] Connecting using {uri.split('://')[0]}")

        self.graph = Neo4jGraph(
            url=uri,
            username=username,
            password=password,
            database=database,
            refresh_schema=True,
        )

        print("[Neo4j] Connection successful!")

        # -------------------------------------------------
        # 2. Initialize GraphCypherQAChain
        # -------------------------------------------------
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

    # -----------------------------------------------------
    # 3. LangGraph node
    # -----------------------------------------------------
    def graph_retriever_node(self, state):
        """
        LangGraph node that:
        1. Receives the question from GraphState.
        2. Generates a Cypher query.
        3. Executes the query against Neo4j.
        4. Returns raw records and generated Cypher.
        """

        question = state["question"]

        try:
            result = self.chain.invoke({"query": question})

            # Extract generated Cypher from intermediate steps.Fi
            steps = result.get("intermediate_steps", [])

            generated_cypher_query = (
                steps[0].get("query", "No query generated")
                if steps
                else "No query generated"
            )

            # return_direct=True means result contains raw Neo4j records.
            context = result.get("result", [])

            if not context:
                print("[GraphRetriever] Empty context.")

                return {
                    "graph_raw_cntx": [],
                    "generated_cypher": generated_cypher_query,
                }

            print(f"[GraphRetriever] Retrieved {len(context)} records.")

            return {
                "generated_cypher": generated_cypher_query,
                "graph_raw_cntx": context,
            }

        except Exception as e:
            print(f"[GraphRetriever] GraphCypherQAChain error: {e}")
            raise
            # return {
            #     "graph_raw_cntx": [],
            #     "generated_cypher": None,
            # }


# ---------------------------------------------------------
# 4. Independent testing
# ---------------------------------------------------------
if __name__ == "__main__":

    print("Testing GraphRetrieverChain... (Press Ctrl+C to exit)")

    try:
        retriever = GraphRetrieverChain()

        # Verify that the LangGraph node method is exposed.
        assert callable(retriever.graph_retriever_node)

        print("[Test] graph_retriever_node is available.")

        while True:
            question = input("\nEnter your question: ")

            result = retriever.graph_retriever_node(
                {"question": question}
            )

            print("\nGenerated Cypher:")
            print(result["generated_cypher"])

            print("\nRetrieved Records:")
            print(result["graph_raw_cntx"])

            print("=" * 100)

    except KeyboardInterrupt:
        print("\nExiting...")

    except Exception as e:
        print(f"[Test] Error: {e}")
