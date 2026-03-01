from config.prompts import cypher_generation_prompt_template, FEW_SHOT_EXAMPLES
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from langchain_groq import ChatGroq 
from langchain_neo4j import Neo4jGraph
from config.settings import translator_llm

import os
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")
groq_api_key = os.getenv("GROQ_API_KEY")


class GraphRetriever:

    ##Initialize connection to the Aura DB
    def __init__(self):
        self.graph = Neo4jGraph(
            url=uri,
            username=user,
            password=password,
            database="neo4j"
        )
        self.translator_llm = translator_llm
    
    
    #Convert natural language questions to cypher queries
    def convert_to_cypher(self, question, translator_llm) -> str:
        prompt = cypher_generation_prompt_template.partial(examples=FEW_SHOT_EXAMPLES)
        chain = prompt | translator_llm | StrOutputParser()
        cypher_query = chain.invoke({
            "schema": self.graph.schema,
            "question": question})

        #Preprocess to clean LLM-generated query 
        cypher_query = cypher_query.strip()
        if cypher_query.startswith("```"):
            lines = cypher_query.splitlines()
            cypher_query = "\n".join(lines[1:-1]).strip()

        return cypher_query



    #Directly query the graph database with the Cypher query language and return raw results
    def query_graph(self, cypher_query):
        try:
            result = self.graph.query(cypher_query)
            #df = pd.DataFrame(result)
            if not result:
                print("Query ran but returned no results")
            else:
                print(f"Data from Retriver: {result}")
            return result
        except Exception as e:
            print(f"Query failed: {e}")
            print(f"Cypher attempted:\n{cypher_query}")
            return None 

    
    def graph_retriever_node(self, state):
        user_question = state['question']
        cypher_query = self.convert_to_cypher(user_question, translator_llm)
        graph_result = self.query_graph(cypher_query)
        return {"graph_result": graph_result}
        


if __name__ == "__main__":
    print("Testing connection to Neo4j...")
    try:
        retriever = GraphRetriever()
        print("Connected successfully!")
        
        # Test if we can retrieve the schema
        schema = retriever.graph.schema
        query = "Find all complications"
        cypher_query = retriever.convert_to_cypher(query, translator_llm)
        print("\n--- GENERATED CYPHER ---")
        print(cypher_query)
        print("------------------------\n")
        result = retriever.query_graph(cypher_query)
        print("Result: ", result)

    except Exception as e:
        print(f"Connection failed with error: {e}")
