# Stores Model Gateways
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

#API Keys
groq_api_key = os.getenv("GROQ_API_KEY")


#Model Gateways

#Translator LLM converts natural language questions to cypher queries
translator_llm = ChatGroq(groq_api_key=groq_api_key, 
                          model_name="openai/gpt-oss-20b", 
                          temperature=0)

#Response Generation LLM
response_llm = ChatGroq(groq_api_key=groq_api_key, 
                          model_name="openai/gpt-oss-20b", 
                          temperature=0)