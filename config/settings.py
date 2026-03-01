# Stores Model Gateways
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()

#VectorDB settings ----------------------------------------------------
QDRANT_URL=os.getenv("QDRANT_URL")
QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")
COLLECTION_NAME="ada_model_medembed_base_v0.1"
EMBED_MODEL="abhinand/MedEmbed-base-v0.1"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

#API Keys --------------------------------------------------------------
groq_api_key = os.getenv("GROQ_API_KEY")





#Model Gateways --------------------------------------------------------

#Translator LLM converts natural language questions to cypher queries
translator_llm = ChatGroq(groq_api_key=groq_api_key, 
                          model_name="openai/gpt-oss-20b", 
                          temperature=0)

#Response Generation LLM
response_llm = ChatGroq(groq_api_key=groq_api_key, 
                          model_name="openai/gpt-oss-20b", 
                          temperature=0)

