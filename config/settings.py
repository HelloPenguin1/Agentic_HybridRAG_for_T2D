# Stores Model Gateways
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from config.output_validation import RouterOutput, GradeHallucination
from dotenv import load_dotenv
from pathlib import Path
import os
load_dotenv()
os.environ.pop("SSL_CERT_FILE", None)

#VectorDB settings ----------------------------------------------------
QDRANT_URL=os.getenv("QDRANT_URL")
QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")
COLLECTION_NAME="ada_model_medembed_base_v0.1"
EMBED_MODEL="abhinand/MedEmbed-base-v0.1"
QDRANT_TOPK = 10
BM25_TOP = 10

CHUNKS_PATH = Path(__file__).resolve().parent.parent / "1_vectordb_ingestion" / "processed_chunks"


EMBEDDINGS = HuggingFaceEmbeddings(
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
qa_llm = ChatGroq(groq_api_key=groq_api_key, 
                          model_name="openai/gpt-oss-20b", 
                          temperature=0)


#Router LLM
router_llm = ChatGroq(groq_api_key=groq_api_key, 
                          model_name="openai/gpt-oss-120b", 
                          temperature=0)
router_llm = router_llm.with_structured_output(RouterOutput)


#Final Response Generation LLM
response_llm = ChatGroq(groq_api_key=groq_api_key, 
                          model_name="llama-3.1-8b-instant", 
                          temperature=0,
                          max_tokens=512)

#Hallucination Checker LLM
hallucination_llm = ChatGroq(groq_api_key=groq_api_key, 
                          model_name="qwen/qwen3-32b", 
                          temperature=0,)
hallucination_llm = hallucination_llm.with_structured_output(GradeHallucination)

