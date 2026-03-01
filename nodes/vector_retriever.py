
from config.settings import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, embeddings

from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever



class VectorRetriever:
    def __init__(self):
        self.vector_store = QdrantVectorStore.from_existing_collection(
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        self.qdrant_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})


    def vector_retriever_node(self, state):
        pass



if __name__ == "__main__":
    print("Testing connection to Neo4j...")
    try:
        retriever = VectorRetriever()
        print("Connected successfully!")
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")