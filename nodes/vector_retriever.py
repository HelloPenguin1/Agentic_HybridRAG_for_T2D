import json
import nltk
from nltk.tokenize import word_tokenize
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank



from config.settings import (
    QDRANT_URL,
    QDRANT_API_KEY,
    COLLECTION_NAME,
    EMBEDDINGS,
    QDRANT_TOPK,
    BM25_TOP,
    CHUNKS_PATH,
)

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def _load_processed_chunks() -> list[dict]:
    """Load all processed chunk JSON files from CHUNKS_PATH into a flat list."""
    chunks = []
    for file_path in CHUNKS_PATH.glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            chunks.extend(json.load(f))
    return chunks


class VectorRetriever:
    def __init__(self):
        self.vector_store = QdrantVectorStore.from_existing_collection(
            collection_name=COLLECTION_NAME,
            embedding=EMBEDDINGS,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        self.qdrant_retriever = self.vector_store.as_retriever(search_kwargs={"k": QDRANT_TOPK})
        self.compressor = FlashrankRerank(top_n=5)


        processed_chunks = _load_processed_chunks()
        bm25_docs = [Document(page_content=chunk.get("page_content", ""),metadata={"chunk_id": chunk["metadata"]["chunk_id"]}) for chunk in processed_chunks]

        self.bm25_retriever = BM25Retriever.from_documents(bm25_docs, k=BM25_TOP, preprocess_func=word_tokenize)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.qdrant_retriever, self.bm25_retriever],
            weights=[0.7, 0.3],
        )
        self.compression_retriever = ContextualCompressionRetriever(base_compressor=self.compressor, base_retriever=self.ensemble_retriever)

    def vector_retriever_node(self, state):
        """LangGraph node: retrieves documents for the given question via RRF."""
        query = state["question"]
        docs = self.compression_retriever.invoke(query)

        vector_result = "\n\n".join(doc.page_content for doc in docs)
        return {"vector_result": vector_result, "vector_docs": docs}


if __name__ == "__main__":
    print("Testing VectorRetriever (Qdrant + BM25 ensemble)…")
    try:
        retriever = VectorRetriever()
        test_query = "What is the recommended HbA1c target for type 2 diabetes?"
        docs = retriever.compression_retriever.invoke(test_query)
        print(f"Retrieved {len(docs)} docs")

    except Exception as e:
        print(f"Error: {e}")    