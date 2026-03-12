from core.state import GraphState
from langchain_community.tools.tavily_search import TavilySearchResults


def web_search_node(state: GraphState) -> dict:
    """Fetch evidence from trusted medical sources via Tavily.

    Stores structured web_docs (for citation agent) and a formatted
    web_result string (for the synthesizer), mirroring how vector_docs
    and vector_result work for local retrieval.
    """
    search_tool = TavilySearchResults(
        include_domains=[
            "pubmed.ncbi.nlm.nih.gov",
            "diabetes.org",
            "nih.gov",
            "cdc.gov",
            "mayoclinic.org",
            "medlineplus.gov",
        ],
        k=3,
    )

    results = search_tool.invoke({"query": state["question"]})

    # Build structured docs for citation agent
    web_docs = [
        {
            "url": r.get("url", ""),
            "title": r.get("title", r.get("url", "")),
            "content": r.get("content", ""),
        }
        for r in results
        if isinstance(r, dict)
    ]

    # Build formatted string for synthesizer
    web_result = "\n\n".join(doc["content"] for doc in web_docs)

    print(f"[WebSearch] Retrieved {len(web_docs)} results from trusted sources.")
    return {"web_docs": web_docs, "web_result": web_result, "web_search_used": True}