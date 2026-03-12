from core.state import GraphState
from langchain_community.tools.tavily_search import TavilySearchResults


def web_search_node(state: GraphState) -> dict:
    """Fetch evidence from trusted medical sources via Tavily.

    Stores results in 'vector_result' so the downstream synthesiser
    treats them the same way it treats local vector evidence.
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

    # Format each result with its URL for traceability
    formatted = "\n\n".join(
        f"[Web{i+1}] {r.get('url', '')}\n{r.get('content', '')}"
        for i, r in enumerate(results)
        if isinstance(r, dict)
    )

    print(f"[WebSearch] Retrieved {len(results)} results from trusted sources.")
    return {"vector_result": formatted, "web_search_used": True}