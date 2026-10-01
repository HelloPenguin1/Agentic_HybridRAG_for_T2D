"""
Minimal FastAPI wrapper for Agentic GraphRAG Workflow.
Three endpoints: /query, /health, /docs (auto-generated).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import sys
import os

# Add parent directory to path to import workflow
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from workflows.final_workflow import workflow


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, description="Nursing question about diabetes care"
    )

    class Config:
        json_schema_extra = {
            "example": {"question": "When should I hold metformin before a procedure?"}
        }


class Citation(BaseModel):
    id: str
    source_type: str  # "vector", "graph", or "web"
    label: str


class QueryMetadata(BaseModel):
    router_choice: str
    router_reasoning: str
    web_search_used: bool
    hallucination_score: Optional[str]
    agents_executed: List[str]


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]
    metadata: QueryMetadata


class HealthResponse(BaseModel):
    status: str
    version: str


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# APP SETUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app = FastAPI(
    title="Diabetes Nursing GraphRAG API",
    description="Agentic RAG system for Type 2 Diabetes nursing care",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify Streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Main endpoint: Submit a nursing question and receive an evidence-based answer.

    The workflow:
    1. Router decides retrieval strategy (graph/vector/both/real_time)
    2. Retrieves evidence from Neo4j KG, Qdrant vector DB, and/or web
    3. Evidence gate checks if local knowledge is sufficient
    4. Synthesizes coherent answer with citations
    5. Grades for hallucinations and refines if needed
    6. Adds numbered citations for traceability
    """
    try:
        # Invoke LangGraph workflow
        result = workflow.invoke({"question": request.question})

        # Extract agents that were executed (trace the path)
        agents_executed = _extract_executed_agents(result)

        # Build response
        return QueryResponse(
            question=request.question,
            answer=result.get("final_answer", ""),
            citations=[
                Citation(id=c["id"], source_type=c["source_type"], label=c["label"])
                for c in (result.get("citations") or [])
            ],
            metadata=QueryMetadata(
                router_choice=result.get("router_choice", "unknown"),
                router_reasoning=result.get("router_reasoning", ""),
                web_search_used=result.get("web_search_used", False),
                hallucination_score=result.get("hallucination_score"),
                agents_executed=agents_executed,
            ),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Workflow execution failed: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint - redirects to docs."""
    return {
        "message": "Diabetes Nursing GraphRAG API",
        "docs": "/docs",
        "health": "/health",
        "query": "POST /query",
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _extract_executed_agents(result: Dict[str, Any]) -> List[str]:
    """Infer which agents were executed based on state."""
    agents = ["router"]  # Always starts with router

    choice = result.get("router_choice", "")

    # Add retrievers based on router choice
    if choice == "graph":
        agents.append("graph_retriever")
    elif choice == "vector":
        agents.append("vector_retriever")
    elif choice == "both":
        agents.extend(["graph_retriever", "vector_retriever"])
    elif choice == "real_time":
        agents.append("web_search")

    # Evidence gate always runs (unless real_time path)
    if choice != "real_time":
        agents.append("evidence_gate")

        # If web search was triggered by evidence gate
        if result.get("web_search_used"):
            agents.append("web_search")

    # Synthesis pipeline always runs
    agents.extend(["synthesizer", "hallucination_grader"])

    # Refiner only if hallucination detected
    if result.get("hallucination_score") == "hallucinated":
        agents.append("refiner")

    # Citation agent always runs last
    agents.append("citation_agent")

    return agents


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RUN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
