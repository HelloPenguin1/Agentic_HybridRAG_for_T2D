"""Evaluation arms: one factory, seven configs. Web search/evidence gate only in `final_workflow`."""
import sys
from functools import cache
from pathlib import Path
import re
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

load_dotenv()

import time
from core.state import GraphState
from nodes.graph_retriever_chain import GraphRetrieverChain
from nodes.vector_retriever import VectorRetriever
from nodes.router import Router
from nodes.synthesizer import synthesizer, docs_to_text
from nodes.fallback_to_vector import fallback_to_vector
from nodes.hallucinaton_checker import hallucination_grader, hallucination_decision
from nodes.refiner_node import refiner_node
from nodes.citation_agent import citation_agent

ARMS = {
    #"no_rag":           dict(route="none"),  # LLM only, separate prompt
    #"vector_only":      dict(route="vector"),
    #"graph_only":       dict(route="graph"),
    #"always_both":      dict(route="both"),
    #"adaptive_lite":    dict(route="adaptive", fallback=True),
    #"adaptive_full":    dict(route="adaptive", fallback=True, checker=True),
    "always_both_full": dict(route="both", checker=True),
    #"final_workflow":   dict(route="adaptive", fallback=True, checker=True, web=True),  # = workflows/final_workflow.py
}
TARGETS = {"graph": ["graph_retriever"], "vector": ["vector_retriever"],
           "both": ["graph_retriever", "vector_retriever"]}


@cache  # heavy objects (Neo4j, Qdrant, BM25) are built once and shared across arms
def _retrievers():
    return GraphRetrieverChain(), VectorRetriever()


def build(route="adaptive", fallback=False, checker=False, web=False):
    gr, vr = _retrievers()
    g = StateGraph(GraphState)
    g.add_node("router", Router if route == "adaptive" else (lambda s: {"router_choice": route}))
    g.add_node("graph_retriever", gr.graph_retriever_node)
    g.add_node("vector_retriever", vr.vector_retriever_node)
    g.add_node("synthesizer", synthesizer)
    g.add_node("citation_agent", citation_agent)

    sink, targets = "synthesizer", dict(TARGETS)
    if web:  # evidence gate + Tavily, exactly as in final_workflow.py
        from nodes.evidence_gate import evidence_gate, evidence_gate_decision
        from nodes.web_search import web_search_node
        g.add_node("evidence_gate", evidence_gate)
        g.add_node("web_search", web_search_node)
        g.add_conditional_edges("evidence_gate", evidence_gate_decision, ["synthesizer", "web_search"])
        g.add_edge("web_search", "synthesizer")
        sink, targets["real_time"] = "evidence_gate", ["web_search"]

    g.add_edge(START, "router")
    # without web, real_time -> vector (choice is still logged for router metrics)
    g.add_conditional_edges("router", lambda s: targets.get(s["router_choice"], targets["vector"]),
                            ["graph_retriever", "vector_retriever"] + (["web_search"] if web else []))
    if fallback:  # empty graph + router "graph" -> vector; otherwise on to the sink
        g.add_conditional_edges("graph_retriever", fallback_to_vector,
                                {"vector_retriever": "vector_retriever", "evidence_gate": sink})
    else:
        g.add_edge("graph_retriever", sink)
    g.add_edge("vector_retriever", sink)

    if checker:
        g.add_node("grader", hallucination_grader)
        g.add_node("refiner", refiner_node)
        g.add_edge("synthesizer", "grader")
        g.add_conditional_edges("grader", hallucination_decision,
                                {"end": "citation_agent", "refiner": "refiner"})
        g.add_edge("refiner", "citation_agent")
    else:
        g.add_edge("synthesizer", "citation_agent")
    g.add_edge("citation_agent", END)
    return g.compile()


NO_RAG_PROMPT = ("You are a nursing assistant for Type 2 Diabetes care. Answer the question from your own "
                 "knowledge, concisely and clinically.\n\nQuestion: {question}")


def _no_rag_target():
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from config.settings import response_llm
    chain = ChatPromptTemplate.from_template(NO_RAG_PROMPT) | response_llm | StrOutputParser()

    def target(inputs: dict) -> dict:
        t0 = time.perf_counter()
        ans = chain.invoke({"question": inputs["question"]})
        return {"final_answer": ans, "draft_answer": ans, "context": "", "router_choice": None,
                "adaptive": False, "generated_cypher": None, "n_graph": 0, "hallucination_score": None,
                "citations": [], "web_search_used": False, "latency_s": round(time.perf_counter() - t0, 2)}
    return target


def make_target(arm):
    if arm == "no_rag":
        return _no_rag_target()
    cfg = ARMS[arm]
    app = build(**cfg)

    def target(inputs: dict) -> dict:
        t0 = time.perf_counter()
        state, draft = {}, None
        for chunk in app.stream({"question": inputs["question"]}, stream_mode="updates"):
            for node, upd in chunk.items():
                state.update(upd or {})
                if node == "synthesizer":
                    draft = upd["final_answer"]  # pre-refiner / pre-citation answer
        graph_ctx = docs_to_text(state.get("graph_raw_cntx") or [])
        return {
            "final_answer": state.get("final_answer", ""),
            "draft_answer": re.sub(r"\[[VGW]\d+\]", "", draft) if draft else draft,
            "context": f"VECTOR:\n{state.get('vector_result') or ''}\n\nGRAPH:\n{graph_ctx}"
                       f"\n\nWEB:\n{state.get('web_result') or ''}",
            "web_search_used": bool(state.get("web_search_used")),
            "router_choice": state.get("router_choice"),
            "adaptive": cfg["route"] == "adaptive",
            "generated_cypher": state.get("generated_cypher"),
            "n_graph": len(state.get("graph_raw_cntx") or []),
            "n_evidence": len(state.get("evidence") or []),
            "hallucination_score": state.get("hallucination_score"),
            "citations": state.get("citations") or [],
            "latency_s": round(time.perf_counter() - t0, 2),
        }

    return target
