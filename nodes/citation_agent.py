"""
Citation Agent Node — Traceability layer for Agentic GraphRAG.

Reads the final answer + raw evidence from state, builds a numbered
evidence index with nurse-friendly source labels, and asks the LLM
to annotate each sentence with [V1], [G1] etc. citation tags.
"""

from langchain_core.output_parsers import StrOutputParser
from config.settings import response_llm
from config.prompts2 import citation_prompt


def _build_vector_evidence(vector_docs: list) -> tuple[list[dict], list[str]]:
    """Build numbered evidence entries from vector Document objects.

    Returns (citations_list, formatted_lines) where each line is like:
      [V1] ADA — "Chapter Name" > "Section", p.3
           Content: "first 200 chars of chunk..."
    """
    citations = []
    lines = []
    for i, doc in enumerate(vector_docs, start=1):
        meta = getattr(doc, "metadata", {})
        tag = f"V{i}"

        # Build label from metadata: chapter > section > subsection, p.N
        parts = []
        if meta.get("chapter_name"):
            parts.append(f'"{meta["chapter_name"]}"')
        if meta.get("section_heading"):
            parts.append(f'> "{meta["section_heading"]}"')
        if meta.get("subsection_heading"):
            parts.append(f'> "{meta["subsection_heading"]}"')
        if meta.get("page_number"):
            parts.append(f"p.{meta['page_number']}")

        label = "ADA — " + ", ".join(parts) if parts else "ADA — Unknown Section"
        snippet = (doc.page_content[:200] + "...") if len(doc.page_content) > 200 else doc.page_content

        citations.append({
            "id": tag,
            "source_type": "vector",
            "label": label,
            "chapter_name": meta.get("chapter_name", ""),
            "section_heading": meta.get("section_heading", ""),
            "page_number": meta.get("page_number"),
        })
        lines.append(f"[{tag}] {label}\n     Content: \"{snippet}\"")

    return citations, lines


def _build_graph_evidence(graph_docs: list, generated_cypher: str | None) -> tuple[list[dict], list[str]]:
    """Build numbered evidence entries from Neo4j result dicts.

    Returns (citations_list, formatted_lines) where each line is like:
      [G1] DrugBank KG — (Metformin)-[:INTERACTS_WITH]->(Glipizide)
           Query: MATCH (d1:Drug)-[r:INTERACTS_WITH]-(d2:Drug)...
    """
    citations = []
    lines = []

    cypher_display = generated_cypher.strip() if generated_cypher else "N/A"

    for i, record in enumerate(graph_docs, start=1):
        tag = f"G{i}"
        record_str = str(record)
        summary = (record_str[:250] + "...") if len(record_str) > 250 else record_str

        label = f"DrugBank KG — Result: {summary}"

        citations.append({
            "id": tag,
            "source_type": "graph",
            "label": label,
            "cypher_query": cypher_display,
        })
        lines.append(f"[{tag}] {label}\n     Cypher: {cypher_display}")

    return citations, lines


def citation_agent(state):
    """LangGraph node: annotates the final answer with formal citations."""
    print("---CITATION AGENT: ADDING TRACEABILITY---")

    final_answer = state.get("final_answer", "")
    vector_docs = state.get("vector_docs") or []
    graph_docs = state.get("graph_docs") or []
    generated_cypher = state.get("generated_cypher")

    # ── Build evidence index ────────────────────────────────────────
    v_citations, v_lines = _build_vector_evidence(vector_docs)
    g_citations, g_lines = _build_graph_evidence(graph_docs, generated_cypher)

    all_citations = v_citations + g_citations
    all_lines = v_lines + g_lines

    # If there is no evidence at all just return unchanged
    if not all_lines:
        print("[CitationAgent] No evidence docs found — skipping citation.")
        return {"final_answer": final_answer, "citations": []}

    evidence_index = "\n\n".join(all_lines)

    # ── Run citation LLM ────────────────────────────────────────────
    chain = citation_prompt | response_llm | StrOutputParser()

    cited_answer = chain.invoke({
        "evidence_index": evidence_index,
        "final_answer": final_answer,
    })

    return {"final_answer": cited_answer, "citations": all_citations}
