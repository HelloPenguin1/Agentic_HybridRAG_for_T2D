"""
Citation Agent Node — Traceability layer for Agentic GraphRAG.

Reads the final answer + raw evidence from state, builds a numbered
evidence index with nurse-friendly source labels, and asks the LLM
to annotate each sentence with [V1], [G1] etc. citation tags.
"""

import re
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

        # Build label: chapter > section > subsection
        parts = []
        if meta.get("chapter_name"):
            parts.append(f'"{meta["chapter_name"]}"')
        if meta.get("section_heading"):
            parts.append(f'> "{meta["section_heading"]}"')
        if meta.get("subsection_heading"):
            parts.append(f'> "{meta["subsection_heading"]}"')

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


def _extract_relationship(cypher: str) -> str:
    """Extract the relationship type from a Cypher query, e.g. 'INTERACTS_WITH'."""
    match = re.search(r'\[:(\w+)', cypher)
    return match.group(1) if match else "RELATED_TO"


def _extract_drug_name(cypher: str) -> str:
    """Extract the primary drug name from toLower(...) CONTAINS toLower('name')."""
    match = re.search(r"toLower\(['\"](.+?)['\"]\)", cypher)
    return match.group(1).title() if match else "Unknown Drug"


def _format_record_as_path(record: dict, drug_name: str, relationship: str) -> str:
    """Format a Neo4j result dict as a human-readable relationship path.

    Examples:
      Acarbose —[BELONGS_TO]→ Oral Hypoglycemics
      Metformin —[INTERACTS_WITH]→ Glipizide: "may increase risk"
    """
    values = list(record.values())
    if not values:
        return f"{drug_name} —[{relationship}]→ (empty result)"

    target = str(values[0])
    detail = ""
    if len(values) > 1 and values[1]:
        detail = f': "{values[1]}"'

    return f"{drug_name} —[{relationship}]→ {target}{detail}"


def _build_graph_evidence(graph_docs: list, generated_cypher: str | None) -> tuple[list[dict], list[str]]:
    """Build numbered evidence entries from Neo4j result dicts.

    Uses relationship-path format for nurse-friendly citations:
      [G1] DrugBank KG — Acarbose —[BELONGS_TO]→ Oral Hypoglycemics
    """
    citations = []
    lines = []

    if not graph_docs:
        return citations, lines

    relationship = _extract_relationship(generated_cypher) if generated_cypher else "RELATED_TO"
    drug_name = _extract_drug_name(generated_cypher) if generated_cypher else "Unknown Drug"

    for i, record in enumerate(graph_docs, start=1):
        tag = f"G{i}"
        path = _format_record_as_path(record, drug_name, relationship)
        label = f"DrugBank KG — {path}"

        citations.append({
            "id": tag,
            "source_type": "graph",
            "label": label,
            "relationship": relationship,
            "drug_name": drug_name,
        })
        lines.append(f"[{tag}] {label}")

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
