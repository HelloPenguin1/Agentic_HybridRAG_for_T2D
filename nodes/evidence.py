"""Evidence index: every retrieved item gets a stable ID (V1.., G1.., W1..) that the synthesizer,
refiner and citation renderer all share, so citations are looked up by identity, never guessed."""


def docs_to_text(docs):  # re-exported by synthesizer.py (arms.py imports it from there)
    return "\n".join("; ".join(f"{k}: {v}" for k, v in row.items()) for row in docs)


def build_evidence(state) -> list[dict]:
    ev = []
    for i, d in enumerate(state.get("vector_docs") or [], 1):
        m = d.metadata or {}
        label = " > ".join(filter(None, [m.get("chapter_name"), m.get("section_heading"),
                                         m.get("subsection_heading")])) or "Unknown section"
        ev.append({"id": f"V{i}", "source_type": "vector", "label": f"ADA — {label}",
                   "text": d.page_content, "chapter_name": m.get("chapter_name", ""),
                   "section_heading": m.get("section_heading", ""), "page_number": m.get("page_number")})
    for i, rec in enumerate(state.get("graph_raw_cntx") or [], 1):
        text = "; ".join(f"{k}: {v}" for k, v in rec.items())
        ev.append({"id": f"G{i}", "source_type": "graph", "label": f"DrugBank KG — {text[:120]}", "text": text})
    for i, w in enumerate(state.get("web_docs") or [], 1):
        ev.append({"id": f"W{i}", "source_type": "web", "text": w.get("content", ""),
                   "label": f"{w.get('title', 'Web source')} — {w.get('url', '')}", "url": w.get("url", "")})
    return ev


def format_evidence(ev: list[dict]) -> str:
    return "\n\n".join(f"[{e['id']}] {e['label']}\n{e['text']}" for e in ev) or "No evidence was retrieved."