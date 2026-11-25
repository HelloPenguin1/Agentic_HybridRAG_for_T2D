
"""Code-based evaluators. Missing output fields return None, not KeyError."""

import re

NEEDS = {
    "graph": {"graph"},
    "vector": {"vector"},
    "both": {"graph", "vector"},
}


def unavailable(*keys):
    return {
        "results": [
            {"key": key, "score": None}
            for key in keys
        ]
    }


def router_metrics(outputs: dict, reference_outputs: dict) -> dict:
    if not isinstance(outputs, dict):
        return unavailable("router_exact", "router_under", "router_over")

    if not outputs.get("adaptive") or not outputs.get("router_choice"):
        return unavailable("router_exact", "router_under", "router_over")

    gold_route = (reference_outputs or {}).get("gold_route")

    if gold_route not in NEEDS:
        return unavailable("router_exact", "router_under", "router_over")

    need = NEEDS[gold_route]
    got = NEEDS.get(outputs.get("router_choice"), set())

    return {
        "results": [
            {"key": "router_exact", "score": int(got == need)},
            {"key": "router_under", "score": int(not need <= got)},
            {"key": "router_over", "score": int(bool(got - need))},
        ]
    }


def cypher_metrics(outputs: dict, reference_outputs: dict) -> dict:
    if not isinstance(outputs, dict):
        return unavailable("cypher_generated", "cypher_records")

    choice = outputs.get("router_choice")

    if choice not in ("graph", "both"):
        return unavailable("cypher_generated", "cypher_records")

    cypher = outputs.get("generated_cypher")
    n_graph = outputs.get("n_graph")

    if cypher is None and n_graph is None:
        return unavailable("cypher_generated", "cypher_records")

    return {
        "results": [
            {
                "key": "cypher_generated",
                "score": int(bool(cypher) and cypher != "No query generated"),
            },
            {
                "key": "cypher_records",
                "score": int((n_graph or 0) > 0),
            },
        ]
    }


def checker_flagged(outputs: dict, reference_outputs: dict) -> dict:
    if not isinstance(outputs, dict):
        return {"key": "checker_flagged", "score": None}

    score = outputs.get("hallucination_score")

    return {
        "key": "checker_flagged",
        "score": None if score is None else int(score == "hallucinated"),
    }


def citation_metrics(outputs: dict, reference_outputs: dict) -> dict:
    if not isinstance(outputs, dict) or not outputs.get("n_evidence"):
        return unavailable("citation_validity", "citation_coverage")
    ids = {str(c["id"]) for c in outputs.get("citations") or []}
    body = re.split(r"#*\s*\**\s*References", outputs.get("final_answer") or "", flags=re.I)[0]
    body = re.sub(r"([.!?])\s+((?:\(\d+\)\s*)+)", r"\2\1 ", body)  # glue ". (1)" back to its sentence
    cited = re.findall(r"\((\d+)\)", body)
    units = [u for u in re.split(r"(?<=[.!?])\s+|\n+", body)
             if len(u.split()) > 4 and "not found" not in u.lower()]
    cov = sum(bool(re.search(r"\(\d+\)", u)) for u in units) / len(units) if units else None
    return {"results": [
        {"key": "citation_validity", "score": sum(c in ids for c in cited) / len(cited) if cited else 0.0},
        {"key": "citation_coverage", "score": cov}]}


EVALUATORS = [
    router_metrics,
    cypher_metrics,
    checker_flagged,
    citation_metrics,
]
