
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


def checker_flagged(outputs: dict, reference_outputs: dict) -> dict:
    if not isinstance(outputs, dict):
        return {"key": "checker_flagged", "score": None}

    score = outputs.get("hallucination_score")

    return {
        "key": "checker_flagged",
        "score": None if score is None else int(score == "hallucinated"),
    }


def citation_validity(outputs: dict, reference_outputs: dict) -> dict:
    if not isinstance(outputs, dict):
        return {"key": "citation_validity", "score": None}

    citations = outputs.get("citations") or []
    final_answer = outputs.get("final_answer") or ""

    ids = {str(c["id"]) for c in citations if isinstance(c, dict) and "id" in c}

    if not ids:
        return {"key": "citation_validity", "score": None}

    body = re.split(
        r"#*\s*\**\s*References",
        final_answer,
        flags=re.I,
    )[0]

    cited = re.findall(r"\((\d+)\)", body)

    # Evidence exists but the answer contains no inline citations.
    if not cited:
        return {"key": "citation_validity", "score": 0.0}

    return {
        "key": "citation_validity",
        "score": sum(c in ids for c in cited) / len(cited),
    }


EVALUATORS = [
    router_metrics,
    cypher_metrics,
    checker_flagged,
    citation_validity,
]
