
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
