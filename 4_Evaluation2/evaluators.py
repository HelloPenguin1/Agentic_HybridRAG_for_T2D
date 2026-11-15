"""Code (non-LLM) evaluators. A score of None = metric not applicable for that run."""
import re

NEEDS = {"graph": {"graph"}, "vector": {"vector"}, "both": {"graph", "vector"}}


def router_metrics(outputs: dict, reference_outputs: dict) -> dict:
    if not outputs["adaptive"]:
        return {"results": [{"key": k, "score": None} for k in ("router_exact", "router_under", "router_over")]}
    need = NEEDS[reference_outputs["gold_route"]]
    got = NEEDS.get(outputs["router_choice"], set())  # real_time -> nothing retrieved -> under-routed
    return {"results": [
        {"key": "router_exact", "score": int(got == need)},
        {"key": "router_under", "score": int(not need <= got)},  # missing a needed source
        {"key": "router_over", "score": int(bool(got - need))},   # used an unneeded source
    ]}


def cypher_metrics(outputs: dict, reference_outputs: dict) -> dict:
    if outputs["router_choice"] not in ("graph", "both"):
        return {"results": [{"key": "cypher_generated", "score": None},
                            {"key": "cypher_records", "score": None}]}
    cy = outputs["generated_cypher"]
    return {"results": [
        {"key": "cypher_generated", "score": int(bool(cy) and cy != "No query generated")},
        {"key": "cypher_records", "score": int(outputs["n_graph"] > 0)},  # 1 - this = empty/failed rate
    ]}


def checker_flagged(outputs: dict, reference_outputs: dict) -> dict:
    s = outputs["hallucination_score"]  # only set in checker arms
    return {"key": "checker_flagged", "score": None if s is None else int(s == "hallucinated")}


def citation_validity(outputs: dict, reference_outputs: dict) -> dict:
    """Share of inline (n) markers that map to a real evidence id. Assumes '(n)' marker format."""
    ids = {c["id"] for c in outputs["citations"]}
    body = re.split(r"#*\s*\**\s*References", outputs["final_answer"], flags=re.I)[0]
    cited = re.findall(r"\((\d+)\)", body)
    if not ids or not cited:
        return {"key": "citation_validity", "score": None}
    return {"key": "citation_validity", "score": sum(c in ids for c in cited) / len(cited)}


EVALUATORS = [router_metrics, cypher_metrics, checker_flagged, citation_validity]
