"""Citation renderer (deterministic, no LLM). Turns the synthesizer's inline [V1]/[G2] tags into
numbered (1)(2) refs in order of first use and builds the References block from evidence metadata."""
import re

TAG = re.compile(r"[ \t]*\[\s*([VGW]\d+(?:\s*[,;]\s*[VGW]\d+)*)\s*\]")
NATIVE = re.compile(r"[ \t]*【[^】]*】")  # citation tokens gpt-oss emits on its own; they map to nothing
REFS = re.compile(r"^[ \t]*#{0,6}[ \t]*\**[ \t]*References\b.*\Z", re.I | re.M | re.S)


def citation_agent(state):
    ev = {e["id"]: e for e in state.get("evidence") or []}
    answer = REFS.sub("", NATIVE.sub("", state.get("final_answer", ""))).rstrip()
    order: dict[str, int] = {}

    def number(m):  # unknown (invented) IDs are dropped; known ones are numbered by first use
        ids = [i for i in re.split(r"\s*[,;]\s*", m.group(1)) if i in ev]
        return " " + "".join(f"({order.setdefault(i, len(order) + 1)})" for i in ids) if ids else ""

    body = TAG.sub(number, answer)
    if not order:  # nothing cited: say so honestly instead of attaching an unrelated evidence list
        return {"final_answer": body, "citations": []}

    cites = [{"id": str(n), "evidence_id": e, **{k: v for k, v in ev[e].items() if k != "id"}}
             for e, n in order.items()]
    refs = "\n".join(f"({c['id']}) {c['evidence_id']} — {c['label']}" for c in cites)
    return {"final_answer": f"{body}\n\n### References\n{refs}", "citations": cites}