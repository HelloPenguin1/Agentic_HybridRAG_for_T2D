# Citation fix: remaining edits (5 small ones)

Drop-in replacements are in `nodes/`: `evidence.py` (new), `synthesizer.py`, `refiner_node.py`, `citation_agent.py`.
The node name `citation_agent` is unchanged, so `arms.py` graph wiring needs no changes.

## 1. `config/prompts2.py` — `base_synthesizer_prompt`

Replace the whole `**RETRIEVED INFORMATION:**` block (the three `{graph_raw_cntx}` / `{vector_result}` / `{web_result}` sections) with:

```
    **RETRIEVED EVIDENCE** (each item has an ID: V = ADA guideline, G = DrugBank record, W = web):
    {evidence}

    **CITATION RULES:**
    - End every sentence or bullet that states a fact with the ID(s) of the evidence
      item(s) that directly contain it, in square brackets: [V2] or [G1][V3].
    - Cite only IDs listed above, and only the item that actually contains the fact.
      Never invent IDs.
    - Use only this ASCII bracket format. Never use 【】, footnote numbers, or a
      References section; references are added automatically.
    - Do not cite a sentence that says information was not found.
```

and add to **FINAL CHECK**: "every factual sentence ends with a valid evidence ID."

## 2. `config/prompts2.py` — `refiner_prompt`

Human message, last line. Replace
`Please provide the refined Final Answer with citations as before. `
with
`Return the refined answer. Keep the [V#]/[G#]/[W#] tags of claims that remain supported; add tags only for IDs in the evidence.`

## 3. `core/state.py`

```python
    evidence: Optional[List[dict]]  # ID-tagged evidence index built by the synthesizer
```

## 4. `evals/arms.py` — in the dict returned by `target`

```python
            "n_evidence": len(state.get("evidence") or []),
```
(`draft_answer` now holds raw `[V1]` tags. Strip with `re.sub(r"\[[VGW]\d+\]", "", draft)` before judging it.)

## 5. `evaluators.py` — replace `citation_validity`

`citations[]` now contains only sources actually cited, so "evidence retrieved but nothing cited" must be scored 0 instead of None.

```python
def citation_validity(outputs: dict, reference_outputs: dict) -> dict:
    """Every inline (n) resolves to a citations[] entry; 0.0 if evidence was retrieved but nothing was cited."""
    if not isinstance(outputs, dict) or not outputs.get("n_evidence"):
        return {"key": "citation_validity", "score": None}
    ids = {str(c["id"]) for c in outputs.get("citations") or []}
    body = re.split(r"#*\s*\**\s*References", outputs.get("final_answer") or "", flags=re.I)[0]
    cited = re.findall(r"\((\d+)\)", body)
    return {"key": "citation_validity", "score": sum(c in ids for c in cited) / len(cited) if cited else 0.0}
```

Remove the mojibake patterns I added earlier (`_GRAPH_OPEN` etc.); they were chasing a marker the synthesizer was leaking, not a real citation format.

## Optional: `config/settings.py`

`response_llm` has `max_tokens=1024`. On gpt-oss the reasoning tokens share that budget, so long answers plus tags can truncate. Consider 2048.
