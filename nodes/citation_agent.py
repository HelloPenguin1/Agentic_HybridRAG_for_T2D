"""
Citation renderer (deterministic, no LLM).

Responsibilities:
1. Normalize citation formatting variations.
2. Resolve inline evidence IDs against the existing evidence index.
3. Assign sequential reference numbers in order of first use.
4. Build the final References section and structured citation metadata.

The synthesizer/refiner remains responsible for selecting the correct
evidence ID for each claim. This node trusts that attribution.
"""

import re

# ------------------------------------------------------------
# 1. PATTERNS
# ------------------------------------------------------------

# Matches ASCII and fullwidth citation brackets.
# Captures any preceding horizontal whitespace so formatting is preserved.
BRACKET_TOKEN = re.compile(
    r"(?P<leading>[ \t]*)(?:\[(?P<ascii>[^\]]*)\]|【(?P<fullwidth>[^】]*)】)"
)

# Common invisible Unicode formatting characters.
# These are removed only while examining a bracketed token.
ZERO_WIDTH_CHARS = re.compile(r"[\u200B\u200C\u200D\u2060\uFEFF]")

# Valid evidence IDs and comma/semicolon-separated ID lists.
ID_LIST = re.compile(r"[VGW]\d+(?:\s*[,;]\s*[VGW]\d+)*")

# Matches canonical ASCII citation tags after normalization.
TAG = re.compile(r"[ \t]*\[\s*([VGW]\d+(?:\s*[,;]\s*[VGW]\d+)*)\s*\]")

# Removes an existing References section to prevent duplication.
REFS = re.compile(r"^[ \t]*#{0,6}[ \t]*\**[ \t]*References\b.*\Z", re.IGNORECASE | re.MULTILINE | re.DOTALL)


# ------------------------------------------------------------
# 2. CITATION TOKEN NORMALIZATION
# ------------------------------------------------------------


def normalize_citation_tokens(text: str) -> str:
    """
    Converts recognized citation tokens into canonical ASCII format.

    Supported:
        [G1]       -> [G1]
        【G1】      -> [G1]
        [\u200bV4]      -> [V4]  (U+200B removed)
        [G1, V2]   -> [G1, V2]
        【native】   -> removed

    Ordinary, unrecognized ASCII bracketed text is preserved.
    """

    def normalize(match: re.Match) -> str:

        # Determine which bracket format was encountered.
        raw = (
            match.group("ascii")
            if match.group("ascii") is not None
            else match.group("fullwidth")
        )

        # Remove invisible characters inside the token only.
        cleaned = ZERO_WIDTH_CHARS.sub("", raw).strip()

        # If it contains valid evidence IDs, convert to ASCII brackets.
        if ID_LIST.fullmatch(cleaned):
            return f"{match.group('leading')}[{cleaned}]"

        # Preserve existing behaviour:
        # Unrecognized fullwidth/native model citation tokens are removed.
        if match.group("fullwidth") is not None:
            return ""

        # Leave ordinary ASCII bracketed text untouched.
        return match.group(0)

    return BRACKET_TOKEN.sub(normalize, text)


# ------------------------------------------------------------
# 3. MAIN CITATION AGENT
# ------------------------------------------------------------


def citation_agent(state):

    # Original evidence index created by build_evidence().
    # No retrieval is performed here.
    ev = {e["id"]: e for e in (state.get("evidence") or [])}

    # Obtain draft/final answer.
    raw_answer = state.get("final_answer") or ""

    # Remove any existing References section.
    answer = REFS.sub("", raw_answer).rstrip()

    # Normalize Unicode citation variations before extraction.
    answer = normalize_citation_tokens(answer)

    # Maps evidence IDs to sequential reference numbers.
    order: dict[str, int] = {}

    # --------------------------------------------------------
    # 4. RESOLVE AND NUMBER CITATIONS
    # --------------------------------------------------------

    def number(match: re.Match) -> str:

        # Extract individual IDs from a single citation tag.
        ids = re.split(r"\s*[,;]\s*", match.group(1))

        # Keep only IDs present in the original evidence index.
        valid_ids = [evidence_id for evidence_id in ids if evidence_id in ev]

        # Unknown/invented IDs are dropped.
        if not valid_ids:
            return ""

        # Assign numbers according to first occurrence.
        numbers = [
            order.setdefault(evidence_id, len(order) + 1) for evidence_id in valid_ids
        ]

        return " " + "".join(f"({n})" for n in numbers)

    body = TAG.sub(number, answer)

    # --------------------------------------------------------
    # 5. HANDLE EMPTY CITATION CASE
    # --------------------------------------------------------

    if not order:
        return {"final_answer": body, "citations": []}

    # --------------------------------------------------------
    # 6. BUILD STRUCTURED CITATION METADATA
    # --------------------------------------------------------

    cites = [
        {
            "id": str(number),
            "evidence_id": evidence_id,
            **{key: value for key, value in ev[evidence_id].items() if key != "id"},
        }
        for evidence_id, number in order.items()
    ]

    # --------------------------------------------------------
    # 7. BUILD REFERENCES SECTION
    # --------------------------------------------------------

    refs = "\n".join(f"({c['id']}) {c['evidence_id']} — {c['label']}" for c in cites)

    final_answer = f"{body}\n\n### References\n{refs}"

    return {"final_answer": final_answer, "citations": cites}
