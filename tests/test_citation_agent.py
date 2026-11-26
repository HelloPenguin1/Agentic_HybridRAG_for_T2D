
import pytest

from nodes.citation_agent import citation_agent


@pytest.fixture
def evidence():
    return [
        {
            "id": "G1",
            "source_type": "graph",
            "label": "DrugBank: Metformin",
            "text": "Metformin information"
        },
        {
            "id": "V4",
            "source_type": "vector",
            "label": "ADA Chapter 9",
            "text": "Glycemic management"
        },
        {
            "id": "W1",
            "source_type": "web",
            "label": "NIH Article",
            "text": "Clinical information",
            "url": "https://example.com"
        }
    ]


@pytest.mark.parametrize(
    "token",
    [
        "[G1]",
        "【G1】",
        "[\u200bG1]",
        "[G\u200b1]",
        "【\u200bG1】",
    ]
)
def test_citation_format_variations(token, evidence):

    result = citation_agent({
        "final_answer": f"Claim supported by evidence {token}.",
        "evidence": evidence
    })

    assert "Claim supported by evidence (1)." in result["final_answer"]
    assert result["citations"][0]["evidence_id"] == "G1"
    assert "### References" in result["final_answer"]


def test_multiple_sources(evidence):

    result = citation_agent({
        "final_answer": "First claim [G1]. Second claim 【V4】. Third claim [W1].",
        "evidence": evidence
    })

    assert "(1)" in result["final_answer"]
    assert "(2)" in result["final_answer"]
    assert "(3)" in result["final_answer"]

    assert [
        c["evidence_id"] for c in result["citations"]
    ] == ["G1", "V4", "W1"]


def test_repeated_citation_reuses_number(evidence):

    result = citation_agent({
        "final_answer": "First claim [G1]. Second claim 【G1】.",
        "evidence": evidence
    })

    # Separate answer body from References section.
    body, references = result["final_answer"].split(
        "\n\n### References", 1
    )

    # Both claims should reuse reference number 1.
    assert body.count("(1)") == 2

    # Only one unique evidence record should be generated.
    assert len(result["citations"]) == 1

    assert result["citations"][0]["evidence_id"] == "G1"

    # References section should contain exactly one entry.
    assert references.count("(1) G1 — DrugBank: Metformin") == 1


def test_unknown_evidence_id_is_dropped(evidence):

    result = citation_agent({
        "final_answer": "Claim [G99].",
        "evidence": evidence
    })

    assert result["citations"] == []
    assert "G99" not in result["final_answer"]


def test_native_token_is_removed(evidence):

    result = citation_agent({
        "final_answer": "Claim 【native_token】 [G1].",
        "evidence": evidence
    })

    assert "native_token" not in result["final_answer"]
    assert "(1)" in result["final_answer"]


def test_existing_references_section_is_replaced(evidence):

    result = citation_agent({
        "final_answer": (
            "Claim [G1].\n\n"
            "### References\n"
            "(old) Incorrect reference"
        ),
        "evidence": evidence
    })

    assert "Incorrect reference" not in result["final_answer"]
    assert "(1) G1 — DrugBank: Metformin" in result["final_answer"]
