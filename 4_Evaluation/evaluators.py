"""
evaluators.py
LangSmith-compatible evaluator functions for the T2D Agentic GraphRAG system.

Each evaluator follows the LangSmith signature:
    fn(run, example) -> EvaluationResult

Usage:
    from langsmith import evaluate
    from 6_Evaluation.eval_runners import run_adaptive_router
    from 6_Evaluation.evaluators import (
        answer_correctness, faithfulness, completeness, router_accuracy
    )

    evaluate(
        run_adaptive_router,
        data="diabetes-nursing-qa-v1",
        evaluators=[answer_correctness, faithfulness, completeness, router_accuracy],
        experiment_prefix="adaptive-router",
    )

Note: latency is tracked automatically by LangSmith — no evaluator needed.
"""

import os
from openai import OpenAI
from langsmith.schemas import Run, Example
from langsmith.evaluation import EvaluationResult
from langsmith.evaluation import EvaluationResult, run_evaluator
from config.settings import response_llm 
from config.output_validation import GRAPH_EMPTY

_oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _llm_score(prompt: str, low: int = 0, high: int = 5) -> float:
    """Call GPT-4o-mini, parse an integer score from the response, normalize to 0-1."""
    resp = _oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=8,
    )
    raw = resp.choices[0].message.content.strip()
    try:
        score = int("".join(c for c in raw if c.isdigit() or c == ".").split(".")[0])
        return max(0.0, min(1.0, (score - low) / (high - low)))
    except (ValueError, ZeroDivisionError):
        return 0.0


# ── 1. Answer Correctness ─────────────────────────────────────────────────────
def answer_correctness(run: Run, example: Example) -> EvaluationResult:
    """
    LLM-as-Judge: how factually correct is the system answer vs ground truth?
    Scale 0-5, normalized to 0-1.
    """
    question      = example.inputs["question"]
    ground_truth  = example.outputs["ground_truth_answer"]
    system_answer = run.outputs.get("final_answer", "")

    prompt = f"""You are a clinical accuracy evaluator for a Type 2 Diabetes Q&A system.

Question: {question}

Ground Truth Answer:
{ground_truth}

System Answer:
{system_answer}

Rate the FACTUAL CORRECTNESS of the System Answer compared to the Ground Truth on a scale of 0-5:
  5 = Completely correct, all key facts match
  4 = Mostly correct, minor omissions or slight inaccuracies
  3 = Partially correct, some facts right but missing important information
  2 = Mostly incorrect or significantly inaccurate
  1 = Almost entirely wrong
  0 = Completely wrong or no answer

Reply with a SINGLE integer (0-5) only."""

    score = _llm_score(prompt, low=0, high=5)
    return EvaluationResult(key="correctness", score=score)


# ── 2. Faithfulness / Groundedness ───────────────────────────────────────────
def faithfulness(run: Run, example: Example) -> EvaluationResult:
    """
    LLM-as-Judge: are the claims in the system answer supported by retrieved context?
    Scale 0-5, normalized to 0-1. Catches hallucinations.
    """
    system_answer     = run.outputs.get("final_answer", "")
    retrieved_context = run.outputs.get("retrieved_context", "")

    if not retrieved_context:
        # No context retrieved — treat all claims as unsupported
        return EvaluationResult(key="faithfulness", score=0.0)

    prompt = f"""You are evaluating whether a clinical answer is grounded in the retrieved evidence.

Retrieved Context:
{retrieved_context[:3000]}

System Answer:
{system_answer}

Rate FAITHFULNESS — how well every claim in the System Answer is supported by the Retrieved Context:
  5 = Every claim is directly supported by the context
  4 = Most claims supported, one minor unsupported detail
  3 = Some claims supported, but notable unsupported statements
  2 = Many claims cannot be verified from the context
  1 = Most claims appear hallucinated or contradicted
  0 = Answer is entirely fabricated / not grounded at all

Reply with a SINGLE integer (0-5) only."""

    score = _llm_score(prompt, low=0, high=5)
    return EvaluationResult(key="faithfulness", score=score)


# ── 3. Completeness ───────────────────────────────────────────────────────────
def completeness(run: Run, example: Example) -> EvaluationResult:
    """
    LLM-as-Judge: does the system answer cover all key points from the ground truth?
    Scale 0-5, normalized to 0-1.
    """
    question      = example.inputs["question"]
    ground_truth  = example.outputs["ground_truth_answer"]
    system_answer = run.outputs.get("final_answer", "")

    prompt = f"""You are evaluating the completeness of a clinical answer for nurses.

Question: {question}

Ground Truth Answer (all key information a complete answer should cover):
{ground_truth}

System Answer:
{system_answer}

Rate COMPLETENESS — how much of the critical clinical information from the Ground Truth is present:
  5 = All key facts, doses, criteria, or steps are covered
  4 = Most key information present, one minor point missing
  3 = Core information present but missing notable detail
  2 = Only partially complete, significant gaps
  1 = Barely touches the answer, most information missing
  0 = Completely missing or irrelevant

Reply with a SINGLE integer (0-5) only."""

    score = _llm_score(prompt, low=0, high=5)
    return EvaluationResult(key="completeness", score=score)


# ── 4. Router Accuracy ────────────────────────────────────────────────────────
def router_accuracy(run: Run, example: Example) -> EvaluationResult:
    """
    Exact match: did the system route to the expected retrieval strategy?
    Score: 1.0 = correct, 0.0 = incorrect.
    For fixed-route configs (vector_only, graph_only, fixed_hybrid) this will
    always match if their expected_route aligns — useful as a sanity check,
    but most meaningful for the adaptive router config.
    """
    expected = example.outputs.get("expected_route", "").strip().lower()
    actual   = run.outputs.get("router_choice", "").strip().lower()

    score = 1.0 if (expected and actual and expected == actual) else 0.0
    return EvaluationResult(key="router_accuracy", score=score)


@run_evaluator
def cypher_quality_evaluator(run: Run, example: Example) -> EvaluationResult:
    """Checks the actual Cypher syntax and intent."""
    # Retrieve from the new state field
    query = run.outputs.get("generated_cypher", "")
    
    if not query:
        return EvaluationResult(key="cypher_quality", score=0.0)

    prompt = f"""You are a Neo4j Cypher expert. 
    Evaluate this generated query for a Type 2 Diabetes database:
    Query: {query}

    Score (0-5):
    5: Perfect syntax, uses correct relationship types (e.g., HAS_GENE, TREATS), and is efficient.
    3: Correct syntax but might be missing a filter or using a slightly suboptimal path.
    1: Valid Cypher but logically wrong for the question.
    0: Syntax error or completely hallucinated labels.

    Reply with a SINGLE integer only."""
    
    return EvaluationResult(key="cypher_quality", score=_llm_judge_score(prompt))


@run_evaluator
def context_recall_evaluator(run, example) -> EvaluationResult:
    """Tier 2: Checks if retrieved graph data contains the ground truth facts."""
    ground_truth = example.outputs.get("ground_truth_context", "")
    retrieved = run.outputs.get("graph_result", "")
    
    prompt = f"""Ground truth facts: {ground_truth}
    Retrieved facts: {retrieved}
    Does the Retrieved facts text contain the core clinical information found in the Ground truth facts? 
    Answer only with YES or NO."""
    
    res = response_llm.invoke(prompt).content.strip().upper()
    return EvaluationResult(key="context_recall", score=1 if "YES" in res else 0)



@run_evaluator
def e2e_quality_evaluator(run, example) -> EvaluationResult:
    """Tier 3: Scores the final generated answer against the golden answer (1-5)."""
    gt_answer = example.outputs.get("ground_truth_answer", "")
    final_answer = run.outputs.get("final_answer", "")
    
    prompt = f"""Golden Answer: {gt_answer}
    Agent Answer: {final_answer}
    Rate the Agent Answer from 1 to 5 based on clinical accuracy and alignment with the Golden Answer. 
    Output ONLY the integer (e.g., 4)."""
    
    try:
        res = response_llm.invoke(prompt).content.strip()
        score = int(res) / 5.0  # Normalize to a 0.0 - 1.0 scale for LangSmith
    except ValueError:
        score = 0.0
        
    return EvaluationResult(key="e2e_quality", score=score)