"""Usage (from repo root):
    python 4_Evaluation2/run_eval.py adaptive_lite            # one arm, pilot
    python 4_Evaluation2/run_eval.py                          # main arms (web-enabled final_workflow excluded)
    python 4_Evaluation2/run_eval.py vector_only graph_only --reps 3
"""
import argparse
import subprocess

from langsmith import evaluate

from arms import ARMS, make_target
from evaluators import EVALUATORS

DATASET = "Combined_Dataset_For_E2E_v1"  # <- your LangSmith dataset name

p = argparse.ArgumentParser()
p.add_argument("arms", nargs="*", default=[x for x in ARMS if x != "final_workflow"])  # web arm: run by name
p.add_argument("--reps", type=int, default=1)
p.add_argument("--dataset", default=DATASET)
a = p.parse_args()

sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()

for arm in a.arms:
    evaluate(
        make_target(arm),
        data=a.dataset,
        evaluators=EVALUATORS,  # add LangSmith UI judges on the dataset/experiments separately
        experiment_prefix=arm,
        num_repetitions=a.reps,
        max_concurrency=2,
        metadata={"arm": arm, "git_sha": sha, **ARMS[arm]},
    )
