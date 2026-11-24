
"""
Run LangSmith evaluations with controlled request pacing.

Examples:
    python 4_Evaluation2/run_eval.py adaptive_lite
    python 4_Evaluation2/run_eval.py adaptive_lite --delay-seconds 45
    python 4_Evaluation2/run_eval.py vector_only graph_only --reps 3
"""

import argparse
import subprocess   
import time
from functools import wraps

from langsmith import evaluate

from arms import ARMS, make_target
from evaluators import EVALUATORS

DATASET = "Always_Both_Full_pt-2"


def add_inter_run_delay(target, delay_seconds=30):
    """Pause between successive examples, not before the first example."""

    first_call = True

    @wraps(target)
    def wrapped(inputs: dict):
        nonlocal first_call

        if not first_call and delay_seconds > 0:
            print(
                f"\n[RateLimit] Waiting {delay_seconds}s "
                "before processing the next question..."
            )
            time.sleep(delay_seconds)

        first_call = False
        return target(inputs)

    return wrapped


p = argparse.ArgumentParser()

p.add_argument(
    "arms",
    nargs="*",
    default=[x for x in ARMS if x != "final_workflow"],
)

p.add_argument("--reps", type=int, default=1)
p.add_argument("--dataset", default=DATASET)
p.add_argument("--delay-seconds", type=float, default=30)

a = p.parse_args()

sha = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    capture_output=True,
    text=True,
).stdout.strip()


for i, arm in enumerate(a.arms):

    print(f"\nRunning: {arm} ({i + 1}/{len(a.arms)})")

    target = add_inter_run_delay(
        make_target(arm),
        delay_seconds=a.delay_seconds,
    )

    evaluate(
        target,
        data=a.dataset,
        evaluators=EVALUATORS,
        experiment_prefix=arm,
        num_repetitions=a.reps,
        max_concurrency=1,
        blocking=True,
        metadata={
            "arm": arm,
            "git_sha": sha,
            **ARMS[arm],
        },
    )

    print(f"Completed: {arm}")

    # Additional cooldown between different evaluation configurations.
    if i < len(a.arms) - 1:
        print("[RateLimit] Waiting 60 seconds before the next evaluation arm.")
        time.sleep(60)
