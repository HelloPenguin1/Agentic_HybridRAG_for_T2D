from langsmith import evaluate
from eval_runners import (
    run_adaptive_router,
    run_fixed_hybrid,
    run_vector_only,
    run_graph_only,
)
from evaluators import (
    answer_correctness,
    faithfulness,
    completeness,
    router_accuracy,
    cypher_semantic_correctness,
    context_recall_evaluator,
    e2e_quality_evaluator,
)

# Evaluators for all workflows
common_evaluators = [
    answer_correctness,
    faithfulness,
    completeness,
    router_accuracy,  # Always include (sanity check for fixed workflows)
    e2e_quality_evaluator,
]

# Additional evaluators for workflows that use graph
graph_evaluators = [
    cypher_semantic_correctness,
    context_recall_evaluator,
    e2e_quality_evaluator,
]
import time

# 1. Adaptive Router (uses both graph + vector)
# evaluate(
#     run_adaptive_router,
#     data="diabetes-mixed-eval-v1",
#     evaluators=common_evaluators + graph_evaluators,
#     experiment_prefix="adaptive-router",
#     max_concurrency=1,
#     blocking=True
# )
# # time.sleep(60)
# 2. Fixed Hybrid (uses both graph + vector)
# evaluate(
#     run_fixed_hybrid,
#     data="diabetes-mixed-eval-v1",
#     evaluators=common_evaluators + graph_evaluators,
#     experiment_prefix="fixed-hybrid",
#     max_concurrency=1
# )
# # time.sleep(60)

# # 3. Vector Only
evaluate(
    run_vector_only,
    data="diabetes-mixed-eval-v1",
    evaluators=common_evaluators,  # No graph evaluators
    experiment_prefix="vector-only",
    max_concurrency=1,
    blocking=True,
)
time.sleep(60)
# 4. Graph Only
# evaluate(
#     run_graph_only,
#     data="diabetes-mixed-eval-v1",
#     evaluators=common_evaluators + graph_evaluators,
#     experiment_prefix="graph-only",
#     max_concurrency=1,
#     blocking=True
# )
