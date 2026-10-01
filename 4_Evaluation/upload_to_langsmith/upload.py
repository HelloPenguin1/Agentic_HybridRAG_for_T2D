from langsmith import Client
import json

client = Client()

# Create dataset
client.create_dataset(
    dataset_name="diabetes-mixed-eval-v1",
    description="mixed vector and graph questions for workflow comparison",
)

# Load dataset
with open("4_Evaluation/datasets/mixed_dataset.json", "r") as f:
    examples = json.load(f)

# Upload each example
for example in examples:
    # Build outputs dict dynamically based on what exists
    outputs = {
        "ground_truth_answer": example["ground_truth_answer"],
        "expected_route": example["expected_route"],
    }

    # Only add graph-specific fields if they exist
    if "ground_truth_context" in example:
        outputs["ground_truth_context"] = example["ground_truth_context"]

    if "expected_cypher" in example:
        outputs["expected_cypher"] = example["expected_cypher"]

    client.create_example(
        dataset_name="diabetes-mixed-eval-v1",
        inputs={"question": example["question"]},
        outputs=outputs,
    )

print(f"✅ Uploaded {len(examples)} examples to LangSmith")
