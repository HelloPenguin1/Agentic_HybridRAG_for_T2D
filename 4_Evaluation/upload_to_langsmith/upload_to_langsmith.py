"""
upload_to_langsmith.py
Uploads test_dataset.json to LangSmith as a named dataset for evaluation.

Run from project root:
    python 6_Evaluation/upload_to_langsmith.py
"""

import json
from pathlib import Path
from langsmith import Client
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

client = Client()

with open("4_Evaluation\datasets\mixed_dataset.py", encoding="utf-8") as f:
    dataset = json.load(f)

dataset_name = "diabetes_graphdb_dataset"

ls_dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="Evaluation dataset for the Agentic GraphRAG T2D system: Testing Graph Only"
)

client.create_examples(
    inputs=[{"question": item["question"]} for item in dataset],
    outputs=[{
        "ground_truth_answer": item["ground_truth_answer"],
        "expected_route":      item["expected_route"],
    } for item in dataset],
    dataset_id=ls_dataset.id
)

print(f"Uploaded {len(dataset)} questions to LangSmith dataset: {dataset_name}")
print(f"Dataset ID: {ls_dataset.id}")
