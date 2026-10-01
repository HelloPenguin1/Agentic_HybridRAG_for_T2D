import json
from langsmith import Client

client = Client()
dataset_name = "T2D_Graph_Dataset"  # You will reference this name in the run script


def upload_dataset():
    # Create a new dataset in LangSmith
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="GraphRAG evaluation for Type 2 Diabetes nursing",
    )

    # Load your JSON file
    with open("4_Evaluation\graph_eval_dataset.json", "r") as f:
        data = json.load(f)

    # Upload each question and its ground truths
    for item in data:
        client.create_example(
            inputs={"question": item["question"]},
            outputs={
                "expected_cypher": item["expected_cypher"],
                "ground_truth_context": item["ground_truth_context"],
                "ground_truth_answer": item["ground_truth_answer"],
            },
            dataset_id=dataset.id,
        )
    print(f"Successfully uploaded {len(data)} examples to '{dataset_name}'!")


if __name__ == "__main__":
    upload_dataset()
