
import os
import sys
import json
import uuid
import requests

from dotenv import load_dotenv
from langsmith import Client


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("LANGSMITH_API_KEY")
ENDPOINT = os.getenv(
    "LANGSMITH_ENDPOINT",
).rstrip("/")

CORRECTNESS_RULE_ID = os.getenv(
    "CORRECTNESS_RULE_ID", ""
).strip()

# Use the EXACT experiment/project names displayed in LangSmith.
EXPERIMENT_NAMES = [
    #"Adaptive_Full_E2E2",
     "Adaptive_Lite_E2E2",
     "Always_Both_E2E2",
     "Always_Both_Full_E2E2",
     "Graph_Only_E2E2",
     "Vector_Only_E2E2",
]

# Safety switch: keep True until IDs and configuration are verified.
DRY_RUN = False


# ============================================================
# 2. INITIALIZE CLIENTS
# ============================================================

if not API_KEY:
    raise ValueError("LANGSMITH_API_KEY is missing from .env")

client = Client(
    api_key=API_KEY,
    api_url=ENDPOINT
)

headers = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}

# Optional tenant ID, if required by your workspace.
tenant_id = os.getenv("LANGSMITH_TENANT_ID")
if tenant_id:
    headers["x-tenant-id"] = tenant_id


# ============================================================
# 3. RESOLVE EXISTING EXPERIMENTS
# ============================================================

print("\nResolving existing LangSmith experiments...\n")

experiments = []
dataset_ids = set()

for name in EXPERIMENT_NAMES:

    try:
        project = client.read_project(project_name=name)

        experiment_id = str(project.id)
        dataset_id = getattr(project, "reference_dataset_id", None)

        dataset_id = (
            str(dataset_id) if dataset_id else None
        )

        if not dataset_id:
            raise ValueError(
                f"{name} has no linked reference dataset."
            )

        dataset_ids.add(dataset_id)

        experiments.append({
            "name": name,
            "id": experiment_id,
            "dataset_id": dataset_id
        })

        print(f"[FOUND] {name}")
        print(f"        Experiment ID: {experiment_id}")
        print(f"        Dataset ID:    {dataset_id}")

    except Exception as e:
        print(f"\n[ERROR] Could not resolve {name}: {e}")
        raise


# ============================================================
# 4. VERIFY DATASET CONSISTENCY
# ============================================================

if len(dataset_ids) != 1:
    raise ValueError(
        "\nThe experiments do not all reference the same dataset.\n"
        "Use a separate dataset-bound correctness rule for each "
        "dataset before evaluating."
    )

dataset_id = next(iter(dataset_ids))

print("\nAll experiments reference the same dataset.")
print("Dataset ID:", dataset_id)


# ============================================================
# 5. DISCOVER THE CORRECTNESS RULE
# ============================================================

if not CORRECTNESS_RULE_ID:

    print("\nNo CORRECTNESS_RULE_ID supplied.")
    print("Fetching evaluator rules attached to this dataset...\n")

    response = requests.get(
        f"{ENDPOINT}/api/v1/runs/rules",
        headers=headers,
        params={
            "dataset_id": dataset_id
        },
        timeout=60
    )

    response.raise_for_status()

    rules = response.json()

    print("AVAILABLE DATASET RULES:")
    print(json.dumps(rules, indent=4, default=str))

    print(
        "\nFind the rule whose display_name corresponds "
        "to your existing Correctness evaluator."
    )
    print(
        "Copy its 'id' field into CORRECTNESS_RULE_ID "
        "in your .env file."
    )

    sys.exit(0)


# Validate the supplied rule UUID.
try:
    uuid.UUID(CORRECTNESS_RULE_ID)
except ValueError:
    raise ValueError(
        "CORRECTNESS_RULE_ID must be a valid UUID."
    )


# ============================================================
# 6. DRY RUN
# ============================================================

print("\nCorrectness backfill configuration:")
print("Rule ID:", CORRECTNESS_RULE_ID)
print("Experiments:", len(experiments))
print("Dry run:", DRY_RUN)

if DRY_RUN:
    print("\nDRY RUN COMPLETE.")
    print("No evaluator has been executed.")
    print("Set DRY_RUN = False after verifying the IDs.")
    sys.exit(0)


# ============================================================
# 7. EVALUATE EXISTING EXPERIMENTS
# ============================================================

print("\nStarting correctness backfill...\n")

successful = []
failed = []

for experiment in experiments:

    name = experiment["name"]
    experiment_id = experiment["id"]

    print("=" * 65)
    print("Evaluating:", name)
    print("Experiment ID:", experiment_id)

    url = (
        f"{ENDPOINT}/api/v1/runs/experiments/"
        f"{experiment_id}/evaluate"
    )

    payload = {
        "rule_id": CORRECTNESS_RULE_ID
    }

    try:

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=300
        )

        if not response.ok:
            print("[FAILED]", response.status_code)
            print(response.text)

            failed.append({
                "experiment": name,
                "status": response.status_code,
                "error": response.text
            })

            continue

        try:
            result = response.json()
        except ValueError:
            result = response.text

        print("[REQUEST ACCEPTED]")
        print(json.dumps(result, indent=4, default=str))

        successful.append(name)

    except requests.RequestException as e:

        print("[REQUEST ERROR]", str(e))

        failed.append({
            "experiment": name,
            "error": str(e)
        })


# ============================================================
# 8. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("CORRECTNESS BACKFILL SUMMARY")
print("=" * 65)

print(f"Total experiments: {len(experiments)}")
print(f"Requests accepted: {len(successful)}")
print(f"Request failures:  {len(failed)}")

print("\nAccepted:")
for name in successful:
    print("  +", name)

print("\nFailed:")
for item in failed:
    print("  -", item)

print(
    "\nVerify the correctness feedback in each original "
    "experiment in LangSmith."
)
