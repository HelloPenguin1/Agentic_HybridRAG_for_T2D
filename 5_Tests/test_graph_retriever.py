"""
Test script for DrugBank GraphRetrieverChain
Quick testing of Cypher generation with sample nursing queries
"""

# Sample test queries organized by type

SAMPLE_QUERIES = {
    "Basic Drug Info": [
        "What is the dose of metformin?",
        "What are the available dosages for insulin glargine?",
        "Tell me about the half-life of glipizide",
    ],
    "Drug Class/Category": [
        "What drug class is semaglutide?",
        "What class does metformin belong to?",
        "Is glipizide a sulfonylurea?",
        "List other drugs in the same class as metformin",
    ],
    "Drug Interactions": [
        "Does metformin interact with glipizide?",  # Should return: no interaction found
        "Does metformin interact with insulin lispro?",  # Should find interaction
        "What drugs interact with metformin?",  # Should list all interactions
        "Does metformin interact with acetohexamide?",  # Should find hypoglycemia risk
    ],
    "Food Interactions": [
        "Can metformin be taken with food?",
        "Are there dietary restrictions for insulin?",
        "What food interactions exist for glipizide?",
    ],
    "Mechanism/Target": [
        "How does semaglutide work?",
        "What is the mechanism of action of metformin?",
        "What does insulin act on?",
        "What biological target does semaglutide affect?",
    ],
    "Brand Names": [
        "What's the brand name for insulin glargine?",
        "What are the commercial products for metformin?",
        "What brands of semaglutide are available?",
    ],
    "Toxicity/Safety": [
        "What are the toxic effects of metformin?",
        "Tell me about lactic acidosis risk with metformin",
        "What are the safety concerns for glipizide?",
        "What's the toxicity profile of insulin?",
    ],
}


def print_test_menu():
    """Display test query categories"""
    print("\n" + "=" * 70)
    print("DRUGBANK GRAPH RETRIEVER TEST QUERIES")
    print("=" * 70)

    for i, (category, queries) in enumerate(SAMPLE_QUERIES.items(), 1):
        print(f"\n{i}. {category}:")
        for j, query in enumerate(queries, 1):
            print(f"   {i}.{j} {query}")

    print(f"\n{len(SAMPLE_QUERIES) + 1}. Custom Query")
    print(f"{len(SAMPLE_QUERIES) + 2}. Exit")
    print("=" * 70)


def get_query_by_number(number_str):
    """Get query from category.item format (e.g., '1.2' or '3.1')"""
    try:
        if "." in number_str:
            cat_num, query_num = map(int, number_str.split("."))
            category = list(SAMPLE_QUERIES.keys())[cat_num - 1]
            query = SAMPLE_QUERIES[category][query_num - 1]
            return query
        else:
            return None
    except (ValueError, IndexError, KeyError):
        return None


if __name__ == "__main__":
    from graph_retriever_chain import GraphRetrieverChain

    print("\nInitializing GraphRetrieverChain...")
    try:
        retriever = GraphRetrieverChain()
        print("✓ Connected to Neo4j\n")

        while True:
            print_test_menu()
            choice = input("\nSelect a query (e.g., 1.2) or category number: ").strip()

            if choice == str(len(SAMPLE_QUERIES) + 2):  # Exit
                print("Exiting...")
                break

            elif choice == str(len(SAMPLE_QUERIES) + 1):  # Custom query
                question = input("\nEnter your custom question: ").strip()
                if not question:
                    continue

            else:
                # Try to get query by number
                question = get_query_by_number(choice)
                if not question:
                    print("Invalid selection. Try again.")
                    continue

            print(f"\n{'=' * 70}")
            print(f"QUESTION: {question}")
            print(f"{'=' * 70}\n")

            try:
                result = retriever.chain.invoke({"query": question})

                # Show intermediate steps
                steps = result.get("intermediate_steps", [])
                if len(steps) > 0:
                    print("GENERATED CYPHER:")
                    print(steps[0].get("query", "N/A"))
                    print()

                if len(steps) > 1:
                    context = steps[1].get("context", [])
                    print(f"RAW RESULTS ({len(context)} rows):")
                    for i, row in enumerate(context[:5], 1):  # Show first 5
                        print(f"  {i}. {row}")
                    if len(context) > 5:
                        print(f"  ... and {len(context) - 5} more rows")
                    print()

                print("FINAL ANSWER:")
                print(result.get("result", "No answer generated"))
                print()

            except Exception as e:
                print(f"ERROR: {e}\n")

            input("\nPress Enter to continue...")

    except Exception as e:
        print(f"Failed to initialize: {e}")
