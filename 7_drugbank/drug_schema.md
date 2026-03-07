Overview: This Neo4j property graph maps diabetes-related medications (specifically ATC A10 class) to their clinical profiles, biological targets, commercial products, and safety interactions. It is designed to power an Agentic GraphRAG system for nursing and clinical decision support.
1. Nodes and Properties
`(:Drug)`
The central node representing a generic medication.
* `drugbank_id` (String): Unique primary identifier (e.g., "DB00331").
* `name` (String): Generic name of the drug (e.g., "Metformin").
* `description` (String): General overview of the drug.
* `indication` (String): Approved medical conditions the drug treats.
* `toxicity` (String): Overdose protocols, maximum doses, and severe adverse effects (e.g., lactic acidosis risks).
* `mechanism_of_action` (String): Pharmacological mechanism at the cellular level.
* `half_life` (String): Pharmacokinetic half-life (crucial for dosing schedules).
* `clearance` (String): How the drug is eliminated (useful for renal/hepatic warnings).
* `available_dosages` (List of Strings): Flattened, human-readable list of available forms, routes, and strengths (e.g., `["Tablet (Oral) - 500 mg", "Solution (Subcutaneous) - 100 U/mL"]`).
`(:Target)`
The biological or molecular target of the drug.
* `name` (String): Name of the biological target (e.g., "Glucagon-like peptide 1 receptor").
`(:Category)`
The pharmacological class or grouping.
* `name` (String): Name of the category (e.g., "Sulfonylureas", "Biguanides").
`(:ATC)`
World Health Organization Anatomical Therapeutic Chemical classification.
* `code` (String): The specific ATC code (e.g., "A10BA02").
`(:Product)`
Real-world, commercially branded versions of the drug.
* `brand_name` (String): Commercial name (e.g., "Glucotrol", "Humalog").
* `labeller` (String): The pharmaceutical manufacturing company.
* `country` (String): Market availability (e.g., "US", "Canada").
`(:FoodInteraction)`
Dietary rules and restrictions.
* `description` (String): The specific dietary instruction (e.g., "Take with a meal to reduce GI upset").
2. Relationships (Edges)
* `(:Drug)-[:ACTS_ON]->(:Target)`
   * Edge Property: `action` (String) - Describes the pharmacological action (e.g., "agonist", "inhibitor").
* `(:Drug)-[:BELONGS_TO]->(:Category)`
* `(:Drug)-[:HAS_ATC_CODE]->(:ATC)`
* `(:Drug)-[:MARKETED_AS]->(:Product)`
* `(:Drug)-[:HAS_DIETARY_RULE]->(:FoodInteraction)`
* `(:Drug)-[:INTERACTS_WITH]->(:Drug)`
   * Edge Property: `description` (String) - Detailed clinical warning of what happens when these two drugs are combined (e.g., "The risk or severity of hypoglycemia can be increased..."). Note: This points to both other diabetes drugs AND non-diabetes drugs.
