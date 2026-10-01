import xml.etree.ElementTree as ET
import json
import os
from typing import Dict, List
from tqdm import tqdm


class DrugBankDiabetesParser:
    # Rigorous namespace definition matching the XSD
    NS = {"db": "http://www.drugbank.ca"}

    def __init__(self, xml_path: str):
        self.xml_path = os.path.normpath(xml_path)
        self.diabetes_drugs = []

    def is_diabetes_related(self, drug_elem) -> bool:
        """Filter drugs strictly by the A10 ATC code."""
        for atc in drug_elem.findall(".//db:atc-codes/db:atc-code", self.NS):
            if atc.get("code", "").startswith("A10"):
                return True
        return False

    def extract_drug_node(self, drug_elem) -> Dict:
        """Extract the core Medication Node, flattening messy dosages for the LLM."""
        primary_id = ""
        for db_id in drug_elem.findall("db:drugbank-id", self.NS):
            if db_id.get("primary") == "true":
                primary_id = db_id.text
                break

        atc_codes = [
            atc.get("code")
            for atc in drug_elem.findall(".//db:atc-codes/db:atc-code", self.NS)
            if atc.get("code")
        ]

        # Extract Categories for denser drug-grouping (e.g., "Hypoglycemic Agents")
        categories = [
            cat.text
            for cat in drug_elem.findall(
                ".//db:categories/db:category/db:category", self.NS
            )
            if cat.text
        ]

        # Flatten messy dosages into a simple string list for the LLM to read easily
        raw_dosages = []
        for dosage in drug_elem.findall(".//db:dosages/db:dosage", self.NS):
            form = dosage.findtext("db:form", default="", namespaces=self.NS)
            route = dosage.findtext("db:route", default="", namespaces=self.NS)
            strength = dosage.findtext("db:strength", default="", namespaces=self.NS)

            # Create a readable sentence for the LLM like "Tablet, extended release (Oral) - 10 mg"
            dose_str = f"{form} ({route}) - {strength}".strip(" -()")
            if dose_str:
                raw_dosages.append(dose_str)

        # Extract food interactions as a simple list on the node
        food_interactions = [
            food.text
            for food in drug_elem.findall(
                ".//db:food-interactions/db:food-interaction", self.NS
            )
            if food.text
        ]

        return {
            "drugbank_id": primary_id,
            "name": drug_elem.findtext("db:name", default="", namespaces=self.NS),
            "description": drug_elem.findtext(
                "db:description", default="", namespaces=self.NS
            ),
            "indication": drug_elem.findtext(
                "db:indication", default="", namespaces=self.NS
            ),
            "toxicity": drug_elem.findtext(
                "db:toxicity", default="", namespaces=self.NS
            ),
            "mechanism_of_action": drug_elem.findtext(
                "db:mechanism-of-action", default="", namespaces=self.NS
            ),
            "half_life": drug_elem.findtext(
                "db:half-life", default="", namespaces=self.NS
            ),
            "clearance": drug_elem.findtext(
                "db:clearance", default="", namespaces=self.NS
            ),
            "atc_codes": atc_codes,
            "categories": list(set(categories)),
            "available_dosages": list(set(raw_dosages)),
            "food_interactions": food_interactions,
        }

    def extract_targets(self, drug_elem) -> List[Dict]:
        """Extract Biological Targets for cross-referencing mechanisms."""
        targets = []
        for target in drug_elem.findall(".//db:targets/db:target", self.NS):
            name = target.findtext("db:name", default="", namespaces=self.NS)
            action = target.findtext(
                ".//db:actions/db:action", default="", namespaces=self.NS
            )
            if name:
                targets.append({"target_name": name, "pharmacological_action": action})
        return targets

    def extract_products(self, drug_elem) -> List[Dict]:
        """Extract Real-world Products (brand names on the floor)."""
        products = []
        for product in drug_elem.findall(".//db:products/db:product", self.NS):
            products.append(
                {
                    "brand_name": product.findtext(
                        "db:name", default="", namespaces=self.NS
                    ),
                    "labeller": product.findtext(
                        "db:labeller", default="", namespaces=self.NS
                    ),
                    "country": product.findtext(
                        "db:country", default="", namespaces=self.NS
                    ),
                }
            )
        return products

    def extract_drug_interactions(self, drug_elem) -> List[Dict]:
        """Extract Drug-Drug Interactions to build safety edges."""
        drug_interactions = []
        for interaction in drug_elem.findall(
            ".//db:drug-interactions/db:drug-interaction", self.NS
        ):
            drug_interactions.append(
                {
                    "interacting_drugbank_id": interaction.findtext(
                        "db:drugbank-id", default="", namespaces=self.NS
                    ),
                    "interacting_name": interaction.findtext(
                        "db:name", default="", namespaces=self.NS
                    ),
                    "description": interaction.findtext(
                        "db:description", default="", namespaces=self.NS
                    ),
                }
            )
        return drug_interactions

    def parse_database(self) -> List[Dict]:
        """Single-pass parsing with toxicity filter."""
        print(f"Parsing DrugBank XML from: {self.xml_path}")
        context = ET.iterparse(self.xml_path, events=("end",))
        skipped_no_toxicity = 0

        for event, elem in tqdm(context, desc="Scanning XML"):
            if elem.tag == f"{{{self.NS['db']}}}drug":
                if self.is_diabetes_related(elem):
                    medication = self.extract_drug_node(elem)

                    # Filter: skip outdated/incomplete legacy drugs
                    if not medication.get("toxicity", "").strip():
                        skipped_no_toxicity += 1
                        elem.clear()
                        continue

                    drug_data = {
                        "Medication": medication,
                        "Targets": self.extract_targets(elem),
                        "Products": self.extract_products(elem),
                        "DrugInteractions": self.extract_drug_interactions(elem),
                    }
                    self.diabetes_drugs.append(drug_data)

                elem.clear()  # Free memory after processing

        print(
            f"\nExtraction complete. Found {len(self.diabetes_drugs)} rich diabetes drugs."
        )
        print(f"Skipped {skipped_no_toxicity} drugs with no toxicity data.")
        return self.diabetes_drugs


def main():
    xml_path = r"7_drugbank\full_database.xml"
    parser = DrugBankDiabetesParser(xml_path)
    diabetes_drugs = parser.parse_database()

    output_path = r"7_drugbank\graph_data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(diabetes_drugs, f, indent=2, ensure_ascii=False)
    print(f"Saved rich graph data to {output_path}")


if __name__ == "__main__":
    main()
