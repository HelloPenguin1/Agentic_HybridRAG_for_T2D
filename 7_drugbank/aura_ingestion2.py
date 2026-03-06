import json
from neo4j import GraphDatabase

class DiabetesGraphLoader:
    def __init__(self, uri, user, password):
        # Initialize the connection to Neo4j
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_constraints(self):
        """Create uniqueness constraints to prevent duplicates and speed up queries."""
        with self.driver.session() as session:
            queries = [
                "CREATE CONSTRAINT drug_id IF NOT EXISTS FOR (d:Drug) REQUIRE d.drugbank_id IS UNIQUE",
                "CREATE CONSTRAINT atc_code IF NOT EXISTS FOR (a:ATC) REQUIRE a.code IS UNIQUE",
                "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
                "CREATE CONSTRAINT target_name IF NOT EXISTS FOR (t:Target) REQUIRE t.name IS UNIQUE",
                "CREATE CONSTRAINT product_name IF NOT EXISTS FOR (p:Product) REQUIRE p.brand_name IS UNIQUE",
                "CREATE CONSTRAINT food_rule IF NOT EXISTS FOR (f:FoodInteraction) REQUIRE f.description IS UNIQUE"
            ]
            for query in queries:
                session.run(query)
            print("✓ Constraints created successfully.")

    def load_data(self, json_filepath):
        """Load the JSON file and pass it to Neo4j in modular steps."""
        with open(json_filepath, 'r', encoding='utf-8') as file:
            drugs_data = json.load(file)

        with self.driver.session() as session:
            print("1/6: Ingesting Base Medication Nodes (with Pharmacokinetics & Dosages)...")
            session.execute_write(self._ingest_medications, drugs_data)
            
            print("2/6: Ingesting ATC Codes & Categories...")
            session.execute_write(self._ingest_classifications, drugs_data)
            
            print("3/6: Ingesting Biological Targets...")
            session.execute_write(self._ingest_targets, drugs_data)
            
            print("4/6: Ingesting Commercial Products...")
            session.execute_write(self._ingest_products, drugs_data)
            
            print("5/6: Ingesting Food Interactions...")
            session.execute_write(self._ingest_food, drugs_data)
            
            print("6/6: Ingesting Drug-Drug Interactions...")
            session.execute_write(self._ingest_drug_interactions, drugs_data)
            
        print(f"\n✓ Successfully loaded {len(drugs_data)} rich diabetes drugs into Neo4j!")

    @staticmethod
    def _ingest_medications(tx, drugs_data):
        # The flattened available_dosages are stored directly on the node as a list
        query = """
        UNWIND $drugs AS drug
        MERGE (d:Drug {drugbank_id: drug.Medication.drugbank_id})
        SET d.name = drug.Medication.name,
            d.description = drug.Medication.description,
            d.indication = drug.Medication.indication,
            d.toxicity = drug.Medication.toxicity,
            d.mechanism_of_action = drug.Medication.mechanism_of_action,
            d.half_life = drug.Medication.half_life,
            d.clearance = drug.Medication.clearance,
            d.available_dosages = drug.Medication.available_dosages
        """
        tx.run(query, drugs=drugs_data)

    @staticmethod
    def _ingest_classifications(tx, drugs_data):
        query = """
        UNWIND $drugs AS drug
        MATCH (d:Drug {drugbank_id: drug.Medication.drugbank_id})
        
        // FOREACH safely handles empty lists without dropping the whole row
        FOREACH (code IN drug.Medication.atc_codes |
            MERGE (a:ATC {code: code})
            MERGE (d)-[:HAS_ATC_CODE]->(a)
        )
        
        FOREACH (cat IN drug.Medication.categories |
            MERGE (c:Category {name: cat})
            MERGE (d)-[:BELONGS_TO]->(c)
        )
        """
        tx.run(query, drugs=drugs_data)

    @staticmethod
    def _ingest_targets(tx, drugs_data):
        query = """
        UNWIND $drugs AS drug
        MATCH (d:Drug {drugbank_id: drug.Medication.drugbank_id})
        
        UNWIND drug.Targets AS target
        MERGE (t:Target {name: target.target_name})
        MERGE (d)-[r:ACTS_ON]->(t)
        SET r.action = target.pharmacological_action
        """
        tx.run(query, drugs=drugs_data)

    @staticmethod
    def _ingest_products(tx, drugs_data):
        query = """
        UNWIND $drugs AS drug
        MATCH (d:Drug {drugbank_id: drug.Medication.drugbank_id})
        
        UNWIND drug.Products AS prod
        MERGE (p:Product {brand_name: prod.brand_name})
        ON CREATE SET 
            p.labeller = prod.labeller, 
            p.country = prod.country
        MERGE (d)-[:MARKETED_AS]->(p)
        """
        tx.run(query, drugs=drugs_data)

    @staticmethod
    def _ingest_food(tx, drugs_data):
        query = """
        UNWIND $drugs AS drug
        MATCH (d:Drug {drugbank_id: drug.Medication.drugbank_id})
        
        UNWIND drug.Medication.food_interactions AS food_desc
        MERGE (f:FoodInteraction {description: food_desc})
        MERGE (d)-[:HAS_DIETARY_RULE]->(f)
        """
        tx.run(query, drugs=drugs_data)

    @staticmethod
    def _ingest_drug_interactions(tx, drugs_data):
        query = """
        UNWIND $drugs AS drug
        MATCH (d1:Drug {drugbank_id: drug.Medication.drugbank_id})
        
        UNWIND drug.DrugInteractions AS interaction
        MERGE (d2:Drug {drugbank_id: interaction.interacting_drugbank_id})
        ON CREATE SET d2.name = interaction.interacting_name
        
        MERGE (d1)-[r:INTERACTS_WITH]->(d2)
        SET r.description = interaction.description
        """
        tx.run(query, drugs=drugs_data)

if __name__ == "__main__":
    NEO4J_URI = "neo4j+s://52830101.databases.neo4j.io" # Or bolt://localhost:7687
    NEO4J_USER = "52830101"
    NEO4J_PASSWORD = "T7zK97BX1FVw48eifpBhXSupNXVw_YjBjSesnv0YozU"
    
    JSON_FILE = r"7_drugbank\graph_data.json"
    
    loader = DiabetesGraphLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        loader.create_constraints()
        loader.load_data(JSON_FILE)
    finally:
        loader.close()