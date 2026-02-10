"""
Neo4j Knowledge Graph Importer
Imports 5 category-specific T2D knowledge graphs from CSV files
"""

import os
import csv
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# Neo4j connection settings
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Categories to import
CATEGORIES = [
    "assessment_diagnosis",
    "complications_management",
    "education_lifestyle",
    "pharmacology_technology",
    "special_populations"
]

class Neo4jImporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        # Handle being run from parent directory or t2d_extraction_pipeline directory
        script_dir = Path(__file__).parent
        self.base_dir = script_dir / "outputs" / "neo4j_category_imports"
    
    def close(self):
        self.driver.close()
    
    def clear_database(self, tx):
        """Clear all nodes and relationships"""
        tx.run("MATCH (n) DETACH DELETE n")
    
    def create_constraints(self, tx, category):
        """Create uniqueness constraints for node IDs"""
        # Create constraint for Entity nodes
        tx.run(f"""
            CREATE CONSTRAINT {category}_entity_id IF NOT EXISTS
            FOR (n:Entity) REQUIRE n.node_id IS UNIQUE
        """)
    
    def import_nodes(self, tx, csv_path, category):
        """Import nodes from CSV"""
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            nodes_created = 0
            
            for row in reader:
                # Parse pipe-separated arrays
                sources = row['sources:string[]'].split('|') if row['sources:string[]'] else []
                evidence_levels = row['evidence_levels:string[]'].split('|') if row['evidence_levels:string[]'] else []
                pages = row['pages:string[]'].split('|') if row['pages:string[]'] else []
                
                # Extract primary label from type:LABEL (format: "Entity;SpecificType")
                labels = row['type:LABEL'].split(';')
                primary_label = labels[1] if len(labels) > 1 else labels[0]
                
                # Create node with properties
                tx.run("""
                    CREATE (n:Entity {
                        node_id: $node_id,
                        name: $name,
                        category: $category,
                        sources: $sources,
                        evidence_levels: $evidence_levels,
                        pages: $pages
                    })
                    SET n:%s
                """ % primary_label, {
                    'node_id': row['node_id:ID'],
                    'name': row['name'],
                    'category': category,
                    'sources': sources,
                    'evidence_levels': evidence_levels,
                    'pages': pages
                })
                nodes_created += 1
            
            return nodes_created
    
    def import_relationships(self, tx, csv_path):
        """Import relationships from CSV"""
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rels_created = 0
            
            for row in reader:
                rel_type = row['type:TYPE']
                
                # Create relationship with properties
                tx.run(f"""
                    MATCH (start:Entity {{node_id: $start_id}})
                    MATCH (end:Entity {{node_id: $end_id}})
                    CREATE (start)-[r:`{rel_type}` {{
                        evidence: $evidence,
                        context: $context,
                        origin: $origin
                    }}]->(end)
                """, {
                    'start_id': row[':START_ID'],
                    'end_id': row[':END_ID'],
                    'evidence': row['evidence'],
                    'context': row['context'],
                    'origin': row['origin']
                })
                rels_created += 1
            
            return rels_created
    
    def create_indexes(self, tx):
        """Create indexes for performance"""
        tx.run("CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)")
        tx.run("CREATE INDEX entity_category IF NOT EXISTS FOR (n:Entity) ON (n.category)")
    
    def import_category(self, category):
        """Import a single category's knowledge graph"""
        print(f"\n📦 Importing category: {category.upper()}")
        
        category_dir = self.base_dir / category
        nodes_csv = category_dir / "nodes.csv"
        rels_csv = category_dir / "relationships.csv"
        
        if not nodes_csv.exists() or not rels_csv.exists():
            print(f"  ⚠️ CSV files not found for {category}")
            return
        
        with self.driver.session() as session:
            # Create constraints
            session.execute_write(self.create_constraints, category)
            
            # Import nodes
            nodes_count = session.execute_write(self.import_nodes, nodes_csv, category)
            print(f"  ✅ Imported {nodes_count} nodes")
            
            # Import relationships
            rels_count = session.execute_write(self.import_relationships, rels_csv)
            print(f"  ✅ Imported {rels_count} relationships")
    
    def verify_import(self):
        """Verify the import with count queries"""
        print("\n📊 Verification:")
        
        with self.driver.session() as session:
            # Total counts
            result = session.run("MATCH (n) RETURN count(n) as total_nodes")
            total_nodes = result.single()['total_nodes']
            
            result = session.run("MATCH ()-[r]->() RETURN count(r) as total_rels")
            total_rels = result.single()['total_rels']
            
            print(f"  Total Nodes: {total_nodes}")
            print(f"  Total Relationships: {total_rels}")
            
            # Counts by category
            print("\n  By Category:")
            result = session.run("""
                MATCH (n:Entity)
                RETURN n.category as category, count(n) as count
                ORDER BY category
            """)
            for record in result:
                print(f"    {record['category']}: {record['count']} nodes")
    
    def run_import(self, clear_first=False):
        """Run the complete import process"""
        print("🚀 Starting Neo4j Knowledge Graph Import")
        print(f"   URI: {NEO4J_URI}")
        
        with self.driver.session() as session:
            if clear_first:
                print("\n🗑️  Clearing existing data...")
                session.execute_write(self.clear_database)
            
            # Create indexes
            print("\n📇 Creating indexes...")
            session.execute_write(self.create_indexes)
        
        # Import each category
        for category in CATEGORIES:
            self.import_category(category)
        
        # Verify
        self.verify_import()
        
        print("\n✅ Import complete!")
        print("\n💡 Next steps:")
        print("   1. Open Neo4j Browser: http://localhost:7474")
        print("   2. Run sample query: MATCH (n) RETURN n LIMIT 25")
        print("   3. Explore relationships: MATCH p=()-[]->() RETURN p LIMIT 50")


def main():
    """Main entry point"""
    # Check if .env file exists
    if not Path(".env").exists():
        print("⚠️  No .env file found. Creating template...")
        with open(".env", "a") as f:
            f.write("\n# Neo4j Connection Settings\n")
            f.write("NEO4J_URI=bolt://localhost:7687\n")
            f.write("NEO4J_USER=neo4j\n")
            f.write("NEO4J_PASSWORD=your_password_here\n")
        print("   Please update .env with your Neo4j credentials and run again.")
        return
    
    # Create importer and run
    importer = Neo4jImporter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Set clear_first=True to delete existing data before import
        importer.run_import(clear_first=True)
    finally:
        importer.close()


if __name__ == "__main__":
    main()
