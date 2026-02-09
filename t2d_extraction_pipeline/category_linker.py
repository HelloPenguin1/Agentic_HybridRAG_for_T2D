import json
import csv
import hashlib
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Set

class CategoryLinker:
    """
    Deduplicates and links clinical entities across specific PDF extractions
    within 5 clinical domains to produce CSV files for Neo4j ingestion.
    """

    def __init__(self, entities_dir: Path, output_dir: Path, config_dir: Path = None):
        self.entities_dir = Path(entities_dir)
        self.output_dir = Path(output_dir) / "neo4j_category_imports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build comprehensive synonym map
        self.synonym_map = self._build_synonym_map(config_dir)

        # Updated mapping including Chapter 4 and all provided files
        self.category_map = {
            "assessment_diagnosis": [
                "ada_chapter2_entities.json",
                "ada_chapter4_entities.json",
                "ada_glycemic_hypoglycemia_entities.json"
            ],
            "education_lifestyle": [
                "ada_chapter3_entities.json",
                "ada_fascilitating_pos_behaviors_entities.json",
                "ada_obesity_and_weight_entities.json",
                "ada_chapter17_entities.json"
            ],
            "pharmacology_technology": [
                "ada_pharma_approaches_entities.json",
                "ada_chapter7_entities.json"
            ],
            "complications_management": [
                "ada_cardio_disease_manag_entities.json",
                "ada_chronic_kidney_manag_entities.json",
                "ada_retino_neuro_footcare_entities.json"
            ],
            "special_populations": [
                "ada_children_adolescents_standards_entities.json",
                "ada_chapter13_entities.json",
                "ada_chapter15_entities.json"
            ]
        }

    def _build_synonym_map(self, config_dir: Path = None) -> Dict[str, str]:
        """
        Builds comprehensive synonym map by loading medical abbreviations from JSON
        and merging with hardcoded clinical synonyms.
        """
        synonym_map = {}
        
        # Load medical abbreviations from config
        if config_dir is None:
            config_dir = Path(__file__).parent / "config"
        else:
            config_dir = Path(config_dir)
            
        abbrev_file = config_dir / "medical_abbreviations.json"
        
        if abbrev_file.exists():
            with open(abbrev_file, 'r') as f:
                abbreviations = json.load(f)
            
            # Build bidirectional mapping: all synonyms point to the abbreviation (canonical form)
            for abbrev, full_forms in abbreviations.items():
                abbrev_upper = abbrev.upper()
                # Map each full form to the abbreviation
                for full_form in full_forms:
                    full_upper = full_form.upper()
                    synonym_map[full_upper] = abbrev_upper
                    # Also map variations without special characters
                    clean_full = re.sub(r'[^\w\s]', '', full_upper)
                    if clean_full != full_upper:
                        synonym_map[clean_full] = abbrev_upper
        
        # Merge with hardcoded synonyms (these take precedence)
        hardcoded_synonyms = {
            "HBA1C": "A1C",
            "HEMOGLOBIN A1C": "A1C",
            "GLYCATED HEMOGLOBIN": "A1C",
            "METFORMIN HYDROCHLORIDE": "METFORMIN",
            "GLUCOPHAGE": "METFORMIN",
            "FASTING GLUCOSE": "FASTING PLASMA GLUCOSE",
            "SMBG": "SELF-MONITORING OF BLOOD GLUCOSE",
            "SELF MONITORING OF BLOOD GLUCOSE": "SELF-MONITORING OF BLOOD GLUCOSE",
            "HF": "HEART FAILURE",
            "URINARY ALBUMIN TO CREATININE RATIO": "UACR",
            "URINARY ALBUMINTOCREATININE RATIO": "UACR",
            "ESTIMATED GLOMERULAR FILTRATION RATE": "EGFR",
            "DIABETES SELF MANAGEMENT EDUCATION AND SUPPORT": "DSMES",
            "DIABETES SELFMANAGEMENT EDUCATION AND SUPPORT": "DSMES"
        }
        synonym_map.update(hardcoded_synonyms)
        
        return synonym_map

    def _normalize_name(self, name: str) -> str:
        """Normalizes clinical names for consistent matching across documents."""
        if not name: return "UNKNOWN"
        # Standardize: uppercase, remove special chars, apply synonym mapping
        clean = name.strip().upper()
        clean = re.sub(r'[^\w\s]', '', clean)
        return self.synonym_map.get(clean, clean)

    def _generate_id(self, name: str, category: str) -> str:
        """Generates a stable, category-scoped ID based on normalized name."""
        norm_name = self._normalize_name(name)
        # Category prefix prevents ID collisions between separate databases
        raw_key = f"{category.lower()}_{norm_name}"
        return hashlib.md5(raw_key.encode()).hexdigest()

    def process_and_export(self):
        """Processes each category group and exports independent CSV files."""
        for category, files in self.category_map.items():
            print(f"📦 Processing Domain: {category.upper()}")
            
            # Master storage for this category
            nodes = {} # id -> data
            relationships = []

            for filename in files:
                file_path = self.entities_dir / filename
                if not file_path.exists():
                    print(f"  ⚠️ File not found: {filename}")
                    continue

                with open(file_path, 'r') as f:
                    data = json.load(f)
                    self._link_data(data, category, filename, nodes, relationships)

            self._write_csvs(category, nodes, relationships)

    def _link_data(self, data: Dict, cat: str, source_file: str, nodes: Dict, rels: List):
        source_doc = source_file.replace("_entities.json", "")

        for key, entities in data.items():
            if key == "relationships":
                for rel in entities:
                    # Map source/target to normalized category-scoped IDs
                    rel['start_id'] = self._generate_id(rel['source_entity'], cat)
                    rel['end_id'] = self._generate_id(rel['target_entity'], cat)
                    rel['source_doc'] = source_doc
                    rels.append(rel)
                continue

            for ent in entities:
                node_id = self._generate_id(ent['name'], cat)
                norm_name = self._normalize_name(ent['name'])

                if node_id not in nodes:
                    nodes[node_id] = {
                        "name": norm_name,
                        "label": key,
                        "sources": set(),
                        "evidence_levels": set(),
                        "page_numbers": set(),
                        "trace_snippets": []
                    }

                n = nodes[node_id]
                n["sources"].add(source_doc)
                if ent.get('evidence_level') and ent['evidence_level'] != "Unknown":
                    n["evidence_levels"].add(ent['evidence_level'])
                if ent.get('page_number'):
                    n["page_numbers"].add(str(ent['page_number']))
                if ent.get('source_text'):
                    n["trace_snippets"].append({
                        "doc": source_doc,
                        "text": ent['source_text']
                    })

    def _write_csvs(self, category: str, nodes: Dict, rels: List):
        cat_dir = self.output_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        # 1. NODES CSV
        with open(cat_dir / "nodes.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["node_id:ID", "name", "type:LABEL", "sources:string[]", "evidence_levels:string[]", "pages:string[]"])
            for nid, d in nodes.items():
                writer.writerow([
                    nid,
                    d["name"],
                    f"Entity;{d['label']}",
                    "|".join(list(d["sources"])),
                    "|".join(list(d["evidence_levels"])),
                    "|".join(list(d["page_numbers"]))
                ])

        # 2. RELATIONSHIPS CSV
        with open(cat_dir / "relationships.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([":START_ID", ":END_ID", "type:TYPE", "evidence", "context", "origin"])
            for r in rels:
                writer.writerow([
                    r['start_id'],
                    r['end_id'],
                    r['relationship_type'],
                    r.get('evidence_level', 'Unknown'),
                    r.get('conditional_context', ''),
                    r['source_doc']
                ])
        
        # 3. TRACEABILITY LOG (Contextual Audit Trail)
        with open(cat_dir / "traceability.json", 'w') as f:
            trace = {nid: d["trace_snippets"] for nid, d in nodes.items()}
            json.dump(trace, f, indent=2)

        print(f"  ✅ Exported {len(nodes)} nodes and {len(rels)} relationships to {cat_dir}")

if __name__ == "__main__":
    # Ensure current directory paths are correct for your environment
    pkg_dir = Path(__file__).parent
    entities_path = pkg_dir / "outputs" / "extracted_entities"
    output_path = pkg_dir / "outputs"
    config_path = pkg_dir / "config"
    
    linker = CategoryLinker(entities_path, output_path, config_path)
    linker.process_and_export()