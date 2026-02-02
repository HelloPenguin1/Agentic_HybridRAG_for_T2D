"""
Post-Processing Module for Entity Normalization and Linking
Handles deduplication, normalization, and cross-document entity linking
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict
from difflib import SequenceMatcher


class EntityNormalizer:
    """Normalizes entity names and handles variants"""
    
    # Common medical abbreviation mappings
    ABBREVIATION_MAP = {
        'A1C': ['HbA1c', 'Hemoglobin A1C', 'Glycated Hemoglobin'],
        'FPG': ['Fasting Plasma Glucose', 'Fasting Glucose'],
        'OGTT': ['Oral Glucose Tolerance Test'],
        'T2D': ['Type 2 Diabetes', 'Type 2 Diabetes Mellitus', 'T2DM'],
        'T1D': ['Type 1 Diabetes', 'Type 1 Diabetes Mellitus', 'T1DM'],
        'GLP-1': ['GLP-1 RA', 'GLP-1 Receptor Agonist', 'Glucagon-like Peptide-1'],
        'SGLT2': ['SGLT2i', 'SGLT2 Inhibitor', 'Sodium-Glucose Cotransporter-2'],
        'CGM': ['Continuous Glucose Monitor', 'Continuous Glucose Monitoring'],
        'DSMES': ['Diabetes Self-Management Education and Support'],
        'MNT': ['Medical Nutrition Therapy'],
        'DPP': ['Diabetes Prevention Program'],
        'CKD': ['Chronic Kidney Disease'],
        'CVD': ['Cardiovascular Disease'],
        'ASCVD': ['Atherosclerotic Cardiovascular Disease'],
        'eGFR': ['estimated Glomerular Filtration Rate'],
        'UACR': ['Urine Albumin-to-Creatinine Ratio'],
        'ACE': ['ACE inhibitor', 'Angiotensin-Converting Enzyme inhibitor'],
        'ARB': ['Angiotensin Receptor Blocker'],
        'BMI': ['Body Mass Index']
    }
    
    @staticmethod
    def normalize_name(name: str) -> str:
        """
        Normalize an entity name to canonical form.
        
        Args:
            name: Original entity name
        
        Returns:
            Normalized name
        """
        # Remove extra whitespace
        name = ' '.join(name.split())
        
        # Convert to title case for consistency
        normalized = name.strip()
        
        return normalized
    
    @staticmethod
    def get_canonical_form(name: str) -> str:
        """
        Get the canonical form of an entity, expanding abbreviations.
        
        Args:
            name: Entity name (possibly abbreviated)
        
        Returns:
            Canonical form
        """
        normalized = EntityNormalizer.normalize_name(name)
        
        # Check if it's a known abbreviation
        for canonical, variants in EntityNormalizer.ABBREVIATION_MAP.items():
            if normalized == canonical or normalized in variants:
                return canonical
        
        return normalized
    
    @staticmethod
    def are_similar(name1: str, name2: str, threshold: float = 0.85) -> bool:
        """
        Check if two entity names are similar enough to be considered the same.
        
        Args:
            name1: First entity name
            name2: Second entity name
            threshold: Similarity threshold (0-1)
        
        Returns:
            True if similar enough
        """
        norm1 = EntityNormalizer.normalize_name(name1).lower()
        norm2 = EntityNormalizer.normalize_name(name2).lower()
        
        # Exact match
        if norm1 == norm2:
            return True
        
        # Check if one is an abbreviation of the other
        can1 = EntityNormalizer.get_canonical_form(name1)
        can2 = EntityNormalizer.get_canonical_form(name2)
        if can1 == can2:
            return True
        
        # Use sequence matching for fuzzy comparison
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        return similarity >= threshold
    
    @staticmethod
    def extract_numeric_value(text: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Extract numeric value and unit from text.
        
        Args:
            text: Text containing a value (e.g., "6.5%", "126 mg/dL")
        
        Returns:
            Tuple of (value, unit) or (None, None)
        """
        # Pattern to match number with optional unit
        pattern = r'([<>=≤≥]?\s*\d+\.?\d*)\s*(%|mg/dL|mmol/L|mg/g|years?|min|weeks?|months?)?'
        
        match = re.search(pattern, text)
        if match:
            value_str = match.group(1).strip()
            unit = match.group(2) if match.group(2) else None
            
            # Extract just the number
            num_pattern = r'\d+\.?\d*'
            num_match = re.search(num_pattern, value_str)
            if num_match:
                try:
                    value = float(num_match.group())
                    return value, unit
                except ValueError:
                    pass
        
        return None, None


class EntityLinker:
    """Links entities across documents and builds unified entity set"""
    
    def __init__(self):
        """Initialize entity linker"""
        self.entity_clusters = defaultdict(list)
        self.entity_id_map = {}
        self.next_id = 1
    
    def add_entity(self, entity_data: Dict[str, Any], source_pdf: str, entity_type: str):
        """
        Add an entity to the linking system.
        
        Args:
            entity_data: Entity dict from extraction
            source_pdf: Source PDF name
            entity_type: Type of entity
        """
        entity_name = entity_data.get('name', '')
        if not entity_name:
            return
        
        canonical_name = EntityNormalizer.get_canonical_form(entity_name)
        
        # Check if this entity matches any existing cluster
        matched_cluster = None
        for cluster_key, cluster_entities in self.entity_clusters.items():
            cluster_canonical = cluster_key
            
            if EntityNormalizer.are_similar(canonical_name, cluster_canonical):
                matched_cluster = cluster_key
                break
        
        # Add to existing cluster or create new one
        if matched_cluster:
            cluster_key = matched_cluster
        else:
            cluster_key = canonical_name
        
        # Assign unique ID if not already assigned
        if cluster_key not in self.entity_id_map:
            self.entity_id_map[cluster_key] = f"{entity_type}_{self.next_id:05d}"
            self.next_id += 1
        
        entity_id = self.entity_id_map[cluster_key]
        
        # Add entity to cluster
        self.entity_clusters[cluster_key].append({
            'entity_id': entity_id,
            'canonical_name': cluster_key,
            'original_name': entity_name,
            'entity_type': entity_type,
            'source_pdf': source_pdf,
            'data': entity_data
        })
    
    def get_unified_entities(self) -> List[Dict[str, Any]]:
        """
        Get unified entity list with merged information.
        
        Returns:
            List of unified entities
        """
        unified = []
        
        for canonical_name, entity_instances in self.entity_clusters.items():
            if not entity_instances:
                continue
            
            # Get common entity ID
            entity_id = entity_instances[0]['entity_id']
            entity_type = entity_instances[0]['entity_type']
            
            # Collect all variant names
            all_names = {inst['original_name'] for inst in entity_instances}
            
            # Collect all source PDFs
            sources = {inst['source_pdf'] for inst in entity_instances}
            
            # Merge data from all instances
            merged_data = self._merge_entity_data(entity_instances)
            
            unified.append({
                'entity_id': entity_id,
                'canonical_name': canonical_name,
                'entity_type': entity_type,
                'variant_names': list(all_names),
                'sources': list(sources),
                'occurrence_count': len(entity_instances),
                'merged_data': merged_data
            })
        
        return unified
    
    def _merge_entity_data(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge data from multiple instances of the same entity"""
        merged = {}
        
        # Collect all unique values for each field
        for instance in instances:
            data = instance['data']
            
            for key, value in data.items():
                if key == 'name':
                    continue
                
                if key not in merged:
                    merged[key] = []
                
                if isinstance(value, list):
                    merged[key].extend(value)
                elif value is not None:
                    merged[key].append(value)
        
        # Deduplicate lists
        for key in merged:
            if isinstance(merged[key], list):
                merged[key] = list(set(merged[key]))
        
        return merged


class PostProcessor:
    """Main post-processing orchestrator"""
    
    def __init__(self, entities_dir: str, output_dir: str):
        """
        Initialize post-processor.
        
        Args:
            entities_dir: Directory with extracted entities
            output_dir: Directory for post-processed outputs
        """
        self.entities_dir = Path(entities_dir)
        self.output_dir = Path(output_dir)
        
        # Create output subdirectories
        self.normalized_dir = self.output_dir / "normalized"
        self.linked_dir = self.output_dir / "linked"
        self.normalized_dir.mkdir(parents=True, exist_ok=True)
        self.linked_dir.mkdir(parents=True, exist_ok=True)
        
        self.entity_linkers = {}  # One linker per entity type
    
    def process_all(self):
        """Process all extracted entity files"""
        entity_files = list(self.entities_dir.glob("*_entities.json"))
        
        if not entity_files:
            print("❌ No entity files found for post-processing")
            return
        
        print(f"\n🔧 Post-processing {len(entity_files)} entity files...")
        
        for entity_file in entity_files:
            print(f"\n📄 Processing: {entity_file.name}")
            self._process_file(entity_file)
        
        # Generate unified entity sets
        print("\n🔗 Linking entities across documents...")
        self._generate_unified_entities()
        
        # Generate summary statistics
        print("\n📊 Generating statistics...")
        self._generate_statistics()
        
        print("\n✅ Post-processing complete!")
    
    def _process_file(self, entity_file: Path):
        """Process a single entity extraction file"""
        with open(entity_file, 'r') as f:
            data = json.load(f)
        
        pdf_name = data['pdf']
        category = data['category']
        extractions = data['extractions']
        
        # Process each extraction chunk
        for extraction in extractions:
            extraction_data = extraction['extraction']
            
            # Process each entity type in this extraction
            for entity_type_key, entities in extraction_data.items():
                if not isinstance(entities, list):
                    continue
                
                # Initialize linker for this entity type if needed
                if entity_type_key not in self.entity_linkers:
                    self.entity_linkers[entity_type_key] = EntityLinker()
                
                linker = self.entity_linkers[entity_type_key]
                
                # Add each entity to the linker
                for entity in entities:
                    linker.add_entity(entity, pdf_name, entity_type_key)
        
        print(f"   ✅ Processed {len(extractions)} extraction chunks")
    
    def _generate_unified_entities(self):
        """Generate unified entity sets across all documents"""
        all_unified = {}
        
        for entity_type, linker in self.entity_linkers.items():
            unified = linker.get_unified_entities()
            all_unified[entity_type] = unified
            
            # Save per-type unified entities
            output_file = self.linked_dir / f"{entity_type}_unified.json"
            with open(output_file, 'w') as f:
                json.dump(unified, f, indent=2)
            
            print(f"   ✅ {entity_type}: {len(unified)} unique entities")
        
        # Save combined file
        combined_file = self.linked_dir / "all_entities_unified.json"
        with open(combined_file, 'w') as f:
            json.dump(all_unified, f, indent=2)
        
        print(f"\n   📁 Unified entities saved to: {self.linked_dir}")
    
    def _generate_statistics(self):
        """Generate statistics about extracted and linked entities"""
        stats = {
            'total_entity_types': len(self.entity_linkers),
            'entity_type_counts': {},
            'cross_document_entities': [],
            'single_source_entities': []
        }
        
        for entity_type, linker in self.entity_linkers.items():
            unified = linker.get_unified_entities()
            
            stats['entity_type_counts'][entity_type] = len(unified)
            
            # Find entities that appear in multiple documents
            for entity in unified:
                if len(entity['sources']) > 1:
                    stats['cross_document_entities'].append({
                        'entity_id': entity['entity_id'],
                        'name': entity['canonical_name'],
                        'type': entity_type,
                        'sources': entity['sources'],
                        'occurrences': entity['occurrence_count']
                    })
                else:
                    stats['single_source_entities'].append({
                        'entity_id': entity['entity_id'],
                        'name': entity['canonical_name'],
                        'type': entity_type
                    })
        
        # Save statistics
        stats_file = self.output_dir / "post_processing_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n   📊 Statistics:")
        print(f"      Total entity types: {stats['total_entity_types']}")
        print(f"      Cross-document entities: {len(stats['cross_document_entities'])}")
        print(f"      Single-source entities: {len(stats['single_source_entities'])}")
        print(f"      Stats saved to: {stats_file}")


def main():
    """Main entry point for post-processing"""
    
    BASE_DIR = Path(__file__).parent.parent
    ENTITIES_DIR = BASE_DIR / "outputs" / "extracted_entities"
    OUTPUT_DIR = BASE_DIR / "outputs"
    
    processor = PostProcessor(
        entities_dir=str(ENTITIES_DIR),
        output_dir=str(OUTPUT_DIR)
    )
    
    processor.process_all()


if __name__ == "__main__":
    main()