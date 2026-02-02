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
    """
    Normalizes entity names and handles medical abbreviation variants.
    
    For medical terminology, we use exact matching and abbreviation expansion.
    Fuzzy matching is disabled by default to prevent incorrect entity merges
    (e.g., "Type 1 Diabetes" vs "Type 2 Diabetes").
    """
    
    # Load abbreviation mappings from config file
    _abbreviation_map = None
    
    @classmethod
    def _load_abbreviations(cls) -> Dict[str, List[str]]:
        """Load medical abbreviations from config file (cached)."""
        if cls._abbreviation_map is None:
            config_path = Path(__file__).parent / "config" / "medical_abbreviations.json"
            try:
                with open(config_path, 'r') as f:
                    cls._abbreviation_map = json.load(f)
            except FileNotFoundError:
                print(f"Warning: Abbreviation config not found at {config_path}, using empty map")
                cls._abbreviation_map = {}
        return cls._abbreviation_map
    
    @staticmethod
    def normalize_name(name: str, expand_abbreviations: bool = True) -> str:
        """
        Normalize an entity name, optionally expanding abbreviations.
        
        This combines basic normalization (whitespace cleanup) with optional
        abbreviation expansion into a single efficient method.
        
        Args:
            name: Original entity name
            expand_abbreviations: If True, expand known abbreviations to canonical form
        
        Returns:
            Normalized name (canonical form if abbreviation found)
        
        Examples:
            >>> normalize_name("  HbA1c  ")
            "A1C"  # Canonical form
            >>> normalize_name("Type 2 Diabetes")
            "T2D"  # Canonical form
        """
        # Remove extra whitespace and strip
        normalized = ' '.join(name.split()).strip()
        
        # Expand abbreviations if requested
        if expand_abbreviations:
            abbreviations = EntityNormalizer._load_abbreviations()
            for canonical, variants in abbreviations.items():
                if normalized == canonical or normalized in variants:
                    return canonical
        
        return normalized
    
    @staticmethod
    def are_similar(
        name1: str, 
        name2: str, 
        use_fuzzy: bool = False,
        threshold: float = 0.95
    ) -> bool:
        """
        Check if two entity names should be considered the same entity.
        
        For medical terminology, exact matching and abbreviation expansion are
        preferred. Fuzzy matching is DISABLED by default to prevent dangerous
        false positives (e.g., merging "Type 1 Diabetes" with "Type 2 Diabetes").
        
        Args:
            name1: First entity name
            name2: Second entity name
            use_fuzzy: Enable fuzzy string matching (default: False for safety)
            threshold: Similarity threshold for fuzzy matching (default: 0.95)
        
        Returns:
            True if entities should be considered the same
        
        Examples:
            >>> are_similar("HbA1c", "Hemoglobin A1C")
            True  # Abbreviation match
            >>> are_similar("Type 1 Diabetes", "Type 2 Diabetes")
            False  # Different entities (fuzzy disabled by default)
        """
        # Normalize both names (with abbreviation expansion)
        norm1 = EntityNormalizer.normalize_name(name1).lower()
        norm2 = EntityNormalizer.normalize_name(name2).lower()
        
        # Exact match (after normalization and abbreviation expansion)
        if norm1 == norm2:
            return True
        
        # Fuzzy matching (opt-in only, higher threshold for safety)
        if use_fuzzy:
            similarity = SequenceMatcher(None, norm1, norm2).ratio()
            return similarity >= threshold
        
        # Default: entities are different
        return False


class EntityLinker:
    """Links entities across documents and builds unified entity set"""
    
    def __init__(self):
        """Initialize entity linker with optimized lookup structures."""
        self.entity_clusters = defaultdict(list)
        self.entity_id_map = {}
        self.canonical_to_cluster = {}  # Cache for O(1) lookup
        self.next_id = 1
    
    def add_entity(self, entity_data: Dict[str, Any], source_pdf: str, entity_type: str):
        """
        Add an entity to the linking system.
        
        Uses O(1) hash lookup for efficient clustering instead of O(n) linear search.
        
        Args:
            entity_data: Entity dict from extraction
            source_pdf: Source PDF name
            entity_type: Type of entity
        """
        entity_name = entity_data.get('name', '')
        if not entity_name:
            return
        
        # Normalize to canonical form (handles abbreviations)
        canonical_name = EntityNormalizer.normalize_name(entity_name)
        
        # O(1) lookup instead of O(n) linear search
        if canonical_name in self.canonical_to_cluster:
            cluster_key = self.canonical_to_cluster[canonical_name]
        else:
            # New cluster
            cluster_key = canonical_name
            self.canonical_to_cluster[canonical_name] = cluster_key
        
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
        
        # Create output directory for linked entities
        self.linked_dir = self.output_dir / "linked"
        self.linked_dir.mkdir(parents=True, exist_ok=True)
        
        self.entity_linkers = {}  # One linker per entity type
    
    def process_all(self):
        """Process all extracted entity files"""
        entity_files = list(self.entities_dir.glob("*_entities.json"))
        
        if not entity_files:
            print("No entity files found for post-processing")
            return
        
        for entity_file in entity_files:
            print(f"Processing: {entity_file.name}")
            self._process_file(entity_file)
        
        # Generate unified entity sets
        self._generate_unified_entities()
        
        # Generate summary statistics
        self._generate_statistics()
        
        print("Post-processing complete")
    
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
        
        print(f"Processed {len(extractions)} extraction chunks")
    
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
            
            print(f"{entity_type}: {len(unified)} unique entities")
        
        # Save combined file
        combined_file = self.linked_dir / "all_entities_unified.json"
        with open(combined_file, 'w') as f:
            json.dump(all_unified, f, indent=2)
        
        print(f"Unified entities saved to: {self.linked_dir}")
    
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
        
        print(f"Statistics: Total entity types: {stats['total_entity_types']}, Cross-document entities: {len(stats['cross_document_entities'])}, Single-source entities: {len(stats['single_source_entities'])}")
        print(f"Stats saved to: {stats_file}")


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