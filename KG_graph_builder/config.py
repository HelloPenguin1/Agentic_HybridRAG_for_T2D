"""
Configuration for LangChain Knowledge Graph Builder
"""
import os
from pathlib import Path
from typing import Dict, List

# ============================================================================
# Paths
# ============================================================================

BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
EXTRACTED_TEXT_DIR = OUTPUTS_DIR / "extracted_text"
METADATA_DIR = OUTPUTS_DIR / "metadata"

# ============================================================================
# Neo4j Configuration
# ============================================================================

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# ============================================================================
# Gemini Configuration
# ============================================================================

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash-exp"
GEMINI_TEMPERATURE = 0  # Deterministic extraction

# ============================================================================
# Processing Configuration
# ============================================================================

# Chunking parameters
MAX_CHUNK_SIZE = 8000  # tokens
CHUNK_OVERLAP = 200  # characters

# Batch processing
BATCH_SIZE = 5  # Process 5 documents at a time
MAX_CONCURRENT_REQUESTS = 3  # Async concurrency limit

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# ============================================================================
# Category Schemas
# ============================================================================

CATEGORY_SCHEMAS: Dict[str, Dict[str, List[str]]] = {
    "assessment_diagnosis": {
        "nodes": [
            "DiagnosticTest",
            "Condition", 
            "MetricValue",
            "ScreeningFrequency",
            "TargetGoal",
            "PatientProfile"
        ],
        "relationships": [
            "DIAGNOSES",
            "REQUIRES_SCREENING",
            "HAS_TARGET",
            "APPLIES_TO",
            "DEFINED_BY",
            "MEASURES"
        ]
    },
    "patient_education_lifestyle": {
        "nodes": [
            "Intervention",
            "Behavior",
            "SocialDeterminant",
            "Outcome",
            "ScreeningTool"
        ],
        "relationships": [
            "RECOMMENDS",
            "IMPROVES",
            "BARRIER_TO",
            "ASSESSES",
            "MODIFIES",
            "INFLUENCES",
            "LEADS_TO",
            "PREVENTS"
        ]
    },
    "pharmacology_technology": {
        "nodes": [
            "MedicationClass",
            "ActiveIngredient",
            "Device",
            "ClinicalIndication",
            "AdverseEvent",
            "Administration",
            "Dosage"
        ],
        "relationships": [
            "CONTAINS",
            "TREATS",
            "CAUSES",
            "MONITORS",
            "REQUIRES_EDUCATION",
            "CONTRAINDICATED_WITH",
            "TITRATED_BY",
            "INDICATED_FOR"
        ]
    },
    "complications_management": {
        "nodes": [
            "Complication",
            "ScreeningTest",
            "TherapeuticAgent",
            "RiskFactor",
            "ReferralCriteria"
        ],
        "relationships": [
            "SCREENS_FOR",
            "TREATS",
            "INCREASES_RISK",
            "REQUIRES_REFERRAL",
            "DETECTED_BY",
            "MANAGED_BY",
            "REQUIRES_MONITORING"
        ]
    },
    "special_populations": {
        "nodes": [
            "PopulationSegment",
            "SpecificGoal",
            "AllowedMedication",
            "ContraindicatedMedication",
            "AgeRange"
        ],
        "relationships": [
            "HAS_GOAL",
            "ALLOWS",
            "CONTRAINDICATES",
            "APPLIES_TO",
            "HAS_TARGET_OVERRIDE",
            "CAN_USE",
            "MUST_AVOID",
            "REQUIRES_DEINTENSIFICATION"
        ]
    }
}

# Node and relationship properties
NODE_PROPERTIES = ["description", "source_text", "page_number", "evidence_level"]
RELATIONSHIP_PROPERTIES = ["evidence_level", "source_text", "conditional_context"]

# ============================================================================
# Logging Configuration
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FILE = BASE_DIR / "kg_builder.log"
