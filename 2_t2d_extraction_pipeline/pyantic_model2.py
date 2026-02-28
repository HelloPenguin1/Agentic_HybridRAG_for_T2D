from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union
from enum import Enum

class EvidenceLevel(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    E = "E"
    Unknown = "Unknown"

class BaseEntity(BaseModel):
    name: str = Field(default="Unknown")
    synonyms: List[str] = Field(default_factory=list)
    source_text: Optional[str] = Field(default="Unknown")
    page_number: Optional[Union[int, str]] = Field(default=None)
    evidence_level: Optional[Union[EvidenceLevel, str]] = Field(default="Unknown")

class BaseRelationship(BaseModel):
    source_entity: str = Field(default="Unknown")
    target_entity: str = Field(default="Unknown")
    relationship_type: str = Field(default="Unknown")
    evidence_level: Optional[Union[EvidenceLevel, str]] = Field(default="Unknown")
    conditional_context: Optional[str] = Field(default="None")

    # Added for traceability
    source_text: Optional[str] = Field(default="")
    confidence: Optional[float] = Field(default=None)

# --- Category specific models ---
# Keep the same high-level fields as your original file but relaxed. If you have the original
# classes defined elsewhere, you can merge or import them. The goal here is to ensure that
# relationship objects accept source_text and confidence so downstream code preserves evidence.

class AssessmentDiagnosisExtraction(BaseModel):
    diagnostic_tests: List[BaseEntity] = Field(default_factory=list)
    conditions: List[BaseEntity] = Field(default_factory=list)
    metric_values: List[BaseEntity] = Field(default_factory=list)
    screening_frequencies: List[BaseEntity] = Field(default_factory=list)
    target_goals: List[BaseEntity] = Field(default_factory=list)
    patient_profiles: List[BaseEntity] = Field(default_factory=list)
    relationships: List[BaseRelationship] = Field(default_factory=list)

class EducationLifestyleExtraction(BaseModel):
    interventions: List[BaseEntity] = Field(default_factory=list)
    behaviors: List[BaseEntity] = Field(default_factory=list)
    social_determinants: List[BaseEntity] = Field(default_factory=list)
    outcomes: List[BaseEntity] = Field(default_factory=list)
    screening_tools: List[BaseEntity] = Field(default_factory=list)
    relationships: List[BaseRelationship] = Field(default_factory=list)

class PharmacologyExtraction(BaseModel):
    medication_classes: List[BaseEntity] = Field(default_factory=list)
    active_ingredients: List[BaseEntity] = Field(default_factory=list)
    devices: List[BaseEntity] = Field(default_factory=list)
    clinical_indications: List[BaseEntity] = Field(default_factory=list)
    adverse_events: List[BaseEntity] = Field(default_factory=list)
    administration: List[BaseEntity] = Field(default_factory=list)
    dosage: List[BaseEntity] = Field(default_factory=list)
    relationships: List[BaseRelationship] = Field(default_factory=list)

class ComplicationsExtraction(BaseModel):
    complications: List[BaseEntity] = Field(default_factory=list)
    screening_tests: List[BaseEntity] = Field(default_factory=list)
    therapeutic_agents: List[BaseEntity] = Field(default_factory=list)
    risk_factors: List[BaseEntity] = Field(default_factory=list)
    referral_criteria: List[BaseEntity] = Field(default_factory=list)
    relationships: List[BaseRelationship] = Field(default_factory=list)

class SpecialPopulationsExtraction(BaseModel):
    special_populations: List[BaseEntity] = Field(default_factory=list)
    exception_rules: List[BaseEntity] = Field(default_factory=list)
    relationships: List[BaseRelationship] = Field(default_factory=list)
