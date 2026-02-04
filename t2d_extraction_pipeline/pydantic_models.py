"""
Pydantic Models for T2D Clinical Knowledge Graph Extraction
Enforces loose typing and high-recall validation for LLM-based extraction.
Updated to bypass strict validation issues while preserving clinical data.
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union
from enum import Enum


# ============================================================================
# Common Base Models
# ============================================================================

class EvidenceLevel(str, Enum):
    """ADA Evidence Levels - Relaxed to allow string-based fallback"""
    A = "A"
    B = "B"
    C = "C"
    E = "E"
    Unknown = "Unknown"


class BaseEntity(BaseModel):
    """Base model with relaxed validation to prevent extraction crashes."""
    name: str = Field(default="Unknown", description="The canonical name of the entity")
    synonyms: List[str] = Field(default_factory=list, description="Alternative names")
    source_text: Optional[str] = Field(default="Unknown", description="The exact text snippet")
    page_number: Optional[Union[int, str]] = Field(default=None) 
    evidence_level: Optional[Union[EvidenceLevel, str]] = Field(default="Unknown")


class BaseRelationship(BaseModel):
    """Base model for all relationships with safe defaults."""
    source_entity: str = Field(default="Unknown")
    target_entity: str = Field(default="Unknown")
    relationship_type: str = Field(default="Unknown")
    evidence_level: Optional[Union[EvidenceLevel, str]] = Field(default="Unknown")
    conditional_context: Optional[str] = Field(default="None")


# ============================================================================
# CATEGORY 1: Assessment and Diagnosis Models
# ============================================================================

class DiagnosticTest(BaseEntity):
    test_type: Optional[str] = Field(default="Unknown")
    normal_range: Optional[str] = Field(default=None)
    units: Optional[str] = Field(default=None)


class Condition(BaseEntity):
    icd10_code: Optional[str] = Field(default=None)
    classification: Optional[str] = Field(default="Unknown")


class MetricValue(BaseEntity):
    value: Optional[Union[float, str]] = Field(default=None)
    unit: Optional[str] = Field(default=None)
    comparator: Optional[str] = Field(default="Unknown")
    upper_bound: Optional[Union[float, str]] = Field(default=None)


class ScreeningFrequency(BaseEntity):
    frequency: str = Field(default="Unknown")
    applies_to: str = Field(default="Unknown")


class TargetGoal(BaseEntity):
    goal_type: Optional[str] = Field(default="Unknown")
    target_value: str = Field(default="Unknown")
    patient_population: str = Field(default="Unknown")


class PatientProfile(BaseEntity):
    age_range: Optional[str] = Field(default=None)
    bmi_range: Optional[str] = Field(default=None)
    comorbidities: List[str] = Field(default_factory=list)
    special_status: Optional[str] = Field(default=None)


class AssessmentDiagnosisRelationship(BaseRelationship):
    relationship_type: Optional[str] = Field(default="Unknown")


class AssessmentDiagnosisExtraction(BaseModel):
    diagnostic_tests: List[DiagnosticTest] = Field(default_factory=list)
    conditions: List[Condition] = Field(default_factory=list)
    metric_values: List[MetricValue] = Field(default_factory=list)
    screening_frequencies: List[ScreeningFrequency] = Field(default_factory=list)
    target_goals: List[TargetGoal] = Field(default_factory=list)
    patient_profiles: List[PatientProfile] = Field(default_factory=list)
    relationships: List[AssessmentDiagnosisRelationship] = Field(default_factory=list)


# ============================================================================
# CATEGORY 2: Patient Education and Lifestyle Models
# ============================================================================

class Intervention(BaseEntity):
    intervention_type: Optional[str] = Field(default="Unknown")
    duration: Optional[str] = Field(default=None)
    frequency: Optional[str] = Field(default=None)


class Behavior(BaseEntity):
    behavior_category: Optional[str] = Field(default="Unknown")
    target_amount: Optional[str] = Field(default=None)


class SocialDeterminant(BaseEntity):
    sdoh_category: Optional[str] = Field(default="Unknown")
    screening_tool: Optional[str] = Field(default=None)


class Outcome(BaseEntity):
    outcome_type: Optional[str] = Field(default="Unknown")
    magnitude: Optional[str] = Field(default=None)


class ScreeningTool(BaseEntity):
    tool_type: Optional[str] = Field(default="Unknown")
    threshold: Optional[str] = Field(default=None)
    what_it_measures: str = Field(default="Unknown")


class EducationLifestyleRelationship(BaseRelationship):
    relationship_type: Optional[str] = Field(default="Unknown")


class EducationLifestyleExtraction(BaseModel):
    interventions: List[Intervention] = Field(default_factory=list)
    behaviors: List[Behavior] = Field(default_factory=list)
    social_determinants: List[SocialDeterminant] = Field(default_factory=list)
    outcomes: List[Outcome] = Field(default_factory=list)
    screening_tools: List[ScreeningTool] = Field(default_factory=list)
    relationships: List[EducationLifestyleRelationship] = Field(default_factory=list)


# ============================================================================
# CATEGORY 3: Pharmacology and Technology Models
# ============================================================================

class MedicationClass(BaseEntity):
    class_name: str = Field(default="Unknown")
    mechanism_of_action: Optional[str] = Field(default=None)
    route: Optional[str] = Field(default="Unknown")


class ActiveIngredient(BaseEntity):
    generic_name: str = Field(default="Unknown")
    brand_names: List[str] = Field(default_factory=list)
    dosage_forms: List[str] = Field(default_factory=list)


class Device(BaseEntity):
    device_type: Optional[str] = Field(default="Unknown")
    brand: Optional[str] = Field(default=None)
    features: List[str] = Field(default_factory=list)


class ClinicalIndication(BaseEntity):
    indication_type: Optional[str] = Field(default="Unknown")
    condition: str = Field(default="Unknown")
    priority: Optional[Union[int, str]] = Field(default=None)


class AdverseEvent(BaseEntity):
    severity: Optional[str] = Field(default="Unknown")
    frequency: Optional[str] = Field(default=None)


class Administration(BaseEntity):
    route: str = Field(default="Unknown")
    site: Optional[str] = Field(default=None)
    technique: Optional[str] = Field(default=None)


class Dosage(BaseEntity):
    starting_dose: str = Field(default="Unknown")
    max_dose: str = Field(default="Unknown")
    titration_schedule: Optional[str] = Field(default=None)
    adjustments: Optional[str] = Field(default=None)


class PharmacologyRelationship(BaseRelationship):
    relationship_type: Optional[str] = Field(default="Unknown")


class PharmacologyExtraction(BaseModel):
    medication_classes: List[MedicationClass] = Field(default_factory=list)
    active_ingredients: List[ActiveIngredient] = Field(default_factory=list)
    devices: List[Device] = Field(default_factory=list)
    clinical_indications: List[ClinicalIndication] = Field(default_factory=list)
    adverse_events: List[AdverseEvent] = Field(default_factory=list)
    administrations: List[Administration] = Field(default_factory=list)
    dosages: List[Dosage] = Field(default_factory=list)
    relationships: List[PharmacologyRelationship] = Field(default_factory=list)


# ============================================================================
# CATEGORY 4: Complications Management Models
# ============================================================================

class Complication(BaseEntity):
    complication_category: Optional[str] = Field(default="Unknown")
    stage: Optional[str] = Field(default=None)


class ScreeningTest(BaseEntity):
    test_name: str = Field(default="Unknown")
    frequency: str = Field(default="Unknown")
    indication: str = Field(default="Unknown")


class TherapeuticAgent(BaseEntity):
    agent_type: Optional[str] = Field(default="Unknown")
    when_to_use: str = Field(default="Unknown")


class RiskFactor(BaseEntity):
    factor_type: Optional[str] = Field(default="Unknown")
    impact_magnitude: Optional[str] = Field(default=None)


class ReferralCriteria(BaseEntity):
    specialist_type: str = Field(default="Unknown")
    urgency: Optional[str] = Field(default="Unknown")
    criteria: str = Field(default="Unknown")


class ComplicationsRelationship(BaseRelationship):
    relationship_type: Optional[str] = Field(default="Unknown")


class ComplicationsExtraction(BaseModel):
    complications: List[Complication] = Field(default_factory=list)
    screening_tests: List[ScreeningTest] = Field(default_factory=list)
    therapeutic_agents: List[TherapeuticAgent] = Field(default_factory=list)
    risk_factors: List[RiskFactor] = Field(default_factory=list)
    referral_criteria: List[ReferralCriteria] = Field(default_factory=list)
    relationships: List[ComplicationsRelationship] = Field(default_factory=list)


# ============================================================================
# CATEGORY 5: Special Populations Models
# ============================================================================

class PopulationSegment(BaseEntity):
    population_type: Optional[str] = Field(default="Unknown")
    age_range: Optional[str] = Field(default=None)
    defining_characteristics: List[str] = Field(default_factory=list)


class SpecificGoal(BaseEntity):
    goal_parameter: str = Field(default="Unknown")
    target_value: str = Field(default="Unknown")
    rationale: Optional[str] = Field(default=None)


class AllowedMedication(BaseEntity):
    medication_name: str = Field(default="Unknown")
    safety_level: Optional[str] = Field(default="Unknown")
    special_instructions: Optional[str] = Field(default=None)


class ContraindicatedMedication(BaseEntity):
    medication_name: str = Field(default="Unknown")
    reason: str = Field(default="Unknown")
    alternative: Optional[str] = Field(default=None)


class AgeRange(BaseEntity):
    minimum_age: Optional[Union[int, str]] = Field(default=None)
    maximum_age: Optional[Union[int, str]] = Field(default=None)
    age_description: str = Field(default="Unknown")


class SpecialPopulationsRelationship(BaseRelationship):
    relationship_type: Optional[str] = Field(default="Unknown")


class SpecialPopulationsExtraction(BaseModel):
    population_segments: List[PopulationSegment] = Field(default_factory=list)
    specific_goals: List[SpecificGoal] = Field(default_factory=list)
    allowed_medications: List[AllowedMedication] = Field(default_factory=list)
    contraindicated_medications: List[ContraindicatedMedication] = Field(default_factory=list)
    age_ranges: List[AgeRange] = Field(default_factory=list)
    relationships: List[SpecialPopulationsRelationship] = Field(default_factory=list)