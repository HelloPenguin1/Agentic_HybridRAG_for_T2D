"""
Pydantic Models for T2D Clinical Knowledge Graph Extraction
Enforces strict typing and validation for LLM-based entity and relationship extraction.
Updated with "Unknown" literals and optional fields to prevent extraction crashes.
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union
from enum import Enum


# ============================================================================
# Common Base Models
# ============================================================================

class EvidenceLevel(str, Enum):
    """ADA Evidence Levels"""
    A = "A"  # Clear evidence from RCTs
    B = "B"  # Supportive evidence from cohort studies
    C = "C"  # Supportive evidence from poorly controlled studies
    E = "E"  # Expert consensus
    Unknown = "Unknown"



class BaseEntity(BaseModel):
    name: str = Field(description="The canonical name of the entity")
    synonyms: List[str] = Field(default_factory=list, description="Alternative names or abbreviations")
    source_text: Optional[str] = Field(default="", description="The exact text snippet")
    page_number: Optional[int] = Field(default=None, description="Source page number")
    evidence_level: Optional[EvidenceLevel] = Field(default=EvidenceLevel.Unknown)


class BaseRelationship(BaseModel):
    """Base model for all relationships"""
    source_entity: str = Field(description="The source entity name")
    target_entity: str = Field(description="The target entity name")
    relationship_type: str = Field(description="The type of relationship")
    evidence_level: Optional[EvidenceLevel] = Field(default=EvidenceLevel.Unknown)
    conditional_context: Optional[str] = Field(default=None, description="Any conditions that apply (e.g., 'if eGFR < 30')")


# ============================================================================
# CATEGORY 1: Assessment and Diagnosis Models
# ============================================================================

class DiagnosticTest(BaseEntity):
    """Model for diagnostic tests"""
    test_type: Optional[Literal["Lab", "Physical", "Imaging", "Questionnaire", "Unknown"]] = Field(default="Unknown")
    normal_range: Optional[str] = Field(default=None, description="Normal reference range if applicable")
    units: Optional[str] = Field(default=None, description="Units of measurement (e.g., 'mg/dL', '%')")


class Condition(BaseEntity):
    """Model for medical conditions"""
    icd10_code: Optional[str] = Field(default=None)
    classification: Optional[Literal["Type 1 Diabetes", "Type 2 Diabetes", "Prediabetes", "Gestational Diabetes", "Other", "Unknown"]] = Field(default="Unknown")


class MetricValue(BaseEntity):
    """Model for specific metric values (thresholds, targets)"""
    value: Optional[float] = Field(default=None, description="The numeric value")
    unit: Optional[str] = Field(default=None, description="Unit of measurement")
    comparator: Optional[Literal["<", "<=", ">", ">=", "=", "range", "Unknown"]] = Field(default="Unknown", description="How to interpret the value")
    upper_bound: Optional[float] = Field(default=None, description="For range values")


class ScreeningFrequency(BaseEntity):
    """Model for screening schedules"""
    frequency: str = Field(description="How often (e.g., 'Annually', 'Every 3 years')")
    applies_to: str = Field(description="Patient population this applies to")


class TargetGoal(BaseEntity):
    """Model for treatment targets"""
    goal_type: Optional[Literal["A1C", "Fasting Glucose", "Postprandial Glucose", "Time in Range", "Blood Pressure", "LDL", "Other", "Unknown"]] = Field(default="Unknown")
    target_value: str = Field(description="The target value (e.g., '<7.0%')")
    patient_population: str = Field(description="Which patient group this applies to")


class PatientProfile(BaseEntity):
    """Model for patient demographics/characteristics"""
    age_range: Optional[str] = Field(default=None, description="e.g., 'Adult', '<18', '>65'")
    bmi_range: Optional[str] = Field(default=None)
    comorbidities: List[str] = Field(default_factory=list)
    special_status: Optional[str] = Field(default=None, description="e.g., 'Pregnant', 'Frail', 'Asymptomatic'")


class AssessmentDiagnosisRelationship(BaseRelationship):
    """Relationships for assessment/diagnosis category"""
    relationship_type: Optional[Literal[
        "DIAGNOSES_CONDITION_AT_VALUE",
        "REQUIRES_SCREENING",
        "HAS_DEFAULT_TARGET",
        "DEFINED_BY",
        "Unknown"
    ]] = Field(default="Unknown")


class AssessmentDiagnosisExtraction(BaseModel):
    """Complete extraction for Assessment & Diagnosis category"""
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
    """Model for healthcare interventions"""
    intervention_type: Optional[Literal["Education", "Counseling", "Program", "Surgery", "Other", "Unknown"]] = Field(default="Unknown")
    duration: Optional[str] = Field(default=None, description="Duration if specified")
    frequency: Optional[str] = Field(default=None, description="How often administered")


class Behavior(BaseEntity):
    """Model for patient behaviors"""
    behavior_category: Optional[Literal["Diet", "Exercise", "Smoking", "Sleep", "Medication Adherence", "Other", "Unknown"]] = Field(default="Unknown")
    target_amount: Optional[str] = Field(default=None, description="e.g., '150 min/week', '>5% weight loss'")


class SocialDeterminant(BaseEntity):
    """Model for social determinants of health"""
    sdoh_category: Optional[Literal["Food Security", "Housing", "Financial", "Transportation", "Health Literacy", "Other", "Unknown"]] = Field(default="Unknown")
    screening_tool: Optional[str] = Field(default=None, description="Tool to assess this SDOH")


class Outcome(BaseEntity):
    """Model for health outcomes"""
    outcome_type: Optional[Literal["Weight Loss", "A1C Reduction", "Improved Distress", "Remission", "Other", "Unknown"]] = Field(default="Unknown")
    magnitude: Optional[str] = Field(default=None, description="Expected improvement (e.g., '>5%')")


class ScreeningTool(BaseEntity):
    """Model for screening instruments"""
    tool_type: Optional[Literal["Questionnaire", "Scale", "Assessment", "Unknown"]] = Field(default="Unknown")
    threshold: Optional[str] = Field(default=None, description="Cutoff score for positive screen")
    what_it_measures: str = Field(description="What this tool screens for")


class EducationLifestyleRelationship(BaseRelationship):
    """Relationships for education/lifestyle category"""
    relationship_type: Optional[Literal[
        "RECOMMENDS_BEHAVIOR",
        "IMPROVES",
        "BARRIER_TO",
        "ASSESSES",
        "Unknown"
    ]] = Field(default="Unknown")


class EducationLifestyleExtraction(BaseModel):
    """Complete extraction for Education & Lifestyle category"""
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
    """Model for medication classes"""
    class_name: str = Field(description="Drug class name (e.g., 'SGLT2i', 'GLP-1 RA')")
    mechanism_of_action: Optional[str] = Field(default=None)
    route: Optional[Literal["Oral", "Subcutaneous", "Intravenous", "Inhaled", "Other", "Unknown"]] = Field(default="Unknown")


class ActiveIngredient(BaseEntity):
    """Model for specific medications"""
    generic_name: str
    brand_names: List[str] = Field(default_factory=list)
    dosage_forms: List[str] = Field(default_factory=list, description="e.g., ['500mg tablet', '1000mg tablet']")


class Device(BaseEntity):
    """Model for diabetes technology devices"""
    device_type: Optional[Literal["CGM", "Insulin Pump", "AID System", "Connected Pen", "Meter", "Other", "Unknown"]] = Field(default="Unknown")
    brand: Optional[str] = Field(default=None)
    features: List[str] = Field(default_factory=list)


class ClinicalIndication(BaseEntity):
    """Model for clinical indications for treatment"""
    indication_type: Optional[Literal["Primary", "Secondary", "Off-label", "Unknown"]] = Field(default="Unknown")
    condition: str = Field(description="The condition being treated")
    priority: Optional[int] = Field(default=None, description="1 = first-line, 2 = second-line, etc.")


class AdverseEvent(BaseEntity):
    """Model for side effects and adverse events"""
    severity: Optional[Literal["Mild", "Moderate", "Severe", "Life-threatening", "Unknown"]] = Field(default="Unknown")
    frequency: Optional[str] = Field(default=None, description="How common (e.g., 'Common', 'Rare')")


class Administration(BaseEntity):
    """Model for administration instructions"""
    route: str
    site: Optional[str] = Field(default=None, description="Injection site if applicable")
    technique: Optional[str] = Field(default=None, description="Special technique requirements")


class Dosage(BaseEntity):
    """Model for dosing information"""
    starting_dose: str
    max_dose: str
    titration_schedule: Optional[str] = Field(default=None)
    adjustments: Optional[str] = Field(default=None, description="Dose adjustments for renal/hepatic impairment")


class PharmacologyRelationship(BaseRelationship):
    """Relationships for pharmacology/technology category"""
    relationship_type: Optional[Literal[
        "CONTAINS_INGREDIENT",
        "IS_TREATED_BY_PREFERRED",
        "HAS_RISK",
        "REQUIRES_EDUCATION_ON",
        "CONTRAINDICATED_WITH",
        "TITRATED_BY",
        "Unknown"
    ]] = Field(default="Unknown")


class PharmacologyExtraction(BaseModel):
    """Complete extraction for Pharmacology & Technology category"""
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
    """Model for diabetes complications"""
    complication_category: Optional[Literal["Cardiovascular", "Renal", "Retinopathy", "Neuropathy", "Foot", "Other", "Unknown"]] = Field(default="Unknown")
    stage: Optional[str] = Field(default=None, description="Staging if applicable (e.g., 'CKD Stage 3')")


class ScreeningTest(BaseEntity):
    """Model for complication screening tests"""
    test_name: str
    frequency: str = Field(description="How often to perform")
    indication: str = Field(description="When to screen")


class TherapeuticAgent(BaseEntity):
    """Model for medications used to manage complications"""
    agent_type: Optional[Literal["Medication", "Procedure", "Lifestyle Modification", "Unknown"]] = Field(default="Unknown")
    when_to_use: str = Field(description="Clinical scenario for use")


class RiskFactor(BaseEntity):
    """Model for risk factors"""
    factor_type: Optional[Literal["Modifiable", "Non-modifiable", "Unknown"]] = Field(default="Unknown")
    impact_magnitude: Optional[str] = Field(default=None, description="e.g., '2x risk', 'High risk'")


class ReferralCriteria(BaseEntity):
    """Model for when to refer to specialist"""
    specialist_type: str = Field(description="e.g., 'Ophthalmologist', 'Nephrologist'")
    urgency: Optional[Literal["Routine", "Urgent", "Emergent", "Unknown"]] = Field(default="Unknown")
    criteria: str = Field(description="What triggers the referral")


class ComplicationsRelationship(BaseRelationship):
    """Relationships for complications category"""
    relationship_type: Optional[Literal[
        "DETECTED_BY",
        "MANAGED_BY",
        "INCREASES_RISK_OF",
        "REQUIRES_MONITORING",
        "TRIGGERS_REFERRAL",
        "Unknown"
    ]] = Field(default="Unknown")


class ComplicationsExtraction(BaseModel):
    """Complete extraction for Complications Management category"""
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
    """Model for special populations"""
    population_type: Optional[Literal["Pediatric", "Older Adult", "Pregnancy", "Other", "Unknown"]] = Field(default="Unknown")
    age_range: Optional[str] = Field(default=None)
    defining_characteristics: List[str] = Field(default_factory=list)


class SpecificGoal(BaseEntity):
    """Model for population-specific goals"""
    goal_parameter: str = Field(description="What is being measured (e.g., 'A1C', 'Fasting Glucose')")
    target_value: str = Field(description="The specific target")
    rationale: Optional[str] = Field(default=None, description="Why this target differs")


class AllowedMedication(BaseEntity):
    """Model for medications safe for population"""
    medication_name: str
    safety_level: Optional[Literal["Preferred", "Acceptable", "Use with Caution", "Unknown"]] = Field(default="Unknown")
    special_instructions: Optional[str] = Field(default=None)


class ContraindicatedMedication(BaseEntity):
    """Model for medications to avoid"""
    medication_name: str
    reason: str = Field(description="Why it's contraindicated")
    alternative: Optional[str] = Field(default=None, description="What to use instead")


class AgeRange(BaseEntity):
    """Model for age-based categories"""
    minimum_age: Optional[int] = Field(default=None)
    maximum_age: Optional[int] = Field(default=None)
    age_description: str = Field(description="e.g., 'School-age', 'Geriatric'")


class SpecialPopulationsRelationship(BaseRelationship):
    """Relationships for special populations category"""
    relationship_type: Optional[Literal[
        "HAS_TARGET_OVERRIDE",
        "CAN_USE_MEDICATION",
        "MUST_AVOID_MEDICATION",
        "REQUIRES_DEINTENSIFICATION_IF",
        "Unknown"
    ]] = Field(default="Unknown")


class SpecialPopulationsExtraction(BaseModel):
    """Complete extraction for Special Populations category"""
    population_segments: List[PopulationSegment] = Field(default_factory=list)
    specific_goals: List[SpecificGoal] = Field(default_factory=list)
    allowed_medications: List[AllowedMedication] = Field(default_factory=list)
    contraindicated_medications: List[ContraindicatedMedication] = Field(default_factory=list)
    age_ranges: List[AgeRange] = Field(default_factory=list)
    relationships: List[SpecialPopulationsRelationship] = Field(default_factory=list)