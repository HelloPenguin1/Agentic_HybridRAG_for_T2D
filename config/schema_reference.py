
#Schema for Nodes and Relationship

SCHEMAS: Dict[str, Dict[str, List[str]]] = {
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