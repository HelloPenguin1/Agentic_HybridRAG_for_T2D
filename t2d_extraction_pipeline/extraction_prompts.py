"""
Extraction Prompts for T2D Clinical Knowledge Graph
Category-specific prompts based on ADA Standards of Care structure
"""

# ============================================================================
# CATEGORY 1: Assessment and Diagnosis
# ============================================================================

ASSESSMENT_DIAGNOSIS_PROMPT = """You are a clinical data specialist analyzing diabetes diagnosis and assessment guidelines.

Extract entities and relationships to construct a knowledge graph for clinical decision support.

**CRITICAL INSTRUCTIONS:**
1. Differentiate between 'Diagnosis' criteria and 'Goal' criteria
2. Link A1C targets to specific patient profiles (e.g., nonpregnant adults vs. healthy older adults)
3. Extract EXACT numeric values with their units
4. Capture screening frequencies with temporal context

**Text to Analyze:**
{text}

**Extract the following structured data:**

ENTITIES to identify:
- Diagnostic_Test: Tests used for diagnosis (e.g., A1C, FPG, OGTT)
- Condition: Diabetes types and related conditions
- Metric_Value: Specific numeric thresholds with units
- Screening_Frequency: How often to screen (with patient population context)
- Target_Goal: Treatment targets (distinguish from diagnostic thresholds)
- Patient_Profile: Patient characteristics that affect screening/targets

RELATIONSHIPS to identify:
- (Diagnostic_Test)-[:DIAGNOSES_CONDITION_AT_VALUE]->(Metric_Value)
- (Patient_Profile)-[:REQUIRES_SCREENING]->(Diagnostic_Test)
- (Condition)-[:HAS_DEFAULT_TARGET]->(Target_Goal)
- (Target_Goal)-[:DEFINED_BY]->(Metric_Value)

Return your response as a JSON object matching the AssessmentDiagnosisExtraction schema.
"""

# ============================================================================
# CATEGORY 2: Patient Education and Lifestyle
# ============================================================================

EDUCATION_LIFESTYLE_PROMPT = """You are a diabetes educator assistant analyzing lifestyle and prevention guidelines.

Extract entities and relationships to build a care-plan knowledge graph.

**CRITICAL INSTRUCTIONS:**
1. Link 'Metabolic Surgery' to the BMI criteria mentioned (if present)
2. Link 'DSMES' to the four critical times mentioned (if present)
3. Capture specific diet types with their recommendations (e.g., Mediterranean diet)
4. Extract mental health screening tools with thresholds

**Text to Analyze:**
{text}

**Extract the following structured data:**

ENTITIES to identify:
- Intervention: Healthcare interventions (e.g., DSMES, MNT, DPP, Metabolic Surgery)
- Behavior: Patient behaviors (e.g., Mediterranean diet, 150 min/week activity, Smoking cessation)
- Social_Determinant: SDOH barriers (e.g., Food insecurity, Financial barriers, Housing instability)
- Outcome: Expected outcomes (e.g., Weight loss >5%, A1C reduction, Reduced distress)
- Screening_Tool: Assessment instruments (e.g., Diabetes Distress Scale, PHQ-9)

RELATIONSHIPS to identify:
- (Intervention)-[:RECOMMENDS_BEHAVIOR]->(Behavior)
- (Behavior)-[:IMPROVES]->(Outcome)
- (Social_Determinant)-[:BARRIER_TO]->(Intervention)
- (Screening_Tool)-[:ASSESSES]->(Social_Determinant or Outcome)

Return your response as a JSON object matching the EducationLifestyleExtraction schema.
"""

# ============================================================================
# CATEGORY 3: Pharmacology and Technology
# ============================================================================

PHARMACOLOGY_TECHNOLOGY_PROMPT = """You are a clinical pharmacist assistant analyzing pharmacology and diabetes technology guidelines.

Extract entities for a medication decision-support graph with HIGH PRECISION.

**CRITICAL INSTRUCTIONS:**
1. Prioritize the 'Comorbidity-First' logic (e.g., If ASCVD → GLP-1 RA/SGLT2i)
2. Capture eGFR cutoffs for medication dosing adjustments
3. Extract injection techniques and lipohypertrophy prevention strategies
4. Map CGM metrics to clinical decision thresholds

**Text to Analyze:**
{text}

**Extract the following structured data:**

ENTITIES to identify:
- Medication_Class: Drug classes (e.g., SGLT2i, GLP-1 RA, Biguanides)
- Active_Ingredient: Specific medications (e.g., Metformin, Semaglutide, Tirzepatide)
- Device: Technology (e.g., CGM, AID System, Insulin Pen)
- Clinical_Indication: When to use (e.g., High ASCVD Risk, Heart Failure, CKD, Obesity)
- Adverse_Event: Side effects (e.g., Hypoglycemia, DKA, Genital mycotic infection)
- Administration: How to administer (e.g., Oral, Subcutaneous, Pump)
- Dosage: Dosing information including titration schedules

RELATIONSHIPS to identify:
- (Medication_Class)-[:CONTAINS_INGREDIENT]->(Active_Ingredient)
- (Clinical_Indication)-[:IS_TREATED_BY_PREFERRED]->(Medication_Class)
- (Medication_Class)-[:HAS_RISK]->(Adverse_Event)
- (Device)-[:REQUIRES_EDUCATION_ON]->(Administration)
- (Medication_Class)-[:CONTRAINDICATED_WITH]->(Condition/eGFR value)
- (Medication_Class)-[:TITRATED_BY]->(Dosage)

Return your response as a JSON object matching the PharmacologyExtraction schema.
"""

# ============================================================================
# CATEGORY 4: Complications Management
# ============================================================================

COMPLICATIONS_MANAGEMENT_PROMPT = """You are a nursing care coordinator analyzing complications management guidelines.

Extract the "If X complication, do Y protocol" logic for clinical workflows.

**CRITICAL INSTRUCTIONS:**
1. Map the CKD "Heat Map" logic (eGFR + Albuminuria stages) to specific actions
2. Extract BP targets with patient profile context (e.g., <130/80 for most, may vary)
3. Capture monofilament testing protocols and foot care algorithms
4. Link cardiovascular risk categories to specific medication recommendations

**Text to Analyze:**
{text}

**Extract the following structured data:**

ENTITIES to identify:
- Complication: Diabetes complications (e.g., Diabetic Retinopathy, CKD Stage 3, Charcot Foot)
- Screening_Test: Tests to detect complications (e.g., UACR, Dilated Eye Exam, 10g Monofilament)
- Therapeutic_Agent: Treatments (e.g., ACE inhibitor, Statin, Finerenone, Gabapentin)
- Risk_Factor: Risk factors (e.g., Hypertension, Dyslipidemia, Albuminuria)
- Referral_Criteria: When to refer to specialist

RELATIONSHIPS to identify:
- (Complication)-[:DETECTED_BY]->(Screening_Test)
- (Complication)-[:MANAGED_BY]->(Therapeutic_Agent)
- (Risk_Factor)-[:INCREASES_RISK_OF]->(Complication)
- (Therapeutic_Agent)-[:REQUIRES_MONITORING]->(Screening_Test)
- (Complication)-[:TRIGGERS_REFERRAL]->(ReferralCriteria)

Return your response as a JSON object matching the ComplicationsExtraction schema.
"""

# ============================================================================
# CATEGORY 5: Special Populations
# ============================================================================

SPECIAL_POPULATIONS_PROMPT = """You are a specialist nurse analyzing special population guidelines.

Extract "Exception Rules" that override general adult diabetes care guidelines.

**CRITICAL INSTRUCTIONS:**
1. Identify where targets differ from general adult population (e.g., A1C <8.0% for elderly vs. <7.0% for adults)
2. Extract pregnancy-specific glucose targets (e.g., Fasting <95 mg/dL)
3. Capture medication safety profiles by population (what's safe, what's contraindicated)
4. Link frailty/cognitive status to treatment deintensification decisions

**Text to Analyze:**
{text}

**Extract the following structured data:**

ENTITIES to identify:
- Population_Segment: Special populations (e.g., Older Adult with Frailty, Pregnant with T2D, Pediatric T2D)
- Specific_Goal: Population-specific targets (e.g., A1C < 8.0%, Fasting Glucose < 95 mg/dL)
- Allowed_Medication: Safe medications for population (e.g., Insulin, Metformin in pregnancy)
- Contraindicated_Medication: Medications to avoid (e.g., ACE inhibitors in pregnancy)
- Age_Range: Age-based categories with specific care considerations

RELATIONSHIPS to identify:
- (Population_Segment)-[:HAS_TARGET_OVERRIDE]->(Specific_Goal)
- (Population_Segment)-[:CAN_USE_MEDICATION]->(Allowed_Medication)
- (Population_Segment)-[:MUST_AVOID_MEDICATION]->(Contraindicated_Medication)
- (Population_Segment)-[:REQUIRES_DEINTENSIFICATION_IF]->(Condition)

Return your response as a JSON object matching the SpecialPopulationsExtraction schema.
"""

# ============================================================================
# General Instructions for All Categories
# ============================================================================

GENERAL_EXTRACTION_INSTRUCTIONS = """
**GENERAL RULES FOR ALL EXTRACTIONS:**

1. **Be Precise:** Extract exact wording from the text when possible. For numeric values, include the exact number and unit.

2. **Capture Context:** If a statement is conditional (e.g., "for patients with eGFR <30"), capture this in the conditional_context field.

3. **Evidence Levels:** If the text mentions ADA evidence levels (A, B, C, E), capture them.

4. **Synonyms:** Include common abbreviations and alternative names (e.g., "A1C" = "HbA1c" = "Hemoglobin A1C").

5. **Source Tracking:** Always include the source_text field with the exact snippet you extracted from.

6. **Relationships:** For every entity, try to identify at least one relationship with another entity.

7. **Avoid Hallucination:** Only extract information explicitly stated in the text. Do not infer or add information not present.

8. **Numeric Precision:** For dosages, targets, and thresholds, exact numbers matter. "6.5%" is different from "6.0%".

9. **Patient Populations:** Always specify which patient population a guideline applies to when mentioned.

10. **Temporal Information:** Capture timing information (e.g., "annually", "every 3-5 years", "at diagnosis").

11. **Relationship Density:** Extract approximately 1 relationship per entity. If you extract 20 entities, aim for 15-25 relationships.

12. **Mandatory Connections:** Before finalizing, scan your entity list:
    - Every Test → Must link to a Condition (DETECTED_BY)
    - Every Drug → Must link to a Condition (MANAGED_BY)
    - Every Complication → Must link to a Test or Risk Factor
    
13. **Context is Mandatory:** NEVER leave conditional_context empty. Use "General population" if no specific condition is stated.

14. **Interconnect Entities:** The goal is a CONNECTED graph, not isolated nodes. Force connections between related entities.
"""

# ============================================================================
# Prompt Templates Dictionary
# ============================================================================

EXTRACTION_PROMPTS = {
    "assessment_diagnosis": ASSESSMENT_DIAGNOSIS_PROMPT,
    "patient_education_lifestyle": EDUCATION_LIFESTYLE_PROMPT,
    "pharmacology_technology": PHARMACOLOGY_TECHNOLOGY_PROMPT,
    "complications_management": COMPLICATIONS_MANAGEMENT_PROMPT,
    "special_populations": SPECIAL_POPULATIONS_PROMPT
}


def get_extraction_prompt(category: str, text: str) -> str:
    """
    Get the appropriate extraction prompt for a category.
    
    Args:
        category: One of the 5 category names
        text: The text chunk to analyze
    
    Returns:
        Formatted prompt with instructions
    """
    base_prompt = EXTRACTION_PROMPTS.get(category, "")
    full_prompt = GENERAL_EXTRACTION_INSTRUCTIONS + "\n\n" + base_prompt.format(text=text)
    return full_prompt