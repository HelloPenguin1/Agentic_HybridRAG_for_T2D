
# ---------------------------------------------------------------------------
# Few-shot examples: Natural Language → Cypher Query

from langchain_core.prompts import ChatPromptTemplate

FEW_SHOT_EXAMPLES = """
Example 1
Question: List all diagnostic tests in the knowledge graph.
Cypher:
MATCH (n:DiagnosticTest)
RETURN n.id AS diagnostic_test
LIMIT 20

Example 2
Question: Find all A1C and glucose-related entities across all node types.
Cypher:
MATCH (n)
WHERE n.id IS NOT NULL
  AND (toLower(n.id) CONTAINS 'a1c'
    OR toLower(n.id) CONTAINS 'glucose'
    OR toLower(n.id) CONTAINS 'hemoglobin')
RETURN DISTINCT labels(n) AS node_types, n.id AS entity_id
LIMIT 30

Example 3
Question: What medications (therapeutic agents) are available for diabetes management?
Cypher:
MATCH (n:TherapeuticAgent)
RETURN n.id AS medication
LIMIT 30

Example 4
Question: Search for entities related to insulin, metformin, GLP-1, or SGLT2 medications.
Cypher:
MATCH (n)
WHERE n.id IS NOT NULL
  AND (toLower(n.id) CONTAINS 'metformin'
    OR toLower(n.id) CONTAINS 'insulin'
    OR toLower(n.id) CONTAINS 'glp'
    OR toLower(n.id) CONTAINS 'sglt')
RETURN DISTINCT labels(n) AS node_types, n.id AS entity_id
LIMIT 30

Example 5
Question: List all diabetes complications.
Cypher:
MATCH (n:Complication)
RETURN n.id AS complication
LIMIT 30

Example 6
Question: Find all entities related to kidney disease, retinopathy, neuropathy, or cardiovascular complications.
Cypher:
MATCH (n)
WHERE n.id IS NOT NULL
  AND (toLower(n.id) CONTAINS 'kidney'
    OR toLower(n.id) CONTAINS 'retinopathy'
    OR toLower(n.id) CONTAINS 'neuropathy'
    OR toLower(n.id) CONTAINS 'cardiovascular'
    OR toLower(n.id) CONTAINS 'ckd')
RETURN DISTINCT labels(n) AS node_types, n.id AS entity_id
LIMIT 30

Example 7
Question: Which medications treat which complications?
Cypher:
MATCH (agent:TherapeuticAgent)-[:TREATS]->(comp:Complication)
RETURN agent.id AS treatment, comp.id AS complication
LIMIT 20

Example 8
Question: What risk factors increase the likelihood of developing complications?
Cypher:
MATCH (rf:RiskFactor)-[:INCREASES_RISK]->(comp:Complication)
RETURN rf.id AS risk_factor, comp.id AS complication
LIMIT 20

Example 9
Question: What conditions or diseases do diagnostic tests diagnose?
Cypher:
MATCH (test:DiagnosticTest)-[:DIAGNOSES]->(condition)
RETURN test.id AS diagnostic_test,
       labels(condition) AS condition_type,
       condition.id AS condition
LIMIT 20

Example 10
Question: What target goals apply to which patient profiles?
Cypher:
MATCH (goal:TargetGoal)-[:APPLIES_TO]->(profile:PatientProfile)
RETURN goal.id AS target_goal, profile.id AS patient_profile
LIMIT 20

Example 11
Question: What medications are used to treat cardiovascular complications?
Cypher:
MATCH (c:Complication {id: "CVD"})<-[:TREATS]-(t:TherapeuticAgent)
RETURN t.id AS medication

Example 12
Question: What are the screening tests for diabetic retinopathy and how often should they be done?
Cypher:
MATCH (comp:Complication)-[:REQUIRES_SCREENING]->(freq:ScreeningFrequency)
WHERE toLower(comp.id) CONTAINS 'retinopathy'
RETURN comp.id AS complication, freq.id AS screening_frequency
LIMIT 10

Example 13
Question: Show me entity relationships between non-chunk nodes in the graph.
Cypher:
MATCH (a)-[r]->(b)
WHERE a.id IS NOT NULL AND b.id IS NOT NULL
  AND NOT 'Chunk' IN labels(a)
  AND NOT 'Chunk' IN labels(b)
RETURN labels(a)[0] AS from_type,
       a.id AS from_entity,
       type(r) AS relationship,
       labels(b)[0] AS to_type,
       b.id AS to_entity
LIMIT 30

Example 14
Question: Which conditions require monitoring of what tests or metrics?
Cypher:
MATCH (cond:Condition)-[:REQUIRES_MONITORING]->(test)
RETURN cond.id AS condition, labels(test) AS test_type, test.id AS monitored_item
LIMIT 20

Example 15
Question: What referral criteria exist for kidney-related complications?
Cypher:
MATCH (comp:Complication)-[:REQUIRES_REFERRAL]->(ref:ReferralCriteria)
WHERE toLower(comp.id) CONTAINS 'kidney' OR toLower(comp.id) CONTAINS 'ckd'
RETURN comp.id AS complication, ref.id AS referral_criteria
LIMIT 15
"""

# ---------------------------------------------------------------------------
# Main Cypher generation prompt template (inject {schema} and {question})
# ---------------------------------------------------------------------------

cypher_generation_prompt_template = ChatPromptTemplate.from_template(
    """Generate a Cypher query for Neo4j graph database (Type 2 Diabetes nursing knowledge).

Schema:
{schema}

Important:
Return ONLY the cypher query. Do not include any other text.

Rules:
1. Use ONLY schema elements above
2. Entities identified by 'id' property
3. Case-insensitive search: WHERE toLower(n.id) CONTAINS toLower('term')
4. READ-ONLY: Use MATCH/WHERE/RETURN/WITH/LIMIT only
5. Return properties, not counts

Examples:
{examples}

Question: {question}

Output (Cypher only, no explanation):"""
)

### Synthesizer Prompt ###

synthesizer_prompt = ChatPromptTemplate.from_template(
    """Your task is to transform retrieved information from a graph database into
a complete, coherent response for the user.

Your ONLY source of information is the vector retrieved data provided below.
Do NOT use any outside knowledge, assumptions, or information not present in the graph data.
If the graph data is empty or does not contain enough information to answer the question,
say so clearly — do not fabricate an answer.

Vector Data:
{vector_result}

Question: {question}

Answer:""")



