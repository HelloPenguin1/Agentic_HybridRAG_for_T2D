
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
6. Multi-relationship syntax: use [:REL_A|REL_B] NOT [:REL_A|:REL_B] (no colon after pipe)

Domain Vocabulary (always map these user terms to the correct node label):
- medication / drug / medicine / treatment / agent    → TherapeuticAgent
- complication / condition / disease / disorder       → Complication
- test / lab / screening / diagnostic / measurement   → DiagnosticTest
- risk factor / risk                                  → RiskFactor
- goal / target / threshold / recommended value       → TargetGoal
- patient profile / patient type / population         → PatientProfile
- referral / specialist                               → ReferralCriteria
- monitoring / follow-up / frequency                  → ScreeningFrequency

Examples:
{examples}

Question: {question}

Output (Cypher only, no explanation):"""
)


router_prompt = ChatPromptTemplate.from_template(
    """You are a query routing agent for a hybrid retrieval system that combines:
    
1. **Knowledge Graph** (Neo4j): Structured facts, entities, relationships
   - Good for: Entity lists, relationships, medications, contraindications, diagnostic criteria
   - Examples: "What treats Y?", "List all Z", "What causes X?"

2. **Vector Database** (semantic search): Clinical guidelines, protocols, procedures, explanations
   - Good for: How-to instructions, patient education, clinical reasoning, best practices, definitions
   - Examples: "How do I...", "What is the protocol for...", "Explain why..."

Your task: Analyze the question and decide which retrieval source(s) to use.

Question: {question}

Classification Guidelines:

**Route to GRAPH_ONLY when:**
- Question requests specific entity lists or counts ("list all medications", "list all complications")
- Question involves a known named relationship between two specific entities
- Question asks for a concrete property of one specific entity (dose, frequency)
- Examples:
  * "List all diabetes complications"
  * "What medications treat cardiovascular disease?"
  * "Which TherapeuticAgents are linked to metformin?"

**Route to VECTOR_ONLY when:**
- Question asks for procedures, protocols, or clinical guidelines
- Question involves patient education or teaching strategies
- Question asks "how to", "why", "explain the process of"
- Examples:
  * "How do I teach insulin injection technique?"
  * "Explain the protocol for hypoglycemia management"
  * "What are best practices for diabetic foot care?"

**Route to BOTH when:**
- Question asks "what is X?" or requests a definition/overview of a condition, medication, or concept
  (graph provides relationships; vector provides explanatory text)
- Question requires both factual data AND contextual explanation
- Question involves clinical decision-making combining facts + protocols
- Question has multiple parts requiring different sources
- Examples:
  * "What is CKD?" → graph: relationships of CKD; vector: definition and context
  * "What is metformin?" → graph: what it treats; vector: mechanism and guidance
  * "Patient on metformin has eGFR 28, what should I do?"
  * "What is the dose of insulin and how do I teach injection technique?"
  * "Compare metformin and SGLT2 inhibitors for CKD patients"

Respond ONLY based on the structured output schema and provide router reasoning/decision thought process.
You must always provide a reasoning
"""
)



### Synthesizer Prompt ###

synthesizer_prompt = ChatPromptTemplate.from_template(
    """Your task is to synthesize a complete, coherent response using the retrieved data below.

Your ONLY sources of information are the Vector Data and Graph Data provided.
Do NOT use any outside knowledge or information not present in the retrieved data.
If both sources are empty or insufficient to answer the question, say so clearly — do not fabricate an answer.
If one source is empty, rely on the other.

Vector Data (semantic chunk retrieval):
{vector_result}

Graph Data (knowledge graph retrieval):
{graph_result}

Question: {question}

Answer:""")


### GraphCypherQAChain Prompts ###
# GraphCypherQAChain requires PromptTemplate (not ChatPromptTemplate)
from langchain_core.prompts import PromptTemplate

cypher_chain_generation_prompt = PromptTemplate(
    input_variables=["schema", "question"],
    template="""You are an expert in Neo4j Cypher for a Type 2 Diabetes nursing knowledge graph.
Generate a Cypher query to answer the question below.

Schema:
{schema}

Rules:
1. Use ONLY node labels and relationship types present in the schema above
2. Entities are identified by their 'id' property
3. Case-insensitive search: WHERE toLower(n.id) CONTAINS toLower('term')
4. READ-ONLY queries only: MATCH, WHERE, RETURN, WITH, LIMIT
5. Return properties, not node objects
6. Multi-relationship syntax: [:REL_A|REL_B] NOT [:REL_A|:REL_B]
7. ALWAYS use descriptive aliases in RETURN (e.g. RETURN t.id AS medication, NOT RETURN t.id)
8. NEVER return embedding fields — only return .id and other human-readable properties
9. For definition/overview questions ("what is X?"), explore all relationships of that entity:
   MATCH (n)-[r]->(m) WHERE toLower(n.id) CONTAINS 'term' AND n.id IS NOT NULL AND m.id IS NOT NULL
   RETURN type(r) AS relationship, m.id AS related_entity, labels(m)[0] AS related_type LIMIT 20

Domain Vocabulary (map user terms → graph node labels):
- medication / drug / medicine / treatment → TherapeuticAgent
- complication / condition / disease       → Complication
- test / screening / lab / diagnostic      → DiagnosticTest
- risk factor / risk                       → RiskFactor
- goal / target / threshold               → TargetGoal
- patient profile / patient type          → PatientProfile
- referral / specialist                   → ReferralCriteria
- monitoring / follow-up / frequency      → ScreeningFrequency

Return ONLY the Cypher query. No explanations, no markdown code blocks.

Question: {question}
Cypher query:"""
)

cypher_chain_qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a clinical assistant helping nurses manage Type 2 Diabetes patients.
Use the information below (retrieved from a graph database) to answer the question.
The information is authoritative — use it as-is, do not use outside knowledge.
Make the answer sound like a direct response to the question.
Do not mention the graph or database in your answer.
If the information is empty, say that you don't know.

Example:
Question: What medications treat CVD?
Information: [{{'medication': 'GLP-1 RAs'}}, {{'medication': 'SGLT2 inhibitor'}}]
Answer: The medications that treat CVD include GLP-1 RAs and SGLT2 inhibitors.

Information:
{context}

Question: {question}
Answer:"""
)
