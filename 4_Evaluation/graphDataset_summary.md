# Question Classification Summary

## Overview
All 28 questions in the dataset have been classified into three categories based on their graph traversal patterns and the schema structure.

## Classification Results

### Single-hop Property Retrieval: 9 questions (Q1-Q9)
**Definition**: Questions that retrieve properties directly from Drug nodes without traversing any relationships.

**Questions**:
1. What are the available dosages for Insulin lispro?
2. What is the half-life of Metformin?
3. What are the toxicity risks associated with Metformin?
4. What is the mechanism of action for Glimepiride?
5. What are the indications for Acarbose?
6. What is the half-life of Rosiglitazone?
7. What are the toxicity concerns for Nateglinide?
8. What is the mechanism of action for Acarbose?
9. What are the available dosages for Tolazamide?

**Why Single-hop**: These questions use Cypher patterns like `MATCH (d:Drug) WHERE ... RETURN d.property` without any relationship traversal (no `-[]->`). They access intrinsic Drug node properties such as `available_dosages`, `half_life`, `toxicity`, `mechanism_of_action`, and `indication`.

---

### Multi-hop Relationships: 9 questions (Q10-Q18)
**Definition**: Questions that traverse relationships to access connected nodes like other Drugs (interactions), Products, or FoodInteractions.

**Questions**:
10. Does Metformin interact with Glimepiride?
11. What drugs interact with Metformin?
12. What are the brand names for Insulin glargine?
13. What are the dietary rules for Acarbose?
14. What products is Metformin marketed as?
15. Does Rosiglitazone interact with Insulin glargine?
16. What are all the drugs that interact with Glimepiride?
17. What food interactions exist for Miglitol?
18. What products is Nateglinide sold as?

**Why Multi-hop**: These questions traverse specific relationships:
- **INTERACTS_WITH → Drug**: Accesses drug-drug interactions and interaction severity descriptions (Q10, Q11, Q15, Q16)
- **MARKETED_AS → Product**: Accesses commercial brand information (Q12, Q14, Q18)
- **HAS_DIETARY_RULE → FoodInteraction**: Retrieves dietary restrictions and food-related warnings (Q13, Q17)

The Cypher patterns include relationship traversal like `MATCH (d1:Drug)-[r:INTERACTS_WITH]-(d2:Drug)`, `MATCH (d:Drug)-[:MARKETED_AS]->(p:Product)`, or `MATCH (d:Drug)-[:HAS_DIETARY_RULE]->(f:FoodInteraction)`.

---

### Categorization & Mechanisms: 10 questions (Q19-Q28)
**Definition**: Questions about drug classifications, biological targets, and pharmacological mechanisms that require traversing categorical or target relationships.

**Questions**:
19. What pharmacological category does Metformin belong to?
20. What is the drug class for Glimepiride?
21. What biological target does Insulin lispro act on?
22. What drugs are in the same class as Acetohexamide?
23. What is the mechanism and target for Rosiglitazone?
24. What category does Acarbose belong to?
25. What other drugs are in the same class as Miglitol?
26. What target does Nateglinide act on and how?
27. What drugs belong to the same class as Tolazamide?
28. What is the pharmacological target and action of Insulin glargine?

**Why Categorization & Mechanisms**:
- **Drug Classification (Q19, Q20, Q24)**: Single-hop traversal via `BELONGS_TO → Category` to identify drug classes
- **Biological Targets (Q21, Q23, Q26, Q28)**: Traversal via `ACTS_ON → Target` to retrieve molecular targets and action types (agonist, inhibitor, etc.)
- **Same-class Drugs (Q22, Q25, Q27)**: Two-hop traversal pattern: `Drug → BELONGS_TO → Category ← BELONGS_TO ← Drug` to find drugs sharing the same pharmacological class

The Cypher patterns include `MATCH (d:Drug)-[:BELONGS_TO]->(c:Category)` or `MATCH (d:Drug)-[:ACTS_ON]->(t:Target)` with bidirectional class-matching patterns.

---

## Distribution Analysis

| Category | Count | Percentage |
|----------|-------|------------|
| Single-hop property retrieval | 9 | 32.1% |
| Multi-hop relationships | 9 | 32.1% |
| Categorization & Mechanisms | 10 | 35.7% |
| **Total** | **28** | **100%** |

**Note**: The dataset contains 28 questions (not the 30 mentioned in the original spec). The distribution is nearly balanced across the three categories.

---

## Key Insights

1. **Complete Coverage**: All 28 questions are accounted for with no gaps in the sequence (Q1-Q28)

2. **Graph Complexity**: The questions span from simple property retrieval to complex two-hop graph traversals, testing different levels of graph query complexity

3. **Clinical Relevance**: 
   - Single-hop questions focus on core pharmacological data (dosing, pharmacokinetics, safety)
   - Multi-hop questions address practical clinical needs (drug interactions, brands, food interactions)
   - Categorization questions support clinical decision-making (finding alternative drugs, understanding mechanisms)

4. **Relationship Coverage**: The dataset exercises **5 of 6** relationship types in the schema:
   - ✅ ACTS_ON (biological targets) - Q21, Q23, Q26, Q28
   - ✅ BELONGS_TO (drug categories) - Q19, Q20, Q22, Q24, Q25, Q27
   - ✅ MARKETED_AS (commercial products) - Q12, Q14, Q18
   - ✅ HAS_DIETARY_RULE (food interactions) - Q13, Q17
   - ✅ INTERACTS_WITH (drug-drug interactions) - Q10, Q11, Q15, Q16
   - ❌ HAS_ATC_CODE (not tested)

5. **Safety Focus**: 4 questions (Q10, Q11, Q15, Q16) specifically test drug interaction queries, which is critical for medication safety and clinical decision support
