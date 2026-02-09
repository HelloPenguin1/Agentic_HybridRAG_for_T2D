import asyncio
import os
import time
import re
import json
from typing import Optional, Type
from google import genai
from pydantic import BaseModel
from extraction_prompts import get_extraction_prompt


class LLMExtractor:
    """
    Refactored LLM-based entity extractor using Gemini 2.5 Flash-Lite.
    Optimized for JSON reliability and large document handling.
    """
    final_prompt = """
    You are a Senior Clinical Data Scientist specializing in Type 2 Diabetes (T2D) guidelines. 
Your goal is to extract structured knowledge (entities and respective relationships) with zero hallucination, with priority of EXPLAINABILITY and ACCURACY.

### OPERATIONAL RULES:
1. **Evidence-First Extraction**: For every entity AND relationship, you MUST extract the `source_text`. If you cannot find a direct quote supporting the relationship, DO NOT extract it.
2. **Handle Ambiguity**: If a field is a Literal (Enum) and the text is not explicit, use "Unknown". Do not guess.
3. **Numeric Precision**: Capture exact values (e.g., "A1C <7.0%") including their units and comparators.
4. **Contextual Fidelity**: If a recommendation is conditional (e.g., "for patients with ASCVD"), capture that condition in the `conditional_context` field.
5. **No Hallucinations**: Do not use outside medical knowledge. Only extract what is present in the provided text.
6. **No Nulls**: If you cannot find a snippet, use an empty string ""—NEVER use null or None.
7. **Markdown Table Parsing**: The input text is in Markdown format. Pay special attention to Markdown tables (denoted by | separators) for medication dosages, screening thresholds, and diagnostic criteria, as these contain the most critical clinical data.
8. **OUTPUT CONSTRAINTS - CRITICAL**: 
    - Extract a MAXIMUM of 30 entities per entity type (e.g., max 30 medications, max 30 complications)
    - Extract a MAXIMUM of 30 relationships, refer to the specific extraction prompt according to the PDF category.
    - PRIORITIZE the most clinically significant entities (evidence level A/B, specific numeric targets, FDA-approved medications)
    - SKIP generic/redundant entities (e.g., don't extract "Type 2 Diabetes" 10 times, extract it once)
    - Focus on ACTIONABLE clinical knowledge (specific dosages, screening frequencies, contraindications)

### EXTRACTION WORKFLOW:
Step 1: Scan the text for clinical entities matching the requested schema.
Step 2: RANK entities by clinical significance (evidence level, specificity, actionability)
Step 3: Select TOP entities up to the 30.
Step 4: **RELATIONSHIP EXTRACTION (EVIDENCE-BASED ONLY)**:
    a) For each pair of extracted entities, scan the source text for EXPLICIT connections
    b) Valid relationship evidence includes:
       - Direct statements: "Metformin TREATS Type 2 Diabetes"
       - Causal links: "SGLT2 inhibitors REDUCE cardiovascular risk"
       - Clinical associations: "A1C >9% INDICATES poor glycemic control"
       - Contraindications: "Metformin is CONTRAINDICATED in severe renal impairment"
       - Dosage specifications: "Metformin DOSAGE 500-2000mg daily"
       - Monitoring requirements: "Statins REQUIRE lipid panel monitoring"
    c) REJECT relationships based on:
       - Inference or general medical knowledge not in the text
       - Entities mentioned in different sections without explicit connection
       - Assumed clinical logic not stated in the document
    d) For each relationship, identify the EXACT sentence/phrase that establishes the connection
    e) Classify the relationship type based ONLY on what the text explicitly states
Step 5: **RELATIONSHIP VALIDATION**:
    a) Verify both source_entity and target_entity names EXACTLY match extracted entity names
    b) Confirm the relationship type is supported by the source_text evidence
    c) Check that relationship direction is correct (e.g., MEDICATION → TREATS → CONDITION, not reversed)
    d) Ensure conditional context is captured if the relationship only applies in specific scenarios
Step 6: **RELATIONSHIP RANKING & SELECTION**:
    a) Rank relationships by clinical significance:
       - Tier 1: Treatment efficacy, contraindications, dosing (highest priority)
       - Tier 2: Monitoring requirements, drug interactions, adverse events
       - Tier 3: Risk associations, screening recommendations
       - Tier 4: General associations, supportive context
    b) Select TOP 30 relationships ensuring diversity across relationship types
    c) Prioritize relationships that form clinically actionable knowledge paths
    d) Avoid redundant relationships (e.g., don't extract 5 similar TREATS relationships for the same drug)
Step 7: Identify the exact 'source_text' snippet for each selected entity.
Step 8: Map the entity to the most appropriate category. If it doesn't fit a specific category, use "Other" or "Unknown" as defined in the schema.
Step 9: Formulate the JSON response ensuring all Pydantic types (floats, literals, lists) are strictly followed.

### RELATIONSHIP-SPECIFIC RULES:
1. **Source Text Requirement**: Every relationship MUST have a `source_text` field containing the exact quote that establishes the connection. If no quote exists, DO NOT create the relationship.
2. **Entity Name Matching**: Use the EXACT entity names as extracted. Do not paraphrase or abbreviate.
3. **Relationship Type Fidelity**: Only use relationship types explicitly supported by the text. Do not infer types based on medical knowledge.
4. **Bidirectional Caution**: Some relationships are bidirectional (e.g., INTERACTS_WITH), others are not (e.g., TREATS). Respect the directionality implied by the source text.
5. **Conditional Relationships**: If a relationship only applies under certain conditions (e.g., "for patients with CKD"), capture this in the relationship's conditional_context field.
6. **Evidence Strength**: Prefer relationships with explicit evidence levels (A, B, C) or strong clinical language ("recommended", "indicated", "contraindicated").

### SCHEMA ADHERENCE:
- If a numeric 'value' is expected but only a range is given, use the 'range' comparator and specify 'upper_bound'.
- For relationships, ensure the 'source_entity' and 'target_entity' names match the names you used in the entity lists exactly.
- Include relationship metadata: evidence_level, strength (e.g., "strong", "moderate", "weak"), and source_text.

### KNOWLEDGE GRAPH OPTIMIZATION:
The extracted entities and relationships will populate a Neo4j knowledge graph for trustable medical RAG. Therefore:
- **Connectivity**: Prioritize relationships that create meaningful clinical pathways (e.g., Diagnosis → Screening → Medication → Monitoring → Outcome)
- **Queryability**: Favor relationships that answer common clinical questions (What treats X? What are contraindications for Y? How to monitor Z?)
- **Traceability**: Every node and edge must trace back to source evidence for explainability
- **Clinical Utility**: Focus on actionable knowledge (treatment protocols, screening criteria, risk stratification) over descriptive facts

### RELATIONSHIP EXTRACTION EXAMPLES:

**GOOD EXAMPLES (Extract these)**:
✓ Text: "Metformin is the first-line therapy for Type 2 Diabetes"
  → Medication("Metformin") -[TREATS]-> Condition("Type 2 Diabetes")
  → source_text: "Metformin is the first-line therapy for Type 2 Diabetes"

✓ Text: "SGLT2 inhibitors reduce cardiovascular events in patients with established ASCVD"
  → Medication("SGLT2 inhibitors") -[REDUCES_RISK_OF]-> Complication("cardiovascular events")
  → conditional_context: "in patients with established ASCVD"
  → source_text: "SGLT2 inhibitors reduce cardiovascular events in patients with established ASCVD"

✓ Text: "Metformin is contraindicated in patients with eGFR <30 mL/min"
  → Medication("Metformin") -[CONTRAINDICATED_IN]-> Condition("eGFR <30 mL/min")
  → source_text: "Metformin is contraindicated in patients with eGFR <30 mL/min"

**BAD EXAMPLES (Do NOT extract these)**:
✗ Text mentions "Metformin" in section 2 and "kidney disease" in section 5 without connecting them
  → DO NOT create: Medication("Metformin") -[AFFECTS]-> Complication("kidney disease")
  → Reason: No explicit connection in text

✗ Text: "Diabetes increases cardiovascular risk"
  → DO NOT create: Condition("Diabetes") -[TREATED_BY]-> Medication("Statins")
  → Reason: Statins not mentioned; this is inferred medical knowledge

✗ Using general medical knowledge to create relationships not stated in the document
  → DO NOT create relationships based on "common clinical practice" unless explicitly stated

### FINAL CHECKLIST BEFORE OUTPUT:
- [ ] Every relationship has a non-empty source_text field with exact quote
- [ ] Entity names in relationships match exactly with extracted entities
- [ ] No relationships inferred from medical knowledge outside the document
- [ ] Relationship types accurately reflect the connection stated in the text
- [ ] Maximum 30 relationships selected, ranked by clinical significance
- [ ] Relationships form clinically meaningful knowledge pathways
- [ ] All conditional contexts captured where applicable
- [ ] No hallucinated connections between entities mentioned in isolation
    """


    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key: raise ValueError("GOOGLE_API_KEY not set")
        self.client = genai.Client(api_key=self.api_key)
        self.model, self.min_interval, self.last_call = "gemini-2.5-flash-lite", 7.0, 0

    def _surgical_json_repair(self, text: str) -> str:
        """Fixes common JSON errors: truncation, stray escapes, and trailing commas."""
        match = re.search(r'(\{.*)', text, re.DOTALL)
        if not match: return text
        text = match.group(1).strip()
        
        # Strip illegal escapes and ensure quote balance
        text = re.sub(r'\\(?![\\"/bfnrtu])', r'', text)
        if text.count('"') % 2 != 0: text += '"'
        
        # Clean trailing commas/whitespace
        text = re.sub(r',\s*$', '', text.rstrip())
        
        # Balance structural braces using a stack
        stack = []
        for char in text:
            if char in '{[': stack.append('}' if char == '{' else ']')
            elif char in '}]' and stack and stack[-1] == char: stack.pop()
        
        repaired = text + ''.join(reversed(stack))
        return re.sub(r',\s*([\]\}])', r'\1', repaired) # Remove internal trailing commas

    def _merge_extractions(self, res1: BaseModel, res2: BaseModel) -> BaseModel:
        """Unified merging and deduplication of Pydantic model results."""
        d1, d2 = res1.model_dump(), res2.model_dump()
        for k in d1:
            if isinstance(d1[k], list):
                combined, seen = d1[k] + d2.get(k, []), set()
                unique = []
                for item in combined:
                    # Deduplicate based on name or serialized content
                    key = item.get('name', str(item)) if isinstance(item, dict) else str(item)
                    if key not in seen:
                        seen.add(key)
                        unique.append(item)
                d1[k] = unique
        return res1.__class__(**d1)

    async def extract_entire_pdf(self, text: str, cat: str, model: Type[BaseModel], retries: int = 3) -> Optional[BaseModel]:
        # Handle large documents by splitting into THREE chunks (not two)
        # This prevents hitting the ~155K character output limit
        if len(text) > 150000:
            chunk_size = len(text) // 3
            print(f"📄 Large doc ({len(text)} chars). Processing in THREE chunks of ~{chunk_size:,} chars each...")
            
            # Extract from all 3 chunks
            r1 = await self._extract_single_chunk(text[:chunk_size], cat, model, retries)
            if not r1: return None
            
            r2 = await self._extract_single_chunk(text[chunk_size:chunk_size*2], cat, model, retries)
            if not r2: return r1  # Return partial if chunk 2 fails
            
            r3 = await self._extract_single_chunk(text[chunk_size*2:], cat, model, retries)
            if not r3: return self._merge_extractions(r1, r2)  # Return 2 chunks if chunk 3 fails
            
            # Merge all 3 results
            merged_12 = self._merge_extractions(r1, r2)
            return self._merge_extractions(merged_12, r3)
        
        return await self._extract_single_chunk(text, cat, model, retries)

    async def _extract_single_chunk(self, text: str, cat: str, model: Type[BaseModel], retries: int = 3) -> Optional[BaseModel]:
        prompt = get_extraction_prompt(cat, text)
        for attempt in range(retries):
            try:
                # Throttling
                elapsed = time.time() - self.last_call
                if elapsed < self.min_interval: await asyncio.sleep(self.min_interval - elapsed)
                
                self.last_call = time.time()
                print(f"Calling Gemini ({cat}) - attempt {attempt + 1}/{retries}")
                
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=f"{self.final_prompt}\n\n{prompt}",
                    config={
                        'response_mime_type': 'application/json',
                        'response_schema': model.model_json_schema(),
                        'temperature': 0,
                        'max_output_tokens': 49152  # Increased for complex schemas
                    }
                )
                
                # Debug: Log response size
                response_text = response.text
                print(f"  📊 Response size: {len(response_text):,} chars")
                
                # Try to repair and validate
                repaired = self._surgical_json_repair(response_text)
                result = model.model_validate_json(repaired)
                
                # SOLUTION 4: Trim verbose source_text fields to max 200 chars
                result_dict = result.model_dump()
                for key, entities in result_dict.items():
                    if isinstance(entities, list):
                        for entity in entities:
                            if isinstance(entity, dict) and 'source_text' in entity:
                                if entity['source_text'] and len(entity['source_text']) > 200:
                                    entity['source_text'] = entity['source_text'][:197] + "..."
                
                # Reconstruct the model with trimmed data
                return model.model_validate(result_dict)
                
            except Exception as e:
                error_msg = str(e)
                
                # Save failed response for debugging
                if "Invalid JSON" in error_msg or "EOF" in error_msg:
                    debug_file = f"debug_failed_response_{cat}_attempt{attempt+1}.txt"
                    try:
                        with open(debug_file, 'w', encoding='utf-8') as f:
                            f.write(f"=== ERROR ===\n{error_msg}\n\n")
                            f.write(f"=== RAW RESPONSE ({len(response_text)} chars) ===\n")
                            f.write(response_text[:5000])  # First 5K chars
                            f.write("\n\n=== LAST 2000 CHARS ===\n")
                            f.write(response_text[-2000:])  # Last 2K chars
                        print(f"  💾 Saved debug info to: {debug_file}")
                    except:
                        pass
                
                if "429" in error_msg or "quota" in error_msg.lower():
                    wait = 60 * (attempt + 1)
                    print(f"⏳ Quota hit. Sleeping {wait}s...")
                    await asyncio.sleep(wait)
                elif attempt < retries - 1:
                    print(f"⚠️ Error: {error_msg[:120]}... Retrying.")
                    await asyncio.sleep(10)
                else:
                    print(f"❌ Max retries reached for {cat}.")
                    print(f"   Last error: {error_msg[:200]}")
                    return None
        return None