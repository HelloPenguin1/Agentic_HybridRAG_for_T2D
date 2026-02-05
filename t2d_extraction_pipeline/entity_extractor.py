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

    SYSTEM_PROMPT = """You are a Senior Clinical Data Scientist specializing in Type 2 Diabetes (T2D) guidelines. 
    Your goal is to extract structured knowledge with zero hallucination, with priority of EXPLAINABILITY and ACCURACY

    ### OPERATIONAL RULES:
    1. **Evidence-First Extraction**: For every entity, you MUST extract the `source_text`. If you cannot find a direct quote, DO NOT extract the entity.
    2. **Handle Ambiguity**: If a field is a Literal (Enum) and the text is not explicit, use "Unknown". Do not guess.
    3. **Numeric Precision**: Capture exact values (e.g., "A1C <7.0%") including their units and comparators.
    4. **Contextual Fidelity**: If a recommendation is conditional (e.g., "for patients with ASCVD"), capture that condition in the `conditional_context` field.
    5. **No Hallucinations**: Do not use outside medical knowledge. Only extract what is present in the provided text.
    6. **No Nulls**: If you cannot find a snippet, use an empty string ""—NEVER use null or None.
    7. **Markdown Table Parsing**: The input text is in Markdown format. Pay special attention to Markdown tables (denoted by | separators) for medication dosages, screening thresholds, and diagnostic criteria, as these contain the most critical clinical data.
    8. **OUTPUT CONSTRAINTS - CRITICAL**: 
        - Extract a MAXIMUM of 30 entities per entity type (e.g., max 30 medications, max 30 complications)
        - Extract a MAXIMUM of 30 relationships
        - PRIORITIZE the most clinically significant entities (evidence level A/B, specific numeric targets, FDA-approved medications)
        - SKIP generic/redundant entities (e.g., don't extract "Type 2 Diabetes" 10 times, extract it once)
        - Focus on ACTIONABLE clinical knowledge (specific dosages, screening frequencies, contraindications)

    ### EXTRACTION WORKFLOW:
    Step 1: Scan the text for clinical entities matching the requested schema.
    Step 2: RANK entities by clinical significance (evidence level, specificity, actionability)
    Step 3: Select TOP entities up to the maximum limits
    Step 4: Identify the exact 'source_text' snippet for each selected entity.
    Step 5: Map the entity to the most appropriate category. If it doesn't fit a specific category, use "Other" or "Unknown" as defined in the schema.
    Step 6: Formulate the JSON response ensuring all Pydantic types (floats, literals, lists) are strictly followed.

    ### SCHEMA ADHERENCE:
    - If a numeric 'value' is expected but only a range is given, use the 'range' comparator and specify 'upper_bound'.
    - For relationships, ensure the 'source_entity' and 'target_entity' names match the names you used in the entity lists exactly.
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
        # Handle large documents by splitting into two halves
        if len(text) > 150000:
            print(f"📄 Large doc ({len(text)} chars). Processing in two chunks...")
            mid = len(text) // 2
            r1 = await self._extract_single_chunk(text[:mid], cat, model, retries)
            if not r1: return None
            r2 = await self._extract_single_chunk(text[mid:], cat, model, retries)
            return self._merge_extractions(r1, r2) if r2 else r1
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
                    contents=f"{self.SYSTEM_PROMPT}\n\n{prompt}",
                    config={
                        'response_mime_type': 'application/json',
                        'response_schema': model.model_json_schema(),
                        'temperature': 0,
                        'max_output_tokens': 8192 
                    }
                )
                return model.model_validate_json(self._surgical_json_repair(response.text))
                
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    wait = 60 * (attempt + 1)
                    print(f"⏳ Quota hit. Sleeping {wait}s...")
                    await asyncio.sleep(wait)
                elif attempt < retries - 1:
                    print(f"⚠️ Error: {str(e)[:80]}... Retrying.")
                    await asyncio.sleep(10)
                else:
                    print(f"❌ Max retries reached for {cat}.")
                    return None
        return None