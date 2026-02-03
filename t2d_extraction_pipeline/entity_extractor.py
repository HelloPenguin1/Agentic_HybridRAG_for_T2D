import asyncio
import json
import os
from typing import List, Dict, Any, Optional, Type
from pathlib import Path
from groq import AsyncGroq
from pydantic import BaseModel
from extraction_prompts import get_extraction_prompt

class LLMExtractor:

 
    SYSTEM_PROMPT = """You are a Senior Clinical Data Scientist specializing in Type 2 Diabetes (T2D) guidelines. 
    Your goal is to extract structured knowledge with zero hallucination.

    ### OPERATIONAL RULES:
    1. **Evidence-First Extraction**: For every entity, you MUST extract the `source_text`. If you cannot find a direct quote, DO NOT extract the entity.
    2. **Handle Ambiguity**: If a field is a Literal (Enum) and the text is not explicit, use "Unknown". Do not guess.
    3. **Numeric Precision**: Capture exact values (e.g., "A1C <7.0%") including their units and comparators.
    4. **Contextual Fidelity**: If a recommendation is conditional (e.g., "for patients with ASCVD"), capture that condition in the `conditional_context` field.
    5. **No Hallucinations**: Do not use outside medical knowledge. Only extract what is present in the provided text chunk.

    ### EXTRACTION WORKFLOW:
    Step 1: Scan the text for clinical entities matching the requested schema.
    Step 2: Identify the exact 'source_text' snippet for each entity.
    Step 3: Map the entity to the most appropriate category. If it doesn't fit a specific category, use "Other" or "Unknown" as defined in the schema.
    Step 4: Formulate the JSON response ensuring all Pydantic types (floats, literals, lists) are strictly followed.

    ### SCHEMA ADHERENCE:
    - If a numeric 'value' is expected but only a range is given, use the 'range' comparator and specify 'upper_bound'.
    - For relationships, ensure the 'source_entity' and 'target_entity' names match the names you used in the entity lists exactly.
    """


    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile", tpm_safe_limit: int = 3):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")    
        self.client = AsyncGroq(api_key=self.api_key)
        self.model = model
        # Lower limit to 3 to stay under 12,000 Tokens Per Minute
        self.semaphore = asyncio.Semaphore(tpm_safe_limit)

    async def _call_llm_async(self, prompt: str, model_class: Type[BaseModel], retry_count=3) -> Optional[BaseModel]:
        async with self.semaphore:
            for attempt in range(retry_count):
                try:
                    # Small delay to let Groq's token bucket refill
                    await asyncio.sleep(1.5 * (attempt + 1)) 
                    
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.SYSTEM_PROMPT},

                            {"role": "user", "content": f"{prompt}\n\nSchema: {model_class.model_json_schema()}"}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0
                    )
                    return model_class.model_validate_json(response.choices[0].message.content)
                except Exception as e:
                    if "429" in str(e) and attempt < retry_count - 1:
                        continue # Automatic retry on rate limit
                    print(f"  ⚠️ Request failed: {e}")
                    return None

    async def extract_pdf_chunks(self, chunks: List[Dict[str, Any]], category: str, model_class: Type[BaseModel]):
        tasks = [self._call_llm_async(get_extraction_prompt(category, c['text']), model_class) for c in chunks]
        return await asyncio.gather(*tasks)