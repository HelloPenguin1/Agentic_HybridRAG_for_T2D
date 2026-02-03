import asyncio
import json
import os
import time
from typing import Dict, Any, Optional, Type
from pathlib import Path
from google import genai
from pydantic import BaseModel
from extraction_prompts import get_extraction_prompt

class LLMExtractor:
    """
    LLM-based entity extractor using Gemini 2.5 Flash-Lite.
    
    Key Features:
    - Full-document processing (1M token context window)
    - 1,000 requests per day (free tier)
    - 15 requests per minute rate limit
    - Structured JSON output
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

    ### EXTRACTION WORKFLOW:
    Step 1: Scan the text for clinical entities matching the requested schema.
    Step 2: Identify the exact 'source_text' snippet for each entity.
    Step 3: Map the entity to the most appropriate category. If it doesn't fit a specific category, use "Other" or "Unknown" as defined in the schema.
    Step 4: Formulate the JSON response ensuring all Pydantic types (floats, literals, lists) are strictly followed.

    ### SCHEMA ADHERENCE:
    - If a numeric 'value' is expected but only a range is given, use the 'range' comparator and specify 'upper_bound'.
    - For relationships, ensure the 'source_entity' and 'target_entity' names match the names you used in the entity lists exactly.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-2.5-flash-lite"  # 1M token context, 1000 RPD free tier
        
        # Rate limiting: 15 RPM (requests per minute)
        self.min_request_interval = 4.0  # 60s / 15 = 4s between requests
        self.last_request_time = 0

    async def _rate_limit_wait(self):
        """Ensure we don't exceed 15 requests per minute."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            print(f"  ⏳ Rate limiting: waiting {wait_time:.1f}s...")
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()

    async def extract_entire_pdf(
        self, 
        full_text: str, 
        category: str, 
        model_class: Type[BaseModel],
        retry_count: int = 3
    ) -> Optional[BaseModel]:
        """
        Extract entities from an entire PDF in one API call.
        
        Args:
            full_text: Complete text of the PDF
            category: Category for extraction prompt
            model_class: Pydantic model for validation
            retry_count: Number of retries on failure
            
        Returns:
            Validated Pydantic model or None on failure
        """
        prompt = get_extraction_prompt(category, full_text)
        
        for attempt in range(retry_count):
            try:
                # Rate limiting
                await self._rate_limit_wait()
                
                # Construct the full prompt with system instructions
                full_prompt = f"{self.SYSTEM_PROMPT}\n\n{prompt}\n\nSchema: {model_class.model_json_schema()}"
                
                print(f"  🤖 Calling Gemini API (attempt {attempt + 1}/{retry_count})...")
                
                # Call Gemini with structured output
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=full_prompt,
                    config={
                        'response_mime_type': 'application/json',
                        'temperature': 0
                    }
                )
                
                # Validate and return
                result = model_class.model_validate_json(response.text)
                print(f"  ✅ Extraction successful")
                return result
                
            except Exception as e:
                error_str = str(e)
                
                # Handle 429 Rate Limit errors
                if "429" in error_str or "Resource has been exhausted" in error_str:
                    if attempt < retry_count - 1:
                        wait_time = 60 * (attempt + 1)  # Exponential backoff: 60s, 120s, 180s
                        print(f"  ⚠️ Rate limit hit (429). Waiting {wait_time}s before retry...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"  ❌ Rate limit exhausted after {retry_count} attempts.")
                        print(f"  💡 Daily limit may be reached. Resume pipeline later.")
                        raise Exception("RATE_LIMIT_EXHAUSTED") from e
                
                # Handle other errors
                if attempt < retry_count - 1:
                    print(f"  ⚠️ Request failed: {e}. Retrying...")
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                else:
                    print(f"  ❌ Request failed after {retry_count} attempts: {e}")
                    return None
        
        return None