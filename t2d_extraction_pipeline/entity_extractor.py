import asyncio
import json
import os
from typing import List, Dict, Any, Optional, Type
from pathlib import Path
from groq import AsyncGroq
from pydantic import BaseModel
# Relative import from the same subfolder
from .extraction_prompts import get_extraction_prompt

class LLMExtractor:
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile", rpm_limit: int = 28):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")    
        self.client = AsyncGroq(api_key=self.api_key)
        self.model = model
        # Limits concurrent requests to stay safe under 30 RPM
        self.semaphore = asyncio.Semaphore(rpm_limit)

    async def _call_llm_async(self, prompt: str, model_class: Type[BaseModel]) -> Optional[BaseModel]:
        async with self.semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Expert clinical data extractor. Respond in valid JSON."},
                        {"role": "user", "content": f"{prompt}\n\nSchema: {model_class.model_json_schema()}"}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0
                )
                return model_class.model_validate_json(response.choices[0].message.content)
            except Exception as e:
                print(f"  ⚠️ Request failed: {e}")
                return None

    async def extract_pdf_chunks(self, chunks: List[Dict[str, Any]], category: str, model_class: Type[BaseModel]):
        """Parallel chunk processing"""
        tasks = [self._call_llm_async(get_extraction_prompt(category, c['text']), model_class) for c in chunks]
        return await asyncio.gather(*tasks)