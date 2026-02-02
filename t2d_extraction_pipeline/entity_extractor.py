"""
LLM-Based Entity and Relationship Extractor
Uses Groq API with structured outputs and Pydantic models for schema enforcement
"""

import json
import os
from typing import List, Dict, Any, Optional, Type
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# Import our Pydantic models
from pydantic_models import (
    AssessmentDiagnosisExtraction,
    EducationLifestyleExtraction,
    PharmacologyExtraction,
    ComplicationsExtraction,
    SpecialPopulationsExtraction
)

# Import extraction prompts
from extraction_prompts import get_extraction_prompt


class LLMExtractor:
    """Handles LLM-based extraction with structured outputs"""
    
    # Map category names to their Pydantic models
    CATEGORY_MODELS = {
        "assessment_diagnosis": AssessmentDiagnosisExtraction,
        "patient_education_lifestyle": EducationLifestyleExtraction,
        "pharmacology_technology": PharmacologyExtraction,
        "complications_management": ComplicationsExtraction,
        "special_populations": SpecialPopulationsExtraction
    }
    
    def __init__(self, api_key: Optional[str] = None, model: str = "openai/gpt-oss-120b"):
        """
        Initialize LLM extractor.
        
        Args:
            api_key: Groq API key (or uses GROQ_API_KEY env var)
            model: Groq model to use (default: gpt-oss-20b with structured outputs)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")    
        self.client = Groq(api_key=self.api_key)
        self.model = model
    
    def _call_llm_with_retry(self, prompt: str, model_class: Type[BaseModel], max_retries: int = 3) -> Optional[BaseModel]:
        """
        Call LLM with automatic retry logic.
        
        Args:
            prompt: The extraction prompt
            model_class: Pydantic model class for structured output
            max_retries: Number of retry attempts
        
        Returns:
            Parsed Pydantic model or None on failure
        """
        for attempt in range(max_retries):
            try:
                # Groq uses standard chat completions with JSON mode
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert clinical data extraction assistant specializing in Type 1/2 diabetes care guidelines. Extract structured information precisely as requested. Always respond with valid JSON matching the requested schema."
                        },
                        {
                            "role": "user",
                            "content": f"{prompt}\n\nRespond with valid JSON matching this Pydantic schema: {model_class.model_json_schema()}"
                        }
                    ],
                    response_format={"type": "json_object"},
                    temperature=0
                )
                
                # Parse JSON response into Pydantic model
                json_response = json.loads(response.choices[0].message.content)
                return model_class.model_validate(json_response)
            
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    print(f"All {max_retries} attempts failed")
                    return None
        
        return None
    
    def extract_from_text(
        self, 
        text: str, 
        category: str,
        max_retries: int = 3
    ) -> Optional[BaseModel]:
        """
        Extract entities and relationships from text using category-specific logic.
        
        Args:
            text: The text chunk to analyze
            category: Category name (determines which prompt and model to use)
            max_retries: Number of retry attempts on failure
        
        Returns:
            Pydantic model instance with extracted data, or None on failure
        """
        # Get the appropriate Pydantic model for this category
        model_class = self.CATEGORY_MODELS.get(category)
        
        if not model_class:
            raise ValueError(f"Unknown category: {category}. Must be one of: {list(self.CATEGORY_MODELS.keys())}")
        
        # Get the extraction prompt for this category
        prompt = get_extraction_prompt(category, text)
        
        # Call LLM with retry logic
        return self._call_llm_with_retry(prompt, model_class, max_retries)
    
    def extract_batch(
        self,
        text_chunks: List[Dict[str, Any]],
        category: str,
        intermediate_dir: str
    ) -> List[Dict[str, Any]]:
        """
        Extract from multiple text chunks.
        
        Intermediate results are always saved for debugging (helpful for students).
        **Automatically resumes from where it left off** by skipping already-processed chunks.
        
        Args:
            text_chunks: List of chunk dicts with 'text', 'chunk_id', etc.
            category: Category name
            intermediate_dir: Directory to save intermediate results
        
        Returns:
            List of extraction results with metadata
        """
        results = []
        intermediate_path = Path(intermediate_dir)
        intermediate_path.mkdir(parents=True, exist_ok=True)
        
        # Count already-processed chunks
        existing_files = list(intermediate_path.glob("*_extraction.json"))
        already_processed = len(existing_files)
        
        if already_processed > 0:
            print(f"  ✅ Found {already_processed} already-processed chunks, resuming...")
        
        for idx, chunk in enumerate(text_chunks, 1):
            chunk_file = intermediate_path / f"{chunk['chunk_id']}_extraction.json"
            
            # Skip if already processed
            if chunk_file.exists():
                print(f"  ⏭️  Skipping chunk {idx}/{len(text_chunks)}: {chunk['chunk_id']} (already done)")
                # Load existing result
                with open(chunk_file, 'r') as f:
                    result = json.load(f)
                results.append(result)
                continue
            
            print(f"Processing chunk {idx}/{len(text_chunks)}: {chunk['chunk_id']}")
            
            extracted = self.extract_from_text(
                text=chunk['text'],
                category=category
            )
            
            if extracted:
                result = {
                    'chunk_id': chunk['chunk_id'],
                    'page': chunk['page'],
                    'source': chunk['source'],
                    'category': category,
                    'extraction': json.loads(extracted.model_dump_json()),
                    'char_count': chunk['char_count']
                }
                
                results.append(result)
                
                # Always save intermediate results for debugging
                with open(chunk_file, 'w') as f:
                    json.dump(result, f, indent=2)
            else:
                print(f"  ❌ Extraction failed for chunk {chunk['chunk_id']}")
        
        return results


class ExtractionOrchestrator:
    """Orchestrates extraction across all PDFs with category-specific logic"""
    
    def __init__(
        self, 
        extracted_text_dir: str,
        metadata_dir: str,
        output_dir: str,
        api_key: Optional[str] = None
    ):
        """
        Initialize orchestrator.
        
        Args:
            extracted_text_dir: Directory with extracted PDF text
            metadata_dir: Directory with PDF metadata
            output_dir: Directory for extraction outputs
            api_key: OpenAI API key (optional, defaults to OPENAI_API_KEY env var)
        """
        self.extracted_text_dir = Path(extracted_text_dir)
        self.metadata_dir = Path(metadata_dir)
        self.output_dir = Path(output_dir)
        
        # Use provided API key or load from environment
        api_key = api_key or os.getenv('GROQ_API_KEY')
        self.extractor = LLMExtractor(api_key=api_key)
        
        # Create output directories
        self.entities_dir = self.output_dir / "extracted_entities"
        self.entities_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_json_file(self, file_path: Path, file_description: str) -> Optional[Dict[str, Any]]:
        """
        Helper method to load JSON files with error handling.
        
        Args:
            file_path: Path to JSON file
            file_description: Description for error messages
        
        Returns:
            Loaded JSON data or None if file not found
        """
        if not file_path.exists():
            print(f"❌ {file_description} not found: {file_path}")
            return None
        
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def process_pdf(self, pdf_stem: str) -> Optional[Dict[str, Any]]:
        """
        Process a single PDF's extracted text.
        
        Args:
            pdf_stem: PDF filename without extension
        
        Returns:
            Extraction results or None
        """
        # Load extracted text
        text_file = self.extracted_text_dir / f"{pdf_stem}_extracted.json"
        text_chunks = self._load_json_file(text_file, "Extracted text")
        if not text_chunks:
            return None
        
        # Load metadata to get category
        metadata_file = self.metadata_dir / f"{pdf_stem}_metadata.json"
        metadata = self._load_json_file(metadata_file, "Metadata")
        if not metadata:
            return None
        
        category = metadata['category']
        print(f"\n📄 Processing: {pdf_stem}")
        print(f"   Category: {category}, Chunks: {len(text_chunks)}")
        
        # Create intermediate directory for this PDF
        intermediate_dir = self.entities_dir / pdf_stem
        intermediate_dir.mkdir(exist_ok=True)
        
        # Extract entities from all chunks
        extractions = self.extractor.extract_batch(
            text_chunks=text_chunks,
            category=category,
            intermediate_dir=str(intermediate_dir)
        )
        
        # Save consolidated extraction
        output_file = self.entities_dir / f"{pdf_stem}_entities.json"
        with open(output_file, 'w') as f:
            json.dump({
                'pdf': pdf_stem,
                'category': category,
                'metadata': metadata,
                'total_chunks': len(text_chunks),
                'successful_extractions': len(extractions),
                'extractions': extractions
            }, f, indent=2)
        
        print(f"Saved consolidated extraction to: {output_file}")
        
        return {
            'pdf': pdf_stem,
            'category': category,
            'chunks_processed': len(text_chunks),
            'successful_extractions': len(extractions)
        }
    
    def process_all(self) -> Dict[str, Any]:
        """
        Process all PDFs that have extracted text.
        
        Returns:
            Summary of all processing
        """
        # Find all extracted text files
        text_files = list(self.extracted_text_dir.glob("*_extracted.json"))
        
        if not text_files:
            print("No extracted text files found")
            return {}
        
        results = []
        
        for text_file in text_files:
            pdf_stem = text_file.stem.replace('_extracted', '')
            result = self.process_pdf(pdf_stem)
            
            if result:
                results.append(result)
        
        # Save overall summary
        summary = {
            'total_pdfs': len(text_files),
            'successfully_processed': len(results),
            'results': results
        }
        
        summary_file = self.output_dir / "entity_extraction_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Entity extraction complete. Processed: {len(results)}/{len(text_files)} PDFs")
        print(f"Summary: {summary_file}")
        
        return summary


def main():
    """Main entry point for entity extraction"""
    
    BASE_DIR = Path(__file__).parent.parent
    EXTRACTED_TEXT_DIR = BASE_DIR / "outputs" / "extracted_text"
    METADATA_DIR = BASE_DIR / "outputs" / "metadata"
    OUTPUT_DIR = BASE_DIR / "outputs"
    
    # Initialize orchestrator
    orchestrator = ExtractionOrchestrator(
        extracted_text_dir=str(EXTRACTED_TEXT_DIR),
        metadata_dir=str(METADATA_DIR),
        output_dir=str(OUTPUT_DIR)
    )
    
    # Process all PDFs
    orchestrator.process_all()


if __name__ == "__main__":
    main()