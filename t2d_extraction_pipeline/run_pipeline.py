import asyncio
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Internal imports
from .extraction_pipeline import ExtractionPipeline
from .entity_extractor import LLMExtractor
from .post_processor import PostProcessor
from .pydantic_models import (
    AssessmentDiagnosisExtraction, EducationLifestyleExtraction,
    PharmacologyExtraction, ComplicationsExtraction, SpecialPopulationsExtraction
)

load_dotenv()

CAT_MODELS = {
    "assessment_diagnosis": AssessmentDiagnosisExtraction,
    "patient_education_lifestyle": EducationLifestyleExtraction,
    "pharmacology_technology": PharmacologyExtraction,
    "complications_management": ComplicationsExtraction,
    "special_populations": SpecialPopulationsExtraction
}

async def run_pipeline():
    # Path Setup - All contained within this subfolder
    PKG_DIR = Path(__file__).parent
    PDF_DIR = PKG_DIR / "data" / "raw_pdfs"
    OUT_DIR = PKG_DIR / "outputs"
    CONFIG_DIR = PKG_DIR / "config"
    
    # 1. Text Extraction
    pipe = ExtractionPipeline(str(CONFIG_DIR), str(PDF_DIR), str(OUT_DIR))
    pipe.process_all_pdfs(str(PDF_DIR))

    # 2. Async Map Phase
    extractor = LLMExtractor()
    tasks = []
    
    meta_files = list((OUT_DIR / "metadata").glob("*.json"))
    for meta_file in meta_files:
        with open(meta_file, 'r') as f: meta = json.load(f)
        chunk_file = OUT_DIR / "extracted_text" / meta_file.name.replace("_metadata", "_extracted")
        with open(chunk_file, 'r') as f: chunks = json.load(f)
        
        print(f"🚀 Mapping: {meta['filename']}")
        tasks.append(extractor.extract_pdf_chunks(chunks, meta['category'], CAT_MODELS[meta['category']]))

    all_results = await asyncio.gather(*tasks)

    # Save Mapping Results
    ent_dir = OUT_DIR / "extracted_entities"
    ent_dir.mkdir(exist_ok=True)
    for meta_file, result in zip(meta_files, all_results):
        save_path = ent_dir / meta_file.name.replace("_metadata", "_entities")
        with open(save_path, 'w') as f:
            json.dump([r.model_dump() for r in result if r], f, indent=2)

    # 3. Global Reduction
    post = PostProcessor(ent_dir, OUT_DIR)
    post.process_and_reduce()

if __name__ == "__main__":
    asyncio.run(run_pipeline())