import asyncio
import json
import os
from pathlib import Path
from dotenv import load_dotenv

from extraction_pipeline import ExtractionPipeline
from entity_extractor import LLMExtractor
from category_linker import CategoryLinker
from pydantic_models import (
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
    """Resilient pipeline with local progress saving."""
    PKG_DIR = Path(__file__).parent
    PDF_DIR = PKG_DIR / "data" / "raw_pdfs"
    OUT_DIR = PKG_DIR / "outputs"
    ENT_DIR = OUT_DIR / "extracted_entities"
    ENT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Text Extraction
    pipe = ExtractionPipeline(str(PKG_DIR / "config"), str(PDF_DIR), str(OUT_DIR))
    pipe.process_all_pdfs(str(PDF_DIR))

    # 2. Entity Extraction
    extractor = LLMExtractor()
    meta_files = list((OUT_DIR / "metadata").glob("*.json"))
    
    for meta_file in meta_files:
        save_path = ENT_DIR / meta_file.name.replace("_metadata", "_entities")
        if save_path.exists():
            continue
            
        with open(meta_file, 'r') as f: meta = json.load(f)
        chunk_file = OUT_DIR / "extracted_text" / meta_file.name.replace("_metadata", "_extracted")
        
        with open(chunk_file, 'r') as f: chunks = json.load(f)
        full_text = " ".join([c['text'] for c in chunks])
        
        print(f"🚀 Processing: {meta['filename']}")
        result = await extractor.extract_entire_pdf(full_text, meta['category'], CAT_MODELS[meta['category']])
        
        if result:
            with open(save_path, 'w') as f:
                json.dump(result.model_dump(), f, indent=2)

    # 3. Category-Based Entity Linking and Neo4j Export
    if len(list(ENT_DIR.glob("*_entities.json"))) == len(meta_files):
        print("\n" + "=" * 60)
        print("STEP 3: Category-based entity linking and Neo4j CSV export")
        print("=" * 60)
        
        config_dir = PKG_DIR / "config"
        linker = CategoryLinker(ENT_DIR, OUT_DIR, config_dir)
        linker.process_and_export()
        
        print("\n✅ Pipeline complete!")
        print(f"Neo4j CSV files generated in: {OUT_DIR / 'neo4j_category_imports'}")
    else:
        print("\nNot all files processed. Skipping post-processing.")
        print("Run the pipeline again to resume.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())