import asyncio
import json
import os
from pathlib import Path
from dotenv import load_dotenv

from extraction_pipeline import ExtractionPipeline
from entity_extractor import LLMExtractor
from post_processor import PostProcessor
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

    # 3. Global Reduction (only if all files processed)
    # TEMPORARILY DISABLED: Review post_processor.py before running
    print("\n" + "=" * 60)
    print("⏸️  POST-PROCESSING SKIPPED")
    print("=" * 60)
    print(f"All {len(meta_files)} PDFs have been extracted successfully!")
    print(f"\nTo run post-processing (map-reduce operation):")
    print(f"  1. Review the logic in 'post_processor.py'")
    print(f"  2. Uncomment lines 72-81 in 'run_pipeline.py'")
    print(f"  3. Re-run: python run_pipeline.py")
    
    # if len(list(ENT_DIR.glob("*_entities.json"))) == len(meta_files):
    #     print("\n" + "=" * 60)
    #     print("STEP 3: Post-processing and reduction")
    #     print("=" * 60)
    #     post = PostProcessor(ENT_DIR, OUT_DIR)
    #     post.process_and_reduce()
    #     print("✅ Pipeline complete!")
    # else:
    #     print("\n⚠️ Not all files processed. Skipping post-processing.")
    #     print("Run the pipeline again to resume.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())