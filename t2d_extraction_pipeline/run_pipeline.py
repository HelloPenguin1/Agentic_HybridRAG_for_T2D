import asyncio
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Internal imports
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
    """
    Run the extraction pipeline with full-document batching and resume capability.
    
    Key Features:
    - Processes entire PDFs in one API call (not chunks)
    - Checks for existing entity files to resume from interruptions
    - Saves immediately after each successful extraction
    - Handles 429 rate limit errors gracefully
    """
    # Path Setup - All contained within this subfolder
    PKG_DIR = Path(__file__).parent
    PDF_DIR = PKG_DIR / "data" / "raw_pdfs"
    OUT_DIR = PKG_DIR / "outputs"
    CONFIG_DIR = PKG_DIR / "config"
    
    # 1. Text Extraction
    print("=" * 60)
    print("STEP 1: Extracting text from PDFs")
    print("=" * 60)
    pipe = ExtractionPipeline(str(CONFIG_DIR), str(PDF_DIR), str(OUT_DIR))
    pipe.process_all_pdfs(str(PDF_DIR))

    # 2. Entity Extraction with Resume Logic
    print("\n" + "=" * 60)
    print("STEP 2: Extracting entities with Gemini 2.5 Flash-Lite")
    print("=" * 60)
    
    extractor = LLMExtractor()
    
    # Prepare output directory
    ent_dir = OUT_DIR / "extracted_entities"
    ent_dir.mkdir(exist_ok=True)
    
    meta_files = list((OUT_DIR / "metadata").glob("*.json"))
    total_files = len(meta_files)
    processed_count = 0
    skipped_count = 0
    
    print(f"\nFound {total_files} PDFs to process\n")
    
    for idx, meta_file in enumerate(meta_files, 1):
        # Load metadata
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        
        filename = meta['filename']
        category = meta['category']
        
        # 1. RESUME CHECK: Skip if already processed
        save_path = ent_dir / meta_file.name.replace("_metadata", "_entities")
        if save_path.exists():
            print(f"[{idx}/{total_files}] ⏩ Already processed: {filename}")
            skipped_count += 1
            continue
        
        print(f"\n[{idx}/{total_files}] 🚀 Processing: {filename}")
        print(f"  Category: {category}")
        
        # 2. LOAD FULL TEXT (Consolidate all chunks)
        chunk_file = OUT_DIR / "extracted_text" / meta_file.name.replace("_metadata", "_extracted")
        
        try:
            with open(chunk_file, 'r') as f:
                chunks = json.load(f)
            
            # Combine all chunks into full document text
            full_text = " ".join([c['text'] for c in chunks])
            print(f"  📄 Consolidated {len(chunks)} chunks into full document ({len(full_text)} chars)")
            
            # 3. EXTRACT & SAVE IMMEDIATELY
            try:
                result = await extractor.extract_entire_pdf(
                    full_text, 
                    category, 
                    CAT_MODELS[category]
                )
                
                if result:
                    # Save immediately to enable resume
                    with open(save_path, 'w') as f:
                        json.dump(result.model_dump(), f, indent=2)
                    print(f"  💾 Saved entities to: {save_path.name}")
                    processed_count += 1
                else:
                    print(f"  ⚠️ Extraction returned None for {filename}")
                    
            except Exception as e:
                if "RATE_LIMIT_EXHAUSTED" in str(e):
                    print("\n" + "=" * 60)
                    print("⛔ RATE LIMIT REACHED")
                    print("=" * 60)
                    print(f"Processed {processed_count}/{total_files} documents before hitting limit.")
                    print("The pipeline can be resumed by running this script again.")
                    print("Already-processed files will be skipped automatically.")
                    print("\n💡 Tip: Daily limits reset at midnight Pacific Time.")
                    return  # Graceful exit
                else:
                    print(f"  ❌ Error processing {filename}: {e}")
                    continue
                    
        except FileNotFoundError:
            print(f"  ⚠️ Chunk file not found: {chunk_file}")
            continue

    # Summary
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total PDFs: {total_files}")
    print(f"Already processed (skipped): {skipped_count}")
    print(f"Newly processed: {processed_count}")
    print(f"Remaining: {total_files - skipped_count - processed_count}")

    # 3. Global Reduction (only if all files processed)
    if skipped_count + processed_count == total_files:
        print("\n" + "=" * 60)
        print("STEP 3: Post-processing and reduction")
        print("=" * 60)
        post = PostProcessor(ent_dir, OUT_DIR)
        post.process_and_reduce()
        print("✅ Pipeline complete!")
    else:
        print("\n⚠️ Not all files processed. Skipping post-processing.")
        print("Run the pipeline again to resume.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
