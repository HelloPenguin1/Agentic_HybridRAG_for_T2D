#!/usr/bin/env python3
"""
Main Orchestration Script for T2D Data Extraction Pipeline
Runs all steps: PDF text extraction → Entity extraction → Post-processing
"""

import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from extraction_pipeline import ExtractionPipeline
from entity_extractor import ExtractionOrchestrator
from post_processor import PostProcessor


def run_full_pipeline(pdf_directory: str, skip_text_extraction: bool = False, skip_entity_extraction: bool = False):
    """
    Run the complete extraction pipeline.
    
    Args:
        pdf_directory: Directory containing PDF files
        skip_text_extraction: Skip text extraction if already done
        skip_entity_extraction: Skip entity extraction if already done
    """
    # Define all paths in one place
    BASE_DIR = Path(__file__).parent
    CONFIG_DIR = BASE_DIR / "config"
    OUTPUT_DIR = BASE_DIR / "outputs"
    EXTRACTED_TEXT_DIR = OUTPUT_DIR / "extracted_text"
    METADATA_DIR = OUTPUT_DIR / "metadata"
    ENTITIES_DIR = OUTPUT_DIR / "extracted_entities"
    
    print("="*60)
    print("T2D CLINICAL KNOWLEDGE GRAPH EXTRACTION PIPELINE")
    print("="*60)
    
    # STEP 1: PDF Text Extraction
    if not skip_text_extraction:
        print("\n[STEP 1/3] PDF TEXT EXTRACTION")
        print("-" * 60)
        
        pipeline = ExtractionPipeline(
            config_dir=str(CONFIG_DIR),
            data_dir=str(pdf_directory),
            output_dir=str(OUTPUT_DIR)
        )
        
        pipeline.export_metadata_catalog()
        pipeline.process_all_pdfs(pdf_directory)
    else:
        print("\n[STEP 1/3] SKIPPED - Text extraction already completed")
    
    # STEP 2: Entity and Relationship Extraction
    if not skip_entity_extraction:
        print("\n[STEP 2/3] ENTITY AND RELATIONSHIP EXTRACTION")
        print("-" * 60)
        
        orchestrator = ExtractionOrchestrator(
            extracted_text_dir=str(EXTRACTED_TEXT_DIR),
            metadata_dir=str(METADATA_DIR),
            output_dir=str(OUTPUT_DIR)
        )
        
        orchestrator.process_all()
    else:
        print("\n[STEP 2/3] SKIPPED - Entity extraction already completed")
    
    # STEP 3: Post-Processing (Normalization & Linking)
    print("\n[STEP 3/3] POST-PROCESSING (NORMALIZATION & LINKING)")
    print("-" * 60)
    
    processor = PostProcessor(
        entities_dir=str(ENTITIES_DIR),
        output_dir=str(OUTPUT_DIR)
    )
    
    processor.process_all()
    
    # Pipeline Complete
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE!")
    print("="*60)
    print("\nOutput files:")
    print(f"  📁 Extracted text:    {EXTRACTED_TEXT_DIR}")
    print(f"  📁 Extracted entities: {ENTITIES_DIR}")
    print(f"  📁 Linked entities:    {OUTPUT_DIR / 'linked'}")
    print(f"  📁 Metadata:           {METADATA_DIR}")
    print(f"  📊 Statistics:         {OUTPUT_DIR / 'post_processing_statistics.json'}")
    print()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="T2D Clinical Knowledge Graph Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python run_pipeline.py --pdf-dir data/raw_pdfs

  # Skip text extraction if already done
  python run_pipeline.py --pdf-dir data/raw_pdfs --skip-text

  # Skip both text and entity extraction, only run post-processing
  python run_pipeline.py --pdf-dir data/raw_pdfs --skip-text --skip-entities
        """
    )
    
    parser.add_argument(
        '--pdf-dir',
        type=str,
        default='data/raw_pdfs',
        help='Directory containing PDF files (default: data/raw_pdfs)'
    )
    
    parser.add_argument(
        '--skip-text',
        action='store_true',
        help='Skip PDF text extraction (if already completed)'
    )
    
    parser.add_argument(
        '--skip-entities',
        action='store_true',
        help='Skip entity extraction (if already completed)'
    )
    
    args = parser.parse_args()
    
    # Run the pipeline
    run_full_pipeline(
        pdf_directory=args.pdf_dir,
        skip_text_extraction=args.skip_text,
        skip_entity_extraction=args.skip_entities
    )


if __name__ == "__main__":
    main()