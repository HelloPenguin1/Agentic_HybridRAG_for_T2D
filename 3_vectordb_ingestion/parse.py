"""
Medical-Grade PDF Extraction Pipeline
Implements 3-phase extraction for Type 2 Diabetes documents.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vectordb_ingestion.data_loader import MedicalDataLoader

# Define paths
PDF_DIR = r"t2d_extraction_pipeline\data\raw_pdfs"
OUTPUT_DIR = r"vectordb_ingestion\processed_chunks"

def main():
    """
    Process all PDFs with medical-grade extraction pipeline.
    """
    print("=" * 60)
    print("Medical-Grade PDF Extraction Pipeline")
    print("3-Phase Process: Parse → Split → Enrich")
    print("=" * 60)
    print()
    
    # Initialize loader
    loader = MedicalDataLoader(
        pdf_dir=PDF_DIR,
        output_dir=OUTPUT_DIR,
        chunk_size=1000,  # Adjust based on your embedding model
        chunk_overlap=200
    )
    
    # Process all PDFs
    loader.process_all_pdfs()
    
    print()
    print("=" * 60)
    print("✓ All PDFs processed successfully!")
    print(f"Output location: {Path(OUTPUT_DIR).absolute()}")
    print()
    print("Each chunk contains:")
    print("  - chapter_name (if applicable)")
    print("  - section_heading (if applicable)")
    print("  - subsection_heading (if applicable)")
    print("  - page_number")
    print("  - source (filename)")
    print("  - source_path (full path)")
    print("=" * 60)


def inspect_chunks():
    """
    Example: Inspect chunks from a processed file.
    """
    loader = MedicalDataLoader(
        pdf_dir=PDF_DIR,
        output_dir=OUTPUT_DIR
    )
    
    # Load chunks from a specific file
    chunk_file = Path(OUTPUT_DIR) / "ada_cardio_disease_manag_chunks.json"
    
    if chunk_file.exists():
        chunks = loader.load_chunks_from_disk(str(chunk_file))
        stats = loader.get_chunk_stats(chunks)
        
        print(f"\nChunk Statistics for {chunk_file.name}:")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Avg chunk size: {stats['avg_chunk_size']:.0f} chars")
        print(f"  Min/Max size: {stats['min_chunk_size']}/{stats['max_chunk_size']} chars")
        print(f"  Unique pages: {stats['unique_pages']}")
        print(f"  Unique chapters: {stats['unique_chapters']}")
        
        # Show first chunk as example
        if chunks:
            print("\nFirst chunk preview:")
            print(f"Content: {chunks[0].page_content[:200]}...")
            print(f"Metadata: {chunks[0].metadata}")
    else:
        print(f"Chunk file not found: {chunk_file}")


if __name__ == "__main__":
    # Run the main processing
    main()
    
    # Uncomment to inspect chunks after processing
    # inspect_chunks()
