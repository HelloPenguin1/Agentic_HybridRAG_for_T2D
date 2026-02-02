"""
Main Extraction Pipeline for T2D Clinical Knowledge Graph
Handles PDF text extraction and routes to category-specific extraction logic
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import pdfplumber
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PDFMetadata:
    """Metadata for a single PDF file"""
    filename: str
    category: str
    title: str
    focus: str
    source_chapters: List[int]
    entity_types: List[str]
    relationship_types: List[str]
    extraction_priority: List[str]


class MetadataManager:
    """Manages PDF categorization and metadata"""
    
    def __init__(self, schema_path: str):
        """
        Initialize with metadata schema.
        
        Args:
            schema_path: Path to metadata_schema.yaml
        """
        with open(schema_path, 'r') as f:
            self.schema = yaml.safe_load(f)
        
        self.categories = self.schema['categories']
        self.pdf_to_category_map = self._build_pdf_mapping()
    
    def _build_pdf_mapping(self) -> Dict[str, PDFMetadata]:
        """Build a mapping from PDF filename to its metadata"""
        mapping = {}
        
        for category_name, category_data in self.categories.items():
            for pdf_info in category_data.get('pdf_mappings', []):
                metadata = PDFMetadata(
                    filename=pdf_info['filename'],
                    category=category_name,
                    title=pdf_info['title'],
                    focus=pdf_info['focus'],
                    source_chapters=category_data.get('source_chapters', []),
                    entity_types=category_data.get('entity_types', []),
                    relationship_types=category_data.get('relationship_types', []),
                    extraction_priority=category_data.get('extraction_priority', [])
                )
                mapping[pdf_info['filename']] = metadata
        
        return mapping
    
    def get_metadata(self, filename: str) -> Optional[PDFMetadata]:
        """Get metadata for a PDF file"""
        return self.pdf_to_category_map.get(filename)
    
    def get_category(self, filename: str) -> Optional[str]:
        """Get category name for a PDF file"""
        metadata = self.get_metadata(filename)
        return metadata.category if metadata else None
    
    def list_all_pdfs(self) -> List[str]:
        """List all expected PDF filenames"""
        return list(self.pdf_to_category_map.keys())
    
    def export_metadata_catalog(self, output_path: str):
        """Export a human-readable catalog of all PDFs and their metadata."""
        catalog = {}
        
        # Build catalog by category
        for category_name, category_data in self.categories.items():
            # Get all PDFs for this category
            category_pdfs = [
                {
                    'filename': filename,
                    'title': metadata.title,
                    'focus': metadata.focus,
                    'entity_types': metadata.entity_types,
                    'relationship_types': metadata.relationship_types
                }
                for filename, metadata in self.pdf_to_category_map.items()
                if metadata.category == category_name
            ]
            
            catalog[category_name] = {
                'description': category_data['description'],
                'pdfs': category_pdfs
            }
        
        with open(output_path, 'w') as f:
            json.dump(catalog, f, indent=2)


class PDFTextExtractor:
    """Extracts text from PDF files with page-level granularity"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Extract text from PDF, chunked by page and further split for manageable processing.
        
        Args:
            pdf_path: Path to PDF file
            chunk_size: Target character count per chunk
            chunk_overlap: Overlap between chunks to preserve context
        
        Returns:
            List of dicts with 'text', 'page', 'chunk_id', 'source'
        """
        chunks = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text()
                    
                    if not page_text or len(page_text.strip()) == 0:
                        continue
                    
                    # Split page into chunks if it's too long
                    if len(page_text) <= chunk_size:
                        chunks.append({
                            'text': page_text,
                            'page': page_num,
                            'chunk_id': f"p{page_num}_c1",
                            'source': os.path.basename(pdf_path),
                            'char_count': len(page_text)
                        })
                    else:
                        # Split long pages into overlapping chunks
                        page_chunks = PDFTextExtractor._split_text(
                            page_text, 
                            chunk_size, 
                            chunk_overlap
                        )
                        
                        for chunk_idx, chunk_text in enumerate(page_chunks, start=1):
                            chunks.append({
                                'text': chunk_text,
                                'page': page_num,
                                'chunk_id': f"p{page_num}_c{chunk_idx}",
                                'source': os.path.basename(pdf_path),
                                'char_count': len(chunk_text)
                            })
        
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return []
        
        return chunks
    
    @staticmethod
    def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    @staticmethod
    def save_extracted_text(chunks: List[Dict[str, Any]], output_path: str):
        """Save extracted text chunks to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(chunks, f, indent=2)


class ExtractionPipeline:
    """Main pipeline orchestrating the entire extraction process"""
    
    def __init__(self, config_dir: str, data_dir: str, output_dir: str):
        """
        Initialize extraction pipeline.
        
        Args:
            config_dir: Directory containing config files
            data_dir: Directory containing raw PDFs
            output_dir: Directory for output files
        """
        # Convert all paths to Path objects
        self.config_dir = Path(config_dir)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Set up output directories
        self.extracted_dir = self.output_dir / "extracted_text"
        self.metadata_dir = self.output_dir / "metadata"
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata manager
        schema_path = self.config_dir / "metadata_schema.yaml"
        self.metadata_manager = MetadataManager(str(schema_path))
    
    def process_all_pdfs(self, pdf_directory: str):
        """
        Process all PDFs in the specified directory.
        
        Args:
            pdf_directory: Path to directory containing PDFs
        """
        pdf_dir = Path(pdf_directory)
        
        if not pdf_dir.exists():
            print(f"PDF directory not found: {pdf_directory}")
            return
        
        # Find all PDF files
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in: {pdf_directory}")
            return
        
        results = {
            'processed': [],
            'errors': [],
            'unknown_pdfs': []
        }
        
        for pdf_path in pdf_files:
            filename = pdf_path.name
            print(f"Processing: {filename}")
            
            # Get metadata
            metadata = self.metadata_manager.get_metadata(filename)
            
            if not metadata:
                print(f"Unknown PDF (not in metadata schema): {filename}")
                results['unknown_pdfs'].append(filename)
                continue
            
            print(f"Category: {metadata.category}")
            
            # Extract text
            try:
                chunks = PDFTextExtractor.extract_text_from_pdf(str(pdf_path))
                
                if not chunks:
                    print("No text extracted")
                    results['errors'].append({
                        'filename': filename,
                        'error': 'No text extracted'
                    })
                    continue
                
                print(f"Extracted {len(chunks)} text chunks")
                
                # Save extracted text
                output_filename = f"{pdf_path.stem}_extracted.json"
                output_path = self.extracted_dir / output_filename
                PDFTextExtractor.save_extracted_text(chunks, str(output_path))
                
                # Save metadata for this PDF
                pdf_metadata_file = self.metadata_dir / f"{pdf_path.stem}_metadata.json"
                with open(pdf_metadata_file, 'w') as f:
                    json.dump({
                        'filename': filename,
                        'category': metadata.category,
                        'title': metadata.title,
                        'focus': metadata.focus,
                        'source_chapters': metadata.source_chapters,
                        'entity_types': metadata.entity_types,
                        'relationship_types': metadata.relationship_types,
                        'extraction_priority': metadata.extraction_priority,
                        'chunks_extracted': len(chunks),
                        'total_characters': sum(c['char_count'] for c in chunks),
                        'processed_at': datetime.now().isoformat()
                    }, f, indent=2)
                
                results['processed'].append({
                    'filename': filename,
                    'category': metadata.category,
                    'chunks': len(chunks)
                })
                
            except Exception as e:
                print(f"Error: {e}")
                results['errors'].append({
                    'filename': filename,
                    'error': str(e)
                })
        
        # Save processing summary
        summary_path = self.output_dir / "extraction_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Processing complete. Processed: {len(results['processed'])}, Errors: {len(results['errors'])}, Unknown: {len(results['unknown_pdfs'])}")
        print(f"Summary saved to: {summary_path}")
    
    def get_pdf_category_mapping(self, filename: str) -> Optional[str]:
        """Get the category for a given PDF filename"""
        return self.metadata_manager.get_category(filename)
    
    def export_metadata_catalog(self):
        """Export full metadata catalog"""
        catalog_path = self.metadata_dir / "metadata_catalog.json"
        self.metadata_manager.export_metadata_catalog(str(catalog_path))


def main():
    """Main entry point for extraction pipeline"""
    
    # Configuration
    BASE_DIR = Path(__file__).parent.parent
    CONFIG_DIR = BASE_DIR / "config"
    DATA_DIR = BASE_DIR / "data" / "raw_pdfs"
    OUTPUT_DIR = BASE_DIR / "outputs"
    
    # Initialize pipeline
    pipeline = ExtractionPipeline(
        config_dir=str(CONFIG_DIR),
        data_dir=str(DATA_DIR),
        output_dir=str(OUTPUT_DIR)
    )
    
    # Export metadata catalog
    pipeline.export_metadata_catalog()
    
    # Process all PDFs
    pipeline.process_all_pdfs(str(DATA_DIR))


if __name__ == "__main__":
    main()