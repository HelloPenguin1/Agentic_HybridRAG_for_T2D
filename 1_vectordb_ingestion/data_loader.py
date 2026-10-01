import os
import json
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from llama_parse import LlamaParse
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document

# Load environment variables
load_dotenv()


class MedicalDataLoader:
    """
    Medical-grade data loader implementing 3-phase extraction:
    1. LlamaParse with layout agent (preserves tables, multi-column)
    2. Two-stage splitting (headers → recursive)
    3. Metadata enrichment (page numbers, source, section hierarchy)
    """

    def __init__(
        self,
        pdf_dir: str,
        output_dir: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ):
        """
        Initialize the Medical Data Loader.

        Args:
            pdf_dir: Directory containing raw PDF files
            output_dir: Directory to save processed chunks
            chunk_size: Target size for final chunks (tokens)
            chunk_overlap: Overlap between chunks (tokens)
        """
        self.pdf_dir = Path(pdf_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: Initialize LlamaParse with medical-specific configuration
        self.parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            result_type="markdown",
            verbose=True,
            disable_ocr=True,
            num_workers=4,
            system_prompt="""
            This is a medical document about clinical Diabetes guidelines.
            - Extract all text in logical reading order, ignoring column breaks.
            - Preserve all tables in Markdown format.
            - Identify section headers clearly (use # for main sections, ## for subsections).
            - Maintain the hierarchical structure of the document.
            """,
        )

        # Phase 2: Initialize two-stage splitters
        # Stage 1: Split by structural headers
        headers_to_split_on = [
            ("#", "chapter_name"),
            ("##", "section_heading"),
            ("###", "subsection_heading"),
        ]
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=False
        )

        # Stage 2: Split into manageable chunks for LLM context
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ".", " "],
        )

    def process_pdf(self, pdf_path: Path) -> List[Document]:
        """
        Process a single PDF through the 3-phase pipeline.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of traceable chunks ready for vector DB
        """
        print(f"Processing: {pdf_path.name}")

        # Phase 1: Extract with LlamaParse
        print(f"  → Phase 1: Parsing with layout agent...")
        llama_docs = self.parser.load_data(str(pdf_path))

        all_final_chunks = []

        # Process each page
        for i, page in enumerate(llama_docs):
            page_num = i + 1  # 1-indexed page number

            # Phase 2: Two-stage splitting
            # Stage 1: Split by headers to get structural metadata
            sections = self.header_splitter.split_text(page.text)

            for section in sections:
                # Phase 3: Metadata enrichment
                section.metadata["page_number"] = page_num
                section.metadata["source"] = pdf_path.name
                section.metadata["source_path"] = str(pdf_path)

                # Stage 2: Split into small chunks (inherits all metadata)
                chunks = self.text_splitter.split_documents([section])
                all_final_chunks.extend(chunks)

        print(f"  → Created {len(all_final_chunks)} traceable chunks")
        return all_final_chunks

    def save_chunks(self, chunks: List[Document], pdf_name: str):
        """
        Save chunks to disk as JSON.

        Args:
            chunks: List of Document objects with metadata
            pdf_name: Name of the source PDF
        """
        output_file = self.output_dir / f"{Path(pdf_name).stem}_chunks.json"

        # Convert documents to serializable format
        chunks_data = [
            {"page_content": chunk.page_content, "metadata": chunk.metadata}
            for chunk in chunks
        ]

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        print(f"  → Saved to: {output_file}")

    def process_all_pdfs(self):
        """
        Process all PDFs in the pdf_dir and save chunks to disk.
        Sequential processing for simplicity and reliability.
        """
        pdf_files = list(self.pdf_dir.glob("*.pdf"))

        if not pdf_files:
            print(f"No PDF files found in {self.pdf_dir}")
            return

        print(f"Found {len(pdf_files)} PDF files to process\n")
        print("=" * 60)

        success_count = 0
        error_count = 0

        for pdf_path in pdf_files:
            try:
                chunks = self.process_pdf(pdf_path)
                self.save_chunks(chunks, pdf_path.name)
                success_count += 1
                print()
            except Exception as e:
                print(f"  ✗ Error processing {pdf_path.name}: {e}\n")
                error_count += 1
                continue

        print("=" * 60)
        print(f"Processing complete!")
        print(f"  ✓ Success: {success_count}/{len(pdf_files)}")
        print(f"  ✗ Errors: {error_count}/{len(pdf_files)}")
        print(f"Chunks saved to: {self.output_dir}")

    def load_chunks_from_disk(self, chunk_file: str) -> List[Document]:
        """
        Load previously saved chunks from disk.

        Args:
            chunk_file: Path to the JSON file containing chunks

        Returns:
            List of Document objects
        """
        with open(chunk_file, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        documents = [
            Document(page_content=chunk["page_content"], metadata=chunk["metadata"])
            for chunk in chunks_data
        ]

        return documents

    def get_chunk_stats(self, chunks: List[Document]) -> dict:
        """
        Get statistics about the chunks for quality assessment.

        Args:
            chunks: List of Document objects

        Returns:
            Dictionary with statistics
        """
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes) if chunks else 0,
            "min_chunk_size": min(chunk_sizes) if chunks else 0,
            "max_chunk_size": max(chunk_sizes) if chunks else 0,
            "unique_pages": len(
                set(chunk.metadata.get("page_number") for chunk in chunks)
            ),
            "unique_chapters": len(
                set(
                    chunk.metadata.get("chapter_name")
                    for chunk in chunks
                    if "chapter_name" in chunk.metadata
                )
            ),
        }
