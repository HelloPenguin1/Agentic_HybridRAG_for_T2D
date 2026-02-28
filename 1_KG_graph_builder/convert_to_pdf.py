"""
Convert text files to PDF for LangChain web UI
"""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT

# Directories
text_dir = Path("outputs/text_files")
pdf_dir = Path("outputs/pdf_files")
pdf_dir.mkdir(exist_ok=True)

# Get all text files
text_files = list(text_dir.glob("*.txt"))

print(f"Found {len(text_files)} text files to convert to PDF\n")

# Create custom style for medical text
styles = getSampleStyleSheet()
custom_style = ParagraphStyle(
    'CustomStyle',
    parent=styles['Normal'],
    fontSize=10,
    leading=12,
    alignment=TA_LEFT,
    fontName='Helvetica'
)

for text_file in text_files:
    print(f"Converting: {text_file.name}...")
    
    # Read text
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create PDF
    pdf_file = pdf_dir / f"{text_file.stem}.pdf"
    doc = SimpleDocTemplate(
        str(pdf_file),
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    # Build content
    story = []
    
    # Split into paragraphs
    paragraphs = text.split('\n\n')
    
    for para_text in paragraphs:
        if para_text.strip():
            # Clean text for PDF
            para_text = para_text.replace('\r', '')
            para_text = para_text.replace('\x00', '')  # Remove null bytes
            
            # Handle special characters
            para_text = para_text.replace('&', '&amp;')
            para_text = para_text.replace('<', '&lt;')
            para_text = para_text.replace('>', '&gt;')
            
            try:
                para = Paragraph(para_text, custom_style)
                story.append(para)
                story.append(Spacer(1, 0.1*inch))
            except Exception as e:
                # Skip problematic paragraphs
                print(f"  Warning: Skipped paragraph due to: {e}")
                continue
    
    # Build PDF
    try:
        doc.build(story)
        print(f"  ✓ Created: {pdf_file.name}")
        print(f"    Size: {pdf_file.stat().st_size / 1024:.1f} KB\n")
    except Exception as e:
        print(f"  ✗ Failed: {e}\n")

print(f"\nAll PDFs saved to: {pdf_dir.absolute()}")
print("\nYou can now upload the PDF files to the LangChain web UI!")
