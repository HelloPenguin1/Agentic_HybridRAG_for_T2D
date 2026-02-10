"""
Convert extracted JSON files to plain text for LangChain web UI
"""
import json
from pathlib import Path

# Directories
extracted_text_dir = Path("outputs/extracted_text")
output_dir = Path("outputs/text_files")
output_dir.mkdir(exist_ok=True)

# Process all JSON files
json_files = list(extracted_text_dir.glob("*_extracted.json"))

print(f"Found {len(json_files)} JSON files to convert\n")

for json_file in json_files:
    # Read JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract text from all chunks
    all_text = []
    for chunk in data:
        text = chunk.get("text", "")
        all_text.append(text)
    
    # Combine chunks with separator
    combined_text = "\n\n=== CHUNK SEPARATOR ===\n\n".join(all_text)
    
    # Write to .txt file
    output_file = output_dir / f"{json_file.stem.replace('_extracted', '')}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(combined_text)
    
    print(f"✓ Converted: {json_file.name} → {output_file.name}")
    print(f"  Characters: {len(combined_text):,}")
    print(f"  Chunks: {len(data)}\n")

print(f"\nAll files saved to: {output_dir.absolute()}")
print("\nYou can now upload the .txt files to the LangChain web UI!")
