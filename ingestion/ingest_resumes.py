import pandas as pd
from pathlib import Path
from tqdm import tqdm
from pdf_parser import extract_text_from_pdf

# Define paths
RESUME_DIR = Path("data/resumes")
OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "processed_resumes.csv"

# Ensure directories exist
RESUME_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    """Ingest all PDF resumes and save to CSV."""
    
    # Get list of PDFs
    pdf_files = list(RESUME_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"⚠️  No PDFs found in {RESUME_DIR}. Please add your 20 resumes there first!")
        return
    
    print(f" Starting ingestion of {len(pdf_files)} resumes...")
    
    records = []
    failed_files = []
    
    for pdf_file in tqdm(pdf_files, desc="Processing"):
        text = extract_text_from_pdf(pdf_file)
        
        if text:  # Only add if extraction succeeded
            records.append({
                "resume_id": pdf_file.stem,
                "file_name": pdf_file.name,
                "raw_text": text,
                "text_length": len(text)  # Track length for quality check
            })
        else:
            failed_files.append(pdf_file.name)
    
    # Create DataFrame and save
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print(f" Successfully ingested: {len(df)} resumes")
    print(f" Failed to process: {len(failed_files)} resumes")
    
    if failed_files:
        print(f"\n  Failed files:")
        for fname in failed_files:
            print(f"   - {fname}")
    
    if len(df) > 0:
        print(f"\n Statistics:")
        print(f"   Average text length: {df['text_length'].mean():.0f} characters")
        print(f"   Min length: {df['text_length'].min()} characters")
        print(f"   Max length: {df['text_length'].max()} characters")
    
    print(f"\n Data saved to: {OUTPUT_FILE}")
    print("="*60)
    
    return df

if __name__ == "__main__":
    df = main()
    
    # Show sample preview
    if df is not None and len(df) > 0:
        print("\n Sample preview (first resume):")
        sample_text = df.iloc[0]['raw_text'][:300]
        print(sample_text + "...\n")