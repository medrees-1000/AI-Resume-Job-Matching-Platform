from pypdf import PdfReader
from pathlib import Path
from typing import Optional

def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    """
    Extracts and cleans text from a single PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Cleaned text or None if extraction fails
    """
    try:
        reader = PdfReader(pdf_path)
        pages_text = []
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                # Basic cleaning: collapse multiple spaces and newlines
                text = " ".join(text.split())
                pages_text.append(text)
        
        full_text = "\n".join(pages_text)
        
        # Return None if text is suspiciously short (likely a scanned PDF)
        if len(full_text.strip()) < 50:
            print(f"⚠️  {pdf_path.name}: Text too short ({len(full_text)} chars) - might be scanned image")
            return None
            
        return full_text
        
    except Exception as e:
        print(f"❌ Error reading {pdf_path.name}: {e}")
        return None