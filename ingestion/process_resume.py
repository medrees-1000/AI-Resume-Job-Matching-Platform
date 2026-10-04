"""
Single resume processing for Streamlit uploads.
Replaces the old batch ingestion system.
"""

from pathlib import Path
import sys

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from ingestion.pdf_parser import extract_text_from_pdf, UnreliableExtractionError
from ingestion.chunking import chunk_text
from sentence_transformers import SentenceTransformer

# Initialize model once (will be cached)
_model = None

def get_embedding_model():
    """Lazy load the embedding model."""
    global _model
    if _model is None:
        # Using all-mpnet-base-v2: Higher quality embeddings (768 dims vs 384)
        # Better for resume matching - industry standard for semantic search
        _model = SentenceTransformer("all-mpnet-base-v2")
    return _model


def process_uploaded_resume(pdf_file):
    """
    Process a single uploaded PDF resume.
    
    Args:
        pdf_file: Streamlit UploadedFile object or file path
    
    Returns:
        dict: {
            "text": full resume text,
            "chunks": list of text chunks,
            "embeddings": list of embedding vectors,
            "success": bool,
            "error": str or None
        }
    """
    try:
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_file)
        
        if not text or len(text.strip()) < 50:
            return {
                "success": False,
                "error": "Could not extract sufficient text from PDF"
            }
        
        # Chunk the text (smaller chunks for better granularity)
        chunks = chunk_text(text, chunk_size=200, overlap=75)
        
        if not chunks:
            return {
                "success": False,
                "error": "Text chunking failed"
            }
        
        # Generate embeddings
        model = get_embedding_model()
        embeddings = model.encode(chunks)
        
        return {
            "success": True,
            "text": text,
            "chunks": chunks,
            "embeddings": embeddings,
            "error": None
        }
        
    except UnreliableExtractionError as e:
        return {
            "success": False,
            "extraction_warning": True,
            "error": str(e)
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Processing failed: {str(e)}"
        }


def process_job_description(job_text):
    """
    Process a job description text.
    
    Args:
        job_text: string of job description
    
    Returns:
        dict: {
            "embedding": embedding vector,
            "success": bool,
            "error": str or None
        }
    """
    try:
        if not job_text or len(job_text.strip()) < 20:
            return {
                "success": False,
                "error": "Job description is too short"
            }
        
        model = get_embedding_model()
        embedding = model.encode(job_text)
        
        return {
            "success": True,
            "embedding": embedding,
            "error": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Embedding failed: {str(e)}"
        }