import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_match_score(resume_embedding, job_embedding):
    """
    Calculates cosine similarity between resume and job embeddings.
    
    Args:
        resume_embedding: numpy array or list (384-dim vector)
        job_embedding: numpy array or list (384-dim vector)
    
    Returns:
        float: Similarity score between 0 and 1
    """
    # Convert to numpy arrays if needed
    resume_vec = np.array(resume_embedding).reshape(1, -1)
    job_vec = np.array(job_embedding).reshape(1, -1)
    
    # Calculate cosine similarity
    score = cosine_similarity(resume_vec, job_vec)[0][0]
    
    return float(score)


def get_top_matching_chunks(resume_chunks, resume_embeddings, job_embedding, top_k=3):
    """
    Finds the most relevant resume chunks for a job description.
    
    Args:
        resume_chunks: list of text chunks from resume
        resume_embeddings: list of embeddings for each chunk
        job_embedding: embedding vector for job description
        top_k: number of top matches to return
    
    Returns:
        list of dicts: [{"chunk": text, "score": float}, ...]
    """
    scores = []
    
    for i, chunk_embedding in enumerate(resume_embeddings):
        score = calculate_match_score(chunk_embedding, job_embedding)
        scores.append({
            "chunk": resume_chunks[i],
            "score": score,
            "index": i
        })
    
    # Sort by score descending
    scores.sort(key=lambda x: x["score"], reverse=True)
    
    return scores[:top_k]