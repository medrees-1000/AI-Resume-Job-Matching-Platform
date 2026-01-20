"""
Database utilities for resume-job matching system.
Uses SQLite for local storage.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime

DB_PATH = Path("data/resumes.db")

def get_connection():
    """Create a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Access columns by name
    return conn


def initialize_database():
    """Create tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Resumes table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Resume chunks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resume_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding BLOB,
            FOREIGN KEY (resume_id) REFERENCES resumes(id)
        )
    """)
    
    # Jobs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            description TEXT NOT NULL,
            embedding BLOB,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()


def save_resume(filename, raw_text, chunks, embeddings):
    """
    Save a resume and its chunks to the database.
    
    Args:
        filename: original PDF filename
        raw_text: full extracted text
        chunks: list of text chunks
        embeddings: list of numpy arrays
    
    Returns:
        int: resume_id
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Insert resume
    cursor.execute(
        "INSERT INTO resumes (filename, raw_text) VALUES (?, ?)",
        (filename, raw_text)
    )
    resume_id = cursor.lastrowid
    
    # Insert chunks with embeddings
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # Convert numpy array to bytes
        embedding_blob = embedding.tobytes()
        
        cursor.execute(
            """INSERT INTO resume_chunks 
               (resume_id, chunk_index, chunk_text, embedding) 
               VALUES (?, ?, ?, ?)""",
            (resume_id, i, chunk, embedding_blob)
        )
    
    conn.commit()
    conn.close()
    
    return resume_id


def save_job(title, description, embedding):
    """
    Save a job description to the database.
    
    Args:
        title: job title
        description: job description text
        embedding: numpy array
    
    Returns:
        int: job_id
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    embedding_blob = embedding.tobytes()
    
    cursor.execute(
        "INSERT INTO jobs (title, description, embedding) VALUES (?, ?, ?)",
        (title, description, embedding_blob)
    )
    job_id = cursor.lastrowid
    
    conn.commit()
    conn.close()
    
    return job_id


def get_all_resumes():
    """Get list of all resumes."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, filename, upload_date FROM resumes ORDER BY upload_date DESC")
    resumes = cursor.fetchall()
    
    conn.close()
    return resumes


def get_resume_chunks(resume_id):
    """Get all chunks for a specific resume."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        """SELECT chunk_index, chunk_text, embedding 
           FROM resume_chunks 
           WHERE resume_id = ? 
           ORDER BY chunk_index""",
        (resume_id,)
    )
    chunks = cursor.fetchall()
    
    conn.close()
    return chunks


def clear_database():
    """Delete all data (useful for testing)."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM resume_chunks")
    cursor.execute("DELETE FROM resumes")
    cursor.execute("DELETE FROM jobs")
    
    conn.commit()
    conn.close()