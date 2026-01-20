"""
AI Resume-Job Matching Platform
Interactive Streamlit UI for semantic resume matching
"""

import streamlit as st
import sys
from pathlib import Path
import os

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from ingestion.process_resume import process_uploaded_resume, process_job_description
from matching.similarity import calculate_match_score, get_top_matching_chunks
from rag.explainer import (
    generate_match_explanation, 
    generate_simple_explanation,
    get_remaining_credit,
    FreeAPILimitError
)

# Page config
st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .match-score {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
    }
    .high-match {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .medium-match {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .low-match {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🎯 AI Resume-Job Matching Platform</p>', unsafe_allow_html=True)
st.markdown("**Semantic matching powered by Sentence-BERT + RAG explanations**")

# Sidebar - API Credit Monitor
with st.sidebar:
    st.header("⚙️ System Status")
    
    credit_info = get_remaining_credit()
    st.metric("API Credits Used", f"${credit_info['used']:.2f}")
    st.metric("Remaining", f"${credit_info['remaining']:.2f}")
    
    progress = min(credit_info['percentage'] / 100, 1.0)
    st.progress(progress)
    
    if credit_info['remaining'] < 0.5:
        st.warning("⚠️ Low credit! Explanations may be limited.")
    
    st.divider()
    st.info("💡 **How it works:**\n\n1. Upload resume PDF\n2. Paste job description\n3. Get AI-powered match analysis")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📄 Upload Resume")
    uploaded_file = st.file_uploader(
        "Upload candidate resume (PDF only)",
        type=["pdf"],
        help="Upload a single PDF resume to analyze"
    )
    
    if uploaded_file:
        st.success(f"✅ Uploaded: {uploaded_file.name}")

with col2:
    st.subheader("💼 Job Description")
    
    # Option to load sample jobs
    sample_jobs = {
        "Select a sample...": "",
        "Data Scientist": Path("data/jobs/data_scientist.txt"),
        "Data Analyst": Path("data/jobs/data_analyst.txt"),
        "Software Engineer": Path("data/jobs/software_engineer.txt"),
        "DevOps Engineer": Path("data/jobs/devops_engineer.txt"),
        "Cybersecurity Specialist": Path("data/jobs/cybersecurity.txt")
    }
    
    selected_sample = st.selectbox("Load sample job:", list(sample_jobs.keys()))
    
    # Load sample if selected
    default_text = ""
    if selected_sample != "Select a sample..." and sample_jobs[selected_sample].exists():
        default_text = sample_jobs[selected_sample].read_text()
    
    job_description = st.text_area(
        "Paste or edit job description:",
        value=default_text,
        height=300,
        help="Paste the full job description here"
    )

# Analysis button
st.divider()

if st.button("🚀 Analyze Match", type="primary"):
    # Validation
    if not uploaded_file:
        st.error("❌ Please upload a resume first!")
        st.stop()
    
    if not job_description or len(job_description.strip()) < 50:
        st.error("❌ Please provide a job description (at least 50 characters)!")
        st.stop()
    
    # Processing
    with st.spinner("🔄 Processing resume and analyzing match..."):
        
        # Step 1: Process resume
        resume_result = process_uploaded_resume(uploaded_file)
        
        if not resume_result["success"]:
            st.error(f"❌ Resume processing failed: {resume_result['error']}")
            st.stop()
        
        # Step 2: Process job description
        job_result = process_job_description(job_description)
        
        if not job_result["success"]:
            st.error(f"❌ Job processing failed: {job_result['error']}")
            st.stop()
        
        # Step 3: Calculate similarity scores
        top_chunks = get_top_matching_chunks(
            resume_result["chunks"],
            resume_result["embeddings"],
            job_result["embedding"],
            top_k=5
        )
        
        # Overall match score (average of top 3 chunks)
        overall_score = sum([c["score"] for c in top_chunks[:3]]) / 3
        
    # Display Results
    st.success("✅ Analysis Complete!")
    
    # Match Score Display
    st.subheader("📊 Match Score")
    
    if overall_score >= 0.7:
        score_class = "high-match"
        verdict = "🔥 Excellent Match"
    elif overall_score >= 0.5:
        score_class = "medium-match"
        verdict = "✅ Good Match"
    else:
        score_class = "low-match"
        verdict = "⚠️ Moderate Match"
    
    st.markdown(
        f'<div class="match-score {score_class}">{overall_score:.1%}<br><small>{verdict}</small></div>',
        unsafe_allow_html=True
    )
    
    # Results tabs
    tab1, tab2, tab3 = st.tabs(["🤖 AI Explanation", "🔍 Matching Sections", "📋 Full Resume"])
    
    with tab1:
        st.subheader("Why This Candidate Matches")
        
        try:
            # Generate RAG explanation
            with st.spinner("Generating AI explanation..."):
                explanation = generate_match_explanation(
                    [c["chunk"] for c in top_chunks],
                    job_description,
                    overall_score
                )
            
            st.write(explanation["explanation"])
            
            if explanation["strengths"]:
                st.markdown("**✨ Key Strengths:**")
                for strength in explanation["strengths"]:
                    st.markdown(f"- {strength}")
            
            if explanation["gaps"]:
                st.markdown("**⚠️ Potential Gaps:**")
                for gap in explanation["gaps"]:
                    st.markdown(f"- {gap}")
            
            st.caption(f"💰 Cost: ${explanation.get('cost', 0):.4f}")
            
        except FreeAPILimitError as e:
            st.warning(str(e))
            st.info("Showing basic analysis instead:")
            
            fallback = generate_simple_explanation(
                [c["chunk"] for c in top_chunks],
                overall_score
            )
            st.write(fallback["explanation"])
            
        except Exception as e:
            st.error(f"Explanation generation failed: {str(e)}")
    
    with tab2:
        st.subheader("Top Matching Resume Sections")
        
        for i, chunk in enumerate(top_chunks[:3], 1):
            with st.expander(f"Match #{i} - Score: {chunk['score']:.1%}", expanded=(i==1)):
                st.write(chunk["chunk"])
    
    with tab3:
        st.subheader("Full Resume Text")
        st.text_area("Complete resume content:", resume_result["text"], height=400)

# Footer
st.divider()
st.caption("🧠 Powered by Sentence-BERT embeddings + Claude API | Built for semantic job matching")