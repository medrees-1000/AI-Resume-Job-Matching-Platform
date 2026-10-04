"""
AI Resume-Job Matching Platform
Integrated UI: Gemini's clean design + Original backend logic
"""

import streamlit as st
import sys
from pathlib import Path
import os

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import core functions
from ingestion.process_resume import process_uploaded_resume, process_job_description

# Try to use enhanced versions, fall back to originals if not available
from ingestion.job_cleaner import extract_requirements_section

from matching.keyword_matcher import extract_keywords, calculate_keyword_match, get_improvement_suggestions

from matching.similarity import calculate_match_score, get_top_matching_chunks
from matching.hybrid_scorer import calculate_hybrid_score, generate_score_explanation
from explanation.groq_explainer import generate_match_explanation_groq, generate_simple_explanation_fallback

# Score components: (key in score_breakdown, label, share of final score, what it measures)
SCORE_COMPONENTS = [
    ("technical_score", "Technical skills", 40, "Required and preferred skills found in the resume"),
    ("semantic_score", "Semantic fit", 30, "How closely the resume's content matches the job description"),
    ("experience_score", "Experience level", 20, "Experience-level wording compared with the posting"),
    ("education_score", "Education", 10, "Education-level wording compared with the posting"),
]


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Nikola AI | Smart Resume Analyzer",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Developer mode is off by default; open the app with ?dev=1 to show internals.
DEV_MODE = st.query_params.get("dev") == "1"

# --- STYLING ---
# Colours come from .streamlit/config.toml; this only adds layout details.
st.markdown("""
    <style>
    .header-container { padding: 0.5rem 0 1rem 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 1.5rem; }
    .main-title { font-size: 2rem; font-weight: 700; color: #0f172a; margin: 0; line-height: 1.2; }
    .sub-title { color: #64748b; font-size: 1rem; margin: 0; }

    .score-number { font-size: 4rem; font-weight: 700; line-height: 1; color: #0f172a; }
    .score-label { font-size: 1.1rem; font-weight: 600; color: #2563eb; margin-top: 0.4rem; }
    .score-label.muted { color: #475569; }
    .score-note { color: #64748b; font-size: 0.9rem; margin-top: 0.4rem; }

    .component-head { display: flex; justify-content: space-between; font-weight: 600; margin-top: 0.6rem; }
    .component-desc { color: #64748b; font-size: 0.85rem; margin-bottom: 0.2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### How it works")
    st.markdown(
        "1. Upload a text-based PDF resume\n"
        "2. Paste the full job posting\n"
        "3. Review the score, skill gaps and AI explanation"
    )

    st.divider()

    st.markdown("### How the score is built")
    st.markdown(
        "- **40%** Technical skills\n"
        "- **30%** Semantic fit\n"
        "- **20%** Experience level\n"
        "- **10%** Education"
    )
    st.caption("Tuned for technical roles: data, engineering, IT.")

    st.divider()

    if os.getenv("GROQ_API_KEY"):
        st.caption("AI explanations: on")
    else:
        st.caption("AI explanations: off. Set GROQ_API_KEY to enable them ([get a key](https://console.groq.com)).")

    if DEV_MODE:
        st.caption("Developer mode is on.")

# --- HEADER ---
st.markdown("""
<div class="header-container">
    <div style="display: flex; align-items: center; gap: 14px;">
        <svg width="44" height="44" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M13 2L3 14H12L11 22L21 10H12L13 2Z"
                stroke="#2563eb" stroke-width="2" fill="#2563eb"
                stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <div>
            <p class="main-title">Nikola AI</p>
            <p class="sub-title">Resume and job description matching with semantic embeddings and keyword analysis</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- INPUT COLUMNS ---
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Resume")
    st.caption(
        "PDF only, and it must contain selectable text (scanned images aren't supported). "
        "Standard layouts work best; password-protected files can't be read."
    )

    uploaded_file = st.file_uploader("Upload PDF Resume", type=["pdf"], label_visibility="collapsed")

with col2:
    st.subheader("Job description")
    st.caption(
        "Paste the full posting. Company boilerplate is removed and required skills are "
        "separated from preferred ones automatically."
    )

    job_description = st.text_area(
        "Paste the complete job posting:",
        value="",
        height=200,
        placeholder="Paste job description here...",
        label_visibility="collapsed"
    )

# --- ANALYSIS TRIGGER ---
st.markdown("<br>", unsafe_allow_html=True)

if st.button("Run Match Analysis", use_container_width=True, type="primary"):
    
    # Validation
    if not uploaded_file:
        st.error("Please upload a resume first.")
        st.stop()
    
    if not job_description or len(job_description.strip()) < 50:
        st.error("Please provide a job description (at least 50 characters).")
        st.stop()
    
    # Processing with progress
    with st.spinner("Processing resume and analyzing match..."):
        
        # Step 1: Process resume
        progress_bar = st.progress(0)
        st.caption("Step 1/5: Extracting text from PDF...")
        
        resume_result = process_uploaded_resume(uploaded_file)
        
        if resume_result.get("extraction_warning"):
            st.warning(f"{resume_result['error']} Scoring was skipped so you don't get a misleadingly low result.")
            st.stop()

        if not resume_result["success"]:
            st.error(f"Resume processing failed: {resume_result['error']}")
            st.stop()
        
        progress_bar.progress(20)
        st.caption("Step 2/5: Cleaning job description...")
        
        # Step 2: Clean and process job description
        job_sections = extract_requirements_section(job_description)
        cleaned_job_text = job_sections["cleaned_text"]
        
        job_result = process_job_description(cleaned_job_text)
        
        if not job_result["success"]:
            st.error(f"Job processing failed: {job_result['error']}")
            st.stop()
        
        progress_bar.progress(40)
        st.caption("Step 3/5: Calculating semantic similarity...")
        
        # Step 3: Calculate semantic similarity
        top_chunks = get_top_matching_chunks(
            resume_result["chunks"],
            resume_result["embeddings"],
            job_result["embedding"],
            top_k=5
        )
        
        # Average of top 3 chunks for semantic score
        semantic_score = sum([c["score"] for c in top_chunks[:3]]) / 3
        
        progress_bar.progress(60)
        st.caption("Step 4/5: Matching keywords (Required vs Preferred)...")
        
        # Step 4: Keyword matching with job sections
        resume_keywords = extract_keywords(resume_result["text"])
        job_keywords = extract_keywords(cleaned_job_text)
        keyword_results = calculate_keyword_match(resume_keywords, job_keywords, job_sections)
        
        progress_bar.progress(80)
        st.caption("Step 5/5: Generating hybrid score...")
        
        # Step 5: Hybrid scoring
        score_breakdown = calculate_hybrid_score(
            semantic_score,
            keyword_results,
            top_chunks
        )
        
        progress_bar.progress(100)
        st.caption("Analysis complete.")
        
    # Clear progress indicators
    progress_bar.empty()
    
    # --- RESULT ---
    st.markdown("---")
    
    hybrid_score = score_breakdown["hybrid_score"]

    if score_breakdown.get("job_description_too_short"):
        st.warning(
            "This job description mentions very few recognized skills, so the skill-match "
            "score is capped and less reliable. Paste the full posting for a better result."
        )
    
    # Match category thresholds
    if hybrid_score >= 0.85:
        match_category, strong = "Excellent match", True
    elif hybrid_score >= 0.71:
        match_category, strong = "Good match", True
    elif hybrid_score >= 0.40:
        match_category, strong = "Fair match", False
    else:
        match_category, strong = "Low match", False
    
    # One card: overall score on the left, the parts that make it up on the right.
    with st.container(border=True):
        left, right = st.columns([1, 2], gap="large")

        with left:
            st.markdown(
                f"""
                <div class="score-number">{hybrid_score * 100:.0f}%</div>
                <div class="score-label{'' if strong else ' muted'}">{match_category}</div>
                <div class="score-note">Overall match, combining the four components shown here.</div>
                """,
                unsafe_allow_html=True,
            )

        with right:
            st.markdown("**What makes up this score**")
            for key, label, weight, description in SCORE_COMPONENTS:
                value = score_breakdown[key]
                st.markdown(
                    f'<div class="component-head"><span>{label} <span style="font-weight:400;color:#64748b">'
                    f'({weight}% of score)</span></span><span>{value * 100:.0f}%</span></div>'
                    f'<div class="component-desc">{description}</div>',
                    unsafe_allow_html=True,
                )
                st.progress(min(max(value, 0.0), 1.0))
    
    # --- SKILLS ANALYSIS ---
    st.subheader("Skills analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Matched skills**")
        matched_skills = score_breakdown.get("matched_skills", [])
        if matched_skills:
            st.write(", ".join(matched_skills[:15]))  # Show up to 15 skills
            if len(matched_skills) > 15:
                st.caption(f"and {len(matched_skills) - 15} more")
        else:
            st.caption("No listed technical skills were found in the resume. It may use different terminology than the posting.")
    
    with col2:
        st.markdown("**Missing skills**")
        
        missing_required = score_breakdown.get("missing_required", [])
        missing_preferred = score_breakdown.get("missing_preferred", [])
        
        if missing_required:
            st.caption("Required (high priority)")
            st.write(", ".join(missing_required[:8]))  # Show up to 8
            if len(missing_required) > 8:
                st.caption(f"and {len(missing_required) - 8} more")
        
        if missing_preferred:
            st.caption("Preferred (nice to have)")
            st.write(", ".join(missing_preferred[:5]))  # Show up to 5
            if len(missing_preferred) > 5:
                st.caption(f"and {len(missing_preferred) - 5} more")
        
        if not missing_required and not missing_preferred:
            st.caption("No required or preferred skills are missing.")

    # --- DEVELOPER MODE (?dev=1) ---
    if DEV_MODE:
        with st.expander("Developer mode: extracted keywords and raw results"):
            st.write("**Resume keywords:**", resume_keywords if resume_keywords else "None detected")
            st.write("**Job keywords:**", job_keywords if job_keywords else "None detected")
            st.write("**Keyword results:**", keyword_results)
            st.write("**Chunk similarity scores:**", [round(c["score"], 3) for c in top_chunks])
    
    # --- TABS ---
    tab1, tab2 = st.tabs(["AI explanation", "Relevant resume sections"])
    
    with tab1:
        if os.getenv("GROQ_API_KEY"):
            with st.spinner("Generating AI explanation..."):
                explanation = generate_match_explanation_groq(
                    [c["chunk"] for c in top_chunks],
                    cleaned_job_text,
                    score_breakdown
                )
        else:
            explanation = generate_simple_explanation_fallback(score_breakdown)
        
        if explanation.get("error"):
            st.info(explanation["explanation"])
        else:
            st.markdown("**Why this score**")
            st.write(explanation["explanation"])
            
            if explanation["strengths"]:
                st.markdown("**Strengths**")
                for strength in explanation["strengths"]:
                    st.markdown(f"- {strength}")
            
            if explanation["gaps"]:
                st.markdown("**Areas for improvement**")
                for gap in explanation["gaps"]:
                    st.markdown(f"- {gap}")
            
            if explanation.get("suggestions"):
                st.markdown("**Suggestions**")
                for suggestion in explanation["suggestions"]:
                    st.markdown(f"- {suggestion}")
    
    with tab2:
        st.markdown("**Resume sections most relevant to this job**")
        st.caption("The passages of the resume closest in meaning to the job description. They determine the semantic fit score.")
        
        for i, chunk in enumerate(top_chunks[:3], 1):
            with st.container(border=True):
                st.markdown(chunk["chunk"])
                if DEV_MODE:
                    st.caption(f"Section {i}, similarity {chunk['score']:.1%}")

# --- FOOTER ---
st.markdown("---")
st.caption("Embeddings: all-mpnet-base-v2. Explanations: Groq. Scores are heuristic and meant as a screening aid.")
