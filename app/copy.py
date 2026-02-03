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
from ingestion.job_cleaner import extract_requirements_section
from matching.similarity import calculate_match_score, get_top_matching_chunks
from matching.keyword_matcher import extract_keywords, calculate_keyword_match, get_improvement_suggestions
from matching.hybrid_scorer import calculate_hybrid_score, generate_score_explanation
from rag.groq_explainer import generate_match_explanation_groq, generate_simple_explanation_fallback

# --- CUSTOM SMOOTH CIRCLE COMPONENT ---
def render_full_circle_gauge(percent, label, size=150, color="#6366f1", font_size="22px"):
    """Renders a smooth, full-circle SVG gauge with a professional look."""
    radius = 40
    circumference = 2 * 3.14159 * radius
    offset = circumference - (percent / 100) * circumference
    
    # Adjust vertical position based on size for better centering
    text_y = 58 if size > 200 else 55
    
    return f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin: 10px;">
        <svg width="{size}" height="{size}" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="{radius}" stroke="#f3f4f6" stroke-width="8" fill="transparent" />
            <circle cx="50" cy="50" r="{radius}" stroke="{color}" stroke-width="8" 
                stroke-dasharray="{circumference}" stroke-dashoffset="{offset}" 
                stroke-linecap="round" fill="transparent" transform="rotate(-92 50 50)"
                style="transition: stroke-dashoffset 1s ease-in-out;" />
            <text x="50" y="54" font-family="'Inter', sans-serif" font-size="{font_size}" font-weight="700" text-anchor="middle" dominant-baseline="middle" fill="#1f2937">{int(percent)}%</text>
        </svg>
        <div style="font-family: 'Inter', sans-serif; font-weight: 600; color: #6b7280; font-size: 0.85rem; margin-top: -5px;">{label}</div>
    </div>
    """

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Resume Matcher", page_icon="🎯", layout="wide")

# --- STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* Overall Font */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Header Styling */
    .header-container { text-align: left; padding: 1rem 0; border-bottom: 1px solid #e5e7eb; margin-bottom: 2rem; }
    .main-title { font-size: 2.2rem; font-weight: 800; color: #111827; margin: 0; }
    .sub-title { color: #6b7280; font-size: 1.1rem; }

    /* Info Boxes - Keeping the "Alive" look but smoothing edges */
    .alive-box { padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem; border-left: 6px solid; }
    .info { background: #eff6ff; border-left-color: #3b82f6; color: #1e3a8a; }
    .warning { background: #fffbeb; border-left-color: #f59e0b; color: #78350f; }
    .success { background: #f0fdf4; border-left-color: #22c55e; color: #14532d; }
    
    /* Verdict Badge */
    .verdict-badge {
        padding: 8px 20px; border-radius: 50px; font-weight: 700; font-size: 1.2rem;
        display: inline-block; margin-top: 1rem;
    }
    .excellent { background: #dcfce7; color: #166534; }
    .good { background: #dbeafe; color: #1e40af; }
    .fair { background: #fef9c3; color: #854d0e; }
    .low { background: #fee2e2; color: #991b1b; }
    
    /* Sidebar Cleanup */
    .css-163ttbj { background-color: #f9fafb; }
    
    /* Run Analysis Button - Red/Coral */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #f43f5e 0%, #fb7185 100%) !important;
        color: white !important;
        font-weight: bold;
        padding: 0.75rem;
        font-size: 1.1rem;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #e11d48 0%, #f43f5e 100%) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 📋 How It Works")
    st.markdown("""
    **1. Upload Resume**
    - PDF format only
    - Text-based (not scanned)
    
    **2. Paste Job Description**
    - Paste everything - we clean it!
    
    **3. Get Analysis**
    - Match score breakdown
    - Skill gap analysis
    - AI-powered insights
    """)
    
    st.divider()
    
    st.markdown("### 🎯 Scoring Method")
    st.caption("Hybrid Breakdown:")
    st.markdown("- 40% Keyword matching\n- 30% Semantic similarity\n- 20% Experience match\n- 10% Education match")
    
    st.info("**Note:** This system is optimized for technical roles (Data Science, Engineering, IT, etc.)")
    
    st.divider()
    
    # GROQ Check
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        st.success("✅ AI Explanations: Active")
    else:
        st.warning("⚠️ AI Explanations: Disabled\n\nAdd GROQ_API_KEY to .env")
        st.caption("[Get free key](https://console.groq.com)")

# --- HEADER ---
st.markdown("""
<div class="header-container">
    <p class="main-title">Resume Match Intelligence</p>
    <p class="sub-title">Semantic Embeddings • Keyword Matching • AI Insights</p>
</div>
""", unsafe_allow_html=True)

# --- INPUT COLUMNS ---
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("#### 📄 Resume Source")
    
    st.markdown("""
    <div class="alive-box info">
    <b>✅ What Works Best:</b><br><br>
    • <b>PDF format only</b> (no Word docs)<br>
    • <b>Text-based PDF</b> - you can select/copy text<br>
    • <b>Standard layouts</b> - traditional resume format
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="alive-box warning">
    <b>⚠️ Won't Work:</b><br><br>
    • <b>Scanned images</b> saved as PDF<br>
    • <b>Highly graphic/creative</b> resumes<br>
    • <b>Password-protected</b> files
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload PDF Resume", type=["pdf"], label_visibility="collapsed")
    
    if uploaded_file:
        st.markdown(f"""
        <div class="alive-box success">
        <b>✅ File Uploaded!</b><br>
        {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("#### 💼 Job Description")
    
    st.markdown("""
    <div class="alive-box success">
    <b>✨ Pro Tip:</b><br><br>
    • Paste the <b>entire</b> job post. Our AI cleans the fluff automatically!<br>
    • The more detail, the better your results!
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="alive-box info">
    <b>✨ Auto-Cleaning:</b><br><br>
    Our AI automatically removes:<br>
    • Company fluff & benefits<br>
    • Separates Required vs Preferred skills<br>
    • Focuses on actual qualifications
    </div>
    """, unsafe_allow_html=True)
    
    job_description = st.text_area(
        "Paste the complete job posting:",
        value="",
        height=200,
        placeholder="Paste job description here...",
        label_visibility="collapsed"
    )

# --- ANALYSIS TRIGGER ---
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 Run Match Analysis", use_container_width=True, type="primary"):
    
    # Validation
    if not uploaded_file:
        st.error("❌ Please upload a resume first!")
        st.stop()
    
    if not job_description or len(job_description.strip()) < 50:
        st.error("❌ Please provide a job description (at least 50 characters)!")
        st.stop()
    
    # Processing with progress
    with st.spinner("🔄 Processing resume and analyzing match..."):
        
        # Step 1: Process resume
        progress_bar = st.progress(0)
        st.caption("Step 1/5: Extracting text from PDF...")
        
        resume_result = process_uploaded_resume(uploaded_file)
        
        if not resume_result["success"]:
            st.error(f"❌ Resume processing failed: {resume_result['error']}")
            st.stop()
        
        progress_bar.progress(20)
        st.caption("Step 2/5: Cleaning job description...")
        
        # Step 2: Clean and process job description
        job_sections = extract_requirements_section(job_description)
        cleaned_job_text = job_sections["cleaned_text"]
        
        job_result = process_job_description(cleaned_job_text)
        
        if not job_result["success"]:
            st.error(f"❌ Job processing failed: {job_result['error']}")
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
        st.caption("✅ Analysis complete!")
        
    # Clear progress indicators
    progress_bar.empty()
    
    # Display Results
    st.success("✅ Analysis Complete!")
    
    # --- OVERALL SCORE DISPLAY ---
    st.markdown("---")
    
    hybrid_score = score_breakdown["hybrid_score"]
    
    # Determine match category based on NEW thresholds
    if hybrid_score >= 0.85:
        match_category = "Excellent Match"
        badge_class = "excellent"
        gauge_color = "#22c55e"
    elif hybrid_score >= 0.71:
        match_category = "Good Match"
        badge_class = "good"
        gauge_color = "#3b82f6"
    elif hybrid_score >= 0.40:
        match_category = "Fair Match"
        badge_class = "fair"
        gauge_color = "#f59e0b"
    else:
        match_category = "Low Match"
        badge_class = "low"
        gauge_color = "#ef4444"
    
    # Big circle centered, small circles underneath in a row
    st.markdown("### Match Verdict")
    
    # Main gauge
    col_center = st.columns([1, 2, 1])
    with col_center[1]:
        st.components.v1.html(
            render_full_circle_gauge(hybrid_score * 100, "", size=280, color=gauge_color, font_size="22px"), 
            height=300
        )
        st.markdown(f'<div class="verdict-badge {badge_class}" style="text-align: center;">{match_category}</div>', 
                   unsafe_allow_html=True)
    
    st.markdown("### Score Breakdown")
    
    # Small circles in a row
    m1, m2, m3, m4 = st.columns(4)
    
    tech_score = score_breakdown['technical_score'] * 100
    sem_score = score_breakdown['semantic_score'] * 100
    exp_score = score_breakdown['experience_score'] * 100
    edu_score = score_breakdown['education_score'] * 100
    
    with m1: 
        st.components.v1.html(render_full_circle_gauge(tech_score, "Technical", size=130, color="#818cf8", font_size="20px"), height=160)
    with m2: 
        st.components.v1.html(render_full_circle_gauge(sem_score, "Semantic", size=130, color="#818cf8", font_size="20px"), height=160)
    with m3: 
        st.components.v1.html(render_full_circle_gauge(exp_score, "Experience", size=130, color="#818cf8", font_size="20px"), height=160)
    with m4: 
        st.components.v1.html(render_full_circle_gauge(edu_score, "Education", size=130, color="#818cf8", font_size="20px"), height=160)
    
    # --- SKILLS ANALYSIS ---
    st.markdown("---")
    st.markdown("### 🎯 Skills Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**✅ Matched Skills**")
        matched_skills = score_breakdown["matched_skills"]
        if matched_skills:
            for skill in matched_skills[:10]:
                st.markdown(f"✓ {skill}")
            if len(matched_skills) > 10:
                st.caption(f"...and {len(matched_skills) - 10} more")
        else:
            st.info("No specific technical skills detected")
    
    with col2:
        st.markdown("**⚠️ Missing Skills**")
        
        missing_required = score_breakdown.get("missing_required", [])
        missing_preferred = score_breakdown.get("missing_preferred", [])
        
        if missing_required:
            st.markdown("**🔴 Required (High Priority):**")
            for skill in missing_required[:5]:
                st.markdown(f"❌ {skill}")
        
        if missing_preferred:
            st.markdown("**🟡 Preferred (Nice to Have):**")
            for skill in missing_preferred[:3]:
                st.markdown(f"⚠️ {skill}")
        
        if not missing_required and not missing_preferred:
            st.success("All skills found!")
    
    # --- TABS ---
    st.markdown("---")
    tab1, tab2 = st.tabs(["🤖 AI Explanation", "🔍 Matching Sections"])
    
    with tab1:
        st.subheader("AI-Powered Analysis")
        
        if os.getenv("GROQ_API_KEY"):
            with st.spinner("Generating AI explanation..."):
                explanation = generate_match_explanation_groq(
                    [c["chunk"] for c in top_chunks],
                    cleaned_job_text,
                    score_breakdown
                )
        else:
            explanation = generate_simple_explanation_fallback(score_breakdown)
        
        st.markdown("**Why This Candidate Matches:**")
        st.write(explanation["explanation"])
        
        if explanation["strengths"]:
            st.markdown("**✨ Key Strengths:**")
            for strength in explanation["strengths"]:
                st.markdown(f"- {strength}")
        
        if explanation["gaps"]:
            st.markdown("**⚠️ Areas for Improvement:**")
            for gap in explanation["gaps"]:
                st.markdown(f"- {gap}")
        
        if explanation.get("suggestions"):
            st.markdown("**💡 Suggestions:**")
            for suggestion in explanation["suggestions"]:
                st.markdown(f"- {suggestion}")
    
    with tab2:
        st.subheader("Top Matching Resume Sections")
        
        for i, chunk in enumerate(top_chunks[:5], 1):
            with st.expander(f"Match #{i} - Relevance: {chunk['score']:.1%}", expanded=(i==1)):
                st.markdown(f"**Similarity Score:** {chunk['score']:.1%}")
                st.text_area(
                    f"Section {i}",
                    chunk["chunk"],
                    height=150,
                    key=f"chunk_{i}",
                    label_visibility="collapsed"
                )

# --- FOOTER ---
st.markdown("---")
st.caption("🧠 Powered by all-mpnet-base-v2 embeddings + Groq Llama 3.3 | Weighted Required/Preferred scoring")
st.caption("💡 Auto-cleans job descriptions • Smaller chunks • Cross-domain compatible")