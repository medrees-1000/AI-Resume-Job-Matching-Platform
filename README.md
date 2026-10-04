# AI Resume-Job Matching Platform

## Project Overview

A Streamlit app that scores how well a resume PDF matches a job description, using **Python**, **sentence-transformer embeddings** and **LLM-generated explanations**.

A hybrid score combines semantic similarity (all-mpnet-base-v2 embeddings) with weighted keyword analysis (required vs. preferred skills), and an optional Groq (Llama 3.3) call writes a short human-readable explanation of the result.

### What is and isn't validated

- **Tested:** unit tests cover the keyword matcher, the hybrid scorer and the PDF text extraction (`pytest`). See [Testing and evaluation](#testing-and-evaluation).
- **Reproducible evaluation:** `tests/evaluate_matching.py` runs the keyword matcher over every sample resume × job description and writes the scores to `tests/results/`.
- **Not validated:** there is no labelled dataset in this repo, so no accuracy figure is claimed. The semantic-boost factor (1.8×) and the score weights are hand-tuned heuristics, not fitted values. Earlier versions of this README cited a comparison against a commercial ATS; that comparison cannot be reproduced from this repo and has been removed.

---

## Business Problem

Recruiters and hiring teams struggle with:

- **Keyword-only ATS** that miss qualified candidates with different terminology
- **Black-box scoring** with no explanation of why candidates match or don't
- **Equal weighting** of required vs. preferred skills leading to unfair assessments
- **Slow processing** taking minutes per resume in traditional systems
- **Poor handling** of career transitions (e.g., Software Engineer → Data Scientist)

This project is an attempt at those problems: semantic matching, required/preferred weighting, and explanations of why a score came out as it did.

---

## Tech Stack

Exact versions are pinned in [requirements.txt](requirements.txt).

* **Python 3.10+**
* **sentence-transformers** (all-mpnet-base-v2, 768-dim embeddings), **PyTorch**, **transformers**, **scikit-learn** (cosine similarity)
* **pypdf** – PDF text extraction
* **Groq API (Llama 3.3 70B)** – optional match explanations
* **Streamlit** + **Plotly** – web UI and gauge charts
* **pytest** – tests (see below)

---

## Project Architecture
```
AI-Resume-Job-Matching-Platform/
│
├── app/
│   └── streamlit_app.py          # Streamlit web application
│
├── ingestion/
│   ├── pdf_parser.py             # PDF → text (detects/repairs letter-spaced extraction)
│   ├── chunking.py               # Word-based chunking (200 words, 75 overlap)
│   ├── job_cleaner.py            # Strips company fluff, splits required/preferred
│   └── process_resume.py         # Resume/job → text, chunks, embeddings
│
├── matching/
│   ├── similarity.py             # Cosine similarity, top matching chunks
│   ├── keyword_matcher.py        # Skill extraction + required/preferred scoring
│   └── hybrid_scorer.py          # Weighted combination into the final score
│
├── explanation/
│   └── groq_explainer.py         # LLM explanation (Groq) + offline fallback
│
├── data/
│   ├── jobs/                     # 5 sample job descriptions
│   └── sample_resumes/           # 25 sample resume PDFs used for evaluation
│
├── tests/
│   ├── test_keyword_matcher.py   # pytest unit tests
│   ├── test_hybrid_scorer.py     # pytest unit tests
│   ├── test_pdf_parser.py        # pytest unit tests
│   ├── evaluate_matching.py      # standalone evaluation script (not a pytest test)
│   └── results/                  # evaluation output (CSV + JSON)
│
├── pytest.ini
├── .env.example                  # Environment template
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Data Pipeline Workflow
```
Resume PDF → Text Extraction → Chunking → Embeddings → Similarity Matching → Hybrid Scoring → AI Explanation
     ↓              ↓              ↓           ↓              ↓                    ↓                ↓
   pypdf      pdf_parser    chunking.py   all-mpnet    similarity.py      hybrid_scorer.py   groq_explainer.py
                              200 words     (768-dim)    + keyword_matcher   (4 components)    (Llama 3.3 70B)
                              75 overlap                 Required 80%
                                                         Preferred 20%
```

### Hybrid Scoring Algorithm

**Four-component weighted formula:**
```python
hybrid_score = (
    0.40 × technical_skill_score +    # Keyword matching (Required 80%, Preferred 20%)
    0.30 × boosted_semantic_score +   # Semantic similarity × 1.8 boost (capped at 1)
    0.20 × experience_match_score +   # Experience level alignment
    0.10 × education_match_score      # Education requirements
)

# Cross-domain boost for career transitions
if boosted_semantic_score > 0.4 and technical_score < 0.5:
    hybrid_score += 0.05  # +5% for transitioning roles (e.g., SWE → DS)
```

**Key Innovations:**
- **Semantic boost (1.8×)**: Raw cosine similarity for chunk-vs-job comparisons tends to be low, so it is scaled up and capped at 1.0. This factor is a hand-picked heuristic.
- **Required vs. Preferred weighting**: Required skills weighted 4× more than preferred (80/20 split).
- **Penalty system**: 15% reduction of the technical score when more than 3 required skills are missing.
- **Short-posting correction**: a job description with fewer than 5 recognized skills would let almost any resume match 100% of them, so the technical score is scaled by `skills_found / 5` and the UI warns that the posting is too short to score reliably.
- **Cross-domain intelligence**: +5% boost when semantic fit is strong but keyword overlap is low.

---

## Notes on Design Choices

- **Semantic boost:** the 1.8× multiplier and the score thresholds were tuned by hand on a few examples. They are not derived from labelled data.
- **Embedding model:** all-mpnet-base-v2 (768-dim) is used instead of all-MiniLM-L6-v2 (384-dim) for higher general-purpose embedding quality; no benchmark of the two on this task is included in the repo.
- **PDF robustness:** some PDFs make pypdf emit letter-spaced text ("D a t a  A n a l y s t"), which silently zeroes keyword matching. `ingestion/pdf_parser.py` detects this, tries to re-join the letters, and otherwise raises `UnreliableExtractionError` so the UI can tell the user instead of showing a misleadingly low score.

---

## Matching Pipeline

### 1. Job Description Cleaning
**Automated preprocessing removes noise:**
- Company overviews, mission statements, "about us" sections
- Benefits packages, perks, salary ranges
- Application instructions, contact details, EEO statements

**Retains only relevant content:**
- Job responsibilities and key duties
- Required qualifications and must-have skills
- Preferred qualifications and nice-to-have skills
- Technical requirements and tools


---

### 2. Semantic Matching
**Transformer-based deep understanding:**
- **Model:** all-mpnet-base-v2 (768-dimensional embeddings)
- **Chunking strategy:** 200 words with 75-word overlap for context preservation
- **Similarity metric:** Cosine similarity between resume chunks and job description
- **Output:** Top 5 most relevant resume sections with similarity scores

**Why this matters:** Understands "machine learning" = "ML" = "predictive modeling" semantically, unlike keyword systems.

---

### 3. Keyword Matching
**Structured skill extraction:**
- **Skill database:** ~290 technical skills, tools, and frameworks (Python, SQL, AWS, Docker, etc.)
- **Separation logic:** Automatically distinguishes required vs. preferred qualifications
- **Weighted scoring:** 80% weight on required skills, 20% on preferred
- **Gap analysis:** Identifies exact missing skills for candidate feedback

**Example output:**
```
Matched: python, sql, pandas, machine learning, git
Missing Required: docker, airflow
Missing Preferred: aws, kubernetes
```

---

### 4. AI Explanation (optional)
**Natural language insights:**
- **Model:** Groq Llama 3.3 70B (fast, free inference)
- **Generates:**
  - Match reasoning (2-3 sentences)
  - Top 3 key strengths
  - Top 2 skill gaps
  - 2-3 actionable improvement suggestions

Requires a Groq API key; without one the app uses a simple rule-based fallback explanation.

---

## How to Run This Project

### Prerequisites
- Python 3.10 or higher
- pip package manager
- (Optional) Groq API key for AI explanations

### Installation Steps

**1. Clone the repository**
```bash
git clone https://github.com/medrees-1000/AI-Resume-Job-Matching-Platform.git
cd AI-Resume-Job-Matching-Platform
```

**2. Create virtual environment**
```bash
python3 -m venv env

# Activate (Mac/Linux)
source env/bin/activate

# Activate (Windows)
.\env\Scripts\activate.ps1
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

This will install all required packages including:
- PyTorch and transformers for embeddings
- Streamlit for the web interface
- Groq client for AI explanations
- All data processing libraries

**4. Set up API key (optional)**
```bash
# Copy template
cp .env.example .env

# Edit .env and add your Groq API key
# GROQ_API_KEY=your_actual_key_here
```

Get free API key at: **https://console.groq.com** (no credit card required)

**5. Run the application**
```bash
cd app
streamlit run streamlit_app.py
```

Application will open at **http://localhost:8501**

---

## Testing and Evaluation

Install dependencies first (`pip install -r requirements.txt`, plus `pip install pytest`), then run from the repo root.

**Unit tests**
```bash
pytest
```
Covers `extract_keywords` / `calculate_keyword_match` (including the short-job-description cap and the missing-required-skills penalty), `calculate_hybrid_score` (weighting, category thresholds, cross-domain boost) and the PDF letter-spacing detection/repair. They use small hand-written strings, not the sample resumes, and don't need the embedding model.

**Evaluation script**
```bash
python tests/evaluate_matching.py
```
Runs the keyword matcher on every PDF in `data/sample_resumes/` against every job in `data/jobs/` and writes `tests/results/keyword_matching_results.csv` and `.json` (resume, job, score, matched/missing skills). It prints the top 5 resumes per job. It evaluates keyword matching only (deterministic, no model download), so scores are not the full hybrid scores shown in the app. There are no ground-truth labels, so it reports scores and rankings rather than an accuracy percentage.

---

### Troubleshooting

**"ModuleNotFoundError: No module named 'groq'"**
```bash
pip install groq
```

**"Model not found" error**
```bash
# First run downloads the embedding model (~90MB)
# Wait for download to complete
```

**"API key not set" warning**
```bash
# The app works without an API key; only the AI explanation requires it
```

**"This PDF's text could not be read reliably"**
Re-export the resume from the original document (e.g. "Save as PDF") and upload it again.

---

## Limitations

- Skill matching is a fixed vocabulary lookup (`TECH_SKILLS`), so skills outside that list are invisible to the keyword score.
- Scanned/image-only PDFs have no extractable text and can't be processed (no OCR).
- Scores are heuristic; treat them as a screening aid, not a hiring decision.

---

## License

This project is licensed under the MIT License.
