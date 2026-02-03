"""
Enhanced Keyword extraction and matching for resume-job comparison.
Extracts technical skills, tools, and requirements with better coverage.
"""

import re
from typing import Dict, List, Set

# COMPREHENSIVE Technical Skills Database
TECH_SKILLS = {
    # Programming Languages
    "python", "java", "javascript", "c++", "c#", "r", "sql", "scala", "go", "rust",
    "typescript", "php", "swift", "kotlin", "ruby", "matlab", "julia", "perl", "bash",
    
    # Data Science & ML Core
    "machine learning", "ml", "deep learning", "nlp", "natural language processing",
    "computer vision", "data science", "data analysis", "statistics", "statistical analysis",
    "pandas", "numpy", "scikit-learn", "sklearn", "tensorflow", "pytorch", "keras", 
    "lightgbm", "xgboost", "catboost", "scikit learn",
    
    # AI & Generative AI
    "generative ai", "genai", "gen ai", "llm", "llms", "large language model", 
    "large language models", "langchain", "hugging face", "huggingface", "openai",
    "vertex ai", "prompt engineering", "ai agents", "ai agent", "copilot", "copilots",
    "chatgpt", "gpt", "gpt-4", "gpt-3", "claude", "bedrock", "sagemaker",
    "fine-tuning", "rag", "retrieval augmented generation", "embedding", "embeddings",
    "transformer", "transformers", "bert", "gpt", "attention mechanism",
    
    # Big Data & Cloud
    "aws", "amazon web services", "azure", "microsoft azure", "gcp", "google cloud", 
    "google cloud platform", "spark", "pyspark", "hadoop", "kafka", "airflow", 
    "apache airflow", "databricks", "snowflake", "bigquery", "redshift", "s3",
    "lambda", "ec2", "emr", "glue", "kinesis", "cloud computing",
    
    # Databases
    "mysql", "postgresql", "postgres", "mongodb", "redis", "cassandra", "dynamodb", 
    "oracle", "sql server", "mariadb", "sqlite", "neo4j", "elasticsearch", "nosql",
    "database", "databases", "rdbms", "data modeling",
    
    # Tools & Frameworks - Web/API
    "git", "github", "gitlab", "docker", "kubernetes", "k8s", "jenkins", "terraform",
    "ansible", "flask", "django", "fastapi", "react", "reactjs", "node.js", "nodejs",
    "spring boot", "express", "vue", "angular", "rest api", "restful", "graphql",
    "microservices", "api", "apis",
    
    # BI & Visualization
    "tableau", "power bi", "powerbi", "looker", "qlik", "excel", "powerpoint",
    "google sheets", "data visualization", "dataviz", "dashboards", "reporting",
    "matplotlib", "seaborn", "plotly", "d3.js", "d3",
    
    # Data Engineering & ETL
    "etl", "elt", "data pipeline", "data pipelines", "data warehouse", "data warehousing",
    "data lake", "data lakes", "dbt", "fivetran", "stitch", "talend", "informatica",
    "alteryx", "dataflow", "beam", "apache beam", "data integration",
    
    # MLOps & DevOps
    "mlops", "devops", "ci/cd", "continuous integration", "continuous deployment",
    "mlflow", "kubeflow", "github actions", "circleci", "travis ci", "gitlab ci",
    "monitoring", "logging", "observability", "prometheus", "grafana",
    
    # Testing & Quality
    "pytest", "unittest", "testing", "test automation", "selenium", "junit",
    "integration testing", "unit testing", "tdd", "test driven development",
    
    # Collaboration & Project Management
    "jira", "confluence", "slack", "teams", "notion", "asana", "trello",
    "agile", "scrum", "kanban", "sprint", "git", "version control",
    
    # Business & Productivity
    "powerpoint", "excel", "word", "microsoft office", "google workspace",
    "presentations", "documentation", "technical writing",
    
    # Automation & RPA
    "rpa", "robotic process automation", "uipath", "automation anywhere",
    "blue prism", "power automate", "automation", "workflow automation",
    
    # Security & Compliance
    "security", "cybersecurity", "encryption", "authentication", "authorization",
    "oauth", "jwt", "ssl", "tls", "compliance", "gdpr", "hipaa", "soc2",
    
    # Networking & Systems
    "networking", "tcp/ip", "http", "https", "dns", "load balancing",
    "linux", "unix", "windows", "macos", "system administration",
    
    # Specific Tools & Platforms
    "jupyter", "jupyter notebook", "colab", "google colab", "vscode", "pycharm",
    "intellij", "sublime", "vim", "emacs", "postman", "swagger",
    
    # MCPs and Advanced Topics
    "mcp", "mcps", "model context protocol", "langchain", "llamaindex",
    "semantic search", "vector database", "vector databases", "pinecone", "weaviate",
    "chromadb", "faiss", "annoy",
    
    # Methodologies & Concepts
    "agile", "scrum", "waterfall", "design patterns", "system design",
    "distributed systems", "scalability", "performance optimization",
    "data structures", "algorithms", "object oriented programming", "oop",
    "functional programming", "async", "asynchronous programming"
}

# Education keywords
EDUCATION_LEVELS = {
    "phd", "ph.d", "doctorate", "doctoral", "master", "masters", "master's", "msc", "m.sc",
    "bachelor", "bachelors", "bachelor's", "bs", "b.s", "ba", "b.a", "ms", "m.s", 
    "mba", "ma", "m.a", "undergraduate", "graduate", "postgraduate", "associate",
    "degree", "certification", "certificate", "certified"
}

# Experience levels
EXPERIENCE_KEYWORDS = {
    "intern", "internship", "entry level", "entry-level", "junior", "mid level",
    "mid-level", "senior", "sr", "lead", "principal", "staff", "architect",
    "years experience", "years of experience", "yoe"
}


def normalize_skill(skill: str) -> str:
    """Normalize skill for better matching (lowercase, remove special chars)."""
    return re.sub(r'[^\w\s]', '', skill.lower()).strip()


def extract_keywords(text: str) -> Dict[str, Set[str]]:
    """
    Extract structured keywords from text with improved matching.
    
    Returns:
        dict: {
            "technical_skills": set of tech skills found,
            "education": set of education levels,
            "experience_level": set of experience indicators
        }
    """
    text_lower = text.lower()
    
    # Find technical skills - with better matching
    found_skills = set()
    for skill in TECH_SKILLS:
        # Create flexible pattern that handles plurals and variations
        skill_pattern = re.escape(skill)
        
        # Handle common variations
        if skill.endswith('s'):
            # If skill ends in 's', also match without 's'
            base_skill = skill[:-1]
            skill_pattern = f"({skill_pattern}|{re.escape(base_skill)})"
        else:
            # If skill doesn't end in 's', also match with 's'
            skill_pattern = f"({skill_pattern}|{skill_pattern}s?)"
        
        # Use word boundaries for single words, looser matching for phrases
        if ' ' in skill:
            # Multi-word phrase - use looser matching
            pattern = skill_pattern
        else:
            # Single word - use word boundaries
            pattern = r'\b' + skill_pattern + r'\b'
        
        if re.search(pattern, text_lower):
            found_skills.add(skill)
    
    # Find education levels
    found_education = set()
    for edu in EDUCATION_LEVELS:
        pattern = r'\b' + re.escape(edu) + r'\b'
        if re.search(pattern, text_lower):
            found_education.add(edu)
    
    # Find experience level
    found_experience = set()
    for exp in EXPERIENCE_KEYWORDS:
        # For phrases like "years experience", use exact match
        pattern = re.escape(exp)
        if re.search(pattern, text_lower):
            found_experience.add(exp)
    
    return {
        "technical_skills": found_skills,
        "education": found_education,
        "experience_level": found_experience
    }


def calculate_keyword_match(resume_keywords: Dict, job_keywords: Dict, job_sections: Dict = None) -> Dict[str, float]:
    """
    Calculate keyword overlap between resume and job.
    Weights required skills more heavily than preferred.
    
    Args:
        resume_keywords: Skills found in resume
        job_keywords: Skills found in full job description
        job_sections: Optional dict with "required_skills" and "preferred_skills" text
    
    Returns:
        dict: Scoring breakdown with matched/missing skills
    """
    resume_skills = resume_keywords.get("technical_skills", set())
    job_skills = job_keywords.get("technical_skills", set())
    
    # Try to separate required vs preferred
    if job_sections and (job_sections.get("required_skills") or job_sections.get("preferred_skills")):
        # Extract keywords from separated sections
        required_text = job_sections.get("required_skills", "")
        preferred_text = job_sections.get("preferred_skills", "")
        
        required_keywords = extract_keywords(required_text)
        preferred_keywords = extract_keywords(preferred_text)
        
        required_skills = required_keywords.get("technical_skills", set())
        preferred_skills = preferred_keywords.get("technical_skills", set())
        
        # If both are empty, fall back to heuristic
        if not required_skills and not preferred_skills:
            # Use heuristic split
            all_job_skills = list(job_skills)
            split_point = int(len(all_job_skills) * 0.7)
            required_skills = set(all_job_skills[:split_point]) if split_point > 0 else job_skills
            preferred_skills = set(all_job_skills[split_point:]) if split_point > 0 else set()
    else:
        # Fallback: Assume 70% required, 30% preferred
        all_job_skills = list(job_skills)
        split_point = int(len(all_job_skills) * 0.7)
        required_skills = set(all_job_skills[:split_point]) if split_point > 0 else job_skills
        preferred_skills = set(all_job_skills[split_point:]) if split_point > 0 else set()
    
    # Calculate matches
    matched_required = resume_skills.intersection(required_skills)
    matched_preferred = resume_skills.intersection(preferred_skills)
    
    missing_required = required_skills - resume_skills
    missing_preferred = preferred_skills - resume_skills
    
    # Weighted scoring
    if required_skills:
        required_score = len(matched_required) / len(required_skills)
    else:
        required_score = 0.8  # Neutral if no requirements specified
    
    if preferred_skills:
        preferred_score = len(matched_preferred) / len(preferred_skills)
    else:
        preferred_score = 1.0  # Perfect if no preferences specified
    
    # Technical score: 80% required, 20% preferred
    technical_score = (0.80 * required_score) + (0.20 * preferred_score)
    
    # Combine all matched skills
    all_matched = list(matched_required.union(matched_preferred))
    all_missing = list(missing_required.union(missing_preferred))
    
    # Penalty for missing critical required skills
    if len(missing_required) > 3:
        technical_score *= 0.85  # 15% penalty for many missing required skills
    
    # Education match
    resume_edu = resume_keywords.get("education", set())
    job_edu = job_keywords.get("education", set())
    
    if resume_edu and job_edu:
        # Check for any overlap
        education_score = 1.0 if resume_edu.intersection(job_edu) else 0.5
    elif not job_edu:
        # No education requirements specified
        education_score = 1.0
    else:
        # Education required but not found in resume
        education_score = 0.5
    
    # Experience level match
    resume_exp = resume_keywords.get("experience_level", set())
    job_exp = job_keywords.get("experience_level", set())
    
    if resume_exp and job_exp:
        experience_score = 1.0 if resume_exp.intersection(job_exp) else 0.7
    else:
        experience_score = 0.8  # Neutral if unclear
    
    # Overall keyword score
    overall = (
        0.70 * technical_score +
        0.20 * education_score +
        0.10 * experience_score
    )
    
    return {
        "technical_score": technical_score,
        "required_skill_score": required_score,
        "preferred_skill_score": preferred_score,
        "education_score": education_score,
        "experience_score": experience_score,
        "overall_keyword_score": overall,
        "matched_skills": sorted(all_matched),
        "missing_skills": sorted(all_missing),
        "missing_required": sorted(list(missing_required)),
        "missing_preferred": sorted(list(missing_preferred)),
        # Debug info
        "total_resume_skills": len(resume_skills),
        "total_job_skills": len(job_skills),
        "total_required_skills": len(required_skills),
        "total_preferred_skills": len(preferred_skills)
    }


def get_improvement_suggestions(missing_skills: List[str], matched_skills: List[str]) -> List[str]:
    """
    Generate actionable improvement suggestions.
    """
    suggestions = []
    
    if len(missing_skills) > 0:
        # Prioritize most important missing skills
        top_missing = missing_skills[:5]
        suggestions.append(f"Add these key skills to your resume: {', '.join(top_missing)}")
    
    if len(matched_skills) < 5:
        suggestions.append("Expand your technical skills section with more specific tools and frameworks")
    
    # Specific skill suggestions
    missing_set = set(missing_skills)
    
    if "python" in missing_set:
        suggestions.append("Python is highly valued - add Python projects to your experience section")
    
    if any(skill in missing_set for skill in ["aws", "azure", "gcp", "google cloud"]):
        suggestions.append("Consider getting cloud platform experience (AWS/Azure/GCP)")
    
    if any(skill in missing_set for skill in ["docker", "kubernetes", "k8s"]):
        suggestions.append("Container technologies (Docker/Kubernetes) are in high demand")
    
    if any(skill in missing_set for skill in ["llm", "llms", "generative ai", "genai"]):
        suggestions.append("Generative AI skills are trending - consider adding LLM experience")
    
    if not suggestions:
        suggestions.append("Strong skill match! Consider highlighting achievements and quantifiable impact")
    
    return suggestions