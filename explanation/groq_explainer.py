"""
Free RAG explanations using Groq API (Llama 3.3 70B).
100% free with 14,400 requests/day limit.
"""

import os
from groq import Groq


def generate_match_explanation_groq(
    resume_chunks: list,
    job_description: str,
    score_breakdown: dict
) -> dict:
    """
    Generate AI explanation using Groq (free & fast).
    
    Args:
        resume_chunks: Top matching resume sections
        job_description: Job description text
        score_breakdown: Hybrid scoring results
    
    Returns:
        dict: {
            "explanation": str,
            "strengths": list,
            "gaps": list,
            "suggestions": list
        }
    """
    
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {
            "explanation": "⚠️ Groq API key not set. Add GROQ_API_KEY to your .env file.",
            "strengths": ["Semantic match found in resume"],
            "gaps": ["Unable to generate detailed analysis without API key"],
            "suggestions": ["Get free API key at https://console.groq.com"]
        }
    
    # Build context from top chunks
    chunks_text = "\n\n---\n\n".join(resume_chunks[:3])
    
    # Extract scores
    hybrid_score = score_breakdown.get("hybrid_score", 0)
    matched_skills = score_breakdown.get("matched_skills", [])
    missing_skills = score_breakdown.get("missing_skills", [])
    
    # Create prompt
    prompt = f"""You are an expert technical recruiter analyzing a resume-job match.

JOB REQUIREMENTS:
{job_description[:1000]}

TOP MATCHING RESUME SECTIONS:
{chunks_text}

MATCH DATA:
- Overall Score: {hybrid_score:.1%}
- Matched Skills: {', '.join(matched_skills[:10]) if matched_skills else 'None found'}
- Missing Skills: {', '.join(missing_skills[:10]) if missing_skills else 'None'}

Provide a concise analysis in this EXACT format:

EXPLANATION:
[2-3 sentences explaining why this candidate matches or doesn't match]

STRENGTHS:
- [Key strength 1]
- [Key strength 2]
- [Key strength 3]

GAPS:
- [Gap 1]
- [Gap 2]

SUGGESTIONS:
- [Actionable suggestion 1]
- [Actionable suggestion 2]

Keep it professional, specific, and actionable."""

    try:
        # Initialize Groq client
        client = Groq(api_key=api_key)
        
        # Make API call
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Fast and free
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # More focused responses
            max_tokens=500
        )
        
        # Parse response
        raw_text = response.choices[0].message.content
        
        # Simple parsing
        explanation = ""
        strengths = []
        gaps = []
        suggestions = []
        
        current_section = None
        
        for line in raw_text.split('\n'):
            line = line.strip()
            
            if line.startswith("EXPLANATION:"):
                current_section = "explanation"
                explanation = line.replace("EXPLANATION:", "").strip()
            elif line.startswith("STRENGTHS:"):
                current_section = "strengths"
            elif line.startswith("GAPS:"):
                current_section = "gaps"
            elif line.startswith("SUGGESTIONS:"):
                current_section = "suggestions"
            elif line.startswith("-") or line.startswith("•"):
                item = line.lstrip("-•").strip()
                if current_section == "strengths":
                    strengths.append(item)
                elif current_section == "gaps":
                    gaps.append(item)
                elif current_section == "suggestions":
                    suggestions.append(item)
            elif current_section == "explanation" and line:
                explanation += " " + line
        
        return {
            "explanation": explanation or raw_text[:200],
            "strengths": strengths[:3],
            "gaps": gaps[:2],
            "suggestions": suggestions[:3]
        }
        
    except Exception as e:
        return {
            "explanation": f"Error generating explanation: {str(e)}",
            "strengths": ["Technical skills present in resume"],
            "gaps": ["Some required skills may be missing"],
            "suggestions": ["Review job requirements carefully"]
        }


def generate_simple_explanation_fallback(score_breakdown: dict) -> dict:
    """
    Fallback explanation without API (when no key available).
    """
    matched = score_breakdown.get("matched_skills", [])
    missing = score_breakdown.get("missing_skills", [])
    score = score_breakdown.get("hybrid_score", 0)
    
    return {
        "explanation": f"This candidate has a {score:.1%} match based on semantic analysis and keyword matching.",
        "strengths": [
            f"Matches {len(matched)} required skills" if matched else "Some relevant experience found",
            "Resume structure is clear and readable",
            "Technical background is relevant"
        ],
        "gaps": [
            f"Missing {len(missing)} key skills: {', '.join(missing[:5])}" if missing else "Some skills need verification",
            "Consider adding more specific technical details"
        ],
        "suggestions": [
            "Highlight specific tools and technologies used",
            "Quantify achievements with numbers and metrics",
            "Add relevant certifications if available"
        ]
    }