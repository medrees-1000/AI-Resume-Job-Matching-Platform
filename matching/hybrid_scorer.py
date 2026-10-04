"""
Hybrid scoring system combining semantic similarity and keyword matching.
Produces realistic scores similar to professional ATS systems.
"""

from typing import Dict, List


def calculate_hybrid_score(
    semantic_score: float,
    keyword_match_results: Dict,
    top_chunks: List[Dict]
) -> Dict:
    """
    Combine semantic similarity and keyword matching for final score.
    Uses boosted semantic scoring for better alignment with human judgment.
    
    Args:
        semantic_score: Cosine similarity score (0-1)
        keyword_match_results: Output from keyword_matcher
        top_chunks: Top matching resume sections
    
    Returns:
        dict: Complete scoring breakdown
    """
    
    # Extract individual scores
    technical_score = keyword_match_results.get("technical_score", 0.5)
    education_score = keyword_match_results.get("education_score", 0.5)
    experience_score = keyword_match_results.get("experience_score", 0.5)
    
    # Boost semantic score (it's usually too conservative)
    # If any chunk scored above 0.3, it's likely a relevant match
    boosted_semantic = min(semantic_score * 1.8, 1.0)  # 80% boost, capped at 100%
    
    # Weighted hybrid formula
    # 40% keyword matching (technical skills) - most important
    # 30% semantic similarity (overall fit)
    # 20% experience match
    # 10% education match
    
    hybrid_score = (
        0.40 * technical_score +
        0.30 * boosted_semantic +
        0.20 * experience_score +
        0.10 * education_score
    )
    
    # Apply cross-domain boost
    # If semantic score is decent but keyword score is low,
    # candidate might be transitioning roles (e.g., SWE → DS)
    if boosted_semantic > 0.4 and technical_score < 0.5:
        hybrid_score += 0.05  # 5% boost for potential cross-domain fit
    
    # Determine match category
    if hybrid_score >= 0.70:
        match_category = "Excellent Match"
        match_icon = "🔥"
        recommendation = "Strong candidate - Recommend immediate interview"
    elif hybrid_score >= 0.55:
        match_category = "Good Match"
        match_icon = "✅"
        recommendation = "Solid candidate - Review in detail"
    elif hybrid_score >= 0.40:
        match_category = "Moderate Match"
        match_icon = "⚠️"
        recommendation = "Some gaps exist - Consider with reservations"
    else:
        match_category = "Low Match"
        match_icon = "❌"
        recommendation = "Significant gaps - May not be suitable"
    
    # Calculate section scores (for visualization)
    section_scores = {
        "Technical Skills": technical_score,
        "Overall Fit": boosted_semantic,
        "Experience Level": experience_score,
        "Education": education_score
    }
    
    return {
        "hybrid_score": hybrid_score,
        "semantic_score": boosted_semantic,
        "raw_semantic_score": semantic_score,
        "technical_score": technical_score,
        "education_score": education_score,
        "experience_score": experience_score,
        "match_category": match_category,
        "match_icon": match_icon,
        "recommendation": recommendation,
        "section_scores": section_scores,
        "matched_skills": keyword_match_results.get("matched_skills", []),
        "missing_skills": keyword_match_results.get("missing_skills", []),
        "missing_required": keyword_match_results.get("missing_required", []),
        "missing_preferred": keyword_match_results.get("missing_preferred", []),
        "job_description_too_short": keyword_match_results.get("job_description_too_short", False)
    }


def generate_score_explanation(score_breakdown: Dict) -> str:
    """
    Generate human-readable explanation of the score.
    """
    hybrid = score_breakdown["hybrid_score"]
    technical = score_breakdown["technical_score"]
    semantic = score_breakdown["semantic_score"]
    
    matched_count = len(score_breakdown["matched_skills"])
    missing_count = len(score_breakdown["missing_skills"])
    
    explanation = f"""
**Overall Match: {hybrid:.1%}** - {score_breakdown['match_category']}

**What This Means:**
- Found {matched_count} matching technical skills
- {missing_count} required skills are missing
- Semantic fit score: {semantic:.1%}
- Technical skills coverage: {technical:.1%}

**Recommendation:** {score_breakdown['recommendation']}
    """.strip()
    
    return explanation