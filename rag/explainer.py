"""
RAG-based explanation generator using Claude API.
Includes free tier protection - stops after $5 credit exhausted.
"""

import os
from anthropic import Anthropic

# Free tier tracking
FREE_TIER_LIMIT = 5.0  # $5 USD
_total_cost = 0.0


class FreeAPILimitError(Exception):
    """Raised when free API credits are exhausted."""
    pass


def estimate_cost(input_tokens, output_tokens):
    """
    Estimate cost for Claude API call.
    Sonnet 4 pricing (as of Jan 2025):
    - Input: $3 / 1M tokens
    - Output: $15 / 1M tokens
    """
    input_cost = (input_tokens / 1_000_000) * 3.0
    output_cost = (output_tokens / 1_000_000) * 15.0
    return input_cost + output_cost


def check_free_tier():
    """Ensure we haven't exceeded free tier."""
    global _total_cost
    if _total_cost >= FREE_TIER_LIMIT:
        raise FreeAPILimitError(
            f"Free API limit (${FREE_TIER_LIMIT}) reached. "
            f"Total cost: ${_total_cost:.2f}. "
            "Please add credits to continue."
        )


def generate_match_explanation(resume_chunks, job_description, match_score):
    """
    Generate human-readable explanation of why candidate matches.
    
    Args:
        resume_chunks: list of top matching text chunks
        job_description: string of job description
        match_score: float (0-1) similarity score
    
    Returns:
        dict: {
            "explanation": str,
            "strengths": list of str,
            "gaps": list of str,
            "cost": float
        }
    """
    global _total_cost
    
    # Check if we've hit the limit
    check_free_tier()
    
    # Get API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {
            "explanation": "⚠️ Claude API key not set. Set ANTHROPIC_API_KEY environment variable.",
            "strengths": [],
            "gaps": [],
            "cost": 0.0
        }
    
    # Build prompt
    chunks_text = "\n\n---\n\n".join(resume_chunks[:3])  # Top 3 chunks only
    
    prompt = f"""Analyze this resume-job match:

JOB DESCRIPTION:
{job_description}

TOP MATCHING RESUME SECTIONS:
{chunks_text}

MATCH SCORE: {match_score:.1%}

Provide a concise analysis in this format:

EXPLANATION:
[2-3 sentences explaining why this candidate matches]

STRENGTHS:
- [Key strength 1]
- [Key strength 2]
- [Key strength 3]

SKILL GAPS:
- [Gap 1]
- [Gap 2]

Keep it professional and specific."""

    try:
        client = Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,  # Keep it concise to save costs
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Calculate cost
        usage = response.usage
        call_cost = estimate_cost(usage.input_tokens, usage.output_tokens)
        _total_cost += call_cost
        
        # Parse response
        raw_text = response.content[0].text
        
        # Simple parsing (you can make this more robust)
        explanation = ""
        strengths = []
        gaps = []
        
        sections = raw_text.split("\n\n")
        current_section = None
        
        for section in sections:
            if "EXPLANATION:" in section:
                explanation = section.replace("EXPLANATION:", "").strip()
            elif "STRENGTHS:" in section:
                current_section = "strengths"
            elif "SKILL GAPS:" in section or "GAPS:" in section:
                current_section = "gaps"
            elif section.strip().startswith("-"):
                item = section.strip().lstrip("- ")
                if current_section == "strengths":
                    strengths.append(item)
                elif current_section == "gaps":
                    gaps.append(item)
        
        return {
            "explanation": explanation or raw_text[:200],
            "strengths": strengths[:3],
            "gaps": gaps[:2],
            "cost": call_cost,
            "total_cost": _total_cost
        }
        
    except Exception as e:
        return {
            "explanation": f"Error generating explanation: {str(e)}",
            "strengths": [],
            "gaps": [],
            "cost": 0.0
        }


def generate_simple_explanation(resume_chunks, match_score):
    """
    Fallback explanation without API (when limit reached).
    """
    return {
        "explanation": f"This candidate has a {match_score:.1%} semantic match based on resume content.",
        "strengths": [
            "Relevant experience found in resume",
            "Skills align with job requirements",
            "Background matches role expectations"
        ],
        "gaps": [
            "Some specialized skills may need verification",
            "Additional details available in full resume"
        ],
        "cost": 0.0
    }


def get_remaining_credit():
    """Check how much free credit is left."""
    global _total_cost
    remaining = max(0, FREE_TIER_LIMIT - _total_cost)
    return {
        "used": _total_cost,
        "remaining": remaining,
        "limit": FREE_TIER_LIMIT,
        "percentage": (_total_cost / FREE_TIER_LIMIT) * 100
    }