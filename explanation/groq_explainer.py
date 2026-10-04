"""
Match explanations generated with the Groq API.

The model is not hardcoded: it is resolved from Groq's live model list
(GET /openai/v1/models) so the app doesn't break when Groq retires a model.
If the list can't be fetched, a short hardcoded fallback list is tried in order.
Raw API errors are logged server-side and never shown to the user.
"""

import logging
import os
import re
import threading
import time

logger = logging.getLogger(__name__)

# Used only when the live model list can't be fetched or has no usable chat model.
FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
]

# Non-chat / specialised models returned by the models endpoint.
_EXCLUDED = re.compile(
    r"whisper|tts|playai|orpheus|guard|safeguard|compound|embed|moderation|vision-preview",
    re.IGNORECASE,
)

# Preference order: fast general-purpose instruct models first (what
# llama-3.3-70b-versatile was), reasoning models later because they spend the
# token budget on hidden reasoning. Anything else usable is ranked after these.
_PREFERRED = [
    r"llama-3\.3-70b",
    r"llama-4-maverick",
    r"llama-4-scout",
    r"llama-3\.1-70b",
    r"llama-3\.1-8b",
    r"gpt-oss-120b",
    r"gpt-oss-20b",
]

MAX_CANDIDATES = 3
CACHE_TTL_OK = 6 * 3600     # re-check the live list a few times a day
CACHE_TTL_FAILED = 5 * 60   # retry sooner if the models call itself failed

FRIENDLY_ERROR = (
    "Couldn't generate an AI explanation right now. The match score and skill "
    "breakdown above are still accurate."
)

_cache_lock = threading.Lock()
_cache = {"models": None, "expires": 0.0}


def rank_models(model_ids):
    """Filter a live model list to usable chat models and order by preference."""
    usable = sorted({m for m in model_ids if not _EXCLUDED.search(m)})

    def rank(model_id):
        for i, pattern in enumerate(_PREFERRED):
            if re.search(pattern, model_id):
                return i
        return len(_PREFERRED)

    return sorted(usable, key=rank)  # stable: ties stay alphabetical


def get_candidate_models(client):
    """
    Models to try, best first. Cached in memory so the models endpoint is not
    called on every explanation request.
    """
    with _cache_lock:
        if _cache["models"] is not None and time.monotonic() < _cache["expires"]:
            return list(_cache["models"])

        try:
            live = [
                m.id for m in client.models.list().data
                if getattr(m, "active", True) is not False
            ]
            ranked = rank_models(live)[:MAX_CANDIDATES]
            if not ranked:
                raise ValueError("models endpoint returned no usable chat models")
            models, ttl = ranked, CACHE_TTL_OK
            logger.info("Groq models resolved from live list: %s", models)
        except Exception as e:
            logger.warning("Could not fetch Groq model list (%s); using fallback models", e)
            models, ttl = list(FALLBACK_MODELS), CACHE_TTL_FAILED

        _cache["models"] = models
        _cache["expires"] = time.monotonic() + ttl
        return list(models)


def reset_model_cache():
    with _cache_lock:
        _cache["models"] = None
        _cache["expires"] = 0.0


def _make_client(api_key):
    from groq import Groq  # imported lazily so tests don't need the package
    return Groq(api_key=api_key, timeout=20)


def _is_model_error(error):
    text = str(error).lower()
    return getattr(error, "status_code", None) == 404 or "model_not_found" in text


def generate_match_explanation_groq(
    resume_chunks: list,
    job_description: str,
    score_breakdown: dict,
    client=None
) -> dict:
    """
    Generate AI explanation using Groq (free & fast).
    
    Args:
        resume_chunks: Top matching resume sections
        job_description: Job description text
        score_breakdown: Hybrid scoring results
        client: Optional Groq-compatible client (for tests)
    
    Returns:
        dict: {
            "explanation": str,
            "strengths": list,
            "gaps": list,
            "suggestions": list,
            "error": True only if generation failed (explanation is then a
                     short user-safe message and the lists are empty)
        }
    """
    
    api_key = os.getenv("GROQ_API_KEY")
    if client is None and not api_key:
        return generate_simple_explanation_fallback(score_breakdown)
    
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
        client = client or _make_client(api_key)
        candidates = get_candidate_models(client)
    except Exception as e:
        logger.error("Groq setup failed: %s", e)
        return _failure_result()

    raw_text = None
    for model in candidates:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # More focused responses
                max_tokens=500
            )
            raw_text = response.choices[0].message.content
            break
        except Exception as e:
            logger.error("Groq explanation failed with model %s: %s", model, e)
            if _is_model_error(e):
                reset_model_cache()  # re-resolve from the live list next time
            elif getattr(e, "status_code", None) in (401, 403):
                break  # a bad key won't be fixed by trying another model
    
    if not raw_text:
        return _failure_result()

    try:
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
            "explanation": explanation.strip() or raw_text[:200],
            "strengths": strengths[:3],
            "gaps": gaps[:2],
            "suggestions": suggestions[:3]
        }
        
    except Exception as e:
        logger.error("Could not parse Groq response: %s", e)
        return _failure_result()


def _failure_result() -> dict:
    return {
        "explanation": FRIENDLY_ERROR,
        "strengths": [],
        "gaps": [],
        "suggestions": [],
        "error": True
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