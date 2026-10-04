import pytest

from matching.hybrid_scorer import calculate_hybrid_score, generate_score_explanation


def keyword_results(technical=0.5, education=1.0, experience=1.0, **extra):
    return {
        "technical_score": technical,
        "education_score": education,
        "experience_score": experience,
        "matched_skills": ["python"],
        "missing_skills": ["aws"],
        "missing_required": ["aws"],
        "missing_preferred": [],
        **extra,
    }


def score(semantic, **kw):
    return calculate_hybrid_score(semantic, keyword_results(**kw), top_chunks=[])


def test_weighted_formula():
    # 0.4*tech + 0.3*boosted_semantic + 0.2*exp + 0.1*edu (no cross-domain boost: tech >= 0.5)
    r = score(0.5, technical=0.8, education=0.6, experience=0.7)
    boosted = 0.5 * 1.8
    assert r["hybrid_score"] == pytest.approx(0.4 * 0.8 + 0.3 * boosted + 0.2 * 0.7 + 0.1 * 0.6)


def test_semantic_boost_is_capped_at_one():
    assert score(0.9)["semantic_score"] == 1.0
    assert score(0.9)["raw_semantic_score"] == 0.9


def test_strong_candidate_is_excellent():
    r = score(0.6, technical=1.0)
    assert r["match_category"] == "Excellent Match"


def test_weak_candidate_is_low():
    r = score(0.0, technical=0.0, education=0.0, experience=0.0)
    assert r["hybrid_score"] == 0
    assert r["match_category"] == "Low Match"


@pytest.mark.parametrize("technical, semantic, expected", [
    (0.9, 0.5, "Excellent Match"),
    (0.6, 0.3, "Good Match"),
    (0.5, 0.2, "Moderate Match"),
    (0.0, 0.0, "Low Match"),
])
def test_category_thresholds(technical, semantic, expected):
    assert score(semantic, technical=technical, education=0.5, experience=0.5)["match_category"] == expected


# --- cross-domain boost: boosted_semantic > 0.4 and technical < 0.5 ----------

def formula(semantic, technical, education=1.0, experience=1.0):
    boosted = min(semantic * 1.8, 1.0)
    return 0.4 * technical + 0.3 * boosted + 0.2 * experience + 0.1 * education


def test_cross_domain_boost_applies():
    r = score(0.3, technical=0.3)  # boosted semantic 0.54 > 0.4, technical < 0.5
    assert r["hybrid_score"] == pytest.approx(formula(0.3, 0.3) + 0.05)


def test_no_boost_when_technical_is_high_enough():
    r = score(0.3, technical=0.5)
    assert r["hybrid_score"] == pytest.approx(formula(0.3, 0.5))


def test_no_boost_when_semantic_is_low():
    r = score(0.2, technical=0.3)  # boosted semantic 0.36 <= 0.4
    assert r["hybrid_score"] == pytest.approx(formula(0.2, 0.3))


# --- keyword results are passed through (the missing-skills *penalty* itself
# lives in calculate_keyword_match; see test_keyword_matcher.py) ---------------

def test_skill_lists_are_passed_through():
    r = score(0.5)
    assert r["matched_skills"] == ["python"]
    assert r["missing_skills"] == ["aws"]
    assert r["missing_required"] == ["aws"]


def test_short_job_flag_is_passed_through():
    assert score(0.5, job_description_too_short=True)["job_description_too_short"] is True
    assert score(0.5)["job_description_too_short"] is False


def test_explanation_mentions_score_and_category():
    text = generate_score_explanation(score(0.5, technical=0.8))
    assert "Overall Match" in text and "Match" in text
