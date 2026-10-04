from matching.keyword_matcher import (
    MIN_RELIABLE_JOB_SKILLS,
    calculate_keyword_match,
    extract_keywords,
)

JOB_8 = "Requirements: python, sql, docker, kubernetes, aws, git, linux, terraform"


def skills(text):
    return extract_keywords(text)["technical_skills"]


# --- extract_keywords -------------------------------------------------------

def test_extracts_known_skills_case_insensitively():
    assert {"python", "sql", "docker"} <= skills("Built APIs in PYTHON with Docker and SQL.")


def test_unknown_words_are_not_skills():
    assert skills("I enjoy hiking, cooking and travelling.") == set()


def test_plural_variants_match():
    assert "docker" in skills("Deployed Dockers across clusters")


def test_single_word_skills_need_word_boundaries():
    # "r" and "go" are skills; they must not match inside other words.
    found = skills("Strong research and good programming ability")
    assert "r" not in found and "go" not in found


def test_extracts_education_and_experience():
    kw = extract_keywords("Master's degree. 5 years experience in backend work.")
    assert kw["education"]
    assert kw["experience_level"]


# --- calculate_keyword_match ------------------------------------------------

def match(resume_text, job_text=JOB_8):
    return calculate_keyword_match(extract_keywords(resume_text), extract_keywords(job_text))


def test_matched_and_missing_are_disjoint_and_cover_job_skills():
    r = match("python sql docker")
    assert set(r["matched_skills"]) == {"python", "sql", "docker"}
    assert set(r["missing_skills"]) == skills(JOB_8) - {"python", "sql", "docker"}
    assert not set(r["matched_skills"]) & set(r["missing_skills"])


def test_full_match_scores_higher_than_partial_than_none():
    full = match(JOB_8)["technical_score"]
    partial = match("python sql docker git")["technical_score"]
    none = match("I enjoy hiking and cooking")["technical_score"]
    assert full > partial > none


def test_scores_stay_in_range():
    for resume in (JOB_8, "python", "nothing relevant here"):
        r = match(resume)
        for key in ("technical_score", "education_score", "experience_score", "overall_keyword_score"):
            assert 0.0 <= r[key] <= 1.0, key


def test_many_missing_required_skills_are_penalised():
    # Same job; resume with <=3 missing vs >3 missing required skills.
    # Heuristic split makes ~70% of the 8 skills "required" (5), so matching
    # 4 of 5 required leaves 1 missing, while matching 1 leaves 4 missing.
    few_missing = match("python sql docker git linux")
    many_missing = match("python")
    assert len(many_missing["missing_required"]) > 3
    assert many_missing["technical_score"] < few_missing["technical_score"] * 0.85


def test_skill_split_is_deterministic():
    results = {tuple(match("python sql")["missing_required"]) for _ in range(5)}
    assert len(results) == 1


# --- short job description handling ----------------------------------------

SHORT_JOB = "We need cybersecurity, networking and security expertise."


def test_short_job_is_flagged_and_confidence_scaled():
    r = match("cybersecurity networking security", SHORT_JOB)
    assert r["total_job_skills"] < MIN_RELIABLE_JOB_SKILLS
    assert r["job_description_too_short"] is True
    assert r["keyword_confidence"] < 1.0


def test_long_job_is_not_flagged():
    r = match(JOB_8)
    assert r["job_description_too_short"] is False
    assert r["keyword_confidence"] == 1.0


def test_perfect_match_on_short_job_cannot_reach_full_technical_score():
    r = match("cybersecurity networking security", SHORT_JOB)
    assert r["matched_skills"] and not r["missing_skills"]
    assert r["technical_score"] <= r["total_job_skills"] / MIN_RELIABLE_JOB_SKILLS + 1e-9


def test_perfect_match_on_long_job_is_not_capped():
    assert match(JOB_8)["technical_score"] > 0.99
