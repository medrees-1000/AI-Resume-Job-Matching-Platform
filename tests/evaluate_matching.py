"""
Standalone evaluation: keyword-match every sample resume against every sample
job description and write the results table to tests/results/.

Run from the repo root:
    python tests/evaluate_matching.py

This evaluates the keyword matcher only (deterministic, no model download).
It does not compute the semantic score, so it does not reproduce the full
hybrid score shown in the app. There are no ground-truth labels in the repo,
so this reports scores and rankings, not an accuracy percentage.
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingestion.job_cleaner import extract_requirements_section
from ingestion.pdf_parser import UnreliableExtractionError, extract_text_from_pdf
from matching.keyword_matcher import calculate_keyword_match, extract_keywords

RESUME_DIR = ROOT / "data" / "sample_resumes"
JOB_DIR = ROOT / "data" / "jobs"
RESULTS_DIR = ROOT / "tests" / "results"
TOP_SKILLS = 8


def load_resumes():
    resumes, skipped = {}, {}
    for pdf in sorted(RESUME_DIR.glob("*.pdf")):
        try:
            text = extract_text_from_pdf(str(pdf))
        except UnreliableExtractionError as e:
            skipped[pdf.name] = f"unreliable extraction: {e}"
            continue
        if text is None:
            skipped[pdf.name] = "no readable text"
            continue
        resumes[pdf.name] = extract_keywords(text)
    return resumes, skipped


def load_jobs():
    jobs = {}
    for path in sorted(JOB_DIR.glob("*.txt")):
        sections = extract_requirements_section(path.read_text(encoding="utf-8"))
        jobs[path.stem] = (extract_keywords(sections["cleaned_text"]), sections)
    return jobs


def main():
    resumes, skipped = load_resumes()
    jobs = load_jobs()
    if not resumes or not jobs:
        sys.exit(f"Nothing to evaluate (resumes: {len(resumes)}, jobs: {len(jobs)})")

    rows = []
    for resume_name, resume_kw in resumes.items():
        for job_name, (job_kw, sections) in jobs.items():
            r = calculate_keyword_match(resume_kw, job_kw, sections)
            rows.append({
                "resume": resume_name,
                "job": job_name,
                "keyword_score": round(r["overall_keyword_score"], 4),
                "technical_score": round(r["technical_score"], 4),
                "job_skills_found": r["total_job_skills"],
                "job_too_short": r["job_description_too_short"],
                "matched_skills": r["matched_skills"][:TOP_SKILLS],
                "missing_skills": r["missing_skills"][:TOP_SKILLS],
            })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "keyword_matching_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row,
                             "matched_skills": "; ".join(row["matched_skills"]),
                             "missing_skills": "; ".join(row["missing_skills"])})
    with open(RESULTS_DIR / "keyword_matching_results.json", "w", encoding="utf-8") as f:
        json.dump({"skipped_resumes": skipped, "results": rows}, f, indent=2)

    print(f"{len(resumes)} resumes x {len(jobs)} jobs = {len(rows)} pairs")
    for name, reason in skipped.items():
        print(f"  skipped {name}: {reason}")
    for job_name in jobs:
        ranked = sorted((r for r in rows if r["job"] == job_name),
                        key=lambda r: r["keyword_score"], reverse=True)
        print(f"\n{job_name} ({ranked[0]['job_skills_found']} job skills) - top 5 resumes:")
        for r in ranked[:5]:
            print(f"  {r['keyword_score']:.2f}  {r['resume']}")
    print(f"\nWrote {RESULTS_DIR.relative_to(ROOT)}/keyword_matching_results.{{csv,json}}")


if __name__ == "__main__":
    main()
