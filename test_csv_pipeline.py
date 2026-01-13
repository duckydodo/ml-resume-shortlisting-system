import pandas as pd

from src.anonymizer import anonymize_text
from src.matcher import compute_similarity, fit_jd_vectorizer
from src.scorer import compute_final_score
from src.feature_extraction import (
    extract_jd_skills,
    extract_resume_features,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

CSV_PATH = "data/Resume.csv"
JOB_DESCRIPTION_PATH = "data/job_descriptions/sample_jd.txt"
OUTPUT_PATH = "output/shortlisted_candidates.csv"
TOP_N = 10

MAX_RESUME_CHARS = 3000  # truncate long resumes for speed


# --------------------------------------------------
# Load inputs
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)

with open(JOB_DESCRIPTION_PATH, "r", encoding="utf-8") as f:
    job_description = f.read().lower()


# --------------------------------------------------
# Prepare job description features
# --------------------------------------------------

fit_jd_vectorizer(job_description)
jd_skills = extract_jd_skills(job_description)


# --------------------------------------------------
# Process resumes
# --------------------------------------------------

results = []

for _, row in df.iterrows():
    candidate_id = row["ID"]

    raw_resume = str(row["Resume_str"])
    resume_text = anonymize_text(raw_resume).lower()
    resume_text = resume_text[:MAX_RESUME_CHARS]

    resume_features = extract_resume_features(
        resume_text=resume_text,
        jd_skills=jd_skills,
    )

    similarity = compute_similarity(resume_text, job_description)

    final_score = compute_final_score(
        similarity_score=similarity,
        skill_match_ratio=resume_features["skill_match_ratio"],
        experience_years=resume_features["experience_years"],
    )

    results.append({
        "candidate_id": candidate_id,
        "score": round(final_score, 3),
        "similarity": round(similarity, 3),
        "matched_skills": resume_features["matched_skills"],
        "missing_skills": resume_features["missing_skills"],
    })


# --------------------------------------------------
# Output results
# --------------------------------------------------

results_df = (
    pd.DataFrame(results)
      .sort_values(by="score", ascending=False)
)

results_df.to_csv(OUTPUT_PATH, index=False)

print(results_df.head(TOP_N))
