import pandas as pd
from src.anonymizer import anonymize_text
from src.matcher import compute_similarity
from src.scorer import compute_final_score
from src.matcher import fit_jd_vectorizer
from src.feature_extraction import extract_jd_skills, extract_resume_features






# --------------------------------------------------
# CONFIG
# --------------------------------------------------

CSV_PATH = "data/Resume.csv"   # adjust if filename differs
JOB_DESCRIPTION_PATH = "data/job_descriptions/sample_jd.txt"
TOP_N = 10

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)

with open(JOB_DESCRIPTION_PATH, "r", encoding="utf-8") as f:
    job_description = f.read().lower()

# --------------------------------------------------
# PROCESS JOB DESCRIPTION
# --------------------------------------------------
fit_jd_vectorizer(job_description)
jd_features = extract_jd_skills(job_description)

results = []

# --------------------------------------------------
# PROCESS RESUMES
# --------------------------------------------------

for _, row in df.iterrows():
    candidate_id = row["ID"]

    # 1️⃣ extract + anonymize resume
    raw_resume = str(row["Resume_str"])
    resume_text = anonymize_text(raw_resume).lower()
    resume_text = resume_text[:3000]

    # 2️⃣ extract features
    resume_features = extract_resume_features(resume_text, jd_features)



    similarity = compute_similarity(resume_text, job_description)

    matched_skills = resume_features["matched_skills"]
    missing_skills = resume_features["missing_skills"]

    score = compute_final_score(
        similarity_score=similarity,
        skill_match_ratio=resume_features["skill_match_ratio"],
        experience_years=resume_features["experience_years"]
    )


    results.append({
        "candidate_id": candidate_id,
        "score": round(score, 3),
        "similarity": round(similarity, 3),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills
    })

# --------------------------------------------------
# OUTPUT
# --------------------------------------------------

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="score", ascending=False)

results_df.to_csv("output/shortlisted_candidates.csv", index=False)

print(results_df.head(TOP_N))
