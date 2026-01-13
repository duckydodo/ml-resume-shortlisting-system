from src.parser import extract_text
from src.anonymizer import anonymize_text
from src.feature_extraction import extract_jd_skills, extract_resume_features
from src.matcher import fit_jd_vectorizer, compute_similarity
from src.scorer import compute_final_score

# --------------------------------------------------
# INPUTS (CHANGE THESE PATHS ONLY)
# --------------------------------------------------

PDF_PATH = "data/resumes_raw/sample.pdf"   # put ONE resume here

JOB_DESCRIPTION = """
Looking for a data analyst with experience in Python, SQL,
data analysis, statistics, and machine learning.
"""

# --------------------------------------------------
# PIPELINE
# --------------------------------------------------

# Parse resume
raw_resume_text = extract_text(PDF_PATH)

# Anonymize
resume_text = anonymize_text(raw_resume_text).lower()

# Optional: cap size for speed
resume_text = resume_text[:3000]

# Prepare JD
jd_text = anonymize_text(JOB_DESCRIPTION.lower())

# Extract JD skills ONCE
jd_skills = extract_jd_skills(jd_text)
print("JD SKILLS:", jd_skills)

# Fit TF-IDF ONCE
fit_jd_vectorizer(jd_text)

# Resume features
resume_features = extract_resume_features(resume_text, jd_skills)

# Similarity
similarity = compute_similarity(resume_text, jd_text)

# Final score
final_score = compute_final_score(
    similarity_score=similarity,
    skill_match_ratio=resume_features["skill_match_ratio"],
    experience_years=resume_features["experience_years"]
)

# --------------------------------------------------
# OUTPUT
# --------------------------------------------------

print("\n===== RESULT =====")
print("Similarity:", round(similarity, 3))
print("Final Score:", round(final_score, 3))
print("Matched Skills:", resume_features["matched_skills"])
print("Missing Skills:", resume_features["missing_skills"])
