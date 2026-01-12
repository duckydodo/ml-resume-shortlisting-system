import pandas as pd

from src.parser import parse_resume
from src.anonymizer import anonymize_text
from src.feature_extraction import extract_features
from src.matcher import compute_similarity
from src.scorer import compute_final_score


def score_single_resume(resume_file, job_description: str) -> dict:
    """
    Scores a single resume against a job description.
    Used in Candidate Mode (Streamlit).
    """

    # 1. Parse PDF → raw text
    resume_text = parse_resume(resume_file)

    # 2. Remove bias (names, gender, religion, etc.)
    clean_text = anonymize_text(resume_text)

    # 3. Extract interpretable features
    features = extract_features(clean_text, job_description)

    # 4. Compute similarity score
    similarity_score = compute_similarity(clean_text, job_description)

    # 5. Final weighted score
    final_score = compute_final_score(
        similarity_score=similarity_score,
        experience_years=features["experience_years"],
        skill_match_ratio=features["skill_match_ratio"]
    )

    return {
        "final_score": round(final_score, 3),
        "similarity_score": round(similarity_score, 3),
        "experience_years": features["experience_years"],
        "matched_skills": features["matched_skills"],
        "missing_skills": features["missing_skills"]
    }


def shortlist_resumes(resume_files: list, job_description: str, top_n: int = 10) -> pd.DataFrame:
    """
    Scores and ranks multiple resumes.
    Used in Recruiter Mode (Streamlit).
    """

    results = []

    for resume_file in resume_files:
        try:
            result = score_single_resume(resume_file, job_description)
            result["candidate_id"] = getattr(resume_file, "name", "unknown")
            results.append(result)
        except Exception as e:
            # Skip bad PDFs instead of crashing the whole app
            continue

    df = pd.DataFrame(results)
    df = df.sort_values(by="final_score", ascending=False)

    return df.head(top_n)
