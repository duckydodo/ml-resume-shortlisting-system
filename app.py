import streamlit as st

from src.parser import extract_text
from src.anonymizer import anonymize_text
from src.feature_extraction import extract_jd_skills, extract_resume_features
from src.matcher import fit_jd_vectorizer, compute_similarity
from src.scorer import compute_final_score

# --------------------------------------------------
# UI SETUP
# --------------------------------------------------

st.set_page_config(page_title="Resume Scorer", layout="centered")

st.title("📄 Resume–Job Match Scorer")
st.write(
    "Upload a resume and paste a job description to see how well they match. "
    "Sensitive information is anonymized before scoring."
)

# --------------------------------------------------
# INPUTS
# --------------------------------------------------

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

job_description = st.text_area(
    "Paste Job Description",
    height=200,
    placeholder="Enter the job description here..."
)

# --------------------------------------------------
# ACTION
# --------------------------------------------------

if st.button("Score Resume"):
    if uploaded_file is None or not job_description.strip():
        st.warning("Please upload a resume and paste a job description.")
    else:
        with st.spinner("Analyzing resume..."):

            # 1️⃣ Parse resume
            raw_resume_text = extract_text(uploaded_file)

            # 2️⃣ Anonymize + normalize
            resume_text = anonymize_text(raw_resume_text).lower()
            resume_text = resume_text[:3000]  # speed safeguard

            jd_text = anonymize_text(job_description.lower())

            # 3️⃣ Extract JD skills ONCE
            jd_skills = extract_jd_skills(jd_text)

            # 4️⃣ Fit TF-IDF ONCE
            fit_jd_vectorizer(jd_text)

            # 5️⃣ Resume features
            resume_features = extract_resume_features(resume_text, jd_skills)

            # 6️⃣ Similarity
            similarity = compute_similarity(resume_text, jd_text)

            # 7️⃣ Final score
            final_score = compute_final_score(
                similarity_score=similarity,
                skill_match_ratio=resume_features["skill_match_ratio"],
                experience_years=resume_features["experience_years"]
            )

        # --------------------------------------------------
        # OUTPUT
        # --------------------------------------------------

        st.success("Analysis complete!")

        st.metric("Final Score", round(final_score * 100, 1))
        st.metric("Semantic Similarity", round(similarity, 3))

        st.subheader("Matched Skills")
        if resume_features["matched_skills"]:
            st.write(", ".join(resume_features["matched_skills"]))
        else:
            st.write("None")

        st.subheader("Missing Skills")
        if resume_features["missing_skills"]:
            st.write(", ".join(resume_features["missing_skills"]))
        else:
            st.write("None 🎉")
