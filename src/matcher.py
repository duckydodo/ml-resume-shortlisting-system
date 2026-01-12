from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# create vectorizer ONCE
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

def fit_jd_vectorizer(jd_text: str):
    """
    Fit TF-IDF vectorizer on job description once.
    """
    vectorizer.fit([jd_text])


def compute_similarity(resume_text: str, jd_text: str) -> float:
    """
    Compute similarity using pre-fitted vectorizer.
    """
    resume_vec = vectorizer.transform([resume_text])
    jd_vec = vectorizer.transform([jd_text])

    similarity = cosine_similarity(resume_vec, jd_vec)
    return similarity[0][0]
