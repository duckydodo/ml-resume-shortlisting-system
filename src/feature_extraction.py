import spacy

nlp = spacy.load("en_core_web_sm")
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

GENERIC_SKILLS = {
    "data", "system", "systems", "tools", "tool",
    "process", "processes", "project", "projects",
    "management", "education", "bachelor", "master",
    "degree", "communication", "presentation",
    "skills", "responsibilities", "experience",
    "work", "analysis", "analytical"
}
SOFT_SKILLS = {
    "and", "big", "business", "computer", "diverse",
    "interactive", "messy", "raw", "soft", "teamwork",
    "clear", "efficient", "valuable", "familiarity",
    "integrity", "consistency"
}

def _normalize_skill(skill: str) -> str:
    skill = skill.lower()

    fillers = [
        "its", "strong", "excellent", "potentially",
        "technical", "non-technical", "multiple",
        "different", "various", "complex"
    ]

    for f in fillers:
        skill = skill.replace(f, "")

    return " ".join(skill.split())

def _canonicalize_skill(skill: str) -> str:
    """
    Reduce skill phrase to its core concept.
    """
    tokens = skill.split()

    # prioritize known core heads
    for head in [
        "python", "sql", "statistics", "machine learning",
        "data analysis", "data visualization", "databases",
        "spark", "hadoop", "tableau", "power bi", "numpy", "pandas"
    ]:
        if head in skill:
            return head

    # fallback: first meaningful token
    for tok in tokens:
        if tok not in GENERIC_SKILLS and len(tok) > 2:
            return tok

    return None



def extract_jd_skills(jd_text: str) -> list:
    """
    Extract skill phrases from job description ONCE.
    """
    doc = nlp(jd_text.lower())
    skills = []

    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip()

        if phrase.startswith(("a ", "an ", "the ")):
            continue
        if any(role in phrase for role in ROLE_WORDS):
            continue
        if any(word in phrase for word in ["experience", "knowledge", "ability"]):
            continue
        if len(phrase.split()) > 4 or len(phrase) < 3:
            continue

        phrase = re.sub(r"[^a-zA-Z0-9+\-\. ]", "", phrase)
        skills.append(phrase)

    return list(set(skills))


def extract_resume_features(resume_text: str, jd_skills: list) -> dict:
    resume_text = resume_text.lower()

    resume_tokens = set(resume_text.split())

    matched = set()
    missing = set()

    for skill in jd_skills:
        norm_skill = _normalize_skill(skill)
        canonical = _canonicalize_skill(norm_skill)
        if not canonical:
            continue
        if canonical in SOFT_SKILLS:
            continue

        skill_tokens = norm_skill.split()

        if any(tok in resume_tokens for tok in skill_tokens):
            matched.add(canonical)
        else:
            missing.add(canonical)

    experience_years = _estimate_experience(resume_text)

    skill_match_ratio = (
        len(matched) / len(jd_skills) if jd_skills else 0
    )
    missing = missing - matched

    return {
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing),
        "skill_match_ratio": round(skill_match_ratio, 3),
        "experience_years": experience_years
    }

ROLE_WORDS = {
    "analyst", "engineer", "developer", "manager",
    "specialist", "professional", "candidate", "role"
}

def _estimate_experience(text: str) -> int:
    """
    Rough estimation of experience using year mentions.
    """
    years = re.findall(r"\b(19|20)\d{2}\b", text)
    return len(set(years))
