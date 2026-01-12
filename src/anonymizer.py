import re
import spacy

nlp = spacy.load("en_core_web_sm")

GENDER_TERMS = [
    "he", "she", "him", "her", "his", "hers",
    "male", "female", "man", "woman", "men", "women"
]

RELIGION_TERMS = [
    "hindu", "muslim", "christian", "sikh", "jewish",
    "buddhist", "jain"
]

TITLE_TERMS = [
    "mr", "mrs", "ms", "miss", "sir", "madam"
]


def anonymize_text(text: str) -> str:
    """
    Removes bias-inducing attributes:
    names, gender, religion, nationality
    """
    text = text.lower()
    text = _remove_named_entities(text)
    text = _remove_terms(text, GENDER_TERMS)
    text = _remove_terms(text, RELIGION_TERMS)
    text = _remove_terms(text, TITLE_TERMS)
    return text


def _remove_named_entities(text: str) -> str:
    doc = nlp(text)
    anonymized = text

    for ent in doc.ents:
        if ent.label_ in ["PERSON", "NORP", "GPE"]:
            anonymized = anonymized.replace(ent.text, "<REDACTED>")

    return anonymized


def _remove_terms(text: str, terms: list) -> str:
    pattern = r"\b(" + "|".join(terms) + r")\b"
    return re.sub(pattern, "<REDACTED>", text)
