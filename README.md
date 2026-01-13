# Resume Shortlisting System

## Overview

This project implements a **bias-aware resume shortlisting system** designed to assist in early-stage candidate screening. The system evaluates how well resumes align with a given job description using lightweight NLP techniques, with an emphasis on **interpretability**, **modularity**, and **real-world usability** rather than opaque black-box predictions.

At its core, the system is intended to **filter out clearly unsuitable resumes**, not to make final hiring decisions. It provides structured signals (scores, matched skills, missing skills) that help reduce manual screening effort.

---

## What This Project Is (and Is Not)

**This project *is***:
- A practical screening and comparison tool
- Suitable for shortlisting large pools of applicants
- Designed to be explainable and debuggable

**This project is *not***:
- A final decision-making system
- A psychological or performance predictor
- A replacement for human judgment

Shortlisting is fundamentally about **rejecting resumes that are not a fit**, not selecting the single best candidate. This system reflects that philosophy.

---

## Key Features

- **PDF Resume Parsing**  
  Extracts text from real-world PDF resumes.

- **Bias Mitigation**  
  Anonymizes resumes before analysis to reduce the influence of personally identifiable attributes.

- **Job Description–Driven Skill Extraction**  
  Skills are dynamically extracted from the job description instead of relying on predefined keyword lists.

- **Skill Matching and Gap Analysis**  
  Canonicalizes and filters extracted skills to produce clear lists of matched and missing competencies.

- **Semantic Similarity Scoring**  
  Uses TF-IDF and cosine similarity to measure alignment between resume and job description.

- **Explainable Composite Scoring**  
  Combines similarity, skill coverage, and experience signals into a single numeric score.

- **Interactive Streamlit Interface**  
  Allows users to upload resumes and immediately view results.

---

## System Architecture

```
resume-shortlisting-system/
│
├── src/
│   ├── parser.py              # PDF text extraction
│   ├── anonymizer.py          # Bias mitigation and PII removal
│   ├── feature_extraction.py  # Skill extraction and matching logic
│   ├── matcher.py             # TF-IDF semantic similarity
│   ├── scorer.py              # Final scoring logic
│
├── app.py                     # Streamlit application
├── test_pdf_pipeline.py       # Single-resume CLI-style test pipeline
├── test_csv_pipeline.py       # Experimental CSV-based batch pipeline
├── .gitignore
└── README.md
```

Each module has a single responsibility, making the system easy to extend or refactor.

---

## Scoring Logic

The final score is a weighted combination of:

- **Semantic Similarity**: Textual alignment between resume and job description
- **Skill Match Ratio**: Fraction of job-relevant skills present in the resume
- **Experience Signal**: Approximate experience inferred from temporal patterns

The score is **relative**, not absolute.

- A score is meaningful **only when comparing multiple candidates for the same role**
- A single score in isolation does not imply "good" or "bad"

---

## Bias Mitigation

Before any scoring or feature extraction:
- Resume text is anonymized
- Names and demographic signals are removed
- All downstream processing operates on anonymized content

This helps reduce unintended bias during automated screening.

---

## CSV Pipeline (Experimental)

An experimental CSV-based pipeline is included for batch shortlisting scenarios. This pipeline allows scoring and ranking multiple candidates at once, which is where numeric scores become meaningful.

### Design Notes
- CSV schemas vary widely across datasets
- Column naming conventions are inconsistent
- This pipeline assumes **user-specified columns** (e.g., applicant ID, resume text)

Because of this variability, the CSV pipeline is considered **experimental** and is not exposed through the Streamlit UI by default. It is better suited for controlled or internal use cases.

---

## Limitations

This system intentionally favors simplicity and transparency, which introduces limitations:

- Skill extraction is heuristic-based and may count generic or borderline terms as skills
- Experience estimation is approximate
- No normalization exists across different job descriptions
- Scores are not calibrated across roles

These limitations are acknowledged by design to keep the system interpretable and extensible.

---

## Future Work

- Multi-resume recruiter mode with ranking and export
- User-defined CSV column mapping in the UI
- Improved skill ontology and phrase clustering
- More robust experience modeling

---

## Disclaimer

This project is intended for **educational and experimental purposes**. It should be used as a decision-support system, not a sole authority for hiring decisions.

---

## Author

Chittaranjan Dutta  


