def compute_final_score(
    similarity_score: float,
    skill_match_ratio: float,
    experience_years: int
) -> float:
    """
    Compute final candidate score using weighted components.

    similarity_score: cosine similarity between resume & JD (0–1)
    skill_match_ratio: fraction of JD skills matched (0–1)
    experience_years: estimated experience signal (integer)
    """

    # Normalize experience (cap at 10 years to avoid domination)
    experience_component = min(experience_years / 10, 1.0)

    final_score = (
        0.5 * similarity_score +        # semantic relevance (most important)
        0.3 * skill_match_ratio +       # JD-specific skill coverage
        0.2 * experience_component      # timeline signal
    )

    return final_score
