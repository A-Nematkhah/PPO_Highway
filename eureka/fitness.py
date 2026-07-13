"""
fitness.py

Converts the metrics from evaluate_candidate.py (crash_rate, mean_speed,
mean_overtakes) into a single scalar used to rank candidates.

Deliberately does NOT use the candidate's own shaped training reward or
mean_raw_return - fitness must be computed from metrics the candidate
cannot directly manipulate by "gaming" its own shaping function. See
evaluate_candidate.py for why these particular metrics were chosen.
"""


def compute_fitness(metrics: dict, weights: dict) -> float:
    return (
        -weights["crash"] * metrics["crash_rate"]
        + weights["speed"] * metrics["mean_speed"]
        + weights["overtakes"] * metrics["mean_overtakes"]
    )
