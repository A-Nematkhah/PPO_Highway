"""
fitness.py

Converts the metrics from evaluate_candidate.py (crash_rate, mean_speed,
mean_overtakes) into a single scalar score.

With MULTI_OBJECTIVE_MODE="pareto" (current default in eureka_config.py),
this scalar is diagnostic / logged only — survivor selection and LLM
reflection elites come from eureka.objectives. With mode "shadow", the
loop still uses this score to update the legacy `best` parent.

Deliberately does NOT use the candidate's own shaped training reward or
mean_raw_return - the score must come from metrics the candidate cannot
directly manipulate by "gaming" its own shaping function.
"""


def compute_fitness(metrics: dict, weights: dict) -> float:
    return (
        -weights["crash"] * metrics["crash_rate"]
        + weights["speed"] * metrics["mean_speed"]
        + weights["overtakes"] * metrics["mean_overtakes"]
    )
