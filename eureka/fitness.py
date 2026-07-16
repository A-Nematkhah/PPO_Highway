"""
fitness.py

Converts the metrics from evaluate_candidate.py (crash_rate, mean_speed,
mean_overtakes) into a single scalar score.

With MULTI_OBJECTIVE_MODE="pareto" (current default in eureka_config.py),
this scalar is diagnostic / logged only - survivor selection and LLM
reflection elites come from eureka.objectives. With mode "shadow", the
loop still uses this score to update the legacy `best` parent.

Deliberately does NOT use the candidate's own shaped training reward or
mean_raw_return - the score must come from metrics the candidate cannot
directly manipulate by "gaming" its own shaping function.

--------------------------------------------------------------------------
NOT SAFETY-AWARE BY DEFAULT WEIGHTING - read this before trusting the number
--------------------------------------------------------------------------
`compute_fitness` is a flat linear combination: it has no epsilon deadband,
no notion of a safety threshold, and no concept of a Pareto trade-off front.
A large enough gain on `speed` or `overtakes` can always outweigh a
worsening `crash_rate`, for ANY finite set of weights - that is a structural
property of a linear scalarization, not a bug you can fully weight away.

eureka_config.FITNESS_WEIGHTS has been rebalanced (see the comment there)
so that a doubling of crash_rate is not casually offset by a few m/s of
speed - it now takes a much larger, less plausible speed/overtake gain to
produce the same effect. That reduces how often the number is actively
misleading, but it is still a rough, safety-weighted SANITY CHECK for
logs/console output, not a ranking signal. Use the Pareto archive
(eureka.objectives) - which reasons about crash_rate/mean_speed/
mean_overtakes as independent, epsilon-quantized objectives - for anything
that should actually drive candidate selection.
"""


def compute_fitness(metrics: dict, weights: dict) -> float:
    return (
        -weights["crash"] * metrics["crash_rate"]
        + weights["speed"] * metrics["mean_speed"]
        + weights["overtakes"] * metrics["mean_overtakes"]
    )