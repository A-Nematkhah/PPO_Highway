"""
reflection.py

Builds reward-reflection text from one legacy best candidate or multiple
Pareto elites. Multi-elite prompts expose the real safety/speed/overtaking
trade-off instead of pretending that one hand-weighted scalar is a universal
definition of "best."
"""

from eureka.eureka_config import N_EVAL_EPISODES


_TARGET_INSTRUCTIONS = {
    "balanced_knee": (
        "Seek a balanced improvement across all three objectives without "
        "sacrificing safety for a small speed gain."
    ),
    "safest": (
        "Use the safest elite as the parent and improve speed/overtaking while "
        "keeping crash_rate within the configured safety epsilon."
    ),
    "fastest_safe": (
        "Use the fastest safety-eligible elite as the parent; preserve its speed "
        "while reducing crash_rate or improving overtakes."
    ),
    "overtaking_safe": (
        "Use the strongest safety-eligible overtaking elite as the parent; "
        "preserve safety while making overtakes smoother and repeatable."
    ),
}


def build_reflection(elites: dict | list[dict] | None, target_role: str | None = None) -> str:
    """
    elites: one legacy best dict, a Pareto-elite list, or None on generation 0.
    target_role: optional direction for this particular LLM offspring request.
    """
    if not elites:
        return ("Design an initial reward shaping function to encourage safe, "
                "skillful overtaking behavior in highway driving.")

    candidates = [elites] if isinstance(elites, dict) else list(elites)
    sections = []
    for index, candidate in enumerate(candidates, start=1):
        metrics = candidate["metrics"]
        role = candidate.get("reflection_role", "legacy_best")
        pareto = ""
        if "pareto_rank" in candidate:
            pareto = (
                f"- Pareto rank: {candidate['pareto_rank']}\n"
                f"- reflection role: {role}\n"
            )
        legacy = (
            f"- legacy scalar score (diagnostic only): {candidate['fitness']:.3f}\n"
            if "fitness" in candidate
            else ""
        )
        sections.append(
            f"Candidate {index} ({role}):\n"
            f"```python\n{candidate['code']}\n```\n"
            f"Metrics over {N_EVAL_EPISODES} deterministic evaluation episodes:\n"
            f"- crash_rate: {metrics['crash_rate']:.2%}\n"
            f"- mean_speed: {metrics['mean_speed']:.2f} m/s\n"
            f"- mean_overtakes: {metrics['mean_overtakes']:.2f} per episode\n"
            f"{pareto}{legacy}"
        )

    target = _TARGET_INSTRUCTIONS.get(
        target_role,
        "Propose a meaningfully different candidate that improves the Pareto "
        "front rather than optimizing a hidden weighted sum.",
    )
    return (
        "The following reward programs are non-dominated trade-off elites. "
        "There is no single globally best candidate: lower crash_rate is better, "
        "while higher mean_speed and mean_overtakes are better.\n\n"
        + "\n".join(sections)
        + "\n"
        + target
        + "\nReturn an IMPROVED reward function, not a trivial constant tweak."
    )
