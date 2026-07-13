"""
reflection.py

Builds reward-reflection text from one legacy best candidate or multiple
Pareto elites. Multi-elite prompts expose the real safety/speed/overtaking
trade-off instead of pretending that one hand-weighted scalar is a universal
definition of "best."

When available, also surfaces per-component reward means and chronological
training snapshots (EUREKA paper Sec 3.3 / Prompt 2) so the LLM can revise
mis-scaled terms instead of guessing from scalar outcomes alone.
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


def _sample_checkpoint_values(values: list[float], max_points: int = 10) -> list[float]:
    if len(values) <= max_points:
        return list(values)
    if max_points <= 1:
        return [values[-1]]
    indices = [
        round(i * (len(values) - 1) / (max_points - 1))
        for i in range(max_points)
    ]
    return [values[index] for index in indices]


def _format_component_sections(candidate: dict, metrics: dict) -> str:
    parts = []
    component_means = metrics.get("component_means") or candidate.get("component_means") or {}
    if component_means:
        lines = ["Reward component means over this evaluation:"]
        for key in sorted(component_means):
            lines.append(f"- {key}: {float(component_means[key]):.4f}")
        parts.append("\n".join(lines) + "\n")

    history = candidate.get("component_history") or {}
    if history:
        lines = [
            "Reward component values at checkpoints during training (chronological):"
        ]
        for key in sorted(history):
            sampled = _sample_checkpoint_values(list(history[key]))
            formatted = ", ".join(f"{float(v):.2f}" for v in sampled)
            lines.append(f"- {key}: [{formatted}]")
        parts.append("\n".join(lines) + "\n")

    return "".join(parts)


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
        component_block = _format_component_sections(candidate, metrics)
        sections.append(
            f"Candidate {index} ({role}):\n"
            f"```python\n{candidate['code']}\n```\n"
            f"Metrics over {N_EVAL_EPISODES} deterministic evaluation episodes:\n"
            f"- crash_rate: {metrics['crash_rate']:.2%}\n"
            f"- mean_speed: {metrics['mean_speed']:.2f} m/s\n"
            f"- mean_overtakes: {metrics['mean_overtakes']:.2f} per episode\n"
            f"{component_block}{pareto}{legacy}"
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
