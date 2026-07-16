"""
reflection.py

Builds reward-reflection text from one legacy best candidate or multiple
Pareto elites. Multi-elite prompts expose the real safety/speed/overtaking
trade-off instead of pretending that one hand-weighted scalar is a universal
definition of "best."

When available, also surfaces per-component reward means and chronological
training snapshots (EUREKA paper Sec 3.3 / Prompt 2) so the LLM can revise
mis-scaled terms instead of guessing from scalar outcomes alone.

--------------------------------------------------------------------------
Prompt-size budget (P1 fix)
--------------------------------------------------------------------------
Groq (and most hosted LLM APIs) enforce a per-request token limit that is
independent of any per-key rate limit - a single call can be rejected with
HTTP 413 / "rate_limit_exceeded" ("Request too large ... Limit 8000,
Requested 8140") even when the account has plenty of remaining quota.
Nothing about which API key sends the request fixes this: the payload
itself is too big. With REFLECTION_ELITES=3 and multi-generation
component_history accumulating (each entry sampled down to at most 10
points, but per component, and possibly many components), a 3-elite
prompt with full history can silently cross that boundary - which is
exactly what happened when generation 2 lost all 8 LLM calls at once
with only a one-line WARNING in the log.

build_reflection() now measures the assembled prompt against a character
budget (a cheap proxy for tokens - roughly 4 chars/token for English text/
code) and degrades in stages, cheapest-to-drop first, until it fits:
    1. full detail (component means + chronological history for all elites)
    2. drop component_history (checkpoint snapshots), keep component means
    3. drop component means too, keep only code + scalar metrics
    4. reduce the elite count (drop lowest-priority roles first)
    5. last resort: keep only the single highest-priority elite

Each stage is retried on the FULL elite set before the elite count is
reduced, since dropping detail is nearly free (no information about *which*
candidates exist is lost) whereas dropping an elite loses an entire
trade-off perspective from the LLM's context.
"""

from eureka.eureka_config import N_EVAL_EPISODES


# Cheap token-count proxy: ~4 characters per token for typical English +
# Python source. Deliberately conservative (biased toward triggering
# degradation a bit early) since token counts for code can run higher than
# prose. Groq's observed limit for this model was ~8000 tokens; budgeting
# well under that leaves headroom for the system prompt and completion
# tokens, which are billed against the same per-request ceiling.
_CHARS_PER_TOKEN_ESTIMATE = 4
MAX_PROMPT_TOKENS_ESTIMATE = 6000
MAX_PROMPT_CHARS = MAX_PROMPT_TOKENS_ESTIMATE * _CHARS_PER_TOKEN_ESTIMATE


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

# Priority order used only to decide which elite to drop FIRST if the
# prompt still doesn't fit after dropping detail (stage 4/5). Earlier
# entries are kept longest. Any role not listed (e.g. "diverse_front",
# "legacy_best") falls back to this default priority.
_ELITE_DROP_PRIORITY = {
    "balanced_knee": 0,
    "safest": 1,
    "fastest_safe": 2,
    "overtaking_safe": 3,
}
_DEFAULT_ELITE_PRIORITY = 4


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


def _format_component_sections(
    candidate: dict, metrics: dict, include_means: bool, include_history: bool
) -> str:
    parts = []
    if include_means:
        component_means = metrics.get("component_means") or candidate.get("component_means") or {}
        if component_means:
            lines = ["Reward component means over this evaluation:"]
            for key in sorted(component_means):
                lines.append(f"- {key}: {float(component_means[key]):.4f}")
            parts.append("\n".join(lines) + "\n")

    if include_history:
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


def _render_candidate_section(
    index: int, candidate: dict, include_means: bool, include_history: bool
) -> str:
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
    component_block = _format_component_sections(
        candidate, metrics, include_means=include_means, include_history=include_history
    )
    return (
        f"Candidate {index} ({role}):\n"
        f"```python\n{candidate['code']}\n```\n"
        f"Metrics over {N_EVAL_EPISODES} deterministic evaluation episodes:\n"
        f"- crash_rate: {metrics['crash_rate']:.2%}\n"
        f"- mean_speed: {metrics['mean_speed']:.2f} m/s\n"
        f"- mean_overtakes: {metrics['mean_overtakes']:.2f} per episode\n"
        f"{component_block}{pareto}{legacy}"
    )


def _render_prompt(
    candidates: list[dict], target_role: str | None,
    include_means: bool, include_history: bool,
) -> str:
    sections = [
        _render_candidate_section(index, candidate, include_means, include_history)
        for index, candidate in enumerate(candidates, start=1)
    ]
    target = _TARGET_INSTRUCTIONS.get(
        target_role,
        "Propose a meaningfully different candidate that improves the Pareto "
        "front rather than optimizing a hidden weighted sum.",
    )
    if len(candidates) == 1:
        intro = (
            "The following reward program is the current best candidate. "
        )
    else:
        intro = (
            "The following reward programs are non-dominated trade-off elites. "
            "There is no single globally best candidate: lower crash_rate is better, "
            "while higher mean_speed and mean_overtakes are better.\n\n"
        )
    return (
        intro
        + "\n".join(sections)
        + "\n"
        + target
        + "\nReturn an IMPROVED reward function, not a trivial constant tweak."
    )


def _elite_sort_key(candidate: dict) -> tuple:
    role = candidate.get("reflection_role", "")
    return _ELITE_DROP_PRIORITY.get(role, _DEFAULT_ELITE_PRIORITY)


def build_reflection(elites: dict | list[dict] | None, target_role: str | None = None) -> str:
    """
    elites: one legacy best dict, a Pareto-elite list, or None on generation 0.
    target_role: optional direction for this particular LLM offspring request.

    Degrades detail (then elite count) until the assembled prompt fits under
    MAX_PROMPT_CHARS - see module docstring for the staged degradation order.
    """
    if not elites:
        return ("Design an initial reward shaping function to encourage safe, "
                "skillful overtaking behavior in highway driving.")

    all_candidates = [elites] if isinstance(elites, dict) else list(elites)
    # Keep highest-priority elites first so any later truncation drops the
    # least valuable trade-off perspective, not an arbitrary one.
    ordered_candidates = sorted(all_candidates, key=_elite_sort_key)

    degradation_stages = (
        {"include_means": True, "include_history": True},
        {"include_means": True, "include_history": False},
        {"include_means": False, "include_history": False},
    )

    fallback_prompt = None
    for candidate_count in range(len(ordered_candidates), 0, -1):
        candidates_to_use = ordered_candidates[:candidate_count]
        for stage in degradation_stages:
            prompt = _render_prompt(candidates_to_use, target_role, **stage)
            fallback_prompt = prompt
            if len(prompt) <= MAX_PROMPT_CHARS:
                return prompt

    # Every stage (down to a single elite, minimal detail) still overflowed
    # the budget - this should be rare (it implies one candidate's own code
    # is enormous). Return the smallest version we built rather than raising,
    # since the caller (generate_candidates) still needs *something* to send
    # and a real API-side 413 will surface the same information downstream.
    return fallback_prompt