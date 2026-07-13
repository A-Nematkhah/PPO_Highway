"""
reflection.py

Builds the "reward reflection" text: the best candidate's code plus its
actual measured metrics, fed back to the LLM as the prompt for the next
generation. This is what makes the loop evolutionary rather than just
repeated random sampling - each generation's proposals are explicitly
conditioned on concrete numeric feedback about what worked and what
didn't, not just a vague "try again."
"""

from eureka.eureka_config import N_EVAL_EPISODES


def build_reflection(best: dict) -> str:
    """
    best: {"code": str, "metrics": dict, "fitness": float, ...} or None
          (None on generation 0, when there's no prior candidate yet)
    """
    if best is None:
        return ("Design an initial reward shaping function to encourage safe, "
                "skillful overtaking behavior in highway driving.")

    m = best["metrics"]
    return (
        "Here is the best reward shaping function found so far, and how it "
        "performed when trained and evaluated:\n\n"
        f"```python\n{best['code']}\n```\n\n"
        f"Resulting metrics (over {N_EVAL_EPISODES} deterministic evaluation "
        "episodes):\n"
        f"- crash_rate: {m['crash_rate']:.2%}\n"
        f"- mean_speed: {m['mean_speed']:.2f} m/s\n"
        f"- mean_overtakes: {m['mean_overtakes']:.2f} per episode\n"
        f"- fitness score: {best['fitness']:.3f}\n\n"
        "Propose an IMPROVED reward shaping function. Keep what's working, fix "
        "what isn't - e.g. if crash_rate is still high, add stronger "
        "safety-oriented shaping; if mean_speed or mean_overtakes are low, add "
        "a stronger incentive for that. Make a meaningfully different attempt, "
        "not just a trivial constant tweak."
    )
