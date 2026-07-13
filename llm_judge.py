"""
llm_judge.py

Phase 1 of the LLM integration roadmap: the LLM acts as a binary judge of
episode quality, not as a reward-function designer (that's a later phase).

judge_episode() sends a short summary of one finished episode (crashed?,
mean speed, overtakes, length) to a Groq model and asks for a single
normalized score: 1 (good driving) or 0 (bad driving). This score gets
multiplied by a weight and added to the reward at the terminal step - see
reward_wrapper.py.

IMPORTANT PERFORMANCE NOTE:
This makes a blocking network call. It is only ever called once per
FINISHED EPISODE (not per step), and even then only every
LLM_JUDGE_EVERY_N_EPISODES episodes (config.py), specifically because a
per-step call would make training network-bound instead of compute-bound.
With N_ENVS parallel envs each calling this independently, expect training
fps to drop noticeably whenever USE_LLM_JUDGE=True - this is an inherent
cost of Phase 1's design, not a bug.

Requires:
    pip install groq
    groq_keys.json filled in with at least one key (see key_manager.py)
"""


def _build_prompt(stats: dict) -> str:
    return (
        "You are judging one episode of a self-driving RL agent in a "
        "highway simulator. Based on the stats below, respond with ONLY "
        "the single character 1 if this was good, skillful driving "
        "(reasonably fast, made progress, no crash), or ONLY the single "
        "character 0 if it was poor driving (crashed, or drove unusually "
        "slowly/passively with no progress). Respond with nothing else.\n\n"
        f"crashed: {stats['crashed']}\n"
        f"mean_speed_mps: {stats['mean_speed']:.2f}\n"
        f"overtakes: {stats['overtakes']}\n"
        f"episode_length_steps: {stats['length']}\n"
    )


def judge_episode(stats: dict, model: str = "openai/gpt-oss-20b") -> int:
    """
    stats: dict with keys "crashed" (bool), "mean_speed" (float),
           "overtakes" (int), "length" (int)

    Returns 0 or 1. On any API error, returns 0 and prints a warning rather
    than raising - a flaky network call should never crash training.
    """
    try:
        from key_manager import get_key_manager
        manager = get_key_manager()
    except Exception as e:
        print(f"[llm_judge] WARNING: key manager unavailable ({e}). Returning 0.")
        return 0

    try:
        response = manager.chat_completion(
            model=model,
            messages=[{"role": "user", "content": _build_prompt(stats)}],
            # gpt-oss models are reasoning models: internal reasoning tokens
            # count against max_tokens before any visible answer appears, so
            # max_tokens=2 would always yield empty content (and a constant
            # score of 0). Give headroom and minimize reasoning instead.
            max_tokens=1000,
            temperature=0,
            reasoning_effort="low",
        )
        text = (response.choices[0].message.content or "").strip()
        return 1 if "1" in text else 0
    except Exception as e:  # noqa: BLE001 - deliberately broad: never crash training
        print(f"[llm_judge] WARNING: Groq call failed ({e}). Returning 0.")
        return 0


if __name__ == "__main__":
    # quick manual test - requires GROQ_API_KEY to be set
    good_episode = {"crashed": False, "mean_speed": 27.5, "overtakes": 3, "length": 140}
    bad_episode = {"crashed": True, "mean_speed": 12.0, "overtakes": 0, "length": 18}

    print("good_episode score:", judge_episode(good_episode))
    print("bad_episode score:", judge_episode(bad_episode))
