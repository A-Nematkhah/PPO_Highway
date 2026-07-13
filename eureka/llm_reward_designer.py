"""
llm_reward_designer.py

Asks the LLM for K reward-shaping CODE candidates (not just numeric
weights - this is what distinguishes Phase 4 from Phase 2). Each call
returns one candidate; we make K separate calls (rather than one call
asking for K variants) so each response is a single, cleanly-parseable
code block instead of relying on fragile multi-block parsing.
"""

import re
import time

from eureka.logging_utils import get_logger
from eureka.reflection import build_reflection

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are an expert reward-function designer for a reinforcement \
learning highway-driving agent (5 discrete actions: change lane left/right, idle, \
accelerate, decelerate). Your job is to write ONE Python function that computes an \
ADDITIONAL shaping reward term, added on top of the environment's own built-in \
reward (which already penalizes collisions and rewards speed). Your term should \
encourage nuanced, skillful behavior - safe overtaking, maintaining safe following \
distance, smooth lane changes - without simply duplicating collision/speed rewards.

Function contract:
    def shaping_reward(ego, road, info: dict):
        # Prefer returning (total_reward, reward_components) so future revisions
        # can see which term is under/over-scaled (EUREKA paper Prompt 3).
        # A bare float is still accepted as a fallback.
        ttc_temp = 5.0
        ...
        ttc_component = ...
        overtake_component = ...
        total = ttc_component + overtake_component
        return total, {"ttc_penalty": ttc_component, "overtake_bonus": overtake_component}

Available attributes:
- ego.position: [x, y] (x = longitudinal position along the road, y = lateral position)
- ego.speed: float, current speed (m/s)
- ego.lane_index: tuple (from_node, to_node, lane_id) - lane_id is an integer lane number
- road.vehicles: list of all Vehicle objects in the scene (including ego); each has the
  same .position / .speed / .lane_index attributes
- info: dict with keys "crashed" (bool), "speed" (float), "raw_reward" (float, this
  step's built-in reward), "n_overtakes" (int, vehicles overtaken this exact step,
  already computed for you)

Constraints:
- The `math` module is already available in scope - do NOT write any import
  statements (no `import math`, no `from ... import ...`, no numpy)
- If you apply a transformation (e.g. math.exp, math.tanh, a sigmoid-like
  squashing, or any function with a tunable scale/steepness constant) to
  a reward term, expose that constant as a locally-assigned named
  variable inside the function body (e.g. `ttc_temp = 5.0` used as
  `math.exp(-ttc_temp * severity)`), rather than an inline numeric
  literal. Use a descriptive name that indicates what it scales (e.g.
  `ttc_temp`, `overtake_scale`, `speed_bonus_weight`). This keeps future
  revisions of this function easy to tune precisely.
- Return either a single float OR a 2-tuple
  `(total_reward: float, reward_components: dict[str, float])`. Prefer the
  2-tuple form: expose named sub-terms that combine into total_reward so
  later feedback can revise specific components. total_reward should be
  roughly in [-1, 1] per step (it's added on top of an already-normalized
  base reward). Components are diagnostic only — training uses total_reward.
- This function is called every single step - do not reference terminal/episode state
- Respond with ONLY one fenced ```python code block containing the complete function
  definition. No explanation, no other text.
"""


def _extract_code(text: str) -> str | None:
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if not match:
        match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # fallback: some reasoning models (e.g. gpt-oss) don't reliably wrap
    # their answer in a fenced code block despite being told to. If we can
    # find the function definition directly, grab from there to the end.
    # Anything after the function (accidental trailing prose) will make the
    # exec() in smoke_test.py fail cleanly, so this is safe to try.
    idx = text.find("def shaping_reward")
    if idx != -1:
        return text[idx:].strip()

    return None


def generate_candidates(elites: dict | list[dict] | None, k: int, generation: int,
                         model: str, temperature: float) -> list[str]:
    """
    elites: legacy best dict, Pareto elite list, or None on generation 0.
            Different calls receive different trade-off targets so the LLM
            acts as a diverse mutation operator over the archive.

    Returns a list of up to k code strings (fewer if some API calls failed
    or didn't contain a parseable code block - loop.py handles that via
    smoke_test.py rejecting/skipping bad candidates).
    """
    from key_manager import get_key_manager
    manager = get_key_manager()

    candidates = []
    roles = ("balanced_knee", "safest", "fastest_safe", "overtaking_safe")
    for i in range(k):
        target_role = roles[i % len(roles)] if elites else None
        user_prompt = build_reflection(elites, target_role=target_role)
        logger.info(
            "LLM candidate request",
            extra={
                "event": "llm_call_start",
                "generation": generation,
                "index": i + 1,
                "k": k,
                "model": model,
                "target_role": target_role,
            },
        )
        call_start = time.perf_counter()
        try:
            response = manager.chat_completion(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                # gpt-oss models are reasoning models: their internal
                # chain-of-thought counts against max_tokens BEFORE any
                # visible answer is produced. 500 was entirely eaten by
                # reasoning (finish_reason="length", content=""), so give
                # plenty of headroom and cap the reasoning effort.
                max_tokens=4000,
                reasoning_effort="low",
            )
            text = response.choices[0].message.content
            code = _extract_code(text)
            elapsed = round(time.perf_counter() - call_start, 4)
            if code:
                candidates.append(code)
                logger.info(
                    "LLM code received",
                    extra={
                        "event": "llm_call_success",
                        "generation": generation,
                        "index": i + 1,
                        "lines": len(code.splitlines()),
                        "duration_s": elapsed,
                    },
                )
            else:
                preview = (text or "").strip().replace("\n", " ")[:200]
                logger.warning(
                    "LLM response had no parseable code",
                    extra={
                        "event": "llm_call_no_code",
                        "generation": generation,
                        "index": i + 1,
                        "duration_s": elapsed,
                        "preview": preview,
                    },
                )
        except Exception as e:
            elapsed = round(time.perf_counter() - call_start, 4)
            logger.error(
                "LLM API call failed",
                extra={
                    "event": "llm_call_error",
                    "generation": generation,
                    "index": i + 1,
                    "duration_s": elapsed,
                    "error": str(e),
                },
            )

    return candidates