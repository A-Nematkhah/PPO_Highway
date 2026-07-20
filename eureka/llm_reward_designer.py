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
- Do NOT define any inner/nested function (no `def` inside `shaping_reward`).
  Use inline expressions, comprehensions, or extra local variables instead.
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

# The per-candidate reflection target roles cycled through in
# generate_candidates() below when elites are available (i.e. every
# generation after the first). Hoisted to module level - rather than a
# local literal inside generate_candidates() - purely so other modules
# (e.g. loop.py, archiving the reflection prompt actually used for a
# generation) can reference the exact same tuple instead of duplicating
# it; the cycling logic and values are unchanged.
REFLECTION_TARGET_ROLES = ("balanced_knee", "safest", "fastest_safe", "overtaking_safe")


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


def _call_llm(manager, model: str, temperature: float, user_prompt: str):
    """One raw chat_completion call. Raises RequestTooLargeError / other
    exceptions to the caller unchanged - retry policy lives in the caller."""
    return manager.chat_completion(
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


def generate_candidates(elites: dict | list[dict] | None, k: int, generation: int,
                         model: str, temperature: float) -> list[str]:
    """
    elites: legacy best dict, Pareto elite list, or None on generation 0.
            Different calls receive different trade-off targets so the LLM
            acts as a diverse mutation operator over the archive.

    Returns a list of up to k code strings (fewer if some API calls failed
    or didn't contain a parseable code block - loop.py handles that via
    smoke_test.py rejecting/skipping bad candidates).

    P1 fix: build_reflection() already caps prompt size at the source (see
    reflection.py), so a 413/"request too large" response should be rare.
    As defense in depth, a RequestTooLargeError is still caught here and
    retried EXACTLY ONCE per candidate with a forcibly minimal context
    (single highest-priority elite, no component detail) rather than
    silently losing that candidate slot entirely - this is what let
    generation 2 lose all 8 LLM calls at once in a prior run, with nothing
    but a one-line WARNING in the log.
    """
    from key_manager import get_key_manager, RequestTooLargeError
    manager = get_key_manager()

    candidates = []
    roles = REFLECTION_TARGET_ROLES
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
            try:
                response = _call_llm(manager, model, temperature, user_prompt)
            except RequestTooLargeError as e:
                logger.warning(
                    "LLM request too large, retrying with minimal context",
                    extra={
                        "event": "llm_call_request_too_large",
                        "generation": generation,
                        "index": i + 1,
                        "prompt_chars": len(user_prompt),
                        "error": str(e),
                    },
                )
                minimal_elites = None
                if isinstance(elites, dict):
                    minimal_elites = elites
                elif elites:
                    minimal_elites = elites[0]
                minimal_prompt = build_reflection(minimal_elites, target_role=target_role)
                response = _call_llm(manager, model, temperature, minimal_prompt)

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