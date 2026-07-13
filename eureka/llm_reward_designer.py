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

from eureka.reflection import build_reflection

SYSTEM_PROMPT = """You are an expert reward-function designer for a reinforcement \
learning highway-driving agent (5 discrete actions: change lane left/right, idle, \
accelerate, decelerate). Your job is to write ONE Python function that computes an \
ADDITIONAL shaping reward term, added on top of the environment's own built-in \
reward (which already penalizes collisions and rewards speed). Your term should \
encourage nuanced, skillful behavior - safe overtaking, maintaining safe following \
distance, smooth lane changes - without simply duplicating collision/speed rewards.

Function contract (must match exactly):
    def shaping_reward(ego, road, info: dict) -> float:

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
- Return a single float, roughly in [-1, 1] per step (it's added on top of an
  already-normalized base reward)
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


def generate_candidates(best: dict, k: int, generation: int,
                         model: str, temperature: float) -> list[str]:
    """
    best: the current best candidate dict (or None on generation 0) - used
          to build the reflection prompt.

    Returns a list of up to k code strings (fewer if some API calls failed
    or didn't contain a parseable code block - loop.py handles that via
    smoke_test.py rejecting/skipping bad candidates).
    """
    from key_manager import get_key_manager
    manager = get_key_manager()

    user_prompt = build_reflection(best)

    candidates = []
    for i in range(k):
        print(f"  [llm] requesting candidate {i + 1}/{k} from {model}...", flush=True)
        call_start = time.time()
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
            elapsed = time.time() - call_start
            if code:
                candidates.append(code)
                print(f"  [llm] candidate {i + 1}/{k}: code received "
                      f"({len(code.splitlines())} lines, {elapsed:.1f}s)", flush=True)
            else:
                preview = (text or "").strip().replace("\n", " ")[:200]
                print(f"  [llm] candidate {i + 1}/{k}: no code block found "
                      f"({elapsed:.1f}s), skipping. Preview: {preview!r}", flush=True)
        except Exception as e:
            elapsed = time.time() - call_start
            print(f"  [llm] candidate {i + 1}/{k}: API call failed "
                  f"({elapsed:.1f}s): {e}", flush=True)

    return candidates