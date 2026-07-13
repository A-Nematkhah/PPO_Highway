"""
smoke_test.py

Validates one candidate's code BEFORE it's ever used for training:
    1. executes it in a restricted namespace (no imports, no file/network
       access - only a small whitelist of safe builtins + `math`)
    2. checks it defines a callable named `shaping_reward`
    3. runs it against several real (ego, road, info) states pulled from an
       actual highway-env rollout with random actions
    4. rejects it if it ever raises, returns a non-numeric/non-finite
       value, or returns something wildly out of the expected range

This mirrors the smoke-testing approach already used in text2reward-Rag
before writing LLM-generated reward code to disk - never trust generated
code with a full training run until it's been exercised on real states.
"""

import math

import gymnasium as gym
import highway_env  # noqa: F401

from config import ENV_CONFIG, ENV_ID

# deliberately tiny whitelist - blocks import, open, exec, eval, __import__,
# and anything else that could do something unexpected. Includes common
# type names because the LLM prompt asks for `info: dict` annotations.
_SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "len": len, "range": range,
    "float": float, "int": int, "bool": bool, "sum": sum, "sorted": sorted,
    "dict": dict, "list": list, "tuple": tuple, "str": str, "set": set,
    "isinstance": isinstance, "enumerate": enumerate, "zip": zip,
    "round": round, "pow": pow, "True": True, "False": False, "None": None,
}

MAX_ABS_VALUE = 5.0  # a shaping term returning something this large per step
                      # is almost certainly a bug, not intentional design


def _sanitize_candidate_code(code_str: str) -> str:
    """
    Strip import lines from LLM-generated code. `math` is already injected
    into the exec namespace; without __import__ in the sandbox, any
    `import ...` statement fails with "__import__ not found".
    """
    lines = []
    for line in code_str.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def smoke_test(code_str: str, n_trials: int = 5) -> tuple[bool, str]:
    """
    Returns (passed: bool, message: str).
    """
    code_str = _sanitize_candidate_code(code_str)
    namespace = {}
    try:
        exec(compile(code_str, "<candidate>", "exec"),
             {"__builtins__": _SAFE_BUILTINS, "math": math}, namespace)
    except Exception as e:
        return False, f"exec failed: {e}"

    fn = namespace.get("shaping_reward")
    if fn is None or not callable(fn):
        return False, "no `shaping_reward` function defined"

    try:
        env = gym.make(ENV_ID)
        env.unwrapped.configure(ENV_CONFIG)
        obs, info = env.reset(seed=0)
    except Exception as e:
        return False, f"failed to build test env: {e}"

    passed, message = True, "ok"

    for i in range(n_trials):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        ego = env.unwrapped.vehicle
        road = env.unwrapped.road
        sample_info = {
            "crashed": bool(info.get("crashed", False)),
            "speed": info.get("speed", 0.0),
            "raw_reward": reward,
            "n_overtakes": 0,
        }

        try:
            value = fn(ego, road, sample_info)
        except Exception as e:
            passed, message = False, f"runtime error on trial {i}: {e}"
            break

        if not isinstance(value, (int, float)) or not math.isfinite(value):
            passed, message = False, f"invalid return value on trial {i}: {value!r}"
            break

        if abs(value) > MAX_ABS_VALUE:
            passed, message = False, f"return value out of range on trial {i}: {value}"
            break

        if terminated or truncated:
            obs, info = env.reset(seed=i + 1)

    env.close()
    return passed, message
