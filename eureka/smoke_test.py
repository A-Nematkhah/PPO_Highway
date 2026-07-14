"""
smoke_test.py

Validates one candidate's code BEFORE it's ever used for training:
    1. parses the EXACT code string with the AST allowlist in sandbox.py
    2. executes that same unmodified string in a restricted namespace (no
       imports, no file/network access - only a small whitelist of safe
       builtins + `math`) inside an isolated subprocess
    3. checks it defines a callable named `shaping_reward`
    4. runs it against several real (ego, road, info) states pulled from an
       actual highway-env rollout with random actions, varying `n_overtakes`
       across trials (including explicit nonzero probes)
    5. rejects it if it ever raises, returns a non-numeric/non-finite
       value, or returns something wildly out of the expected range

The AST allowlist and restricted exec are defense-in-depth only: they do NOT
make in-process exec() sandboxing fully secure against all escape vectors.
Training-time load (env_factory.py) now uses the same restricted exec path
instead of importlib, but workers still share the parent OS user's privileges
— see docs/SECURITY.md for remaining risk and the Phase 3 DSL roadmap.

This mirrors the smoke-testing approach already used in text2reward-Rag
before writing LLM-generated reward code to disk - never trust generated
code with a full training run until it's been exercised on real states.
"""

import multiprocessing
import sys

import gymnasium as gym
import highway_env  # noqa: F401

from config import ENV_CONFIG, ENV_ID
from eureka.sandbox import exec_shaping_reward, normalize_shaping_output, validate_candidate_ast
from reward_wrapper import compute_overtakes

MAX_ABS_VALUE = 5.0  # a shaping term returning something this large per step
                      # is almost certainly a bug, not intentional design
_RUNTIME_PROBE_TIMEOUT_S = 60.0
# POSIX-only defense-in-depth inside the probe worker (skipped on Windows).
_RUNTIME_PROBE_CPU_LIMIT_S = 30
_RUNTIME_PROBE_AS_LIMIT_BYTES = 512 * 1024 * 1024


def _apply_runtime_probe_resource_limits() -> None:
    """
    Cap CPU time and address space for the smoke-test subprocess on POSIX.
    Windows spawn semantics are supported but lack a portable rlimit equivalent.
    """
    if sys.platform == "win32":
        return
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (_RUNTIME_PROBE_CPU_LIMIT_S, _RUNTIME_PROBE_CPU_LIMIT_S))
    resource.setrlimit(resource.RLIMIT_AS, (_RUNTIME_PROBE_AS_LIMIT_BYTES, _RUNTIME_PROBE_AS_LIMIT_BYTES))


def _probe_shaping_fn(fn, n_trials: int) -> tuple[bool, str]:
    """
    Run `fn` against real highway-env states. Varies `n_overtakes` using
    compute_overtakes() across steps and also probes explicit nonzero values
    so candidates that only work when n_overtakes == 0 cannot slip through.
    """
    try:
        env = gym.make(ENV_ID)
        env.unwrapped.configure(ENV_CONFIG)
        env.reset(seed=0)
    except Exception as e:
        return False, f"failed to build test env: {e}"

    prev_relative_x: dict = {}
    passed, message = True, "ok"

    try:
        for i in range(n_trials):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            ego = env.unwrapped.vehicle
            road = env.unwrapped.road
            n_overtakes, prev_relative_x = compute_overtakes(road, ego, prev_relative_x)

            sample_infos = [
                {
                    "crashed": bool(info.get("crashed", False)),
                    "speed": info.get("speed", 0.0),
                    "raw_reward": reward,
                    "n_overtakes": n_overtakes,
                },
                {
                    "crashed": bool(info.get("crashed", False)),
                    "speed": info.get("speed", 0.0),
                    "raw_reward": reward,
                    "n_overtakes": max(n_overtakes, 2),
                },
            ]

            for probe_idx, sample_info in enumerate(sample_infos):
                try:
                    total, _components = normalize_shaping_output(fn(ego, road, sample_info))
                except Exception as e:
                    passed = False
                    message = (
                        f"runtime error on trial {i} "
                        f"(n_overtakes={sample_info['n_overtakes']}): {e}"
                    )
                    break

                if abs(total) > MAX_ABS_VALUE:
                    passed = False
                    message = (
                        f"return value out of range on trial {i} "
                        f"(probe {probe_idx}, n_overtakes={sample_info['n_overtakes']}): "
                        f"{total}"
                    )
                    break

            if not passed:
                break

            if terminated or truncated:
                env.reset(seed=i + 1)
                prev_relative_x = {}
    finally:
        env.close()

    return passed, message


def _runtime_probe_worker(code_str: str, n_trials: int, result_queue) -> None:
    """
    Subprocess entry point: exec + runtime probe with restricted builtins.
    Keeps a compromised candidate from running with the parent's full
    privileges during the smoke test (defense in depth, not a full jail).
    """
    _apply_runtime_probe_resource_limits()
    fn, error = exec_shaping_reward(code_str)
    if error is not None:
        result_queue.put((False, error))
        return
    result_queue.put(_probe_shaping_fn(fn, n_trials))


def _run_runtime_probe_subprocess(code_str: str, n_trials: int) -> tuple[bool, str]:
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_runtime_probe_worker,
        args=(code_str, n_trials, result_queue),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=_RUNTIME_PROBE_TIMEOUT_S)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return False, "runtime probe timed out"

    if result_queue.empty():
        return False, "runtime probe produced no result"
    return result_queue.get()


def smoke_test(code_str: str, n_trials: int = 5) -> tuple[bool, str]:
    """
    Validate the exact `code_str` that loop.py will write to disk and
    env_factory.py will load. Returns (passed: bool, message: str).
    """
    passed, message = validate_candidate_ast(code_str)
    if not passed:
        return False, message

    # AST allowlist is necessary but not sufficient: in-process exec() can still
    # be escaped by determined adversarial code. Run the runtime probe in a
    # separate subprocess so a failing/exploitative candidate does not inherit
    # the parent's full filesystem/network privileges during the smoke test.
    return _run_runtime_probe_subprocess(code_str, n_trials)
