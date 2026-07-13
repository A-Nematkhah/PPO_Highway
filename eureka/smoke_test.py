"""
smoke_test.py

Validates one candidate's code BEFORE it's ever used for training:
    1. parses the EXACT code string with `ast` and rejects forbidden constructs
       (imports, dunder access, exec/eval/open/compile/__import__, etc.)
    2. executes that same unmodified string in a restricted namespace (no
       imports, no file/network access - only a small whitelist of safe
       builtins + `math`) inside an isolated subprocess
    3. checks it defines a callable named `shaping_reward`
    4. runs it against several real (ego, road, info) states pulled from an
       actual highway-env rollout with random actions, varying `n_overtakes`
       across trials (including explicit nonzero probes)
    5. rejects it if it ever raises, returns a non-numeric/non-finite
       value, or returns something wildly out of the expected range

The AST gate and restricted exec are defense-in-depth only: they do NOT make
in-process exec() sandboxing fully secure against all escape vectors.
Ideally both this runtime probe and the training-time import in
env_factory.py would run inside a further-isolated worker (subprocess with
restricted filesystem/network permissions, or a container). The subprocess
wrapper here limits blast radius for the probe; training-time import still
runs candidate module code with full process privileges (see env_factory.py).

This mirrors the smoke-testing approach already used in text2reward-Rag
before writing LLM-generated reward code to disk - never trust generated
code with a full training run until it's been exercised on real states.
"""

import ast
import math
import multiprocessing

import gymnasium as gym
import highway_env  # noqa: F401

from config import ENV_CONFIG, ENV_ID
from reward_wrapper import compute_overtakes

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

_FORBIDDEN_CALL_NAMES = frozenset({"__import__", "eval", "exec", "open", "compile"})
_FORBIDDEN_NODE_TYPES = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)

MAX_ABS_VALUE = 5.0  # a shaping term returning something this large per step
                      # is almost certainly a bug, not intentional design
_RUNTIME_PROBE_TIMEOUT_S = 60.0


class _ForbiddenConstructVisitor(ast.NodeVisitor):
    """
    Rejects standard exec-sandbox escape vectors before any code runs.
    The AST pass is necessary but not sufficient for security on its own.
    """

    def __init__(self):
        self.error: str | None = None

    def _reject(self, message: str) -> None:
        if self.error is None:
            self.error = message

    def generic_visit(self, node):
        if self.error is not None:
            return
        super().generic_visit(node)

    def visit_Import(self, node):
        self._reject("import statements are not allowed")

    def visit_ImportFrom(self, node):
        self._reject("import statements are not allowed")

    def visit_Global(self, node):
        self._reject("global statements are not allowed")

    def visit_Nonlocal(self, node):
        self._reject("nonlocal statements are not allowed")

    def visit_Attribute(self, node):
        if node.attr.startswith("__"):
            self._reject(f"dunder attribute access is not allowed: {node.attr!r}")
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALL_NAMES:
            self._reject(f"call to {node.func.id!r} is not allowed")
        elif isinstance(node.func, ast.Attribute) and node.func.attr in _FORBIDDEN_CALL_NAMES:
            self._reject(f"call to {node.func.attr!r} is not allowed")
        self.generic_visit(node)


def validate_candidate_ast(code_str: str) -> tuple[bool, str]:
    """
    Parse `code_str` and reject forbidden constructs before any exec().
    Returns (passed, message). On pass, the same string is safe to exec in
    the restricted namespace (modulo remaining in-process escape risk).
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, f"syntax error: {e}"

    visitor = _ForbiddenConstructVisitor()
    visitor.visit(tree)
    if visitor.error is not None:
        return False, visitor.error
    return True, "ok"


def _exec_candidate_in_namespace(code_str: str) -> tuple[object | None, str | None]:
    """
    Exec the validated, unmodified code string in a restricted namespace.
    Returns (shaping_reward_callable_or_None, error_message_or_None).
    """
    namespace: dict = {}
    try:
        exec(compile(code_str, "<candidate>", "exec"),
             {"__builtins__": _SAFE_BUILTINS, "math": math}, namespace)
    except Exception as e:
        return None, f"exec failed: {e}"

    fn = namespace.get("shaping_reward")
    if fn is None or not callable(fn):
        return None, "no `shaping_reward` function defined"
    return fn, None


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
                    value = fn(ego, road, sample_info)
                except Exception as e:
                    passed = False
                    message = (
                        f"runtime error on trial {i} "
                        f"(n_overtakes={sample_info['n_overtakes']}): {e}"
                    )
                    break

                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    passed = False
                    message = (
                        f"invalid return value on trial {i} "
                        f"(probe {probe_idx}, n_overtakes={sample_info['n_overtakes']}): "
                        f"{value!r}"
                    )
                    break

                if abs(value) > MAX_ABS_VALUE:
                    passed = False
                    message = (
                        f"return value out of range on trial {i} "
                        f"(probe {probe_idx}, n_overtakes={sample_info['n_overtakes']}): "
                        f"{value}"
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
    fn, error = _exec_candidate_in_namespace(code_str)
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
    env_factory.py will import. Returns (passed: bool, message: str).
    """
    passed, message = validate_candidate_ast(code_str)
    if not passed:
        return False, message

    # AST rejection is necessary but not sufficient: in-process exec() can still
    # be escaped by determined adversarial code. Run the runtime probe in a
    # separate subprocess so a failing/exploitative candidate does not inherit
    # the parent's full filesystem/network privileges during the smoke test.
    return _run_runtime_probe_subprocess(code_str, n_trials)
