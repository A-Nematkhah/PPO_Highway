"""
shaping_call.py

Per-step timeout wrapper for candidate shaping_reward calls.

A candidate that passes the short smoke-test probe can still hang on later
steps (e.g. infinite loop triggered only after many calls). Thread-based
timeout works on Windows (no SIGALRM). Note: timed-out threads cannot be
forcibly killed in CPython — a pathological candidate may leave a stray
background thread until the worker process exits.

Returns (total: float, components: dict) — components empty on degrade or
when the candidate uses the legacy bare-float contract.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Callable

from eureka.eureka_config import SHAPING_FN_TIMEOUT_S
from eureka.sandbox import normalize_shaping_output

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="shaping_fn")


def call_shaping_fn(
    shaping_fn: Callable,
    ego,
    road,
    candidate_info: dict,
    timeout_s: float = SHAPING_FN_TIMEOUT_S,
) -> tuple[float, dict]:
    """
    Invoke shaping_fn with a per-call timeout. Returns (0.0, {}) on timeout,
    exception, or invalid output (same degrade semantics as
    CandidateRewardWrapper).
    """
    future = _executor.submit(shaping_fn, ego, road, candidate_info)
    try:
        raw = future.result(timeout=timeout_s)
    except FuturesTimeoutError:
        return 0.0, {}
    except Exception:
        return 0.0, {}

    try:
        return normalize_shaping_output(raw)
    except ValueError:
        return 0.0, {}
