"""
shaping_call.py

Per-step timeout wrapper for candidate shaping_reward calls.

A candidate that passes the short smoke-test probe can still hang on later
steps (e.g. infinite loop triggered only after many calls). Thread-based
timeout works on Windows (no SIGALRM). Note: timed-out threads cannot be
forcibly killed in CPython — a pathological candidate may leave a stray
background thread until that thread finishes on its own.

When enough timed-out calls saturate the shared executor (all workers
leaked), the pool is replaced with a fresh ThreadPoolExecutor so later
calls are not permanently stuck returning (0.0, {}).

Returns (total: float, components: dict) — components empty on degrade or
when the candidate uses the legacy bare-float contract.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Callable

from eureka.eureka_config import SHAPING_FN_EXECUTOR_WORKERS, SHAPING_FN_TIMEOUT_S
from eureka.logging_utils import get_logger
from eureka.sandbox import normalize_shaping_output

logger = get_logger(__name__)

_lock = threading.Lock()
_executor: ThreadPoolExecutor | None = None
_max_workers = SHAPING_FN_EXECUTOR_WORKERS
_leaked_futures: set[Future] = set()


def _create_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=_max_workers, thread_name_prefix="shaping_fn")


def _ensure_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = _create_executor()
    return _executor


def _prune_completed_leaks() -> None:
    global _leaked_futures
    if _leaked_futures:
        _leaked_futures = {future for future in _leaked_futures if not future.done()}


def _shaping_fn_context(shaping_fn: Callable, candidate_info: dict) -> dict:
    return {
        "shaping_fn": getattr(shaping_fn, "__name__", "shaping_reward"),
        "candidate_module": candidate_info.get("candidate_module")
        or candidate_info.get("module_path"),
    }


def _log_timeout(timeout_s: float, context: dict, leaked_count: int) -> None:
    logger.warning(
        "shaping_reward call timed out",
        extra={
            "event": "shaping_call_timeout",
            "timeout_s": timeout_s,
            "outstanding_leaked_threads": leaked_count,
            **{key: value for key, value in context.items() if value is not None},
        },
    )


def _replace_executor(leaked_count: int, context: dict) -> None:
    global _executor, _leaked_futures
    logger.warning(
        "shaping executor replaced due to leaked worker saturation",
        extra={
            "event": "shaping_executor_replaced",
            "leaked_threads": leaked_count,
            "max_workers": _max_workers,
            **{key: value for key, value in context.items() if value is not None},
        },
    )
    _executor = _create_executor()
    _leaked_futures = set()


def reset_executor_for_tests() -> None:
    """Reset singleton executor state (unit tests only)."""
    global _executor, _leaked_futures
    with _lock:
        _executor = None
        _leaked_futures = set()


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
    context = _shaping_fn_context(shaping_fn, candidate_info)

    with _lock:
        _prune_completed_leaks()
        executor = _ensure_executor()

    future = executor.submit(shaping_fn, ego, road, candidate_info)
    try:
        raw = future.result(timeout=timeout_s)
    except FuturesTimeoutError:
        with _lock:
            _leaked_futures.add(future)
            _prune_completed_leaks()
            leaked_count = len(_leaked_futures)
            _log_timeout(timeout_s, context, leaked_count)
            if leaked_count >= _max_workers:
                _replace_executor(leaked_count, context)
        return 0.0, {}
    except Exception:
        return 0.0, {}

    try:
        return normalize_shaping_output(raw)
    except ValueError:
        return 0.0, {}
