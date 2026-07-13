"""Unit tests for per-step shaping_reward timeout."""

import time

import pytest

from eureka.shaping_call import call_shaping_fn


def _fast_fn(ego, road, info):
    return float(info.get("n_overtakes", 0)) * 0.1


def _slow_fn(ego, road, info):
    time.sleep(1.0)
    return 1.0


def test_call_shaping_fn_returns_value_when_fast():
    assert call_shaping_fn(_fast_fn, None, None, {"n_overtakes": 3}, timeout_s=0.5) == pytest.approx(0.3)


def test_call_shaping_fn_returns_zero_on_timeout():
    assert call_shaping_fn(_slow_fn, None, None, {}, timeout_s=0.05) == 0.0


def test_call_shaping_fn_returns_zero_on_exception():
    def _boom(ego, road, info):
        raise RuntimeError("boom")

    assert call_shaping_fn(_boom, None, None, {}, timeout_s=0.5) == 0.0
