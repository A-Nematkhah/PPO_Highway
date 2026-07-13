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
    value, components = call_shaping_fn(
        _fast_fn, None, None, {"n_overtakes": 3}, timeout_s=0.5
    )
    assert value == pytest.approx(0.3)
    assert components == {}


def test_call_shaping_fn_returns_zero_on_timeout():
    value, components = call_shaping_fn(_slow_fn, None, None, {}, timeout_s=0.05)
    assert value == 0.0
    assert components == {}


def test_call_shaping_fn_returns_zero_on_exception():
    def _boom(ego, road, info):
        raise RuntimeError("boom")

    value, components = call_shaping_fn(_boom, None, None, {}, timeout_s=0.5)
    assert value == 0.0
    assert components == {}


def test_call_shaping_fn_propagates_component_dict():
    def _with_components(ego, road, info):
        return 0.3, {"a": 0.1, "b": 0.2}

    value, components = call_shaping_fn(
        _with_components, None, None, {}, timeout_s=0.5
    )
    assert value == pytest.approx(0.3)
    assert components == {"a": 0.1, "b": 0.2}


def test_call_shaping_fn_degrades_malformed_tuple():
    def _bad_arity(ego, road, info):
        return (0.1, 0.2, 0.3)

    value, components = call_shaping_fn(_bad_arity, None, None, {}, timeout_s=0.5)
    assert value == 0.0
    assert components == {}

    def _bad_components(ego, road, info):
        return (0.1, [0.05, 0.05])

    value, components = call_shaping_fn(_bad_components, None, None, {}, timeout_s=0.5)
    assert value == 0.0
    assert components == {}
