"""Unit tests for per-step shaping_reward timeout and executor recovery."""

import logging
import threading
import time

import pytest

from eureka import shaping_call as sc
from eureka.shaping_call import call_shaping_fn, reset_executor_for_tests


@pytest.fixture(autouse=True)
def _isolated_executor():
    reset_executor_for_tests()
    yield
    reset_executor_for_tests()


def _fast_fn(ego, road, info):
    return float(info.get("n_overtakes", 0)) * 0.1


def _slow_fn(ego, road, info):
    time.sleep(1.0)
    return 1.0


def _hang_fn(ego, road, info):
    time.sleep(60.0)
    return 1.0


def test_call_shaping_fn_returns_value_when_fast():
    value, components = call_shaping_fn(
        _fast_fn, None, None, {"n_overtakes": 3}, timeout_s=0.5
    )
    assert value == pytest.approx(0.3)
    assert components == {}


def test_call_shaping_fn_returns_zero_on_timeout(caplog):
    # caplog.at_level(level, logger=name) already attaches caplog's handler
    # to that named logger for the duration of the block - this is pytest's
    # documented workaround for loggers that don't propagate to root (see
    # eureka.logging_utils.setup_logging(), which sets propagate=False on
    # the "eureka" logger). Manually addHandler(caplog.handler) here too
    # would register the SAME handler twice on the SAME logger, so every
    # record gets appended to caplog.records twice - which is exactly what
    # was causing this test to intermittently see 2 (or 4, run alongside
    # test_executor_replaced_after_leak_saturation below) log entries where
    # only 1 (or 2) actually occurred.
    with caplog.at_level(logging.WARNING, logger="eureka.shaping_call"):
        value, components = call_shaping_fn(
            _slow_fn, None, None, {}, timeout_s=0.05
        )
    assert value == 0.0
    assert components == {}
    timeout_logs = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "shaping_call_timeout"
    ]
    assert len(timeout_logs) == 1


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


def test_executor_replaced_after_leak_saturation(monkeypatch, caplog):
    monkeypatch.setattr(sc, "_max_workers", 2)

    # See test_call_shaping_fn_returns_zero_on_timeout above for why the
    # manual addHandler/removeHandler pair is not needed (and actively
    # harmful): caplog.at_level(level, logger=...) already attaches its
    # handler to that logger for the block's duration.
    with caplog.at_level(logging.WARNING, logger="eureka.shaping_call"):
        call_shaping_fn(_hang_fn, None, None, {}, timeout_s=0.01)
        call_shaping_fn(_hang_fn, None, None, {}, timeout_s=0.01)
        value, components = call_shaping_fn(
            _fast_fn, None, None, {"n_overtakes": 4}, timeout_s=0.5
        )

    assert value == pytest.approx(0.4)
    assert components == {}

    timeout_logs = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "shaping_call_timeout"
    ]
    replacement_logs = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "shaping_executor_replaced"
    ]
    assert len(timeout_logs) == 2
    assert len(replacement_logs) == 1
    assert replacement_logs[0].leaked_threads == 2
    assert replacement_logs[0].max_workers == 2


def test_concurrent_calls_do_not_corrupt_executor_state(monkeypatch, caplog):
    monkeypatch.setattr(sc, "_max_workers", 4)
    errors: list[BaseException] = []
    results: list[tuple[float, dict]] = []
    barrier = threading.Barrier(12)

    def _worker():
        try:
            barrier.wait(timeout=5.0)
            results.append(
                call_shaping_fn(_fast_fn, None, None, {"n_overtakes": 2}, timeout_s=0.5)
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
        assert not thread.is_alive()

    assert errors == []
    assert len(results) == 12
    assert all(value == pytest.approx(0.2) for value, _ in results)
    assert all(components == {} for _, components in results)

    replacement_logs = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "shaping_executor_replaced"
    ]
    assert replacement_logs == []