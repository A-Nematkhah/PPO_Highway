"""Unit tests for structured logging and telemetry helpers."""

import json
import logging

from eureka.logging_utils import JsonLogFormatter, get_logger, setup_logging
from eureka.telemetry import Telemetry


def test_json_log_formatter_emits_parseable_line():
    setup_logging()
    formatter = JsonLogFormatter()
    record = logging.LogRecord(
        name="eureka.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.event = "unit_test"
    record.generation = 0
    line = formatter.format(record)
    payload = json.loads(line)
    assert payload["message"] == "hello"
    assert payload["event"] == "unit_test"
    assert payload["generation"] == 0


def test_get_logger_returns_eureka_prefixed_logger():
    logger = get_logger("loop")
    assert logger.name == "eureka.loop"


def test_telemetry_writes_jsonl(tmp_path):
    path = tmp_path / "metrics.jsonl"
    tel = Telemetry(path=str(path))
    tel.record("smoke_test", generation=0, candidate=0, passed=True, duration_s=1.2)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["event"] == "smoke_test"
    assert row["passed"] is True
    assert row["duration_s"] == 1.2


def test_telemetry_timed_context_manager(tmp_path):
    path = tmp_path / "metrics.jsonl"
    tel = Telemetry(path=str(path))
    with tel.timed("train", generation=1, candidate=2) as ctx:
        ctx["checkpoint"] = "x.pt"
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["event"] == "train"
    assert row["checkpoint"] == "x.pt"
    assert "duration_s" in row
    assert row["duration_s"] >= 0
