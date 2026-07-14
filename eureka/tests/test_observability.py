"""Unit tests for structured logging and telemetry helpers."""

import json
import logging

from eureka.logging_utils import (
    JsonLogFormatter,
    TrainProgressTable,
    get_logger,
    setup_logging,
)
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


def test_train_progress_table_prints_aligned_rows(capsys):
    table = TrainProgressTable("gen0_cand0", 97)
    table.add_row(
        update=1,
        global_step=512,
        fps=60,
        mean_return=22.25,
        crash_rate_pct=100.0,
        mean_speed=25.2,
        mean_overtakes=0.6,
    )
    table.add_row(
        update=9,
        global_step=4608,
        fps=61,
        mean_return=79.24,
        crash_rate_pct=70.0,
        mean_speed=22.4,
        mean_overtakes=1.2,
    )
    output = capsys.readouterr().out
    assert "[gen0_cand0] training" in output
    assert "upd    step  fps  return  crash  speed  otk" in output
    assert "1/97" in output
    assert "9/97" in output
    assert "22.2" in output
    assert "100%" in output


def test_console_formatter_uses_compact_context_not_raw_dict():
    from eureka.logging_utils import _ConsoleFormatter

    setup_logging()
    formatter = _ConsoleFormatter()
    record = logging.LogRecord(
        name="eureka.llm_reward_designer",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="LLM code received",
        args=(),
        exc_info=None,
    )
    record.event = "llm_call_success"
    record.generation = 0
    record.index = 2
    record.lines = 52
    record.duration_s = 2.14
    line = formatter.format(record)
    assert "LLM code received" in line
    assert "52 lines" in line
    assert "2.1s" in line
    assert "{" not in line


def test_json_log_file_is_optional_and_console_stays_readable(tmp_path, monkeypatch):
    monkeypatch.setenv("EUREKA_LOG_JSON", "1")
    monkeypatch.setenv("EUREKA_LOG_JSON_PATH", str(tmp_path / "run.jsonl"))

    import eureka.logging_utils as logging_utils

    logging_utils._configured = False
    setup_logging()

    logger = get_logger("loop")
    logger.info("hello", extra={"event": "unit_test", "generation": 0})

    json_lines = (tmp_path / "run.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(json_lines) == 1
    payload = json.loads(json_lines[0])
    assert payload["message"] == "hello"
    assert payload["event"] == "unit_test"

    logging_utils._configured = False


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
