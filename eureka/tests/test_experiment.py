"""Unit tests for eureka/experiment.py."""

import json

import pytest

from eureka.experiment import ExperimentRun, build_experiment_config_snapshot
from eureka.run_metadata import collect_run_metadata


def test_start_creates_first_run_as_0001(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = ExperimentRun.start()
    assert run.run_name == "run_0001"
    assert run.run_dir.is_dir()
    for subdir in (run.plots_dir, run.checkpoints_dir, run.reflection_dir, run.reward_candidates_dir):
        assert subdir.is_dir()


def test_start_increments_across_calls(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run1 = ExperimentRun.start()
    run2 = ExperimentRun.start()
    run3 = ExperimentRun.start()
    assert [r.run_name for r in (run1, run2, run3)] == ["run_0001", "run_0002", "run_0003"]


def test_start_ignores_unrelated_entries_under_runs_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "not_a_run_dir").mkdir()
    (tmp_path / "runs" / "somefile.txt").write_text("x")
    run = ExperimentRun.start()
    assert run.run_name == "run_0001"


def test_start_resumes_numbering_from_existing_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs" / "run_0005").mkdir(parents=True)
    run = ExperimentRun.start()
    assert run.run_name == "run_0006"


def test_write_config_snapshot_round_trips(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = ExperimentRun.start()
    config = build_experiment_config_snapshot()
    run.write_config_snapshot(config)

    loaded = json.loads(run.config_path.read_text(encoding="utf-8"))
    assert loaded["n_generations"] == config["n_generations"]
    assert loaded["multi_objective_mode"] == config["multi_objective_mode"]


def test_write_metadata_includes_run_identity_and_returns_same_payload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = ExperimentRun.start()
    metadata = collect_run_metadata("test-model")

    payload = run.write_metadata(metadata, execution_time_s=12.3456)

    assert payload["run_id"] == run.run_id
    assert payload["run_name"] == run.run_name
    assert payload["execution_time_s"] == pytest.approx(12.346, abs=1e-3)
    assert payload["llm_model"] == "test-model"

    on_disk = json.loads(run.metadata_path.read_text(encoding="utf-8"))
    assert on_disk == payload


def test_archive_candidate_code_writes_exact_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = ExperimentRun.start()
    code = "def shaping_reward(ego, road, info):\n    return 0.0\n"

    path = run.archive_candidate_code("gen0_cand0", code)

    assert path == run.reward_candidates_dir / "gen0_cand0.py"
    assert path.read_text(encoding="utf-8") == code


def test_archive_final_reward_writes_to_run_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = ExperimentRun.start()
    code = "def shaping_reward(ego, road, info):\n    return 0.5\n"

    path = run.archive_final_reward(code)

    assert path == run.final_reward_path
    assert path.read_text(encoding="utf-8") == code


def test_archive_checkpoint_copies_existing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = ExperimentRun.start()
    source = tmp_path / "some_checkpoint.pt"
    source.write_bytes(b"fake torch bytes")

    dest = run.archive_checkpoint(str(source))

    assert dest == run.checkpoints_dir / "some_checkpoint.pt"
    assert dest.read_bytes() == b"fake torch bytes"


def test_archive_checkpoint_missing_source_returns_none_without_raising(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = ExperimentRun.start()
    assert run.archive_checkpoint("does/not/exist.pt") is None


def test_archive_reflection_prompt_names_file_by_role(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = ExperimentRun.start()

    path = run.archive_reflection_prompt(2, 1, "safest", "some prompt text")

    assert path.name == "gen2_req1_safest.txt"
    assert path.read_text(encoding="utf-8") == "some prompt text"


def test_archive_reflection_prompt_without_role(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = ExperimentRun.start()

    path = run.archive_reflection_prompt(0, 0, None, "initial prompt")

    assert path.name == "gen0_req0.txt"


def test_capture_console_writes_to_console_log_and_restores_stdout(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    run = ExperimentRun.start()
    import sys
    original_stdout = sys.stdout

    with run.capture_console():
        print("hello from inside the run")

    assert sys.stdout is original_stdout  # restored after the context exits
    content = run.console_log_path.read_text(encoding="utf-8")
    assert "hello from inside the run" in content

    # still visible on the real stdout too (tee, not a redirect)
    captured = capsys.readouterr()
    assert "hello from inside the run" in captured.out


def test_capture_console_captures_structured_logger_output_too(tmp_path, monkeypatch):
    """Regression test: the console handler is normally bound to whatever
    sys.stdout was at module-import time (before capture_console can
    possibly run) - capture_console must force a rebind, or structured log
    lines silently never reach console.log."""
    monkeypatch.chdir(tmp_path)
    run = ExperimentRun.start()

    from eureka.logging_utils import get_logger
    logger = get_logger("test_capture_console_logger")

    with run.capture_console():
        logger.info("structured log line", extra={"event": "test_event"})

    content = run.console_log_path.read_text(encoding="utf-8")
    assert "structured log line" in content


def test_build_experiment_config_snapshot_is_json_serializable():
    config = build_experiment_config_snapshot()
    # round-trips cleanly - would raise TypeError on anything non-serializable
    json.loads(json.dumps(config))
    assert "n_generations" in config
    assert "objective_specs" in config
    assert isinstance(config["objective_specs"], list)
