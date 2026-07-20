"""Unit tests for eureka/report_html.py."""

import json

import pytest

from eureka.report_html import (
    compute_execution_stats,
    generate_html_report,
    generation_reason,
)


def _metadata(run_name="run_0001"):
    return {
        "run_name": run_name,
        "timestamp_utc": "2026-07-19T12:00:00+00:00",
        "git_commit": "abcdef1234567890",
        "git_dirty": False,
        "os_name": "Linux",
        "python_version": "3.12.3",
        "cpu_model": "x86_64",
        "cpu_cores_logical": 8,
        "ram_total_gb": 16.0,
        "llm_model": "test-model",
    }


def _config():
    return {
        "n_generations": 3,
        "k_candidates": 4,
        "train_steps_per_candidate": 50000,
        "n_eval_episodes": 30,
        "multi_objective_mode": "pareto",
        "confirmation_seeds": [10000, 20000],
    }


def _candidate(candidate_id, module_path, generation, crash, speed, overtakes, rank=0):
    return {
        "candidate_id": candidate_id,
        "module_path": module_path,
        "generation": generation,
        "pareto_rank": rank,
        "crowding_distance": 1.0,
        "metrics": {
            "crash_rate": crash,
            "mean_speed": speed,
            "mean_overtakes": overtakes,
            "mean_raw_return": 10.0,
        },
    }


def _full_log():
    return [
        {
            "generation": 0,
            "selection_mode": "pareto",
            "results": [
                dict(_candidate("a", "eureka.candidates.gen0_cand0", 0, 0.1, 20.0, 1.0),
                     candidate_index=0, source="llm", legacy_fitness=1.1),
                dict(_candidate("b", "eureka.candidates.gen0_cand1", 0, 0.3, 25.0, 2.0),
                     candidate_index=1, source="llm", legacy_fitness=0.9),
            ],
            "generation_front_candidate_ids": ["a", "b"],
            "representative_id": "a",
            "legacy_scalar_winner_id": "a",
            "legacy_scalar_winner_on_front": True,
        },
    ]


# --------------------------------------------------------------------------- #
# generation_reason
# --------------------------------------------------------------------------- #


def test_generation_reason_all_candidates_failed():
    record = {"all_candidates_failed": True}
    assert "no candidates survived" in generation_reason(record).lower()


def test_generation_reason_no_results():
    record = {"results": []}
    assert "smoke test" in generation_reason(record).lower()


def test_generation_reason_shadow_mode_on_front():
    record = {"selection_mode": "shadow", "results": [{}], "legacy_scalar_winner_on_front": True}
    reason = generation_reason(record)
    assert "legacy scalar fitness" in reason.lower()
    assert "NOT on the Pareto front" not in reason


def test_generation_reason_shadow_mode_off_front_adds_note():
    record = {"selection_mode": "shadow", "results": [{}], "legacy_scalar_winner_on_front": False}
    reason = generation_reason(record)
    assert "NOT on the Pareto front" in reason


def test_generation_reason_pareto_mode():
    record = {"selection_mode": "pareto", "results": [{}]}
    assert "knee" in generation_reason(record).lower()


# --------------------------------------------------------------------------- #
# compute_execution_stats
# --------------------------------------------------------------------------- #


def test_compute_execution_stats_aggregates_by_event(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    rows = [
        {"event": "train", "duration_s": 10.0},
        {"event": "train", "duration_s": 5.0},
        {"event": "eval", "duration_s": 2.0},
        {"event": "llm_generation", "duration_s": 1.5},
        {"event": "candidate_complete", "duration_s": 999.0},  # not aggregated
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    stats = compute_execution_stats(path, total_runtime_s=123.0)

    assert stats["train_time_s"] == pytest.approx(15.0)
    assert stats["eval_time_s"] == pytest.approx(2.0)
    assert stats["llm_time_s"] == pytest.approx(1.5)
    assert stats["total_runtime_s"] == 123.0


def test_compute_execution_stats_missing_file_still_returns_dict(tmp_path):
    stats = compute_execution_stats(tmp_path / "does_not_exist.jsonl", total_runtime_s=5.0)
    assert stats["train_time_s"] == 0.0
    assert stats["total_runtime_s"] == 5.0


def test_compute_execution_stats_skips_malformed_lines(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    path.write_text('{"event": "train", "duration_s": 3.0}\nnot valid json\n', encoding="utf-8")
    stats = compute_execution_stats(path)
    assert stats["train_time_s"] == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# generate_html_report - structure and safety
# --------------------------------------------------------------------------- #


def test_generate_html_report_contains_all_required_sections(tmp_path):
    archive = [_candidate("a", "eureka.candidates.gen0_cand0", 0, 0.1, 20.0, 1.0)]
    out = generate_html_report(
        run_dir=tmp_path, full_log=_full_log(), archive=archive, representative_id="a",
        metadata=_metadata(), config=_config(), execution_stats={"total_runtime_s": 42.0},
    )
    content = out.read_text(encoding="utf-8")

    for section in (
        "Experiment Summary", "Training Summary", "Generation Summary",
        "Pareto Archive", "Reflection", "Plots", "Execution Statistics",
    ):
        assert f"<h2>{section}</h2>" in content

    assert "run_0001" in content
    assert "winner-row" in content  # candidate "a" is the representative


def test_generate_html_report_escapes_malicious_reflection_text(tmp_path):
    """Reflection prompt text is raw LLM-adjacent text rendered verbatim
    inside <pre> blocks and must never be trusted as literal HTML."""
    reflection_dir = tmp_path / "reflection"
    reflection_dir.mkdir()
    malicious = "<script>alert('xss')</script>"
    (reflection_dir / "gen0_req0.txt").write_text(malicious, encoding="utf-8")

    out = generate_html_report(
        run_dir=tmp_path, full_log=_full_log(), archive=[], representative_id=None,
        metadata=_metadata(), config=_config(), execution_stats={},
        reflection_dir=reflection_dir,
    )
    content = out.read_text(encoding="utf-8")

    assert "<script>alert" not in content
    assert "&lt;script&gt;" in content


def test_generate_html_report_handles_empty_archive_and_log(tmp_path):
    out = generate_html_report(
        run_dir=tmp_path, full_log=[], archive=[], representative_id=None,
        metadata=_metadata(), config=_config(), execution_stats={},
    )
    content = out.read_text(encoding="utf-8")
    assert "Final archive is empty" in content
    assert "No generations recorded" in content


def test_generate_html_report_embeds_plot_as_base64(tmp_path):
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    # minimal valid 1x1 PNG
    png_bytes = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000a49444154789c6360000002000100faa5b2b8000000004945"
        "4e44ae426082"
    )
    (plots_dir / "run_summary_test.png").write_bytes(png_bytes)

    out = generate_html_report(
        run_dir=tmp_path, full_log=[], archive=[], representative_id=None,
        metadata=_metadata(), config=_config(), execution_stats={},
        plots_dir=plots_dir,
    )
    content = out.read_text(encoding="utf-8")
    assert "data:image/png;base64," in content


def test_generate_html_report_no_plots_dir_shows_placeholder(tmp_path):
    out = generate_html_report(
        run_dir=tmp_path, full_log=[], archive=[], representative_id=None,
        metadata=_metadata(), config=_config(), execution_stats={},
    )
    content = out.read_text(encoding="utf-8")
    assert "No plots were generated" in content


def test_generate_html_report_respects_out_path_override(tmp_path):
    custom_path = tmp_path / "custom_report.html"
    out = generate_html_report(
        run_dir=tmp_path, full_log=[], archive=[], representative_id=None,
        metadata=_metadata(), config=_config(), execution_stats={},
        out_path=custom_path,
    )
    assert out == custom_path
    assert custom_path.is_file()
