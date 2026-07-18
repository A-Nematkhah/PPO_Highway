"""
Unit tests for eureka/evaluate_cli.py.

Deliberately does NOT exercise _build_render_env / _load_policy /
_run_render_episode / _save_frames_as_gif directly: those need a real
torch model, a real highway-env instance, and (for --render-live) a real
display or SDL fallback - they're integration-shaped, not unit-shaped, the
same reasoning that already excludes eureka/loop.py, train_candidate.py,
smoke_test.py, and env_factory.py from this project's coverage gate (see
.coveragerc). Everything else in this module - path resolution, candidate
discovery, CLI validation, and per-candidate orchestration - is pure logic
and is covered here by mocking evaluate_candidate() and render_candidate()
at the boundary.
"""

import pytest

import eureka.evaluate_cli as cli


# --------------------------------------------------------------------------- #
# CandidateRef
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,expected_short,expected_module",
    [
        ("gen2_cand5", "gen2_cand5", "eureka.candidates.gen2_cand5"),
        ("eureka.candidates.gen2_cand5", "gen2_cand5", "eureka.candidates.gen2_cand5"),
        ("gen2_cand5.py", "gen2_cand5", "eureka.candidates.gen2_cand5"),
        ("eureka.candidates.gen2_cand5.py", "gen2_cand5", "eureka.candidates.gen2_cand5"),
        ("  gen2_cand5  ", "gen2_cand5", "eureka.candidates.gen2_cand5"),
    ],
)
def test_candidate_ref_from_raw_normalizes_all_accepted_forms(raw, expected_short, expected_module):
    ref = cli.CandidateRef.from_raw(raw)
    assert ref.short_name == expected_short
    assert ref.module_path == expected_module
    assert ref.source_path == cli.CANDIDATES_DIR / f"{expected_short}.py"
    assert ref.default_checkpoint == cli.CHECKPOINTS_DIR / f"{expected_short}.pt"


def test_candidate_ref_resolve_checkpoint_uses_override_when_given():
    ref = cli.CandidateRef.from_raw("gen0_cand0")
    assert ref.resolve_checkpoint(None) == ref.default_checkpoint
    assert ref.resolve_checkpoint("some/other.pt") == cli.Path("some/other.pt")


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def _make_candidate_files(base, short_name, trained=True, rejected=False):
    candidates_dir = base / "eureka" / "candidates"
    checkpoints_dir = base / "eureka" / "checkpoints"
    rejected_dir = candidates_dir / "rejected"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if rejected:
        rejected_dir.mkdir(parents=True, exist_ok=True)
        (rejected_dir / f"{short_name}.py").write_text("def shaping_reward(): pass")
        return

    (candidates_dir / f"{short_name}.py").write_text("def shaping_reward(): pass")
    if trained:
        (checkpoints_dir / f"{short_name}.pt").write_bytes(b"fake")


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    """Point the module's directory constants at an isolated tmp_path tree,
    rather than chdir - keeps tests independent of cwd side effects."""
    candidates_dir = tmp_path / "eureka" / "candidates"
    monkeypatch.setattr(cli, "CANDIDATES_DIR", candidates_dir)
    monkeypatch.setattr(cli, "REJECTED_DIR", candidates_dir / "rejected")
    monkeypatch.setattr(cli, "CHECKPOINTS_DIR", tmp_path / "eureka" / "checkpoints")
    return tmp_path


def test_discover_trained_candidates_requires_both_checkpoint_and_source(isolated_dirs):
    _make_candidate_files(isolated_dirs, "gen0_cand0", trained=True)
    _make_candidate_files(isolated_dirs, "gen0_cand1", trained=False)

    trained = cli.discover_trained_candidates()

    assert trained == ["gen0_cand0"]


def test_discover_trained_candidates_empty_when_checkpoints_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "CHECKPOINTS_DIR", tmp_path / "does_not_exist")
    assert cli.discover_trained_candidates() == []


def test_discover_all_candidate_names_separates_rejected_from_saved(isolated_dirs):
    _make_candidate_files(isolated_dirs, "gen0_cand0", trained=True)
    _make_candidate_files(isolated_dirs, "gen0_cand1", rejected=True)

    saved, rejected = cli.discover_all_candidate_names()

    assert saved == {"gen0_cand0"}
    assert rejected == {"gen0_cand1"}


# --------------------------------------------------------------------------- #
# CLI parsing / validation (build_arg_parser + build_run_plan)
# --------------------------------------------------------------------------- #


def _plan_for(argv, isolated_dirs):
    parser = cli.build_arg_parser()
    args = parser.parse_args(argv)
    return cli.build_run_plan(parser, args)


def test_checkpoint_override_rejected_with_multiple_candidates(isolated_dirs):
    parser = cli.build_arg_parser()
    args = parser.parse_args(["gen0", "gen1", "--checkpoint", "x.pt"])
    with pytest.raises(SystemExit):
        cli.build_run_plan(parser, args)


def test_render_live_rejected_with_multiple_candidates(isolated_dirs):
    parser = cli.build_arg_parser()
    args = parser.parse_args(["gen0", "gen1", "--render-live"])
    with pytest.raises(SystemExit):
        cli.build_run_plan(parser, args)


def test_render_live_rejected_with_all_flag(isolated_dirs):
    parser = cli.build_arg_parser()
    args = parser.parse_args(["--all", "--render-live"])
    with pytest.raises(SystemExit):
        cli.build_run_plan(parser, args)


def test_render_episodes_must_be_positive(isolated_dirs):
    parser = cli.build_arg_parser()
    args = parser.parse_args(["gen0", "--render-episodes", "0"])
    with pytest.raises(SystemExit):
        cli.build_run_plan(parser, args)


def test_render_only_implies_render_job_without_render_flag(isolated_dirs):
    plan = _plan_for(["gen0", "--render-only"], isolated_dirs)
    assert plan.render_only is True
    assert plan.render_job is not None
    assert plan.render_job.live is False


def test_render_live_implies_render_job(isolated_dirs):
    plan = _plan_for(["gen0", "--render-live"], isolated_dirs)
    assert plan.render_job is not None
    assert plan.render_job.live is True


def test_no_render_flags_means_no_render_job(isolated_dirs):
    plan = _plan_for(["gen0"], isolated_dirs)
    assert plan.render_job is None


def test_all_flag_discovers_trained_candidates(isolated_dirs):
    _make_candidate_files(isolated_dirs, "gen0_cand0", trained=True)
    _make_candidate_files(isolated_dirs, "gen1_cand0", trained=True)

    plan = _plan_for(["--all"], isolated_dirs)

    assert sorted(plan.names) == ["gen0_cand0", "gen1_cand0"]


def test_no_candidates_and_no_all_falls_back_to_discovery(isolated_dirs):
    _make_candidate_files(isolated_dirs, "gen0_cand0", trained=True)
    plan = _plan_for([], isolated_dirs)
    assert plan.names == ["gen0_cand0"]


# --------------------------------------------------------------------------- #
# process_candidate - skip paths
# --------------------------------------------------------------------------- #


def _default_plan(**overrides):
    defaults = dict(names=[], episodes=5, checkpoint_override=None, render_only=False, render_job=None)
    defaults.update(overrides)
    return cli.RunPlan(**defaults)


def test_process_candidate_skips_missing_source(isolated_dirs, capsys):
    result = cli.process_candidate("nonexistent", _default_plan())
    assert result.succeeded is False
    assert result.outcome is None
    assert "no source file" in capsys.readouterr().out


def test_process_candidate_skips_rejected_candidate(isolated_dirs, capsys):
    _make_candidate_files(isolated_dirs, "gen0_cand0", rejected=True)
    result = cli.process_candidate("gen0_cand0", _default_plan())
    assert result.succeeded is False
    assert "rejected by the sandbox" in capsys.readouterr().out


def test_process_candidate_skips_missing_checkpoint(isolated_dirs, capsys):
    _make_candidate_files(isolated_dirs, "gen0_cand0", trained=False)
    result = cli.process_candidate("gen0_cand0", _default_plan())
    assert result.succeeded is False
    assert "no checkpoint at" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# process_candidate - success / failure paths (evaluate_candidate mocked)
# --------------------------------------------------------------------------- #


_FAKE_METRICS = {
    "crash_rate": 0.1,
    "mean_speed": 20.0,
    "mean_overtakes": 1.0,
    "mean_raw_return": 5.0,
}


def test_process_candidate_success_path_produces_outcome(isolated_dirs, monkeypatch, capsys):
    _make_candidate_files(isolated_dirs, "gen0_cand0", trained=True)
    monkeypatch.setattr(cli, "evaluate_candidate", lambda checkpoint, module_path, n_episodes: _FAKE_METRICS)

    result = cli.process_candidate("gen0_cand0", _default_plan(episodes=3))

    assert result.succeeded is True
    assert result.outcome is not None
    assert result.outcome.metrics == _FAKE_METRICS
    assert "legacy_fitness" in capsys.readouterr().out


def test_process_candidate_eval_exception_is_caught_and_reported(isolated_dirs, monkeypatch, capsys):
    _make_candidate_files(isolated_dirs, "gen0_cand0", trained=True)

    def _boom(checkpoint, module_path, n_episodes):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "evaluate_candidate", _boom)

    result = cli.process_candidate("gen0_cand0", _default_plan())

    assert result.succeeded is False
    assert result.outcome is None
    assert "FAILED to evaluate" in capsys.readouterr().out


def test_process_candidate_render_only_skips_eval_entirely(isolated_dirs, monkeypatch, capsys):
    _make_candidate_files(isolated_dirs, "gen0_cand0", trained=True)

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("evaluate_candidate should not be called with --render-only")

    monkeypatch.setattr(cli, "evaluate_candidate", _should_not_be_called)
    render_calls = []
    monkeypatch.setattr(cli, "render_candidate", lambda ref, checkpoint, job: render_calls.append(ref.short_name))

    job = cli.RenderJob(episodes=1, live=False, render_dir=isolated_dirs / "renders", fps=5)
    result = cli.process_candidate("gen0_cand0", _default_plan(render_only=True, render_job=job))

    assert result.succeeded is True
    assert result.outcome is None  # render-only never produces numeric metrics
    assert render_calls == ["gen0_cand0"]


def test_process_candidate_invokes_render_job_after_normal_eval(isolated_dirs, monkeypatch):
    _make_candidate_files(isolated_dirs, "gen0_cand0", trained=True)
    monkeypatch.setattr(cli, "evaluate_candidate", lambda checkpoint, module_path, n_episodes: _FAKE_METRICS)
    render_calls = []
    monkeypatch.setattr(cli, "render_candidate", lambda ref, checkpoint, job: render_calls.append(ref.short_name))

    job = cli.RenderJob(episodes=2, live=False, render_dir=isolated_dirs / "renders", fps=5)
    result = cli.process_candidate("gen0_cand0", _default_plan(render_job=job))

    assert result.succeeded is True
    assert result.outcome is not None
    assert render_calls == ["gen0_cand0"]


# --------------------------------------------------------------------------- #
# render_candidate - episode-level failure isolation (real render episode mocked out)
# --------------------------------------------------------------------------- #


def test_render_candidate_continues_after_one_episode_fails(isolated_dirs, monkeypatch, capsys):
    ref = cli.CandidateRef.from_raw("gen0_cand0")
    calls = {"n": 0}

    def _fake_run_episode(module_path, checkpoint_path, seed, live):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient render failure")
        return ["frame"]

    monkeypatch.setattr(cli, "_run_render_episode", _fake_run_episode)
    monkeypatch.setattr(cli, "_save_frames_as_gif", lambda frames, out_path, fps: None)

    job = cli.RenderJob(episodes=2, live=False, render_dir=isolated_dirs / "renders", fps=5)
    cli.render_candidate(ref, isolated_dirs / "fake.pt", job)

    out = capsys.readouterr().out
    assert calls["n"] == 2
    assert "RENDER FAILED" in out
    assert "saved render" in out


def test_render_candidate_live_mode_never_saves_a_gif(isolated_dirs, monkeypatch):
    ref = cli.CandidateRef.from_raw("gen0_cand0")
    monkeypatch.setattr(cli, "_run_render_episode", lambda module_path, checkpoint_path, seed, live: None)
    save_calls = []
    monkeypatch.setattr(cli, "_save_frames_as_gif", lambda *a, **k: save_calls.append(a))

    job = cli.RenderJob(episodes=1, live=True, render_dir=isolated_dirs / "renders", fps=5)
    cli.render_candidate(ref, isolated_dirs / "fake.pt", job)

    assert save_calls == []


# --------------------------------------------------------------------------- #
# main() - end to end with evaluate_candidate/render_candidate mocked
# --------------------------------------------------------------------------- #


def test_main_list_flag_exits_zero_without_evaluating(isolated_dirs, monkeypatch, capsys):
    monkeypatch.setattr(cli, "evaluate_candidate", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    assert cli.main(["--list"]) == 0
    assert "Trained candidates" in capsys.readouterr().out


def test_main_returns_1_when_no_trained_candidates_found(isolated_dirs):
    assert cli.main([]) == 1


def test_main_returns_0_and_prints_archive_table_for_multiple_candidates(isolated_dirs, monkeypatch, capsys):
    _make_candidate_files(isolated_dirs, "gen0_cand0", trained=True)
    _make_candidate_files(isolated_dirs, "gen1_cand0", trained=True)
    monkeypatch.setattr(cli, "evaluate_candidate", lambda checkpoint, module_path, n_episodes: dict(_FAKE_METRICS))

    exit_code = cli.main(["--all"])

    assert exit_code == 0
    assert "Final Pareto archive" in capsys.readouterr().out


def test_main_returns_1_when_every_candidate_fails(isolated_dirs, monkeypatch):
    _make_candidate_files(isolated_dirs, "gen0_cand0", trained=True)

    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "evaluate_candidate", _boom)

    assert cli.main(["gen0_cand0"]) == 1


def test_main_render_only_returns_0_without_any_eval_calls(isolated_dirs, monkeypatch):
    _make_candidate_files(isolated_dirs, "gen0_cand0", trained=True)

    def _should_not_be_called(*a, **k):
        raise AssertionError("evaluate_candidate must not run under --render-only")

    monkeypatch.setattr(cli, "evaluate_candidate", _should_not_be_called)
    monkeypatch.setattr(cli, "render_candidate", lambda ref, checkpoint, job: None)

    assert cli.main(["gen0_cand0", "--render-only"]) == 0
