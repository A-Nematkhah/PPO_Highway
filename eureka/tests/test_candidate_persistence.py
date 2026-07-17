"""Regression tests for candidate-file lifecycle: rejected-candidate
persistence and stale-file cleanup between runs.

Context: a user reported uploading gen0_cand1/2/3/5.py after a run that
rejected candidates at those exact slots, expecting to see the code that
tripped the sandbox - but the files on disk were untouched leftovers from
an EARLIER run that had happened to pass at those same slot indices,
because rejected candidates were never written anywhere and nothing ever
cleared old files between runs. This file locks in the fix: rejected
source is now preserved (for debugging) under eureka/candidates/rejected/,
and eureka/loop.main() clears both directories at the start of every run
so a stale file can never again be mistaken for the current run's output.
"""

import os

_REJECTED_CODE = (
    "def shaping_reward(ego, road, info):\n"
    "    f = lambda: 1.0\n"
    "    return f()\n"
)
_VALID_CODE = "def shaping_reward(ego, road, info):\n    return 0.0\n"


def test_rejected_candidate_source_is_persisted_with_reason(tmp_path, monkeypatch):
    import eureka.loop as loop

    monkeypatch.chdir(tmp_path)
    os.makedirs("eureka/candidates", exist_ok=True)

    from eureka.telemetry import Telemetry

    telemetry = Telemetry(str(tmp_path / "metrics.jsonl"))
    survivors = loop._smoke_test_and_save(
        [_REJECTED_CODE], generation=0, human_seed_index=None, telemetry=telemetry
    )

    assert survivors == []

    rejected_path = tmp_path / "eureka" / "candidates" / "rejected" / "gen0_cand0.py"
    assert rejected_path.is_file()
    content = rejected_path.read_text(encoding="utf-8")
    assert "REJECTED" in content
    assert "generation=0 candidate=0" in content
    assert "lambda" in content.lower()  # exact rejection reason preserved
    assert "f = lambda: 1.0" in content  # exact source preserved

    # The main candidates dir must NOT contain a file for the rejected slot.
    assert not (tmp_path / "eureka" / "candidates" / "gen0_cand0.py").is_file()


def test_survivor_candidate_is_not_written_to_rejected_dir(tmp_path, monkeypatch):
    import eureka.loop as loop

    monkeypatch.chdir(tmp_path)
    os.makedirs("eureka/candidates", exist_ok=True)

    from eureka.telemetry import Telemetry

    telemetry = Telemetry(str(tmp_path / "metrics.jsonl"))
    survivors = loop._smoke_test_and_save(
        [_VALID_CODE], generation=0, human_seed_index=None, telemetry=telemetry
    )

    assert len(survivors) == 1
    assert (tmp_path / "eureka" / "candidates" / "gen0_cand0.py").is_file()
    assert not (tmp_path / "eureka" / "candidates" / "rejected" / "gen0_cand0.py").is_file()


def test_main_clears_stale_candidate_files_from_a_previous_run(tmp_path, monkeypatch):
    """
    Regression test for the exact scenario reported: a stale
    gen0_cand1.py left over from an earlier run must NOT still be present
    (and therefore must not be mistakable for this run's output) after a
    new run starts - regardless of whether this run's candidate at that
    same slot passes or fails.
    """
    import eureka.loop as loop

    monkeypatch.chdir(tmp_path)
    os.makedirs("eureka/candidates/rejected", exist_ok=True)
    os.makedirs("eureka/checkpoints", exist_ok=True)

    stale_survivor = tmp_path / "eureka" / "candidates" / "gen0_cand1.py"
    stale_survivor.write_text(_VALID_CODE, encoding="utf-8")
    stale_rejected = tmp_path / "eureka" / "candidates" / "rejected" / "gen0_cand3.py"
    stale_rejected.write_text("# old rejected code\n", encoding="utf-8")

    monkeypatch.setattr(loop, "N_GENERATIONS", 1)
    monkeypatch.setattr(loop, "K_CANDIDATES", 1)
    monkeypatch.setattr(loop, "MULTI_OBJECTIVE_MODE", "shadow")
    monkeypatch.setattr(loop, "CONFIRMATION_SEEDS", ())
    monkeypatch.setattr(loop, "SEED_GENERATION_0_WITH_HUMAN_REWARD", False)
    monkeypatch.setattr(loop, "LOG_PATH", "eureka/eureka_log.json")
    monkeypatch.setattr(
        loop, "generate_candidates",
        lambda context, k, generation, model, temperature: [_VALID_CODE],
    )
    monkeypatch.setattr(loop, "smoke_test", lambda code: (True, "ok"))
    monkeypatch.setattr(
        loop, "train_candidate",
        lambda module_path, total_timesteps, seed: f"{module_path}.pt",
    )
    monkeypatch.setattr(
        loop, "evaluate_candidate",
        lambda checkpoint, module_path, n_episodes: {
            "crash_rate": 0.0, "mean_speed": 20.0,
            "mean_overtakes": 0.0, "mean_raw_return": 1.0,
        },
    )

    loop.main()

    # The stale files from the "previous run" must be gone - this run's
    # own candidate 0 output must be the only thing present.
    assert not stale_survivor.is_file()
    assert not stale_rejected.is_file()
    assert (tmp_path / "eureka" / "candidates" / "gen0_cand0.py").is_file()
