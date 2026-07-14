"""Tests for optional multi-seed Pareto finalist confirmation."""

import json

import pytest

from eureka.eureka_config import OBJECTIVE_SPECS
from eureka.objectives import annotate_population
from eureka.telemetry import Telemetry


def _candidate():
    return {
        "module_path": "eureka.candidates.finalist",
        "code": "def shaping_reward(ego, road, info):\n    return 0.0\n",
        "checkpoint": "screening.pt",
        "metrics": {
            "crash_rate": 0.2,
            "mean_speed": 20.0,
            "mean_overtakes": 1.0,
            "mean_raw_return": 0.5,
        },
    }


def test_confirmation_aggregates_independent_seed_metrics(tmp_path, monkeypatch):
    import eureka.loop as loop

    archive = [_candidate()]
    annotate_population(archive, OBJECTIVE_SPECS)
    monkeypatch.setattr(loop, "CONFIRMATION_SEEDS", (101, 202))
    monkeypatch.setattr(
        loop,
        "train_candidate",
        lambda module_path, total_timesteps, seed: f"{seed}.pt",
    )
    rows = iter([
        {
            "crash_rate": 0.0,
            "mean_speed": 22.0,
            "mean_overtakes": 2.0,
            "mean_raw_return": 0.7,
        },
        {
            "crash_rate": 0.4,
            "mean_speed": 24.0,
            "mean_overtakes": 3.0,
            "mean_raw_return": 0.9,
        },
    ])
    monkeypatch.setattr(
        loop,
        "evaluate_candidate",
        lambda checkpoint, module_path, n_episodes: next(rows),
    )

    telemetry = Telemetry(str(tmp_path / "metrics.jsonl"))
    confirmed = loop._confirm_archive(archive, telemetry)

    assert len(confirmed) == 1
    metrics = confirmed[0]["metrics"]
    assert metrics["crash_rate"] == pytest.approx(0.2)
    assert metrics["mean_speed"] == pytest.approx(22.0)
    assert metrics["mean_overtakes"] == pytest.approx(2.0)
    assert len(confirmed[0]["confirmation_runs"]) == 2

    events = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == ["confirmation", "confirmation"]


def test_confirmation_disabled_is_noop(tmp_path, monkeypatch):
    import eureka.loop as loop

    archive = [_candidate()]
    monkeypatch.setattr(loop, "CONFIRMATION_SEEDS", ())
    telemetry = Telemetry(str(tmp_path / "metrics.jsonl"))
    assert loop._confirm_archive(archive, telemetry) is archive
