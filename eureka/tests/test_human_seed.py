"""Tests for the human-authored generation-0 reward seed (EUREKA Sec 4.4)."""

import json
import os
from unittest.mock import MagicMock

import pytest

from eureka.human_seed import HUMAN_SEED_CODE
from eureka.sandbox import load_shaping_reward_from_code, validate_candidate_ast
from eureka.smoke_test import smoke_test
from reward_wrapper import RewardShapingWrapper


class _FakeVehicle:
    def __init__(self, x: float, speed: float, lane_id: int = 1):
        self.position = [x, 0.0]
        self.speed = speed
        self.lane_index = ("a", "b", lane_id)


class _FakeRoad:
    def __init__(self, vehicles):
        self.vehicles = vehicles


class _FakeUnwrapped:
    def __init__(self, ego, vehicles):
        self.vehicle = ego
        self.road = _FakeRoad(vehicles)


def test_human_seed_passes_ast_validation():
    passed, message = validate_candidate_ast(HUMAN_SEED_CODE)
    assert passed is True
    assert message == "ok"


def test_human_seed_passes_smoke_test():
    passed, message = smoke_test(HUMAN_SEED_CODE, n_trials=3)
    assert passed is True, message


def test_human_seed_matches_reward_wrapper_ttc_logic():
    shaping_fn = load_shaping_reward_from_code(HUMAN_SEED_CODE)

    ego = _FakeVehicle(x=0.0, speed=30.0, lane_id=1)
    ahead_close = _FakeVehicle(x=15.0, speed=20.0, lane_id=1)  # ttc = 1.5s < 3.0
    other_lane = _FakeVehicle(x=10.0, speed=10.0, lane_id=0)
    behind = _FakeVehicle(x=-5.0, speed=10.0, lane_id=1)
    vehicles = [ego, ahead_close, other_lane, behind]
    road = _FakeRoad(vehicles)

    ttc = 15.0 / 10.0
    severity = (3.0 - ttc) / 3.0
    expected_ttc = -0.1 * severity

    # Case 1: TTC penalty only
    info = {"n_overtakes": 0, "crashed": False, "speed": ego.speed, "raw_reward": 0.0}
    assert shaping_fn(ego, road, info) == pytest.approx(expected_ttc)

    wrapper = object.__new__(RewardShapingWrapper)
    wrapper.env = MagicMock()
    wrapper.env.unwrapped = _FakeUnwrapped(ego, vehicles)
    wrapper.ttc_threshold = 3.0
    wrapper.ttc_weight = 0.1
    wrapper.overtake_bonus = 0.2
    assert wrapper._compute_ttc_penalty() == pytest.approx(expected_ttc)
    assert shaping_fn(ego, road, info) == pytest.approx(
        wrapper._compute_ttc_penalty() + 0.2 * info["n_overtakes"]
    )

    # Case 2: other-lane vehicle alone -> no TTC penalty
    road_other = _FakeRoad([ego, other_lane])
    wrapper.env.unwrapped = _FakeUnwrapped(ego, [ego, other_lane])
    assert shaping_fn(ego, road_other, info) == pytest.approx(0.0)
    assert wrapper._compute_ttc_penalty() == pytest.approx(0.0)

    # Case 3: overtakes via info["n_overtakes"]
    info_ot = {"n_overtakes": 2, "crashed": False, "speed": ego.speed, "raw_reward": 0.0}
    expected_with_ot = expected_ttc + 0.2 * 2
    assert shaping_fn(ego, road, info_ot) == pytest.approx(expected_with_ot)
    wrapper.env.unwrapped = _FakeUnwrapped(ego, vehicles)
    assert shaping_fn(ego, road, info_ot) == pytest.approx(
        wrapper._compute_ttc_penalty() + 0.2 * info_ot["n_overtakes"]
    )


def test_loop_prepends_human_seed_in_generation_zero_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("eureka/candidates", exist_ok=True)

    llm_code = "def shaping_reward(ego, road, info):\n    return 0.0\n"
    metric = {
        "crash_rate": 0.1,
        "mean_speed": 20.0,
        "mean_overtakes": 1.0,
        "mean_raw_return": 0.0,
    }
    # gen0: human + 1 LLM; gen1: 1 LLM -> 3 evaluations
    metrics = iter([dict(metric) for _ in range(3)])

    monkeypatch.setattr("eureka.loop.N_GENERATIONS", 2)
    monkeypatch.setattr("eureka.loop.K_CANDIDATES", 1)
    monkeypatch.setattr("eureka.loop.SEED_GENERATION_0_WITH_HUMAN_REWARD", True)
    monkeypatch.setattr("eureka.loop.MULTI_OBJECTIVE_MODE", "shadow")
    monkeypatch.setattr("eureka.loop.CONFIRMATION_SEEDS", ())
    # This test pins an exact count of evaluate_candidate calls (one per
    # gen0/gen1 candidate) and isn't exercising second-seed screening -
    # disable it here so it doesn't consume extra items from `metrics`.
    monkeypatch.setattr("eureka.loop.SCREENING_SECOND_SEED_ENABLED", False)
    monkeypatch.setattr("eureka.loop.LOG_PATH", "eureka/eureka_log.json")
    monkeypatch.setattr(
        "eureka.loop.generate_candidates",
        lambda context, k, generation, model, temperature: [llm_code],
    )
    monkeypatch.setattr("eureka.loop.smoke_test", lambda code: (True, "ok"))
    monkeypatch.setattr(
        "eureka.loop.train_candidate",
        lambda module_path, total_timesteps, seed: f"{module_path}.pt",
    )
    monkeypatch.setattr(
        "eureka.loop.evaluate_candidate",
        lambda checkpoint, module_path, n_episodes: next(metrics),
    )

    from eureka.loop import main

    main()

    log_data = json.loads(
        (tmp_path / "eureka" / "eureka_log.json").read_text(encoding="utf-8")
    )
    assert len(log_data) == 2

    gen0 = log_data[0]["results"]
    gen1 = log_data[1]["results"]
    assert len(gen0) == 2  # K_CANDIDATES + 1
    assert sum(1 for r in gen0 if r.get("source") == "human_seed") == 1
    assert sum(1 for r in gen0 if r.get("source") == "llm") == 1
    assert len(gen1) == 1
    assert all(r.get("source") == "llm" for r in gen1)