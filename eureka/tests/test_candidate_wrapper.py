"""Unit tests for eureka/candidate_wrapper.py."""

import numpy as np
import pytest
import gymnasium as gym

from eureka.candidate_wrapper import CandidateRewardWrapper


class _FakeVehicle:
    def __init__(self, x: float):
        self.position = [x, 0.0]


class _FakeRoad:
    def __init__(self, vehicles):
        self.vehicles = vehicles


class _MinimalEnv(gym.Env):
    def __init__(self, ego, other):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self.vehicle = ego
        self.road = _FakeRoad([ego, other])

    def reset(self, *, seed=None, options=None):
        return np.zeros(5, dtype=np.float32), {}

    def step(self, action):
        return (
            np.zeros(5, dtype=np.float32),
            1.0,
            False,
            False,
            {"speed": 20.0, "crashed": False},
        )


def test_candidate_wrapper_applies_shaping_fn():
    ego = _FakeVehicle(10.0)
    other = _FakeVehicle(12.0)
    env = _MinimalEnv(ego, other)
    wrapper = CandidateRewardWrapper(env, lambda ego, road, info: 0.5)
    obs, reward, term, trunc, info = wrapper.step(0)

    assert reward == pytest.approx(1.5)
    assert info["shaping_value"] == pytest.approx(0.5)
    assert "n_overtakes" in info
    wrapper.close()


def test_candidate_wrapper_zero_on_bad_return():
    ego = _FakeVehicle(0.0)
    other = _FakeVehicle(5.0)
    env = _MinimalEnv(ego, other)
    wrapper = CandidateRewardWrapper(env, lambda ego, road, info: float("nan"))
    _, reward, _, _, info = wrapper.step(0)
    assert info["shaping_value"] == 0.0
    assert info["shaping_components"] == {}
    assert reward == pytest.approx(1.0)
    wrapper.close()


def test_candidate_wrapper_propagates_components_without_double_counting():
    ego = _FakeVehicle(10.0)
    other = _FakeVehicle(12.0)
    env = _MinimalEnv(ego, other)
    wrapper = CandidateRewardWrapper(
        env, lambda ego, road, info: (0.5, {"ttc": 0.3, "overtake": 0.2})
    )
    _, reward, _, _, info = wrapper.step(0)

    assert info["shaping_value"] == pytest.approx(0.5)
    assert info["shaping_components"] == {"ttc": 0.3, "overtake": 0.2}
    assert reward == pytest.approx(1.5)  # raw 1.0 + total 0.5 only
    wrapper.close()
