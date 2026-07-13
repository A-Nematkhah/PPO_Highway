"""Unit tests for shared overtake counting in reward_wrapper.py."""

from unittest.mock import MagicMock

from reward_wrapper import RewardShapingWrapper, compute_overtakes


class _FakeVehicle:
    def __init__(self, x: float):
        self.position = [x, 0.0]


class _FakeRoad:
    def __init__(self, vehicles):
        self.vehicles = vehicles


class _FakeUnwrapped:
    def __init__(self, ego, other):
        self.vehicle = ego
        self.road = _FakeRoad([ego, other])


def test_compute_overtakes_detects_single_pass():
    ego = _FakeVehicle(10.0)
    other = _FakeVehicle(15.0)
    prev = {id(other): 5.0}
    other.position[0] = 8.0

    n_overtakes, current = compute_overtakes(_FakeRoad([ego, other]), ego, prev)

    assert n_overtakes == 1
    assert current[id(other)] == -2.0


def test_reward_shaping_wrapper_uses_shared_overtake_logic():
    ego = _FakeVehicle(10.0)
    other = _FakeVehicle(15.0)
    fake_env = MagicMock()
    fake_env.unwrapped = _FakeUnwrapped(ego, other)

    wrapper = object.__new__(RewardShapingWrapper)
    wrapper.env = fake_env
    wrapper.overtake_bonus = 0.25
    wrapper._prev_relative_x = {id(other): 5.0}
    other.position[0] = 8.0

    bonus, n_overtakes = wrapper._compute_overtake_bonus()

    expected_n, _ = compute_overtakes(
        fake_env.unwrapped.road,
        ego,
        {id(other): 5.0},
    )
    assert n_overtakes == expected_n == 1
    assert bonus == 0.25 * expected_n
