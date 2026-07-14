"""Unit tests for eureka/fitness.py."""

import pytest

from eureka.fitness import compute_fitness

WEIGHTS = {"crash": 1.0, "speed": 0.05, "overtakes": 0.3}


def _metrics(crash_rate=0.0, mean_speed=20.0, mean_overtakes=1.0, mean_raw_return=0.5):
    return {
        "crash_rate": crash_rate,
        "mean_speed": mean_speed,
        "mean_overtakes": mean_overtakes,
        "mean_raw_return": mean_raw_return,
    }


def test_fitness_decreases_with_higher_crash_rate():
    low_crash = compute_fitness(_metrics(crash_rate=0.1), WEIGHTS)
    high_crash = compute_fitness(_metrics(crash_rate=0.5), WEIGHTS)
    assert low_crash > high_crash


def test_fitness_increases_with_higher_mean_speed():
    slow = compute_fitness(_metrics(mean_speed=10.0), WEIGHTS)
    fast = compute_fitness(_metrics(mean_speed=25.0), WEIGHTS)
    assert fast > slow


def test_fitness_increases_with_higher_mean_overtakes():
    few = compute_fitness(_metrics(mean_overtakes=0.0), WEIGHTS)
    many = compute_fitness(_metrics(mean_overtakes=4.0), WEIGHTS)
    assert many > few


def test_fitness_zero_metrics():
    result = compute_fitness(_metrics(crash_rate=0.0, mean_speed=0.0, mean_overtakes=0.0), WEIGHTS)
    assert result == pytest.approx(0.0)


def test_fitness_ignores_mean_raw_return():
    a = compute_fitness(_metrics(mean_raw_return=0.0), WEIGHTS)
    b = compute_fitness(_metrics(mean_raw_return=999.0), WEIGHTS)
    assert a == b


def test_fitness_negative_speed_weight_would_penalize_speed():
    weights = {"crash": 0.0, "speed": -0.1, "overtakes": 0.0}
    assert compute_fitness(_metrics(mean_speed=20.0), weights) < compute_fitness(
        _metrics(mean_speed=5.0), weights
    )
